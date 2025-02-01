#pragma once

#include <iostream>
#include <vector>
#include <functional>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iterator>
#include <immintrin.h> // For AVX/SSE intrinsics
#include <string>
#include <chrono>
#include <unordered_set>
#include <atomic>
#include <memory>
#include <future>
#include <thread>
#include <unordered_map>
#include <mutex>
#include <shared_mutex>
#include <optional>

namespace drb {

namespace detail {
    // SIMD helper functions
    template<typename T>
    struct SimdTraits {
        static constexpr bool has_simd = false;
        static constexpr size_t vector_size = 1;
    };

    template<>
    struct SimdTraits<float> {
        static constexpr bool has_simd = true;
        static constexpr size_t vector_size = 8; // AVX: 256 bits = 8 floats
        using simd_type = __m256;

        static simd_type load(const float* ptr) { return _mm256_loadu_ps(ptr); }
        static void store(float* ptr, simd_type val) { _mm256_storeu_ps(ptr, val); }
        static simd_type add(simd_type a, simd_type b) { return _mm256_add_ps(a, b); }
        static simd_type multiply(simd_type a, simd_type b) { return _mm256_mul_ps(a, b); }
        static simd_type set1(float val) { return _mm256_set1_ps(val); }
        
        static float sum(simd_type v) {
            __m128 high = _mm256_extractf128_ps(v, 1);
            __m128 low = _mm256_castps256_ps128(v);
            __m128 sum = _mm_add_ps(high, low);
            sum = _mm_hadd_ps(sum, sum);
            sum = _mm_hadd_ps(sum, sum);
            return _mm_cvtss_f32(sum);
        }
    };

    template<>
    struct SimdTraits<double> {
        static constexpr bool has_simd = true;
        static constexpr size_t vector_size = 4; // AVX: 256 bits = 4 doubles
        using simd_type = __m256d;

        static simd_type load(const double* ptr) { return _mm256_loadu_pd(ptr); }
        static void store(double* ptr, simd_type val) { _mm256_storeu_pd(ptr, val); }
        static simd_type add(simd_type a, simd_type b) { return _mm256_add_pd(a, b); }
        static simd_type multiply(simd_type a, simd_type b) { return _mm256_mul_pd(a, b); }
        static simd_type set1(double val) { return _mm256_set1_pd(val); }
        
        static double sum(simd_type v) {
            __m128d high = _mm256_extractf128_pd(v, 1);
            __m128d low = _mm256_castpd256_pd128(v);
            __m128d sum = _mm_add_pd(high, low);
            sum = _mm_hadd_pd(sum, sum);
            return _mm_cvtsd_f64(sum);
        }
    };
}

template<typename T>
class DynamicRingBuffer {
private:
    struct BufferState {
        std::vector<T> data;
        std::atomic<size_t> head{0};
        std::atomic<size_t> tail{0};
        std::atomic<size_t> size{0};
        std::atomic<double> cached_mean{0.0};
        mutable std::shared_mutex data_mutex;
        std::mutex resize_mutex;

        explicit BufferState(size_t capacity = 4) : data(capacity) {}
    };

    std::shared_ptr<BufferState> state;

public:
    template<bool IsConst>
    class Iterator {
    private:
        using BufferPtr = std::shared_ptr<typename std::conditional<IsConst, const BufferState, BufferState>::type>;
        BufferPtr rb;
        size_t index;

    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = typename std::conditional<IsConst, const T, T>::type;
        using difference_type = std::ptrdiff_t;
        using pointer = value_type*;
        using reference = value_type&;

        Iterator(BufferPtr rb, size_t index) : rb(rb), index(index) {}

        reference operator*() const {
            std::shared_lock lock(rb->data_mutex);
            size_t head = rb->head.load(std::memory_order_acquire);
            return rb->data[(head + index) % rb->data.size()];
        }

        Iterator& operator++() {
            ++index;
            return *this;
        }

        Iterator operator++(int) {
            Iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        bool operator==(const Iterator& other) const {
            return rb == other.rb && index == other.index;
        }

        bool operator!=(const Iterator& other) const {
            return !(*this == other);
        }
    };

    using iterator = Iterator<false>;
    using const_iterator = Iterator<true>;

    explicit DynamicRingBuffer(size_t initial_capacity = 4)
        : state(std::make_shared<BufferState>(initial_capacity)) {}

    DynamicRingBuffer(const DynamicRingBuffer& other) {
        std::shared_lock lock(other.state->data_mutex);
        state = std::make_shared<BufferState>();
        state->data = other.state->data;
        state->head.store(other.state->head.load(std::memory_order_acquire), std::memory_order_release);
        state->tail.store(other.state->tail.load(std::memory_order_acquire), std::memory_order_release);
        state->size.store(other.state->size.load(std::memory_order_acquire), std::memory_order_release);
        state->cached_mean.store(other.state->cached_mean.load(std::memory_order_acquire), std::memory_order_release);
    }

    DynamicRingBuffer& operator=(const DynamicRingBuffer& other) {
        if (this != &other) {
            std::shared_lock lock(other.state->data_mutex);
            auto new_state = std::make_shared<BufferState>();
            new_state->data = other.state->data;
            new_state->head.store(other.state->head.load(std::memory_order_acquire), std::memory_order_release);
            new_state->tail.store(other.state->tail.load(std::memory_order_acquire), std::memory_order_release);
            new_state->size.store(other.state->size.load(std::memory_order_acquire), std::memory_order_release);
            new_state->cached_mean.store(other.state->cached_mean.load(std::memory_order_acquire), std::memory_order_release);
            state = std::move(new_state);
        }
        return *this;
    }

    bool push(const T& value) {
        std::unique_lock resize_lock(state->resize_mutex);
        std::unique_lock data_lock(state->data_mutex);
        size_t current_size = state->size.load(std::memory_order_relaxed);
        if (current_size >= state->data.size()) {
            // Resize with locks already held
            size_t new_capacity = state->data.size() * 2;
            std::vector<T> new_data(new_capacity);
            size_t head = state->head.load(std::memory_order_relaxed);

            for (size_t i = 0; i < current_size; ++i) {
                new_data[i] = state->data[(head + i) % state->data.size()];
            }

            state->data = std::move(new_data);
            state->head.store(0, std::memory_order_release);
            state->tail.store(current_size, std::memory_order_release);
        }
        
        size_t tail = state->tail.load(std::memory_order_relaxed);
        state->data[tail] = value;
        state->tail.store((tail + 1) % state->data.size(), std::memory_order_release);
        state->size.fetch_add(1, std::memory_order_release);
        return true;
    }

    std::optional<T> pop() {
        std::unique_lock lock(state->data_mutex);
        if (isEmpty()) {
            return std::nullopt;
        }
        
        size_t head = state->head.load(std::memory_order_relaxed);
        T value = state->data[head];
        state->head.store((head + 1) % state->data.size(), std::memory_order_release);
        state->size.fetch_sub(1, std::memory_order_release);
        return value;
    }

    void resize(size_t new_capacity) {
        std::unique_lock resize_lock(state->resize_mutex);
        std::unique_lock data_lock(state->data_mutex);
        if (new_capacity <= state->data.size()) {
            return;
        }

        std::vector<T> new_data(new_capacity);
        size_t current_size = state->size.load(std::memory_order_relaxed);
        size_t head = state->head.load(std::memory_order_relaxed);

        for (size_t i = 0; i < current_size; ++i) {
            new_data[i] = state->data[(head + i) % state->data.size()];
        }

        state->data = std::move(new_data);
        state->head.store(0, std::memory_order_release);
        state->tail.store(current_size, std::memory_order_release);
    }

    size_t size() const {
        return state->size.load(std::memory_order_acquire);
    }

    size_t getCapacity() const {
        std::shared_lock lock(state->data_mutex);
        return state->data.size();
    }

    bool isEmpty() const {
        return size() == 0;
    }

    double mean() const {
        std::shared_lock lock(state->data_mutex);
        if (isEmpty()) return 0.0;
        
        double sum = 0.0;
        size_t count = 0;
        size_t head = state->head.load(std::memory_order_relaxed);
        size_t current_size = state->size.load(std::memory_order_relaxed);
        
        for (size_t i = 0; i < current_size; ++i) {
            sum += static_cast<double>(state->data[(head + i) % state->data.size()]);
            count++;
        }
        
        double result = sum / count;
        state->cached_mean.store(result, std::memory_order_release);
        return result;
    }

    // Helper function to calculate mean and variance together to avoid lock recursion
    std::pair<double, double> calculate_mean_and_variance() const {
        std::shared_lock lock(state->data_mutex);
        if (isEmpty()) return {0.0, 0.0};
        
        double sum = 0.0;
        double sum_sq = 0.0;
        size_t count = 0;
        size_t head = state->head.load(std::memory_order_relaxed);
        size_t current_size = state->size.load(std::memory_order_relaxed);
        
        for (size_t i = 0; i < current_size; ++i) {
            double value = static_cast<double>(state->data[(head + i) % state->data.size()]);
            sum += value;
            sum_sq += value * value;
            count++;
        }
        
        double mean = sum / count;
        double variance = (sum_sq / count) - (mean * mean);
        
        state->cached_mean.store(mean, std::memory_order_release);
        return {mean, variance};
    }

    double variance() const {
        return calculate_mean_and_variance().second;
    }

    double stddev() const {
        return std::sqrt(calculate_mean_and_variance().second);
    }

    void clear() {
        std::unique_lock lock(state->data_mutex);
        state->head.store(0, std::memory_order_release);
        state->tail.store(0, std::memory_order_release);
        state->size.store(0, std::memory_order_release);
        state->cached_mean.store(0.0, std::memory_order_release);
    }

    iterator begin() { return iterator(state, 0); }
    iterator end() { return iterator(state, size()); }
    const_iterator begin() const { return const_iterator(state, 0); }
    const_iterator end() const { return const_iterator(state, size()); }
    const_iterator cbegin() const { return const_iterator(state, 0); }
    const_iterator cend() const { return const_iterator(state, size()); }

    std::vector<T> get_snapshot() const {
        std::shared_lock lock(state->data_mutex);
        std::vector<T> result;
        size_t current_size = state->size.load(std::memory_order_relaxed);
        result.reserve(current_size);
        
        size_t head = state->head.load(std::memory_order_relaxed);
        for (size_t i = 0; i < current_size; ++i) {
            result.push_back(state->data[(head + i) % state->data.size()]);
        }
        
        return result;
    }

    void add(const T& value) {
        std::unique_lock lock(state->data_mutex);
        size_t current_size = state->size.load(std::memory_order_relaxed);
        size_t head = state->head.load(std::memory_order_relaxed);
        
        for (size_t i = 0; i < current_size; ++i) {
            state->data[(head + i) % state->data.size()] += value;
        }
    }

    void multiply(const T& value) {
        std::unique_lock lock(state->data_mutex);
        size_t current_size = state->size.load(std::memory_order_relaxed);
        size_t head = state->head.load(std::memory_order_relaxed);
        
        for (size_t i = 0; i < current_size; ++i) {
            state->data[(head + i) % state->data.size()] *= value;
        }
    }

    T sum() const {
        std::shared_lock lock(state->data_mutex);
        if (isEmpty()) return T{};
        
        T result{};
        size_t current_size = state->size.load(std::memory_order_relaxed);
        size_t head = state->head.load(std::memory_order_relaxed);
        
        for (size_t i = 0; i < current_size; ++i) {
            result += state->data[(head + i) % state->data.size()];
        }
        
        return result;
    }

    T max() const {
        std::shared_lock lock(state->data_mutex);
        if (isEmpty()) throw std::runtime_error("Buffer is empty");
        
        size_t current_size = state->size.load(std::memory_order_relaxed);
        size_t head = state->head.load(std::memory_order_relaxed);
        T result = state->data[head];
        
        for (size_t i = 1; i < current_size; ++i) {
            T value = state->data[(head + i) % state->data.size()];
            if (value > result) {
                result = value;
            }
        }
        
        return result;
    }

    void rotate(size_t n) {
        std::unique_lock lock(state->data_mutex);
        if (isEmpty()) return;

        size_t current_size = state->size.load(std::memory_order_relaxed);
        n = n % current_size;
        if (n == 0) return;

        std::vector<T> temp;
        temp.reserve(current_size);
        size_t head = state->head.load(std::memory_order_relaxed);

        // Copy elements to temporary vector
        for (size_t i = 0; i < current_size; ++i) {
            temp.push_back(state->data[(head + i) % state->data.size()]);
        }

        // Rotate the temporary vector
        std::rotate(temp.begin(), temp.begin() + n, temp.end());

        // Copy back to the buffer
        for (size_t i = 0; i < current_size; ++i) {
            state->data[i] = temp[i];
        }

        // Reset head and tail
        state->head.store(0, std::memory_order_release);
        state->tail.store(current_size, std::memory_order_release);
    }

    void reverse() {
        std::unique_lock lock(state->data_mutex);
        size_t current_size = state->size.load(std::memory_order_relaxed);
        if (current_size <= 1) return;
        
        size_t head = state->head.load(std::memory_order_relaxed);
        for (size_t i = 0; i < current_size / 2; ++i) {
            size_t j = current_size - 1 - i;
            std::swap(
                state->data[(head + i) % state->data.size()],
                state->data[(head + j) % state->data.size()]
            );
        }
    }

    DynamicRingBuffer<T> compress(const std::vector<bool>& mask) {
        if (mask.empty()) return DynamicRingBuffer<T>();
        
        std::shared_lock lock(state->data_mutex);
        size_t current_size = state->size.load(std::memory_order_relaxed);
        if (mask.size() != current_size) {
            throw std::invalid_argument("Mask size must match buffer size");
        }
        
        DynamicRingBuffer<T> result;
        size_t head = state->head.load(std::memory_order_relaxed);
        
        for (size_t i = 0; i < current_size; ++i) {
            if (mask[i]) {
                result.push(state->data[(head + i) % state->data.size()]);
            }
        }
        
        return result;
    }

    // APL-style operations
    
    // Outer product: applies a binary operation between all pairs of elements
    template<typename BinaryOp>
    std::vector<std::vector<T>> outer_product(const DynamicRingBuffer<T>& other, BinaryOp op) const {
        std::vector<T> this_data = get_snapshot();
        std::vector<T> other_data = other.get_snapshot();
        
        std::vector<std::vector<T>> result(this_data.size(), std::vector<T>(other_data.size()));
        for (size_t i = 0; i < this_data.size(); ++i) {
            for (size_t j = 0; j < other_data.size(); ++j) {
                result[i][j] = op(this_data[i], other_data[j]);
            }
        }
        return result;
    }

    // Compress: keep only elements where mask is true
    DynamicRingBuffer<T> compress(const std::vector<bool>& mask) const {
        std::vector<T> data = get_snapshot();
        if (mask.size() != data.size()) {
            throw std::invalid_argument("Mask size must match buffer size");
        }

        DynamicRingBuffer<T> result;
        for (size_t i = 0; i < data.size(); ++i) {
            if (mask[i]) {
                result.push(data[i]);
            }
        }
        return result;
    }

    // Expand: insert default values where mask is false
    DynamicRingBuffer<T> expand(const std::vector<bool>& mask, const T& default_value = T()) const {
        std::vector<T> data = get_snapshot();
        DynamicRingBuffer<T> result;
        size_t buffer_idx = 0;

        for (bool keep : mask) {
            if (keep && buffer_idx < data.size()) {
                result.push(data[buffer_idx++]);
            } else {
                result.push(default_value);
            }
        }
        return result;
    }

    // Scan/Prefix operations with binary operator
    template<typename BinaryOp>
    DynamicRingBuffer<T> scan(BinaryOp op, const T& initial = T()) const {
        std::vector<T> data = get_snapshot();
        DynamicRingBuffer<T> result;
        if (data.empty()) return result;

        T accumulator = initial;
        for (const T& value : data) {
            accumulator = op(accumulator, value);
            result.push(accumulator);
        }
        return result;
    }

    // Rotate elements by n positions
    void rotate(int n) {
        std::vector<T> data = get_snapshot();
        if (data.empty()) return;
        
        n = ((n % data.size()) + data.size()) % data.size(); // Normalize n to positive value
        if (n == 0) return;

        std::vector<T> temp(data.size());
        for (size_t i = 0; i < data.size(); ++i) {
            size_t new_idx = (i + n) % data.size();
            temp[new_idx] = std::move(data[i]);
        }

        std::unique_lock lock(state->data_mutex);
        state->data = std::move(temp);
        state->head.store(0, std::memory_order_release);
        state->tail.store(data.size(), std::memory_order_release);
    }

    bool isFull() const {
        std::shared_lock lock(state->data_mutex);
        return state->size.load(std::memory_order_acquire) == state->data.size();
    }

    size_t getSize() const { 
        return state->size.load(std::memory_order_acquire); 
    }

    T median() {
        auto snapshot = get_snapshot();
        if (snapshot.empty()) {
            throw std::runtime_error("Buffer is empty");
        }
        
        std::sort(snapshot.begin(), snapshot.end());
        size_t mid = snapshot.size() / 2;
        
        if (snapshot.size() % 2 == 0) {
            return (snapshot[mid - 1] + snapshot[mid]) / static_cast<T>(2);
        } else {
            return snapshot[mid];
        }
    }

    // MapReduce functionality
    template<typename K, typename V, typename MapFunc, typename ReduceFunc>
    std::unordered_map<K, V> mapReduce(
        MapFunc mapFunc,      // Function: T -> vector<pair<K, V>>
        ReduceFunc reduceFunc,// Function: (V, V) -> V
        size_t num_workers = std::thread::hardware_concurrency()
    ) const {
        std::vector<T> data = get_snapshot();
        if (data.empty()) return {};

        // 1. Map Phase
        std::vector<std::future<std::vector<std::pair<K, V>>>> map_futures;
        std::vector<std::vector<std::pair<K, V>>> mapped_results;
        
        size_t chunk_size = std::max(size_t(1), data.size() / num_workers);
        for (size_t i = 0; i < data.size(); i += chunk_size) {
            size_t end = std::min(i + chunk_size, data.size());
            map_futures.push_back(std::async(std::launch::async, [&, i, end]() {
                std::vector<std::pair<K, V>> chunk_result;
                for (size_t j = i; j < end; ++j) {
                    auto mapped = mapFunc(data[j]);
                    chunk_result.insert(chunk_result.end(), mapped.begin(), mapped.end());
                }
                return chunk_result;
            }));
        }

        // Collect map results
        for (auto& future : map_futures) {
            mapped_results.push_back(future.get());
        }

        // 2. Shuffle Phase
        std::unordered_map<K, std::vector<V>> shuffled;
        for (const auto& chunk : mapped_results) {
            for (const auto& [key, value] : chunk) {
                shuffled[key].push_back(value);
            }
        }

        // 3. Reduce Phase
        std::vector<std::future<std::pair<K, V>>> reduce_futures;
        for (const auto& [key, values] : shuffled) {
            reduce_futures.push_back(std::async(std::launch::async, [&, key, values]() {
                if (values.empty()) {
                    return std::make_pair(key, V{});
                }
                V result = values[0];
                for (size_t i = 1; i < values.size(); ++i) {
                    result = reduceFunc(result, values[i]);
                }
                return std::make_pair(key, result);
            }));
        }

        // Collect reduce results
        std::unordered_map<K, V> result;
        for (auto& future : reduce_futures) {
            auto [key, value] = future.get();
            result[key] = value;
        }

        return result;
    }

    // Convenience method for word count
    std::unordered_map<std::string, size_t> wordCount(
        size_t num_workers = std::thread::hardware_concurrency()
    ) const requires std::is_same_v<T, std::string> {
        return mapReduce<std::string, size_t>(
            // Map function: Split string into words and count each as 1
            [](const std::string& str) {
                std::string word;
                std::vector<std::pair<std::string, size_t>> result;
                for (char c : str) {
                    if (std::isalnum(c)) {
                        word += std::tolower(c);
                    } else if (!word.empty()) {
                        result.emplace_back(word, 1);
                        word.clear();
                    }
                }
                if (!word.empty()) {
                    result.emplace_back(word, 1);
                }
                return result;
            },
            // Reduce function: Sum the counts
            [](size_t a, size_t b) { return a + b; },
            num_workers
        );
    }

    // Parallel aggregation with custom functions
    template<typename R, typename MapFunc, typename ReduceFunc>
    R parallelAggregate(
        MapFunc mapFunc,      // Function: T -> R
        ReduceFunc reduceFunc,// Function: (R, R) -> R
        R initial_value,
        size_t num_workers = std::thread::hardware_concurrency()
    ) const {
        std::vector<T> data = get_snapshot();
        if (data.empty()) return initial_value;

        std::vector<std::future<R>> futures;
        size_t chunk_size = std::max(size_t(1), data.size() / num_workers);
        
        for (size_t i = 0; i < data.size(); i += chunk_size) {
            size_t end = std::min(i + chunk_size, data.size());
            futures.push_back(std::async(std::launch::async, [&, i, end]() {
                R result = initial_value;
                for (size_t j = i; j < end; ++j) {
                    result = reduceFunc(result, mapFunc(data[j]));
                }
                return result;
            }));
        }

        R final_result = initial_value;
        for (auto& future : futures) {
            final_result = reduceFunc(final_result, future.get());
        }

        return final_result;
    }

    // Parallel filter
    DynamicRingBuffer<T> parallelFilter(
        std::function<bool(const T&)> predicate,
        size_t num_workers = std::thread::hardware_concurrency()
    ) const {
        std::vector<T> data = get_snapshot();
        if (data.empty()) return DynamicRingBuffer<T>();

        std::vector<std::future<std::vector<T>>> futures;
        size_t chunk_size = std::max(size_t(1), data.size() / num_workers);
        
        for (size_t i = 0; i < data.size(); i += chunk_size) {
            size_t end = std::min(i + chunk_size, data.size());
            futures.push_back(std::async(std::launch::async, [&, i, end]() {
                std::vector<T> result;
                for (size_t j = i; j < end; ++j) {
                    if (predicate(data[j])) {
                        result.push_back(data[j]);
                    }
                }
                return result;
            }));
        }

        DynamicRingBuffer<T> result;
        for (auto& future : futures) {
            auto filtered = future.get();
            for (const auto& item : filtered) {
                result.push(item);
            }
        }

        return result;
    }

    size_t capacity() const {
        std::shared_lock<std::shared_mutex> lock(state->data_mutex);
        return state->data.size();
    }

    // MapReduce functionality
    template<typename Func>
    void map(Func func) {
        size_t current_size = state->size.load(std::memory_order_acquire);
        if (current_size == 0) {
            return;
        }

        std::unique_lock<std::shared_mutex> lock(state->data_mutex);
        size_t head = state->head.load(std::memory_order_acquire);
        for (size_t i = 0; i < current_size; ++i) {
            size_t idx = (head + i) % state->data.size();
            state->data[idx] = func(state->data[idx]);
        }
    }

    template<typename Func>
    T reduce(Func func, T initial) {
        size_t current_size = state->size.load(std::memory_order_acquire);
        if (current_size == 0) {
            return initial;
        }

        std::shared_lock<std::shared_mutex> lock(state->data_mutex);
        T result = initial;
        size_t head = state->head.load(std::memory_order_acquire);
        for (size_t i = 0; i < current_size; ++i) {
            result = func(result, state->data[(head + i) % state->data.size()]);
        }
        return result;
    }
};

} // namespace drb 