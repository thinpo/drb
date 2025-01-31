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
#include <mutex>
#include <shared_mutex>
#include <memory>

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
        int head;
        int tail;
        int size;
        double cached_mean;
        bool mean_is_valid;

        BufferState(int capacity = 4) 
            : data(capacity), head(0), tail(0), size(0), 
              cached_mean(0), mean_is_valid(false) {}
    };

    std::shared_ptr<BufferState> state;
    mutable std::unique_ptr<std::shared_mutex> mutex;
    static constexpr double SHRINK_FACTOR = 0.25;

    // Helper function to get a snapshot of the current data
    std::vector<T> get_snapshot() const {
        std::vector<T> snapshot;
        {
            std::shared_lock lock(*mutex);
            snapshot.reserve(state->size);
            
            if (state->head < state->tail) {
                snapshot.insert(snapshot.end(),
                              state->data.begin() + state->head,
                              state->data.begin() + state->tail);
            } else if (state->size > 0) {
                snapshot.insert(snapshot.end(),
                              state->data.begin() + state->head,
                              state->data.end());
                snapshot.insert(snapshot.end(),
                              state->data.begin(),
                              state->data.begin() + state->tail);
            }
        }
        return snapshot;
    }

    void resize(int new_capacity) {
        std::vector<T> new_buffer(new_capacity);
        int old_count;
        std::vector<T> old_data;
        
        {
            std::unique_lock lock(*mutex);
            old_count = state->size;
            old_data = std::move(state->data);
        }
        
        if (state->head < state->tail) {
            std::copy(old_data.begin() + state->head, 
                     old_data.begin() + state->tail, 
                     new_buffer.begin());
        } else if (old_count > 0) {
            std::copy(old_data.begin() + state->head, 
                     old_data.end(), 
                     new_buffer.begin());
            std::copy(old_data.begin(), 
                     old_data.begin() + state->tail,
                     new_buffer.begin() + (old_data.size() - state->head));
        }
        
        {
            std::unique_lock lock(*mutex);
            state->head = 0;
            state->tail = old_count;
            state->data = std::move(new_buffer);
            state->mean_is_valid = false;
        }
    }

    template<bool IsConst>
    class Iterator {
    private:
        using BufferType = typename std::conditional<IsConst, const DynamicRingBuffer*, DynamicRingBuffer*>::type;
        std::vector<T> snapshot;
        size_t pos;

    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = typename std::conditional<IsConst, const T, T>::type;
        using difference_type = std::ptrdiff_t;
        using pointer = typename std::conditional<IsConst, const T*, T*>::type;
        using reference = typename std::conditional<IsConst, const T&, T&>::type;

        Iterator(BufferType rb, size_t pos) {
            if (rb) {
                snapshot = rb->get_snapshot();
            }
            this->pos = pos;
        }

        reference operator*() { return snapshot[pos]; }
        pointer operator->() { return &snapshot[pos]; }
        const reference operator*() const { return snapshot[pos]; }
        const pointer operator->() const { return &snapshot[pos]; }

        Iterator& operator++() { ++pos; return *this; }
        Iterator operator++(int) { Iterator tmp = *this; ++pos; return tmp; }
        Iterator& operator--() { --pos; return *this; }
        Iterator operator--(int) { Iterator tmp = *this; --pos; return tmp; }

        Iterator& operator+=(difference_type n) { pos += n; return *this; }
        Iterator operator+(difference_type n) const { Iterator tmp = *this; return tmp += n; }
        Iterator& operator-=(difference_type n) { pos -= n; return *this; }
        Iterator operator-(difference_type n) const { Iterator tmp = *this; return tmp -= n; }
        difference_type operator-(const Iterator& other) const { return pos - other.pos; }

        bool operator==(const Iterator& other) const { 
            return pos == other.pos && snapshot.size() == other.snapshot.size(); 
        }
        bool operator!=(const Iterator& other) const { return !(*this == other); }
        bool operator<(const Iterator& other) const { return pos < other.pos; }
        bool operator>(const Iterator& other) const { return other < *this; }
        bool operator<=(const Iterator& other) const { return !(other < *this); }
        bool operator>=(const Iterator& other) const { return !(*this < other); }

        reference operator[](difference_type n) { return snapshot[pos + n]; }
        const reference operator[](difference_type n) const { return snapshot[pos + n]; }
    };

public:
    using iterator = Iterator<false>;
    using const_iterator = Iterator<true>;

    explicit DynamicRingBuffer(int initial_capacity = 4)
        : state(std::make_shared<BufferState>(initial_capacity)),
          mutex(std::make_unique<std::shared_mutex>()) {}

    ~DynamicRingBuffer() {
        // No need to delete buffer, std::vector handles it
    }

    DynamicRingBuffer(const DynamicRingBuffer& other)
        : state(std::make_shared<BufferState>(*other.state)),
          mutex(std::make_unique<std::shared_mutex>()) {}

    DynamicRingBuffer(DynamicRingBuffer&& other) noexcept
        : state(std::move(other.state)),
          mutex(std::make_unique<std::shared_mutex>()) {
        other.state = std::make_shared<BufferState>();
    }

    DynamicRingBuffer& operator=(const DynamicRingBuffer& other) {
        if (this != &other) {
            std::unique_lock lock(*mutex);
            std::shared_lock other_lock(*other.mutex);
            state = std::make_shared<BufferState>(*other.state);
        }
        return *this;
    }

    DynamicRingBuffer& operator=(DynamicRingBuffer&& other) noexcept {
        if (this != &other) {
            std::unique_lock lock(*mutex);
            std::unique_lock other_lock(*other.mutex);
            state = std::move(other.state);
            other.state = std::make_shared<BufferState>();
        }
        return *this;
    }

    void push(const T& value) {
        bool needs_resize = false;
        {
            std::shared_lock lock(*mutex);
            needs_resize = (state->size == state->data.size());
        }
        
        if (needs_resize) {
            int new_capacity;
            {
                std::shared_lock lock(*mutex);
                new_capacity = state->data.size() * 2;
            }
            resize(new_capacity);
        }
        
        {
            std::unique_lock lock(*mutex);
            state->data[state->tail] = value;
            state->tail = (state->tail + 1) % state->data.size();
            state->size++;
            state->mean_is_valid = false;
        }
    }

    T pop() {
        T value;
        bool needs_resize = false;
        int current_size;
        int current_capacity;
        
        {
            std::unique_lock lock(*mutex);
            if (state->size == 0) {
                throw std::runtime_error("Buffer is empty!");
            }
            value = state->data[state->head];
            state->head = (state->head + 1) % state->data.size();
            state->size--;
            current_size = state->size;
            current_capacity = state->data.size();
            needs_resize = (current_size < current_capacity * SHRINK_FACTOR && current_capacity > 4);
        }

        if (needs_resize) {
            resize(current_capacity / 2);
        }
        
        return value;
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

        {
            std::unique_lock lock(*mutex);
            for (size_t i = 0; i < data.size(); ++i) {
                state->data[(state->head + i) % state->data.size()] = std::move(temp[i]);
            }
        }
    }

    bool isEmpty() const { 
        std::shared_lock lock(*mutex);
        return state->size == 0; 
    }
    
    bool isFull() const { 
        std::shared_lock lock(*mutex);
        return state->size == state->data.size(); 
    }
    
    int getCapacity() const { 
        std::shared_lock lock(*mutex);
        return state->data.size(); 
    }
    
    int getSize() const { 
        std::shared_lock lock(*mutex);
        return state->size; 
    }

    void add(T value) {
        if constexpr (detail::SimdTraits<T>::has_simd) {
            using Traits = detail::SimdTraits<T>;
            using SimdType = typename Traits::simd_type;
            
            if (state->size >= Traits::vector_size) {
                const SimdType value_vec = Traits::set1(value);
                const size_t vec_size = Traits::vector_size;
                
                std::vector<T> aligned_data;
                if (state->head < state->tail) {
                    aligned_data.assign(state->data.begin() + state->head, state->data.begin() + state->tail);
                } else {
                    aligned_data.reserve(state->size);
                    aligned_data.insert(aligned_data.end(), state->data.begin() + state->head, state->data.end());
                    aligned_data.insert(aligned_data.end(), state->data.begin(), state->data.begin() + state->tail);
                }

                size_t i = 0;
                for (; i + vec_size <= aligned_data.size(); i += vec_size) {
                    SimdType data = Traits::load(&aligned_data[i]);
                    data = Traits::add(data, value_vec);
                    Traits::store(&aligned_data[i], data);
                }

                if (state->head < state->tail) {
                    std::copy(aligned_data.begin(), aligned_data.end(), state->data.begin() + state->head);
                } else {
                    size_t first_part = state->data.size() - state->head;
                    std::copy(aligned_data.begin(), aligned_data.begin() + first_part, state->data.begin() + state->head);
                    std::copy(aligned_data.begin() + first_part, aligned_data.end(), state->data.begin());
                }

                for (; i < aligned_data.size(); ++i) {
                    aligned_data[i] += value;
                }
                return;
            }
        }
        
        for (int i = 0; i < state->size; i++) {
            int idx = (state->head + i) % state->data.size();
            state->data[idx] += value;
        }
    }

    void multiply(T value) {
        if constexpr (detail::SimdTraits<T>::has_simd) {
            using Traits = detail::SimdTraits<T>;
            using SimdType = typename Traits::simd_type;
            
            if (state->size >= Traits::vector_size) {
                const SimdType value_vec = Traits::set1(value);
                const size_t vec_size = Traits::vector_size;
                
                std::vector<T> aligned_data;
                if (state->head < state->tail) {
                    aligned_data.assign(state->data.begin() + state->head, state->data.begin() + state->tail);
                } else {
                    aligned_data.reserve(state->size);
                    aligned_data.insert(aligned_data.end(), state->data.begin() + state->head, state->data.end());
                    aligned_data.insert(aligned_data.end(), state->data.begin(), state->data.begin() + state->tail);
                }

                size_t i = 0;
                for (; i + vec_size <= aligned_data.size(); i += vec_size) {
                    SimdType data = Traits::load(&aligned_data[i]);
                    data = Traits::multiply(data, value_vec);
                    Traits::store(&aligned_data[i], data);
                }

                if (state->head < state->tail) {
                    std::copy(aligned_data.begin(), aligned_data.end(), state->data.begin() + state->head);
                } else {
                    size_t first_part = state->data.size() - state->head;
                    std::copy(aligned_data.begin(), aligned_data.begin() + first_part, state->data.begin() + state->head);
                    std::copy(aligned_data.begin() + first_part, aligned_data.end(), state->data.begin());
                }

                for (; i < aligned_data.size(); ++i) {
                    aligned_data[i] *= value;
                }
                return;
            }
        }
        
        for (int i = 0; i < state->size; i++) {
            int idx = (state->head + i) % state->data.size();
            state->data[idx] *= value;
        }
    }

    T sum() const {
        if constexpr (detail::SimdTraits<T>::has_simd) {
            using Traits = detail::SimdTraits<T>;
            using SimdType = typename Traits::simd_type;
            
            if (state->size >= Traits::vector_size) {
                SimdType sum_vec = Traits::set1(0);
                const size_t vec_size = Traits::vector_size;
                std::vector<T> aligned_data;
                
                if (state->head < state->tail) {
                    aligned_data.assign(state->data.begin() + state->head, state->data.begin() + state->tail);
                } else {
                    aligned_data.reserve(state->size);
                    aligned_data.insert(aligned_data.end(), state->data.begin() + state->head, state->data.end());
                    aligned_data.insert(aligned_data.end(), state->data.begin(), state->data.begin() + state->tail);
                }

                size_t i = 0;
                for (; i + vec_size <= aligned_data.size(); i += vec_size) {
                    SimdType data = Traits::load(&aligned_data[i]);
                    sum_vec = Traits::add(sum_vec, data);
                }

                T total = Traits::sum(sum_vec);

                for (; i < aligned_data.size(); ++i) {
                    total += aligned_data[i];
                }
                
                return total;
            }
        }
        
        T total = 0;
        for (int i = 0; i < state->size; i++) {
            int idx = (state->head + i) % state->data.size();
            total += state->data[idx];
        }
        return total;
    }

    T max() const {
        if (isEmpty()) throw std::runtime_error("Buffer is empty!");
        T max_val = state->data[state->head];
        for (int i = 1; i < state->size; i++) {
            int idx = (state->head + i) % state->data.size();
            max_val = std::max(max_val, state->data[idx]);
        }
        return max_val;
    }

    std::vector<T> slice(int start, int end) const {
        std::vector<T> result;
        int actual_size = state->size;
        start = (start < 0) ? actual_size + start : start;
        end = (end < 0) ? actual_size + end : end;
        end = std::min(end, actual_size);

        for (int i = start; i < end; i++) {
            int idx = (state->head + i) % state->data.size();
            result.push_back(state->data[idx]);
        }
        return result;
    }

    void map(const std::function<T(T)>& func) {
        for (int i = 0; i < state->size; i++) {
            int idx = (state->head + i) % state->data.size();
            state->data[idx] = func(state->data[idx]);
        }
    }

    void reverse() {
        int left = 0;
        int right = state->size - 1;
        while (left < right) {
            int left_idx = (state->head + left) % state->data.size();
            int right_idx = (state->head + right) % state->data.size();
            std::swap(state->data[left_idx], state->data[right_idx]);
            left++;
            right--;
        }
    }

    void concatenate(const DynamicRingBuffer& other) {
        for (int i = 0; i < other.state->size; i++) {
            int idx = (other.state->head + i) % other.state->data.size();
            this->push(other.state->data[idx]);
        }
    }

    double mean() const {
        if (isEmpty()) throw std::runtime_error("Buffer is empty!");
        if (!state->mean_is_valid) {
            state->cached_mean = static_cast<double>(sum()) / state->size;
            state->mean_is_valid = true;
        }
        return state->cached_mean;
    }

    double variance() const {
        if (state->size < 2) throw std::runtime_error("Need at least 2 elements!");
        double m = mean();  // Use cached mean if available
        double sum_sq = 0;
        
        if (state->head < state->tail) {
            for (int i = state->head; i < state->tail; ++i) {
                double diff = state->data[i] - m;
                sum_sq += diff * diff;
            }
        } else {
            for (int i = state->head; i < state->data.size(); ++i) {
                double diff = state->data[i] - m;
                sum_sq += diff * diff;
            }
            for (int i = 0; i < state->tail; ++i) {
                double diff = state->data[i] - m;
                sum_sq += diff * diff;
            }
        }
        
        return sum_sq / (state->size - 1);
    }

    double stddev() const {
        return std::sqrt(variance());
    }

    T median() {
        if (isEmpty()) throw std::runtime_error("Buffer is empty!");
        
        // Use nth_element instead of full sort
        std::vector<T> temp;
        temp.reserve(state->size);
        
        if (state->head < state->tail) {
            temp.insert(temp.end(), state->data.begin() + state->head, state->data.begin() + state->tail);
        } else {
            temp.insert(temp.end(), state->data.begin() + state->head, state->data.end());
            temp.insert(temp.end(), state->data.begin(), state->data.begin() + state->tail);
        }
        
        if (state->size % 2 == 0) {
            int mid = state->size / 2;
            std::nth_element(temp.begin(), temp.begin() + mid - 1, temp.end());
            T left = temp[mid - 1];
            std::nth_element(temp.begin() + mid, temp.begin() + mid, temp.end());
            return (left + temp[mid]) / T(2);
        } else {
            int mid = state->size / 2;
            std::nth_element(temp.begin(), temp.begin() + mid, temp.end());
            return temp[mid];
        }
    }

    iterator begin() { return iterator(this, 0); }
    iterator end() { return iterator(this, state->size); }
    const_iterator begin() const { return const_iterator(this, 0); }
    const_iterator end() const { return const_iterator(this, state->size); }
    const_iterator cbegin() const { return const_iterator(this, 0); }
    const_iterator cend() const { return const_iterator(this, state->size); }
};

} // namespace drb 