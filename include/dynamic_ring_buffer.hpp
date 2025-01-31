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
    std::vector<T> buffer;
    int head;
    int tail;
    int size;
    static constexpr double SHRINK_FACTOR = 0.25;
    mutable double cached_mean;
    mutable bool mean_is_valid;

    void resize(int new_capacity) {
        std::vector<T> new_buffer(new_capacity);
        int old_count = size;

        if (head < tail) {
            std::copy(buffer.begin() + head, buffer.begin() + tail, new_buffer.begin());
        } else {
            std::copy(buffer.begin() + head, buffer.end(), new_buffer.begin());
            std::copy(buffer.begin(), buffer.begin() + tail, new_buffer.begin() + (buffer.size() - head));
        }

        head = 0;
        tail = old_count;
        buffer = std::move(new_buffer);
        mean_is_valid = false;
    }

    template<bool IsConst>
    class Iterator {
    private:
        using BufferType = typename std::conditional<IsConst, const DynamicRingBuffer*, DynamicRingBuffer*>::type;
        BufferType rb;
        int index;
        int actual_index;  // Cache the actual index to avoid repeated modulo

    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;
        using pointer = typename std::conditional<IsConst, const T*, T*>::type;
        using reference = typename std::conditional<IsConst, const T&, T&>::type;

        Iterator(BufferType rb, int index) : rb(rb), index(index) {
            actual_index = (rb->head + index) % rb->buffer.size();
        }

        reference operator*() { return rb->buffer[actual_index]; }
        pointer operator->() { return &(operator*()); }

        Iterator& operator++() {
            ++index;
            actual_index = (rb->head + index) % rb->buffer.size();
            return *this;
        }

        Iterator operator++(int) { 
            Iterator tmp = *this; 
            ++(*this);
            return tmp; 
        }

        Iterator& operator--() {
            --index;
            actual_index = (rb->head + index) % rb->buffer.size();
            return *this;
        }

        Iterator operator--(int) { 
            Iterator tmp = *this; 
            --(*this);
            return tmp; 
        }

        Iterator& operator+=(difference_type n) {
            index += n;
            actual_index = (rb->head + index) % rb->buffer.size();
            return *this;
        }

        Iterator operator+(difference_type n) const {
            Iterator tmp = *this;
            return tmp += n;
        }

        Iterator& operator-=(difference_type n) {
            return *this += -n;
        }

        Iterator operator-(difference_type n) const {
            Iterator tmp = *this;
            return tmp -= n;
        }

        difference_type operator-(const Iterator& other) const {
            return index - other.index;
        }

        bool operator==(const Iterator& other) const { return index == other.index; }
        bool operator!=(const Iterator& other) const { return !(*this == other); }
        bool operator<(const Iterator& other) const { return index < other.index; }
        bool operator>(const Iterator& other) const { return other < *this; }
        bool operator<=(const Iterator& other) const { return !(other < *this); }
        bool operator>=(const Iterator& other) const { return !(*this < other); }

        reference operator[](difference_type n) const {
            return *(*this + n);
        }
    };

    friend Iterator<false> operator+(typename Iterator<false>::difference_type n, const Iterator<false>& it) {
        return it + n;
    }

    friend Iterator<true> operator+(typename Iterator<true>::difference_type n, const Iterator<true>& it) {
        return it + n;
    }

public:
    using iterator = Iterator<false>;
    using const_iterator = Iterator<true>;

    explicit DynamicRingBuffer(int initial_capacity = 4)
        : buffer(initial_capacity), head(0), tail(0), size(0), mean_is_valid(false) {}

    ~DynamicRingBuffer() {
        // No need to delete buffer, std::vector handles it
    }

    DynamicRingBuffer(const DynamicRingBuffer& other)
        : buffer(other.buffer), head(other.head), tail(other.tail), size(other.size), mean_is_valid(false) {}

    DynamicRingBuffer(DynamicRingBuffer&& other) noexcept
        : buffer(std::move(other.buffer)), head(other.head), tail(other.tail), size(other.size), mean_is_valid(false) {
        other.head = other.tail = other.size = 0;
    }

    void push(const T& value) {
        if (isFull()) {
            resize(buffer.size() * 2);
        }
        buffer[tail] = value;
        tail = (tail + 1) % buffer.size();
        size++;
        mean_is_valid = false;
    }

    T pop() {
        if (isEmpty()) {
            throw std::runtime_error("Buffer is empty!");
        }
        T value = buffer[head];
        head = (head + 1) % buffer.size();
        size--;

        if (size < buffer.size() * SHRINK_FACTOR && buffer.size() > 4) {
            resize(buffer.size() / 2);
        }
        return value;
    }

    bool isEmpty() const { return size == 0; }
    bool isFull() const { return size == buffer.size(); }
    int getCapacity() const { return buffer.size(); }
    int getSize() const { return size; }

    void add(T value) {
        if constexpr (detail::SimdTraits<T>::has_simd) {
            using Traits = detail::SimdTraits<T>;
            using SimdType = typename Traits::simd_type;
            
            if (size >= Traits::vector_size) {
                const SimdType value_vec = Traits::set1(value);
                const size_t vec_size = Traits::vector_size;
                
                std::vector<T> aligned_data;
                if (head < tail) {
                    aligned_data.assign(buffer.begin() + head, buffer.begin() + tail);
                } else {
                    aligned_data.reserve(size);
                    aligned_data.insert(aligned_data.end(), buffer.begin() + head, buffer.end());
                    aligned_data.insert(aligned_data.end(), buffer.begin(), buffer.begin() + tail);
                }

                size_t i = 0;
                for (; i + vec_size <= aligned_data.size(); i += vec_size) {
                    SimdType data = Traits::load(&aligned_data[i]);
                    data = Traits::add(data, value_vec);
                    Traits::store(&aligned_data[i], data);
                }

                if (head < tail) {
                    std::copy(aligned_data.begin(), aligned_data.end(), buffer.begin() + head);
                } else {
                    size_t first_part = buffer.size() - head;
                    std::copy(aligned_data.begin(), aligned_data.begin() + first_part, buffer.begin() + head);
                    std::copy(aligned_data.begin() + first_part, aligned_data.end(), buffer.begin());
                }

                for (; i < aligned_data.size(); ++i) {
                    aligned_data[i] += value;
                }
                return;
            }
        }
        
        for (int i = 0; i < size; i++) {
            int idx = (head + i) % buffer.size();
            buffer[idx] += value;
        }
    }

    void multiply(T value) {
        if constexpr (detail::SimdTraits<T>::has_simd) {
            using Traits = detail::SimdTraits<T>;
            using SimdType = typename Traits::simd_type;
            
            if (size >= Traits::vector_size) {
                const SimdType value_vec = Traits::set1(value);
                const size_t vec_size = Traits::vector_size;
                
                std::vector<T> aligned_data;
                if (head < tail) {
                    aligned_data.assign(buffer.begin() + head, buffer.begin() + tail);
                } else {
                    aligned_data.reserve(size);
                    aligned_data.insert(aligned_data.end(), buffer.begin() + head, buffer.end());
                    aligned_data.insert(aligned_data.end(), buffer.begin(), buffer.begin() + tail);
                }

                size_t i = 0;
                for (; i + vec_size <= aligned_data.size(); i += vec_size) {
                    SimdType data = Traits::load(&aligned_data[i]);
                    data = Traits::multiply(data, value_vec);
                    Traits::store(&aligned_data[i], data);
                }

                if (head < tail) {
                    std::copy(aligned_data.begin(), aligned_data.end(), buffer.begin() + head);
                } else {
                    size_t first_part = buffer.size() - head;
                    std::copy(aligned_data.begin(), aligned_data.begin() + first_part, buffer.begin() + head);
                    std::copy(aligned_data.begin() + first_part, aligned_data.end(), buffer.begin());
                }

                for (; i < aligned_data.size(); ++i) {
                    aligned_data[i] *= value;
                }
                return;
            }
        }
        
        for (int i = 0; i < size; i++) {
            int idx = (head + i) % buffer.size();
            buffer[idx] *= value;
        }
    }

    T sum() const {
        if constexpr (detail::SimdTraits<T>::has_simd) {
            using Traits = detail::SimdTraits<T>;
            using SimdType = typename Traits::simd_type;
            
            if (size >= Traits::vector_size) {
                SimdType sum_vec = Traits::set1(0);
                const size_t vec_size = Traits::vector_size;
                std::vector<T> aligned_data;
                
                if (head < tail) {
                    aligned_data.assign(buffer.begin() + head, buffer.begin() + tail);
                } else {
                    aligned_data.reserve(size);
                    aligned_data.insert(aligned_data.end(), buffer.begin() + head, buffer.end());
                    aligned_data.insert(aligned_data.end(), buffer.begin(), buffer.begin() + tail);
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
        for (int i = 0; i < size; i++) {
            int idx = (head + i) % buffer.size();
            total += buffer[idx];
        }
        return total;
    }

    T max() const {
        if (isEmpty()) throw std::runtime_error("Buffer is empty!");
        T max_val = buffer[head];
        for (int i = 1; i < size; i++) {
            int idx = (head + i) % buffer.size();
            max_val = std::max(max_val, buffer[idx]);
        }
        return max_val;
    }

    std::vector<T> slice(int start, int end) const {
        std::vector<T> result;
        int actual_size = size;
        start = (start < 0) ? actual_size + start : start;
        end = (end < 0) ? actual_size + end : end;
        end = std::min(end, actual_size);

        for (int i = start; i < end; i++) {
            int idx = (head + i) % buffer.size();
            result.push_back(buffer[idx]);
        }
        return result;
    }

    void map(const std::function<T(T)>& func) {
        for (int i = 0; i < size; i++) {
            int idx = (head + i) % buffer.size();
            buffer[idx] = func(buffer[idx]);
        }
    }

    void reverse() {
        int left = 0;
        int right = size - 1;
        while (left < right) {
            int left_idx = (head + left) % buffer.size();
            int right_idx = (head + right) % buffer.size();
            std::swap(buffer[left_idx], buffer[right_idx]);
            left++;
            right--;
        }
    }

    void concatenate(const DynamicRingBuffer& other) {
        for (int i = 0; i < other.size; i++) {
            int idx = (other.head + i) % other.buffer.size();
            this->push(other.buffer[idx]);
        }
    }

    double mean() const {
        if (isEmpty()) throw std::runtime_error("Buffer is empty!");
        if (!mean_is_valid) {
            cached_mean = static_cast<double>(sum()) / size;
            mean_is_valid = true;
        }
        return cached_mean;
    }

    double variance() const {
        if (size < 2) throw std::runtime_error("Need at least 2 elements!");
        double m = mean();  // Use cached mean if available
        double sum_sq = 0;
        
        if (head < tail) {
            for (int i = head; i < tail; ++i) {
                double diff = buffer[i] - m;
                sum_sq += diff * diff;
            }
        } else {
            for (int i = head; i < buffer.size(); ++i) {
                double diff = buffer[i] - m;
                sum_sq += diff * diff;
            }
            for (int i = 0; i < tail; ++i) {
                double diff = buffer[i] - m;
                sum_sq += diff * diff;
            }
        }
        
        return sum_sq / (size - 1);
    }

    double stddev() const {
        return std::sqrt(variance());
    }

    T median() {
        if (isEmpty()) throw std::runtime_error("Buffer is empty!");
        
        // Use nth_element instead of full sort
        std::vector<T> temp;
        temp.reserve(size);
        
        if (head < tail) {
            temp.insert(temp.end(), buffer.begin() + head, buffer.begin() + tail);
        } else {
            temp.insert(temp.end(), buffer.begin() + head, buffer.end());
            temp.insert(temp.end(), buffer.begin(), buffer.begin() + tail);
        }
        
        if (size % 2 == 0) {
            int mid = size / 2;
            std::nth_element(temp.begin(), temp.begin() + mid - 1, temp.end());
            T left = temp[mid - 1];
            std::nth_element(temp.begin() + mid, temp.begin() + mid, temp.end());
            return (left + temp[mid]) / T(2);
        } else {
            int mid = size / 2;
            std::nth_element(temp.begin(), temp.begin() + mid, temp.end());
            return temp[mid];
        }
    }

    iterator begin() { return iterator(this, 0); }
    iterator end() { return iterator(this, size); }
    const_iterator begin() const { return const_iterator(this, 0); }
    const_iterator end() const { return const_iterator(this, size); }
    const_iterator cbegin() const { return const_iterator(this, 0); }
    const_iterator cend() const { return const_iterator(this, size); }
};

} // namespace drb 