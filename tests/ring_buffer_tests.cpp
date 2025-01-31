#include <gtest/gtest.h>
#include "dynamic_ring_buffer.hpp"

class RingBufferTest : public ::testing::Test {
protected:
    drb::DynamicRingBuffer<int> buffer;
};

TEST_F(RingBufferTest, InitiallyEmpty) {
    EXPECT_TRUE(buffer.isEmpty());
    EXPECT_EQ(buffer.getSize(), 0);
    EXPECT_EQ(buffer.getCapacity(), 4);  // Default initial capacity
}

TEST_F(RingBufferTest, PushAndPop) {
    buffer.push(1);
    buffer.push(2);
    buffer.push(3);

    EXPECT_EQ(buffer.getSize(), 3);
    EXPECT_FALSE(buffer.isEmpty());

    EXPECT_EQ(buffer.pop(), 1);
    EXPECT_EQ(buffer.pop(), 2);
    EXPECT_EQ(buffer.pop(), 3);

    EXPECT_TRUE(buffer.isEmpty());
}

TEST_F(RingBufferTest, AutoResize) {
    int initial_capacity = buffer.getCapacity();

    // Fill buffer to trigger resize
    for (int i = 0; i < initial_capacity + 1; ++i) {
        buffer.push(i);
    }

    EXPECT_GT(buffer.getCapacity(), initial_capacity);
    EXPECT_EQ(buffer.getSize(), initial_capacity + 1);
}

TEST_F(RingBufferTest, Iterator) {
    std::vector<int> values = {1, 2, 3, 4, 5};
    for (int val : values) {
        buffer.push(val);
    }

    std::vector<int> retrieved;
    for (const auto& val : buffer) {
        retrieved.push_back(val);
    }

    EXPECT_EQ(values, retrieved);
}

TEST_F(RingBufferTest, Statistics) {
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0, 5.0};
    drb::DynamicRingBuffer<double> double_buffer;
    
    for (double val : values) {
        double_buffer.push(val);
    }

    EXPECT_DOUBLE_EQ(double_buffer.mean(), 3.0);
    EXPECT_DOUBLE_EQ(double_buffer.median(), 3.0);
    EXPECT_NEAR(double_buffer.stddev(), 1.5811388300841898, 1e-10);
}

TEST_F(RingBufferTest, BulkOperations) {
    std::vector<int> values = {1, 2, 3, 4, 5};
    for (int val : values) {
        buffer.push(val);
    }

    buffer.add(10);
    std::vector<int> expected_add = {11, 12, 13, 14, 15};
    std::vector<int> result_add;
    for (const auto& val : buffer) {
        result_add.push_back(val);
    }
    EXPECT_EQ(expected_add, result_add);

    buffer.multiply(2);
    std::vector<int> expected_mul = {22, 24, 26, 28, 30};
    std::vector<int> result_mul;
    for (const auto& val : buffer) {
        result_mul.push_back(val);
    }
    EXPECT_EQ(expected_mul, result_mul);
}

TEST_F(RingBufferTest, ExceptionHandling) {
    EXPECT_THROW(buffer.pop(), std::runtime_error);  // Empty buffer
    
    drb::DynamicRingBuffer<double> double_buffer;
    EXPECT_THROW(double_buffer.mean(), std::runtime_error);  // Empty buffer
    EXPECT_THROW(double_buffer.stddev(), std::runtime_error);  // Need at least 2 elements
    
    double_buffer.push(1.0);
    EXPECT_THROW(double_buffer.stddev(), std::runtime_error);  // Still need 2 elements
} 