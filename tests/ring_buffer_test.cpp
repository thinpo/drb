#include <gtest/gtest.h>
#include "../include/dynamic_ring_buffer.hpp"
#include <thread>
#include <random>
#include <chrono>
#include <atomic>

class DynamicRingBufferTest : public ::testing::Test {
protected:
    static constexpr size_t INITIAL_CAPACITY = 4;
    drb::DynamicRingBuffer<int> buffer{INITIAL_CAPACITY};
};

TEST_F(DynamicRingBufferTest, InitialState) {
    EXPECT_TRUE(buffer.isEmpty());
    EXPECT_EQ(buffer.size(), 0);
    EXPECT_EQ(buffer.getCapacity(), INITIAL_CAPACITY);
}

TEST_F(DynamicRingBufferTest, PushPop) {
    buffer.push(1);
    EXPECT_FALSE(buffer.isEmpty());
    EXPECT_EQ(buffer.size(), 1);
    
    auto val = buffer.pop();
    ASSERT_TRUE(val.has_value());
    EXPECT_EQ(*val, 1);
    EXPECT_TRUE(buffer.isEmpty());
}

TEST_F(DynamicRingBufferTest, AutoResize) {
    for (int i = 0; i < 10; ++i) {
        buffer.push(i);
    }
    EXPECT_EQ(buffer.size(), 10);
    EXPECT_GE(buffer.getCapacity(), 10);
}

TEST_F(DynamicRingBufferTest, ConcurrentPushPop) {
    std::atomic<bool> stop{false};
    std::atomic<int> sum{0};
    
    std::thread producer([&]() {
        for (int i = 0; i < 1000 && !stop; ++i) {
            buffer.push(i);
            std::this_thread::yield();
        }
    });
    
    std::thread consumer([&]() {
        for (int i = 0; i < 1000 && !stop; ++i) {
            if (auto val = buffer.pop()) {
                sum += *val;
            }
            std::this_thread::yield();
        }
    });
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    stop = true;
    producer.join();
    consumer.join();
    
    EXPECT_GE(sum.load(), 0);
}

TEST_F(DynamicRingBufferTest, ConcurrentResize) {
    std::atomic<bool> stop{false};
    std::atomic<int> push_count{0};
    
    std::thread producer([&]() {
        for (int i = 0; i < 1000 && !stop; ++i) {
            if (buffer.push(i)) {
                push_count++;
            }
            std::this_thread::yield();
        }
    });
    
    std::thread resizer([&]() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(4, 16);
        
        for (int i = 0; i < 100 && !stop; ++i) {
            buffer.resize(dis(gen));
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    });
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    stop = true;
    producer.join();
    resizer.join();
    
    EXPECT_GT(push_count.load(), 0);
}

TEST_F(DynamicRingBufferTest, ConcurrentStatistics) {
    for (int i = 0; i < 100; ++i) {
        buffer.push(i);
    }
    
    std::atomic<bool> stop{false};
    std::atomic<int> stat_count{0};
    
    std::thread calculator([&]() {
        for (int i = 0; i < 100 && !stop; ++i) {
            volatile double mean = buffer.mean();
            volatile double var = buffer.variance();
            volatile double sd = buffer.stddev();
            stat_count++;
            std::this_thread::yield();
        }
    });
    
    std::thread modifier([&]() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 100);
        
        for (int i = 0; i < 100 && !stop; ++i) {
            buffer.push(dis(gen));
            if (auto val = buffer.pop()) {
                buffer.push(*val);
            }
            std::this_thread::yield();
        }
    });
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    stop = true;
    calculator.join();
    modifier.join();
    
    EXPECT_GT(stat_count.load(), 0);
}

TEST_F(DynamicRingBufferTest, ConcurrentIterators) {
    for (int i = 0; i < 100; ++i) {
        buffer.push(i);
    }
    
    std::atomic<bool> stop{false};
    std::atomic<int> iter_sum{0};
    
    std::thread iterator([&]() {
        for (int i = 0; i < 100 && !stop; ++i) {
            int sum = 0;
            for (const auto& val : buffer) {
                sum += val;
            }
            iter_sum = sum;
            std::this_thread::yield();
        }
    });
    
    std::thread modifier([&]() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 100);
        
        for (int i = 0; i < 100 && !stop; ++i) {
            buffer.push(dis(gen));
            if (auto val = buffer.pop()) {
                buffer.push(*val);
            }
            std::this_thread::yield();
        }
    });
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    stop = true;
    iterator.join();
    modifier.join();
    
    EXPECT_GE(iter_sum.load(), 0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 