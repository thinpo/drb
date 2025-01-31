#include <gtest/gtest.h>
#include "dynamic_ring_buffer.hpp"
#include <string>
#include <unordered_map>

class MapReduceTest : public ::testing::Test {
protected:
    drb::DynamicRingBuffer<std::string> text_buffer;
    drb::DynamicRingBuffer<int> number_buffer;

    void SetUp() override {
        // Add sample text data
        text_buffer.push("hello world");
        text_buffer.push("hello mapreduce");
        text_buffer.push("testing mapreduce implementation");
        text_buffer.push("world of distributed computing");

        // Add sample numeric data
        for (int i = 1; i <= 100; ++i) {
            number_buffer.push(i);
        }
    }
};

TEST_F(MapReduceTest, WordCount) {
    auto word_counts = text_buffer.mapReduce<std::string, size_t>(
        // Map: Split into words and count each as 1
        [](const std::string& line) {
            std::string word;
            std::vector<std::pair<std::string, size_t>> result;
            
            for (char c : line) {
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
        // Reduce: Sum the counts
        [](size_t a, size_t b) { return a + b; }
    );

    EXPECT_EQ(word_counts["hello"], 2);
    EXPECT_EQ(word_counts["world"], 2);
    EXPECT_EQ(word_counts["mapreduce"], 2);
    EXPECT_EQ(word_counts["testing"], 1);
    EXPECT_EQ(word_counts["implementation"], 1);
    EXPECT_EQ(word_counts["of"], 1);
    EXPECT_EQ(word_counts["distributed"], 1);
    EXPECT_EQ(word_counts["computing"], 1);
}

TEST_F(MapReduceTest, NumberAggregation) {
    // Test sum of squares
    auto sum_of_squares = number_buffer.parallelAggregate<int64_t>(
        [](int x) { return static_cast<int64_t>(x) * x; },  // Map: Square each number
        [](int64_t a, int64_t b) { return a + b; },         // Reduce: Sum
        0LL
    );

    // Formula for sum of squares: n(n+1)(2n+1)/6
    int64_t expected = (100LL * 101LL * 201LL) / 6LL;
    EXPECT_EQ(sum_of_squares, expected);

    // Test parallel filtering of even numbers
    auto even_numbers = number_buffer.parallelFilter(
        [](int x) { return x % 2 == 0; }
    );

    EXPECT_EQ(even_numbers.getSize(), 50);  // Should have 50 even numbers
    
    // Verify all numbers are even
    for (const auto& num : even_numbers) {
        EXPECT_EQ(num % 2, 0);
    }
}

TEST_F(MapReduceTest, ParallelProcessing) {
    // Test parallel processing with different numbers of workers
    std::vector<size_t> worker_counts = {1, 2, 4, 8};
    
    for (size_t workers : worker_counts) {
        auto result = number_buffer.mapReduce<int, int64_t>(
            // Map: Create key-value pairs where key is number mod 3
            [](int x) { return std::make_pair(x % 3, static_cast<int64_t>(x)); },
            // Reduce: Sum values
            [](int64_t a, int64_t b) { return a + b; },
            workers
        );

        // Verify results are consistent regardless of worker count
        EXPECT_EQ(result.size(), 3);
        int64_t total = 0;
        for (const auto& [key, value] : result) {
            total += value;
        }
        EXPECT_EQ(total, 5050);  // Sum of numbers 1 to 100
    }
}

TEST_F(MapReduceTest, EmptyBuffer) {
    drb::DynamicRingBuffer<std::string> empty_buffer;
    
    auto result = empty_buffer.mapReduce<std::string, int>(
        [](const std::string&) { return std::make_pair(std::string(), 1); },
        [](int a, int b) { return a + b; }
    );

    EXPECT_TRUE(result.empty());
}

TEST_F(MapReduceTest, LargeDataSet) {
    drb::DynamicRingBuffer<int> large_buffer;
    const size_t size = 1000000;  // 1M elements
    
    // Fill buffer with numbers
    for (size_t i = 0; i < size; ++i) {
        large_buffer.push(static_cast<int>(i));
    }

    // Test parallel sum calculation
    auto sum = large_buffer.parallelAggregate<int64_t>(
        [](int x) { return static_cast<int64_t>(x); },
        [](int64_t a, int64_t b) { return a + b; },
        0LL
    );

    // Expected sum using arithmetic sequence formula: n(n-1)/2
    int64_t expected = (static_cast<int64_t>(size) * (size - 1)) / 2;
    EXPECT_EQ(sum, expected);
} 