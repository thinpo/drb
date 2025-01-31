#include "dynamic_ring_buffer.hpp"
#include "market_data.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <vector>
#include <unordered_set>
#include "sale_condition.hpp"

using namespace drb::market;
using namespace std::chrono;

// Demo class for custom type support
class SensorReading {
public:
    double value;
    std::chrono::system_clock::time_point timestamp;

    SensorReading(double v = 0.0)
        : value(v), timestamp(std::chrono::system_clock::now()) {}

    SensorReading& operator+=(const SensorReading& other) {
        value += other.value;
        return *this;
    }

    SensorReading& operator*=(const SensorReading& other) {
        value *= other.value;
        return *this;
    }

    friend std::ostream& operator<<(std::ostream& os, const SensorReading& sr) {
        auto time = std::chrono::system_clock::to_time_t(sr.timestamp);
        os << "[" << std::put_time(std::localtime(&time), "%H:%M:%S") << "] " << sr.value;
        return os;
    }
};

// Demo scenarios
void demoBasicOperations() {
    std::cout << "\n=== Basic Operations Demo ===\n";
    drb::DynamicRingBuffer<int> buffer(3);

    std::cout << "Pushing values: 1, 2, 3, 4\n";
    buffer.push(1);
    buffer.push(2);
    buffer.push(3);
    buffer.push(4);  // This will trigger resize

    std::cout << "Buffer contents: ";
    for (const auto& val : buffer) {
        std::cout << val << " ";
    }
    std::cout << "\nCapacity: " << buffer.getCapacity() << "\n";
}

void demoStatistics() {
    std::cout << "\n=== Statistics Demo ===\n";
    drb::DynamicRingBuffer<double> buffer;

    std::cout << "Adding measurements: 10.5, 20.7, 15.3, 18.2, 12.9\n";
    buffer.push(10.5);
    buffer.push(20.7);
    buffer.push(15.3);
    buffer.push(18.2);
    buffer.push(12.9);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Mean: " << buffer.mean() << "\n";
    std::cout << "Standard Deviation: " << buffer.stddev() << "\n";
    std::cout << "Median: " << buffer.median() << "\n";
}

void demoSensorReadings() {
    std::cout << "\n=== Sensor Readings Demo ===\n";
    drb::DynamicRingBuffer<SensorReading> buffer(5);

    std::cout << "Simulating sensor readings...\n";
    for (int i = 0; i < 5; ++i) {
        buffer.push(SensorReading(20.0 + (rand() % 100) / 10.0));
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "Last 5 readings:\n";
    for (const auto& reading : buffer) {
        std::cout << reading << "\n";
    }
}

void demoTransformations() {
    std::cout << "\n=== Transformations Demo ===\n";
    drb::DynamicRingBuffer<int> buffer;

    std::cout << "Initial values: 1, 2, 3, 4, 5\n";
    for (int i = 1; i <= 5; ++i) {
        buffer.push(i);
    }

    std::cout << "After adding 10 to each element: ";
    buffer.add(10);
    for (const auto& val : buffer) {
        std::cout << val << " ";
    }

    std::cout << "\nAfter multiplying each element by 2: ";
    buffer.multiply(2);
    for (const auto& val : buffer) {
        std::cout << val << " ";
    }

    std::cout << "\nAfter mapping (x -> x/2): ";
    buffer.map([](int x) { return x / 2; });
    for (const auto& val : buffer) {
        std::cout << val << " ";
    }
    std::cout << "\n";
}

template<typename Func>
double measureLatency(Func&& func, int iterations = 1000) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    return static_cast<double>(duration.count()) / iterations;
}

void demoLatencyTests() {
    std::cout << "\n=== Latency Tests ===\n";
    std::cout << std::fixed << std::setprecision(2);

    // Test push operation latency
    {
        drb::DynamicRingBuffer<int> buffer(1000);
        auto pushLatency = measureLatency([&]() {
            buffer.push(42);
        });
        std::cout << "Average push latency: " << pushLatency << " ns\n";
    }

    // Test statistical operations latency
    {
        drb::DynamicRingBuffer<double> buffer(1000);
        // Fill buffer with random values
        for (int i = 0; i < 1000; ++i) {
            buffer.push(rand() % 100 + (rand() % 100) / 100.0);
        }

        auto meanLatency = measureLatency([&]() {
            volatile double m = buffer.mean();
        });
        std::cout << "Average mean() latency: " << meanLatency << " ns\n";

        auto stddevLatency = measureLatency([&]() {
            volatile double sd = buffer.stddev();
        });
        std::cout << "Average stddev() latency: " << stddevLatency << " ns\n";

        auto medianLatency = measureLatency([&]() {
            volatile double med = buffer.median();
        }, 100);  // Fewer iterations for median as it's typically slower
        std::cout << "Average median() latency: " << medianLatency << " ns\n";
    }

    // Test transformation operations latency
    {
        drb::DynamicRingBuffer<int> buffer(1000);
        for (int i = 0; i < 1000; ++i) {
            buffer.push(i);
        }

        auto addLatency = measureLatency([&]() {
            buffer.add(1);
        }, 100);
        std::cout << "Average add() latency: " << addLatency << " ns\n";

        auto multiplyLatency = measureLatency([&]() {
            buffer.multiply(2);
        }, 100);
        std::cout << "Average multiply() latency: " << multiplyLatency << " ns\n";

        auto mapLatency = measureLatency([&]() {
            buffer.map([](int x) { return x + 1; });
        }, 100);
        std::cout << "Average map() latency: " << mapLatency << " ns\n";
    }
}

void demoLargeBuffer() {
    std::cout << "\n=== Large Buffer Tests ===\n";
    const int LARGE_SIZE = 1000000;  // 1 million elements
    drb::DynamicRingBuffer<double> buffer;

    // Test large sequential push
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < LARGE_SIZE; ++i) {
            buffer.push(i);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Time to push " << LARGE_SIZE << " elements: " << duration.count() << "ms\n";
        std::cout << "Final capacity: " << buffer.getCapacity() << "\n";
    }

    // Test statistical operations on large buffer
    {
        auto start = std::chrono::high_resolution_clock::now();
        double mean_val = buffer.mean();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Mean calculation time for " << LARGE_SIZE << " elements: " << duration.count() << "µs\n";
        std::cout << "Mean value: " << mean_val << "\n";
    }

    {
        auto start = std::chrono::high_resolution_clock::now();
        double stddev_val = buffer.stddev();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Standard deviation calculation time for " << LARGE_SIZE << " elements: " << duration.count() << "µs\n";
        std::cout << "StdDev value: " << stddev_val << "\n";
    }

    // Test bulk operations on large buffer
    {
        auto start = std::chrono::high_resolution_clock::now();
        buffer.add(1.0);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Time to add 1.0 to " << LARGE_SIZE << " elements: " << duration.count() << "ms\n";
    }

    // Test iteration performance
    {
        auto start = std::chrono::high_resolution_clock::now();
        double sum = 0;
        for (const auto& val : buffer) {
            sum += val;
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Time to iterate and sum " << LARGE_SIZE << " elements: " << duration.count() << "ms\n";
        std::cout << "Sum: " << sum << "\n";
    }

    // Test random access performance
    {
        auto start = std::chrono::high_resolution_clock::now();
        const int NUM_RANDOM_ACCESS = 10000;
        double sum = 0;
        for (int i = 0; i < NUM_RANDOM_ACCESS; ++i) {
            int random_idx = rand() % LARGE_SIZE;
            auto it = buffer.begin();
            std::advance(it, random_idx);
            sum += *it;
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Time for " << NUM_RANDOM_ACCESS << " random accesses: " << duration.count() << "µs\n";
    }

    // Test median on large dataset
    {
        auto start = std::chrono::high_resolution_clock::now();
        double median_val = buffer.median();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Median calculation time for " << LARGE_SIZE << " elements: " << duration.count() << "ms\n";
        std::cout << "Median value: " << median_val << "\n";
    }
}

void demoSIMD() {
    std::cout << "\n=== SIMD Performance Tests ===\n";
    const int TEST_SIZE = 10000000;  // 10 million elements
    
    // Test float operations (AVX: 8 floats per vector)
    {
        std::cout << "\nFloat Operations (AVX 256-bit):\n";
        drb::DynamicRingBuffer<float> buffer;
        
        // Fill buffer
        for (int i = 0; i < TEST_SIZE; ++i) {
            buffer.push(static_cast<float>(i));
        }

        // Test add operation
        {
            auto start = std::chrono::high_resolution_clock::now();
            buffer.add(1.0f);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "Time to add 1.0 to " << TEST_SIZE << " floats: " << duration.count() << "ms\n";
        }

        // Test multiply operation
        {
            auto start = std::chrono::high_resolution_clock::now();
            buffer.multiply(2.0f);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "Time to multiply " << TEST_SIZE << " floats by 2.0: " << duration.count() << "ms\n";
        }

        // Test sum operation
        {
            auto start = std::chrono::high_resolution_clock::now();
            float sum = buffer.sum();
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "Time to sum " << TEST_SIZE << " floats: " << duration.count() << "ms\n";
            std::cout << "Sum: " << sum << "\n";
        }
    }

    // Test double operations (AVX: 4 doubles per vector)
    {
        std::cout << "\nDouble Operations (AVX 256-bit):\n";
        drb::DynamicRingBuffer<double> buffer;
        
        // Fill buffer
        for (int i = 0; i < TEST_SIZE; ++i) {
            buffer.push(static_cast<double>(i));
        }

        // Test add operation
        {
            auto start = std::chrono::high_resolution_clock::now();
            buffer.add(1.0);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "Time to add 1.0 to " << TEST_SIZE << " doubles: " << duration.count() << "ms\n";
        }

        // Test multiply operation
        {
            auto start = std::chrono::high_resolution_clock::now();
            buffer.multiply(2.0);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "Time to multiply " << TEST_SIZE << " doubles by 2.0: " << duration.count() << "ms\n";
        }

        // Test sum operation
        {
            auto start = std::chrono::high_resolution_clock::now();
            double sum = buffer.sum();
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "Time to sum " << TEST_SIZE << " doubles: " << duration.count() << "ms\n";
            std::cout << "Sum: " << sum << "\n";
        }
    }

    // Test non-SIMD type (int) for comparison
    {
        std::cout << "\nInteger Operations (No SIMD):\n";
        drb::DynamicRingBuffer<int> buffer;
        
        // Fill buffer
        for (int i = 0; i < TEST_SIZE; ++i) {
            buffer.push(i);
        }

        // Test add operation
        {
            auto start = std::chrono::high_resolution_clock::now();
            buffer.add(1);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "Time to add 1 to " << TEST_SIZE << " integers: " << duration.count() << "ms\n";
        }

        // Test multiply operation
        {
            auto start = std::chrono::high_resolution_clock::now();
            buffer.multiply(2);
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "Time to multiply " << TEST_SIZE << " integers by 2: " << duration.count() << "ms\n";
        }

        // Test sum operation
        {
            auto start = std::chrono::high_resolution_clock::now();
            int sum = buffer.sum();
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "Time to sum " << TEST_SIZE << " integers: " << duration.count() << "ms\n";
            std::cout << "Sum: " << sum << "\n";
        }
    }
}

void demoMarketData() {
    std::cout << "\n=== Market Data Demo ===\n";
    
    // Create buffers for trades and quotes
    drb::market::TradeBuffer trade_buffer;
    drb::market::QuoteBuffer quote_buffer;
    
    const std::string symbol = "AAPL";
    std::unordered_set<SaleCondition> excluded_conditions = {
        sale_condition::from_string("Z  "), // Out of sequence
        sale_condition::from_string("L  ")  // Late
    };
    
    // Simulate a trading day with some sample data
    std::cout << "\nSimulating market data...\n";
    
    // Morning session trades
    trade_buffer.push(drb::market::Trade(symbol, 150.00, 100, "@  ")); // Regular trade
    trade_buffer.push(drb::market::Trade(symbol, 150.25, 200, "F  ")); // Intermarket sweep
    trade_buffer.push(drb::market::Trade(symbol, 150.15, 150, "Z  ")); // Out of sequence - should be filtered
    trade_buffer.push(drb::market::Trade(symbol, 150.50, 300, "C  ")); // Cash trade
    trade_buffer.push(drb::market::Trade(symbol, 150.75, 250, "E  ")); // Automatic execution
    
    // Morning session quotes
    quote_buffer.push(drb::market::Quote(symbol, 150.00, 150.05, 100, 100));
    quote_buffer.push(drb::market::Quote(symbol, 150.20, 150.25, 200, 200));
    quote_buffer.push(drb::market::Quote(symbol, 150.45, 150.50, 150, 150));
    quote_buffer.push(drb::market::Quote(symbol, 150.70, 150.80, 300, 300));
    
    // Calculate 1-minute OHLC bars
    std::cout << "\nCalculating 1-minute OHLC bars:\n";
    auto ohlc_bars = drb::market::calculateOHLC(trade_buffer, symbol, std::chrono::minutes(1), excluded_conditions);
    
    for (const auto& bar : ohlc_bars) {
        auto time = std::chrono::system_clock::to_time_t(bar.timestamp);
        std::cout << "Time: " << std::put_time(std::localtime(&time), "%H:%M:%S") << "\n"
                  << "  Open:  " << bar.open << "\n"
                  << "  High:  " << bar.high << "\n"
                  << "  Low:   " << bar.low << "\n"
                  << "  Close: " << bar.close << "\n"
                  << "  Volume: " << bar.volume << "\n"
                  << "  Trades: " << bar.trade_count << "\n";
    }
    
    // Get filtered trades
    std::cout << "\nFiltered trades:\n";
    auto filtered_trades = drb::market::getFilteredTrades(trade_buffer, symbol, excluded_conditions);
    for (const auto& trade : filtered_trades) {
        auto time = std::chrono::system_clock::to_time_t(trade.timestamp);
        std::cout << std::put_time(std::localtime(&time), "%H:%M:%S")
                  << " - Price: " << trade.price
                  << ", Size: " << trade.size 
                  << ", Condition: " << sale_condition::to_string(trade.sale_condition) << "\n";
    }
    
    // Calculate VWAP
    double vwap = drb::market::calculateVWAP(trade_buffer, symbol, excluded_conditions);
    std::cout << "\nVWAP: " << vwap << "\n";
    
    // Get best bid/ask
    auto now = std::chrono::system_clock::now();
    auto [best_bid, best_ask] = drb::market::getBestBidAsk(quote_buffer, symbol, now);
    std::cout << "\nBest Bid: " << best_bid << "\n"
              << "Best Ask: " << best_ask << "\n"
              << "Spread: " << (best_ask - best_bid) << "\n";
}

void print_trade(const Trade& trade) {
    std::cout << std::fixed << std::setprecision(2)
              << trade.symbol << " @ $" << trade.price 
              << " x " << trade.size 
              << " [" << sale_condition::to_string(trade.sale_condition) << "]"
              << " on " << trade.exchange << "\n";
}

void demo_sale_conditions() {
    std::cout << "\n=== Sale Condition Demo ===\n";
    
    // Create a trade buffer
    TradeBuffer buffer(10);
    
    // Add trades with various UTP conditions
    buffer.push(Trade("AAPL", 150.50, 100, "@  ", "NASDAQ")); // Regular sale
    buffer.push(Trade("AAPL", 150.55, 200, "F T", "NASDAQ")); // Intermarket sweep, Form T
    buffer.push(Trade("AAPL", 150.45, 150, "I  ", "NASDAQ")); // Odd lot
    buffer.push(Trade("AAPL", 150.60, 300, "XC ", "NASDAQ")); // Cross trade, Cash
    
    std::cout << "\nAll trades:\n";
    for (const auto& trade : buffer) {
        print_trade(trade);
    }
    
    // Filter out odd lots and Form T trades
    std::unordered_set<SaleCondition> excluded = {
        sale_condition::from_string("I  "),
        sale_condition::from_string("F T")
    };
    
    std::cout << "\nFiltered trades (excluding odd lots and Form T):\n";
    auto filtered = getFilteredTrades(buffer, "AAPL", excluded);
    for (const auto& trade : filtered) {
        print_trade(trade);
    }
    
    // Validate UTP and CTA conditions
    std::cout << "\nValidating sale conditions:\n";
    SaleCondition utp_cond = sale_condition::from_string("@FT");
    SaleCondition cta_cond = sale_condition::from_string("@CT");
    
    std::cout << "UTP condition '@FT': " 
              << (sale_condition::is_valid_utp_condition(utp_cond) ? "Valid" : "Invalid")
              << " for UTP\n";
              
    std::cout << "CTA condition '@CT': " 
              << (sale_condition::is_valid_cta_condition(cta_cond) ? "Valid" : "Invalid")
              << " for CTA\n";
    
    // Calculate VWAP excluding certain conditions
    std::unordered_set<SaleCondition> vwap_excluded = {
        sale_condition::from_string("I  "), // Exclude odd lots
        sale_condition::from_string("X  ")  // Exclude cross trades
    };
    
    double vwap = calculateVWAP(buffer, "AAPL", vwap_excluded);
    std::cout << "\nVWAP (excluding odd lots and cross trades): $" << std::fixed << std::setprecision(2) << vwap << "\n";
    
    // Calculate OHLC bars
    std::cout << "\nOHLC bars (1 second intervals):\n";
    auto bars = calculateOHLC(buffer, "AAPL", seconds(1), vwap_excluded);
    for (const auto& bar : bars) {
        std::cout << "O: " << bar.open << " H: " << bar.high 
                  << " L: " << bar.low << " C: " << bar.close 
                  << " V: " << bar.volume << " #: " << bar.trade_count << "\n";
    }
}

void demoThreadSafetyAndAPL() {
    std::cout << "\n=== Thread Safety and APL Operations Demo ===\n";
    
    // Create two buffers
    drb::DynamicRingBuffer<int> buffer1;
    drb::DynamicRingBuffer<int> buffer2;
    
    // Fill buffers
    for (int i = 1; i <= 3; ++i) buffer1.push(i);  // [1, 2, 3]
    for (int i = 4; i <= 6; ++i) buffer2.push(i);  // [4, 5, 6]
    
    // Demonstrate outer product (multiplication)
    std::cout << "\nOuter product (multiplication):\n";
    auto outer = buffer1.outer_product(buffer2, std::multiplies<int>());
    for (const auto& row : outer) {
        for (int val : row) {
            std::cout << std::setw(4) << val;
        }
        std::cout << '\n';
    }
    
    // Demonstrate compress
    std::vector<bool> mask = {true, false, true};  // Keep first and third elements
    auto compressed = buffer1.compress(mask);
    std::cout << "\nCompress ([1,2,3] with mask [1,0,1]):\n";
    for (const auto& val : compressed) {
        std::cout << val << " ";
    }
    std::cout << '\n';
    
    // Demonstrate expand
    std::vector<bool> expand_mask = {true, false, true, false, true};
    auto expanded = buffer1.expand(expand_mask, 0);
    std::cout << "\nExpand ([1,2,3] with mask [1,0,1,0,1]):\n";
    for (const auto& val : expanded) {
        std::cout << val << " ";
    }
    std::cout << '\n';
    
    // Demonstrate scan (prefix sum)
    auto scanned = buffer1.scan(std::plus<int>());
    std::cout << "\nPrefix sum:\n";
    for (const auto& val : scanned) {
        std::cout << val << " ";
    }
    std::cout << '\n';
    
    // Demonstrate rotate
    buffer1.rotate(1);
    std::cout << "\nAfter rotating right by 1:\n";
    for (const auto& val : buffer1) {
        std::cout << val << " ";
    }
    std::cout << '\n';
    
    // Demonstrate thread safety
    std::cout << "\nTesting thread safety with concurrent operations...\n";
    drb::DynamicRingBuffer<int> shared_buffer;
    std::vector<std::thread> threads;
    
    // Producer threads
    for (int i = 0; i < 3; ++i) {
        threads.emplace_back([&shared_buffer, i]() {
            for (int j = 0; j < 100; ++j) {
                shared_buffer.push(i * 100 + j);
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
    }
    
    // Consumer threads
    for (int i = 0; i < 2; ++i) {
        threads.emplace_back([&shared_buffer]() {
            for (int j = 0; j < 150; ++j) {
                if (!shared_buffer.isEmpty()) {
                    shared_buffer.pop();
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    std::cout << "Final buffer size after concurrent operations: " 
              << shared_buffer.getSize() << "\n";
}

int main() {
    try {
        demoBasicOperations();
        demoStatistics();
        demoSensorReadings();
        demoTransformations();
        demoLatencyTests();
        demoLargeBuffer();
        demoSIMD();
        demoMarketData();  // Add market data demo
        demo_sale_conditions();
        demoThreadSafetyAndAPL();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
} 