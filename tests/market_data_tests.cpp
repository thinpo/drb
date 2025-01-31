#include <gtest/gtest.h>
#include "market_data.hpp"
#include <chrono>
#include <thread>

class MarketDataTest : public ::testing::Test {
protected:
    drb::market::TradeBuffer trade_buffer;
    drb::market::QuoteBuffer quote_buffer;
    const std::string test_symbol = "TEST";
    std::unordered_set<drb::market::SaleCondition> excluded_conditions;

    void SetUp() override {
        // Add some test trades
        trade_buffer.push(drb::market::Trade(test_symbol, 100.00, 100));
        trade_buffer.push(drb::market::Trade(test_symbol, 101.00, 200));
        trade_buffer.push(drb::market::Trade(test_symbol, 100.50, 150, "OUT"));  // Out of sequence
        trade_buffer.push(drb::market::Trade(test_symbol, 102.00, 300));
        trade_buffer.push(drb::market::Trade(test_symbol, 101.50, 250));

        // Set up excluded conditions
        excluded_conditions.insert(drb::market::sale_condition::from_string("OUT"));

        // Add some test quotes
        quote_buffer.push(drb::market::Quote(test_symbol, 100.00, 100.10, 100, 100));
        quote_buffer.push(drb::market::Quote(test_symbol, 100.20, 100.30, 200, 200));
        quote_buffer.push(drb::market::Quote(test_symbol, 100.40, 100.50, 150, 150));
        quote_buffer.push(drb::market::Quote(test_symbol, 100.60, 100.70, 300, 300));
    }
};

TEST_F(MarketDataTest, FilteredTrades) {
    auto filtered = drb::market::getFilteredTrades(trade_buffer, test_symbol, excluded_conditions);
    EXPECT_EQ(filtered.size(), 4);  // One trade should be filtered out
    
    // Verify the filtered trade is not included
    auto it = std::find_if(filtered.begin(), filtered.end(),
        [](const drb::market::Trade& t) { 
            return t.sale_condition == drb::market::sale_condition::from_string("OUT"); 
        });
    EXPECT_EQ(it, filtered.end());
}

TEST_F(MarketDataTest, VWAP) {
    double vwap = drb::market::calculateVWAP(trade_buffer, test_symbol, excluded_conditions);
    
    // Manual VWAP calculation for verification
    // (100.00 * 100 + 101.00 * 200 + 102.00 * 300 + 101.50 * 250) / (100 + 200 + 300 + 250)
    // Note: 100.50 trade is excluded due to "OUT" condition
    double expected_vwap = (100.00 * 100 + 101.00 * 200 + 102.00 * 300 + 101.50 * 250) / 
                          static_cast<double>(100 + 200 + 300 + 250);
    
    EXPECT_NEAR(vwap, expected_vwap, 1e-10);
}

TEST_F(MarketDataTest, OHLC) {
    auto ohlc = drb::market::calculateOHLC(trade_buffer, test_symbol, 
                                         std::chrono::minutes(1), excluded_conditions);
    EXPECT_EQ(ohlc.size(), 1);  // All trades should be in one bar
    
    if (!ohlc.empty()) {
        auto& bar = ohlc[0];
        EXPECT_DOUBLE_EQ(bar.open, 100.00);
        EXPECT_DOUBLE_EQ(bar.high, 102.00);
        EXPECT_DOUBLE_EQ(bar.low, 100.00);
        EXPECT_DOUBLE_EQ(bar.close, 101.50);
        EXPECT_EQ(bar.volume, 850);  // Total volume excluding filtered trade
        EXPECT_EQ(bar.trade_count, 4);  // Number of trades excluding filtered
    }
}

TEST_F(MarketDataTest, BestBidAsk) {
    auto now = std::chrono::system_clock::now();
    auto [best_bid, best_ask] = drb::market::getBestBidAsk(quote_buffer, test_symbol, now);
    
    EXPECT_DOUBLE_EQ(best_bid, 100.60);  // Highest bid
    EXPECT_DOUBLE_EQ(best_ask, 100.10);  // Lowest ask
}

TEST_F(MarketDataTest, MultipleSymbols) {
    // Add trades for a different symbol
    const std::string other_symbol = "OTHER";
    trade_buffer.push(drb::market::Trade(other_symbol, 200.00, 100));
    trade_buffer.push(drb::market::Trade(other_symbol, 201.00, 200));
    
    auto filtered_test = drb::market::getFilteredTrades(trade_buffer, test_symbol, excluded_conditions);
    auto filtered_other = drb::market::getFilteredTrades(trade_buffer, other_symbol, excluded_conditions);
    
    EXPECT_EQ(filtered_test.size(), 4);  // Original trades minus filtered
    EXPECT_EQ(filtered_other.size(), 2);  // New symbol trades
}

TEST_F(MarketDataTest, TimeBasedOHLC) {
    using namespace std::chrono_literals;
    
    drb::market::TradeBuffer time_buffer;
    
    // Add trades with different timestamps
    auto trade1 = drb::market::Trade(test_symbol, 100.00, 100);
    std::this_thread::sleep_for(100ms);
    
    auto trade2 = drb::market::Trade(test_symbol, 101.00, 200);
    std::this_thread::sleep_for(1s);  // Force new bar
    
    auto trade3 = drb::market::Trade(test_symbol, 102.00, 300);
    
    time_buffer.push(trade1);
    time_buffer.push(trade2);
    time_buffer.push(trade3);
    
    auto ohlc = drb::market::calculateOHLC(time_buffer, test_symbol, 500ms);
    EXPECT_EQ(ohlc.size(), 2);  // Should have two bars due to time gap
}

TEST_F(MarketDataTest, EmptyBuffers) {
    drb::market::TradeBuffer empty_trade_buffer;
    drb::market::QuoteBuffer empty_quote_buffer;
    
    auto ohlc = drb::market::calculateOHLC(empty_trade_buffer, test_symbol, std::chrono::minutes(1));
    EXPECT_TRUE(ohlc.empty());
    
    auto filtered = drb::market::getFilteredTrades(empty_trade_buffer, test_symbol, excluded_conditions);
    EXPECT_TRUE(filtered.empty());
    
    double vwap = drb::market::calculateVWAP(empty_trade_buffer, test_symbol);
    EXPECT_DOUBLE_EQ(vwap, 0.0);
    
    auto [bid, ask] = drb::market::getBestBidAsk(empty_quote_buffer, test_symbol, 
                                                std::chrono::system_clock::now());
    EXPECT_DOUBLE_EQ(bid, 0.0);
    EXPECT_DOUBLE_EQ(ask, std::numeric_limits<double>::max());
} 