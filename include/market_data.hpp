#pragma once

#include <string>
#include <chrono>
#include <vector>
#include <unordered_set>
#include <limits>
#include "dynamic_ring_buffer.hpp"
#include "sale_condition.hpp"

namespace drb {
namespace market {

// Market data structures
struct Quote {
    std::chrono::system_clock::time_point timestamp;
    std::string symbol;
    double bid_price;
    double ask_price;
    int bid_size;
    int ask_size;
    std::string exchange;

    Quote() = default;
    Quote(const std::string& sym, double bid_p, double ask_p, int bid_s, int ask_s, 
          const std::string& ex = "")
        : timestamp(std::chrono::system_clock::now()),
          symbol(sym), bid_price(bid_p), ask_price(ask_p),
          bid_size(bid_s), ask_size(ask_s), exchange(ex) {}
};

struct Trade {
    std::chrono::system_clock::time_point timestamp;
    std::string symbol;
    double price;
    int size;
    SaleCondition sale_condition;
    std::string exchange;

    Trade() = default;
    Trade(const std::string& sym, double p, int s, 
          const SaleCondition& cond = sale_condition::from_string("   "), 
          const std::string& ex = "")
        : timestamp(std::chrono::system_clock::now()),
          symbol(sym), price(p), size(s),
          sale_condition(cond), exchange(ex) {}

    // Convenience constructor that takes string for sale condition
    Trade(const std::string& sym, double p, int s, 
          const std::string& cond, const std::string& ex = "")
        : Trade(sym, p, s, sale_condition::from_string(cond), ex) {}
};

struct OHLC {
    std::chrono::system_clock::time_point timestamp;
    std::string symbol;
    double open;
    double high;
    double low;
    double close;
    int volume;
    int trade_count;

    OHLC() : open(0), high(0), low(0), close(0), volume(0), trade_count(0) {}
};

// Market data buffer types
using TradeBuffer = DynamicRingBuffer<Trade>;
using QuoteBuffer = DynamicRingBuffer<Quote>;

// Helper functions
inline bool shouldFilterTrade(const Trade& trade, const std::unordered_set<SaleCondition>& excluded_conditions) {
    return excluded_conditions.find(trade.sale_condition) != excluded_conditions.end();
}

// Market data analysis functions
template<typename Duration>
std::vector<OHLC> calculateOHLC(const TradeBuffer& buffer,
                               const std::string& symbol,
                               Duration interval,
                               const std::unordered_set<SaleCondition>& excluded_conditions = {}) {
    if (buffer.isEmpty()) return {};

    std::vector<OHLC> results;
    OHLC current;
    bool first_trade = true;
    auto interval_start = std::chrono::system_clock::time_point::min();

    for (const auto& trade : buffer) {
        // Skip trades that don't match the symbol
        if (trade.symbol != symbol) continue;

        // Skip filtered trades
        if (shouldFilterTrade(trade, excluded_conditions)) continue;

        // Initialize new interval if needed
        if (trade.timestamp >= interval_start + interval || first_trade) {
            if (!first_trade) {
                results.push_back(current);
            }
            interval_start = trade.timestamp;
            current = OHLC();
            current.timestamp = interval_start;
            current.symbol = symbol;
            current.open = trade.price;
            current.high = trade.price;
            current.low = trade.price;
            first_trade = false;
        }

        // Update OHLC
        current.high = std::max(current.high, trade.price);
        current.low = std::min(current.low, trade.price);
        current.close = trade.price;
        current.volume += trade.size;
        current.trade_count++;
    }

    // Add the last interval if it contains data
    if (!first_trade) {
        results.push_back(current);
    }

    return results;
}

std::vector<Trade> getFilteredTrades(const TradeBuffer& buffer,
                                   const std::string& symbol,
                                   const std::unordered_set<SaleCondition>& excluded_conditions) {
    std::vector<Trade> filtered_trades;
    for (const auto& trade : buffer) {
        if (trade.symbol == symbol && !shouldFilterTrade(trade, excluded_conditions)) {
            filtered_trades.push_back(trade);
        }
    }
    return filtered_trades;
}

double calculateVWAP(const TradeBuffer& buffer,
                    const std::string& symbol,
                    const std::unordered_set<SaleCondition>& excluded_conditions = {}) {
    double sum_pv = 0.0;
    int total_volume = 0;

    for (const auto& trade : buffer) {
        if (trade.symbol == symbol && !shouldFilterTrade(trade, excluded_conditions)) {
            sum_pv += trade.price * trade.size;
            total_volume += trade.size;
        }
    }

    return total_volume > 0 ? sum_pv / total_volume : 0.0;
}

std::pair<double, double> getBestBidAsk(const QuoteBuffer& buffer,
                                       const std::string& symbol,
                                       std::chrono::system_clock::time_point timestamp) {
    double best_bid = 0.0;
    double best_ask = std::numeric_limits<double>::max();

    for (const auto& quote : buffer) {
        if (quote.symbol == symbol && quote.timestamp <= timestamp) {
            best_bid = std::max(best_bid, quote.bid_price);
            best_ask = std::min(best_ask, quote.ask_price);
        }
    }

    return {best_bid, best_ask};
}

} // namespace market
} // namespace drb 