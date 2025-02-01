#pragma once

#include <string>
#include <queue>
#include <memory>
#include <unordered_map>
#include <functional>
#include "market_data.hpp"

namespace exchange {

enum class Side { BUY, SELL };
enum class OrderType { LIMIT, MARKET };

struct Order {
    std::string order_id;
    std::string symbol;
    Side side;
    OrderType type;
    double price;
    int size;
    int filled_size{0};
    std::string client_id;
    std::chrono::system_clock::time_point timestamp;

    bool is_filled() const { return size == filled_size; }
    int remaining_size() const { return size - filled_size; }
};

struct OrderComparator {
    bool operator()(const std::shared_ptr<Order>& a, const std::shared_ptr<Order>& b) const {
        if (a->price == b->price) {
            return a->timestamp > b->timestamp;  // Earlier orders have priority
        }
        return a->price < b->price;  // Higher prices have priority for buy orders
    }
};

struct ReverseOrderComparator {
    bool operator()(const std::shared_ptr<Order>& a, const std::shared_ptr<Order>& b) const {
        if (a->price == b->price) {
            return a->timestamp > b->timestamp;  // Earlier orders have priority
        }
        return a->price > b->price;  // Lower prices have priority for sell orders
    }
};

class OrderBook {
public:
    using TradeCallback = std::function<void(const std::string& symbol,
                                           double price,
                                           int size,
                                           const std::string& buyer_id,
                                           const std::string& seller_id)>;

    OrderBook(const std::string& symbol, TradeCallback trade_cb)
        : symbol_(symbol), trade_callback_(trade_cb) {}

    std::shared_ptr<Order> add_order(std::shared_ptr<Order> order) {
        if (!order) {
            throw std::runtime_error("Null order");
        }
        if (order->symbol != symbol_) {
            throw std::runtime_error("Order symbol does not match order book");
        }
        if (order->size <= 0) {
            throw std::runtime_error("Order size must be positive");
        }
        if (order->type == OrderType::LIMIT && order->price <= 0) {
            throw std::runtime_error("Limit order price must be positive");
        }

        orders_[order->order_id] = order;

        try {
            if (order->type == OrderType::MARKET) {
                match_market_order(order);
            } else {
                if (order->side == Side::BUY) {
                    buy_orders_.push(order);
                } else {
                    sell_orders_.push(order);
                }
                match_limit_orders();
            }
        } catch (const std::exception& e) {
            // If anything goes wrong during matching, remove the order and rethrow
            orders_.erase(order->order_id);
            throw;
        }

        return order;
    }

    void cancel_order(const std::string& order_id) {
        auto it = orders_.find(order_id);
        if (it == orders_.end()) {
            return;
        }

        auto order = it->second;
        orders_.erase(it);

        // Remove from appropriate queue
        if (order->type == OrderType::LIMIT) {
            if (order->side == Side::BUY) {
                remove_from_queue(buy_orders_, order);
            } else {
                remove_from_queue(sell_orders_, order);
            }
        }
    }

    std::pair<double, int> get_best_bid() const {
        if (buy_orders_.empty()) {
            return {0.0, 0};
        }
        auto best_bid = buy_orders_.top();
        return {best_bid->price, best_bid->remaining_size()};
    }

    std::pair<double, int> get_best_ask() const {
        if (sell_orders_.empty()) {
            return {0.0, 0};
        }
        auto best_ask = sell_orders_.top();
        return {best_ask->price, best_ask->remaining_size()};
    }

private:
    void match_market_order(std::shared_ptr<Order> market_order) {
        if (market_order->side == Side::BUY) {
            match_market_buy_order(market_order);
        } else {
            match_market_sell_order(market_order);
        }
    }

    void match_market_buy_order(std::shared_ptr<Order> market_order) {
        while (!market_order->is_filled() && !sell_orders_.empty()) {
            auto contra_order = sell_orders_.top();
            
            int match_size = std::min(market_order->remaining_size(), contra_order->remaining_size());
            double match_price = contra_order->price;

            // Execute trade
            market_order->filled_size += match_size;
            contra_order->filled_size += match_size;

            // Notify trade
            trade_callback_(symbol_, match_price, match_size,
                          market_order->client_id, contra_order->client_id);

            if (contra_order->is_filled()) {
                sell_orders_.pop();
                orders_.erase(contra_order->order_id);
            }
        }
    }

    void match_market_sell_order(std::shared_ptr<Order> market_order) {
        while (!market_order->is_filled() && !buy_orders_.empty()) {
            auto contra_order = buy_orders_.top();
            
            int match_size = std::min(market_order->remaining_size(), contra_order->remaining_size());
            double match_price = contra_order->price;

            // Execute trade
            market_order->filled_size += match_size;
            contra_order->filled_size += match_size;

            // Notify trade
            trade_callback_(symbol_, match_price, match_size,
                          contra_order->client_id, market_order->client_id);

            if (contra_order->is_filled()) {
                buy_orders_.pop();
                orders_.erase(contra_order->order_id);
            }
        }
    }

    void match_limit_orders() {
        while (!buy_orders_.empty() && !sell_orders_.empty()) {
            auto buy_order = buy_orders_.top();
            auto sell_order = sell_orders_.top();

            if (buy_order->price < sell_order->price) {
                break;  // No match possible
            }

            int match_size = std::min(buy_order->remaining_size(), sell_order->remaining_size());
            double match_price = sell_order->price;  // Price-time priority: first order sets price

            // Execute trade
            buy_order->filled_size += match_size;
            sell_order->filled_size += match_size;

            // Notify trade
            trade_callback_(symbol_, match_price, match_size,
                          buy_order->client_id, sell_order->client_id);

            // Remove filled orders
            if (buy_order->is_filled()) {
                buy_orders_.pop();
                orders_.erase(buy_order->order_id);
            }
            if (sell_order->is_filled()) {
                sell_orders_.pop();
                orders_.erase(sell_order->order_id);
            }
        }
    }

    template<typename Queue>
    void remove_from_queue(Queue& queue, std::shared_ptr<Order> order) {
        Queue temp;
        while (!queue.empty()) {
            auto current = queue.top();
            queue.pop();
            if (current->order_id != order->order_id) {
                temp.push(current);
            }
        }
        queue = std::move(temp);
    }

    std::string symbol_;
    std::unordered_map<std::string, std::shared_ptr<Order>> orders_;
    std::priority_queue<std::shared_ptr<Order>, std::vector<std::shared_ptr<Order>>, OrderComparator> buy_orders_;
    std::priority_queue<std::shared_ptr<Order>, std::vector<std::shared_ptr<Order>>, ReverseOrderComparator> sell_orders_;
    TradeCallback trade_callback_;
};

} // namespace exchange 