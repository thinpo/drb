#pragma once

#include <string>
#include <memory>
#include <unordered_map>
#include <atomic>
#include <thread>
#include <nlohmann/json.hpp>
#include "exchange/order_book.hpp"
#include "market_data.hpp"
#include "exchange/config.hpp"
#include "exchange/nats_client.hpp"

namespace exchange {

class MatchEngine {
public:
    explicit MatchEngine(const ExchangeConfig& config) 
        : config_(config), 
          running_(false),
          nats_(config.nats_url) {

        // Initialize order books for each symbol
        for (const auto& bar : config.bars) {
            order_books_[bar.symbol] = std::make_unique<OrderBook>(
                bar.symbol,
                [this](const std::string& symbol, double price, int size,
                      const std::string& buyer_id, const std::string& seller_id) {
                    publish_trade(symbol, price, size, buyer_id, seller_id);
                }
            );
        }
    }

    void start() {
        if (running_) return;
        
        try {
            std::cout << "Starting match engine..." << std::endl;
            
            // Connect to NATS
            std::cout << "Connecting to NATS server..." << std::endl;
            try {
                nats_.connect();
            } catch (const std::exception& e) {
                std::cerr << "Failed to connect to NATS server: " << e.what() << std::endl;
                throw std::runtime_error("NATS client not running");
            }
            
            // Subscribe to order requests
            std::cout << "Setting up order subscriptions..." << std::endl;
            for (const auto& [symbol, _] : order_books_) {
                try {
                    std::string subject = "ORDER." + symbol;
                    std::cout << "Subscribing to " << subject << "..." << std::endl;
                    
                    nats_.subscribe(
                        subject,
                        [this, symbol](const std::string& subject, const std::string& data, const std::string& reply) {
                            try {
                                std::cout << "Handling order request for " << symbol << " with reply subject: " << reply << std::endl;
                                handle_order(subject, data, reply);
                            } catch (const std::exception& e) {
                                std::cerr << "Error handling order for " << symbol << ": " << e.what() << std::endl;
                                if (!reply.empty()) {
                                    try {
                                        nats_.publish(reply, std::string("Error: ") + e.what());
                                    } catch (...) {
                                        std::cerr << "Failed to send error response" << std::endl;
                                    }
                                }
                            }
                        }
                    );
                    std::cout << "Successfully subscribed to orders for " << symbol << std::endl;
                } catch (const std::exception& e) {
                    std::cerr << "Failed to subscribe to orders for " << symbol << ": " << e.what() << std::endl;
                    nats_.disconnect();
                    throw;
                }
            }
            
            running_ = true;
            
            // Start market data thread
            std::cout << "Starting market data thread..." << std::endl;
            market_data_thread_ = std::thread([this]() { publish_market_data(); });
            
            std::cout << "Match engine started successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Failed to start match engine: " << e.what() << std::endl;
            stop();
            throw;
        }
    }

    void stop() {
        if (!running_) return;
        running_ = false;
        
        // Wait for threads to finish
        if (market_data_thread_.joinable()) {
            market_data_thread_.join();
        }

        // Disconnect from NATS
        try {
            nats_.disconnect();
            std::cout << "Match engine stopped successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error during match engine shutdown: " << e.what() << std::endl;
        }
    }

private:
    void handle_order(const std::string& subject, const std::string& data, const std::string& reply) {
        try {
            std::cout << "Received order on subject " << subject << ": " << data << std::endl;
            
            // Always send a response for request-reply pattern
            auto send_response = [this, &reply](const std::string& msg) {
                if (!reply.empty()) {
                    try {
                        std::cout << "Sending response to " << reply << ": " << msg << std::endl;
                        nats_.publish(reply, msg);
                    } catch (const std::exception& e) {
                        std::cerr << "Failed to send response: " << e.what() << std::endl;
                    }
                }
            };
            
            // Parse order
            auto order = parse_order(data);
            if (!order) {
                std::string error = "Failed to parse order";
                std::cerr << error << ": " << data << std::endl;
                send_response(error);
                return;
            }

            order->timestamp = std::chrono::system_clock::now();
            std::cout << "Parsed order: ID=" << order->order_id << ", Symbol=" << order->symbol 
                     << ", Side=" << (order->side == Side::BUY ? "BUY" : "SELL") 
                     << ", Size=" << order->size << std::endl;

            // Process order
            auto it = order_books_.find(order->symbol);
            if (it == order_books_.end()) {
                std::string error = "Invalid symbol: " + order->symbol;
                std::cerr << error << std::endl;
                send_response(error);
                return;
            }

            auto result = it->second->add_order(order);
            if (!result) {
                std::string error = "Failed to add order: " + order->order_id;
                std::cerr << error << std::endl;
                send_response(error);
                return;
            }

            std::string success = "Order accepted: " + result->order_id;
            std::cout << success << std::endl;
            send_response(success);

            // Publish updated order book
            publish_order_book(order->symbol);

        } catch (const std::exception& e) {
            std::string error = std::string("Error handling order: ") + e.what();
            std::cerr << error << std::endl;
            if (!reply.empty()) {
                try {
                    std::cout << "Sending error response to " << reply << ": " << error << std::endl;
                    nats_.publish(reply, error);
                } catch (...) {
                    std::cerr << "Failed to send error response" << std::endl;
                }
            }
        }
    }

    void publish_market_data() {
        while (running_) {
            try {
                // Publish order books periodically
                for (const auto& [symbol, book] : order_books_) {
                    publish_order_book(symbol);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            } catch (const std::exception& e) {
                if (running_) {
                    std::cerr << "Error publishing market data: " << e.what() << std::endl;
                }
            }
        }
    }

    void publish_trade(const std::string& symbol, double price, int size,
                      const std::string& buyer_id, const std::string& seller_id) {
        if (!running_) return;

        using json = nlohmann::json;
        
        json trade = {
            {"symbol", symbol},
            {"price", price},
            {"size", size},
            {"buyer_id", buyer_id},
            {"seller_id", seller_id},
            {"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count()}
        };

        try {
            nats_.publish_json("TRADE." + symbol, trade);
        } catch (const std::exception& e) {
            if (running_) {
                std::cerr << "Error publishing trade: " << e.what() << std::endl;
            }
        }
    }

    void publish_order_book(const std::string& symbol) {
        if (!running_) return;

        auto it = order_books_.find(symbol);
        if (it == order_books_.end()) return;

        auto& book = it->second;
        auto [best_bid, bid_size] = book->get_best_bid();
        auto [best_ask, ask_size] = book->get_best_ask();

        using json = nlohmann::json;
        json book_data = {
            {"bids", {{best_bid, bid_size}}},
            {"asks", {{best_ask, ask_size}}},
            {"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count()}
        };

        try {
            nats_.publish_json("BOOK." + symbol, book_data);
        } catch (const std::exception& e) {
            if (running_) {
                std::cerr << "Error publishing order book: " << e.what() << std::endl;
            }
        }
    }

    std::shared_ptr<Order> parse_order(const std::string& msg) {
        try {
            using json = nlohmann::json;
            auto j = json::parse(msg);

            // Validate required fields
            if (!j.contains("order_id") || !j.contains("symbol") || 
                !j.contains("side") || !j.contains("type") || 
                !j.contains("size")) {
                throw std::runtime_error("Missing required fields in order");
            }

            auto order = std::make_shared<Order>();
            order->order_id = j["order_id"].get<std::string>();
            order->symbol = j["symbol"].get<std::string>();
            
            std::string side = j["side"].get<std::string>();
            if (side != "BUY" && side != "SELL") {
                throw std::runtime_error("Invalid side: " + side);
            }
            order->side = side == "BUY" ? Side::BUY : Side::SELL;
            
            std::string type = j["type"].get<std::string>();
            if (type != "MARKET" && type != "LIMIT") {
                throw std::runtime_error("Invalid order type: " + type);
            }
            order->type = type == "MARKET" ? OrderType::MARKET : OrderType::LIMIT;
            
            if (order->type == OrderType::LIMIT && !j.contains("price")) {
                throw std::runtime_error("Price required for limit orders");
            }
            order->price = j.value("price", 0.0);
            
            int size = j["size"].get<int>();
            if (size <= 0) {
                throw std::runtime_error("Size must be positive");
            }
            order->size = size;
            
            order->client_id = j.value("client_id", "");
            return order;
        } catch (const nlohmann::json::exception& e) {
            throw std::runtime_error("Invalid JSON format: " + std::string(e.what()));
        }
    }

    const ExchangeConfig& config_;
    std::atomic<bool> running_;
    NatsClient nats_;
    std::thread market_data_thread_;
    std::unordered_map<std::string, std::unique_ptr<OrderBook>> order_books_;
};

} // namespace exchange 