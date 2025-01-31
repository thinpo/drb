#pragma once

#include <string>
#include <memory>
#include <thread>
#include <atomic>
#include <unordered_map>
#include "dynamic_ring_buffer.hpp"
#include "market_data.hpp"
#include "exchange/config.hpp"
#include <zmq.hpp>

namespace exchange {

class MarketDataHandler {
public:
    using TradeBuffer = drb::market::TradeBuffer;
    using QuoteBuffer = drb::market::QuoteBuffer;
    using Trade = drb::market::Trade;
    using Quote = drb::market::Quote;

    MarketDataHandler(const ExchangeConfig& config)
        : config_(config), running_(false), context_(1) {
        // Initialize buffers for each symbol
        for (const auto& bar_config : config.bars) {
            trade_buffers_[bar_config.symbol] = std::make_shared<TradeBuffer>();
            quote_buffers_[bar_config.symbol] = std::make_shared<QuoteBuffer>();
        }
    }

    void start() {
        if (running_) return;
        running_ = true;

        // Start subscriber threads for each exchange
        for (const auto& [exchange_name, endpoint] : config_.exchange_endpoints) {
            subscriber_threads_.emplace_back([this, exchange_name, endpoint]() {
                this->subscribe_to_exchange(exchange_name, endpoint);
            });
        }
    }

    void stop() {
        running_ = false;
        for (auto& thread : subscriber_threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

    std::shared_ptr<TradeBuffer> get_trade_buffer(const std::string& symbol) {
        auto it = trade_buffers_.find(symbol);
        return (it != trade_buffers_.end()) ? it->second : nullptr;
    }

    std::shared_ptr<QuoteBuffer> get_quote_buffer(const std::string& symbol) {
        auto it = quote_buffers_.find(symbol);
        return (it != quote_buffers_.end()) ? it->second : nullptr;
    }

private:
    void subscribe_to_exchange(const std::string& exchange_name, const std::string& endpoint) {
        zmq::socket_t subscriber(context_, ZMQ_SUB);
        subscriber.connect(endpoint);

        // Subscribe to all symbols for this exchange
        for (const auto& bar_config : config_.bars) {
            if (std::find(bar_config.source_exchanges.begin(),
                         bar_config.source_exchanges.end(),
                         exchange_name) != bar_config.source_exchanges.end()) {
                // Subscribe to both trade and quote topics for the symbol
                subscriber.set(zmq::sockopt::subscribe, 
                             "TRADE." + bar_config.symbol + "." + exchange_name);
                subscriber.set(zmq::sockopt::subscribe, 
                             "QUOTE." + bar_config.symbol + "." + exchange_name);
            }
        }

        while (running_) {
            zmq::message_t topic;
            zmq::message_t msg;

            try {
                // Receive with timeout to check running_ flag periodically
                if (!subscriber.recv(topic, zmq::recv_flags::dontwait)) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }
                subscriber.recv(msg);

                std::string topic_str(static_cast<char*>(topic.data()), topic.size());
                auto parts = split_topic(topic_str);
                if (parts.size() != 3) continue;

                const auto& msg_type = parts[0];
                const auto& symbol = parts[1];
                const auto& exchange = parts[2];

                if (msg_type == "TRADE") {
                    auto trade = parse_trade(msg.to_string(), symbol, exchange);
                    if (auto buffer = get_trade_buffer(symbol)) {
                        buffer->push(trade);
                    }
                } else if (msg_type == "QUOTE") {
                    auto quote = parse_quote(msg.to_string(), symbol, exchange);
                    if (auto buffer = get_quote_buffer(symbol)) {
                        buffer->push(quote);
                    }
                }
            } catch (const zmq::error_t& e) {
                if (e.num() != EAGAIN) {
                    // Log error
                    std::cerr << "ZMQ error in subscriber thread: " << e.what() << std::endl;
                }
            } catch (const std::exception& e) {
                // Log error
                std::cerr << "Error in subscriber thread: " << e.what() << std::endl;
            }
        }
    }

    std::vector<std::string> split_topic(const std::string& topic) {
        std::vector<std::string> parts;
        std::string::size_type start = 0;
        std::string::size_type end = 0;

        while ((end = topic.find('.', start)) != std::string::npos) {
            parts.push_back(topic.substr(start, end - start));
            start = end + 1;
        }
        parts.push_back(topic.substr(start));
        return parts;
    }

    Trade parse_trade(const std::string& msg, const std::string& symbol, const std::string& exchange) {
        using json = nlohmann::json;
        auto j = json::parse(msg);
        
        return Trade(
            symbol,
            j["price"].get<double>(),
            j["size"].get<int>(),
            j["condition"].get<std::string>(),
            exchange
        );
    }

    Quote parse_quote(const std::string& msg, const std::string& symbol, const std::string& exchange) {
        using json = nlohmann::json;
        auto j = json::parse(msg);
        
        return Quote(
            symbol,
            j["bid_price"].get<double>(),
            j["ask_price"].get<double>(),
            j["bid_size"].get<int>(),
            j["ask_size"].get<int>(),
            exchange
        );
    }

    ExchangeConfig config_;
    std::atomic<bool> running_;
    zmq::context_t context_;
    std::vector<std::thread> subscriber_threads_;
    std::unordered_map<std::string, std::shared_ptr<TradeBuffer>> trade_buffers_;
    std::unordered_map<std::string, std::shared_ptr<QuoteBuffer>> quote_buffers_;
};

} // namespace exchange 