#pragma once

#include <string>
#include <memory>
#include <thread>
#include <atomic>
#include <unordered_map>
#include <unordered_set>
#include "dynamic_ring_buffer.hpp"
#include "market_data.hpp"
#include "exchange/config.hpp"
#include "exchange/market_data_handler.hpp"
#include <zmq.hpp>

namespace exchange {

class BarPublisher {
public:
    using OHLC = drb::market::OHLC;
    using SaleCondition = drb::market::SaleCondition;

    BarPublisher(const ExchangeConfig& config, std::shared_ptr<MarketDataHandler> handler)
        : config_(config), handler_(handler), running_(false), context_(1) {
        
        // Initialize publisher socket
        publisher_ = std::make_unique<zmq::socket_t>(context_, ZMQ_PUB);
        publisher_->bind("tcp://*:" + std::to_string(config_.bar_publish_port));

        // Initialize excluded conditions for each bar config
        for (const auto& bar_config : config_.bars) {
            std::unordered_set<SaleCondition> conditions;
            for (const auto& cond : bar_config.excluded_conditions) {
                conditions.insert(drb::market::sale_condition::from_string(cond));
            }
            excluded_conditions_[bar_config.symbol] = std::move(conditions);
        }
    }

    void start() {
        if (running_) return;
        running_ = true;

        // Start publisher threads for each bar configuration
        for (const auto& bar_config : config_.bars) {
            publisher_threads_.emplace_back([this, bar_config]() {
                this->publish_bars(bar_config);
            });
        }
    }

    void stop() {
        running_ = false;
        for (auto& thread : publisher_threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

private:
    void publish_bars(const BarConfig& config) {
        using namespace std::chrono;
        auto trade_buffer = handler_->get_trade_buffer(config.symbol);
        auto quote_buffer = handler_->get_quote_buffer(config.symbol);
        
        if (!trade_buffer && !quote_buffer) {
            std::cerr << "No buffers available for symbol: " << config.symbol << std::endl;
            return;
        }

        auto next_bar_time = system_clock::now();
        next_bar_time += duration_cast<microseconds>(config.interval);

        while (running_) {
            auto now = system_clock::now();
            if (now < next_bar_time) {
                std::this_thread::sleep_for(milliseconds(1));
                continue;
            }

            try {
                // Calculate OHLC bar
                std::vector<OHLC> bars;
                if (config.include_trades && trade_buffer) {
                    auto trade_bars = drb::market::calculateOHLC(
                        *trade_buffer,
                        config.symbol,
                        config.interval,
                        excluded_conditions_[config.symbol]
                    );
                    bars.insert(bars.end(), trade_bars.begin(), trade_bars.end());
                }

                // Publish bars
                for (const auto& bar : bars) {
                    publish_bar(bar);
                }

                next_bar_time += duration_cast<microseconds>(config.interval);
            } catch (const std::exception& e) {
                std::cerr << "Error publishing bars for " << config.symbol 
                         << ": " << e.what() << std::endl;
                next_bar_time = system_clock::now() + duration_cast<microseconds>(config.interval);
            }
        }
    }

    void publish_bar(const OHLC& bar) {
        using json = nlohmann::json;
        
        json j = {
            {"timestamp", std::chrono::duration_cast<std::chrono::milliseconds>(
                bar.timestamp.time_since_epoch()).count()},
            {"symbol", bar.symbol},
            {"open", bar.open},
            {"high", bar.high},
            {"low", bar.low},
            {"close", bar.close},
            {"volume", bar.volume},
            {"trade_count", bar.trade_count}
        };

        std::string topic = "BAR." + bar.symbol;
        std::string msg = j.dump();

        zmq::message_t topic_msg(topic.data(), topic.size());
        zmq::message_t data_msg(msg.data(), msg.size());

        try {
            publisher_->send(topic_msg, zmq::send_flags::sndmore);
            publisher_->send(data_msg, zmq::send_flags::none);
        } catch (const zmq::error_t& e) {
            std::cerr << "Error publishing bar: " << e.what() << std::endl;
        }
    }

    ExchangeConfig config_;
    std::shared_ptr<MarketDataHandler> handler_;
    std::atomic<bool> running_;
    zmq::context_t context_;
    std::unique_ptr<zmq::socket_t> publisher_;
    std::vector<std::thread> publisher_threads_;
    std::unordered_map<std::string, std::unordered_set<SaleCondition>> excluded_conditions_;
};

} // namespace exchange 