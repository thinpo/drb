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
#include "exchange/nats_client.hpp"
#include <iostream>

namespace exchange {

class BarPublisher {
public:
    using OHLC = drb::market::OHLC;
    using SaleCondition = drb::market::SaleCondition;

    explicit BarPublisher(const ExchangeConfig& config, std::shared_ptr<MarketDataHandler> market_data_handler)
        : config_(config), market_data_handler_(market_data_handler), running_(false) {
        nats_client_ = std::make_unique<NatsClient>(config.nats_url);
        
        // Initialize excluded conditions for each symbol
        for (const auto& bar : config_.bars) {
            excluded_conditions_[bar.symbol] = std::unordered_set<std::string>(
                bar.excluded_conditions.begin(),
                bar.excluded_conditions.end()
            );
        }
    }

    void start() {
        if (running_) return;
        try {
            std::cout << "Starting bar publisher..." << std::endl;
            nats_client_->connect();
            running_ = true;
            
            // Start publisher thread
            publisher_thread_ = std::thread([this]() {
                while (running_) {
                    try {
                        // TODO: Implement bar publishing logic
                        std::this_thread::sleep_for(std::chrono::seconds(1));
                    } catch (const std::exception& e) {
                        if (running_) {
                            std::cerr << "Error in bar publisher thread: " << e.what() << std::endl;
                        }
                    }
                }
            });
            
            std::cout << "Bar publisher started successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Failed to start bar publisher: " << e.what() << std::endl;
            stop();
            throw;
        }
    }

    void stop() {
        if (!running_) return;
        running_ = false;
        if (publisher_thread_.joinable()) {
            publisher_thread_.join();
        }
        if (nats_client_) {
            nats_client_->disconnect();
        }
    }

private:
    const ExchangeConfig& config_;
    std::shared_ptr<MarketDataHandler> market_data_handler_;
    std::unique_ptr<NatsClient> nats_client_;
    std::atomic<bool> running_;
    std::thread publisher_thread_;
    std::unordered_map<std::string, std::unordered_set<std::string>> excluded_conditions_;
};

} // namespace exchange 