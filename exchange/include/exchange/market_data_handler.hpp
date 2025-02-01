#pragma once

#include "exchange/config.hpp"
#include "exchange/nats_client.hpp"
#include <memory>
#include <atomic>
#include <thread>
#include <iostream>

namespace exchange {

class MarketDataHandler {
public:
    explicit MarketDataHandler(const ExchangeConfig& config) 
        : config_(config), running_(false) {
        nats_client_ = std::make_unique<NatsClient>(config.nats_url);
    }

    void start() {
        if (running_) return;
        try {
            std::cout << "Starting market data handler..." << std::endl;
            nats_client_->connect();
            running_ = true;
            
            // Subscribe to market data for each exchange
            for (const auto& [exchange, _] : config_.exchange_endpoints) {
                try {
                    std::string subject = "MARKET_DATA." + exchange;
                    std::cout << "Subscribing to " << subject << "..." << std::endl;
                    
                    nats_client_->subscribe(
                        subject,
                        [this, exchange](const std::string& subject, const std::string& data, const std::string& reply) {
                            try {
                                handle_market_data(exchange, data);
                            } catch (const std::exception& e) {
                                std::cerr << "Error handling market data for " << exchange << ": " << e.what() << std::endl;
                            }
                        }
                    );
                    std::cout << "Successfully subscribed to market data for " << exchange << std::endl;
                } catch (const std::exception& e) {
                    std::cerr << "Failed to subscribe to market data for " << exchange << ": " << e.what() << std::endl;
                    throw;
                }
            }
            
            std::cout << "Market data handler started successfully" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Failed to start market data handler: " << e.what() << std::endl;
            stop();
            throw;
        }
    }

    void stop() {
        if (!running_) return;
        running_ = false;
        if (nats_client_) {
            nats_client_->disconnect();
        }
    }

private:
    void handle_market_data(const std::string& exchange, const std::string& msg) {
        if (!running_) return;
        try {
            // Process market data message
            // TODO: Implement market data processing logic
        } catch (const std::exception& e) {
            if (running_) {
                std::cerr << "Error in market data handler: " << e.what() << std::endl;
            }
        }
    }

    const ExchangeConfig& config_;
    std::unique_ptr<NatsClient> nats_client_;
    std::atomic<bool> running_;
};

} // namespace exchange 