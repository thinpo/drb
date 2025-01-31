#pragma once

#include <memory>
#include <string>
#include "exchange/config.hpp"
#include "exchange/market_data_handler.hpp"
#include "exchange/bar_publisher.hpp"

namespace exchange {

class Engine {
public:
    explicit Engine(const std::string& config_path) {
        config_ = ExchangeConfig::from_json(config_path);
        market_data_handler_ = std::make_shared<MarketDataHandler>(config_);
        bar_publisher_ = std::make_unique<BarPublisher>(config_, market_data_handler_);
    }

    void start() {
        market_data_handler_->start();
        bar_publisher_->start();
    }

    void stop() {
        bar_publisher_->stop();
        market_data_handler_->stop();
    }

    const ExchangeConfig& get_config() const {
        return config_;
    }

private:
    ExchangeConfig config_;
    std::shared_ptr<MarketDataHandler> market_data_handler_;
    std::unique_ptr<BarPublisher> bar_publisher_;
};

} // namespace exchange 