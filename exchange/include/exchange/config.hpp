#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <fstream>
#include <unordered_map>
#include <nlohmann/json.hpp>

namespace exchange {

struct BarConfig {
    std::string symbol;
    std::chrono::seconds interval;
    std::vector<std::string> source_exchanges;
    bool include_trades{true};
    bool include_quotes{false};
    std::vector<std::string> excluded_conditions;
};

struct ExchangeConfig {
    std::string name;
    std::string nats_url{"nats://localhost:4222"};  // Default NATS URL
    uint16_t market_data_port;
    uint16_t bar_publish_port;
    std::vector<BarConfig> bars;
    std::unordered_map<std::string, std::string> exchange_endpoints;

    static ExchangeConfig from_json(const std::string& config_path) {
        using json = nlohmann::json;
        ExchangeConfig config;
        
        std::ifstream file(config_path);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open config file: " + config_path);
        }

        json j;
        file >> j;

        config.name = j["name"].get<std::string>();
        if (j.contains("nats_url")) {
            config.nats_url = j["nats_url"].get<std::string>();
        }
        config.market_data_port = j["market_data_port"].get<uint16_t>();
        config.bar_publish_port = j["bar_publish_port"].get<uint16_t>();

        // Parse exchange endpoints
        for (const auto& [exchange, endpoint] : j["exchange_endpoints"].items()) {
            config.exchange_endpoints[exchange] = endpoint.get<std::string>();
        }

        // Parse bar configurations
        for (const auto& bar : j["bars"]) {
            BarConfig bar_config;
            bar_config.symbol = bar["symbol"].get<std::string>();
            bar_config.interval = std::chrono::seconds(bar["interval_seconds"].get<int>());
            bar_config.source_exchanges = bar["source_exchanges"].get<std::vector<std::string>>();
            
            if (bar.contains("include_trades")) {
                bar_config.include_trades = bar["include_trades"].get<bool>();
            }
            if (bar.contains("include_quotes")) {
                bar_config.include_quotes = bar["include_quotes"].get<bool>();
            }
            if (bar.contains("excluded_conditions")) {
                bar_config.excluded_conditions = bar["excluded_conditions"].get<std::vector<std::string>>();
            }

            config.bars.push_back(bar_config);
        }

        return config;
    }
};

} // namespace exchange 