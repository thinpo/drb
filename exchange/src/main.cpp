#include "exchange/engine.hpp"
#include <iostream>
#include <csignal>
#include <atomic>

std::atomic<bool> running(true);

void signal_handler(int) {
    running = false;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <config_file>" << std::endl;
        return 1;
    }

    try {
        // Set up signal handling
        signal(SIGINT, signal_handler);
        signal(SIGTERM, signal_handler);

        // Create and start the exchange engine
        exchange::Engine engine(argv[1]);
        std::cout << "Starting exchange engine..." << std::endl;
        engine.start();

        // Print configuration summary
        const auto& config = engine.get_config();
        std::cout << "\nExchange Configuration:" << std::endl;
        std::cout << "Name: " << config.name << std::endl;
        std::cout << "Market Data Port: " << config.market_data_port << std::endl;
        std::cout << "Bar Publish Port: " << config.bar_publish_port << std::endl;
        
        std::cout << "\nConnected Exchanges:" << std::endl;
        for (const auto& [exchange, endpoint] : config.exchange_endpoints) {
            std::cout << "- " << exchange << ": " << endpoint << std::endl;
        }

        std::cout << "\nConfigured Bars:" << std::endl;
        for (const auto& bar : config.bars) {
            std::cout << "- " << bar.symbol << " (" << bar.interval.count() << "s):" << std::endl;
            std::cout << "  Sources: ";
            for (const auto& src : bar.source_exchanges) {
                std::cout << src << " ";
            }
            std::cout << "\n  Include Trades: " << (bar.include_trades ? "Yes" : "No");
            std::cout << "\n  Include Quotes: " << (bar.include_quotes ? "Yes" : "No");
            if (!bar.excluded_conditions.empty()) {
                std::cout << "\n  Excluded Conditions: ";
                for (const auto& cond : bar.excluded_conditions) {
                    std::cout << cond << " ";
                }
            }
            std::cout << std::endl;
        }

        std::cout << "\nExchange engine running. Press Ctrl+C to stop." << std::endl;

        // Wait for shutdown signal
        while (running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // Clean shutdown
        std::cout << "\nShutting down exchange engine..." << std::endl;
        engine.stop();
        std::cout << "Exchange engine stopped." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
} 