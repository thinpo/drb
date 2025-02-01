#pragma once

#include <string>
#include <memory>
#include <thread>
#include <atomic>
#include <functional>
#include <nats/nats.h>
#include <nats/status.h>
#include <nlohmann/json.hpp>
#include <vector>
#include <iostream>

namespace exchange {

class NatsClient {
public:
    NatsClient(const std::string& url = "nats://localhost:4222") : url_(url), running_(false), conn_(nullptr), opts_(nullptr) {
        // Create NATS options
        if (natsOptions_Create(&opts_) != NATS_OK) {
            throw std::runtime_error("Failed to create NATS options");
        }
        
        // Set URL
        if (natsOptions_SetURL(opts_, url_.c_str()) != NATS_OK) {
            natsOptions_Destroy(opts_);
            throw std::runtime_error("Failed to set NATS URL");
        }

        // Set async error handler
        if (natsOptions_SetErrorHandler(opts_, [](natsConnection* nc, natsSubscription* sub, natsStatus err, void* closure) {
            std::cerr << "NATS async error: " << natsStatus_GetText(err) << std::endl;
        }, nullptr) != NATS_OK) {
            natsOptions_Destroy(opts_);
            throw std::runtime_error("Failed to set error handler");
        }

        // Set closed handler
        if (natsOptions_SetClosedCB(opts_, [](natsConnection* nc, void* closure) {
            std::cerr << "NATS connection closed" << std::endl;
        }, nullptr) != NATS_OK) {
            natsOptions_Destroy(opts_);
            throw std::runtime_error("Failed to set closed handler");
        }

        // Set reconnect handler
        if (natsOptions_SetReconnectedCB(opts_, [](natsConnection* nc, void* closure) {
            std::cout << "NATS reconnected" << std::endl;
        }, nullptr) != NATS_OK) {
            natsOptions_Destroy(opts_);
            throw std::runtime_error("Failed to set reconnect handler");
        }

        // Set disconnected handler
        if (natsOptions_SetDisconnectedCB(opts_, [](natsConnection* nc, void* closure) {
            std::cerr << "NATS disconnected" << std::endl;
        }, nullptr) != NATS_OK) {
            natsOptions_Destroy(opts_);
            throw std::runtime_error("Failed to set disconnect handler");
        }
    }

    ~NatsClient() {
        disconnect();
        if (opts_) {
            natsOptions_Destroy(opts_);
        }
    }

    void connect() {
        if (running_) return;
        
        std::cout << "Connecting to NATS server at " << url_ << "..." << std::endl;
        
        // Create connection
        auto status = natsConnection_Connect(&conn_, opts_);
        if (status != NATS_OK) {
            const char* error = natsStatus_GetText(status);
            if (conn_) {
                natsConnection_Close(conn_);
                natsConnection_Destroy(conn_);
                conn_ = nullptr;
            }
            throw std::runtime_error("Failed to create NATS connection: " + std::string(error ? error : "Unknown error"));
        }
        
        // Wait for connection to be established
        int max_retries = 30;  // Increased retries
        int retry_count = 0;
        while (retry_count < max_retries) {
            auto status = natsConnection_Status(conn_);
            if (status == NATS_CONN_STATUS_CONNECTED) {
                std::cout << "Successfully connected to NATS server" << std::endl;
                running_ = true;
                return;
            }
            std::cout << "Waiting for NATS connection (attempt " << retry_count + 1 << "/" << max_retries << ")..." << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));  // Increased wait time
            retry_count++;
        }
        
        // If we get here, connection failed
        if (conn_) {
            natsConnection_Close(conn_);
            natsConnection_Destroy(conn_);
            conn_ = nullptr;
        }
        throw std::runtime_error("Failed to establish NATS connection after " + std::to_string(max_retries) + " retries");
    }

    void disconnect() {
        if (!running_) return;
        running_ = false;
        if (conn_) {
            natsConnection_Drain(conn_);
            // Wait for drain to complete
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            natsConnection_Close(conn_);
            natsConnection_Destroy(conn_);
            conn_ = nullptr;
        }
    }

    void publish(const std::string& subject, const std::string& data) {
        if (!running_) return;
        if (natsConnection_PublishString(conn_, subject.c_str(), data.c_str()) != NATS_OK) {
            throw std::runtime_error("Failed to publish message");
        }
    }

    void publish_json(const std::string& subject, const nlohmann::json& data) {
        publish(subject, data.dump());
    }

    std::string request(const std::string& subject, const std::string& data, int timeout_ms = 1000) {
        if (!running_) throw std::runtime_error("NATS client not running");
        
        natsMsg* reply = nullptr;
        if (natsConnection_RequestString(&reply, conn_, subject.c_str(), data.c_str(), timeout_ms) != NATS_OK) {
            throw std::runtime_error("Failed to send request");
        }

        std::string response(natsMsg_GetData(reply));
        natsMsg_Destroy(reply);
        return response;
    }

    nlohmann::json request_json(const std::string& subject, const nlohmann::json& data, int timeout_ms = 1000) {
        auto response = request(subject, data.dump(), timeout_ms);
        return nlohmann::json::parse(response);
    }

    void subscribe(const std::string& subject, std::function<void(const std::string&, const std::string&, const std::string&)> callback) {
        if (!running_) {
            throw std::runtime_error("NATS client not running");
        }
        if (!conn_) {
            throw std::runtime_error("NATS connection is null");
        }

        std::cout << "Subscribing to subject: " << subject << std::endl;

        // Allocate callback on the heap and store it for lifetime management
        auto heapCallback = std::make_shared<std::function<void(const std::string&, const std::string&, const std::string&)>>(std::move(callback));
        subscriptionCallbacks_.push_back(heapCallback);

        natsSubscription* sub = nullptr;
        auto msg_handler = [](natsConnection* nc, natsSubscription* sub, natsMsg* msg, void* closure) {
            auto cb = static_cast<std::function<void(const std::string&, const std::string&, const std::string&)>*>(closure);
            const char* reply = natsMsg_GetReply(msg);
            const char* data = natsMsg_GetData(msg);
            const char* subj = natsMsg_GetSubject(msg);
            
            try {
                (*cb)(subj ? subj : "", data ? data : "", reply ? reply : "");
                
                // If this is a request (has reply subject), send an empty response to avoid timeout
                if (reply && strlen(reply) > 0) {
                    natsConnection_PublishString(nc, reply, "");
                }
            } catch (const std::exception& e) {
                std::cerr << "Error in message handler: " << e.what() << std::endl;
                // If there was an error but we have a reply subject, send error response
                if (reply && strlen(reply) > 0) {
                    std::string error_msg = std::string("Error: ") + e.what();
                    natsConnection_PublishString(nc, reply, error_msg.c_str());
                }
            }
            
            natsMsg_Destroy(msg);
        };

        // For order subjects, use regular subscribe to handle request-reply pattern
        if (subject.find("ORDER.") == 0) {
            auto status = natsConnection_Subscribe(&sub, conn_, subject.c_str(), (natsMsgHandler)msg_handler, heapCallback.get());
            if (status != NATS_OK) {
                const char* error = natsStatus_GetText(status);
                throw std::runtime_error("Failed to subscribe to " + subject + ": " + std::string(error ? error : "Unknown error"));
            }
        } else {
            // For other subjects, use queue subscribe for load balancing
            auto status = natsConnection_QueueSubscribe(&sub, conn_, subject.c_str(), "exchange_workers", (natsMsgHandler)msg_handler, heapCallback.get());
            if (status != NATS_OK) {
                const char* error = natsStatus_GetText(status);
                throw std::runtime_error("Failed to subscribe to " + subject + ": " + std::string(error ? error : "Unknown error"));
            }
        }

        // Enable async subscription mode
        if (natsSubscription_SetPendingLimits(sub, -1, -1) != NATS_OK) {
            natsSubscription_Destroy(sub);
            throw std::runtime_error("Failed to set subscription pending limits");
        }

        std::cout << "Successfully subscribed to " << subject << std::endl;
    }

    void subscribe_json(const std::string& subject, std::function<void(const std::string&, const nlohmann::json&, const std::string&)> callback) {
        subscribe(subject, [callback](const std::string& subject, const std::string& data, const std::string& reply) {
            callback(subject, nlohmann::json::parse(data), reply);
        });
    }

private:
    std::string url_;
    natsConnection* conn_{nullptr};
    natsOptions* opts_{nullptr};
    std::atomic<bool> running_;
    // Container to hold subscription callbacks to ensure they remain valid
    std::vector<std::shared_ptr<std::function<void(const std::string&, const std::string&, const std::string&)>>> subscriptionCallbacks_;
};

} // namespace exchange 