#!/bin/bash

# Kill any existing exchange engine
pkill -f "exchange_engine"

# Wait for a moment to ensure clean shutdown
sleep 1

# Check if NATS is running
if ! lsof -i :4222 > /dev/null; then
    echo "NATS server is not running. Please start NATS first."
    exit 1
fi

# Start exchange engine
./build/exchange/exchange_engine --config exchange/config/config.json &

# Wait for exchange to be ready
sleep 2

# Check if exchange is running
if lsof -i :9555 > /dev/null; then
    echo "Exchange engine started successfully on port 9555"
else
    echo "Failed to start exchange engine"
    exit 1
fi 