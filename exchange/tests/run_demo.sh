#!/bin/bash

# Change to project root directory
cd "$(dirname "$0")/../.."

# Clean up any existing processes
./exchange/tests/cleanup.sh
if [ $? -ne 0 ]; then
    echo "Failed to clean up processes. Please check if any ports are in use."
    exit 1
fi

# Build the project if needed
if [ ! -d "build" ] || [ ! -f "build/exchange/exchange_engine" ]; then
    echo "Building project..."
    rm -rf build
    cmake -B build -DBUILD_TESTS=ON
    cmake --build build
fi

# Setup Python environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment and install dependencies
echo "Activating virtual environment and installing dependencies..."
source venv/bin/activate
# pip install -q dash plotly pandas numpy nats-py

# Export Python path to use virtual environment
# export PYTHONPATH="$PWD/venv/lib/python3.13/site-packages:$PYTHONPATH"

# Verify ports are free
for port in 4222 9555 9556 9557 9558 9559; do
    if lsof -i :${port} >/dev/null 2>&1; then
        echo "Error: Port ${port} is still in use. Please run cleanup.sh again."
        exit 1
    fi
done

# Start NATS server in the background
echo "Starting NATS server..."
nats-server -DV &
NATS_PID=$!

# Wait for NATS server to start and verify it's running
echo "Waiting for NATS server to start..."
max_retries=10
retry_count=0
while ! lsof -i :4222 >/dev/null 2>&1 && [ $retry_count -lt $max_retries ]; do
    sleep 1
    retry_count=$((retry_count + 1))
done

if ! lsof -i :4222 >/dev/null 2>&1; then
    echo "Error: NATS server failed to start"
    exit 1
fi

echo "NATS server is running on port 4222"

# Start the exchange engine in the background
echo "Starting exchange engine..."
./build/exchange/exchange_engine exchange/config/config.json &
EXCHANGE_PID=$!

# Wait for exchange to start and verify it's running
echo "Waiting for exchange engine to start..."
retry_count=0
while ! lsof -i :9555 >/dev/null 2>&1 && [ $retry_count -lt $max_retries ]; do
    sleep 1
    retry_count=$((retry_count + 1))
done

if ! lsof -i :9555 >/dev/null 2>&1; then
    echo "Error: Exchange engine failed to start"
    exit 1
fi

echo "Exchange engine is running"

# Run the demo with default symbols from config
echo "Starting demo components..."
"$PWD/venv/bin/python3" exchange/tests/run_demo.py AAPL MSFT GOOGL

# Cleanup on exit
trap "kill $EXCHANGE_PID $NATS_PID 2>/dev/null" EXIT 