#!/bin/bash

# Kill any existing NATS server
pkill -f "nats-server"

# Wait for a moment to ensure clean shutdown
sleep 1

# Start NATS server
nats-server -p 4222 &

# Wait for NATS to be ready
sleep 2

# Check if NATS is running
if lsof -i :4222 > /dev/null; then
    echo "NATS server started successfully on port 4222"
else
    echo "Failed to start NATS server"
    exit 1
fi 