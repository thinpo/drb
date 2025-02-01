#!/bin/bash

# Install NATS server
brew install nats-server

# Install NATS C client
brew install cnats

# Install Python NATS client
pip install nats-py

# Start NATS server in the background
nats-server &

# Wait for NATS server to start
sleep 1

echo "NATS server is running..." 