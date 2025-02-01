#!/bin/bash

# Kill any existing dashboard
pkill -f "python.*dashboard.py"

# Wait for a moment to ensure clean shutdown
sleep 1

# Check if exchange is running
if ! lsof -i :9555 > /dev/null; then
    echo "Exchange engine is not running. Please start exchange first."
    exit 1
fi

# Start dashboard
python3 exchange/dashboard/dashboard.py &

# Wait for dashboard to initialize
sleep 2

echo "Dashboard started" 