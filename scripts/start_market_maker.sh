#!/bin/bash

# Check if symbol argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <symbol>"
    echo "Example: $0 AAPL"
    exit 1
fi

SYMBOL=$1

# Kill any existing market maker for this symbol
pkill -f "python.*market_maker.*$SYMBOL"

# Wait for a moment to ensure clean shutdown
sleep 1

# Check if exchange is running
if ! lsof -i :9555 > /dev/null; then
    echo "Exchange engine is not running. Please start exchange first."
    exit 1
fi

# Start market maker for the symbol
python3 exchange/market_maker/market_maker.py --symbol $SYMBOL &

# Wait for market maker to initialize
sleep 2

echo "Market maker started for symbol $SYMBOL" 