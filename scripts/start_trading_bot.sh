#!/bin/bash

# Check if symbol argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <symbol>"
    echo "Example: $0 AAPL"
    exit 1
fi

SYMBOL=$1

# Kill any existing trading bot for this symbol
pkill -f "python.*trading_bot.*$SYMBOL"

# Wait for a moment to ensure clean shutdown
sleep 1

# Check if exchange is running
if ! lsof -i :9555 > /dev/null; then
    echo "Exchange engine is not running. Please start exchange first."
    exit 1
fi

# Start trading bot for the symbol
python3 exchange/trading_bot/trading_bot.py --symbol $SYMBOL &

# Wait for trading bot to initialize
sleep 2

echo "Trading bot started for symbol $SYMBOL" 