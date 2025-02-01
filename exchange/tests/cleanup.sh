#!/bin/bash

# Kill exchange engine
pkill -9 -f exchange_engine

# Kill Python processes
pkill -9 -f "python.*market_maker"
pkill -9 -f "python.*trading_bot"
pkill -9 -f "python.*dashboard"
pkill -9 -f "python.*run_demo"

# Kill NATS server
pkill -9 -f "nats-server"

# Kill any remaining ZMQ processes
for port in 4222 9555 9556 9557 9558 9559; do
    pid=$(lsof -ti :${port})
    if [ ! -z "$pid" ]; then
        echo "Killing process using port ${port} (PID: ${pid})"
        kill -9 $pid 2>/dev/null
    fi
done

# Wait for processes to be killed
sleep 1

# Verify ports are free
for port in 4222 9555 9556 9557 9558 9559; do
    if lsof -i :${port} >/dev/null 2>&1; then
        echo "Error: Port ${port} is still in use"
        exit 1
    fi
done

echo "All exchange processes killed and ports freed" 