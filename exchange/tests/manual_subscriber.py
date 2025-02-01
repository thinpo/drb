import asyncio
import json
import nats
import sys
from datetime import datetime

async def main():
    # Connect to NATS
    nc = await nats.connect("nats://localhost:4222")
    print("Connected to NATS server")

    print("Manual Bar Subscriber")
    print("Available commands:")
    print("1. sub SYMBOL  (subscribe to a symbol)")
    print("2. unsub SYMBOL  (unsubscribe from a symbol)")
    print("3. quit")
    print("\nExample:")
    print("sub AAPL")
    print("unsub MSFT")

    subscribed_symbols = set()
    subscriptions = {}

    async def message_handler(msg):
        try:
            data = json.loads(msg.data.decode())
            timestamp = datetime.fromtimestamp(data["timestamp"] / 1000.0)
            print(f"\nReceived Bar at {timestamp}:")
            print(f"Symbol: {data['symbol']}")
            print(f"Open:   {data['open']:.2f}")
            print(f"High:   {data['high']:.2f}")
            print(f"Low:    {data['low']:.2f}")
            print(f"Close:  {data['close']:.2f}")
            print(f"Volume: {data['volume']}")
            print(f"Trades: {data['trade_count']}")
        except Exception as e:
            print(f"Error processing message: {e}")

    while True:
        try:
            command = input("\nEnter command: ").strip().split()
            if not command:
                continue

            if command[0].lower() == "quit":
                break

            if len(command) != 2:
                print("Invalid command format")
                continue

            cmd_type = command[0].lower()
            symbol = command[1].upper()

            if cmd_type == "sub":
                subject = f"BAR.{symbol}"
                if symbol not in subscribed_symbols:
                    sub = await nc.subscribe(subject, cb=message_handler)
                    subscriptions[symbol] = sub
                    subscribed_symbols.add(symbol)
                    print(f"Subscribed to {symbol}")
                else:
                    print(f"Already subscribed to {symbol}")
            elif cmd_type == "unsub":
                if symbol in subscribed_symbols:
                    sub = subscriptions.pop(symbol)
                    await sub.unsubscribe()
                    subscribed_symbols.remove(symbol)
                    print(f"Unsubscribed from {symbol}")
                else:
                    print(f"Not subscribed to {symbol}")
            else:
                print("Unknown command. Use 'sub' or 'unsub'")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    print("\nShutting down subscriber...")
    await nc.drain()
    await nc.close()

if __name__ == "__main__":
    asyncio.run(main()) 