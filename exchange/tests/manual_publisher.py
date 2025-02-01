import asyncio
import json
import nats
import sys

async def main():
    # Connect to NATS
    nc = await nats.connect("nats://localhost:4222")
    print("Connected to NATS server")

    print("Manual Market Data Publisher")
    print("Available commands:")
    print("1. trade EXCHANGE SYMBOL PRICE SIZE CONDITION")
    print("2. quote EXCHANGE SYMBOL BID_PRICE ASK_PRICE BID_SIZE ASK_SIZE")
    print("3. quit")
    print("\nExample:")
    print("trade NASDAQ AAPL 150.25 100 @")
    print("quote NYSE MSFT 200.50 201.00 500 300")

    while True:
        try:
            command = input("\nEnter command: ").strip().split()
            if not command:
                continue

            if command[0].lower() == "quit":
                break

            if len(command) < 2:
                print("Invalid command format")
                continue

            cmd_type = command[0].lower()
            exchange = command[1].upper()
            
            if exchange not in ["NASDAQ", "NYSE", "IEX"]:
                print("Invalid exchange. Use NASDAQ, NYSE, or IEX")
                continue

            if cmd_type == "trade":
                if len(command) != 6:
                    print("Usage: trade EXCHANGE SYMBOL PRICE SIZE CONDITION")
                    continue
                
                _, exchange, symbol, price, size, condition = command
                trade = {
                    "price": float(price),
                    "size": int(size),
                    "condition": condition
                }
                subject = f"TRADE.{symbol}.{exchange}"
                await nc.publish(subject, json.dumps(trade).encode())
                print(f"Published trade: {subject} - {trade}")

            elif cmd_type == "quote":
                if len(command) != 7:
                    print("Usage: quote EXCHANGE SYMBOL BID_PRICE ASK_PRICE BID_SIZE ASK_SIZE")
                    continue
                
                _, exchange, symbol, bid_price, ask_price, bid_size, ask_size = command
                quote = {
                    "bid_price": float(bid_price),
                    "ask_price": float(ask_price),
                    "bid_size": int(bid_size),
                    "ask_size": int(ask_size)
                }
                subject = f"QUOTE.{symbol}.{exchange}"
                await nc.publish(subject, json.dumps(quote).encode())
                print(f"Published quote: {subject} - {quote}")

            else:
                print("Unknown command. Use 'trade' or 'quote'")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    print("\nShutting down publisher...")
    await nc.drain()
    await nc.close()

if __name__ == "__main__":
    asyncio.run(main()) 