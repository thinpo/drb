import asyncio
import json
import nats
import sys
import uuid
from datetime import datetime

async def main():
    # Connect to NATS
    nc = await nats.connect("nats://localhost:4222")
    print("Connected to NATS server")

    # Generate unique client ID
    client_id = str(uuid.uuid4())

    print("Order Client")
    print("Available commands:")
    print("1. market BUY/SELL SYMBOL SIZE")
    print("2. limit BUY/SELL SYMBOL PRICE SIZE")
    print("3. quit")
    print("\nExample:")
    print("market BUY AAPL 100")
    print("limit SELL MSFT 200.50 500")

    async def handle_trade(msg):
        try:
            data = json.loads(msg.data.decode())
            # Only show trades where we're involved
            if data["buyer_id"] == client_id or data["seller_id"] == client_id:
                timestamp = datetime.fromtimestamp(data["timestamp"] / 1000.0)
                print(f"\nTrade Execution at {timestamp}:")
                print(f"Symbol: {data['symbol']}")
                print(f"Price:  {data['price']:.2f}")
                print(f"Size:   {data['size']}")
                print(f"Role:   {'Buyer' if data['buyer_id'] == client_id else 'Seller'}")
        except Exception as e:
            print(f"Error handling trade: {e}")

    # Subscribe to trade notifications
    await nc.subscribe("TRADE.*", cb=handle_trade)

    while True:
        try:
            command = input("\nEnter command: ").strip().split()
            if not command:
                continue

            if command[0].lower() == "quit":
                break

            if command[0].lower() not in ["market", "limit"] or len(command) not in [4, 5]:
                print("Invalid command format")
                continue

            order_type = command[0].lower()
            side = command[1].upper()
            symbol = command[2].upper()

            if order_type == "market" and len(command) == 4:
                size = int(command[3])
                order = {
                    "client_id": client_id,
                    "type": "MARKET",
                    "side": side,
                    "symbol": symbol,
                    "size": size
                }
            elif order_type == "limit" and len(command) == 5:
                price = float(command[3])
                size = int(command[4])
                order = {
                    "client_id": client_id,
                    "type": "LIMIT",
                    "side": side,
                    "symbol": symbol,
                    "price": price,
                    "size": size
                }
            else:
                print("Invalid command format")
                continue

            # Send order and wait for response
            try:
                response = await nc.request(f"ORDER.{symbol}", json.dumps(order).encode(), timeout=1.0)
                print(f"Order Response: {response.data.decode()}")
            except Exception as e:
                print(f"Error sending order: {e}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

    print("\nShutting down order client...")
    await nc.drain()
    await nc.close()

if __name__ == "__main__":
    asyncio.run(main()) 