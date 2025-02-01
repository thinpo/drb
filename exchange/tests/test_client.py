import asyncio
import json
import time
import threading
import random
import nats
from datetime import datetime

class MarketDataPublisher:
    def __init__(self, exchange_name):
        self.exchange_name = exchange_name
        self.running = False
        self.nc = None
        
    async def connect(self):
        self.nc = await nats.connect("nats://localhost:4222")
        print(f"Connected to NATS server for {self.exchange_name}")

    async def disconnect(self):
        if self.nc:
            await self.nc.drain()
            await self.nc.close()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=lambda: asyncio.run(self._publish_data()))
        self.thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()
        asyncio.run(self.disconnect())

    async def _publish_data(self):
        await self.connect()
        symbols = ["AAPL", "MSFT", "GOOGL"]
        while self.running:
            for symbol in symbols:
                # Publish trade
                trade = {
                    "price": round(random.uniform(100, 200), 2),
                    "size": random.randint(100, 1000),
                    "condition": random.choice(["@", "F", "T"])
                }
                subject = f"TRADE.{symbol}.{self.exchange_name}"
                await self.nc.publish(subject, json.dumps(trade).encode())

                # Publish quote
                quote = {
                    "bid_price": round(random.uniform(100, 200), 2),
                    "ask_price": round(random.uniform(100, 200), 2),
                    "bid_size": random.randint(100, 1000),
                    "ask_size": random.randint(100, 1000)
                }
                subject = f"QUOTE.{symbol}.{self.exchange_name}"
                await self.nc.publish(subject, json.dumps(quote).encode())
                await asyncio.sleep(0.1)  # Simulate some delay

class BarSubscriber:
    def __init__(self):
        self.running = False
        self.nc = None

    async def connect(self):
        self.nc = await nats.connect("nats://localhost:4222")
        print("Connected to NATS server for bar subscription")

    async def disconnect(self):
        if self.nc:
            await self.nc.drain()
            await self.nc.close()

    async def subscribe(self, symbols):
        for symbol in symbols:
            await self.nc.subscribe(f"BAR.{symbol}", cb=self._handle_bar)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=lambda: asyncio.run(self._run()))
        self.thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join()
        asyncio.run(self.disconnect())

    async def _run(self):
        await self.connect()
        await asyncio.sleep(0.1)
        while self.running:
            await asyncio.sleep(0.1)

    async def _handle_bar(self, msg):
        try:
            data = json.loads(msg.data.decode())
            print(f"\nReceived Bar: {msg.subject}")
            print(json.dumps(data, indent=2))
        except Exception as e:
            print(f"Error handling bar: {e}")

async def main_async():
    # Create market data publishers
    nasdaq = MarketDataPublisher("NASDAQ")
    nyse = MarketDataPublisher("NYSE")
    iex = MarketDataPublisher("IEX")

    # Create bar subscriber
    subscriber = BarSubscriber()

    try:
        # Start publishers
        print("Starting market data publishers...")
        nasdaq.start()
        nyse.start()
        iex.start()

        # Start subscriber
        print("Starting bar subscriber...")
        subscriber.start()
        await asyncio.sleep(0.5)  # Wait for connections to establish

        print("Test client running. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down test client...")
        nasdaq.stop()
        nyse.stop()
        iex.stop()
        subscriber.stop()
        print("Test client stopped.")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main() 