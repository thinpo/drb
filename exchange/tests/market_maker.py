import asyncio
import json
import sys
import uuid
import random
from datetime import datetime
import signal
from nats.aio.client import Client as NATS
from nats.aio.errors import ErrConnectionClosed, ErrTimeout, ErrNoServers
from config import NATS_URL, TRADE_SUBJECT, ORDER_SUBJECT

class MarketMaker:
    def __init__(self, symbol, spread=0.1, size=100, price_range=(140.0, 160.0)):
        self.symbol = symbol
        self.spread = spread
        self.size = size
        self.min_price, self.max_price = price_range
        self.current_mid = (self.min_price + self.max_price) / 2
        self.running = False
        self.client_id = str(uuid.uuid4())
        self.nc = NATS()
        
        # Track orders
        self.active_bids = {}  # order_id -> price
        self.active_asks = {}  # order_id -> price

    async def start(self):
        self.running = True
        
        # Connect to NATS
        try:
            await self.nc.connect(NATS_URL)
            print(f"Connected to NATS at {NATS_URL}")
            
            # Subscribe to trades
            await self.nc.subscribe(
                f"{TRADE_SUBJECT}.{self.symbol}",
                cb=self._handle_trade
            )
            
            # Start market making
            asyncio.create_task(self._maintain_market())
            print(f"Market Maker started for {self.symbol}")
            
        except Exception as e:
            print(f"Error connecting to NATS: {e}")
            self.running = False

    async def stop(self):
        self.running = False
        await self.nc.drain()
        await self.nc.close()
        print(f"Market Maker stopped for {self.symbol}")

    async def _handle_trade(self, msg):
        try:
            subject = msg.subject
            data = json.loads(msg.data.decode())
            
            if data["buyer_id"] == self.client_id:
                self.active_bids.pop(data.get("order_id", ""), None)
                # Adjust price down slightly after buying
                self.current_mid -= random.uniform(0.01, 0.05)
            elif data["seller_id"] == self.client_id:
                self.active_asks.pop(data.get("order_id", ""), None)
                # Adjust price up slightly after selling
                self.current_mid += random.uniform(0.01, 0.05)

        except Exception as e:
            print(f"Error handling trade: {e}")

    async def _maintain_market(self):
        while self.running:
            try:
                # Add random walk to price
                self.current_mid += random.uniform(-0.05, 0.05)
                self.current_mid = max(min(self.current_mid, self.max_price), self.min_price)

                # Calculate bid and ask prices
                bid_price = round(self.current_mid - self.spread/2, 2)
                ask_price = round(self.current_mid + self.spread/2, 2)

                # Place new orders
                if len(self.active_bids) < 3:
                    await self._place_limit_order("BUY", bid_price)
                if len(self.active_asks) < 3:
                    await self._place_limit_order("SELL", ask_price)

                await asyncio.sleep(0.1)  # Update more frequently

            except Exception as e:
                print(f"Error maintaining market: {e}")

    async def _place_limit_order(self, side, price):
        order_id = str(uuid.uuid4())
        order = {
            "order_id": order_id,
            "type": "LIMIT",
            "side": side,
            "symbol": self.symbol,
            "price": price,
            "size": self.size,
            "client_id": self.client_id
        }

        try:
            print(f"Placing {side} order for {self.symbol}: {self.size} @ {price:.2f}")
            print(f"Sending request to {ORDER_SUBJECT}.{self.symbol}")
            
            # Send order and wait for response with longer timeout
            response = await self.nc.request(
                f"{ORDER_SUBJECT}.{self.symbol}",
                json.dumps(order).encode(),
                timeout=5.0  # Increased timeout to 5 seconds
            )
            
            response_data = response.data.decode()
            print(f"Received response: {response_data}")
            
            if "accepted" in response_data.lower():
                if side == "BUY":
                    self.active_bids[order_id] = price
                else:
                    self.active_asks[order_id] = price
                print(f"Successfully placed {side} order at {price:.2f}")
            else:
                print(f"Order rejected: {response_data}")

        except (ErrTimeout, ErrNoServers) as e:
            error_msg = str(e)
            print(f"Order error: {error_msg}")
            if "no responders available" in error_msg:
                print("No responders available for order requests. Stopping market maker...")
                self.running = False
                await self.stop()
        except Exception as e:
            error_msg = str(e)
            print(f"Error placing order: {error_msg}")
            if "no responders available" in error_msg:
                print("No responders available for order requests. Stopping market maker...")
                self.running = False
                await self.stop()

async def main():
    if len(sys.argv) != 2:
        print("Usage: python market_maker.py SYMBOL")
        sys.exit(1)

    symbol = sys.argv[1].upper()
    market_maker = MarketMaker(symbol)

    # Handle graceful shutdown
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        asyncio.create_task(market_maker.stop())
    
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await market_maker.start()
        # Keep running until stopped
        while market_maker.running:
            await asyncio.sleep(1)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await market_maker.stop()

if __name__ == "__main__":
    asyncio.run(main()) 