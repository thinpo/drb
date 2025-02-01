import asyncio
import json
import sys
import uuid
from datetime import datetime
import signal
import statistics
from nats.aio.client import Client as NATS
from nats.aio.errors import ErrConnectionClosed, ErrTimeout, ErrNoServers
from config import NATS_URL, TRADE_SUBJECT, ORDER_SUBJECT

class TradingBot:
    def __init__(self, symbol, window_size=3, trade_size=100, threshold=0.1):
        self.symbol = symbol
        self.window_size = window_size
        self.trade_size = trade_size
        self.threshold = threshold
        self.running = False
        self.client_id = str(uuid.uuid4())
        self.prices = []
        self.position = 0  # Net position
        self.nc = NATS()

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
            
            print(f"Trading Bot started for {self.symbol}")
            
        except Exception as e:
            print(f"Error connecting to NATS: {e}")
            self.running = False

    async def stop(self):
        self.running = False
        await self.nc.drain()
        await self.nc.close()
        print(f"Trading Bot stopped for {self.symbol}")

    async def _handle_trade(self, msg):
        try:
            data = json.loads(msg.data.decode())
            
            # Update price history
            self.prices.append(data["price"])
            if len(self.prices) > self.window_size:
                self.prices.pop(0)

            # Update position if we were involved in the trade
            if data["buyer_id"] == self.client_id:
                self.position += data["size"]
            elif data["seller_id"] == self.client_id:
                self.position -= data["size"]

            # Trading logic
            if len(self.prices) >= self.window_size:
                await self._execute_strategy()

        except Exception as e:
            print(f"Error handling trade: {e}")

    async def _execute_strategy(self):
        try:
            # Calculate mean and current price
            mean_price = statistics.mean(self.prices[:-1])  # Mean of previous prices
            current_price = self.prices[-1]
            
            # Calculate z-score
            std_dev = statistics.stdev(self.prices[:-1]) if len(self.prices) > 2 else 0
            if std_dev == 0:
                return
            
            z_score = (current_price - mean_price) / std_dev

            # Trading logic - more aggressive thresholds
            if abs(z_score) > self.threshold:
                # Increase position size based on z-score
                size = int(self.trade_size * min(abs(z_score), 3))
                if z_score > self.threshold and self.position <= 1000:  # Allow larger positions
                    # Price is high, sell
                    await self._place_order("SELL", size)
                elif z_score < -self.threshold and self.position >= -1000:  # Allow larger positions
                    # Price is low, buy
                    await self._place_order("BUY", size)

        except Exception as e:
            print(f"Error executing strategy: {e}")

    async def _place_order(self, side, size):
        order = {
            "order_id": str(uuid.uuid4()),
            "type": "MARKET",
            "side": side,
            "symbol": self.symbol,
            "size": size,
            "client_id": self.client_id
        }

        try:
            # Send order and wait for response
            response = await self.nc.request(
                f"{ORDER_SUBJECT}.{self.symbol}",
                json.dumps(order).encode(),
                timeout=1.0
            )
            
            response_data = response.data.decode()
            print(f"Order response: {response_data}")
            
            # Log trade
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"{timestamp} - {side} {size} {self.symbol} @ MARKET")
            print(f"Current position: {self.position}")

        except ErrTimeout:
            print("Order timeout - no response received")
        except Exception as e:
            print(f"Error placing order: {e}")

async def main():
    if len(sys.argv) != 2:
        print("Usage: python trading_bot.py SYMBOL")
        sys.exit(1)

    symbol = sys.argv[1].upper()
    bot = TradingBot(symbol)

    # Handle graceful shutdown
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        asyncio.create_task(bot.stop())
    
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await bot.start()
        # Keep running until stopped
        while bot.running:
            await asyncio.sleep(1)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await bot.stop()

if __name__ == "__main__":
    asyncio.run(main()) 