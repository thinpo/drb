import asyncio
import json
import signal
from datetime import datetime
from nats.aio.client import Client as NATS
from config import NATS_URL

class Dashboard:
    def __init__(self):
        self.nc = NATS()
        self.running = False
        
        # Track latest state
        self.order_books = {}  # symbol -> {bids: [], asks: []}
        self.last_trades = {}  # symbol -> last trade
        self.order_count = 0
        self.trade_count = 0

    async def start(self):
        self.running = True
        
        try:
            await self.nc.connect(NATS_URL)
            print(f"Connected to NATS at {NATS_URL}")
            
            # Subscribe to all relevant subjects
            await self.nc.subscribe("ORDER.*", cb=self._handle_order)
            await self.nc.subscribe("TRADE.*", cb=self._handle_trade)
            await self.nc.subscribe("BOOK.*", cb=self._handle_book)
            
            print("\n=== Exchange Dashboard Started ===")
            print("Monitoring orders, trades, and order books...\n")
            
            # Start display loop
            while self.running:
                self._display_status()
                await asyncio.sleep(1)
                
        except Exception as e:
            print(f"Error: {e}")
            self.running = False

    async def stop(self):
        self.running = False
        await self.nc.drain()
        await self.nc.close()
        print("\nDashboard stopped")

    async def _handle_order(self, msg):
        try:
            subject = msg.subject
            data = json.loads(msg.data.decode())
            symbol = subject.split('.')[1]
            self.order_count += 1
            
            # Print order details
            side = data.get('side', 'UNKNOWN')
            order_type = data.get('type', 'UNKNOWN')
            size = data.get('size', 0)
            price = data.get('price', 0.0)
            
            print(f"\nNew Order ({symbol}): {side} {size} @ {price:.2f} ({order_type})")
            
        except Exception as e:
            print(f"Error handling order: {e}")

    async def _handle_trade(self, msg):
        try:
            subject = msg.subject
            data = json.loads(msg.data.decode())
            symbol = subject.split('.')[1]
            self.trade_count += 1
            self.last_trades[symbol] = data
            
            # Print trade details
            price = data.get('price', 0.0)
            size = data.get('size', 0)
            
            print(f"\nTrade ({symbol}): {size} @ {price:.2f}")
            
        except Exception as e:
            print(f"Error handling trade: {e}")

    async def _handle_book(self, msg):
        try:
            subject = msg.subject
            data = json.loads(msg.data.decode())
            symbol = subject.split('.')[1]
            self.order_books[symbol] = data
            
        except Exception as e:
            print(f"Error handling order book: {e}")

    def _display_status(self):
        print("\033[2J\033[H")  # Clear screen and move cursor to top
        print("=== Exchange Dashboard ===")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Orders: {self.order_count}")
        print(f"Total Trades: {self.trade_count}\n")
        
        # Display order books
        print("Order Books:")
        for symbol, book in self.order_books.items():
            print(f"\n{symbol}:")
            
            # Display asks (in reverse order)
            asks = book.get('asks', [])
            if asks:
                print("  Asks:")
                for price, size in asks:
                    print(f"    {size:6d} @ {price:8.2f}")
            
            # Display bids
            bids = book.get('bids', [])
            if bids:
                print("  Bids:")
                for price, size in bids:
                    print(f"    {size:6d} @ {price:8.2f}")
        
        # Display last trades
        if self.last_trades:
            print("\nLast Trades:")
            for symbol, trade in self.last_trades.items():
                price = trade.get('price', 0.0)
                size = trade.get('size', 0)
                print(f"  {symbol}: {size} @ {price:.2f}")

async def main():
    dashboard = Dashboard()
    
    # Handle graceful shutdown
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        print("\nShutting down...")
        asyncio.create_task(dashboard.stop())
    
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await dashboard.start()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await dashboard.stop()

if __name__ == "__main__":
    asyncio.run(main()) 