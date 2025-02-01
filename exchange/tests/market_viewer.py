import asyncio
import json
import time
import sys
from datetime import datetime
import curses
import threading
import statistics
import nats

class MarketViewer:
    def __init__(self, symbols):
        self.symbols = symbols
        self.running = False
        self.nc = None
        
        # Market data storage
        self.trades = {sym: [] for sym in symbols}  # Recent trades
        self.bars = {sym: None for sym in symbols}  # Latest bar
        self.stats = {sym: {"vwap": 0.0, "volume": 0, "trades": 0} for sym in symbols}

    async def connect(self):
        self.nc = await nats.connect("nats://localhost:4222")
        print("Connected to NATS server")

        # Subscribe to trades and bars
        for symbol in self.symbols:
            await self.nc.subscribe(f"TRADE.{symbol}.*", cb=self._handle_trade)
            await self.nc.subscribe(f"BAR.{symbol}", cb=self._handle_bar)

    async def disconnect(self):
        if self.nc:
            await self.nc.drain()
            await self.nc.close()

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
        while self.running:
            await asyncio.sleep(0.1)

    async def _handle_trade(self, msg):
        try:
            data = json.loads(msg.data.decode())
            symbol = data["symbol"]
            self.trades[symbol].append(data)
            if len(self.trades[symbol]) > 10:  # Keep last 10 trades
                self.trades[symbol].pop(0)

            # Update statistics
            trades = self.trades[symbol]
            if trades:
                total_volume = sum(t["size"] for t in trades)
                vwap = sum(t["price"] * t["size"] for t in trades) / total_volume
                self.stats[symbol] = {
                    "vwap": vwap,
                    "volume": total_volume,
                    "trades": len(trades)
                }
        except Exception as e:
            if self.running:
                print(f"Error handling trade: {e}")

    async def _handle_bar(self, msg):
        try:
            data = json.loads(msg.data.decode())
            symbol = data["symbol"]
            self.bars[symbol] = data
        except Exception as e:
            if self.running:
                print(f"Error handling bar: {e}")

    def display(self, stdscr):
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)

        while self.running:
            try:
                stdscr.clear()
                row = 0

                # Display header
                stdscr.addstr(row, 0, "Market Data Viewer", curses.A_BOLD)
                row += 2

                for symbol in self.symbols:
                    # Symbol header
                    stdscr.addstr(row, 0, f"=== {symbol} ===", curses.A_BOLD)
                    row += 1

                    # Display bar data
                    bar = self.bars[symbol]
                    if bar:
                        timestamp = datetime.fromtimestamp(bar["timestamp"] / 1000.0)
                        stdscr.addstr(row, 0, f"Last Bar ({timestamp}):")
                        row += 1
                        stdscr.addstr(row, 2, f"OHLC: {bar['open']:.2f}/{bar['high']:.2f}/{bar['low']:.2f}/{bar['close']:.2f}")
                        row += 1
                        stdscr.addstr(row, 2, f"Volume: {bar['volume']}")
                        row += 1

                    # Display statistics
                    stats = self.stats[symbol]
                    stdscr.addstr(row, 0, "Statistics:")
                    row += 1
                    stdscr.addstr(row, 2, f"VWAP: {stats['vwap']:.2f}")
                    row += 1
                    stdscr.addstr(row, 2, f"Volume: {stats['volume']}")
                    row += 1
                    stdscr.addstr(row, 2, f"Trades: {stats['trades']}")
                    row += 1

                    # Display recent trades
                    stdscr.addstr(row, 0, "Recent Trades:")
                    row += 1
                    for trade in reversed(self.trades[symbol][-5:]):  # Show last 5 trades
                        timestamp = datetime.fromtimestamp(trade["timestamp"] / 1000.0)
                        trade_str = f"{timestamp.strftime('%H:%M:%S')} - {trade['price']:.2f} @ {trade['size']}"
                        stdscr.addstr(row, 2, trade_str)
                        row += 1

                    row += 1

                stdscr.addstr(row, 0, "Press 'q' to quit")
                stdscr.refresh()

                # Check for quit
                c = stdscr.getch()
                if c == ord('q'):
                    self.running = False

                time.sleep(0.1)

            except Exception as e:
                stdscr.addstr(0, 0, f"Error: {e}")
                stdscr.refresh()
                time.sleep(1)

def main():
    if len(sys.argv) < 2:
        print("Usage: python market_viewer.py SYMBOL1 [SYMBOL2 ...]")
        sys.exit(1)

    symbols = [s.upper() for s in sys.argv[1:]]
    viewer = MarketViewer(symbols)

    try:
        viewer.start()
        curses.wrapper(viewer.display)
    finally:
        viewer.stop()

if __name__ == "__main__":
    main() 