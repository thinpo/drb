import subprocess
import sys
import time
import signal
import os
import atexit

class ExchangeDemo:
    def __init__(self, symbols):
        self.symbols = symbols
        self.processes = []
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def start(self):
        print("Starting Exchange Demo...")
        print("=" * 50)

        # Get the virtual environment Python path
        venv_python = os.path.join(os.getcwd(), "venv", "bin", "python3")

        # Start market makers
        print("\nStarting Market Makers...")
        for symbol in self.symbols:
            cmd = [venv_python, "exchange/tests/market_maker.py", symbol]
            proc = subprocess.Popen(cmd)
            self.processes.append((f"Market Maker ({symbol})", proc))
            time.sleep(1)

        # Start trading bots
        print("\nStarting Trading Bots...")
        for symbol in self.symbols:
            cmd = [venv_python, "exchange/tests/trading_bot.py", symbol]
            proc = subprocess.Popen(cmd)
            self.processes.append((f"Trading Bot ({symbol})", proc))
            time.sleep(1)

        # Start dashboard
        print("\nStarting Dashboard...")
        dashboard_cmd = [venv_python, "exchange/tests/dashboard.py"] + self.symbols
        dashboard_proc = subprocess.Popen(dashboard_cmd)
        self.processes.append(("Dashboard", dashboard_proc))

        print("\n" + "=" * 50)
        print("All components started!")
        print("Dashboard available at: http://localhost:8050")
        print("Press Ctrl+C to stop all components")
        
        # Wait for all processes
        try:
            for name, proc in self.processes:
                proc.wait()
        except KeyboardInterrupt:
            self.cleanup()

    def cleanup(self):
        print("\nShutting down all components...")
        for name, proc in self.processes:
            try:
                print(f"Stopping {name}...")
                proc.terminate()
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"Force killing {name}...")
                proc.kill()
            except Exception as e:
                print(f"Error stopping {name}: {e}")
        print("All components stopped.")

    def signal_handler(self, signum, frame):
        self.cleanup()
        sys.exit(0)

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_demo.py SYMBOL1 [SYMBOL2 ...]")
        print("Example: python run_demo.py AAPL MSFT GOOGL")
        sys.exit(1)

    symbols = [s.upper() for s in sys.argv[1:]]
    demo = ExchangeDemo(symbols)
    demo.start()

if __name__ == "__main__":
    main() 