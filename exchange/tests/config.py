# NATS configuration
NATS_URL = "nats://localhost:4222"

# NATS subjects
MARKET_DATA_SUBJECT = "market.data"  # For order submission and market data
BAR_DATA_SUBJECT = "bar.data"        # For trade and bar data

# NATS subject patterns
TRADE_SUBJECT = "TRADE"
BOOK_SUBJECT = "BOOK"
ORDER_SUBJECT = "ORDER"

# Common port configuration for all components
MARKET_DATA_PORT = 9555  # For order submission and market data
BAR_PUBLISH_PORT = 9556  # For trade and bar data

# ZMQ socket endpoints
MARKET_DATA_ENDPOINT = f"tcp://localhost:{MARKET_DATA_PORT}"
BAR_PUBLISH_ENDPOINT = f"tcp://localhost:{BAR_PUBLISH_PORT}" 