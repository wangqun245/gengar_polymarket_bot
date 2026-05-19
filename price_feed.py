"""Real-time crypto price feed from Binance WebSocket.

Subscribes to a Binance trade stream for tick-by-tick price updates.
Falls back to REST API polling if WebSocket fails.
"""

import json
import time
import threading
import urllib.request
from dataclasses import dataclass, field
from typing import Optional, Callable


BINANCE_WS_BASE = "wss://stream.binance.com:9443/ws"
BINANCE_REST_BASE = "https://api.binance.com/api/v3/ticker/price"


@dataclass
class PriceState:
    """Thread-safe container for current Binance price."""
    price: float = 0.0
    timestamp: float = 0.0
    source: str = "none"
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, price: float, source: str = "ws"):
        with self._lock:
            self.price = price
            self.timestamp = time.time()
            self.source = source

    def get(self) -> tuple[float, float]:
        """Returns (price, age_in_seconds)."""
        with self._lock:
            return self.price, time.time() - self.timestamp

    @property
    def is_fresh(self) -> bool:
        """Price is considered fresh if < 5 seconds old."""
        _, age = self.get()
        return age < 5.0 and self.price > 0


class BinancePriceFeed:
    def __init__(self, symbol: str = "BTCUSDT", label: str = None):
        self.symbol = str(symbol or "BTCUSDT").upper()
        self.label = label or self.symbol.replace("USDT", "")
        self.ws_url = f"{BINANCE_WS_BASE}/{self.symbol.lower()}@trade"
        self.rest_url = f"{BINANCE_REST_BASE}?symbol={self.symbol}"
        self.state = PriceState()
        self._ws_thread: Optional[threading.Thread] = None
        self._running = False
        self._on_price: Optional[Callable] = None

    def start(self, on_price: Callable = None):
        """Start the price feed. Tries WebSocket first, falls back to REST polling."""
        self._on_price = on_price
        self._running = True

        # Try WebSocket first
        self._ws_thread = threading.Thread(target=self._ws_loop, daemon=True)
        self._ws_thread.start()

        # Also start REST poller as backup
        threading.Thread(target=self._rest_poll_loop, daemon=True).start()

        print(f"[price] {self.label} price feed starting...")

    def stop(self):
        self._running = False

    def _ws_loop(self):
        """WebSocket connection to Binance for real-time trades."""
        try:
            import websockets
            import asyncio

            async def connect():
                while self._running:
                    try:
                        async with websockets.connect(self.ws_url) as ws:
                            print(f"[price] {self.label} WebSocket connected to Binance")
                            while self._running:
                                msg = await asyncio.wait_for(ws.recv(), timeout=30)
                                data = json.loads(msg)
                                price = float(data.get("p", 0))
                                if price > 0:
                                    self.state.update(price, source="ws")
                                    self._emit_price(
                                        price,
                                        source="ws",
                                        event_ts=self._event_timestamp(data),
                                        raw=data,
                                    )
                    except Exception as e:
                        if self._running:
                            print(f"[price] {self.label} WebSocket error: {e}, reconnecting in 3s...")
                            await asyncio.sleep(3)

            asyncio.run(connect())
        except ImportError:
            print("[price] websockets not available, using REST polling only")

    def _rest_poll_loop(self):
        """Fallback: poll Binance REST API every 2 seconds."""
        time.sleep(3)  # Give WebSocket a head start
        while self._running:
            try:
                # Only poll if WebSocket data is stale
                if not self.state.is_fresh:
                    req = urllib.request.Request(
                        self.rest_url,
                        headers={"User-Agent": "PolyBot/1.0"},
                    )
                    resp = urllib.request.urlopen(req, timeout=5)
                    data = json.loads(resp.read().decode())
                    price = float(data.get("price", 0))
                    if price > 0:
                        self.state.update(price, source="rest")
                        self._emit_price(price, source="rest", event_ts=time.time(), raw=data)
            except Exception:
                pass
            time.sleep(2)

    def _emit_price(self, price: float, source: str, event_ts: float = None, raw: dict = None):
        if not self._on_price:
            return
        received_ts = time.time()
        try:
            self._on_price(
                price,
                source=source,
                event_ts=event_ts or received_ts,
                received_ts=received_ts,
                raw=raw or {},
            )
        except TypeError:
            self._on_price(price)

    def _event_timestamp(self, data: dict) -> float:
        # Binance trade stream exposes event time E and trade time T in ms.
        for key in ("E", "T"):
            value = data.get(key)
            if value:
                try:
                    return float(value) / 1000.0
                except Exception:
                    pass
        return time.time()

    def get_price(self) -> tuple[float, bool]:
        """Get current price and whether it's fresh.
        
        Returns (price, is_fresh).
        """
        price, age = self.state.get()
        return price, self.state.is_fresh

    def wait_for_price(self, timeout: float = 30) -> float:
        """Block until we have a valid price. Returns price or 0 on timeout."""
        start = time.time()
        while time.time() - start < timeout:
            if self.state.is_fresh:
                return self.state.price
            time.sleep(0.1)
        return 0.0


if __name__ == "__main__":
    feed = BinancePriceFeed()
    feed.start()
    
    print("Waiting for first price...")
    price = feed.wait_for_price(timeout=15)
    if price:
        print(f"BTC price: ${price:,.2f} (source: {feed.state.source})")
    else:
        print("Timeout waiting for price")
    
    # Watch for 10 seconds
    for i in range(10):
        time.sleep(1)
        p, fresh = feed.get_price()
        print(f"  [{i+1}s] ${p:,.2f} {'✓' if fresh else '✗'}")
    
    feed.stop()
