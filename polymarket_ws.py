"""Real-time Polymarket CLOB market WebSocket feed."""

import asyncio
import json
import threading
import time
from dataclasses import dataclass
from typing import Optional


POLYMARKET_MARKET_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


@dataclass
class TokenPrice:
    asset_id: str
    best_bid: float = 0.0
    best_ask: float = 0.0
    last_trade: float = 0.0
    bid_size: float = 0.0
    ask_size: float = 0.0
    timestamp: float = 0.0
    label: str = ""

    @property
    def mid(self) -> float:
        if self.best_bid > 0 and self.best_ask > 0:
            return (self.best_bid + self.best_ask) / 2
        return self.last_trade


class PolymarketMarketFeed:
    """Maintains live best bid/ask for subscribed Polymarket token IDs."""

    def __init__(self, url: str = POLYMARKET_MARKET_WS):
        self.url = url
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._asset_ids: list[str] = []
        self._labels: dict[str, str] = {}
        self._prices: dict[str, TokenPrice] = {}
        self._subscription_version = 0
        self._connected = False
        self._last_message = 0.0

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print("[poly-ws] Market WebSocket feed starting...")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def subscribe(self, asset_ids: list[str], labels: dict[str, str] = None):
        clean_ids = [str(asset_id) for asset_id in asset_ids if asset_id]
        labels = labels or {}
        with self._lock:
            if clean_ids == self._asset_ids:
                return
            self._asset_ids = clean_ids
            self._labels = {str(k): v for k, v in labels.items()}
            self._prices = {}
            for asset_id in clean_ids:
                self._prices[asset_id] = TokenPrice(
                    asset_id=asset_id,
                    label=self._labels.get(asset_id, ""),
                )
            self._subscription_version += 1
            self._last_message = 0.0
        short = ", ".join(f"{self._labels.get(a, '')}:{a[:8]}" for a in clean_ids)
        print(f"[poly-ws] Subscribed assets: {short}")

    def get_price(self, asset_id: str) -> Optional[TokenPrice]:
        with self._lock:
            price = self._prices.get(str(asset_id))
            if not price:
                return None
            return TokenPrice(**price.__dict__)

    def get_prices(self) -> dict[str, TokenPrice]:
        with self._lock:
            return {asset_id: TokenPrice(**price.__dict__) for asset_id, price in self._prices.items()}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def last_message_age(self) -> float:
        if self._last_message <= 0:
            return 999999.0
        return time.time() - self._last_message

    def _run_loop(self):
        try:
            asyncio.run(self._connect_loop())
        except Exception as e:
            if self._running:
                print(f"[poly-ws] Feed stopped: {e}")

    async def _connect_loop(self):
        import websockets

        while self._running:
            try:
                with self._lock:
                    asset_ids = list(self._asset_ids)
                    version = self._subscription_version
                if not asset_ids:
                    await asyncio.sleep(0.25)
                    continue

                async with websockets.connect(self.url, ping_interval=None) as ws:
                    self._connected = True
                    await self._send_subscription(ws, asset_ids)
                    ping_task = asyncio.create_task(self._ping_loop(ws))
                    try:
                        while self._running:
                            with self._lock:
                                if version != self._subscription_version:
                                    break
                            raw = await asyncio.wait_for(ws.recv(), timeout=30)
                            self._last_message = time.time()
                            self._handle_raw_message(raw)
                    finally:
                        ping_task.cancel()
                        self._connected = False
            except Exception as e:
                self._connected = False
                if self._running:
                    print(f"[poly-ws] WebSocket error: {e}; reconnecting in 2s")
                    await asyncio.sleep(2)

    async def _send_subscription(self, ws, asset_ids: list[str]):
        msg = {
            "assets_ids": asset_ids,
            "type": "market",
            "custom_feature_enabled": True,
        }
        await ws.send(json.dumps(msg))

    async def _ping_loop(self, ws):
        while self._running:
            await asyncio.sleep(10)
            try:
                await ws.send("PING")
            except Exception:
                return

    def _handle_raw_message(self, raw: str):
        try:
            payload = json.loads(raw)
        except Exception:
            return

        messages = payload if isinstance(payload, list) else [payload]
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            event_type = msg.get("event_type")
            if event_type == "book":
                self._handle_book(msg)
            elif event_type == "price_change":
                for change in msg.get("price_changes", []):
                    self._handle_price_change(change, msg.get("timestamp"))
            elif event_type == "best_bid_ask":
                self._handle_best_bid_ask(msg)
            elif event_type == "last_trade_price":
                self._handle_last_trade(msg)

    def _handle_book(self, msg: dict):
        asset_id = str(msg.get("asset_id", ""))
        if not asset_id:
            return
        bids = msg.get("bids", []) or []
        asks = msg.get("asks", []) or []
        best_bid, bid_size = self._best_level(bids, highest=True)
        best_ask, ask_size = self._best_level(asks, highest=False)
        self._update_price(
            asset_id,
            best_bid=best_bid,
            best_ask=best_ask,
            bid_size=bid_size,
            ask_size=ask_size,
        )

    def _handle_price_change(self, msg: dict, timestamp=None):
        asset_id = str(msg.get("asset_id", ""))
        if not asset_id:
            return
        self._update_price(
            asset_id,
            best_bid=self._as_float(msg.get("best_bid")),
            best_ask=self._as_float(msg.get("best_ask")),
            timestamp=timestamp,
        )

    def _handle_best_bid_ask(self, msg: dict):
        asset_id = str(msg.get("asset_id", ""))
        if not asset_id:
            return
        self._update_price(
            asset_id,
            best_bid=self._as_float(msg.get("best_bid")),
            best_ask=self._as_float(msg.get("best_ask")),
            timestamp=msg.get("timestamp"),
        )

    def _handle_last_trade(self, msg: dict):
        asset_id = str(msg.get("asset_id", ""))
        if not asset_id:
            return
        self._update_price(
            asset_id,
            last_trade=self._as_float(msg.get("price")),
            timestamp=msg.get("timestamp"),
        )

    def _update_price(
        self,
        asset_id: str,
        best_bid: float = None,
        best_ask: float = None,
        last_trade: float = None,
        bid_size: float = None,
        ask_size: float = None,
        timestamp=None,
    ):
        with self._lock:
            if asset_id not in self._asset_ids:
                return
            price = self._prices.setdefault(
                asset_id,
                TokenPrice(asset_id=asset_id, label=self._labels.get(asset_id, "")),
            )
            if best_bid is not None and best_bid >= 0:
                price.best_bid = best_bid
            if best_ask is not None and best_ask >= 0:
                price.best_ask = best_ask
            if last_trade is not None and last_trade >= 0:
                price.last_trade = last_trade
            if bid_size is not None and bid_size >= 0:
                price.bid_size = bid_size
            if ask_size is not None and ask_size >= 0:
                price.ask_size = ask_size
            price.timestamp = self._normalize_ts(timestamp) or time.time()

    def _best_level(self, levels: list, highest: bool) -> tuple[float, float]:
        best_price = 0.0
        best_size = 0.0
        for level in levels:
            price = self._as_float(level.get("price") if isinstance(level, dict) else None)
            size = self._as_float(level.get("size") if isinstance(level, dict) else None)
            if price <= 0:
                continue
            if best_price <= 0 or (highest and price > best_price) or (not highest and price < best_price):
                best_price = price
                best_size = size
        return best_price, best_size

    def _as_float(self, value) -> float:
        try:
            if value in ("", None):
                return 0.0
            return float(value)
        except Exception:
            return 0.0

    def _normalize_ts(self, value) -> float:
        ts = self._as_float(value)
        if ts <= 0:
            return 0.0
        if ts > 10_000_000_000:
            return ts / 1000.0
        return ts
