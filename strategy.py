"""Single latency strategy for BTC 5-minute Polymarket markets.

The strategy watches Binance trade prints in fixed 50ms buckets and compares
the newest signal window with rolling baseline statistics from the last few hours.
It enters UP when Binance price and trade volume both spike upward, then exits
when Binance price abruptly reverses downward on elevated volume, provided the
Polymarket sell price clears the configured minimum profit.
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional


@dataclass
class LatencyStrategyConfig:
    bucket_ms: int = 50
    baseline_hours: float = 3.0
    min_baseline_buckets: int = 300
    warmup_minutes: float = 10.0
    min_warmup_buckets: int = 300
    entry_price_z: float = 3.0
    entry_volume_z: float = 1.5
    entry_volume_adaptive: bool = True
    entry_volume_adaptive_minutes: float = 30.0
    entry_volume_adaptive_top_n: int = 6
    entry_volume_adaptive_beat_count: int = 3
    entry_volume_adaptive_min_z: float = 0.50
    exit_price_z: float = 2.5
    exit_volume_z: float = 2.5
    entry_min_price_move_pct: float = 0.015
    exit_min_price_move_pct: float = 0.010
    entry_min_qty: float = 0.0
    exit_min_qty: float = 0.0
    push_std_minutes: float = 10.0
    push_std_min_samples: int = 120
    push_signal_ms: int = 1000
    push_signal_min_samples: int = 3
    entry_push_std_mult: float = 3.0
    exit_push_std_mult: float = 3.0
    max_buy_price: float = 0.60
    min_profit_pct: float = 0.06
    profit_retreat_pct: float = 0.35
    trade_amount: float = 5.0

    @classmethod
    def from_env(cls) -> "LatencyStrategyConfig":
        return cls(
            bucket_ms=int(float(os.getenv("LATENCY_BUCKET_MS", "50"))),
            baseline_hours=float(os.getenv("LATENCY_BASELINE_HOURS", "3")),
            min_baseline_buckets=int(float(os.getenv("LATENCY_MIN_BASELINE_BUCKETS", "300"))),
            warmup_minutes=float(os.getenv("LATENCY_WARMUP_MINUTES", "10")),
            min_warmup_buckets=int(float(os.getenv("LATENCY_MIN_WARMUP_BUCKETS", "300"))),
            entry_price_z=float(os.getenv("LATENCY_ENTRY_PRICE_Z", "3.0")),
            entry_volume_z=float(os.getenv("LATENCY_ENTRY_VOLUME_Z", "1.5")),
            entry_volume_adaptive=os.getenv("LATENCY_ENTRY_VOLUME_ADAPTIVE", "true").lower() == "true",
            entry_volume_adaptive_minutes=float(os.getenv("LATENCY_ENTRY_VOLUME_ADAPTIVE_MINUTES", "30")),
            entry_volume_adaptive_top_n=int(float(os.getenv("LATENCY_ENTRY_VOLUME_ADAPTIVE_TOP_N", "6"))),
            entry_volume_adaptive_beat_count=int(float(os.getenv("LATENCY_ENTRY_VOLUME_ADAPTIVE_BEAT_COUNT", "3"))),
            entry_volume_adaptive_min_z=float(os.getenv("LATENCY_ENTRY_VOLUME_ADAPTIVE_MIN_Z", "0.50")),
            exit_price_z=float(os.getenv("LATENCY_EXIT_PRICE_Z", "2.5")),
            exit_volume_z=float(os.getenv("LATENCY_EXIT_VOLUME_Z", "2.5")),
            entry_min_price_move_pct=float(os.getenv("LATENCY_ENTRY_MIN_PRICE_MOVE_PCT", "0.015")),
            exit_min_price_move_pct=float(os.getenv("LATENCY_EXIT_MIN_PRICE_MOVE_PCT", "0.010")),
            entry_min_qty=float(os.getenv("LATENCY_ENTRY_MIN_QTY", "0")),
            exit_min_qty=float(os.getenv("LATENCY_EXIT_MIN_QTY", "0")),
            push_std_minutes=float(os.getenv("LATENCY_PUSH_STD_MINUTES", "10")),
            push_std_min_samples=int(float(os.getenv("LATENCY_PUSH_STD_MIN_SAMPLES", "120"))),
            push_signal_ms=int(float(os.getenv("LATENCY_PUSH_SIGNAL_MS", "1000"))),
            push_signal_min_samples=int(float(os.getenv("LATENCY_PUSH_SIGNAL_MIN_SAMPLES", "3"))),
            entry_push_std_mult=float(os.getenv("LATENCY_ENTRY_PUSH_STD_MULT", "3.0")),
            exit_push_std_mult=float(os.getenv("LATENCY_EXIT_PUSH_STD_MULT", "3.0")),
            max_buy_price=float(os.getenv("MAX_BUY_PRICE", os.getenv("LATENCY_MAX_BUY_PRICE", "0.60"))),
            min_profit_pct=float(os.getenv("MIN_PROFIT_PCT", os.getenv("LATENCY_MIN_PROFIT_PCT", "0.06"))),
            profit_retreat_pct=float(os.getenv("PROFIT_RETREAT_PCT", os.getenv("LATENCY_PROFIT_RETREAT_PCT", "0.35"))),
            trade_amount=float(os.getenv("TRADE_AMOUNT", os.getenv("MIN_BET", "5.0"))),
        )

    @property
    def bucket_seconds(self) -> float:
        return max(0.001, self.bucket_ms / 1000.0)

    @property
    def baseline_seconds(self) -> float:
        return max(60.0, self.baseline_hours * 3600.0)

    @property
    def warmup_seconds(self) -> float:
        return max(60.0, self.warmup_minutes * 60.0)


@dataclass
class TradeBucket:
    bucket_id: int
    start_ts: float
    updated_ts: float
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    qty: float = 0.0
    notional: float = 0.0
    trades: int = 0

    def add(self, price: float, qty: float, event_ts: float) -> None:
        if self.trades == 0:
            self.open_price = price
            self.high_price = price
            self.low_price = price
        self.high_price = max(self.high_price, price)
        self.low_price = min(self.low_price, price)
        self.close_price = price
        self.updated_ts = event_ts
        self.qty += max(0.0, qty)
        self.notional += max(0.0, qty) * price
        self.trades += 1

    @property
    def move_pct(self) -> float:
        if self.open_price <= 0:
            return 0.0
        return (self.close_price - self.open_price) / self.open_price * 100.0

    @property
    def abs_move_pct(self) -> float:
        return abs(self.move_pct)


@dataclass
class BucketStats:
    count: int
    move_mean: float
    move_std: float
    qty_mean: float
    qty_std: float
    mode: str
    window_seconds: float


@dataclass
class PushPricePoint:
    ts: float
    price: float
    qty: float = 0.0


@dataclass
class PushDeviationStats:
    count: int
    current_count: int
    mean_price: float
    std_price: float
    current_mean: float
    current_std: float
    current_baseline_std: float
    deviation: float
    delta: float
    mean_shift_z: float
    std_ratio: float
    direction: str
    source: str = "window"


@dataclass
class EntryVolumeSample:
    window_id: int
    ts: float
    volume_z: float


@dataclass
class StrategySignal:
    action: str
    side: str
    reason: str
    bucket: TradeBucket
    move_pct: float
    price_z: float
    volume_z: float
    push_deviation: float = 0.0
    push_mean: float = 0.0
    push_std: float = 0.0
    push_delta: float = 0.0
    push_count: int = 0
    push_current_count: int = 0
    push_current_mean: float = 0.0
    push_current_std: float = 0.0
    push_mean_shift_z: float = 0.0
    push_std_ratio: float = 0.0
    push_source: str = "window"


class LatencySpikeStrategy:
    """Detect sudden Binance price/volume shocks in 50ms buckets."""

    def __init__(self, config: LatencyStrategyConfig):
        self.config = config
        self._history: Deque[TradeBucket] = deque()
        self._push_prices: Deque[PushPricePoint] = deque()
        self._current: Optional[TradeBucket] = None
        self._last_signal_bucket_id: dict[str, int] = {}
        self._entry_volume_samples: Deque[EntryVolumeSample] = deque()
        self._entry_volume_window_id: Optional[int] = None
        self._entry_volume_window_max_z: Optional[float] = None
        self._entry_volume_window_max_ts: float = 0.0
        self._entry_volume_window_accepted = False
        self._lock = threading.Lock()

    def ingest_trade(
        self,
        price: float,
        qty: float = 0.0,
        event_ts: float | None = None,
    ) -> list[TradeBucket]:
        if price <= 0:
            return []

        event_ts = event_ts or time.time()
        bucket_seconds = self.config.bucket_seconds
        bucket_id = int(event_ts / bucket_seconds)
        closed: list[TradeBucket] = []

        with self._lock:
            self._push_prices.append(PushPricePoint(ts=event_ts, price=price, qty=max(0.0, qty)))
            self._trim_push_prices(event_ts)

            if self._current is None:
                self._current = self._new_bucket(bucket_id, event_ts, price)

            if bucket_id != self._current.bucket_id:
                closed.append(self._current)
                self._history.append(self._current)
                self._trim_history(event_ts)
                self._current = self._new_bucket(bucket_id, event_ts, price)

            self._current.add(price, qty, event_ts)
        return closed

    def entry_signal(self) -> Optional[StrategySignal]:
        bucket = self._current_snapshot()
        if not bucket or bucket.trades <= 0:
            return None
        if (
            self.config.entry_volume_adaptive
            and self._entry_volume_window_id == self._entry_volume_market_window_id(bucket.updated_ts)
            and self._entry_volume_window_accepted
        ):
            return None
        if self._last_signal_bucket_id.get("entry") == bucket.bucket_id:
            return None

        stats = self._stats()
        if not stats:
            return None
        push_stats = self._push_deviation(bucket.updated_ts, preferred_direction="up")
        if not push_stats:
            return None

        move_pct = self._latest_move_pct(bucket)
        price_z = self._z(move_pct, stats.move_mean, stats.move_std)
        signal_qty, volume_z = self._signal_volume_z(bucket.updated_ts)
        candidate_ok = (
            push_stats.deviation >= self.config.entry_push_std_mult
            and push_stats.direction == "up"
            and signal_qty >= self.config.entry_min_qty
        )
        adaptive_threshold = self._entry_volume_threshold(bucket.updated_ts)
        volume_threshold = adaptive_threshold if self.config.entry_volume_adaptive else self.config.entry_volume_z
        if candidate_ok:
            self._record_entry_volume_candidate(bucket.updated_ts, volume_z)
        if (
            candidate_ok
            and volume_z >= volume_threshold
        ):
            self._last_signal_bucket_id["entry"] = bucket.bucket_id
            self._accept_entry_volume_candidate(bucket.updated_ts, volume_z)
            return StrategySignal(
                action="BUY",
                side="UP",
                reason="binance_up_spike",
                bucket=bucket,
                move_pct=move_pct,
                price_z=price_z,
                volume_z=volume_z,
                push_deviation=push_stats.deviation,
                push_mean=push_stats.mean_price,
                push_std=push_stats.std_price,
                push_delta=push_stats.delta,
                push_count=push_stats.count,
                push_current_count=push_stats.current_count,
                push_current_mean=push_stats.current_mean,
                push_current_std=push_stats.current_std,
                push_mean_shift_z=push_stats.mean_shift_z,
                push_std_ratio=push_stats.std_ratio,
                push_source=push_stats.source,
            )
        if (
            candidate_ok
            and self._last_signal_bucket_id.get("entry_volume_block") != bucket.bucket_id
        ):
            self._last_signal_bucket_id["entry_volume_block"] = bucket.bucket_id
            print(
                "  Entry blocked by volume gate: "
                f"push {push_stats.deviation:+.2f}x, mean_z {push_stats.mean_shift_z:+.2f}, "
                f"signal qty {signal_qty:.4f}, qty z {volume_z:.2f} < {volume_threshold:.2f} "
                f"({self._entry_volume_threshold_summary(bucket.updated_ts)})"
            )
        return None

    def exit_signal(self) -> Optional[StrategySignal]:
        bucket = self._current_snapshot()
        if not bucket or bucket.trades <= 0:
            return None
        if self._last_signal_bucket_id.get("exit") == bucket.bucket_id:
            return None

        stats = self._stats()
        if not stats:
            return None
        push_stats = self._push_deviation(bucket.updated_ts, preferred_direction="down")
        if not push_stats:
            return None

        move_pct = self._latest_move_pct(bucket)
        price_z = self._z(move_pct, stats.move_mean, stats.move_std)
        volume_z = self._z(bucket.qty, stats.qty_mean, stats.qty_std)
        if (
            push_stats.deviation <= -self.config.exit_push_std_mult
            and push_stats.direction == "down"
            and bucket.qty >= self.config.exit_min_qty
            and volume_z >= self.config.exit_volume_z
        ):
            self._last_signal_bucket_id["exit"] = bucket.bucket_id
            return StrategySignal(
                action="SELL",
                side="UP",
                reason="binance_down_reversal",
                bucket=bucket,
                move_pct=move_pct,
                price_z=price_z,
                volume_z=volume_z,
                push_deviation=push_stats.deviation,
                push_mean=push_stats.mean_price,
                push_std=push_stats.std_price,
                push_delta=push_stats.delta,
                push_count=push_stats.count,
                push_current_count=push_stats.current_count,
                push_current_mean=push_stats.current_mean,
                push_current_std=push_stats.current_std,
                push_mean_shift_z=push_stats.mean_shift_z,
                push_std_ratio=push_stats.std_ratio,
                push_source=push_stats.source,
            )
        return None

    def baseline_ready(self) -> bool:
        return self._stats() is not None

    def baseline_summary(self) -> str:
        stats = self._stats()
        if not stats:
            span = self._history_span_seconds()
            history_len = len(self._history_snapshot())
            return (
                f"warming up ({span / 60:.1f}/{self.config.warmup_minutes:.1f} min, "
                f"{history_len}/{self.config.min_warmup_buckets} buckets)"
            )
        return (
            f"{stats.mode} {stats.window_seconds / 60:.1f}m | "
            f"{stats.count} buckets | move std {stats.move_std:.5f}% | "
            f"qty avg {stats.qty_mean:.4f} | {self._push_summary()}"
        )

    def entry_volume_summary(self) -> str:
        return self._entry_volume_threshold_summary(time.time())

    def _new_bucket(self, bucket_id: int, event_ts: float, price: float) -> TradeBucket:
        return TradeBucket(
            bucket_id=bucket_id,
            start_ts=event_ts,
            updated_ts=event_ts,
            open_price=price,
            high_price=price,
            low_price=price,
            close_price=price,
        )

    def _entry_volume_market_window_id(self, ts: float) -> int:
        return int(ts // 300.0)

    def _record_entry_volume_candidate(self, ts: float, volume_z: float) -> None:
        if not self.config.entry_volume_adaptive:
            return
        volume_z = max(0.0, volume_z)
        window_id = self._entry_volume_market_window_id(ts)
        if self._entry_volume_window_id is None:
            self._entry_volume_window_id = window_id
        elif window_id != self._entry_volume_window_id:
            self._finalize_entry_volume_window()
            self._entry_volume_window_id = window_id
            self._entry_volume_window_max_z = None
            self._entry_volume_window_max_ts = 0.0
            self._entry_volume_window_accepted = False

        if self._entry_volume_window_max_z is None or volume_z > self._entry_volume_window_max_z:
            self._entry_volume_window_max_z = volume_z
            self._entry_volume_window_max_ts = ts

    def _accept_entry_volume_candidate(self, ts: float, volume_z: float) -> None:
        if not self.config.entry_volume_adaptive:
            return
        volume_z = max(0.0, volume_z)
        window_id = self._entry_volume_market_window_id(ts)
        if self._entry_volume_window_id != window_id:
            self._entry_volume_window_id = window_id
            self._entry_volume_window_max_z = volume_z
            self._entry_volume_window_max_ts = ts
        self._entry_volume_window_accepted = True
        self._append_entry_volume_sample(window_id, ts, volume_z)

    def _finalize_entry_volume_window(self) -> None:
        if self._entry_volume_window_id is None or self._entry_volume_window_accepted:
            return
        if self._entry_volume_window_max_z is None:
            return
        self._append_entry_volume_sample(
            self._entry_volume_window_id,
            self._entry_volume_window_max_ts or time.time(),
            self._entry_volume_window_max_z,
        )

    def _append_entry_volume_sample(self, window_id: int, ts: float, volume_z: float) -> None:
        volume_z = max(0.0, volume_z)
        self._entry_volume_samples = deque(
            sample for sample in self._entry_volume_samples if sample.window_id != window_id
        )
        self._entry_volume_samples.append(EntryVolumeSample(window_id=window_id, ts=ts, volume_z=volume_z))
        self._trim_entry_volume_samples(ts)

    def _trim_entry_volume_samples(self, now_ts: float) -> None:
        cutoff = now_ts - max(300.0, self.config.entry_volume_adaptive_minutes * 60.0)
        while self._entry_volume_samples and self._entry_volume_samples[0].ts < cutoff:
            self._entry_volume_samples.popleft()

    def _entry_volume_recent_values(self, now_ts: float) -> list[float]:
        if not self.config.entry_volume_adaptive:
            return []
        self._trim_entry_volume_samples(now_ts)
        return [sample.volume_z for sample in self._entry_volume_samples]

    def _entry_volume_top_values(self, now_ts: float) -> list[float]:
        values = sorted(self._entry_volume_recent_values(now_ts), reverse=True)
        top_n = max(1, self.config.entry_volume_adaptive_top_n)
        return values[:top_n]

    def _entry_volume_threshold(self, now_ts: float) -> float:
        if not self.config.entry_volume_adaptive:
            return self.config.entry_volume_z
        top_values = self._entry_volume_top_values(now_ts)
        if not top_values:
            return self.config.entry_volume_z
        beat_count = max(1, self.config.entry_volume_adaptive_beat_count)
        index = max(0, len(top_values) - beat_count)
        return max(self.config.entry_volume_adaptive_min_z, top_values[index])

    def _entry_volume_threshold_summary(self, now_ts: float) -> str:
        if not self.config.entry_volume_adaptive:
            return f"static volume z>={self.config.entry_volume_z:.2f}"
        top_values = self._entry_volume_top_values(now_ts)
        threshold = self._entry_volume_threshold(now_ts)
        if not top_values:
            return (
                f"adaptive volume z>={threshold:.2f}, no history, "
                f"bootstrap static {self.config.entry_volume_z:.2f}"
            )
        top_text = ",".join(f"{value:.2f}" for value in top_values[: self.config.entry_volume_adaptive_top_n])
        return (
            f"adaptive volume z>={threshold:.2f}, "
            f"beat {self.config.entry_volume_adaptive_beat_count}/{self.config.entry_volume_adaptive_top_n}, "
            f"top{len(top_values)}/{self.config.entry_volume_adaptive_top_n} [{top_text}]"
        )

    def _trim_history(self, now_ts: float) -> None:
        cutoff = now_ts - self.config.baseline_seconds
        while self._history and self._history[0].start_ts < cutoff:
            self._history.popleft()

    def _trim_push_prices(self, now_ts: float) -> None:
        cutoff = now_ts - max(self.config.baseline_seconds, self.config.push_std_minutes * 60.0)
        while self._push_prices and self._push_prices[0].ts < cutoff:
            self._push_prices.popleft()

    def _stats(self) -> Optional[BucketStats]:
        buckets = self._baseline_buckets()
        if not buckets:
            return None

        moves = []
        prev_close = buckets[0].close_price
        for bucket in buckets[1:]:
            if prev_close > 0:
                moves.append((bucket.close_price - prev_close) / prev_close * 100.0)
            prev_close = bucket.close_price
        if not moves:
            return None
        qtys = [b.qty for b in buckets]
        move_mean, move_std = self._mean_std(moves)
        qty_mean, qty_std = self._mean_std(qtys)
        return BucketStats(
            count=len(buckets),
            move_mean=move_mean,
            move_std=max(move_std, 0.00001),
            qty_mean=qty_mean,
            qty_std=max(qty_std, 0.00001),
            mode="full" if self._history_span_seconds() >= self.config.baseline_seconds else "warmup",
            window_seconds=self.config.baseline_seconds
            if self._history_span_seconds() >= self.config.baseline_seconds
            else self.config.warmup_seconds,
        )

    def _baseline_buckets(self) -> list[TradeBucket]:
        buckets = [b for b in self._history_snapshot() if b.trades > 0]
        if not buckets:
            return []

        span = buckets[-1].start_ts - buckets[0].start_ts
        if span >= self.config.baseline_seconds:
            return buckets if len(buckets) >= self.config.min_baseline_buckets else []

        if span < self.config.warmup_seconds:
            return []

        cutoff = buckets[-1].start_ts - self.config.warmup_seconds
        warmup_buckets = [bucket for bucket in buckets if bucket.start_ts >= cutoff]
        if len(warmup_buckets) < self.config.min_warmup_buckets:
            return []
        return warmup_buckets

    def _history_span_seconds(self) -> float:
        buckets = [b for b in self._history_snapshot() if b.trades > 0]
        if len(buckets) < 2:
            return 0.0
        return max(0.0, buckets[-1].start_ts - buckets[0].start_ts)

    def _history_snapshot(self) -> list[TradeBucket]:
        with self._lock:
            return list(self._history)

    def _push_snapshot(self) -> list[PushPricePoint]:
        with self._lock:
            return list(self._push_prices)

    def _current_snapshot(self) -> Optional[TradeBucket]:
        with self._lock:
            if not self._current:
                return None
            return TradeBucket(**self._current.__dict__)

    def _mean_std(self, values: list[float]) -> tuple[float, float]:
        if not values:
            return 0.0, 0.0
        mean = sum(values) / len(values)
        if len(values) == 1:
            return mean, 0.0
        var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        return mean, math.sqrt(max(0.0, var))

    def _z(self, value: float, mean: float, std: float) -> float:
        if std <= 0:
            return 0.0
        return (value - mean) / std

    def _signal_volume_z(self, current_ts: float) -> tuple[float, float]:
        if current_ts <= 0:
            return 0.0, 0.0

        signal_seconds = max(0.001, self.config.push_signal_ms / 1000.0)
        current_cutoff = current_ts - signal_seconds
        baseline_cutoff = current_cutoff - self.config.push_std_minutes * 60.0
        points = self._push_snapshot()
        current_qty = sum(
            point.qty for point in points
            if current_cutoff <= point.ts <= current_ts
        )
        baseline_points = [
            point for point in points
            if baseline_cutoff <= point.ts < current_cutoff
        ]
        if len(baseline_points) < self.config.push_std_min_samples:
            return current_qty, 0.0

        window_qtys: list[float] = []
        rolling_qty = 0.0
        left = 0
        for right, point in enumerate(baseline_points):
            rolling_qty += point.qty
            cutoff = point.ts - signal_seconds
            while left <= right and baseline_points[left].ts < cutoff:
                rolling_qty -= baseline_points[left].qty
                left += 1
            window_qtys.append(rolling_qty)

        qty_mean, qty_std = self._mean_std(window_qtys)
        return current_qty, self._z(current_qty, qty_mean, max(qty_std, 0.00001))

    def _push_deviation(
        self,
        current_ts: float,
        preferred_direction: str | None = None,
    ) -> Optional[PushDeviationStats]:
        if current_ts <= 0:
            return None
        signal_seconds = max(0.001, self.config.push_signal_ms / 1000.0)
        current_cutoff = current_ts - signal_seconds
        baseline_cutoff = current_cutoff - self.config.push_std_minutes * 60.0
        points = self._push_snapshot()
        baseline_prices = [
            point.price for point in points
            if baseline_cutoff <= point.ts < current_cutoff and point.price > 0
        ]
        current_points = [
            point for point in points
            if current_cutoff <= point.ts <= current_ts and point.price > 0
        ]
        if len(baseline_prices) < self.config.push_std_min_samples:
            return None
        mean_price, std_price = self._mean_std(baseline_prices)
        std_price = max(std_price, 0.01)

        candidates: list[PushDeviationStats] = []
        current_prices = [point.price for point in current_points]
        if len(current_prices) >= max(2, self.config.push_signal_min_samples):
            current_mean, current_std = self._mean_std(current_prices)
            delta = current_prices[-1] - current_prices[0]
            direction = "up" if delta > 0 else "down" if delta < 0 else "flat"
            current_baseline_std = abs(delta)
            unsigned_deviation = current_baseline_std / std_price
            deviation = unsigned_deviation if delta >= 0 else -unsigned_deviation
            candidates.append(PushDeviationStats(
                count=len(baseline_prices),
                current_count=len(current_prices),
                mean_price=mean_price,
                std_price=std_price,
                current_mean=current_mean,
                current_std=current_std,
                current_baseline_std=current_baseline_std,
                deviation=deviation,
                delta=delta,
                mean_shift_z=delta / std_price,
                std_ratio=current_std / std_price,
                direction=direction,
                source="window",
            ))

        recent_points = [point for point in points if point.ts <= current_ts and point.price > 0]
        recent_n = max(2, self.config.push_signal_min_samples)
        if len(recent_points) >= recent_n:
            recent_prices = [point.price for point in recent_points[-recent_n:]]
            recent_mean, recent_std = self._mean_std(recent_prices)
            delta = recent_prices[-1] - recent_prices[0]
            direction = "up" if delta > 0 else "down" if delta < 0 else "flat"
            deviation = abs(delta) / std_price
            if delta < 0:
                deviation = -deviation
            candidates.append(PushDeviationStats(
                count=len(baseline_prices),
                current_count=len(recent_prices),
                mean_price=mean_price,
                std_price=std_price,
                current_mean=recent_mean,
                current_std=recent_std,
                current_baseline_std=abs(delta),
                deviation=deviation,
                delta=delta,
                mean_shift_z=delta / std_price,
                std_ratio=recent_std / std_price,
                direction=direction,
                source="recent",
            ))

        if not candidates:
            return None
        if preferred_direction == "up":
            up_candidates = [stats for stats in candidates if stats.direction == "up" and stats.deviation > 0]
            if up_candidates:
                return max(up_candidates, key=lambda stats: stats.deviation)
            return None
        if preferred_direction == "down":
            down_candidates = [stats for stats in candidates if stats.direction == "down" and stats.deviation < 0]
            if down_candidates:
                return min(down_candidates, key=lambda stats: stats.deviation)
            return None
        return max(candidates, key=lambda stats: abs(stats.deviation))

    def _push_summary(self) -> str:
        bucket = self._current_snapshot()
        if not bucket:
            return "push dev n/a"
        stats = self._push_deviation(bucket.updated_ts)
        if not stats:
            return "push dev warming"
        return (
            f"push {stats.source} {stats.deviation:+.2f}x "
            f"(mean_z {stats.mean_shift_z:+.2f}, d ${stats.delta:+.2f}, "
            f"base_std ${stats.std_price:.2f}, cur_std {stats.std_ratio:.2f}x, "
            f"n={stats.current_count}/{stats.count})"
        )
        return (
            f"push dev {stats.deviation:+.2f}σ "
            f"(Δ ${stats.delta:+.2f}, std ${stats.std_price:.2f}, n={stats.count})"
        )

    def _latest_move_pct(self, bucket: TradeBucket) -> float:
        if self._history:
            prev_close = self._history[-1].close_price
            if prev_close > 0:
                return (bucket.close_price - prev_close) / prev_close * 100.0
        return bucket.move_pct
