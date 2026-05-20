"""PolyBot single-strategy runner.

Only one strategy is active:
  1. Build a rolling Binance BTCUSDT baseline from 50ms trade buckets.
  2. Buy Polymarket BTC UP when Binance price and filled quantity spike upward.
  3. Sell the UP position when Binance price reverses downward on elevated
     quantity and the live Polymarket sell price satisfies the minimum profit.
  4. If Binance does not reverse, trail the best unrealized profit and sell
     when Polymarket profit retreats by the configured percentage.
"""

from __future__ import annotations

import builtins
import atexit
import json
import math
import os
import random
import signal
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
from dotenv import load_dotenv

from executor import Executor
from market import MarketWindow, get_current_market, get_market_winner
from polymarket_ws import PolymarketMarketFeed
from price_feed import BinancePriceFeed
from strategy import LatencySpikeStrategy, LatencyStrategyConfig
from telegram_notifier import TelegramNotifier


_ORIGINAL_PRINT = builtins.print


def _timestamped_print(*args, **kwargs):
    now = datetime.now()
    prefix = now.strftime("[%Y-%m-%d %H:%M:%S.%f")[:-3] + "]"
    if args:
        first = str(args[0])
        if first.startswith("\n"):
            args = ("\n" + prefix + " " + first[1:], *args[1:])
        else:
            args = (prefix, *args)
    else:
        args = (prefix,)
    _ORIGINAL_PRINT(*args, **kwargs)
    try:
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        log_dir = Path(os.getenv("LOG_DIR", "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"bot_{now.strftime('%Y%m%d')}.log"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(sep.join(str(arg) for arg in args) + end)
    except Exception:
        pass


builtins.print = _timestamped_print


@dataclass
class Position:
    side: str
    token_id: str
    window_ts: int
    market_slug: str
    entry_price: float
    shares: float
    cost: float
    entry_btc_price: float
    opening_price: float
    entry_ts: float
    exit_revenue: float = 0.0
    peak_unrealized_profit: float = 0.0
    peak_sell_price: float = 0.0
    last_sell_price: float = 0.0
    closed: bool = False


@dataclass
class PendingSignal:
    kind: str
    created_ts: float
    deadline_ts: float
    ref_poly_price: float
    ref_poly_received_ts: float
    signal: object
    trend_start_ts: float = 0.0
    latency_target_ms: float = 0.0
    trend_window_ms: float = 0.0
    safety_margin_ms: float = 0.0
    waits: int = 0
    observation_logged: bool = False


@dataclass
class PriceSample:
    ts: float
    price: float
    segment: int = 0


@dataclass
class WindowLatencyMatch:
    window_start: int
    btc_start_ts: float
    latency_ms: float
    sse: float
    samples: int
    second_sse: float = 0.0
    top_lags: str = ""
    scale: float = 0.0
    btc_range: float = 0.0
    poly_range: float = 0.0


@dataclass
class FeedGapStats:
    count: int = 0
    avg_gap_ms: float = 0.0
    max_gap_ms: float = 0.0
    last_gap_ms: float = 0.0
    last_ts: float = 0.0

    def update(self, ts: float) -> None:
        if ts <= 0:
            return
        if self.last_ts > 0 and ts > self.last_ts:
            gap_ms = (ts - self.last_ts) * 1000.0
            self.last_gap_ms = gap_ms
            self.max_gap_ms = max(self.max_gap_ms, gap_ms)
            if self.count <= 0:
                self.avg_gap_ms = gap_ms
            else:
                self.avg_gap_ms = ((self.avg_gap_ms * self.count) + gap_ms) / (self.count + 1)
            self.count += 1
        elif self.last_ts <= 0:
            self.count = max(self.count, 0)
        self.last_ts = max(self.last_ts, ts)


@dataclass
class BinanceTradePoint:
    ts: float
    price: float
    qty: float
    segment: int = 0


@dataclass
class EdgeModelFeatures:
    key: tuple[int, int, int, int]
    elapsed_seconds: float
    poly_price: float
    btc_move_pct: float
    qty_ratio: float
    signal_qty: float
    sample_count: int


@dataclass
class PendingEdgeObservation:
    due_ts: float
    created_ts: float
    window_start: int
    key: tuple[int, int, int, int]
    elapsed_seconds: float
    start_poly_price: float
    btc_move_pct: float
    qty_ratio: float
    signal_qty: float
    sample_count: int


@dataclass
class PendingBuyOrder:
    order_id: str
    token_id: str
    window_ts: int
    market_slug: str
    price: float
    shares: float
    amount_usd: float
    opening_price: float
    entry_btc_price: float
    created_ts: float
    next_check_ts: float
    balance_before: float = 0.0
    token_balance_before: Optional[float] = None
    negative_windows: int = 0
    cancel_requested: bool = False
    cancel_reason: str = ""
    next_cancel_ts: float = 0.0
    check_attempts: int = 0
    required_confirm_checks: int = 1
    dry_cancel_decided: bool = False
    dry_cancel_blocked: bool = False


@dataclass
class PendingSellOrder:
    order_id: str
    token_id: str
    window_ts: int
    market_slug: str
    price: float
    shares: float
    reason: str
    created_ts: float
    next_check_ts: float
    balance_before: float = 0.0
    token_balance_before: Optional[float] = None
    cancel_requested: bool = False
    cancel_reason: str = ""
    next_cancel_ts: float = 0.0
    check_attempts: int = 0
    required_confirm_checks: int = 1
    last_signal_seq: int = 0
    dry_cancel_decided: bool = False
    dry_cancel_blocked: bool = False


@dataclass
class EdgeModelPrediction:
    key: tuple[int, int, int, int]
    samples: int
    expected_delta: float
    positive_rate: float
    source: str
    nearest_distance: int = 0
    initialized: bool = False


@dataclass
class EdgeModelSignal:
    action: str
    side: str
    reason: str
    bucket: object
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
    push_source: str = "edge_model"
    expected_delta: float = 0.0
    expected_profit_pct: float = 0.0
    rolling_delta: float = 0.0
    rolling_profit_pct: float = 0.0
    model_name: str = ""


@dataclass
class EdgeModelSpec:
    name: str
    path: Path
    time_bucket_seconds: float
    poly_price_bucket: float
    btc_move_bins: int
    qty_ratio_bins: int
    cell_samples: int
    initializer: dict
    cells: dict[tuple[int, int, int, int], deque[float]]
    recent_deltas: deque[float]
    realized_pnl: float = 0.0
    paper_position_shares: float = 0.0
    paper_position_cost: float = 0.0


class PolyBot:
    def __init__(self):
        load_dotenv()

        self.config = LatencyStrategyConfig.from_env()
        self.strategy = LatencySpikeStrategy(self.config)
        self.dry_run = os.getenv("DRY_RUN", "true").lower() == "true"
        self.period_minutes = int(os.getenv("MARKET_PERIOD", "5"))
        self.entry_window_seconds = float(os.getenv("ENTRY_WINDOW_SECONDS", "240"))
        self.dry_run_balance = float(os.getenv("DRY_RUN_BALANCE", "100"))
        self.daily_loss_limit = float(os.getenv("DAILY_LOSS_LIMIT", "30"))
        self.strategy_peak_loss_drop = float(os.getenv("STRATEGY_PEAK_LOSS_DROP", "0"))
        self.strategy_cooling_down_period = float(os.getenv("STRATEGY_COOLING_DOWN_PERIOD", "5400"))
        self.status_interval = float(os.getenv("STATUS_INTERVAL_SECONDS", "10"))
        self.latency_default_ms = float(os.getenv("CHANNEL_LATENCY_DEFAULT_MS", "300"))
        self.latency_max_wait_ms = float(os.getenv("CHANNEL_LATENCY_MAX_WAIT_MS", "1500"))
        self.latency_poly_sync_min_move = float(os.getenv("CHANNEL_LATENCY_POLY_SYNC_MIN_MOVE", "0.005"))
        self.latency_poly_trend_delay_seconds = float(os.getenv("CHANNEL_LATENCY_POLY_TREND_DELAY_MS", "0")) / 1000.0
        self.latency_poly_trend_ticks = max(2, int(float(os.getenv("CHANNEL_LATENCY_POLY_TREND_TICKS", "3"))))
        self.latency_poly_trend_window_ms = float(
            os.getenv("CHANNEL_LATENCY_POLY_TREND_WINDOW_MS", str(self.latency_poly_trend_ticks * 50))
        )
        self.latency_safety_margin_ms = float(os.getenv("CHANNEL_LATENCY_SAFETY_MARGIN_MS", "200"))
        self.latency_samples_limit = int(float(os.getenv("CHANNEL_LATENCY_SAMPLE_LIMIT", "100")))
        self.waveform_sample_ms = float(os.getenv("CHANNEL_WAVEFORM_SAMPLE_MS", "50"))
        self.waveform_match_seconds = float(os.getenv("CHANNEL_WAVEFORM_MATCH_SECONDS", "120"))
        self.waveform_start_offset_seconds = float(os.getenv("CHANNEL_WAVEFORM_START_OFFSET_SECONDS", "10"))
        self.waveform_max_lag_seconds = float(os.getenv("CHANNEL_WAVEFORM_MAX_LAG_SECONDS", "5"))
        self.waveform_lag_step_ms = float(os.getenv("CHANNEL_WAVEFORM_LAG_STEP_MS", "100"))
        self.waveform_min_points = int(float(os.getenv("CHANNEL_WAVEFORM_MIN_POINTS", "1500")))
        self.waveform_max_sample_deviation_ms = float(os.getenv("CHANNEL_WAVEFORM_MAX_SAMPLE_DEVIATION_MS", "250"))
        self.waveform_min_accept_ms = float(os.getenv("CHANNEL_WAVEFORM_MIN_ACCEPT_MS", "250"))
        self.waveform_max_accept_ms = float(os.getenv("CHANNEL_WAVEFORM_MAX_ACCEPT_MS", "1500"))
        self.waveform_min_baseline_samples = int(float(os.getenv("CHANNEL_WAVEFORM_MIN_BASELINE_SAMPLES", "3")))
        self.waveform_max_sample_gap_seconds = float(os.getenv("CHANNEL_WAVEFORM_MAX_SAMPLE_GAP_SECONDS", "30"))
        self.waveform_min_sse_improvement = float(os.getenv("CHANNEL_WAVEFORM_MIN_SSE_IMPROVEMENT", "1.0"))
        self.waveform_min_sse_improvement_pct = float(os.getenv("CHANNEL_WAVEFORM_MIN_SSE_IMPROVEMENT_PCT", "0.001"))
        self.waveform_min_poly_range = float(os.getenv("CHANNEL_WAVEFORM_MIN_POLY_RANGE", "0.50"))
        self.waveform_max_scale = float(os.getenv("CHANNEL_WAVEFORM_MAX_SCALE", "350"))
        self._waveform_fail_reason = ""
        self.edge_model_enabled = os.getenv("EDGE_MODEL_ENABLED", "true").lower() == "true"
        self.edge_model_time_bucket_seconds = float(os.getenv("EDGE_MODEL_TIME_BUCKET_SECONDS", "5"))
        self.edge_model_poly_price_bucket = float(os.getenv("EDGE_MODEL_POLY_PRICE_BUCKET", "0.05"))
        self.edge_model_btc_move_bins = int(float(os.getenv("EDGE_MODEL_BTC_MOVE_BINS", "16")))
        self.edge_model_btc_move_max_pct = float(os.getenv("EDGE_MODEL_BTC_MOVE_MAX_PCT", "0.10"))
        self.edge_model_qty_ratio_bins = int(float(os.getenv("EDGE_MODEL_QTY_RATIO_BINS", "16")))
        self.edge_model_qty_ratio_max = float(os.getenv("EDGE_MODEL_QTY_RATIO_MAX", "5.0"))
        self.edge_model_cell_samples = int(float(os.getenv("EDGE_MODEL_CELL_SAMPLES", "20")))
        self.edge_model_min_samples = int(float(os.getenv("EDGE_MODEL_MIN_SAMPLES", "3")))
        self.edge_model_nearest_cells = int(float(os.getenv("EDGE_MODEL_NEAREST_CELLS", "8")))
        self.edge_model_signal_ms = float(os.getenv("EDGE_MODEL_BINANCE_SIGNAL_MS", "1000"))
        self.edge_model_min_expected_delta = float(os.getenv("EDGE_MODEL_MIN_EXPECTED_POLY_DELTA", "0.02"))
        self.edge_model_min_expected_profit_pct = self.config.min_profit_pct
        self.edge_model_max_pending = int(float(os.getenv("EDGE_MODEL_MAX_PENDING", "5000")))
        self.edge_model_log_interval_seconds = float(os.getenv("EDGE_MODEL_LOG_INTERVAL_SECONDS", "10"))
        self.edge_model_dir = Path(os.getenv("EDGE_MODEL_DIR", "edge_models"))
        self.edge_model_file = os.getenv("EDGE_MODEL_FILE", "").strip()
        self.edge_model_compare_files = [
            item.strip()
            for item in os.getenv("EDGE_MODEL_COMPARE_FILES", "").split(",")
            if item.strip()
        ]
        self.edge_model_save_on_stop = os.getenv("EDGE_MODEL_SAVE_ON_STOP", "true").lower() == "true"
        self.edge_model_use_initialized_cells = os.getenv("EDGE_MODEL_USE_INITIALIZED_CELLS", "true").lower() == "true"
        self.edge_model_profit_window = int(float(os.getenv("EDGE_MODEL_PROFIT_WINDOW", "3")))
        self.simulated_order_failure_rate = float(os.getenv("SIMULATED_ORDER_FAILURE_RATE", "0.10"))
        self.simulated_cancel_success_rate = float(os.getenv("SIMULATED_CANCEL_SUCCESS_RATE", "0.20"))
        self.pending_sell_cancel_btc_move_pct = float(os.getenv("PENDING_SELL_CANCEL_BTC_MOVE_PCT", "0.015"))
        self.trade_cooldown_seconds = float(os.getenv("TRADE_COOLDOWN_SECONDS", "10"))
        self.max_buys_per_window = int(float(os.getenv("MAX_BUYS_PER_WINDOW", "2")))

        self.executor = Executor(
            private_key=os.getenv("PRIVATE_KEY", ""),
            safe_address=os.getenv("SAFE_ADDRESS", ""),
            dry_run=self.dry_run,
            signature_type=int(os.getenv("SIGNATURE_TYPE", "2")),
            funder_address=os.getenv("FUNDER_ADDRESS", os.getenv("SAFE_ADDRESS", "")),
        )
        self.telegram = TelegramNotifier()
        self.price_feed = BinancePriceFeed("BTCUSDT", "BTC")
        self.poly_feed = PolymarketMarketFeed()

        self.market: Optional[MarketWindow] = None
        self.opening_price = 0.0
        self.current_btc_price = 0.0
        self.position: Optional[Position] = None
        self.session_start_balance = 0.0
        self.realized_pnl = 0.0
        self._strategy_pnl_peak = 0.0
        self._strategy_cooldown_until = 0.0
        self._running = False
        self._last_status_ts = 0.0
        self._last_market_probe_ts = 0.0
        self._consecutive_clob_failures = 0
        self._clob_halted = False
        self._pending_entry: Optional[PendingSignal] = None
        self._pending_exit: Optional[PendingSignal] = None
        self._pending_buy: Optional[PendingBuyOrder] = None
        self._pending_sell: Optional[PendingSellOrder] = None
        self._latency_samples_ms: list[float] = []
        self._latency_avg_ms = self.latency_default_ms
        self._last_latency_log_ts = 0.0
        self._btc_price_samples: list[PriceSample] = []
        self._poly_price_samples: list[PriceSample] = []
        self._feed_gap_stats = {
            "btc": FeedGapStats(),
            "poly": FeedGapStats(),
        }
        self._window_latency_matches: list[WindowLatencyMatch] = []
        self._calibrated_window_starts: set[int] = set()
        self._latency_calibrated = False
        self._binance_trade_points: list[BinanceTradePoint] = []
        self._edge_model_cells: dict[tuple[int, int, int, int], deque[float]] = defaultdict(
            lambda: deque(maxlen=max(1, self.edge_model_cell_samples))
        )
        self._edge_model_pending: deque[PendingEdgeObservation] = deque()
        self._edge_model_recent_deltas: deque[float] = deque(maxlen=max(1, self.edge_model_profit_window))
        self._edge_model_last_log_ts = 0.0
        self._edge_model_last_block_bucket = None
        self._edge_model_last_compare_log_ts = 0.0
        self._edge_model_metadata: dict = {}
        self._edge_model_path: Optional[Path] = None
        self._edge_model_specs: list[EdgeModelSpec] = []
        self._edge_model_signal_seq = 0
        self._edge_model_last_evaluated_seq = 0
        self._pending_buy_last_signal_seq = 0
        self._buy_window_ts = 0
        self._buy_count_in_window = 0
        self._edge_model_init = {
            "intercept": 0.0,
            "btc_slope": 0.0,
            "qty_slope": 0.0,
            "time_slope": 0.0,
            "poly_slope": 0.0,
        }
        self._last_trade_action_ts = 0.0
        self._pending_buy_check_interval_seconds = 3.0
        self._last_poly_extrema_received_ts = 0.0
        self._last_calibration_debug_ts = 0.0
        self._calibration_debug_reason = "waiting for completed 5m waveform windows"
        self._edge_model_load()
        atexit.register(self._edge_model_save)

    def start(self) -> None:
        print("=" * 72)
        print("PolyBot - Binance latency spike strategy only")
        print("=" * 72)
        print(f"Mode: {'DRY RUN' if self.dry_run else 'LIVE'}")
        print(f"Bucket: {self.config.bucket_ms}ms")
        print(f"Baseline: {self.config.baseline_hours:.1f}h")
        print(
            f"Entry: push window >= {self.config.entry_push_std_mult:.2f}x "
            f"over {self.config.push_signal_ms}ms vs {self.config.push_std_minutes:.1f}m, "
            f"{self.strategy.entry_volume_summary()}"
        )
        print(
            f"Exit: push window <= -{self.config.exit_push_std_mult:.2f}x "
            f"over {self.config.push_signal_ms}ms vs {self.config.push_std_minutes:.1f}m, "
            f"qty z>={self.config.exit_volume_z:.2f}"
        )
        print(f"Buy cap: ${self.config.max_buy_price:.2f}")
        print(f"Min profit: {self.config.min_profit_pct:.1%}")
        print(f"Profit retreat: {self.config.profit_retreat_pct:.1%}")
        print(
            f"Peak drawdown freeze: ${self.strategy_peak_loss_drop:.2f} / "
            f"{self.strategy_cooling_down_period:.0f}s"
        )
        print(
            f"Channel latency: default {self.latency_default_ms:.0f}ms, "
            f"max wait {self.latency_max_wait_ms:.0f}ms"
        )
        print(
            f"Poly trend sync: delay {self.latency_poly_trend_delay_seconds * 1000:.0f}ms, "
            f"{self.latency_poly_trend_ticks} ticks, "
            f"window {self.latency_poly_trend_window_ms:.0f}ms, "
            f"safety {self.latency_safety_margin_ms:.0f}ms"
        )
        print(
            f"Waveform latency: sample {self.waveform_sample_ms:.0f}ms, "
            f"start +{self.waveform_start_offset_seconds:.0f}s, "
            f"match {self.waveform_match_seconds:.0f}s, "
            f"lag 0-{self.waveform_max_lag_seconds:.0f}s, "
            f"accept {self.waveform_min_accept_ms:.0f}-{self.waveform_max_accept_ms:.0f}ms, "
            f"baseline samples {self.waveform_min_baseline_samples}, "
            f"max gap {self.waveform_max_sample_gap_seconds:.0f}s, "
            f"max sample dev {self.waveform_max_sample_deviation_ms:.0f}ms, "
            f"min SSE edge {self.waveform_min_sse_improvement:.2f}/"
            f"{self.waveform_min_sse_improvement_pct:.3%}, "
            f"min poly range ${self.waveform_min_poly_range:.3f}, "
            f"max scale {self.waveform_max_scale:.0f}x"
        )
        if self.edge_model_enabled:
            print(
                f"Edge model: time {self.edge_model_time_bucket_seconds:.0f}s, "
                f"poly ${self.edge_model_poly_price_bucket:.3f}, "
                f"btc bins {self.edge_model_btc_move_bins} over +/-{self.edge_model_btc_move_max_pct:.3f}%, "
                f"qty bins {self.edge_model_qty_ratio_bins} over 0-{self.edge_model_qty_ratio_max:.1f}x, "
                f"cell FIFO {self.edge_model_cell_samples}, min samples {self.edge_model_min_samples}, "
                f"window {self.edge_model_profit_window}, "
                f"min edge ${self.edge_model_min_expected_delta:.3f}/{self.edge_model_min_expected_profit_pct:.1%}, "
                f"sim fail {self.simulated_order_failure_rate:.0%}, "
                f"sim cancel {self.simulated_cancel_success_rate:.0%}, "
                f"sell cancel BTC +{self.pending_sell_cancel_btc_move_pct:.4f}%, "
                f"cooldown {self.trade_cooldown_seconds:.0f}s"
            )
        print(f"Trade amount: ${self.config.trade_amount:.2f}")
        print(f"Entry window: first {self.entry_window_seconds:.0f}s of each market")

        executor_ready = self.executor.initialize()
        if not executor_ready and not self.dry_run:
            raise RuntimeError("Executor initialization failed")
        if not executor_ready:
            print("[executor] DRY RUN continuing without CLOB auth initialization")

        self.session_start_balance = self.dry_run_balance if self.dry_run else self.executor.get_balance(refresh=True)
        print(f"Starting balance: ${self.session_start_balance:.2f}")

        self.price_feed.start(on_price=self._on_binance_trade)
        self.poly_feed.start()
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        self.telegram.send(
            "*PolyBot Started*\n"
            "Strategy: Binance 50ms latency spike\n"
            f"Mode: {'DRY RUN' if self.dry_run else 'LIVE'}\n"
            f"Buy cap: ${self.config.max_buy_price:.2f}\n"
            f"Min profit: {self.config.min_profit_pct:.1%}\n"
            f"Profit retreat: {self.config.profit_retreat_pct:.1%}\n"
            f"Peak drawdown freeze: ${self.strategy_peak_loss_drop:.2f}"
        )

        self._running = True
        while self._running:
            self._tick()
            time.sleep(0.05)

    def _on_binance_trade(
        self,
        price: float,
        source: str = "ws",
        event_ts: float = None,
        received_ts: float = None,
        raw: dict = None,
        **_,
    ) -> None:
        self.current_btc_price = float(price or 0.0)
        raw = raw or {}
        qty = self._as_float(raw.get("q"))
        self.strategy.ingest_trade(self.current_btc_price, qty=qty, event_ts=event_ts)
        self._update_price_sample("btc", self.current_btc_price, received_ts or time.time())
        self._edge_model_on_binance_trade(self.current_btc_price, qty, received_ts or time.time())
        self._edge_model_signal_seq += 1

    def _tick(self) -> None:
        if self.current_btc_price <= 0:
            return

        self._ensure_market()
        if not self.market:
            return
        self._update_extrema_latency()
        self._edge_model_process_due()
        self._process_pending_buy()
        self._process_pending_sell()
        if self._pending_sell:
            self._print_status()
            return

        self._check_daily_loss_limit()
        if self.position and not self.position.closed:
            if self.market.window_start != self.position.window_ts:
                self._resolve_position(self.current_btc_price)
            else:
                self._try_exit()
                if self.position and not self.position.closed:
                    self._try_entry()
        elif not self._clob_halted and not self._strategy_is_cooling_down():
            self._try_entry()
        else:
            self._pending_entry = None

        self._print_status()

    def _ensure_market(self) -> None:
        now = time.time()
        if self.market and self.market.window_start <= now < self.market.window_end:
            return
        if now - self._last_market_probe_ts < 1.0:
            return

        self._last_market_probe_ts = now
        previous = self.market
        market = get_current_market(self.period_minutes, asset="btc")
        if not market:
            if previous and now >= previous.window_end and previous.window_start not in self._calibrated_window_starts:
                self._calibrate_completed_window(previous)
                print("[market] current market unavailable after boundary; calibrated previous window only")
            return

        if previous and previous.window_start != market.window_start:
            self._calibrate_completed_window(previous)

        self.market = market
        self.opening_price = self.current_btc_price
        self._pending_entry = None
        self._pending_exit = None
        if self._buy_window_ts != market.window_start:
            self._buy_window_ts = market.window_start
            self._buy_count_in_window = 0
        self.poly_feed.subscribe(
            [market.token_id_up, market.token_id_down],
            {market.token_id_up: "UP", market.token_id_down: "DOWN"},
        )
        print(
            f"\n[new window] {market.slug} | open BTC ${self.opening_price:,.2f} | "
            f"UP {market.token_id_up[:10]}..."
        )

        if previous and previous.window_start != market.window_start:
            self._probe_clob_recovery()

    def _try_entry(self) -> None:
        if self._pending_buy:
            return
        if self._buy_limit_reached():
            return
        signal_result = self._edge_model_entry_signal()
        if not signal_result:
            return
        if not self._entry_window_open():
            return
        if not self.strategy.baseline_ready():
            return
        if not self._clob_ok_for_trade():
            return
        if not self._trade_cooldown_ready("buy"):
            return
        self._execute_entry(signal_result)

    def _buy_limit_reached(self) -> bool:
        if not self.market or self.max_buys_per_window <= 0:
            return False
        if self._buy_window_ts != self.market.window_start:
            self._buy_window_ts = self.market.window_start
            self._buy_count_in_window = 0
        return self._buy_count_in_window >= self.max_buys_per_window

    def _execute_entry(self, signal_result: object) -> None:
        if not self.market:
            return

        token_id = self.market.token_id_up
        buy_price = self._live_buy_price(token_id)
        if buy_price <= 0:
            print("  Entry skipped: no executable UP buy price")
            return
        if buy_price > self.config.max_buy_price:
            print(f"  Entry skipped: UP ${buy_price:.3f} > cap ${self.config.max_buy_price:.2f}")
            return

        balance = self.executor.get_balance(refresh=False)
        amount = round(self.config.trade_amount if self.dry_run else min(self.config.trade_amount, max(0.0, balance)), 2)
        if not self.dry_run and amount < 5.0:
            print(f"  Entry skipped: balance ${balance:.2f} below $5 minimum")
            return
        if self._simulate_order_failure("BUY"):
            return

        existing = self.position and not self.position.closed
        print(
            f"\n[{'ADD' if existing else 'ENTRY'}] {signal_result.reason}: BTC "
            f"{signal_result.move_pct:+.4f}% qty {signal_result.bucket.qty:.4f} "
            f"(push {self._format_push_signal(signal_result)}, qty z {signal_result.volume_z:.2f}) | "
            f"UP ask ${buy_price:.3f} | model {getattr(signal_result, 'model_name', '')} | "
            f"expected {getattr(signal_result, 'rolling_delta', 0.0):+.3f}/"
            f"{getattr(signal_result, 'rolling_profit_pct', 0.0):.1%}"
        )
        result = self.executor.place_buy_order(token_id, amount, price=buy_price)
        if not result.success:
            print(f"  Buy submit failed: {result.error}")
            return

        now = time.time()
        required_confirm_checks = random.choices([2, 3, 4], weights=[25, 50, 25], k=1)[0] if self.dry_run else 1
        self._pending_buy = PendingBuyOrder(
            order_id=result.order_id,
            token_id=token_id,
            window_ts=self.market.window_start,
            market_slug=self.market.slug,
            price=result.price or buy_price,
            shares=result.shares,
            amount_usd=result.amount_usd or amount,
            opening_price=self.opening_price,
            entry_btc_price=self.current_btc_price,
            created_ts=now,
            next_check_ts=now + self._pending_buy_check_interval_seconds,
            balance_before=result.balance_before,
            token_balance_before=result.token_balance_before,
            required_confirm_checks=required_confirm_checks,
        )
        self._pending_buy_last_signal_seq = self._edge_model_signal_seq
        self._last_trade_action_ts = now
        self._buy_count_in_window += 1
        print(
            f"  Buy order pending: {result.order_id} | "
            f"{result.shares:.0f} shares @ ${result.price:.3f}; "
            f"check every {self._pending_buy_check_interval_seconds:.0f}s | "
            f"confirm after {required_confirm_checks} checks | "
            f"window buys {self._buy_count_in_window}/{self.max_buys_per_window}"
        )

    def _apply_buy_fill(self, result, pending: PendingBuyOrder) -> None:
        existing = self.position and not self.position.closed
        if existing and self.position:
            total_cost = self.position.cost + result.amount_usd
            total_shares = self.position.shares + result.shares
            self.position.cost = total_cost
            self.position.shares = total_shares
            self.position.entry_price = total_cost / total_shares if total_shares > 0 else result.price
            self.position.entry_ts = time.time()
            self.position.peak_unrealized_profit = 0.0
            self.position.peak_sell_price = 0.0
            self.position.last_sell_price = 0.0
            print(
                f"  Added position: +{result.shares:.0f} shares ${result.amount_usd:.2f}; "
                f"total {self.position.shares:.0f} @ avg ${self.position.entry_price:.3f}"
            )
        else:
            self.position = Position(
                side="UP",
                token_id=pending.token_id,
                window_ts=pending.window_ts,
                market_slug=pending.market_slug,
                entry_price=result.price,
                shares=result.shares,
                cost=result.amount_usd,
                entry_btc_price=pending.entry_btc_price,
                opening_price=pending.opening_price,
                entry_ts=time.time(),
            )
        self._last_trade_action_ts = time.time()
        print(
            f"  Buy filled: {result.shares:.0f} shares @ ${result.price:.3f}, "
            f"cost ${result.amount_usd:.2f}, status {result.status}"
        )
        self.telegram.strategy_trade_alert(
            strategy="Binance latency spike",
            side="UP",
            price=result.price,
            amount=result.amount_usd,
            market_slug=pending.market_slug,
            dry_run=self.dry_run,
            strategy_pnl=self.realized_pnl,
            total_pnl=self.realized_pnl,
        )

    def _process_pending_buy(self) -> None:
        pending = self._pending_buy
        if not pending:
            return
        if not self.market or pending.window_ts != self.market.window_start:
            print(f"  Pending buy {pending.order_id} window changed; attempting cancel")
            pending.cancel_requested = True
            pending.cancel_reason = "market window changed"
            self._cancel_pending_buy(pending.cancel_reason)
            return

        now = time.time()
        if pending.cancel_requested:
            if now >= pending.next_cancel_ts:
                self._cancel_pending_buy(pending.cancel_reason or "cancel requested")
            return

        if self._pending_buy_cancel_signal(pending):
            self._cancel_pending_buy(pending.cancel_reason or "negative follow-through")
            return

        if now < pending.next_check_ts:
            return

        pending.check_attempts += 1
        result = self.executor.check_pending_buy(
            pending.order_id,
            pending.price,
            pending.shares,
            pending.token_id,
            pending.balance_before,
            pending.token_balance_before,
        )
        pending.next_check_ts = now + self._pending_buy_check_interval_seconds
        if not result:
            print(f"  Pending buy check: {pending.order_id} not filled yet")
            return
        if self.dry_run and pending.check_attempts < pending.required_confirm_checks:
            print(
                f"  Pending buy check {pending.check_attempts}/"
                f"{pending.required_confirm_checks}: simulated order still pending"
            )
            return

        self._pending_buy = None
        self._apply_buy_fill(result, pending)

    def _pending_buy_cancel_signal(self, pending: PendingBuyOrder) -> bool:
        if self.dry_run and pending.dry_cancel_blocked:
            return False
        if self._edge_model_signal_seq > self._pending_buy_last_signal_seq:
            self._pending_buy_last_signal_seq = self._edge_model_signal_seq
            features = self._edge_model_features(time.time(), self.current_btc_price)
            if features and features.btc_move_pct < 0:
                pending.negative_windows += 1
            elif features:
                pending.negative_windows = 0
            if pending.negative_windows >= 2:
                pending.cancel_reason = (
                    f"two negative Binance windows ({features.btc_move_pct:+.4f}%)"
                    if features else "two negative Binance windows"
                )
                return True

        if self._poly_recently_falling(pending.created_ts, self.latency_poly_trend_ticks):
            pending.cancel_reason = f"Polymarket falling over {self.latency_poly_trend_ticks} ticks"
            return True
        return False

    def _cancel_pending_buy(self, reason: str) -> None:
        pending = self._pending_buy
        if not pending:
            return
        if self.dry_run:
            if not pending.dry_cancel_decided:
                pending.dry_cancel_decided = True
                cancel_rate = min(max(0.0, self.simulated_cancel_success_rate), 1.0)
                pending.dry_cancel_blocked = random.random() >= cancel_rate
            if not pending.dry_cancel_blocked:
                print(
                    f"  DRY pending buy cancelled {pending.order_id}: {reason} "
                    f"(sim cancel success {self.simulated_cancel_success_rate:.0%})"
                )
                self._pending_buy = None
                self._last_trade_action_ts = time.time()
                return
            print(
                f"  DRY pending buy cancel failed/already matched {pending.order_id}: {reason} "
                f"(sim cancel success {self.simulated_cancel_success_rate:.0%}); keeping order pending"
            )
            pending.cancel_requested = False
            pending.next_cancel_ts = 0.0
            return
        fill = self.executor.check_pending_buy(
            pending.order_id,
            pending.price,
            pending.shares,
            pending.token_id,
            pending.balance_before,
            pending.token_balance_before,
        )
        if fill:
            print(f"  Pending buy already filled before cancel: {pending.order_id}")
            self._pending_buy = None
            self._apply_buy_fill(fill, pending)
            return
        if self.executor.order_is_cancelled(pending.order_id):
            print(f"  Pending buy cancel confirmed: {pending.order_id}")
            self._pending_buy = None
            self._last_trade_action_ts = time.time()
            return

        print(f"  Cancelling pending buy {pending.order_id}: {reason}")
        pending.cancel_requested = True
        pending.cancel_reason = reason
        if self.executor.cancel_order(pending.order_id):
            now = time.time()
            pending.next_cancel_ts = now + 1.0
            pending.next_check_ts = min(pending.next_check_ts, now + 1.0)
            self._last_trade_action_ts = now
            print("  Pending buy cancel sent; retrying every 1s until order state is confirmed")
            return

        now = time.time()
        pending.next_cancel_ts = now + 1.0
        pending.next_check_ts = min(pending.next_check_ts, now + 1.0)
        print("  Pending buy cancel failed; keeping order under observation")

    def _edge_model_on_binance_trade(self, price: float, qty: float, ts: float) -> None:
        if not self.edge_model_enabled or price <= 0 or ts <= 0:
            return
        segment = self.market.window_start if self.market else 0
        self._binance_trade_points.append(BinanceTradePoint(ts=ts, price=price, qty=max(0.0, qty), segment=segment))
        keep_after = ts - 1800.0
        while self._binance_trade_points and self._binance_trade_points[0].ts < keep_after:
            self._binance_trade_points.pop(0)
        if not self.market:
            return
        features = self._edge_model_features(ts, price)
        if not features:
            return
        latency_ms = self._latency_avg_ms if self._latency_samples_ms else self.latency_default_ms
        latency_ms = min(max(50.0, latency_ms), self.latency_max_wait_ms)
        self._edge_model_pending.append(PendingEdgeObservation(
            due_ts=ts + latency_ms / 1000.0,
            created_ts=ts,
            window_start=self.market.window_start,
            key=features.key,
            elapsed_seconds=features.elapsed_seconds,
            start_poly_price=features.poly_price,
            btc_move_pct=features.btc_move_pct,
            qty_ratio=features.qty_ratio,
            signal_qty=features.signal_qty,
            sample_count=features.sample_count,
        ))
        while len(self._edge_model_pending) > max(1, self.edge_model_max_pending):
            self._edge_model_pending.popleft()

    def _edge_model_process_due(self) -> None:
        if not self.edge_model_enabled or not self.market:
            return
        now = time.time()
        processed = 0
        remaining: deque[PendingEdgeObservation] = deque()
        while self._edge_model_pending:
            observation = self._edge_model_pending.popleft()
            if observation.due_ts > now:
                remaining.append(observation)
                continue
            if observation.window_start != self.market.window_start:
                continue
            poly_price, _ = self._poly_reference_snapshot()
            if poly_price <= 0:
                continue
            delta = poly_price - observation.start_poly_price
            self._edge_model_cells[observation.key].append(delta)
            for spec in self._edge_model_specs:
                spec_features = EdgeModelFeatures(
                    key=observation.key,
                    elapsed_seconds=observation.elapsed_seconds,
                    poly_price=observation.start_poly_price,
                    btc_move_pct=observation.btc_move_pct,
                    qty_ratio=observation.qty_ratio,
                    signal_qty=observation.signal_qty,
                    sample_count=observation.sample_count,
                )
                spec_key = self._edge_model_key_for_spec(spec_features, spec)
                spec.cells.setdefault(spec_key, deque(maxlen=max(1, spec.cell_samples))).append(delta)
            processed += 1
        self._edge_model_pending = remaining
        if processed > 0 and now - self._edge_model_last_log_ts >= self.edge_model_log_interval_seconds:
            self._edge_model_last_log_ts = now
            cell_count = len(self._edge_model_cells)
            sample_count = sum(len(values) for values in self._edge_model_cells.values())
            print(
                f"[edge-model] updated {processed} observations | "
                f"cells {cell_count}, samples {sample_count}, latency {self._latency_avg_ms:.0f}ms"
            )

    def _edge_model_entry_signal(self) -> Optional[EdgeModelSignal]:
        if not self.edge_model_enabled or not self.market:
            return None
        if self._edge_model_signal_seq <= self._edge_model_last_evaluated_seq:
            return None
        self._edge_model_last_evaluated_seq = self._edge_model_signal_seq
        if not self._entry_window_open() or not self.strategy.baseline_ready():
            return None
        features = self._edge_model_features(time.time(), self.current_btc_price)
        if not features:
            return None
        prediction = self._edge_model_prediction(features.key)
        if not prediction or prediction.samples < self.edge_model_min_samples:
            self._edge_model_log_block(features, prediction, "warming")
            return None
        buy_price = self._live_buy_price(self.market.token_id_up)
        if buy_price <= 0:
            buy_price = features.poly_price
        signed_delta = prediction.expected_delta
        self._edge_model_recent_deltas.append(signed_delta)
        rolling_delta = sum(self._edge_model_recent_deltas)
        expected_exit = min(0.99, features.poly_price + rolling_delta)
        expected_profit_pct = 0.0 if buy_price <= 0 else (expected_exit - buy_price) / buy_price
        self._edge_model_log_compare(features, buy_price)
        if rolling_delta <= 0 and self.position and not self.position.closed:
            print(
                f"  Edge model cancel/hold: rolling delta {rolling_delta:+.3f}; "
                "no cancellable open order tracked, holding position"
            )
            return None
        if (
            features.btc_move_pct <= 0
            or rolling_delta < self.edge_model_min_expected_delta
            or expected_profit_pct < self.edge_model_min_expected_profit_pct
        ):
            self._edge_model_log_block(features, prediction, "edge_low", buy_price, expected_profit_pct)
            return None

        print(
            "  Edge model entry candidate: "
            f"poly ${features.poly_price:.3f}, buy ${buy_price:.3f}, "
            f"expected {prediction.expected_delta:+.3f}, rolling {rolling_delta:+.3f}, "
            f"profit {expected_profit_pct:.1%}, "
            f"p+ {prediction.positive_rate:.0%}, samples {prediction.samples} ({prediction.source}), "
            f"btc {features.btc_move_pct:+.4f}%, qty {features.qty_ratio:.2f}x, key {features.key}"
        )
        return EdgeModelSignal(
            action="BUY",
            side="UP",
            reason="edge_model_latency_distribution",
            bucket=SimpleNamespace(qty=features.signal_qty),
            move_pct=features.btc_move_pct,
            price_z=prediction.expected_delta,
            volume_z=features.qty_ratio,
            push_deviation=rolling_delta,
            push_delta=self.current_btc_price * features.btc_move_pct / 100.0,
            push_count=prediction.samples,
            push_current_count=features.sample_count,
            push_current_mean=self.current_btc_price,
            push_mean_shift_z=features.btc_move_pct,
            push_std_ratio=features.qty_ratio,
            push_source=f"edge/{prediction.source}",
            expected_delta=prediction.expected_delta,
            expected_profit_pct=expected_profit_pct,
            rolling_delta=rolling_delta,
            rolling_profit_pct=expected_profit_pct,
            model_name=self._edge_model_path.name if self._edge_model_path else "edge-model",
        )

    def _edge_model_features(self, ts: float, price: float) -> Optional[EdgeModelFeatures]:
        if not self.market or price <= 0:
            return None
        signal_seconds = max(0.05, self.edge_model_signal_ms / 1000.0)
        signal_start = ts - signal_seconds
        previous_start = signal_start - signal_seconds
        segment = self.market.window_start
        window_points = [
            point for point in self._binance_trade_points
            if point.segment == segment and point.ts <= ts and point.price > 0
        ]
        signal_points = [point for point in window_points if point.ts >= signal_start]
        previous_points = [
            point for point in window_points
            if previous_start <= point.ts < signal_start
        ]
        if len(signal_points) < 1 or len(previous_points) < 1:
            return None
        poly_price = self._poly_price_near(signal_start, ts)
        if poly_price <= 0:
            return None
        previous_avg = sum(point.price for point in previous_points) / len(previous_points)
        current_avg = sum(point.price for point in signal_points) / len(signal_points)
        if previous_avg <= 0:
            return None
        btc_move_pct = (current_avg - previous_avg) / previous_avg * 100.0
        signal_qty = sum(point.qty for point in signal_points)
        elapsed = max(0.0, signal_start - segment)
        previous_qty = sum(point.qty for point in previous_points)
        qty_ratio = signal_qty / previous_qty if previous_qty > 0 else 0.0
        key = (
            self._edge_time_bucket(elapsed),
            self._edge_poly_bucket(poly_price),
            self._edge_btc_move_bucket(btc_move_pct),
            self._edge_qty_ratio_bucket(qty_ratio),
        )
        return EdgeModelFeatures(
            key=key,
            elapsed_seconds=elapsed,
            poly_price=poly_price,
            btc_move_pct=btc_move_pct,
            qty_ratio=qty_ratio,
            signal_qty=signal_qty,
            sample_count=len(signal_points),
        )

    def _edge_model_prediction(self, key: tuple[int, int, int, int]) -> Optional[EdgeModelPrediction]:
        exact = list(self._edge_model_cells.get(key, []))
        if len(exact) >= self.edge_model_min_samples:
            return self._edge_prediction_from_values(key, exact, "exact", 0)
        neighbors: list[tuple[int, list[float]]] = []
        for other_key, values in self._edge_model_cells.items():
            if not values:
                continue
            distance = sum(abs(a - b) for a, b in zip(key, other_key))
            neighbors.append((distance, list(values)))
        if not neighbors:
            return None
        neighbors.sort(key=lambda item: item[0])
        merged: list[float] = []
        max_distance = 0
        for distance, values in neighbors[: max(1, self.edge_model_nearest_cells)]:
            merged.extend(values)
            max_distance = max(max_distance, distance)
            if len(merged) >= self.edge_model_min_samples:
                break
        if not merged:
            return self._edge_model_initialized_prediction(key)
        if len(merged) < self.edge_model_min_samples:
            initialized = self._edge_model_initialized_prediction(key)
            if initialized:
                return initialized
        return self._edge_prediction_from_values(key, merged, "nearest", max_distance)

    def _edge_model_initialized_prediction(self, key: tuple[int, int, int, int]) -> Optional[EdgeModelPrediction]:
        if not self.edge_model_use_initialized_cells:
            return None
        value = self._edge_model_initialized_delta(key)
        samples = max(1, self.edge_model_min_samples)
        positive_rate = 1.0 if value > 0 else 0.0
        return EdgeModelPrediction(
            key=key,
            samples=samples,
            expected_delta=value,
            positive_rate=positive_rate,
            source="initialized",
            nearest_distance=0,
            initialized=True,
        )

    def _edge_model_initialized_delta(self, key: tuple[int, int, int, int]) -> float:
        time_idx, poly_idx, btc_idx, qty_idx = key
        time_buckets = max(1, int(math.ceil(self.period_minutes * 60.0 / max(1.0, self.edge_model_time_bucket_seconds))))
        poly_buckets = max(1, int(math.ceil(1.0 / max(0.001, self.edge_model_poly_price_bucket))))
        btc_mid = (max(1, self.edge_model_btc_move_bins) - 1) / 2.0
        qty_mid = (max(1, self.edge_model_qty_ratio_bins) - 1) / 2.0
        time_norm = 0.0 if time_buckets <= 1 else time_idx / (time_buckets - 1)
        poly_norm = 0.0 if poly_buckets <= 1 else poly_idx / (poly_buckets - 1)
        btc_norm = 0.0 if btc_mid <= 0 else (btc_idx - btc_mid) / btc_mid
        qty_norm = 0.0 if qty_mid <= 0 else (qty_idx - qty_mid) / qty_mid
        value = (
            float(self._edge_model_init.get("intercept", 0.0))
            + float(self._edge_model_init.get("btc_slope", 0.0)) * btc_norm
            + float(self._edge_model_init.get("qty_slope", 0.0)) * qty_norm
            + float(self._edge_model_init.get("time_slope", 0.0)) * time_norm
            + float(self._edge_model_init.get("poly_slope", 0.0)) * poly_norm
        )
        return max(-0.99, min(0.99, value))

    def _edge_model_log_compare(self, features: EdgeModelFeatures, buy_price: float) -> None:
        if not self._edge_model_specs:
            return
        now = time.time()
        if now - self._edge_model_last_compare_log_ts < self.edge_model_log_interval_seconds:
            return
        self._edge_model_last_compare_log_ts = now
        parts = []
        for spec in self._edge_model_specs:
            key = self._edge_model_key_for_spec(features, spec)
            pred = self._edge_model_prediction_for_spec(key, spec)
            spec.recent_deltas.append(pred.expected_delta)
            rolling = sum(spec.recent_deltas)
            expected_exit = min(0.99, features.poly_price + rolling)
            profit_pct = 0.0 if buy_price <= 0 else (expected_exit - buy_price) / buy_price
            parts.append(
                f"{spec.name}: d {pred.expected_delta:+.3f}/roll {rolling:+.3f}/"
                f"{profit_pct:+.1%}/pnl ${self.config.trade_amount * profit_pct:+.2f} "
                f"n={pred.samples} {pred.source}"
            )
        print("[edge-compare] " + " | ".join(parts))

    def _edge_model_key_for_spec(
        self,
        features: EdgeModelFeatures,
        spec: EdgeModelSpec,
    ) -> tuple[int, int, int, int]:
        return (
            self._edge_time_bucket_for(features.elapsed_seconds, spec.time_bucket_seconds),
            self._edge_poly_bucket_for(features.poly_price, spec.poly_price_bucket),
            self._edge_btc_move_bucket_for(features.btc_move_pct, spec.btc_move_bins),
            self._edge_qty_ratio_bucket_for(features.qty_ratio, spec.qty_ratio_bins),
        )

    def _edge_model_prediction_for_spec(
        self,
        key: tuple[int, int, int, int],
        spec: EdgeModelSpec,
    ) -> EdgeModelPrediction:
        values = list(spec.cells.get(key, []))
        if values:
            return self._edge_prediction_from_values(key, values, "exact", 0)
        value = self._edge_model_initialized_delta_for_spec(key, spec)
        return EdgeModelPrediction(
            key=key,
            samples=max(1, self.edge_model_min_samples),
            expected_delta=value,
            positive_rate=1.0 if value > 0 else 0.0,
            source="initialized",
            initialized=True,
        )

    def _edge_model_initialized_delta_for_spec(
        self,
        key: tuple[int, int, int, int],
        spec: EdgeModelSpec,
    ) -> float:
        time_idx, poly_idx, btc_idx, qty_idx = key
        time_buckets = max(1, int(math.ceil(self.period_minutes * 60.0 / max(1.0, spec.time_bucket_seconds))))
        poly_buckets = max(1, int(math.ceil(1.0 / max(0.001, spec.poly_price_bucket))))
        btc_mid = (max(1, spec.btc_move_bins) - 1) / 2.0
        qty_mid = (max(1, spec.qty_ratio_bins) - 1) / 2.0
        time_norm = 0.0 if time_buckets <= 1 else time_idx / (time_buckets - 1)
        poly_norm = 0.0 if poly_buckets <= 1 else poly_idx / (poly_buckets - 1)
        btc_norm = 0.0 if btc_mid <= 0 else (btc_idx - btc_mid) / btc_mid
        qty_norm = 0.0 if qty_mid <= 0 else (qty_idx - qty_mid) / qty_mid
        init = spec.initializer
        value = (
            float(init.get("intercept", 0.0))
            + float(init.get("btc_slope", 0.0)) * btc_norm
            + float(init.get("qty_slope", 0.0)) * qty_norm
            + float(init.get("time_slope", 0.0)) * time_norm
            + float(init.get("poly_slope", 0.0)) * poly_norm
        )
        return max(-0.99, min(0.99, value))

    def _edge_prediction_from_values(
        self,
        key: tuple[int, int, int, int],
        values: list[float],
        source: str,
        nearest_distance: int,
    ) -> EdgeModelPrediction:
        samples = len(values)
        expected_delta = sum(values) / samples
        positive_rate = sum(1 for value in values if value > 0) / samples
        return EdgeModelPrediction(
            key=key,
            samples=samples,
            expected_delta=expected_delta,
            positive_rate=positive_rate,
            source=source,
            nearest_distance=nearest_distance,
        )

    def _edge_model_load(self) -> None:
        if not self.edge_model_enabled:
            return
        path = self._edge_model_resolve_path()
        if not path or not path.exists():
            self._edge_model_path = path
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[edge-model] failed to load {path}: {exc}")
            self._edge_model_path = path
            return
        self._edge_model_path = path
        self._edge_model_metadata = payload.get("metadata", {})
        self._edge_model_init.update(payload.get("initializer", {}) or {})
        cells = payload.get("cells", {}) or {}
        for key_text, values in cells.items():
            key = self._edge_model_parse_key(key_text)
            if not key:
                continue
            queue = deque(maxlen=max(1, self.edge_model_cell_samples))
            for value in values[-self.edge_model_cell_samples:]:
                try:
                    queue.append(float(value))
                except Exception:
                    pass
            if queue:
                self._edge_model_cells[key] = queue
        print(
            f"[edge-model] loaded {path.name}: cells {len(self._edge_model_cells)}, "
            f"initializer {self._edge_model_init}"
        )
        self._edge_model_load_compare_specs(path)

    def _edge_model_load_compare_specs(self, active_path: Path) -> None:
        paths: list[Path] = []
        raw_files = self.edge_model_compare_files or [
            "edge_t5s_p5c_btc20_qty20_fifo20.json",
            "edge_t4s_p3c_btc20_qty20_fifo10.json",
            "edge_t3s_p2c_btc20_qty20_fifo10.json",
        ]
        for raw in raw_files:
            path = Path(raw)
            if not path.is_absolute():
                path = Path.cwd() / self.edge_model_dir / path
            if path.exists() and path not in paths:
                paths.append(path)
        if active_path.exists() and active_path not in paths:
            paths.insert(0, active_path)
        self._edge_model_specs = []
        for path in paths:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                meta = payload.get("metadata", {}) or {}
                cells: dict[tuple[int, int, int, int], deque[float]] = {}
                fifo = int(meta.get("cell_samples", self.edge_model_cell_samples))
                for key_text, values in (payload.get("cells", {}) or {}).items():
                    key = self._edge_model_parse_key(key_text)
                    if key:
                        cells[key] = deque((float(value) for value in values[-fifo:]), maxlen=max(1, fifo))
                self._edge_model_specs.append(EdgeModelSpec(
                    name=path.stem,
                    path=path,
                    time_bucket_seconds=float(meta.get("time_bucket_seconds", self.edge_model_time_bucket_seconds)),
                    poly_price_bucket=float(meta.get("poly_price_bucket", self.edge_model_poly_price_bucket)),
                    btc_move_bins=int(meta.get("btc_move_bins", self.edge_model_btc_move_bins)),
                    qty_ratio_bins=int(meta.get("qty_ratio_bins", self.edge_model_qty_ratio_bins)),
                    cell_samples=fifo,
                    initializer=payload.get("initializer", {}) or {},
                    cells=cells,
                    recent_deltas=deque(maxlen=max(1, self.edge_model_profit_window)),
                ))
            except Exception as exc:
                print(f"[edge-model] failed to load compare model {path}: {exc}")
        if self._edge_model_specs:
            print("[edge-model] compare models: " + ", ".join(spec.name for spec in self._edge_model_specs))

    def _edge_model_save(self) -> None:
        if not self.edge_model_enabled or not self.edge_model_save_on_stop:
            return
        path = self._edge_model_path or self._edge_model_resolve_path()
        if not path:
            return
        try:
            metadata = dict(self._edge_model_metadata or {})
            metadata.update({
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "time_bucket_seconds": self.edge_model_time_bucket_seconds,
                "poly_price_bucket": self.edge_model_poly_price_bucket,
                "btc_move_bins": self.edge_model_btc_move_bins,
                "qty_ratio_bins": self.edge_model_qty_ratio_bins,
                "cell_samples": self.edge_model_cell_samples,
            })
            self._edge_model_write_payload(path, metadata, self._edge_model_init, self._edge_model_cells)
            saved = {path.resolve()}
            for spec in self._edge_model_specs:
                spec_path = spec.path.resolve()
                if spec_path in saved:
                    continue
                spec_metadata = self._edge_model_existing_metadata(spec.path)
                spec_metadata.update({
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                    "time_bucket_seconds": spec.time_bucket_seconds,
                    "poly_price_bucket": spec.poly_price_bucket,
                    "btc_move_bins": spec.btc_move_bins,
                    "qty_ratio_bins": spec.qty_ratio_bins,
                    "cell_samples": spec.cell_samples,
                })
                self._edge_model_write_payload(spec.path, spec_metadata, spec.initializer, spec.cells)
                saved.add(spec_path)
        except Exception as exc:
            print(f"[edge-model] failed to save: {exc}")

    def _edge_model_existing_metadata(self, path: Path) -> dict:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return dict(payload.get("metadata", {}) or {})
        except Exception:
            return {}

    def _edge_model_write_payload(
        self,
        path: Path,
        metadata: dict,
        initializer: dict,
        cells: dict[tuple[int, int, int, int], deque[float]],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": metadata,
            "initializer": initializer,
            "cells": {
                self._edge_model_key_text(key): list(values)
                for key, values in cells.items()
                if values
            },
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[edge-model] saved {path} ({len(payload['cells'])} cells)")

    def _edge_model_resolve_path(self) -> Optional[Path]:
        if self.edge_model_file:
            path = Path(self.edge_model_file)
            if not path.is_absolute():
                if len(path.parts) == 1:
                    path = Path.cwd() / self.edge_model_dir / path
                else:
                    path = Path.cwd() / path
            return path
        name = (
            f"edge_t{int(self.edge_model_time_bucket_seconds)}s_"
            f"p{int(round(self.edge_model_poly_price_bucket * 100))}c_"
            f"btc{self.edge_model_btc_move_bins}_qty{self.edge_model_qty_ratio_bins}_"
            f"fifo{self.edge_model_cell_samples}.json"
        )
        return Path.cwd() / self.edge_model_dir / name

    def _edge_model_key_text(self, key: tuple[int, int, int, int]) -> str:
        return ",".join(str(int(value)) for value in key)

    def _edge_model_parse_key(self, key_text: str) -> Optional[tuple[int, int, int, int]]:
        try:
            parts = [int(part) for part in str(key_text).split(",")]
        except Exception:
            return None
        if len(parts) != 4:
            return None
        return tuple(parts)  # type: ignore[return-value]

    def _edge_model_log_block(
        self,
        features: EdgeModelFeatures,
        prediction: Optional[EdgeModelPrediction],
        reason: str,
        buy_price: float = 0.0,
        expected_profit_pct: float = 0.0,
    ) -> None:
        bucket_id = (features.key, int(time.time() // self.edge_model_log_interval_seconds), reason)
        if self._edge_model_last_block_bucket == bucket_id:
            return
        self._edge_model_last_block_bucket = bucket_id
        if prediction:
            print(
                f"  Edge model held ({reason}): key {features.key}, "
                f"btc {features.btc_move_pct:+.4f}%, qty {features.qty_ratio:.2f}x, "
                f"expected {prediction.expected_delta:+.3f}, profit {expected_profit_pct:.1%}, "
                f"samples {prediction.samples} ({prediction.source})"
            )
        else:
            print(
                f"  Edge model held ({reason}): key {features.key}, "
                f"btc {features.btc_move_pct:+.4f}%, qty {features.qty_ratio:.2f}x, "
                f"no comparable samples yet"
            )

    def _edge_time_bucket(self, elapsed_seconds: float) -> int:
        return self._edge_time_bucket_for(elapsed_seconds, self.edge_model_time_bucket_seconds)

    def _edge_time_bucket_for(self, elapsed_seconds: float, bucket_seconds: float) -> int:
        bucket_seconds = max(1.0, bucket_seconds)
        max_bucket = max(0, int(math.ceil(self.period_minutes * 60.0 / bucket_seconds)) - 1)
        return min(max(0, int(elapsed_seconds // bucket_seconds)), max_bucket)

    def _edge_poly_bucket(self, poly_price: float) -> int:
        return self._edge_poly_bucket_for(poly_price, self.edge_model_poly_price_bucket)

    def _edge_poly_bucket_for(self, poly_price: float, bucket_size: float) -> int:
        bucket_size = max(0.001, bucket_size)
        max_bucket = max(0, int(math.ceil(1.0 / bucket_size)) - 1)
        return min(max(0, int(poly_price // bucket_size)), max_bucket)

    def _edge_btc_move_bucket(self, btc_move_pct: float) -> int:
        return self._edge_btc_move_bucket_for(btc_move_pct, self.edge_model_btc_move_bins)

    def _edge_btc_move_bucket_for(self, btc_move_pct: float, bins: int) -> int:
        bins = max(1, bins)
        max_abs = max(0.0001, self.edge_model_btc_move_max_pct)
        clamped = min(max(btc_move_pct, -max_abs), max_abs)
        index = int(((clamped + max_abs) / (2.0 * max_abs)) * bins)
        return min(max(0, index), bins - 1)

    def _edge_qty_ratio_bucket(self, qty_ratio: float) -> int:
        return self._edge_qty_ratio_bucket_for(qty_ratio, self.edge_model_qty_ratio_bins)

    def _edge_qty_ratio_bucket_for(self, qty_ratio: float, bins: int) -> int:
        bins = max(1, bins)
        max_ratio = max(0.1, self.edge_model_qty_ratio_max)
        clamped = min(max(qty_ratio, 0.0), max_ratio)
        index = int((clamped / max_ratio) * bins)
        return min(max(0, index), bins - 1)

    def _entry_window_open(self) -> bool:
        if not self.market:
            return False
        if self.entry_window_seconds <= 0:
            return True
        elapsed = time.time() - self.market.window_start
        return 0 <= elapsed <= self.entry_window_seconds

    def _try_exit(self) -> None:
        if self._pending_sell:
            return
        if not self.position:
            return
        sell_price = self._live_sell_price(self.position.token_id, self.position.shares)
        if sell_price <= 0:
            print("  Exit monitor skipped: no executable sell price")
            return

        signal_result = self.strategy.exit_signal()
        unrealized_profit = self._update_profit_trail(sell_price)
        min_sell_price = self.position.entry_price * (1.0 + self.config.min_profit_pct)

        if signal_result:
            if sell_price < min_sell_price:
                print(
                    f"  Binance down signal held: sell ${sell_price:.3f} < "
                    f"min profit ${min_sell_price:.3f}"
                )
                return
            self._sell_position(
                sell_price,
                reason=(
                    f"{signal_result.reason}: BTC bucket {signal_result.move_pct:+.4f}% "
                    f"qty {signal_result.bucket.qty:.4f} "
                    f"(push {self._format_push_signal(signal_result)}, qty z {signal_result.volume_z:.2f})"
                ),
            )
            return

        retreat_hit, retreat_pct = self._profit_retreat_hit(unrealized_profit)
        if not retreat_hit:
            return
        if sell_price < min_sell_price:
            print(
                f"  Profit retreat held: sell ${sell_price:.3f} < "
                f"min profit ${min_sell_price:.3f}; "
                f"current ${unrealized_profit:+.2f}, peak ${self.position.peak_unrealized_profit:+.2f}, "
                f"retreat {retreat_pct:.1%}"
            )
            return

        price_zone = "above_50" if sell_price > 0.50 else "at_or_below_50"
        self._sell_position(
            sell_price,
            reason=(
                f"profit_retreat_{price_zone}: current ${unrealized_profit:+.2f}, "
                f"peak ${self.position.peak_unrealized_profit:+.2f}, "
                f"retreat {retreat_pct:.1%}"
            ),
        )

    def _process_pending_exit(self, sell_price: float, min_sell_price: float) -> bool:
        pending = self._pending_exit
        if not pending:
            return False

        signal_result = pending.signal
        now = time.time()
        self._log_pending_poly_observation(pending, "exit")
        trend_start_ts = self._pending_trend_start_ts(pending)
        if now < trend_start_ts:
            return True

        current_poly_price, current_poly_received_ts = self._poly_reference_snapshot()
        if current_poly_price <= 0:
            current_poly_price = sell_price
            current_poly_received_ts = time.time()
        trend = self._poly_recent_trend(trend_start_ts, self.latency_poly_trend_ticks)
        if not trend:
            if now < pending.deadline_ts:
                return True
            print(
                f"  Exit latency wait expired without {self.latency_poly_trend_ticks} "
                f"Polymarket ticks after trend start +{(trend_start_ts - pending.created_ts) * 1000:.0f}ms"
            )
            trend = {
                "count": 1,
                "first": current_poly_price,
                "last": current_poly_price,
                "delta": 0.0,
                "duration_ms": 0.0,
            }

        poly_delta = trend["delta"]
        falling = poly_delta < 0

        if falling:
            pass
        elif pending.waits < 1 and now >= pending.deadline_ts:
            pending.waits += 1
            pending.deadline_ts = now + self._latency_wait_seconds()
            print(
                f"  Exit latency extended once: Poly not falling yet | "
                f"{self._format_trend(trend)} | "
                f"next deadline {self._latency_wait_seconds() * 1000:.0f}ms"
            )
            return True
        elif now < pending.deadline_ts:
            return True

        if sell_price < min_sell_price:
            print(
                f"  Binance down signal held: sell ${sell_price:.3f} < "
                f"min profit ${min_sell_price:.3f}"
            )
            self._pending_exit = None
            return True

        self._sell_position(
            sell_price,
            reason=(
                f"{signal_result.reason}: BTC bucket {signal_result.move_pct:+.4f}% "
                f"qty {signal_result.bucket.qty:.4f} "
                f"(push {self._format_push_signal(signal_result)}, qty z {signal_result.volume_z:.2f}) | "
                f"Poly trend {self._format_trend(trend)} | latency avg {self._latency_avg_ms:.0f}ms"
            ),
        )
        self._pending_exit = None
        return True

    def _log_pending_poly_observation(self, pending: PendingSignal, label: str) -> None:
        if pending.observation_logged:
            return
        if self.latency_poly_trend_delay_seconds <= 0:
            pending.observation_logged = True
            return
        observe_end = pending.created_ts + self.latency_poly_trend_delay_seconds
        if time.time() < observe_end:
            return
        pending.observation_logged = True
        samples = self._poly_samples_between(pending.created_ts, observe_end)
        if len(samples) < 2:
            print(
                f"  {label.title()} latency first {self.latency_poly_trend_delay_seconds * 1000:.0f}ms "
                f"Poly observation: insufficient ticks ({len(samples)})"
            )
            return
        first = samples[0]
        last = samples[-1]
        print(
            f"  {label.title()} latency first {self.latency_poly_trend_delay_seconds * 1000:.0f}ms "
            f"Poly observation: {len(samples)} ticks | "
            f"${first.price:.3f}->{last.price:.3f} ({last.price - first.price:+.3f}) | "
            f"{(last.ts - first.ts) * 1000:.0f}ms"
        )

    def _poly_recent_trend(self, start_ts: float, ticks: int) -> Optional[dict[str, float]]:
        samples = self._poly_samples_since(start_ts)
        if len(samples) < ticks:
            return None
        window = samples[-ticks:]
        first = window[0]
        last = window[-1]
        return {
            "count": float(len(window)),
            "first": first.price,
            "last": last.price,
            "delta": last.price - first.price,
            "duration_ms": (last.ts - first.ts) * 1000.0,
        }

    def _poly_recently_falling(self, start_ts: float, ticks: int) -> bool:
        samples = self._poly_samples_since(start_ts)
        if len(samples) < ticks:
            return False
        window = samples[-ticks:]
        if window[-1].price >= window[0].price:
            return False
        return all(
            current.price <= previous.price
            for previous, current in zip(window, window[1:])
        )

    def _poly_samples_since(self, start_ts: float) -> list[PriceSample]:
        if not self.market:
            return []
        segment = self.market.window_start
        return [
            sample for sample in self._poly_price_samples
            if sample.segment == segment and sample.ts >= start_ts
        ]

    def _poly_samples_between(self, start_ts: float, end_ts: float) -> list[PriceSample]:
        if not self.market:
            return []
        segment = self.market.window_start
        return [
            sample for sample in self._poly_price_samples
            if sample.segment == segment and start_ts <= sample.ts <= end_ts
        ]

    def _format_trend(self, trend: dict[str, float]) -> str:
        return (
            f"{int(trend['count'])} ticks "
            f"${trend['first']:.3f}->{trend['last']:.3f} "
            f"({trend['delta']:+.3f}, {trend['duration_ms']:.0f}ms)"
        )

    def _format_push_signal(self, signal_result: object) -> str:
        deviation = getattr(signal_result, "push_deviation", 0.0)
        delta = getattr(signal_result, "push_delta", 0.0)
        mean = getattr(signal_result, "push_mean", 0.0)
        std = getattr(signal_result, "push_std", 0.0)
        count = int(getattr(signal_result, "push_count", 0) or 0)
        current_count = int(getattr(signal_result, "push_current_count", 0) or 0)
        current_mean = getattr(signal_result, "push_current_mean", 0.0)
        mean_shift_z = getattr(signal_result, "push_mean_shift_z", 0.0)
        std_ratio = getattr(signal_result, "push_std_ratio", 0.0)
        source = getattr(signal_result, "push_source", "window")
        if count <= 0:
            return "push dev n/a"
        return (
            f"push {source} {deviation:+.2f}x "
            f"(mean_z {mean_shift_z:+.2f}, mean ${mean:,.2f}->${current_mean:,.2f}, "
            f"d ${delta:+.2f}, base_std ${std:.2f}, cur_std {std_ratio:.2f}x, "
            f"n={current_count}/{count})"
        )
        return (
            f"push dev {deviation:+.2f}σ "
            f"(price-mean ${delta:+.2f}, mean ${mean:,.2f}, std ${std:.2f}, n={count})"
        )

    def _update_profit_trail(self, sell_price: float) -> float:
        if not self.position:
            return 0.0
        self.position.last_sell_price = sell_price
        unrealized_revenue = self.position.exit_revenue + self.position.shares * sell_price
        unrealized_profit = unrealized_revenue - self.position.cost
        if unrealized_profit > self.position.peak_unrealized_profit:
            self.position.peak_unrealized_profit = unrealized_profit
            self.position.peak_sell_price = sell_price
        return unrealized_profit

    def _profit_retreat_hit(self, unrealized_profit: float) -> tuple[bool, float]:
        if not self.position:
            return False, 0.0
        peak = self.position.peak_unrealized_profit
        min_profit_usd = self.position.cost * self.config.min_profit_pct
        if peak < min_profit_usd:
            return False, 0.0
        retreat = (peak - unrealized_profit) / peak
        return retreat >= self.config.profit_retreat_pct, retreat

    def _sell_position(self, sell_price: float, reason: str) -> None:
        if not self.position:
            return
        if self._simulate_order_failure("SELL"):
            return

        print(f"\n[EXIT] {reason} | sell ${sell_price:.3f}")
        result = self.executor.place_sell_order(self.position.token_id, self.position.shares, price=sell_price)
        if not result.success:
            print(f"  Sell submit failed: {result.error}")
            return

        now = time.time()
        required_confirm_checks = random.choices([2, 3, 4], weights=[25, 50, 25], k=1)[0] if self.dry_run else 1
        self._pending_sell = PendingSellOrder(
            order_id=result.order_id,
            token_id=self.position.token_id,
            window_ts=self.position.window_ts,
            market_slug=self.position.market_slug,
            price=result.price or sell_price,
            shares=self.position.shares,
            reason=reason,
            created_ts=now,
            next_check_ts=now + self._pending_buy_check_interval_seconds,
            balance_before=result.balance_before,
            token_balance_before=result.token_balance_before,
            required_confirm_checks=required_confirm_checks,
            last_signal_seq=self._edge_model_signal_seq,
        )
        self._last_trade_action_ts = now
        print(
            f"  Sell order pending: {result.order_id} | "
            f"{self.position.shares:.0f} shares @ ${result.price:.3f}; "
            f"check every {self._pending_buy_check_interval_seconds:.0f}s | "
            f"confirm after {required_confirm_checks} checks"
        )

    def _apply_sell_fill(self, result, pending: PendingSellOrder) -> None:
        if not self.position:
            return
        self.position.exit_revenue += result.amount_usd
        self.position.shares = max(0.0, self.position.shares - result.shares)
        if self.position.shares < 1.0:
            self.position.closed = True

        profit = self.position.exit_revenue - self.position.cost
        self._last_trade_action_ts = time.time()
        if self.position.closed:
            self.realized_pnl += profit
            self._update_strategy_drawdown_guard()
            print(f"  Exit filled: revenue ${result.amount_usd:.2f}, profit ${profit:+.2f}")
            self.telegram.strategy_result_alert(
                strategy="Binance latency spike",
                profit=profit,
                strategy_pnl=self.realized_pnl,
                total_pnl=self.realized_pnl,
            )
            return
        print(
            f"  Exit partial: revenue ${result.amount_usd:.2f}, "
            f"sold {result.shares:.0f}, remaining {self.position.shares:.0f}, "
            f"unrealized profit ${profit:+.2f}"
        )

    def _process_pending_sell(self) -> None:
        pending = self._pending_sell
        if not pending:
            return

        now = time.time()
        if pending.cancel_requested:
            if now >= pending.next_cancel_ts:
                self._cancel_pending_sell(pending.cancel_reason or "cancel requested")
            return

        if self._pending_sell_cancel_signal(pending):
            self._cancel_pending_sell(pending.cancel_reason or "positive Binance follow-through")
            return

        if now < pending.next_check_ts:
            return

        pending.check_attempts += 1
        result = self.executor.check_pending_sell(
            pending.order_id,
            pending.price,
            pending.shares,
            pending.token_id,
            pending.balance_before,
            pending.token_balance_before,
        )
        pending.next_check_ts = now + self._pending_buy_check_interval_seconds
        if not result:
            print(f"  Pending sell check: {pending.order_id} not filled yet")
            return
        if self.dry_run and pending.check_attempts < pending.required_confirm_checks:
            print(
                f"  Pending sell check {pending.check_attempts}/"
                f"{pending.required_confirm_checks}: simulated order still pending"
            )
            return

        self._pending_sell = None
        self._apply_sell_fill(result, pending)

    def _pending_sell_cancel_signal(self, pending: PendingSellOrder) -> bool:
        if self.dry_run and pending.dry_cancel_blocked:
            return False
        if self._edge_model_signal_seq <= pending.last_signal_seq:
            return False
        pending.last_signal_seq = self._edge_model_signal_seq
        features = self._edge_model_features(time.time(), self.current_btc_price)
        if not features or features.btc_move_pct < self.pending_sell_cancel_btc_move_pct:
            return False
        pending.cancel_reason = (
            f"Binance up while sell pending ({features.btc_move_pct:+.4f}% "
            f">= {self.pending_sell_cancel_btc_move_pct:.4f}%)"
        )
        return True

    def _cancel_pending_sell(self, reason: str) -> None:
        pending = self._pending_sell
        if not pending:
            return
        if self.dry_run:
            if not pending.dry_cancel_decided:
                pending.dry_cancel_decided = True
                cancel_rate = min(max(0.0, self.simulated_cancel_success_rate), 1.0)
                pending.dry_cancel_blocked = random.random() >= cancel_rate
            if not pending.dry_cancel_blocked:
                print(
                    f"  DRY pending sell cancelled {pending.order_id}: {reason} "
                    f"(sim cancel success {self.simulated_cancel_success_rate:.0%})"
                )
                self._pending_sell = None
                self._last_trade_action_ts = time.time()
                return
            print(
                f"  DRY pending sell cancel failed/already matched {pending.order_id}: {reason} "
                f"(sim cancel success {self.simulated_cancel_success_rate:.0%}); keeping order pending"
            )
            pending.cancel_requested = False
            pending.next_cancel_ts = 0.0
            return

        fill = self.executor.check_pending_sell(
            pending.order_id,
            pending.price,
            pending.shares,
            pending.token_id,
            pending.balance_before,
            pending.token_balance_before,
        )
        if fill:
            print(f"  Pending sell already filled before cancel: {pending.order_id}")
            self._pending_sell = None
            self._apply_sell_fill(fill, pending)
            return
        if self.executor.order_is_cancelled(pending.order_id):
            print(f"  Pending sell cancel confirmed: {pending.order_id}")
            self._pending_sell = None
            self._last_trade_action_ts = time.time()
            return

        print(f"  Cancelling pending sell {pending.order_id}: {reason}")
        pending.cancel_requested = True
        pending.cancel_reason = reason
        if self.executor.cancel_order(pending.order_id):
            now = time.time()
            pending.next_cancel_ts = now + 1.0
            pending.next_check_ts = min(pending.next_check_ts, now + 1.0)
            self._last_trade_action_ts = now
            print("  Pending sell cancel sent; retrying every 1s until order state is confirmed")
            return

        now = time.time()
        pending.next_cancel_ts = now + 1.0
        pending.next_check_ts = min(pending.next_check_ts, now + 1.0)
        print("  Pending sell cancel failed; keeping order under observation")

    def _trade_cooldown_ready(self, action: str) -> bool:
        if self.trade_cooldown_seconds <= 0 or self._last_trade_action_ts <= 0:
            return True
        elapsed = time.time() - self._last_trade_action_ts
        if elapsed >= self.trade_cooldown_seconds:
            return True
        return False

    def _simulate_order_failure(self, action: str) -> bool:
        rate = min(max(0.0, self.simulated_order_failure_rate), 1.0)
        if rate <= 0:
            return False
        if random.random() >= rate:
            return False
        print(f"  Simulated {action} failure ({rate:.0%}); order not sent")
        self._last_trade_action_ts = time.time()
        return True

    def _resolve_position(self, closing_btc_price: float) -> None:
        if not self.position or self.position.closed:
            return

        winner = None if self.dry_run else get_market_winner(self.period_minutes, self.position.window_ts, asset="btc")
        if not winner:
            winner = "UP" if closing_btc_price >= self.position.opening_price else "DOWN"

        if winner == self.position.side:
            claim_price = 0.99
            result = self.executor.sell(self.position.token_id, self.position.shares, price=claim_price)
            revenue = result.amount_usd if result.success else self.position.shares
            profit = self.position.exit_revenue + revenue - self.position.cost
            print(f"  Resolved WIN: +${profit:.2f}")
        else:
            profit = self.position.exit_revenue - self.position.cost
            print(f"  Resolved LOSS: ${profit:.2f}")

        self.realized_pnl += profit
        self._update_strategy_drawdown_guard()
        self.position.closed = True
        self.telegram.strategy_result_alert(
            strategy="Binance latency spike",
            profit=profit,
            strategy_pnl=self.realized_pnl,
            total_pnl=self.realized_pnl,
        )

    def _live_buy_price(self, token_id: str) -> float:
        ws_price = self.poly_feed.get_price(token_id)
        if ws_price and ws_price.best_ask > 0:
            return round(ws_price.best_ask, 2)
        return round(self.executor.get_market_price(token_id, "BUY", self.config.trade_amount), 2)

    def _live_sell_price(self, token_id: str, shares: float) -> float:
        ws_price = self.poly_feed.get_price(token_id)
        if ws_price and ws_price.best_bid > 0:
            return round(ws_price.best_bid, 2)
        notional = max(1.0, shares * 0.50)
        return round(self.executor.get_market_price(token_id, "SELL", notional), 2)

    def _poly_reference_price(self) -> float:
        return self._poly_reference_snapshot()[0]

    def _poly_reference_snapshot(self) -> tuple[float, float]:
        if not self.market:
            return 0.0, 0.0
        price = self.poly_feed.get_price(self.market.token_id_up)
        if not price:
            return 0.0, 0.0
        received_ts = getattr(price, "received_ts", 0.0) or getattr(price, "timestamp", 0.0) or 0.0
        if price.best_bid > 0 and price.best_ask > 0:
            return round((price.best_bid + price.best_ask) / 2.0, 4), received_ts
        if price.best_ask > 0:
            return round(price.best_ask, 4), received_ts
        if price.best_bid > 0:
            return round(price.best_bid, 4), received_ts
        if price.last_trade > 0:
            return round(price.last_trade, 4), received_ts
        return 0.0, received_ts

    def _poly_price_near(self, target_ts: float, known_until_ts: float, max_distance_seconds: float = 1.0) -> float:
        if not self.market or target_ts <= 0:
            return 0.0
        candidates = [
            sample for sample in self._poly_price_samples
            if sample.segment == self.market.window_start
            and sample.price > 0
            and sample.ts <= known_until_ts
            and abs(sample.ts - target_ts) <= max_distance_seconds
        ]
        if not candidates:
            return 0.0
        return min(candidates, key=lambda sample: abs(sample.ts - target_ts)).price

    def _latency_wait_seconds(self) -> float:
        wait_ms = self._latency_avg_ms if self._latency_samples_ms else self.latency_default_ms
        wait_ms = min(max(50.0, wait_ms), self.latency_max_wait_ms)
        return wait_ms / 1000.0

    def _latency_sync_plan(self) -> dict[str, float]:
        target_ms = self._latency_avg_ms if self._latency_samples_ms else self.latency_default_ms
        target_ms = min(max(50.0, target_ms), self.latency_max_wait_ms)
        trend_window_ms = max(0.0, self.latency_poly_trend_window_ms)
        safety_margin_ms = max(0.0, self.latency_safety_margin_ms)
        trend_start_delay_ms = max(0.0, target_ms - trend_window_ms - safety_margin_ms)
        deadline_ms = min(self.latency_max_wait_ms, max(target_ms, trend_start_delay_ms + trend_window_ms))
        return {
            "target_ms": target_ms,
            "trend_start_delay_ms": trend_start_delay_ms,
            "trend_window_ms": trend_window_ms,
            "safety_margin_ms": safety_margin_ms,
            "deadline_ms": max(50.0, deadline_ms),
        }

    def _pending_trend_start_ts(self, pending: PendingSignal) -> float:
        if pending.trend_start_ts > 0:
            return pending.trend_start_ts
        return pending.created_ts + self.latency_poly_trend_delay_seconds

    def _record_channel_latency(self, latency_seconds: float) -> bool:
        latency_ms = max(0.0, latency_seconds * 1000.0)
        if latency_ms < self.config.bucket_ms:
            return False
        if latency_ms > self.waveform_max_lag_seconds * 1000.0:
            return False
        self._latency_samples_ms.append(latency_ms)
        if len(self._latency_samples_ms) > self.latency_samples_limit:
            self._latency_samples_ms = self._latency_samples_ms[-self.latency_samples_limit:]
        self._latency_avg_ms = sum(self._latency_samples_ms) / len(self._latency_samples_ms)
        now = time.time()
        if now - self._last_latency_log_ts >= 10:
            self._last_latency_log_ts = now
            print(
                f"[latency] Binance->Polymarket avg {self._latency_avg_ms:.0f}ms "
                f"from {len(self._latency_samples_ms)} samples"
            )
        return True

    def _update_extrema_latency(self) -> None:
        poly_price, poly_received_ts = self._poly_reference_snapshot()
        if poly_price <= 0 or poly_received_ts <= 0:
            return
        if poly_received_ts <= self._last_poly_extrema_received_ts:
            return
        self._last_poly_extrema_received_ts = poly_received_ts
        self._update_price_sample("poly", poly_price, poly_received_ts)

    def _update_price_sample(self, source: str, price: float, ts: float) -> None:
        if price <= 0 or ts <= 0:
            return
        samples = self._btc_price_samples if source == "btc" else self._poly_price_samples
        segment = self.market.window_start if source == "poly" and self.market else 0
        samples.append(PriceSample(ts=ts, price=price, segment=segment))
        stats = self._feed_gap_stats.get(source)
        if stats:
            stats.update(ts)
        keep_after = ts - 1800.0
        while samples and samples[0].ts < keep_after:
            samples.pop(0)

    def _calibrate_completed_window(self, window: MarketWindow) -> None:
        if window.window_start in self._calibrated_window_starts:
            return
        self._calibrated_window_starts.add(window.window_start)

        target_start = window.window_start + self.waveform_start_offset_seconds
        match_start = self._waveform_available_start(window.window_start, target_start)
        match_seconds = self.waveform_match_seconds
        match_end = match_start + match_seconds
        required_poly_end = match_end + self.waveform_max_lag_seconds
        if match_seconds <= 0 or required_poly_end > window.window_end + 1e-9:
            print(
                f"[latency] waveform skipped {window.slug}: early match window too short "
                f"(start {self._format_ts(match_start)}, need until {self._format_ts(required_poly_end)})"
            )
            return

        best = self._match_waveform_candidate(window.window_start, match_start, match_seconds)

        if not best:
            detail = self._waveform_fail_reason or "unknown reason"
            self._calibration_debug_reason = f"no waveform match for {window.slug}: {detail}"
            print(f"[latency] waveform skipped {window.slug}: {self._calibration_debug_reason}")
            return

        print(
            f"[latency] waveform candidate {window.slug}: BTC early {self._format_ts(best.btc_start_ts)} "
            f"lag {best.latency_ms:.0f}ms | sse {best.sse:.2f} | second {best.second_sse:.2f} | "
            f"scale {best.scale:.1f}x | range btc ${best.btc_range:.2f}/poly ${best.poly_range:.3f} | "
            f"top {best.top_lags}"
        )

        sse_edge = best.second_sse - best.sse if best.second_sse > 0 else 0.0
        sse_edge_pct = sse_edge / max(abs(best.sse), 1.0)
        if sse_edge < self.waveform_min_sse_improvement or sse_edge_pct < self.waveform_min_sse_improvement_pct:
            print(
                f"[latency] waveform rejected {window.slug}: weak SSE separation "
                f"edge {sse_edge:.2f}/{sse_edge_pct:.3%} below "
                f"{self.waveform_min_sse_improvement:.2f}/{self.waveform_min_sse_improvement_pct:.3%} | "
                f"top {best.top_lags}"
            )
            return

        baseline_ready = len(self._latency_samples_ms) >= self.waveform_min_baseline_samples
        if baseline_ready and (
            best.latency_ms < self.waveform_min_accept_ms or best.latency_ms > self.waveform_max_accept_ms
        ):
            print(
                f"[latency] waveform rejected {window.slug}: lag {best.latency_ms:.0f}ms outside accepted "
                f"range {self.waveform_min_accept_ms:.0f}-{self.waveform_max_accept_ms:.0f}ms | "
                f"sse {best.sse:.2f} | top {best.top_lags}"
            )
            return

        if baseline_ready:
            reference = float(np.median(np.array(self._latency_samples_ms, dtype=np.float64)))
            deviation = abs(best.latency_ms - reference)
            if deviation > self.waveform_max_sample_deviation_ms:
                print(
                    f"[latency] waveform rejected {window.slug}: lag {best.latency_ms:.0f}ms "
                    f"deviates {deviation:.0f}ms from median {reference:.0f}ms "
                    f"(limit {self.waveform_max_sample_deviation_ms:.0f}ms) | "
                    f"BTC early {self._format_ts(best.btc_start_ts)} | sse {best.sse:.2f}"
                )
                return

        if not self._record_channel_latency(best.latency_ms / 1000.0):
            print(
                f"[latency] waveform rejected {window.slug}: lag {best.latency_ms:.0f}ms outside valid "
                f"range {self.config.bucket_ms:.0f}-{self.waveform_max_lag_seconds * 1000:.0f}ms | "
                f"BTC early {self._format_ts(best.btc_start_ts)} | sse {best.sse:.2f}"
            )
            return

        self._latency_calibrated = True
        self._window_latency_matches.append(best)
        del self._window_latency_matches[:-self.latency_samples_limit]
        baseline_state = (
            "baseline locked"
            if len(self._latency_samples_ms) >= self.waveform_min_baseline_samples
            else f"baseline collecting {len(self._latency_samples_ms)}/{self.waveform_min_baseline_samples}"
        )
        print(
            f"[latency] waveform matched {window.slug}: BTC early2m {self._format_ts(best.btc_start_ts)} "
            f"lag {best.latency_ms:.0f}ms | sse {best.sse:.2f} | "
            f"second {best.second_sse:.2f} | scale {best.scale:.1f}x | "
            f"range btc ${best.btc_range:.2f}/poly ${best.poly_range:.3f} | top {best.top_lags} | "
            f"avg {self._latency_avg_ms:.0f}ms/{len(self._latency_samples_ms)} | {baseline_state}"
        )

    def _match_waveform_candidate(
        self,
        window_start: int,
        btc_start_ts: float,
        match_seconds: Optional[float] = None,
    ) -> Optional[WindowLatencyMatch]:
        sample_seconds = self.waveform_sample_ms / 1000.0
        match_seconds = match_seconds or self.waveform_match_seconds
        btc_values = self._resample_price_series(
            self._btc_price_samples,
            btc_start_ts,
            btc_start_ts + match_seconds,
            sample_seconds,
            segment=None,
        )
        if btc_values is None or len(btc_values) < self.waveform_min_points:
            self._waveform_fail_reason = (
                f"BTC resample insufficient for {self._format_ts(btc_start_ts)}-"
                f"{self._format_ts(btc_start_ts + match_seconds)} | "
                f"{self._sample_span_debug(self._btc_price_samples, None)}"
            )
            return None
        btc_wave = self._anchor_waveform(btc_values)
        btc_range = self._waveform_range(btc_wave)
        if btc_wave is None or btc_range <= 1e-9:
            self._waveform_fail_reason = "BTC waveform flat/low variance after anchor"
            return None

        match_len = len(btc_wave)
        lag_step = max(1, int(round(self.waveform_lag_step_ms / self.waveform_sample_ms)))
        max_offset = int(round(self.waveform_max_lag_seconds / sample_seconds))

        best_latency_ms = None
        best_sse = None
        second_sse = None
        best_scale = 0.0
        best_btc_range = btc_range
        best_poly_range = 0.0
        max_seen_poly_range = 0.0
        scored: list[tuple[float, float]] = []
        for offset in range(0, max_offset + 1, lag_step):
            lag_seconds = offset * sample_seconds
            poly_slice = self._resample_price_series(
                self._poly_price_samples,
                btc_start_ts + lag_seconds,
                btc_start_ts + lag_seconds + match_seconds,
                sample_seconds,
                segment=window_start,
            )
            if poly_slice is None or len(poly_slice) != match_len:
                continue
            poly_wave = self._anchor_waveform(poly_slice)
            poly_range = self._waveform_range(poly_wave)
            if poly_wave is None or poly_range <= 1e-9:
                continue
            max_seen_poly_range = max(max_seen_poly_range, poly_range)
            if poly_range < self.waveform_min_poly_range:
                continue
            scale = btc_range / poly_range
            if self.waveform_max_scale > 0 and scale > self.waveform_max_scale:
                continue
            scaled_poly = poly_wave * scale
            diff = btc_wave - scaled_poly
            sse = float(np.dot(diff, diff))
            scored.append((offset * self.waveform_sample_ms, sse))
            if best_sse is None or sse < best_sse:
                second_sse = best_sse
                best_sse = sse
                best_latency_ms = offset * self.waveform_sample_ms
                best_scale = scale
                best_poly_range = poly_range
            elif second_sse is None or sse < second_sse:
                second_sse = sse

        if best_latency_ms is None:
            self._waveform_fail_reason = (
                f"no usable Poly lag slices for segment {window_start} "
                f"{self._format_ts(btc_start_ts)}-{self._format_ts(btc_start_ts + match_seconds + self.waveform_max_lag_seconds)} | "
                f"min poly range ${self.waveform_min_poly_range:.3f}, max seen ${max_seen_poly_range:.3f}, "
                f"max scale {self.waveform_max_scale:.0f}x | "
                f"{self._sample_span_debug(self._poly_price_samples, window_start)}"
            )
            return None
        top_lags = ", ".join(
            f"{latency:.0f}ms:{sse:.1f}" for latency, sse in sorted(scored, key=lambda item: item[1])[:5]
        )
        self._waveform_fail_reason = ""
        return WindowLatencyMatch(
            window_start=window_start,
            btc_start_ts=btc_start_ts,
            latency_ms=best_latency_ms,
            sse=best_sse,
            samples=match_len,
            second_sse=second_sse or 0.0,
            top_lags=top_lags,
            scale=best_scale,
            btc_range=best_btc_range,
            poly_range=best_poly_range,
        )

    def _waveform_available_start(self, window_start: int, target_start: float) -> float:
        btc_first = self._first_sample_ts(self._btc_price_samples, segment=None, after_ts=target_start)
        poly_first = self._first_sample_ts(self._poly_price_samples, segment=window_start, after_ts=target_start)
        available = max(target_start, btc_first or target_start)
        if available > target_start + 0.001:
            print(
                f"[latency] waveform start shifted: target {self._format_ts(target_start)} -> "
                f"{self._format_ts(available)} "
                f"(btc first {self._format_ts(btc_first) if btc_first else 'none'}, "
                f"poly first {self._format_ts(poly_first) if poly_first else 'none'})"
            )
        return available

    def _first_sample_ts(
        self,
        samples: list[PriceSample],
        segment: Optional[int],
        after_ts: float,
    ) -> Optional[float]:
        for sample in samples:
            if sample.ts >= after_ts and (segment is None or sample.segment == segment):
                return sample.ts
        return None

    def _resample_price_series(
        self,
        samples: list[PriceSample],
        start_ts: float,
        end_ts: float,
        step_seconds: float,
        segment: Optional[int],
    ) -> Optional[np.ndarray]:
        points = [
            sample for sample in samples
            if sample.ts <= end_ts + self.waveform_max_sample_gap_seconds
            and sample.ts >= start_ts - self.waveform_max_sample_gap_seconds
            and (segment is None or sample.segment == segment)
        ]
        if len(points) < 2:
            return None
        points.sort(key=lambda sample: sample.ts)
        ts = np.array([sample.ts for sample in points], dtype=np.float64)
        prices = np.array([sample.price for sample in points], dtype=np.float64)
        unique_ts, unique_indices = np.unique(ts, return_index=True)
        unique_prices = prices[unique_indices]
        if len(unique_ts) < 2:
            return None

        first_idx = int(np.searchsorted(unique_ts, start_ts, side="right") - 1)
        if first_idx < 0:
            return None
        grid = np.arange(start_ts, end_ts, step_seconds, dtype=np.float64)
        if len(grid) < self.waveform_min_points:
            return None
        if len(grid) <= 0 or grid[0] < unique_ts[0] or grid[-1] > unique_ts[-1]:
            return None

        if segment is not None:
            return self._resample_poly_price_series(unique_ts, unique_prices, grid, step_seconds)

        right_indices = np.searchsorted(unique_ts, grid, side="left")
        exact = (right_indices < len(unique_ts)) & np.isclose(unique_ts[right_indices], grid, rtol=0.0, atol=1e-9)
        left_indices = np.where(exact, right_indices, right_indices - 1)
        right_indices = np.where(exact, right_indices, right_indices)
        if np.any(left_indices < 0) or np.any(right_indices >= len(unique_ts)):
            return None
        bracket_gaps = unique_ts[right_indices] - unique_ts[left_indices]
        if float(np.max(bracket_gaps)) > self.waveform_max_sample_gap_seconds:
            return None
        return np.interp(grid, unique_ts, unique_prices)

    def _resample_poly_price_series(
        self,
        ts: np.ndarray,
        prices: np.ndarray,
        grid: np.ndarray,
        step_seconds: float,
    ) -> Optional[np.ndarray]:
        bin_sums = np.zeros(len(grid), dtype=np.float64)
        bin_counts = np.zeros(len(grid), dtype=np.int32)
        raw_indices = np.rint((ts - grid[0]) / step_seconds).astype(np.int64)
        valid = (raw_indices >= 0) & (raw_indices < len(grid))
        for idx, price in zip(raw_indices[valid], prices[valid]):
            bin_sums[idx] += float(price)
            bin_counts[idx] += 1

        known = bin_counts > 0
        if int(np.count_nonzero(known)) < self.waveform_min_points:
            return None

        binned = np.full(len(grid), np.nan, dtype=np.float64)
        binned[known] = bin_sums[known] / bin_counts[known]
        known_indices = np.flatnonzero(known)
        max_missing_bins = max(1, int(math.ceil(self.waveform_max_sample_gap_seconds / step_seconds)))
        if int(np.max(np.diff(known_indices))) > max_missing_bins:
            return None
        if known_indices[0] > 0 or known_indices[-1] < len(grid) - 1:
            return None
        all_indices = np.arange(len(grid), dtype=np.float64)
        return np.interp(all_indices, known_indices.astype(np.float64), binned[known])

    def _anchor_waveform(self, values: np.ndarray) -> Optional[np.ndarray]:
        if len(values) < self.waveform_min_points:
            return None
        return values.astype(np.float64) - float(values[0])

    def _waveform_range(self, values: Optional[np.ndarray]) -> float:
        if values is None or len(values) <= 0:
            return 0.0
        return float(np.max(values) - np.min(values))

    def _maybe_print_calibration_debug(self) -> None:
        now = time.time()
        if now - self._last_calibration_debug_ts < 30:
            return
        self._last_calibration_debug_ts = now
        print(f"[latency] calibration pending: {self._calibration_debug_reason}")

    def _format_ts(self, ts: float) -> str:
        return datetime.fromtimestamp(ts).strftime("%H:%M:%S.%f")[:-3]

    def _sample_span_debug(self, samples: list[PriceSample], segment: Optional[int]) -> str:
        selected = [sample for sample in samples if segment is None or sample.segment == segment]
        if not selected:
            return "samples=0"
        gaps = [
            (selected[i].ts - selected[i - 1].ts) * 1000.0
            for i in range(1, len(selected))
            if selected[i].ts > selected[i - 1].ts
        ]
        gap_text = (
            f", max_gap={max(gaps):.0f}ms, avg_gap={sum(gaps) / len(gaps):.0f}ms"
            if gaps
            else ""
        )
        return (
            f"samples={len(selected)}, first={self._format_ts(selected[0].ts)}, "
            f"last={self._format_ts(selected[-1].ts)}{gap_text}"
        )

    def _clob_ok_for_trade(self) -> bool:
        if self.dry_run:
            return True
        try:
            ok = bool(self.executor.client and self.executor.client.get_ok())
        except Exception as exc:
            ok = False
            print(f"  CLOB health check failed: {exc}")

        if ok:
            self._consecutive_clob_failures = 0
            return True

        self._consecutive_clob_failures += 1
        if self._consecutive_clob_failures >= 3:
            self._clob_halted = True
            self.telegram.error_alert("CLOB health failed 3 times; trading halted until next window recovery.")
        return False

    def _strategy_is_cooling_down(self) -> bool:
        if self._strategy_cooldown_until <= 0:
            return False
        remaining = self._strategy_cooldown_until - time.time()
        if remaining <= 0:
            self._strategy_cooldown_until = 0.0
            print("  Strategy cooldown expired - entries enabled")
            return False
        return True

    def _update_strategy_drawdown_guard(self) -> None:
        self._strategy_pnl_peak = max(self._strategy_pnl_peak, self.realized_pnl)
        if self.strategy_peak_loss_drop <= 0:
            return
        drawdown = self._strategy_pnl_peak - self.realized_pnl
        if drawdown < self.strategy_peak_loss_drop:
            return
        self._strategy_cooldown_until = time.time() + max(0.0, self.strategy_cooling_down_period)
        print(
            f"  Strategy peak drawdown ${drawdown:.2f} >= "
            f"${self.strategy_peak_loss_drop:.2f}; cooling down "
            f"{self.strategy_cooling_down_period:.0f}s"
        )
        self.telegram.error_alert(
            f"Strategy cooldown: peak drawdown ${drawdown:.2f} >= "
            f"${self.strategy_peak_loss_drop:.2f}"
        )

    def _probe_clob_recovery(self) -> None:
        if not self._clob_halted or self.dry_run:
            return
        try:
            if self.executor.client and self.executor.client.get_ok():
                self._clob_halted = False
                self._consecutive_clob_failures = 0
                print("  CLOB recovered - trading resumed")
                self.telegram.send("*CLOB recovered* - trading resumed")
        except Exception:
            pass

    def _check_daily_loss_limit(self) -> None:
        if self.dry_run or self.daily_loss_limit <= 0:
            return
        balance = self.executor.get_balance(refresh=False)
        if balance <= 0:
            return
        pnl = balance - self.session_start_balance
        if pnl <= -self.daily_loss_limit:
            self._running = False
            self.telegram.error_alert(f"Daily loss limit hit: ${pnl:.2f}")
            print(f"Daily loss limit hit: ${pnl:.2f}; stopping.")

    def _print_status(self) -> None:
        now = time.time()
        if now - self._last_status_ts < self.status_interval:
            return
        self._last_status_ts = now

        pos = "flat"
        if self._pending_sell:
            pos = f"pending sell {self._pending_sell.shares:.0f} @ ${self._pending_sell.price:.2f}"
        elif self.position and not self.position.closed:
            pos = f"UP {self.position.shares:.0f} @ ${self.position.entry_price:.2f}"
        elif self._strategy_is_cooling_down():
            pos = f"cooldown {self._strategy_cooldown_until - time.time():.0f}s"
        seconds = self.market.seconds_remaining if self.market else 0.0
        print(
            f"[status] BTC ${self.current_btc_price:,.2f} | T-{seconds:.0f}s | "
            f"{pos} | waveform-latency {self._latency_avg_ms:.0f}ms/{len(self._latency_samples_ms)} "
            f"({'locked' if len(self._latency_samples_ms) >= self.waveform_min_baseline_samples else 'collecting'}, "
            f"windows {len(self._window_latency_matches)}) | "
            f"feed gaps {self._feed_gap_summary()} | "
            f"baseline {self.strategy.baseline_summary()}"
        )

    def _feed_gap_summary(self) -> str:
        now = time.time()
        return (
            f"btc last/max/avg {self._format_gap('btc', now)}; "
            f"poly last/max/avg {self._format_gap('poly', now)}"
        )

    def _format_gap(self, source: str, now: float) -> str:
        stats = self._feed_gap_stats[source]
        current_gap_ms = (now - stats.last_ts) * 1000.0 if stats.last_ts > 0 else 0.0
        last_gap_ms = max(stats.last_gap_ms, current_gap_ms)
        max_gap_ms = max(stats.max_gap_ms, current_gap_ms)
        return f"{last_gap_ms:.0f}/{max_gap_ms:.0f}/{stats.avg_gap_ms:.0f}ms"

    def _handle_shutdown(self, *_):
        print("\nStopping PolyBot...")
        self._running = False
        if self._pending_buy:
            print(f"  Cancelling pending buy on shutdown: {self._pending_buy.order_id}")
            self.executor.cancel_order(self._pending_buy.order_id)
            self._pending_buy = None
        if self._pending_sell:
            print(f"  Cancelling pending sell on shutdown: {self._pending_sell.order_id}")
            self.executor.cancel_order(self._pending_sell.order_id)
            self._pending_sell = None
        self._edge_model_save()
        self.price_feed.stop()
        self.poly_feed.stop()
        self.telegram.send("*PolyBot stopped*")

    def _as_float(self, value) -> float:
        try:
            if value in ("", None):
                return 0.0
            return float(value)
        except Exception:
            return 0.0


if __name__ == "__main__":
    PolyBot().start()
