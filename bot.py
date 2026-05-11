#!/usr/bin/env python3
"""
PolyBot v13 — Recalibrated + Safety Systems

Strategy:
  - Brownian motion model with vol=0.12 (recalibrated from 0.08)
  - Entry gate: model confidence >= 80%, market price <= true_prob * 0.85
  - Position sizing: quarter-Kelly, $5–$25 per trade
  - Exit: hold all positions to resolution — no stops, no take-profit

Safety systems:
  1. CLOB health check: get_ok() before every trade; 3 consecutive
     failures halt trading and send Telegram alert. Auto-recovers
     when API comes back at next window boundary.
  2. Daily loss limit: if balance retreats from its daily peak by DAILY_LOSS_LIMIT_RETREAT, halt trading.
  3. Balance-verified buys: snapshot USDC before/after; ghost fills
     caught even when API throws. Never cancels on timeout — returns
     UNVERIFIED_BUY for pending detection at next window boundary.
  4. Pending buy safety net: if buy unverified, check balance at next
     window boundary; retroactively track as filled if balance dropped.
  5. Window-boundary balance sync: real USDC balance overwrites internal
     tracking every 5 minutes. Corrects any accumulated drift.
  6. Minimum notional guard: skip sells below $5 notional; hold to
     resolution instead of hitting Polymarket's minimum-size rejection.
"""

import os
import sys
import time
import signal
import math
import statistics
from typing import Optional
from dotenv import load_dotenv

import logging
logging.getLogger("httpx").setLevel(logging.WARNING)

from market import get_current_market, get_market_winner, current_window_ts, PERIOD_SECONDS
from price_feed import BinancePriceFeed
from polymarket_ws import PolymarketMarketFeed
from strategy import (
    evaluate_trend_entry,
    estimate_true_probability,
    get_trend_entry_skip_reason,
    StrategyConfig,
    TradingStats,
)
from executor import Executor, FILLED, PARTIAL, FAILED, max_buy_price
from telegram_notifier import TelegramNotifier
from tracker import Tracker


FORCED_EXIT_START = 5
FORCED_EXIT_END = 1
POSITION_CHECK_INTERVAL = 0.25
MAX_EXIT_RETRIES = 3
EXIT_RETRY_COOLDOWN = 10


class TeeStream:
    """Write console output to both the original stream and a log file."""

    def __init__(self, console, log_file):
        self.console = console
        self.log_file = log_file

    def write(self, data):
        self.console.write(data)
        self.log_file.write(data)
        self.flush()

    def flush(self):
        self.console.flush()
        self.log_file.flush()

    def isatty(self):
        return hasattr(self.console, "isatty") and self.console.isatty()


class PolyBot:
    def __init__(self):
        load_dotenv()

        self._runtime_log_file = None
        self._setup_runtime_log()

        self.dry_run = os.getenv("DRY_RUN", "true").lower() == "true"
        self.period = int(os.getenv("MARKET_PERIOD", "5"))

        self.strategy_config = StrategyConfig(
            min_edge=float(os.getenv("MIN_EDGE", "0.05")),
            min_prob=float(os.getenv("MIN_PROB", "0.80")),
            min_btc_delta=float(os.getenv("MIN_BTC_DELTA", "0.06")),
            entry_window_start=int(os.getenv("ENTRY_WINDOW_START", "240")),
            entry_window_end=int(os.getenv("ENTRY_WINDOW_END", "10")),
            kelly_fraction=float(os.getenv("KELLY_FRACTION", "0.25")),
            min_bet=float(os.getenv("MIN_BET", "5.0")),
            max_bet=float(os.getenv("MAX_BET", "25.0")),
            trend_entry_window_seconds=int(os.getenv("TREND_ENTRY_WINDOW_SECONDS", "60")),
            trend_entry_threshold_pct=float(os.getenv("TREND_ENTRY_THRESHOLD_PCT", "0.03")),
            trend_entry_skip_threshold_pct=float(os.getenv("TREND_ENTRY_SKIP_THRESHOLD_PCT", "0.20")),
            trend_trade_amount=float(os.getenv("TREND_TRADE_AMOUNT", os.getenv("MAX_BET", "25.0"))),
            invert_trend_entry=os.getenv("INVERT_TREND_ENTRY", "false").lower() == "true",
        )

        initial_bankroll = float(os.getenv("BANKROLL", "100.0"))
        self._daily_loss_limit = float(os.getenv("DAILY_LOSS_LIMIT_RETREAT", os.getenv("DAILY_LOSS_LIMIT", "30.0")))
        self._peak_loss_drop = float(os.getenv("PEAK_LOSS_DROP", "0.0"))
        self._cooling_down_period = float(os.getenv("COOLING_DOWN_PERIOD", "10800"))
        self._strategy_daily_loss_limits = {
            "trend": float(os.getenv("TREND_DAILY_LOSS_LIMIT", os.getenv("DAILY_LOSS_LIMIT", "30.0"))),
            "cheap": float(os.getenv("CHEAP_DAILY_LOSS_LIMIT", os.getenv("DAILY_LOSS_LIMIT", "30.0"))),
            "panic": float(os.getenv("PANIC_DAILY_LOSS_LIMIT", os.getenv("DAILY_LOSS_LIMIT", "30.0"))),
            "relative": float(os.getenv("RELATIVE_DAILY_LOSS_LIMIT", os.getenv("DAILY_LOSS_LIMIT", "30.0"))),
        }
        self._strategy_peak_loss_drops = {
            "trend": float(os.getenv("TREND_PEAK_LOSS_DROP", os.getenv("PEAK_LOSS_DROP", "0.0"))),
            "cheap": float(os.getenv("CHEAP_PEAK_LOSS_DROP", os.getenv("CHEAP_SCALP_PEAK_LOSS_DROP", os.getenv("PEAK_LOSS_DROP", "0.0")))),
            "panic": float(os.getenv("PANIC_PEAK_LOSS_DROP", os.getenv("PEAK_LOSS_DROP", "0.0"))),
            "relative": float(os.getenv("RELATIVE_PEAK_LOSS_DROP", os.getenv("PEAK_LOSS_DROP", "0.0"))),
        }
        self._strategy_cooling_down_periods = {
            "trend": float(os.getenv("TREND_COOLING_DOWN_PERIOD", "5400")),
            "cheap": float(os.getenv("CHEAP_COOLING_DOWN_PERIOD", os.getenv("CHEAP_SCALP_COOLING_DOWN_PERIOD", "5400"))),
            "panic": float(os.getenv("PANIC_COOLING_DOWN_PERIOD", "5400")),
            "relative": float(os.getenv("RELATIVE_COOLING_DOWN_PERIOD", "5400")),
        }
        self._rolling_vol_windows = int(os.getenv("ROLLING_VOL_WINDOWS", "12"))
        self._vol_floor = float(os.getenv("VOL_FLOOR", "0.06"))
        self._vol_cap = float(os.getenv("VOL_CAP", "0.30"))
        self._vol_fallback = 0.12  # used until enough windows accumulate

        self.price_feed = BinancePriceFeed()
        self.poly_feed = PolymarketMarketFeed()
        self.executor = Executor(
            private_key=os.getenv("PRIVATE_KEY", ""),
            safe_address=os.getenv("SAFE_ADDRESS", ""),
            dry_run=self.dry_run,
            signature_type=int(os.getenv("SIGNATURE_TYPE", "3")),
            funder_address=os.getenv("FUNDER_ADDRESS", "") or os.getenv("SAFE_ADDRESS", ""),
        )
        self.telegram = TelegramNotifier()
        self.tracker = Tracker(
            log_dir=os.getenv("LOG_DIR", "logs"),
            log_executions=os.getenv("LOG_EXECUTIONS", "false").lower() == "true",
        )
        self.stats = TradingStats(bankroll=initial_bankroll)
        self.stats.hourly.hour_start = time.time()

        self._poly_ws_enabled = os.getenv("POLYMARKET_WS_ENABLED", "true").lower() == "true"
        self._take_profit_price = float(os.getenv("TAKE_PROFIT_PRICE", "0.70"))
        self._min_profit_pct = float(os.getenv("MIN_PROFIT_PCT", "0.10"))
        self._min_profit_usd = float(os.getenv("MIN_PROFIT_USD", "1.00"))
        self._trend_profit_retreat_pct = float(os.getenv("TREND_PROFIT_RETREAT_PCT", os.getenv("STRATEGY_PROFIT_RETREAT_PCT", "0.20")))
        self._confirm_ws_sell_price = os.getenv("CONFIRM_WS_SELL_PRICE", "true").lower() == "true"
        self._cheap_scalp_enabled = os.getenv("CHEAP_SCALP_ENABLED", "true").lower() == "true"
        self._cheap_entry_price = float(os.getenv("CHEAP_ENTRY_PRICE", "0.20"))
        self._cheap_entry_window_seconds = int(os.getenv("CHEAP_ENTRY_WINDOW_SECONDS", "150"))
        self._cheap_trade_amount = float(os.getenv("CHEAP_TRADE_AMOUNT", os.getenv("TREND_TRADE_AMOUNT", "25.0")))
        self._cheap_take_profit_price = float(os.getenv("CHEAP_BUY_TAKE_PROFIT_PRICE", "0.30"))
        self._cheap_min_profit_pct = float(os.getenv("CHEAP_BUY_MIN_PROFIT_PCT", "0.10"))
        self._cheap_min_profit_usd = float(os.getenv("CHEAP_BUY_MIN_PROFIT_USD", "1.00"))
        self._cheap_profit_retreat_pct = float(os.getenv("CHEAP_PROFIT_RETREAT_PCT", os.getenv("CHEAP_SCALP_PROFIT_RETREAT_PCT", os.getenv("STRATEGY_PROFIT_RETREAT_PCT", "0.20"))))
        self._panic_enabled = os.getenv("PANIC_BUY_ENABLED", "true").lower() == "true"
        self._panic_lookback_seconds = int(os.getenv("PANIC_LOOKBACK_SECONDS", "20"))
        self._panic_drop_pct = float(os.getenv("PANIC_DROP_PCT", "0.25"))
        self._panic_max_buy_price = float(os.getenv("PANIC_MAX_BUY_PRICE", "0.45"))
        self._panic_max_btc_against_pct = float(os.getenv("PANIC_MAX_BTC_AGAINST_PCT", "0.08"))
        self._panic_entry_window_seconds = int(os.getenv("PANIC_ENTRY_WINDOW_SECONDS", "240"))
        self._panic_trade_amount = float(os.getenv("PANIC_TRADE_AMOUNT", os.getenv("TREND_TRADE_AMOUNT", "25.0")))
        self._panic_take_profit_price = float(os.getenv("PANIC_TAKE_PROFIT_PRICE", "0.55"))
        self._panic_min_profit_pct = float(os.getenv("PANIC_MIN_PROFIT_PCT", "0.12"))
        self._panic_min_profit_usd = float(os.getenv("PANIC_MIN_PROFIT_USD", "1.00"))
        self._panic_profit_retreat_pct = float(os.getenv("PANIC_PROFIT_RETREAT_PCT", os.getenv("PANIC_REBOUND_PROFIT_RETREAT_PCT", os.getenv("STRATEGY_PROFIT_RETREAT_PCT", "0.20"))))
        self._realtime_to_save = int(os.getenv("REALTIME_TO_SAVE", "120"))
        self._relative_enabled = os.getenv("RELATIVE_REACTION_ENABLED", "true").lower() == "true"
        self._relative_entry_window_seconds = int(os.getenv("RELATIVE_ENTRY_WINDOW_SECONDS", "120"))
        self._relative_lookback_seconds = int(os.getenv("RELATIVE_LOOKBACK_SECONDS", str(self._realtime_to_save)))
        self._relative_min_btc_move_pct = float(os.getenv("RELATIVE_MIN_BTC_MOVE_PCT", "0.02"))
        self._relative_overreaction_price = float(
            os.getenv("RELATIVE_OVERREACTION_PRICE", os.getenv("RELATIVE_OVERREACTION_CENTS", "0.10"))
        )
        self._relative_trade_amount = float(os.getenv("RELATIVE_TRADE_AMOUNT", os.getenv("TREND_TRADE_AMOUNT", "25.0")))
        self._relative_min_buy_price = float(os.getenv("RELATIVE_MIN_BUY_PRICE", "0.05"))
        self._relative_max_buy_price = float(os.getenv("RELATIVE_MAX_BUY_PRICE", "0.65"))
        self._relative_take_profit_price = float(os.getenv("RELATIVE_TAKE_PROFIT_PRICE", "0.55"))
        self._relative_min_profit_pct = float(os.getenv("RELATIVE_MIN_PROFIT_PCT", "0.10"))
        self._relative_min_profit_usd = float(os.getenv("RELATIVE_MIN_PROFIT_USD", "1.00"))
        self._relative_profit_retreat_pct = float(os.getenv("RELATIVE_PROFIT_RETREAT_PCT", os.getenv("RELATIVE_OVERREACTION_PROFIT_RETREAT_PCT", os.getenv("STRATEGY_PROFIT_RETREAT_PCT", "0.20"))))
        self._poly_ws_warmup_seconds = float(os.getenv("POLYMARKET_WS_WARMUP_SECONDS", "2.0"))

        self._running = False
        self._current_window: int = 0
        self._current_market = None
        self._market_subscription_time: float = 0.0
        self._opening_price: float = 0.0
        self._last_hour_check: int = 0
        self._winner_cache: dict = {}
        self._dry_pending_resolutions: list = []
        self._resolving_window_ts: int = 0
        self._resolving_window_winner: str = ""

        # Trade state
        self._traded: bool = False
        self._trade_attempted: bool = False
        self._trade_side: str = ""
        self._trade_price: float = 0.0
        self._trade_cost: float = 0.0
        self._trade_shares: float = 0.0
        self._trade_token_id: str = ""

        # Exit state
        self._exited: bool = False
        self._exit_revenue: float = 0.0
        self._exit_shares_sold: float = 0.0
        self._residual_shares: float = 0.0  # Shares left after partial fill
        self._last_position_check: float = 0.0
        self._last_status_print: float = 0.0
        self._last_tick_context: dict = {}   # last entry-window state, for window-end signal logging
        self._session_start_time: float = time.time()
        self._recent_window_deltas: list = []  # rolling abs(close_delta_pct) per window
        self._exit_retries: int = 0
        self._exit_gave_up: bool = False
        self._last_sell_price_seen: float = 0.0  # last observed sell price during hold period
        self._trend_trailing_armed: bool = False
        self._trend_peak_profit: float = 0.0

        # Secondary strategy: cheap Polymarket price scalp
        self._cheap_traded: bool = False
        self._cheap_trade_attempted: bool = False
        self._cheap_exited: bool = False
        self._cheap_side: str = ""
        self._cheap_token_id: str = ""
        self._cheap_price: float = 0.0
        self._cheap_cost: float = 0.0
        self._cheap_shares: float = 0.0
        self._cheap_exit_revenue: float = 0.0
        self._cheap_last_check: float = 0.0
        self._cheap_trailing_armed: bool = False
        self._cheap_peak_profit: float = 0.0
        self._cheap_pending_side: str = ""
        self._cheap_pending_token_id: str = ""
        self._cheap_pending_order_id: str = ""
        self._cheap_pending_price: float = 0.0
        self._cheap_pending_amount: float = 0.0
        self._cheap_pending_shares: float = 0.0
        self._cheap_pending_balance_before: float = 0.0
        self._cheap_pending_last_check: float = 0.0

        # Strategy-level risk and P&L
        self._strategy_pnl = {"trend": 0.0, "cheap": 0.0, "panic": 0.0, "relative": 0.0}
        self._strategy_loss_halted = {"trend": False, "cheap": False, "panic": False, "relative": False}
        self._total_pnl_peak: float = 0.0
        self._balance_peak: float = initial_bankroll
        self._bot_cooldown_until: float = 0.0
        self._strategy_pnl_peaks = {"trend": 0.0, "cheap": 0.0, "panic": 0.0, "relative": 0.0}
        self._strategy_cooldown_until = {"trend": 0.0, "cheap": 0.0, "panic": 0.0, "relative": 0.0}

        # Fourth strategy: Binance/Polymarket relative overreaction
        self._realtime_history = []
        self._relative_traded: bool = False
        self._relative_trade_attempted: bool = False
        self._relative_exited: bool = False
        self._relative_side: str = ""
        self._relative_token_id: str = ""
        self._relative_price: float = 0.0
        self._relative_cost: float = 0.0
        self._relative_shares: float = 0.0
        self._relative_exit_revenue: float = 0.0
        self._relative_last_check: float = 0.0
        self._relative_trailing_armed: bool = False
        self._relative_peak_profit: float = 0.0
        self._relative_pending_side: str = ""
        self._relative_pending_token_id: str = ""
        self._relative_pending_order_id: str = ""
        self._relative_pending_price: float = 0.0
        self._relative_pending_amount: float = 0.0
        self._relative_pending_shares: float = 0.0
        self._relative_pending_balance_before: float = 0.0
        self._relative_pending_last_check: float = 0.0

        # Third strategy: panic sell rebound
        self._panic_traded: bool = False
        self._panic_trade_attempted: bool = False
        self._panic_exited: bool = False
        self._panic_side: str = ""
        self._panic_token_id: str = ""
        self._panic_price: float = 0.0
        self._panic_cost: float = 0.0
        self._panic_shares: float = 0.0
        self._panic_exit_revenue: float = 0.0
        self._panic_last_check: float = 0.0
        self._panic_trailing_armed: bool = False
        self._panic_peak_profit: float = 0.0
        self._panic_price_history = {"UP": [], "DOWN": []}
        self._panic_pending_side: str = ""
        self._panic_pending_token_id: str = ""
        self._panic_pending_order_id: str = ""
        self._panic_pending_price: float = 0.0
        self._panic_pending_amount: float = 0.0
        self._panic_pending_shares: float = 0.0
        self._panic_pending_balance_before: float = 0.0
        self._panic_pending_last_check: float = 0.0

        # Pending phantom verification (claim sell reported success but balance didn't move yet)
        # Resolved at next window boundary once Polygon settlement has had time to land.
        self._pending_phantom: dict = {}

        # Pending buy (unverified — Polygon settlement too slow)
        self._pending_buy_side: str = ""
        self._pending_buy_price: float = 0.0
        self._pending_buy_amount: float = 0.0
        self._pending_buy_shares: float = 0.0
        self._pending_buy_token_id: str = ""
        self._pending_buy_edge: float = 0.0
        self._pending_buy_delta: float = 0.0
        self._pending_buy_order_id: str = ""
        self._pending_buy_last_check: float = 0.0
        self._balance_before_buy: float = 0.0

        # Unclaimed
        self._unclaimed_winnings: float = 0.0

        # Real balance tracking (source of truth)
        self._session_start_balance: float = 0.0
        self._last_real_balance: float = 0.0

        # Price cache
        self._cached_up: float = 0.50
        self._cached_down: float = 0.50
        self._price_last_fetched: float = 0.0
        self._PRICE_REFRESH: float = 5.0

        # Circuit breaker — detects CLOB API degradation
        self._consecutive_buy_failures: int = 0
        self._clob_halted: bool = False
        self._HALT_AFTER_FAILURES: int = 3
        self._daily_loss_halted: bool = False

    def _setup_runtime_log(self):
        if os.getenv("BOT_RUNTIME_LOG_ENABLED", "true").lower() != "true":
            return
        if isinstance(sys.stdout, TeeStream):
            return

        log_dir = os.getenv("LOG_DIR", "logs")
        os.makedirs(log_dir, exist_ok=True)
        configured_path = os.getenv("BOT_RUNTIME_LOG_FILE", "").strip()
        if configured_path:
            log_path = configured_path
            parent = os.path.dirname(log_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
        else:
            log_path = os.path.join(log_dir, f"bot_{time.strftime('%Y%m%d')}.log")

        self._runtime_log_file = open(log_path, "a", encoding="utf-8", errors="replace", buffering=1)
        sys.stdout = TeeStream(sys.stdout, self._runtime_log_file)
        sys.stderr = TeeStream(sys.stderr, self._runtime_log_file)
        print(f"  Runtime log file: {log_path}")

    def start(self):
        if not self.dry_run:
            print("  Direct CLOB connection enabled (Tor disabled)")

        print("=" * 55)
        print(f"  PolyBot v13 — Recalibrated (vol=0.12)")
        print(f"  Mode: {'DRY RUN' if self.dry_run else '🔴 LIVE TRADING'}")
        print(f"  Bets: ${self.strategy_config.min_bet:.0f}–${self.strategy_config.max_bet:.0f}")
        print(f"  Entry mode: Binance trend scalp")
        print(f"  Entry direction: {'inverse trend' if self.strategy_config.invert_trend_entry else 'trend follow'}")
        print(f"  Trend trigger: {self.strategy_config.trend_entry_threshold_pct:.3f}% | "
              f"Skip chase: >{self.strategy_config.trend_entry_skip_threshold_pct:.3f}%")
        print(f"  Entry window: first {self.strategy_config.trend_entry_window_seconds}s | "
              f"Trade amount: ${self.strategy_config.trend_trade_amount:.2f}")
        print(f"  Max buy price: ${max_buy_price():.2f}")
        print(f"  Polymarket WS: {'on' if self._poly_ws_enabled else 'off'} | "
              f"Take profit: ${self._take_profit_price:.2f} or {self._min_profit_pct:.0%}+")
        print(f"  Cheap scalp: {'on' if self._cheap_scalp_enabled else 'off'} | "
              f"entry <= ${self._cheap_entry_price:.2f} for first {self._cheap_entry_window_seconds}s")
        print(f"  Cheap scalp exit: ${self._cheap_take_profit_price:.2f} or "
              f"{self._cheap_min_profit_pct:.0%}+ / ${self._cheap_min_profit_usd:.2f}+")
        print(f"  Panic rebound: {'on' if self._panic_enabled else 'off'} | "
              f"drop {self._panic_drop_pct:.0%} / buy <= ${self._panic_max_buy_price:.2f}")
        print(f"  Vol: dynamic (fallback=0.12, floor={self._vol_floor}, cap={self._vol_cap}, windows={self._rolling_vol_windows})")
        print(f"  Exits: arm on profit target, sell on profit retreat; otherwise hold to resolution")
        print(
            "  Profit trailing retreats: "
            f"trend {self._trend_profit_retreat_pct:.0%} | "
            f"cheap {self._cheap_profit_retreat_pct:.0%} | "
            f"panic {self._panic_profit_retreat_pct:.0%} | "
            f"relative {self._relative_profit_retreat_pct:.0%}"
        )
        print(f"  Daily balance retreat hard stop: ${self._daily_loss_limit:.0f}")
        print(
            f"  Peak drawdown cooldown: bot ${self._peak_loss_drop:.0f} drop / "
            f"{self._format_duration(self._cooling_down_period)}"
        )
        print(
            "  Strategy loss limits: "
            f"trend ${self._strategy_daily_loss_limits['trend']:.0f} | "
            f"cheap ${self._strategy_daily_loss_limits['cheap']:.0f} | "
            f"panic ${self._strategy_daily_loss_limits['panic']:.0f} | "
            f"relative ${self._strategy_daily_loss_limits['relative']:.0f}"
        )
        print(
            "  Strategy peak drawdown cooldowns: "
            f"trend ${self._strategy_peak_loss_drops['trend']:.0f}/{self._format_duration(self._strategy_cooling_down_periods['trend'])} | "
            f"cheap ${self._strategy_peak_loss_drops['cheap']:.0f}/{self._format_duration(self._strategy_cooling_down_periods['cheap'])} | "
            f"panic ${self._strategy_peak_loss_drops['panic']:.0f}/{self._format_duration(self._strategy_cooling_down_periods['panic'])} | "
            f"relative ${self._strategy_peak_loss_drops['relative']:.0f}/{self._format_duration(self._strategy_cooling_down_periods['relative'])}"
        )
        print(f"  Relative overreaction: {'on' if self._relative_enabled else 'off'} | "
              f"entry first {self._relative_entry_window_seconds}s | save {self._realtime_to_save}s")
        print(f"  Bankroll: ${self.stats.bankroll:.2f}")
        print("=" * 55)

        if not self.dry_run:
            if not self.executor.initialize():
                print("\n❌ Failed to initialize. Check credentials.")
                return
            balance = self.executor.get_balance()
            print(f"  USDC balance: ${balance:.2f}")
            self.stats.bankroll = balance
            self._session_start_balance = balance
            self._last_real_balance = balance
            self._balance_peak = balance
            self.tracker.set_session_balance(balance)
        else:
            print("  [dry run — no wallet connection]")
            self._session_start_balance = self.stats.bankroll
            self._last_real_balance = self.stats.bankroll
            self._balance_peak = self.stats.bankroll
            self.tracker.set_session_balance(self.stats.bankroll)

        self.price_feed.start()
        if self._poly_ws_enabled:
            self.poly_feed.start()
        print("\n⏳ Waiting for BTC price...")
        price = self.price_feed.wait_for_price(timeout=30)
        if not price:
            print("❌ No price feed. Check internet.")
            return
        print(f"✅ BTC: ${price:,.2f} ({self.price_feed.state.source})")

        self.telegram.startup_alert({
            "dry_run": self.dry_run,
            "kelly_fraction": self.strategy_config.kelly_fraction,
            "min_edge": self.strategy_config.min_edge,
            "min_bet": self.strategy_config.min_bet,
            "max_bet": self.strategy_config.max_bet,
            "entry_start": self.strategy_config.entry_window_start,
            "entry_end": self.strategy_config.entry_window_end,
        })

        self._running = True
        self._last_hour_check = int(time.time() // 3600)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        print("\n🚀 Running. Ctrl+C to stop.\n")
        self._main_loop()

    def _main_loop(self):
        while self._running:
            try:
                self._tick()
                self._check_hourly_summary()
            except Exception as e:
                print(f"[error] {e}")
                self.telegram.error_alert(str(e))
            time.sleep(0.1)

    def _tick(self):
        now = time.time()
        period_secs = PERIOD_SECONDS[self.period]
        window_ts = int(now) - (int(now) % period_secs)

        btc_price, is_fresh = self.price_feed.get_price()
        if not is_fresh or btc_price <= 0:
            return

        self._check_pending_buy_fill(now)
        self._check_pending_cheap_buy(now)
        self._check_pending_panic_buy(now)
        self._check_pending_relative_buy(now)

        if window_ts != self._current_window:
            self._on_new_window(window_ts, closing_btc_price=btc_price)

        seconds_remaining = (window_ts + period_secs) - now

        if self._opening_price <= 0:
            self._opening_price = btc_price
            print(f"  📌 Open: ${btc_price:,.2f}")

        self._ensure_market_subscription()
        ws_warmed = (
            not self._poly_ws_enabled
            or (
                self._market_subscription_time > 0
                and now - self._market_subscription_time >= self._poly_ws_warmup_seconds
            )
        )
        self._refresh_cached_ws_prices()
        if ws_warmed:
            self._update_realtime_history(now, btc_price)
            self._update_panic_price_history(now)

            self._manage_relative_reaction(seconds_remaining, now)
            if not self._relative_traded and not self._relative_trade_attempted:
                self._try_relative_reaction_entry(seconds_remaining)
            self._manage_cheap_scalp(seconds_remaining, now)
            if not self._cheap_traded and not self._cheap_trade_attempted:
                self._try_cheap_scalp_entry(seconds_remaining)
            self._manage_panic_rebound(seconds_remaining, now)
            if not self._panic_traded and not self._panic_trade_attempted:
                self._try_panic_rebound_entry(btc_price, seconds_remaining)

        # HOLDING: active position management
        if self._traded and not self._exited and not self._exit_gave_up:
            self._manage_position(btc_price, seconds_remaining, now)
            return

        # Already done
        if self._traded or self._trade_attempted:
            return

        # IDLE: watch Binance during the first minute and enter on trend.
        signal_result = evaluate_trend_entry(
            btc_price=btc_price,
            opening_price=self._opening_price,
            seconds_remaining=seconds_remaining,
            bankroll=self.stats.bankroll,
            config=self.strategy_config,
        )
        skip_reason = get_trend_entry_skip_reason(
            btc_price=btc_price,
            opening_price=self._opening_price,
            seconds_remaining=seconds_remaining,
            config=self.strategy_config,
        )

        if skip_reason == "trend_move_too_large":
            self._trade_attempted = True
            move = abs(((btc_price - self._opening_price) / self._opening_price) * 100)
            print(
                f"  Skip: BTC move {move:.3f}% > "
                f"{self.strategy_config.trend_entry_skip_threshold_pct:.3f}% chase limit"
            )
        elif skip_reason == "entry_window_closed":
            self._trade_attempted = True

        # Store context for window-end no-trade signal logging
        self._last_tick_context = {
            "btc_price": btc_price,
            "up_price": self._cached_up,
            "down_price": self._cached_down,
            "seconds_remaining": seconds_remaining,
            "window_ts": self._current_window,
            "signal": signal_result,
            "skip_reason": skip_reason,
        }

        if signal_result:
            self._execute_trade(signal_result, seconds_remaining)

        if now - self._last_status_print >= 30:
            self._last_status_print = now
            delta = ((btc_price - self._opening_price) / self._opening_price * 100) if self._opening_price > 0 else 0
            d = "↑" if delta > 0 else "↓" if delta < 0 else "→"
            if self._traded:
                state = "HOLDING"
            elif skip_reason == "trend_below_threshold":
                state = (
                    f"TREND_WAIT "
                    f"({abs(delta):.3f}%<{self.strategy_config.trend_entry_threshold_pct:.3f}%)"
                )
            elif skip_reason == "entry_window_closed":
                state = "ENTRY_CLOSED"
            elif skip_reason == "trend_move_too_large":
                state = "SKIP_CHASE"
            else:
                state = "IDLE"
            n = len(self._recent_window_deltas)
            vol_label = f"{self._compute_realized_vol():.3f}({'r' if n >= 6 else f'fb,n={n}'})"
            print(
                f"  ⏱  T-{seconds_remaining:5.1f}s | "
                f"BTC ${btc_price:,.2f} {d}{abs(delta):.3f}% | "
                f"UP ${self._cached_up:.3f} DN ${self._cached_down:.3f} | "
                f"vol={vol_label} | P&L ${self.stats.total_pnl:+.2f} [{state}]"
            )

    # ── Active position management ──────────────────────────────────

    # ── Position monitoring (hold to resolution) ────────────────────

    def _check_daily_loss_limit(self, label: str = "trade") -> bool:
        if self._daily_loss_halted:
            print(f"  DAILY LOSS LIMIT - skipping {label} "
                  f"(balance ${self.stats.bankroll:.2f}, peak ${self._balance_peak:.2f})")
            return False

        self._balance_peak = max(self._balance_peak, self.stats.bankroll)
        retreat = self._balance_peak - self.stats.bankroll
        if retreat >= self._daily_loss_limit:
            self._daily_loss_halted = True
            msg = (
                f"DAILY BALANCE RETREAT LIMIT HIT: current balance ${self.stats.bankroll:.2f} "
                f"retreated ${retreat:.2f} from peak ${self._balance_peak:.2f} "
                f"(limit ${self._daily_loss_limit:.0f}) - stopping all trades for manual review"
            )
            print(f"\n  {msg}")
            self.telegram.status_update({"alert": msg})
            return False
        return True

    def _check_strategy_daily_loss_limit(self, key: str, label: str) -> bool:
        limit = self._strategy_daily_loss_limits.get(key, 0.0)
        if limit <= 0:
            return True
        strategy_pnl = self._strategy_pnl.get(key, 0.0)
        if self._strategy_loss_halted.get(key, False):
            print(
                f"  {label.upper()} LOSS LIMIT - skipping "
                f"(strategy P&L ${strategy_pnl:+.2f}, limit -${limit:.0f})"
            )
            return False
        if strategy_pnl <= -limit:
            self._strategy_loss_halted[key] = True
            msg = (
                f"{label} DAILY LOSS LIMIT HIT: ${strategy_pnl:+.2f} "
                f"(limit -${limit:.0f}) - stopping only this strategy"
            )
            print(f"\n  {msg}")
            self.telegram.status_update({"alert": msg})
            return False
        return True

    def _format_duration(self, seconds: float) -> str:
        seconds = max(0, int(seconds))
        hours, rem = divmod(seconds, 3600)
        minutes, secs = divmod(rem, 60)
        if hours:
            return f"{hours}h {minutes}m"
        if minutes:
            return f"{minutes}m {secs}s"
        return f"{secs}s"

    def _check_bot_cooldown(self, label: str = "trade") -> bool:
        if self._bot_cooldown_until <= 0:
            return True
        remaining = self._bot_cooldown_until - time.time()
        if remaining > 0:
            print(
                f"  BOT COOLING DOWN - skipping {label} "
                f"(remaining {self._format_duration(remaining)}, total P&L ${self.stats.total_pnl:+.2f})"
            )
            return False
        self._bot_cooldown_until = 0.0
        print("  Bot cooldown expired - trading can resume if other risk checks pass")
        return True

    def _check_strategy_cooldown(self, key: str, label: str) -> bool:
        cooldown_until = self._strategy_cooldown_until.get(key, 0.0)
        if cooldown_until <= 0:
            return True
        remaining = cooldown_until - time.time()
        if remaining > 0:
            print(
                f"  {label.upper()} COOLING DOWN - skipping "
                f"(remaining {self._format_duration(remaining)}, strategy P&L ${self._strategy_pnl.get(key, 0.0):+.2f})"
            )
            return False
        self._strategy_cooldown_until[key] = 0.0
        print(f"  {label} cooldown expired - strategy can resume if other risk checks pass")
        return True

    def _check_total_peak_drawdown(self):
        self._total_pnl_peak = max(self._total_pnl_peak, self.stats.total_pnl)
        if self._peak_loss_drop <= 0:
            return
        if self._bot_cooldown_until > time.time():
            return
        drawdown = self._total_pnl_peak - self.stats.total_pnl
        if drawdown < self._peak_loss_drop:
            return

        self._bot_cooldown_until = time.time() + self._cooling_down_period
        msg = (
            f"BOT PEAK DRAWDOWN COOLDOWN: total P&L ${self.stats.total_pnl:+.2f} "
            f"dropped ${drawdown:.2f} from peak ${self._total_pnl_peak:+.2f}. "
            f"Pausing all strategies for {self._format_duration(self._cooling_down_period)}."
        )
        print(f"\n  {msg}")
        self.telegram.status_update({"alert": msg})
        self._total_pnl_peak = self.stats.total_pnl

    def _check_strategy_peak_drawdown(self, key: str, label: str):
        strategy_pnl = self._strategy_pnl.get(key, 0.0)
        self._strategy_pnl_peaks[key] = max(self._strategy_pnl_peaks.get(key, 0.0), strategy_pnl)
        drop_limit = self._strategy_peak_loss_drops.get(key, 0.0)
        if drop_limit <= 0:
            return
        if self._strategy_cooldown_until.get(key, 0.0) > time.time():
            return
        drawdown = self._strategy_pnl_peaks[key] - strategy_pnl
        if drawdown < drop_limit:
            return

        period = self._strategy_cooling_down_periods.get(key, 5400.0)
        self._strategy_cooldown_until[key] = time.time() + period
        msg = (
            f"{label} PEAK DRAWDOWN COOLDOWN: strategy P&L ${strategy_pnl:+.2f} "
            f"dropped ${drawdown:.2f} from peak ${self._strategy_pnl_peaks[key]:+.2f}. "
            f"Pausing this strategy for {self._format_duration(period)}."
        )
        print(f"\n  {msg}")
        self.telegram.status_update({"alert": msg})
        self._strategy_pnl_peaks[key] = strategy_pnl

    def _check_risk_limits(self, key: str, label: str) -> bool:
        return (
            self._check_daily_loss_limit(label)
            and self._check_bot_cooldown(label)
            and self._check_strategy_daily_loss_limit(key, label)
            and self._check_strategy_cooldown(key, label)
        )

    def _update_strategy_pnl(self, key: str, label: str, profit: float):
        self._strategy_pnl[key] = self._strategy_pnl.get(key, 0.0) + profit
        self._check_strategy_daily_loss_limit(key, label)
        self._check_strategy_peak_drawdown(key, label)
        self._check_total_peak_drawdown()
        self._check_daily_loss_limit(label)

    def _record_cheap_result(self, profit: float):
        self._record_strategy_result("cheap", "Cheap scalp", profit)

    def _record_panic_result(self, profit: float):
        self._record_strategy_result("panic", "Panic rebound", profit)

    def _record_relative_result(self, profit: float):
        self._record_strategy_result("relative", "Relative overreaction", profit)

    def _record_strategy_result(self, key: str, label: str, profit: float):
        self.stats.total_trades += 1
        if profit > 0:
            self.stats.wins += 1
            self.stats.total_pnl += profit
            self.stats.hourly.record_result(profit, won=True)
        else:
            loss = abs(profit)
            self.stats.losses += 1
            self.stats.total_pnl -= loss
            self.stats.hourly.record_result(-loss, won=False)
        self._update_strategy_pnl(key, label, profit)
        self.telegram.strategy_result_alert(
            strategy=label,
            profit=profit,
            strategy_pnl=self._strategy_pnl.get(key, 0.0),
            total_pnl=self.stats.total_pnl,
        )

    def _record_result_stats_only(self, profit: float):
        self.stats.total_trades += 1
        if profit > 0:
            self.stats.wins += 1
            self.stats.total_pnl += profit
            self.stats.hourly.record_result(profit, won=True)
        else:
            loss = abs(profit)
            self.stats.losses += 1
            self.stats.total_pnl -= loss
            self.stats.hourly.record_result(-loss, won=False)

    def _record_trend_result(self, profit: float):
        self._record_result_stats_only(profit)
        self._update_strategy_pnl("trend", "Trend follow", profit)

    def _trailing_profit_exit_ready(
        self, label: str, pnl: float, threshold_reached: bool,
        retreat_pct: float, armed_attr: str, peak_attr: str,
    ) -> bool:
        if pnl <= 0:
            return False

        armed = getattr(self, armed_attr)
        peak = getattr(self, peak_attr)

        if not armed:
            if not threshold_reached:
                return False
            setattr(self, armed_attr, True)
            setattr(self, peak_attr, pnl)
            print(
                f"  {label} trailing profit armed: peak profit ${pnl:+.2f}, "
                f"retreat trigger {retreat_pct:.0%}"
            )
            return retreat_pct <= 0

        if pnl > peak:
            setattr(self, peak_attr, pnl)
            return False

        if peak <= 0:
            return False
        retreat = (peak - pnl) / peak
        if retreat >= retreat_pct:
            print(
                f"  {label} trailing profit retreat: peak ${peak:+.2f} -> "
                f"${pnl:+.2f} ({retreat:.0%})"
            )
            return True
        return False

    def _polymarket_winner_for_window(self, window_ts: int = None) -> str:
        window_ts = window_ts or self._current_window
        if window_ts <= 0:
            return ""
        if window_ts in self._winner_cache:
            return self._winner_cache[window_ts]
        winner = ""
        for attempt in range(3):
            winner = get_market_winner(self.period, window_ts) or ""
            if winner:
                break
            if attempt < 2:
                time.sleep(2)

        window_label = time.strftime("%H:%M", time.localtime(window_ts))
        if winner:
            print(f"  Polymarket final result: {winner} for {window_label} ({window_ts})")
            self._winner_cache[window_ts] = winner
        else:
            print(f"  Polymarket final result not available after 3 tries for {window_label} ({window_ts})")
        return winner

    def _dry_run_resolution_won(self, side: str, window_ts: int = None) -> Optional[bool]:
        window_ts = window_ts or self._current_window
        if window_ts == self._resolving_window_ts:
            winner = self._resolving_window_winner
        else:
            winner = self._polymarket_winner_for_window(window_ts)
        if not winner:
            return None
        return winner == side

    def _queue_dry_resolution(
        self, key: str, label: str, side: str, shares: float,
        cost: float, exit_revenue: float, window_ts: int,
    ):
        if any(item["key"] == key and item["window_ts"] == window_ts for item in self._dry_pending_resolutions):
            return
        self._dry_pending_resolutions.append({
            "key": key,
            "label": label,
            "side": side,
            "shares": shares,
            "cost": cost,
            "exit_revenue": exit_revenue,
            "window_ts": window_ts,
        })

    def _process_dry_pending_resolutions(self):
        if not self.dry_run or not self._dry_pending_resolutions:
            return
        remaining = []
        for item in self._dry_pending_resolutions:
            winner = self._polymarket_winner_for_window(item["window_ts"])
            if not winner:
                remaining.append(item)
                continue
            won = winner == item["side"]
            if won:
                revenue = item["shares"]
                profit = revenue - item["cost"]
                self.stats.bankroll += revenue
            else:
                profit = -(item["cost"] - item["exit_revenue"])
            self._record_strategy_result(item["key"], item["label"], profit)
            result = "WIN" if won else "LOSS"
            print(
                f"  {item['label']} dry-run pending resolved {result} "
                f"{profit:+.2f} via Polymarket final result"
            )
        self._dry_pending_resolutions = remaining

    def _manage_position(self, btc_price: float, seconds_remaining: float, now: float):
        """Monitor only — all trades hold to resolution. No stops.
        Tracker logs hold-period stats for future optimization.
        """
        if self._opening_price <= 0:
            return

        btc_delta_pct = ((btc_price - self._opening_price) / self._opening_price) * 100
        direction_prob = estimate_true_probability(btc_delta_pct, seconds_remaining)
        if btc_delta_pct == 0:
            our_prob = 0.50
        else:
            price_direction = "UP" if btc_delta_pct > 0 else "DOWN"
            our_prob = direction_prob if self._trade_side == price_direction else 1.0 - direction_prob

        # Throttled check
        if now - self._last_position_check < POSITION_CHECK_INTERVAL:
            if now - self._last_status_print >= 30:
                self._last_status_print = now
                d = "↑" if btc_delta_pct > 0 else "↓" if btc_delta_pct < 0 else "→"
                print(
                    f"  ⏱  T-{seconds_remaining:5.1f}s | "
                    f"BTC {d}{abs(btc_delta_pct):.3f}% | "
                    f"Prob: {our_prob:.2f} | "
                    f"P&L ${self.stats.total_pnl:+.2f} [HOLDING→RES]"
                )
            return

        self._last_position_check = now

        current_sell_price = self._get_live_sell_price(self._trade_token_id, our_prob)

        if current_sell_price <= 0:
            return

        self._last_sell_price_seen = current_sell_price

        # Track hold-period extremes
        self.tracker.update_hold_stats(our_prob, current_sell_price)

        current_value = self._trade_shares * current_sell_price
        unrealized_pnl = current_value - self._trade_cost
        return_pct = (current_sell_price - self._trade_price) / self._trade_price if self._trade_price > 0 else 0

        target_price = self._profit_target_price()
        can_sell_notional = self._trade_shares * current_sell_price >= 5.0
        threshold_reached = (
            current_sell_price >= target_price
            and unrealized_pnl >= self._min_profit_usd
        )

        if (threshold_reached or self._trend_trailing_armed) and can_sell_notional:
            sell_price = current_sell_price
            if not self.dry_run and self._confirm_ws_sell_price:
                probe = max(round(self._trade_shares * current_sell_price, 2), 1.0)
                confirmed = self.executor.get_market_price(self._trade_token_id, "SELL", probe)
                if confirmed > 0:
                    sell_price = confirmed
                    current_value = self._trade_shares * sell_price
                    unrealized_pnl = current_value - self._trade_cost
            threshold_reached = (
                sell_price >= target_price
                and unrealized_pnl >= self._min_profit_usd
            )
            if self._trailing_profit_exit_ready(
                "Trend follow", unrealized_pnl, threshold_reached,
                self._trend_profit_retreat_pct,
                "_trend_trailing_armed", "_trend_peak_profit",
            ):
                self._exit_position(sell_price, seconds_remaining, "trailing_profit")
                return

        if now - self._last_status_print < 5:
            return
        self._last_status_print = now

        # Status line
        d = "↑" if btc_delta_pct > 0 else "↓" if btc_delta_pct < 0 else "→"
        pnl_emoji = "📈" if unrealized_pnl > 0 else "📉"
        print(
            f"  {pnl_emoji} T-{seconds_remaining:5.1f}s | "
            f"BTC {d}{abs(btc_delta_pct):.3f}% | "
            f"Prob: {our_prob:.2f} | "
            f"Sell: ${current_sell_price:.3f} | "
            f"Target: ${target_price:.2f} | "
            f"PnL: ${unrealized_pnl:+.2f} ({return_pct:+.0%})"
        )

    # ── Execute exit (balance-verified, partial fill aware) ─────────

    def _exit_position(self, sell_price: float, seconds_remaining: float, reason: str):
        if self.dry_run:
            revenue = self._trade_shares * sell_price
            self._exited = True
            self._exit_revenue = revenue
            self.stats.bankroll += revenue
            profit = revenue - self._trade_cost
            print(f"  💰 EXIT ({reason}, paper): {self._trade_shares:.0f} shares @ "
                  f"${sell_price:.3f} = ${revenue:.2f} | Profit: ${profit:+.2f}")
            return

        result = self.executor.sell(
            token_id=self._trade_token_id,
            shares=self._trade_shares,
            price=sell_price,
        )

        if result.success:
            self._exit_revenue += result.amount_usd
            self._exit_shares_sold += result.shares
            self._residual_shares = result.shares_remaining
            self.stats.bankroll += result.amount_usd

            if result.status == PARTIAL and result.shares_remaining >= 1:
                # Partial fill: got some USDC back, still have shares
                print(f"  💰 EXIT ({reason}, partial): ~{result.shares:.0f} shares @ "
                      f"${result.price:.3f} = ${result.amount_usd:.2f} | "
                      f"~{result.shares_remaining:.0f} shares remaining → holding to resolution")
                # Update shares but keep original cost for clean P&L math
                self._trade_shares = result.shares_remaining
                # Mark exited — residual resolves at window close
                self._exited = True
            else:
                # Full fill (or residual < 1 share)
                self._exited = True
                profit = self._exit_revenue - self._trade_cost
                print(f"  💰 EXIT ({reason}): {result.shares:.0f} shares @ "
                      f"${result.price:.3f} = ${result.amount_usd:.2f} | "
                      f"Profit: ${profit:+.2f}")
        elif "hold to resolution" in result.error:
            # Below $5 minimum — can't sell, hold to resolution
            notional = self._trade_shares * sell_price
            print(f"  📌 Can't sell: ${notional:.2f} below $5 minimum — holding to resolution")
            self._exit_gave_up = True  # Skip further exit attempts
        else:
            self._exit_retries += 1
            if self._exit_retries >= MAX_EXIT_RETRIES:
                print(f"  ❌ Exit failed {MAX_EXIT_RETRIES} times ({reason}) — "
                      f"holding to resolution")
                self._exit_gave_up = True
            else:
                print(f"  ⚠️  Exit failed ({reason}, attempt "
                      f"{self._exit_retries}/{MAX_EXIT_RETRIES}): {result.error}")
                self._last_position_check = time.time() + EXIT_RETRY_COOLDOWN - POSITION_CHECK_INTERVAL

    # ── Window management ───────────────────────────────────────────

    def _on_new_window(self, window_ts: int, closing_btc_price: float = 0.0):
        self._process_dry_pending_resolutions()
        if self._current_window > 0:
            self._resolving_window_ts = self._current_window
            self._resolving_window_winner = (
                self._polymarket_winner_for_window(self._current_window)
                if self.dry_run else ""
            )
            # Resolve any pending phantom sell from the previous window.
            # Must run before trade state is reset below.
            # Balance is fetched once here and reused by the sync below.
            if self._pending_phantom:
                pp = self._pending_phantom
                if not self.dry_run and self.executor._initialized:
                    real_bal = self.executor.get_balance()
                    if real_bal > 0:
                        balance_increase = max(0.0, real_bal - pp["pre_sell_balance"])
                        if balance_increase > pp["expected_revenue"] * 0.50:
                            # Settlement landed — it was a real win
                            profit = balance_increase - pp["cost"]
                            self._record_trend_result(profit)
                            self.stats.bankroll = real_bal
                            self._last_real_balance = real_bal
                            print(f"  ✅ Phantom resolved: WIN +${profit:.2f} [phantom_resolved] | "
                                  f"P&L: ${self.stats.total_pnl:+.2f} | Bank: ${self.stats.bankroll:.2f}")
                            self.telegram.strategy_result_alert(
                                strategy="Trend follow",
                                profit=profit,
                                strategy_pnl=self._strategy_pnl.get("trend", 0.0),
                                total_pnl=self.stats.total_pnl,
                            )
                            btc_price, _ = self.price_feed.get_price()
                            self.tracker.log_trade_resolve(
                                btc_final_price=btc_price,
                                opening_price=pp["opening_price"],
                                won=True,
                                profit=profit,
                                exit_revenue=pp["exit_revenue"],
                                resolution_method="phantom_resolved",
                                claim_result="phantom_resolved",
                            )
                        else:
                            # Balance still hasn't moved — genuine loss
                            net_loss = pp["cost"] - pp["exit_revenue"]
                            profit = -net_loss
                            self._record_trend_result(profit)
                            self.stats.bankroll = real_bal
                            self._last_real_balance = real_bal
                            print(f"  ❌ Phantom confirmed: LOSS -${net_loss:.2f} [phantom_confirmed] | "
                                  f"P&L: ${self.stats.total_pnl:+.2f} | Bank: ${self.stats.bankroll:.2f}")
                            self.telegram.strategy_result_alert(
                                strategy="Trend follow",
                                profit=profit,
                                strategy_pnl=self._strategy_pnl.get("trend", 0.0),
                                total_pnl=self.stats.total_pnl,
                            )
                            btc_price, _ = self.price_feed.get_price()
                            self.tracker.log_trade_resolve(
                                btc_final_price=btc_price,
                                opening_price=pp["opening_price"],
                                won=False,
                                profit=-net_loss,
                                exit_revenue=pp["exit_revenue"],
                                resolution_method="phantom_confirmed",
                                claim_result="phantom_confirmed",
                            )
                        self._pending_phantom = {}
                else:
                    # Dry run or executor not ready — treat as loss
                    net_loss = pp["cost"] - pp["exit_revenue"]
                    self._record_trend_result(-net_loss)
                    self._pending_phantom = {}

            # Record closing delta for rolling vol calculation
            if self._opening_price > 0 and closing_btc_price > 0:
                closing_delta = abs((closing_btc_price - self._opening_price) / self._opening_price * 100)
                self._recent_window_deltas.append(closing_delta)
                if len(self._recent_window_deltas) > self._rolling_vol_windows:
                    self._recent_window_deltas.pop(0)
            # Detect pending buy that settled after our verification timeout
            if self._pending_buy_side and not self._traded:
                if not self.dry_run and self.executor._initialized:
                    self._pending_buy_last_check = 0.0
                    self._check_pending_buy_fill(time.time())
                if self._pending_buy_side and not self._traded and not self.dry_run and self.executor._initialized:
                    real_bal = self.executor.get_balance()
                    if real_bal > 0 and self._balance_before_buy > 0:
                        spent = self._balance_before_buy - real_bal
                        if spent > 1.0:
                            # The buy DID go through — retroactively track it
                            est_shares = spent / self._pending_buy_price if self._pending_buy_price > 0 else 0
                            print(f"\n  👻 LATE FILL: balance dropped ${spent:.2f} since buy attempt")
                            print(f"     Retroactively tracking: ~{est_shares:.0f} shares "
                                  f"{self._pending_buy_side} @ ${self._pending_buy_price:.3f}")

                            self._traded = True
                            self._trade_side = self._pending_buy_side
                            self._trade_price = self._pending_buy_price
                            self._trade_cost = spent
                            self._trade_shares = est_shares
                            self._trade_token_id = self._pending_buy_token_id
                            self.stats.bankroll = real_bal
                            self._last_real_balance = real_bal
                            self.stats.hourly.record_trade(
                                self._pending_buy_edge, self._pending_buy_delta)

            if self._cheap_traded and not self._cheap_exited:
                self._resolve_cheap_scalp(closing_btc_price)
            if self._relative_pending_side and not self._relative_traded:
                if not self.dry_run and self.executor._initialized:
                    self._relative_pending_last_check = 0.0
                    self._check_pending_relative_buy(time.time())
            if self._relative_traded and not self._relative_exited:
                self._resolve_relative_reaction(closing_btc_price)
            if self._panic_pending_side and not self._panic_traded:
                if not self.dry_run and self.executor._initialized:
                    self._panic_pending_last_check = 0.0
                    self._check_pending_panic_buy(time.time())
            if self._panic_traded and not self._panic_exited:
                self._resolve_panic_rebound(closing_btc_price)

            self.stats.hourly.record_window(self._traded)
            if self._traded:
                self._resolve_previous_trade()
            elif self._last_tick_context:
                # Log the no-trade signal for this window using last tick state
                ctx = self._last_tick_context
                skip_reason = ctx.get("skip_reason") or get_trend_entry_skip_reason(
                    btc_price=ctx["btc_price"],
                    opening_price=self._opening_price,
                    seconds_remaining=ctx["seconds_remaining"],
                    config=self.strategy_config,
                )
                sig = ctx.get("signal")
                self.tracker.log_signal(
                    window_ts=ctx["window_ts"],
                    btc_price=ctx["btc_price"],
                    opening_price=self._opening_price,
                    up_price=ctx["up_price"],
                    down_price=ctx["down_price"],
                    seconds_remaining=ctx["seconds_remaining"],
                    side=sig.side if sig else "",
                    true_prob=sig.true_prob if sig else 0.0,
                    market_price=sig.market_price if sig else 0.0,
                    edge=sig.edge if sig else 0.0,
                    kelly_size=sig.kelly_size if sig else 0.0,
                    action="no_signal",
                    skip_reason=skip_reason,
                )

            # Sync real balance at window boundary (catches any drift)
            if not self.dry_run and self.executor._initialized:
                real_bal = self.executor.get_balance()
                if real_bal > 0:
                    drift = abs(real_bal - self.stats.bankroll)
                    if drift > 0.50:
                        print(f"  🔄 Balance sync: ${self.stats.bankroll:.2f} → "
                              f"${real_bal:.2f} (drift ${drift:.2f})")
                    self.stats.bankroll = real_bal
                    self._last_real_balance = real_bal

            self._resolving_window_ts = 0
            self._resolving_window_winner = ""

        self._current_window = window_ts
        self._current_market = None
        self._market_subscription_time = 0.0
        self._opening_price = 0.0
        self._traded = False
        self._trade_attempted = False
        self._exited = False
        self._exit_revenue = 0.0
        self._exit_shares_sold = 0.0
        self._residual_shares = 0.0
        self._last_position_check = 0.0
        self._exit_retries = 0
        self._exit_gave_up = False
        self._last_sell_price_seen = 0.0
        self._trend_trailing_armed = False
        self._trend_peak_profit = 0.0
        self._cheap_traded = False
        self._cheap_trade_attempted = False
        self._cheap_exited = False
        self._cheap_side = ""
        self._cheap_token_id = ""
        self._cheap_price = 0.0
        self._cheap_cost = 0.0
        self._cheap_shares = 0.0
        self._cheap_exit_revenue = 0.0
        self._cheap_last_check = 0.0
        self._cheap_trailing_armed = False
        self._cheap_peak_profit = 0.0
        self._clear_pending_cheap_buy()
        self._relative_traded = False
        self._relative_trade_attempted = False
        self._relative_exited = False
        self._relative_side = ""
        self._relative_token_id = ""
        self._relative_price = 0.0
        self._relative_cost = 0.0
        self._relative_shares = 0.0
        self._relative_exit_revenue = 0.0
        self._relative_last_check = 0.0
        self._relative_trailing_armed = False
        self._relative_peak_profit = 0.0
        self._realtime_history = []
        self._clear_pending_relative_buy()
        self._panic_traded = False
        self._panic_trade_attempted = False
        self._panic_exited = False
        self._panic_side = ""
        self._panic_token_id = ""
        self._panic_price = 0.0
        self._panic_cost = 0.0
        self._panic_shares = 0.0
        self._panic_exit_revenue = 0.0
        self._panic_last_check = 0.0
        self._panic_trailing_armed = False
        self._panic_peak_profit = 0.0
        self._panic_price_history = {"UP": [], "DOWN": []}
        self._clear_pending_panic_buy()
        self._cached_up = 0.50
        self._cached_down = 0.50
        self._price_last_fetched = 0.0
        self._pending_buy_side = ""
        self._pending_buy_price = 0.0
        self._pending_buy_amount = 0.0
        self._pending_buy_shares = 0.0
        self._pending_buy_token_id = ""
        self._pending_buy_edge = 0.0
        self._pending_buy_delta = 0.0
        self._pending_buy_order_id = ""
        self._pending_buy_last_check = 0.0
        self._balance_before_buy = 0.0

        t = time.strftime("%H:%M:%S", time.localtime(window_ts))
        print(f"\n{'─' * 55}")
        print(f"🕐 {t} | Trades: {self.stats.total_trades} | "
              f"W/L: {self.stats.wins}/{self.stats.losses} | "
              f"P&L: ${self.stats.total_pnl:+.2f}")
        print(f"{'─' * 55}")

        # Circuit breaker auto-recovery: ping CLOB each new window
        if self._clob_halted and not self.dry_run and self.executor._initialized:
            try:
                self.executor.client.get_ok()
                self._clob_halted = False
                self._consecutive_buy_failures = 0
                print(f"  ✅ CLOB recovered (health check OK) — resuming trades")
            except Exception:
                print(f"  🔌 CLOB health check still failing — staying halted")

    # ── Polymarket live prices ──────────────────────────────────────

    def _clear_pending_buy(self):
        self._pending_buy_side = ""
        self._pending_buy_price = 0.0
        self._pending_buy_amount = 0.0
        self._pending_buy_shares = 0.0
        self._pending_buy_token_id = ""
        self._pending_buy_edge = 0.0
        self._pending_buy_delta = 0.0
        self._pending_buy_order_id = ""
        self._pending_buy_last_check = 0.0
        self._balance_before_buy = 0.0

    def _activate_pending_buy(
        self, price: float, amount: float, shares: float,
        source: str, balance_already_synced: bool = False,
    ):
        if not self._pending_buy_side or self._traded:
            return
        self._traded = True
        self._trade_side = self._pending_buy_side
        self._trade_price = price
        self._trade_cost = amount
        self._trade_shares = shares
        self._trade_token_id = self._pending_buy_token_id
        self._trend_trailing_armed = False
        self._trend_peak_profit = 0.0
        if not balance_already_synced:
            self.stats.bankroll -= amount
        self.stats.hourly.record_trade(self._pending_buy_edge, self._pending_buy_delta)
        print(
            f"  ✅ Pending buy verified [{source}]: ~{shares:.0f} shares "
            f"{self._trade_side} @ ${price:.3f} = ${amount:.2f}"
        )
        self.tracker.log_trade_entry(
            window_ts=self._current_window,
            side=self._trade_side,
            entry_price=price,
            entry_shares=shares,
            entry_cost=amount,
            edge=self._pending_buy_edge,
            prob=1.0,
            btc_delta=self._pending_buy_delta,
            seconds_remaining=0.0,
            entry_delta_pct=self._pending_buy_delta,
            entry_seconds_remaining=0.0,
        )
        self._clear_pending_buy()

    def _check_pending_buy_fill(self, now: float):
        if not self._pending_buy_side or self._traded or self.dry_run:
            return
        if now - self._pending_buy_last_check < 2.0:
            return
        self._pending_buy_last_check = now

        if self._pending_buy_order_id and self.executor._initialized:
            matched = self.executor.check_buy_fill(
                self._pending_buy_order_id,
                self._pending_buy_price,
            )
            if matched:
                self._activate_pending_buy(
                    price=matched[0],
                    amount=matched[1],
                    shares=matched[2],
                    source="order_api",
                )
                return

        if self.executor._initialized and self._balance_before_buy > 0:
            real_bal = self.executor.get_balance()
            spent = self._balance_before_buy - real_bal if real_bal > 0 else 0.0
            if spent > 1.0:
                shares = spent / self._pending_buy_price if self._pending_buy_price > 0 else self._pending_buy_shares
                self.stats.bankroll = real_bal
                self._last_real_balance = real_bal
                self._activate_pending_buy(
                    price=self._pending_buy_price,
                    amount=spent,
                    shares=shares,
                    source="balance",
                    balance_already_synced=True,
                )

    def _ensure_market_subscription(self):
        if self._current_market:
            return self._current_market

        market = get_current_market(self.period)
        if not market:
            return None

        self._current_market = market
        self._market_subscription_time = time.time()
        if self._poly_ws_enabled:
            self.poly_feed.subscribe(
                [market.token_id_up, market.token_id_down],
                labels={market.token_id_up: "UP", market.token_id_down: "DOWN"},
            )
        return market

    def _live_token_price(self, token_id: str):
        if not self._poly_ws_enabled or not token_id:
            return None
        price = self.poly_feed.get_price(token_id)
        if not price:
            return None
        if self._market_subscription_time > 0 and price.timestamp < self._market_subscription_time:
            return None
        if time.time() - price.timestamp > 10:
            return None
        if price.best_bid > 0 and price.best_ask > 0 and price.best_bid >= price.best_ask:
            return None
        return price

    def _refresh_cached_ws_prices(self):
        market = self._current_market
        if not market:
            return
        up = self._live_token_price(market.token_id_up)
        down = self._live_token_price(market.token_id_down)
        if up:
            up_price = up.best_ask or up.last_trade or up.mid
            if up_price > 0:
                self._cached_up = up_price
        if down:
            down_price = down.best_ask or down.last_trade or down.mid
            if down_price > 0:
                self._cached_down = down_price

    def _update_realtime_history(self, now: float, btc_price: float):
        market = self._current_market
        if not market or btc_price <= 0:
            return
        up = self._live_token_price(market.token_id_up)
        down = self._live_token_price(market.token_id_down)
        up_ask = up.best_ask if up else 0.0
        up_bid = up.best_bid if up else 0.0
        down_ask = down.best_ask if down else 0.0
        down_bid = down.best_bid if down else 0.0
        if up_ask <= 0 and down_ask <= 0:
            return
        self._realtime_history.append({
            "ts": now,
            "btc": btc_price,
            "UP_ask": up_ask,
            "UP_bid": up_bid,
            "DOWN_ask": down_ask,
            "DOWN_bid": down_bid,
        })
        cutoff = now - max(self._realtime_to_save, 1)
        while self._realtime_history and self._realtime_history[0]["ts"] < cutoff:
            self._realtime_history.pop(0)

    def _history_row_at_or_before(self, target_ts: float):
        candidate = None
        for row in self._realtime_history:
            if row["ts"] <= target_ts:
                candidate = row
            else:
                break
        return candidate or (self._realtime_history[0] if self._realtime_history else None)

    def _relative_expected_poly_move(self, side: str, btc_delta_pct: float, current_ts: float):
        actuals = []
        for row in self._realtime_history[:-1]:
            base = self._history_row_at_or_before(row["ts"] - self._relative_lookback_seconds)
            if not base or base is row or base["btc"] <= 0:
                continue
            base_price = base.get(f"{side}_ask", 0.0)
            row_price = row.get(f"{side}_ask", 0.0)
            if base_price <= 0 or row_price <= 0:
                continue
            hist_btc_delta = ((row["btc"] - base["btc"]) / base["btc"]) * 100
            if abs(hist_btc_delta) < self._relative_min_btc_move_pct:
                continue
            if hist_btc_delta * btc_delta_pct <= 0:
                continue
            actuals.append(row_price - base_price)

        if actuals:
            return statistics.median(actuals)

        # Fallback until enough same-direction history exists in the saved buffer.
        direction = 1 if btc_delta_pct > 0 else -1
        side_sign = 1 if side == "UP" else -1
        return direction * side_sign * 0.20

    def _relative_reaction_signal(self):
        if len(self._realtime_history) < 3 or not self._current_market:
            return None
        now = time.time()
        current = self._realtime_history[-1]
        baseline = self._history_row_at_or_before(now - self._relative_lookback_seconds)
        if not baseline or baseline["btc"] <= 0:
            return None
        btc_delta_pct = ((current["btc"] - baseline["btc"]) / baseline["btc"]) * 100
        if abs(btc_delta_pct) < self._relative_min_btc_move_pct:
            return None

        candidates = []
        for side, token_id in (("UP", self._current_market.token_id_up), ("DOWN", self._current_market.token_id_down)):
            base_ask = baseline.get(f"{side}_ask", 0.0)
            current_ask = current.get(f"{side}_ask", 0.0)
            if base_ask <= 0 or current_ask <= 0:
                continue
            actual_move = current_ask - base_ask
            expected_move = self._relative_expected_poly_move(side, btc_delta_pct, now)
            gap = actual_move - expected_move
            if gap <= -self._relative_overreaction_price:
                candidates.append((abs(gap), side, token_id, current_ask, actual_move, expected_move, btc_delta_pct))
        if not candidates:
            return None
        candidates.sort(reverse=True, key=lambda item: item[0])
        return candidates[0]

    def _get_live_sell_price(self, token_id: str, fallback_prob: float) -> float:
        price = self._live_token_price(token_id)
        if price and price.best_bid > 0:
            return price.best_bid
        if self.dry_run:
            return round(max(fallback_prob, 0.01), 2)
        sell_probe = round(self._trade_shares * self._trade_price, 2)
        return self.executor.get_market_price(token_id, "SELL", max(sell_probe, 1.0))

    def _profit_target_for_price(self, entry_price: float) -> float:
        pct_target = entry_price * (1.0 + self._min_profit_pct)
        fixed_target = self._take_profit_price if self._take_profit_price > 0 else 0.0
        return round(max(pct_target, fixed_target), 2)

    def _profit_target_price(self) -> float:
        return self._profit_target_for_price(self._trade_price)

    def _cheap_profit_target_price(self) -> float:
        pct_target = self._cheap_price * (1.0 + self._cheap_min_profit_pct)
        fixed_target = self._cheap_take_profit_price if self._cheap_take_profit_price > 0 else 0.0
        return round(max(pct_target, fixed_target), 2)

    def _panic_profit_target_price(self) -> float:
        pct_target = self._panic_price * (1.0 + self._panic_min_profit_pct)
        fixed_target = self._panic_take_profit_price if self._panic_take_profit_price > 0 else 0.0
        return round(max(pct_target, fixed_target), 2)

    def _update_panic_price_history(self, now: float):
        market = self._current_market
        if not market:
            return
        for side, token_id in (("UP", market.token_id_up), ("DOWN", market.token_id_down)):
            price = self._live_token_price(token_id)
            ask = price.best_ask if price else 0.0
            bid = price.best_bid if price else 0.0
            if ask <= 0:
                continue
            history = self._panic_price_history[side]
            history.append((now, ask, bid))
            cutoff = now - max(self._panic_lookback_seconds, 1)
            while history and history[0][0] < cutoff:
                history.pop(0)

    def _cheap_scalp_candidate(self):
        market = self._current_market
        if not market:
            return None
        candidates = []
        for side, token_id in (("UP", market.token_id_up), ("DOWN", market.token_id_down)):
            price = self._live_token_price(token_id)
            ask = price.best_ask if price else 0.0
            if ask > 0 and ask <= self._cheap_entry_price:
                candidates.append((ask, side, token_id))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0]

    def _clear_pending_cheap_buy(self):
        self._cheap_pending_side = ""
        self._cheap_pending_token_id = ""
        self._cheap_pending_order_id = ""
        self._cheap_pending_price = 0.0
        self._cheap_pending_amount = 0.0
        self._cheap_pending_shares = 0.0
        self._cheap_pending_balance_before = 0.0
        self._cheap_pending_last_check = 0.0

    def _activate_cheap_buy(self, side: str, token_id: str, price: float, amount: float, shares: float):
        self._cheap_traded = True
        self._cheap_side = side
        self._cheap_token_id = token_id
        self._cheap_price = price
        self._cheap_cost = amount
        self._cheap_shares = shares
        self._cheap_trailing_armed = False
        self._cheap_peak_profit = 0.0
        self.stats.bankroll -= amount
        self.stats.hourly.record_trade(0.0, 0.0)
        self._clear_pending_cheap_buy()
        print(
            f"  Cheap scalp bought: {shares:.0f} {side} "
            f"@ ${price:.3f} = ${amount:.2f}"
        )
        self.telegram.strategy_trade_alert(
            strategy="Cheap scalp",
            side=side,
            price=price,
            amount=amount,
            market_slug=f"btc-updown-{self.period}m-{self._current_window}",
            dry_run=self.dry_run,
            strategy_pnl=self._strategy_pnl.get("cheap", 0.0),
            total_pnl=self.stats.total_pnl,
        )

    def _check_pending_cheap_buy(self, now: float):
        if not self._cheap_pending_side or self._cheap_traded or self.dry_run:
            return
        if now - self._cheap_pending_last_check < 2.0:
            return
        self._cheap_pending_last_check = now

        if self._cheap_pending_order_id and self.executor._initialized:
            matched = self.executor.check_buy_fill(
                self._cheap_pending_order_id,
                self._cheap_pending_price,
            )
            if matched:
                self._activate_cheap_buy(
                    self._cheap_pending_side,
                    self._cheap_pending_token_id,
                    matched[0],
                    matched[1],
                    matched[2],
                )
                return

        if self.executor._initialized and self._cheap_pending_balance_before > 0:
            real_bal = self.executor.get_balance()
            spent = self._cheap_pending_balance_before - real_bal if real_bal > 0 else 0.0
            if spent > 1.0:
                shares = spent / self._cheap_pending_price if self._cheap_pending_price > 0 else self._cheap_pending_shares
                self.stats.bankroll = real_bal + spent
                self._last_real_balance = real_bal
                self._activate_cheap_buy(
                    self._cheap_pending_side,
                    self._cheap_pending_token_id,
                    self._cheap_pending_price,
                    spent,
                    shares,
                )

    def _try_cheap_scalp_entry(self, seconds_remaining: float):
        if not self._cheap_scalp_enabled:
            return
        if not self._check_risk_limits("cheap", "cheap scalp"):
            self._cheap_trade_attempted = True
            return
        if not self.dry_run and not self.executor._initialized:
            return
        if not self._current_market:
            return
        elapsed = PERIOD_SECONDS[self.period] - seconds_remaining
        if elapsed < 0:
            return
        if elapsed > self._cheap_entry_window_seconds:
            self._cheap_trade_attempted = True
            return

        candidate = self._cheap_scalp_candidate()
        if not candidate:
            return

        ws_ask, side, token_id = candidate
        trade_amount = round(min(self._cheap_trade_amount, self.strategy_config.max_bet, self.stats.bankroll), 2)
        if trade_amount < self.strategy_config.min_bet:
            return

        actual_price = ws_ask if self.dry_run else self.executor.get_market_price(token_id, "BUY", trade_amount)
        if actual_price <= 0:
            return

        cap_price = min(self._cheap_entry_price, max_buy_price())
        if actual_price > cap_price:
            print(
                f"  Cheap scalp skip {side}: WS ask ${ws_ask:.3f}, "
                f"executable ${actual_price:.3f} > cap ${cap_price:.2f}"
            )
            return

        print(
            f"\n  Cheap scalp {side}: WS ask ${ws_ask:.3f}, "
            f"executable ${actual_price:.3f}, amount ${trade_amount:.2f}"
        )
        self._cheap_trade_attempted = True
        result = self.executor.buy(token_id=token_id, amount_usd=trade_amount, price=actual_price)
        if not result.success:
            if result.error == "UNVERIFIED_BUY":
                self._cheap_pending_side = side
                self._cheap_pending_token_id = token_id
                self._cheap_pending_order_id = result.order_id
                self._cheap_pending_price = result.price
                self._cheap_pending_amount = result.amount_usd
                self._cheap_pending_shares = result.shares
                self._cheap_pending_balance_before = self.stats.bankroll
                self._cheap_pending_last_check = 0.0
                print("  Cheap scalp buy unverified - will poll order API and balance")
                return
            print(f"  Cheap scalp buy failed: {result.error}")
            return

        self._activate_cheap_buy(side, token_id, result.price, result.amount_usd, result.shares)

    def _manage_cheap_scalp(self, seconds_remaining: float, now: float):
        if not self._cheap_traded or self._cheap_exited:
            return
        if now - self._cheap_last_check < POSITION_CHECK_INTERVAL:
            return
        self._cheap_last_check = now

        live = self._live_token_price(self._cheap_token_id)
        sell_price = live.best_bid if live and live.best_bid > 0 else 0.0
        if sell_price <= 0:
            if self.executor._initialized and not self.dry_run:
                probe = max(round(self._cheap_shares * self._cheap_price, 2), 1.0)
                sell_price = self.executor.get_market_price(self._cheap_token_id, "SELL", probe)
        if sell_price <= 0:
            return

        target = self._cheap_profit_target_price()
        pnl = self._cheap_shares * sell_price - self._cheap_cost
        threshold_reached = sell_price >= target and pnl >= self._cheap_min_profit_usd
        if not threshold_reached and not self._cheap_trailing_armed:
            return

        if self._cheap_shares * sell_price < 5.0:
            return

        if not self.dry_run and self._confirm_ws_sell_price:
            probe = max(round(self._cheap_shares * sell_price, 2), 1.0)
            confirmed = self.executor.get_market_price(self._cheap_token_id, "SELL", probe)
            if confirmed > 0:
                sell_price = confirmed
                pnl = self._cheap_shares * sell_price - self._cheap_cost
        threshold_reached = sell_price >= target and pnl >= self._cheap_min_profit_usd
        if not self._trailing_profit_exit_ready(
            "Cheap scalp", pnl, threshold_reached,
            self._cheap_profit_retreat_pct,
            "_cheap_trailing_armed", "_cheap_peak_profit",
        ):
            return

        result = self.executor.sell(self._cheap_token_id, self._cheap_shares, price=sell_price)
        if result.success:
            self._cheap_exited = True
            self._cheap_exit_revenue += result.amount_usd
            self.stats.bankroll += result.amount_usd
            profit = self._cheap_exit_revenue - self._cheap_cost
            self._record_cheap_result(profit)
            print(
                f"  Cheap scalp exit: {result.shares:.0f} {self._cheap_side} "
                f"@ ${result.price:.3f} = ${result.amount_usd:.2f} | Profit ${profit:+.2f}"
            )

    def _resolve_cheap_scalp(self, closing_btc_price: float):
        if not self._cheap_traded or self._cheap_exited:
            return
        if self.dry_run:
            won = self._dry_run_resolution_won(self._cheap_side, self._current_window)
            if won is None:
                print("  Cheap scalp dry-run resolution pending Polymarket final result")
                self._queue_dry_resolution(
                    "cheap", "Cheap scalp", self._cheap_side,
                    self._cheap_shares, self._cheap_cost,
                    self._cheap_exit_revenue, self._current_window,
                )
                self._cheap_exited = True
                return
        else:
            won = False
            if self._opening_price > 0 and closing_btc_price > 0:
                won = (closing_btc_price >= self._opening_price) == (self._cheap_side == "UP")

        if won:
            if self.dry_run:
                revenue = self._cheap_shares
            else:
                claim = self.executor.sell(self._cheap_token_id, self._cheap_shares, price=0.99)
                if not claim.success:
                    print(f"  Cheap scalp claim not verified: {claim.error} - not booking win yet")
                    return
                revenue = claim.amount_usd
            profit = revenue - self._cheap_cost
            self.stats.bankroll += revenue
            self._record_cheap_result(profit)
            print(f"  Cheap scalp resolved WIN +${profit:.2f}")
        else:
            loss = self._cheap_cost - self._cheap_exit_revenue
            self._record_cheap_result(-loss)
            print(f"  Cheap scalp resolved LOSS -${loss:.2f}")
        self._cheap_exited = True

    def _relative_profit_target_price(self) -> float:
        pct_target = self._relative_price * (1.0 + self._relative_min_profit_pct)
        fixed_target = self._relative_take_profit_price if self._relative_take_profit_price > 0 else 0.0
        return round(max(pct_target, fixed_target), 2)

    def _clear_pending_relative_buy(self):
        self._relative_pending_side = ""
        self._relative_pending_token_id = ""
        self._relative_pending_order_id = ""
        self._relative_pending_price = 0.0
        self._relative_pending_amount = 0.0
        self._relative_pending_shares = 0.0
        self._relative_pending_balance_before = 0.0
        self._relative_pending_last_check = 0.0

    def _activate_relative_buy(self, side: str, token_id: str, price: float, amount: float, shares: float):
        self._relative_traded = True
        self._relative_side = side
        self._relative_token_id = token_id
        self._relative_price = price
        self._relative_cost = amount
        self._relative_shares = shares
        self._relative_trailing_armed = False
        self._relative_peak_profit = 0.0
        self.stats.bankroll -= amount
        self.stats.hourly.record_trade(0.0, 0.0)
        self._clear_pending_relative_buy()
        print(f"  Relative overreaction bought: {shares:.0f} {side} @ ${price:.3f} = ${amount:.2f}")
        self.telegram.strategy_trade_alert(
            strategy="Relative overreaction",
            side=side,
            price=price,
            amount=amount,
            market_slug=f"btc-updown-{self.period}m-{self._current_window}",
            dry_run=self.dry_run,
            strategy_pnl=self._strategy_pnl.get("relative", 0.0),
            total_pnl=self.stats.total_pnl,
        )

    def _check_pending_relative_buy(self, now: float):
        if not self._relative_pending_side or self._relative_traded or self.dry_run:
            return
        if now - self._relative_pending_last_check < 2.0:
            return
        self._relative_pending_last_check = now

        if self._relative_pending_order_id and self.executor._initialized:
            matched = self.executor.check_buy_fill(
                self._relative_pending_order_id,
                self._relative_pending_price,
            )
            if matched:
                self._activate_relative_buy(
                    self._relative_pending_side,
                    self._relative_pending_token_id,
                    matched[0],
                    matched[1],
                    matched[2],
                )
                return

        if self.executor._initialized and self._relative_pending_balance_before > 0:
            real_bal = self.executor.get_balance()
            spent = self._relative_pending_balance_before - real_bal if real_bal > 0 else 0.0
            if spent > 1.0:
                shares = spent / self._relative_pending_price if self._relative_pending_price > 0 else self._relative_pending_shares
                self.stats.bankroll = real_bal + spent
                self._last_real_balance = real_bal
                self._activate_relative_buy(
                    self._relative_pending_side,
                    self._relative_pending_token_id,
                    self._relative_pending_price,
                    spent,
                    shares,
                )

    def _try_relative_reaction_entry(self, seconds_remaining: float):
        if not self._relative_enabled:
            return
        if not self._check_risk_limits("relative", "relative overreaction"):
            self._relative_trade_attempted = True
            return
        if not self.dry_run and not self.executor._initialized:
            return
        elapsed = PERIOD_SECONDS[self.period] - seconds_remaining
        if elapsed < 0:
            return
        if elapsed > self._relative_entry_window_seconds:
            self._relative_trade_attempted = True
            return
        signal = self._relative_reaction_signal()
        if not signal:
            return
        gap_abs, side, token_id, ws_ask, actual_move, expected_move, btc_delta_pct = signal
        trade_amount = round(min(self._relative_trade_amount, self.strategy_config.max_bet, self.stats.bankroll), 2)
        if trade_amount < self.strategy_config.min_bet:
            return
        actual_price = ws_ask if self.dry_run else self.executor.get_market_price(token_id, "BUY", trade_amount)
        if actual_price <= 0:
            return
        cap_price = min(self._relative_max_buy_price, max_buy_price())
        if actual_price < self._relative_min_buy_price:
            print(
                f"  Relative skip {side}: executable ${actual_price:.3f} "
                f"< min ${self._relative_min_buy_price:.2f}"
            )
            return
        if actual_price > cap_price:
            print(f"  Relative skip {side}: executable ${actual_price:.3f} > cap ${cap_price:.2f}")
            return

        print(
            f"\n  Relative overreaction {side}: BTC {btc_delta_pct:+.3f}% | "
            f"poly actual {actual_move:+.3f}, expected {expected_move:+.3f}, gap -${gap_abs:.3f}"
        )
        self._relative_trade_attempted = True
        result = self.executor.buy(token_id=token_id, amount_usd=trade_amount, price=actual_price)
        if not result.success:
            if result.error == "UNVERIFIED_BUY":
                self._relative_pending_side = side
                self._relative_pending_token_id = token_id
                self._relative_pending_order_id = result.order_id
                self._relative_pending_price = result.price
                self._relative_pending_amount = result.amount_usd
                self._relative_pending_shares = result.shares
                self._relative_pending_balance_before = self.stats.bankroll
                self._relative_pending_last_check = 0.0
                print("  Relative overreaction buy unverified - will poll order API and balance")
                return
            print(f"  Relative overreaction buy failed: {result.error}")
            return
        self._activate_relative_buy(side, token_id, result.price, result.amount_usd, result.shares)

    def _relative_is_overbought_exit(self) -> bool:
        if not self._relative_traded or len(self._realtime_history) < 3:
            return False
        now = time.time()
        current = self._realtime_history[-1]
        baseline = self._history_row_at_or_before(now - self._relative_lookback_seconds)
        if not baseline or baseline["btc"] <= 0:
            return False
        btc_delta_pct = ((current["btc"] - baseline["btc"]) / baseline["btc"]) * 100
        if abs(btc_delta_pct) < self._relative_min_btc_move_pct:
            return False
        base_ask = baseline.get(f"{self._relative_side}_ask", 0.0)
        current_bid = current.get(f"{self._relative_side}_bid", 0.0)
        if base_ask <= 0 or current_bid <= 0:
            return False
        actual_move = current_bid - base_ask
        expected_move = self._relative_expected_poly_move(self._relative_side, btc_delta_pct, now)
        return (actual_move - expected_move) >= self._relative_overreaction_price

    def _manage_relative_reaction(self, seconds_remaining: float, now: float):
        if not self._relative_traded or self._relative_exited:
            return
        if now - self._relative_last_check < POSITION_CHECK_INTERVAL:
            return
        self._relative_last_check = now
        live = self._live_token_price(self._relative_token_id)
        sell_price = live.best_bid if live and live.best_bid > 0 else 0.0
        if sell_price <= 0 and not self.dry_run and self.executor._initialized:
            probe = max(round(self._relative_shares * self._relative_price, 2), 1.0)
            sell_price = self.executor.get_market_price(self._relative_token_id, "SELL", probe)
        if sell_price <= 0:
            return
        pnl = self._relative_shares * sell_price - self._relative_cost
        target = self._relative_profit_target_price()
        overbought_exit = self._relative_is_overbought_exit()
        threshold_reached = (sell_price >= target or overbought_exit) and pnl >= self._relative_min_profit_usd
        if not threshold_reached and not self._relative_trailing_armed:
            return
        if self._relative_shares * sell_price < 5.0:
            return
        if not self.dry_run and self._confirm_ws_sell_price:
            probe = max(round(self._relative_shares * sell_price, 2), 1.0)
            confirmed = self.executor.get_market_price(self._relative_token_id, "SELL", probe)
            if confirmed > 0:
                sell_price = confirmed
                pnl = self._relative_shares * sell_price - self._relative_cost
        threshold_reached = (sell_price >= target or overbought_exit) and pnl >= self._relative_min_profit_usd
        if not self._trailing_profit_exit_ready(
            "Relative overreaction", pnl, threshold_reached,
            self._relative_profit_retreat_pct,
            "_relative_trailing_armed", "_relative_peak_profit",
        ):
            return
        result = self.executor.sell(self._relative_token_id, self._relative_shares, price=sell_price)
        if result.success:
            self._relative_exited = True
            self._relative_exit_revenue += result.amount_usd
            self.stats.bankroll += result.amount_usd
            profit = self._relative_exit_revenue - self._relative_cost
            self._record_relative_result(profit)
            print(f"  Relative overreaction exit: {result.shares:.0f} {self._relative_side} @ ${result.price:.3f} = ${result.amount_usd:.2f} | Profit ${profit:+.2f}")

    def _resolve_relative_reaction(self, closing_btc_price: float):
        if not self._relative_traded or self._relative_exited:
            return
        if self.dry_run:
            won = self._dry_run_resolution_won(self._relative_side, self._current_window)
            if won is None:
                print("  Relative overreaction dry-run resolution pending Polymarket final result")
                self._queue_dry_resolution(
                    "relative", "Relative overreaction", self._relative_side,
                    self._relative_shares, self._relative_cost,
                    self._relative_exit_revenue, self._current_window,
                )
                self._relative_exited = True
                return
        else:
            won = False
            if self._opening_price > 0 and closing_btc_price > 0:
                won = (closing_btc_price >= self._opening_price) == (self._relative_side == "UP")
        if won:
            if self.dry_run:
                revenue = self._relative_shares
            else:
                claim = self.executor.sell(self._relative_token_id, self._relative_shares, price=0.99)
                if not claim.success:
                    print(f"  Relative overreaction claim not verified: {claim.error} - not booking win yet")
                    return
                revenue = claim.amount_usd
            profit = revenue - self._relative_cost
            self.stats.bankroll += revenue
            self._record_relative_result(profit)
            print(f"  Relative overreaction resolved WIN +${profit:.2f}")
        else:
            loss = self._relative_cost - self._relative_exit_revenue
            self._record_relative_result(-loss)
            print(f"  Relative overreaction resolved LOSS -${loss:.2f}")
        self._relative_exited = True

    def _panic_candidate(self, btc_price: float):
        if not self._current_market or self._opening_price <= 0:
            return None
        btc_delta_pct = ((btc_price - self._opening_price) / self._opening_price) * 100
        candidates = []
        for side in ("UP", "DOWN"):
            history = self._panic_price_history.get(side, [])
            if len(history) < 2:
                continue
            current_ask = history[-1][1]
            recent_high = max(row[1] for row in history)
            if current_ask <= 0 or recent_high <= 0:
                continue
            drop_pct = (recent_high - current_ask) / recent_high
            if drop_pct < self._panic_drop_pct or current_ask > self._panic_max_buy_price:
                continue
            if side == "UP" and btc_delta_pct < -self._panic_max_btc_against_pct:
                continue
            if side == "DOWN" and btc_delta_pct > self._panic_max_btc_against_pct:
                continue
            token_id = self._current_market.token_id_up if side == "UP" else self._current_market.token_id_down
            candidates.append((drop_pct, current_ask, recent_high, side, token_id, btc_delta_pct))
        if not candidates:
            return None
        candidates.sort(reverse=True, key=lambda item: item[0])
        return candidates[0]

    def _clear_pending_panic_buy(self):
        self._panic_pending_side = ""
        self._panic_pending_token_id = ""
        self._panic_pending_order_id = ""
        self._panic_pending_price = 0.0
        self._panic_pending_amount = 0.0
        self._panic_pending_shares = 0.0
        self._panic_pending_balance_before = 0.0
        self._panic_pending_last_check = 0.0

    def _activate_panic_buy(self, side: str, token_id: str, price: float, amount: float, shares: float):
        self._panic_traded = True
        self._panic_side = side
        self._panic_token_id = token_id
        self._panic_price = price
        self._panic_cost = amount
        self._panic_shares = shares
        self._panic_trailing_armed = False
        self._panic_peak_profit = 0.0
        self.stats.bankroll -= amount
        self.stats.hourly.record_trade(0.0, 0.0)
        self._clear_pending_panic_buy()
        print(f"  Panic rebound bought: {shares:.0f} {side} @ ${price:.3f} = ${amount:.2f}")
        self.telegram.strategy_trade_alert(
            strategy="Panic rebound",
            side=side,
            price=price,
            amount=amount,
            market_slug=f"btc-updown-{self.period}m-{self._current_window}",
            dry_run=self.dry_run,
            strategy_pnl=self._strategy_pnl.get("panic", 0.0),
            total_pnl=self.stats.total_pnl,
        )

    def _check_pending_panic_buy(self, now: float):
        if not self._panic_pending_side or self._panic_traded or self.dry_run:
            return
        if now - self._panic_pending_last_check < 2.0:
            return
        self._panic_pending_last_check = now

        if self._panic_pending_order_id and self.executor._initialized:
            matched = self.executor.check_buy_fill(
                self._panic_pending_order_id,
                self._panic_pending_price,
            )
            if matched:
                self._activate_panic_buy(
                    self._panic_pending_side,
                    self._panic_pending_token_id,
                    matched[0],
                    matched[1],
                    matched[2],
                )
                return

        if self.executor._initialized and self._panic_pending_balance_before > 0:
            real_bal = self.executor.get_balance()
            spent = self._panic_pending_balance_before - real_bal if real_bal > 0 else 0.0
            if spent > 1.0:
                shares = spent / self._panic_pending_price if self._panic_pending_price > 0 else self._panic_pending_shares
                self.stats.bankroll = real_bal + spent
                self._last_real_balance = real_bal
                self._activate_panic_buy(
                    self._panic_pending_side,
                    self._panic_pending_token_id,
                    self._panic_pending_price,
                    spent,
                    shares,
                )

    def _try_panic_rebound_entry(self, btc_price: float, seconds_remaining: float):
        if not self._panic_enabled:
            return
        if not self._check_risk_limits("panic", "panic rebound"):
            self._panic_trade_attempted = True
            return
        if not self.dry_run and not self.executor._initialized:
            return
        elapsed = PERIOD_SECONDS[self.period] - seconds_remaining
        if elapsed < 0:
            return
        if elapsed > self._panic_entry_window_seconds:
            self._panic_trade_attempted = True
            return
        candidate = self._panic_candidate(btc_price)
        if not candidate:
            return
        drop_pct, ws_ask, recent_high, side, token_id, btc_delta_pct = candidate
        trade_amount = round(min(self._panic_trade_amount, self.strategy_config.max_bet, self.stats.bankroll), 2)
        if trade_amount < self.strategy_config.min_bet:
            return
        actual_price = ws_ask if self.dry_run else self.executor.get_market_price(token_id, "BUY", trade_amount)
        if actual_price <= 0:
            return
        cap_price = min(self._panic_max_buy_price, max_buy_price())
        if actual_price > cap_price:
            print(f"  Panic skip {side}: WS ask ${ws_ask:.3f}, executable ${actual_price:.3f} > cap ${cap_price:.2f}")
            return
        print(f"\n  Panic rebound {side}: ask ${ws_ask:.3f} from high ${recent_high:.3f} drop {drop_pct:.0%}, BTC {btc_delta_pct:+.3f}%")
        self._panic_trade_attempted = True
        result = self.executor.buy(token_id=token_id, amount_usd=trade_amount, price=actual_price)
        if not result.success:
            if result.error == "UNVERIFIED_BUY":
                self._panic_pending_side = side
                self._panic_pending_token_id = token_id
                self._panic_pending_order_id = result.order_id
                self._panic_pending_price = result.price
                self._panic_pending_amount = result.amount_usd
                self._panic_pending_shares = result.shares
                self._panic_pending_balance_before = self.stats.bankroll
                self._panic_pending_last_check = 0.0
                print("  Panic rebound buy unverified - will poll order API and balance")
                return
            print(f"  Panic rebound buy failed: {result.error}")
            return
        self._activate_panic_buy(side, token_id, result.price, result.amount_usd, result.shares)

    def _manage_panic_rebound(self, seconds_remaining: float, now: float):
        if not self._panic_traded or self._panic_exited:
            return
        if now - self._panic_last_check < POSITION_CHECK_INTERVAL:
            return
        self._panic_last_check = now
        live = self._live_token_price(self._panic_token_id)
        sell_price = live.best_bid if live and live.best_bid > 0 else 0.0
        if sell_price <= 0 and not self.dry_run and self.executor._initialized:
            probe = max(round(self._panic_shares * self._panic_price, 2), 1.0)
            sell_price = self.executor.get_market_price(self._panic_token_id, "SELL", probe)
        if sell_price <= 0:
            return
        target = self._panic_profit_target_price()
        pnl = self._panic_shares * sell_price - self._panic_cost
        threshold_reached = sell_price >= target and pnl >= self._panic_min_profit_usd
        if not threshold_reached and not self._panic_trailing_armed:
            return
        if self._panic_shares * sell_price < 5.0:
            return
        if not self.dry_run and self._confirm_ws_sell_price:
            probe = max(round(self._panic_shares * sell_price, 2), 1.0)
            confirmed = self.executor.get_market_price(self._panic_token_id, "SELL", probe)
            if confirmed > 0:
                sell_price = confirmed
                pnl = self._panic_shares * sell_price - self._panic_cost
        threshold_reached = sell_price >= target and pnl >= self._panic_min_profit_usd
        if not self._trailing_profit_exit_ready(
            "Panic rebound", pnl, threshold_reached,
            self._panic_profit_retreat_pct,
            "_panic_trailing_armed", "_panic_peak_profit",
        ):
            return
        result = self.executor.sell(self._panic_token_id, self._panic_shares, price=sell_price)
        if result.success:
            self._panic_exited = True
            self._panic_exit_revenue += result.amount_usd
            self.stats.bankroll += result.amount_usd
            profit = self._panic_exit_revenue - self._panic_cost
            self._record_panic_result(profit)
            print(f"  Panic rebound exit: {result.shares:.0f} {self._panic_side} @ ${result.price:.3f} = ${result.amount_usd:.2f} | Profit ${profit:+.2f}")

    def _resolve_panic_rebound(self, closing_btc_price: float):
        if not self._panic_traded or self._panic_exited:
            return
        if self.dry_run:
            won = self._dry_run_resolution_won(self._panic_side, self._current_window)
            if won is None:
                print("  Panic rebound dry-run resolution pending Polymarket final result")
                self._queue_dry_resolution(
                    "panic", "Panic rebound", self._panic_side,
                    self._panic_shares, self._panic_cost,
                    self._panic_exit_revenue, self._current_window,
                )
                self._panic_exited = True
                return
        else:
            won = False
            if self._opening_price > 0 and closing_btc_price > 0:
                won = (closing_btc_price >= self._opening_price) == (self._panic_side == "UP")
        if won:
            if self.dry_run:
                revenue = self._panic_shares
            else:
                claim = self.executor.sell(self._panic_token_id, self._panic_shares, price=0.99)
                if not claim.success:
                    print(f"  Panic rebound claim not verified: {claim.error} - not booking win yet")
                    return
                revenue = claim.amount_usd
            profit = revenue - self._panic_cost
            self.stats.bankroll += revenue
            self._record_panic_result(profit)
            print(f"  Panic rebound resolved WIN +${profit:.2f}")
        else:
            loss = self._panic_cost - self._panic_exit_revenue
            self._record_panic_result(-loss)
            print(f"  Panic rebound resolved LOSS -${loss:.2f}")
        self._panic_exited = True

    # ── Market prices (cached, complement engine) ───────────────────

    def _compute_realized_vol(self) -> float:
        """Rolling std dev of recent window closing deltas.

        Returns the realized vol to pass into the Brownian motion model.
        Falls back to 0.12 until at least 6 windows have accumulated.
        Floored/capped to prevent extreme values breaking the model.
        """
        min_samples = max(6, self._rolling_vol_windows // 2)
        if len(self._recent_window_deltas) >= min_samples:
            vol = statistics.stdev(self._recent_window_deltas)
            return max(self._vol_floor, min(self._vol_cap, vol))
        return self._vol_fallback

    def _get_market_prices(self, btc_price: float, seconds_remaining: float) -> tuple:
        if self.dry_run or not self.executor._initialized:
            if self._opening_price <= 0:
                return 0.50, 0.50
            delta_pct = (btc_price - self._opening_price) / self._opening_price
            time_factor = 1 - (seconds_remaining / PERIOD_SECONDS[self.period])
            lag_factor = min(time_factor * 0.7, 0.85)
            implied = 0.5 + lag_factor * math.tanh(delta_pct * 500) * 0.45
            up = round(min(max(implied, 0.02), 0.98), 3)
            return up, round(1.0 - up, 3)

        market = self._ensure_market_subscription()
        if market:
            up_live = self._live_token_price(market.token_id_up)
            down_live = self._live_token_price(market.token_id_down)
            if up_live and down_live:
                up_price = up_live.best_ask or up_live.last_trade or up_live.mid
                down_price = down_live.best_ask or down_live.last_trade or down_live.mid
                if up_price > 0 and down_price > 0:
                    self._cached_up = up_price
                    self._cached_down = down_price
                    return up_price, down_price

        now = time.time()
        if now - self._price_last_fetched < self._PRICE_REFRESH:
            return self._cached_up, self._cached_down

        try:
            market = self._ensure_market_subscription()
            if not market:
                return self._cached_up, self._cached_down

            probe_amount = 5.0
            up_price = self.executor.get_market_price(market.token_id_up, "BUY", probe_amount)
            down_price = self.executor.get_market_price(market.token_id_down, "BUY", probe_amount)

            if up_price <= 0 and down_price <= 0:
                return self._cached_up, self._cached_down
            if up_price <= 0:
                up_price = round(1.0 - down_price, 3)
            if down_price <= 0:
                down_price = round(1.0 - up_price, 3)

            self._cached_up = up_price
            self._cached_down = down_price
            self._price_last_fetched = now

            return up_price, down_price

        except Exception as e:
            print(f"[price] Error: {e}")
            return self._cached_up, self._cached_down

    # ── Entry ───────────────────────────────────────────────────────

    def _execute_trade(self, sig, seconds_remaining: float):
        self._trade_attempted = True

        if not self._check_risk_limits("trend", "trend trade"):
            return

        # ── Circuit breaker: CLOB health check ───────────────────
        if self._clob_halted:
            print(f"  🔌 CLOB HALTED — skipping trade ({self._consecutive_buy_failures} consecutive failures)")
            return

        if not self.dry_run and self.executor._initialized:
            try:
                self.executor.client.get_ok()
            except Exception as e:
                self._consecutive_buy_failures += 1
                print(f"  🔌 CLOB health check failed: {e}")
                if self._consecutive_buy_failures >= self._HALT_AFTER_FAILURES:
                    self._clob_halted = True
                    msg = (f"🔌 CLOB HALTED after {self._consecutive_buy_failures} "
                           f"consecutive health check failures — stopping trades until recovery")
                    print(f"\n  {msg}")
                    self.telegram.status_update({"alert": msg})
                return

        market = self._ensure_market_subscription() if not self.dry_run else None
        token_id = ""
        if market:
            token_id = market.token_id_up if sig.side == "UP" else market.token_id_down
        else:
            token_id = f"DRY-{sig.side}-{self._current_window}"

        slug = f"btc-updown-{self.period}m-{self._current_window}"
        trade_amount = round(sig.kelly_size, 2)

        print(f"\n  🎯 {sig.side} | edge={sig.edge:.3f} | "
              f"prob={sig.true_prob:.2f} | BTC Δ={sig.btc_delta_pct:+.3f}%")
        print(f"     Amount: ${trade_amount:.2f} | T-{seconds_remaining:.0f}s")

        # Preview actual market price, re-check edge, then pass price into buy()
        # so executor skips a second fetch (saves one Tor roundtrip ~500ms)
        hint_price = 0.0
        if self.dry_run:
            hint_price = self._cached_up if sig.side == "UP" else self._cached_down
        elif self.executor._initialized:
            live_price = self._live_token_price(token_id)
            if live_price:
                print(
                    f"  WS {sig.side}: bid ${live_price.best_bid:.3f} | "
                    f"ask ${live_price.best_ask:.3f} | last ${live_price.last_trade:.3f}"
                )
            actual_price = self.executor.get_market_price(token_id, "BUY", trade_amount)
            if actual_price > 0:
                actual_edge = sig.true_prob - actual_price
                print(
                    f"  Executable buy price for ${trade_amount:.2f}: "
                    f"${actual_price:.3f} (edge: {actual_edge:.3f})"
                )

                if False and actual_edge < self.strategy_config.min_edge:
                    print(f"  ⚠️  Edge gone at market price — skipping")
                    btc_approx = self._opening_price * (1 + sig.btc_delta_pct / 100) if self._opening_price > 0 else 0
                    self.tracker.log_signal(
                        window_ts=self._current_window,
                        btc_price=btc_approx,
                        opening_price=self._opening_price,
                        up_price=self._cached_up,
                        down_price=self._cached_down,
                        seconds_remaining=seconds_remaining,
                        side=sig.side,
                        true_prob=sig.true_prob,
                        market_price=actual_price,
                        edge=actual_edge,
                        kelly_size=sig.kelly_size,
                        action="skipped_edge_gone",
                        skip_reason="edge_gone_at_market",
                        actual_price=actual_price,
                        actual_edge=actual_edge,
                    )
                    return

                hint_price = actual_price

        cap_price = max_buy_price()
        if hint_price > cap_price:
            print(f"  SKIP: buy price ${hint_price:.3f} > MAX_BUY_PRICE ${cap_price:.2f}")
            self.tracker.log_signal(
                window_ts=self._current_window,
                btc_price=self._opening_price * (1 + sig.btc_delta_pct / 100) if self._opening_price > 0 else 0,
                opening_price=self._opening_price,
                up_price=self._cached_up,
                down_price=self._cached_down,
                seconds_remaining=seconds_remaining,
                side=sig.side,
                true_prob=sig.true_prob,
                market_price=hint_price,
                edge=sig.edge,
                kelly_size=sig.kelly_size,
                action="skipped_price_cap",
                skip_reason="buy_price_above_cap",
                actual_price=hint_price,
            )
            return

        result = self.executor.buy(token_id=token_id, amount_usd=trade_amount, price=hint_price)

        if result.success:
            self._consecutive_buy_failures = 0  # Reset circuit breaker
            self._traded = True
            self._trade_side = sig.side
            self._trade_price = result.price
            self._trade_cost = result.amount_usd
            self._trade_shares = result.shares
            self._trade_token_id = token_id
            self._trend_trailing_armed = False
            self._trend_peak_profit = 0.0

            self.stats.bankroll -= result.amount_usd
            self.stats.hourly.record_trade(sig.edge, sig.btc_delta_pct)

            btc_approx = self._opening_price * (1 + sig.btc_delta_pct / 100) if self._opening_price > 0 else 0
            self.tracker.log_signal(
                window_ts=self._current_window,
                btc_price=btc_approx,
                opening_price=self._opening_price,
                up_price=self._cached_up,
                down_price=self._cached_down,
                seconds_remaining=seconds_remaining,
                side=sig.side,
                true_prob=sig.true_prob,
                market_price=sig.market_price,
                edge=sig.edge,
                kelly_size=sig.kelly_size,
                action="traded",
                actual_price=result.price,
                actual_edge=sig.true_prob - result.price,
                fill_price=result.price,
            )
            self.tracker.log_trade_entry(
                window_ts=self._current_window,
                side=sig.side,
                entry_price=result.price,
                entry_shares=result.shares,
                entry_cost=result.amount_usd,
                edge=sig.edge,
                prob=sig.true_prob,
                btc_delta=sig.btc_delta_pct,
                seconds_remaining=seconds_remaining,
                entry_delta_pct=sig.btc_delta_pct,
                entry_seconds_remaining=seconds_remaining,
            )

            mode = "PAPER" if self.dry_run else "LIVE"
            print(f"  ✅ {mode}: {result.shares:.0f} shares @ "
                  f"${result.price:.3f} = ${result.amount_usd:.2f}")
            print(f"     Watching live Polymarket bid for profit exit")

            self.telegram.strategy_trade_alert(
                strategy="Trend follow",
                side=sig.side,
                price=result.price,
                amount=result.amount_usd,
                market_slug=slug,
                dry_run=self.dry_run,
                strategy_pnl=self._strategy_pnl.get("trend", 0.0),
                total_pnl=self.stats.total_pnl,
            )
        else:
            if result.error == "UNVERIFIED_BUY":
                # Order likely filled but Polygon hasn't settled.
                # Save details — window boundary sync will detect the fill.
                self._pending_buy_side = sig.side
                self._pending_buy_price = result.price
                self._pending_buy_amount = result.amount_usd
                self._pending_buy_shares = result.shares
                self._pending_buy_token_id = token_id
                self._pending_buy_edge = sig.edge
                self._pending_buy_delta = sig.btc_delta_pct
                self._pending_buy_order_id = result.order_id
                self._pending_buy_last_check = 0.0
                self._balance_before_buy = self.stats.bankroll
                print(f"  Buy sent but unverified - will poll order API and balance")
            else:
                print(f"  ❌ Buy failed: {result.error}")
                # Circuit breaker: track consecutive API failures
                err = str(result.error).lower()
                if "request exception" in err or "service not ready" in err or "status_code=none" in err:
                    self._consecutive_buy_failures += 1
                    if self._consecutive_buy_failures >= self._HALT_AFTER_FAILURES:
                        self._clob_halted = True
                        msg = (f"🔌 CLOB HALTED after {self._consecutive_buy_failures} "
                               f"consecutive API failures — stopping trades until restart")
                        print(f"\n  {msg}")
                        self.telegram.status_update({"alert": msg})

    # ── Resolve (partial fill aware) ────────────────────────────────

    def _resolve_previous_trade(self):
        if self._exited:
            profit = self._exit_revenue - self._trade_cost
            self._record_trend_result(profit)
            result_emoji = "✅ WIN" if profit > 0 else "❌ LOSS"
            residual_note = f" (~{self._residual_shares:.0f} residual)" if self._residual_shares >= 1 else ""
            print(f"  {result_emoji} (exited{residual_note}) ${profit:+.2f} | "
                  f"P&L: ${self.stats.total_pnl:+.2f} | "
                  f"Bank: ${self.stats.bankroll:.2f}")
            self.telegram.strategy_result_alert(
                strategy="Trend follow",
                profit=profit,
                strategy_pnl=self._strategy_pnl.get("trend", 0.0),
                total_pnl=self.stats.total_pnl,
            )
            btc_price, _ = self.price_feed.get_price()
            self.tracker.log_trade_resolve(
                btc_final_price=btc_price,
                opening_price=self._opening_price,
                won=profit > 0,
                profit=profit,
                exit_revenue=self._exit_revenue,
                resolution_method="exited",
            )
            return

        original_cost = self._trade_cost
        remaining_shares = self._trade_shares

        # ── Dry run: Binance price fallback ──────────────────────────
        if self.dry_run:
            won = self._dry_run_resolution_won(self._trade_side, self._current_window)
            if won is None:
                print("  Trend follow dry-run resolution pending Polymarket final result")
                self._queue_dry_resolution(
                    "trend", "Trend follow", self._trade_side,
                    remaining_shares, original_cost,
                    self._exit_revenue, self._current_window,
                )
                return
            self._record_resolution(
                won=won,
                original_cost=original_cost,
                remaining_shares=remaining_shares,
                resolution_method="polymarket_gamma",
                claim_revenue=0.0,
                claim_result="gamma_final",
            )
            return

        # ── Live: attempt claim sell first — result is the truth ─────
        # Binance price and oracle can disagree when BTC is near the opening
        # price at resolution. The claim sell result is ground truth:
        #   - Sell succeeds at ~$0.99 → shares had value → won
        #   - "no match" or near-zero fill → shares worthless → lost
        won = None
        claim_revenue = 0.0
        claim_result = "not_attempted"
        resolution_method = "claim_sell"

        claim_notional = remaining_shares * 0.99
        live_token = (self._trade_token_id
                      and not self._trade_token_id.startswith("DRY-")
                      and self.executor._initialized)

        # Short-circuit: if last observed sell price is below $0.50, market has
        # already priced these shares as worthless — skip the claim API call.
        if self._last_sell_price_seen > 0 and self._last_sell_price_seen < 0.50:
            net_loss = original_cost - self._exit_revenue
            profit = -net_loss
            self._record_trend_result(profit)
            partial_note = f" (partial exit ${self._exit_revenue:.2f})" if self._exit_revenue > 0 else ""
            print(f"  ❌ LOSS{partial_note} -${net_loss:.2f} [market_price] | "
                  f"P&L: ${self.stats.total_pnl:+.2f} | "
                  f"Bank: ${self.stats.bankroll:.2f}")
            self.telegram.strategy_result_alert(
                strategy="Trend follow",
                profit=profit,
                strategy_pnl=self._strategy_pnl.get("trend", 0.0),
                total_pnl=self.stats.total_pnl,
            )
            btc_price, _ = self.price_feed.get_price()
            self.tracker.log_trade_resolve(
                btc_final_price=btc_price,
                opening_price=self._opening_price,
                won=False,
                profit=profit,
                exit_revenue=self._exit_revenue,
                resolution_method="market_price",
                claim_result="skipped_losing",
            )
            return

        pre_sell_balance = 0.0
        if live_token and claim_notional >= 5.0:
            print(f"  💰 Claiming: sell {remaining_shares:.0f} shares @ $0.99...")
            pre_sell_balance = self.executor.get_balance()
            claim = self.executor.sell(
                token_id=self._trade_token_id,
                shares=remaining_shares,
                price=0.99,
            )
            if claim.success and claim.amount_usd > remaining_shares * 0.50:
                # API says success — verify with balance check to catch phantom fills
                time.sleep(2)
                post_sell_balance = self.executor.get_balance()
                balance_increase = max(0.0, post_sell_balance - pre_sell_balance) if (
                    pre_sell_balance > 0 and post_sell_balance > 0
                ) else claim.amount_usd
                if balance_increase > remaining_shares * 0.99 * 0.50:
                    # Balance confirmed — real fill
                    won = True
                    claim_revenue = claim.amount_usd
                    claim_result = "filled"
                    self.stats.bankroll += claim_revenue
                else:
                    # API said success but no USDC arrived yet — defer to next window
                    print(f"  ⏳ Possible phantom sell "
                          f"(api=${claim.amount_usd:.2f}, balance_increase=${balance_increase:.2f})"
                          f" — deferring to next window balance sync")
            elif "no match" in claim.error.lower() or (
                claim.success and claim.amount_usd < remaining_shares * 0.10
            ):
                # No buyers for these shares → shares worthless → definitive loss
                won = False
                claim_result = "no_match"
            elif "not enough balance" in claim.error.lower():
                # Tracked share count is slightly above on-chain balance (rounding).
                # Retry with one fewer share to clear the discrepancy.
                retry_shares = int(remaining_shares) - 1
                print(f"  🔄 Rounding fix: retrying claim with {retry_shares} shares...")
                if retry_shares > 0 and float(retry_shares) * 0.99 >= 5.0:
                    retry = self.executor.sell(
                        token_id=self._trade_token_id,
                        shares=float(retry_shares),
                        price=0.99,
                    )
                    if retry.success and retry.amount_usd > retry_shares * 0.50:
                        time.sleep(2)
                        post_bal = self.executor.get_balance()
                        balance_increase = max(0.0, post_bal - pre_sell_balance)
                        if balance_increase > float(retry_shares) * 0.99 * 0.50:
                            won = True
                            claim_revenue = retry.amount_usd
                            claim_result = "filled"
                            self.stats.bankroll += claim_revenue
                        # else: retry succeeded but balance unconfirmed — fall to defer
                # else: retry failed or too small — fall to defer (won still None)
            # else: any other error — fall to defer (won still None)
        else:
            if live_token and claim_notional < 5.0:
                print(f"  💰 {remaining_shares:.0f} shares below $5 min — deferring to auto-resolution")

        # ── Deferred fallback ────────────────────────────────────────
        # The old balance check fired before auto-resolution settled on-chain.
        # Any unresolved case is now deferred to the next window boundary
        # (~5 min), where Polygon settlement is guaranteed to have landed.
        if won is None:
            if not live_token:
                # No valid token/executor — Binance price as last resort
                btc_price, _ = self.price_feed.get_price()
                if self._opening_price > 0 and btc_price > 0:
                    won = (btc_price >= self._opening_price) == (self._trade_side == "UP")
                    resolution_method = "binance_fallback"
                    print(f"  ⚠️  No live token — using Binance fallback")
                else:
                    print(f"  ⚠️  Cannot determine resolution outcome — skipping")
                    return
            else:
                if pre_sell_balance <= 0:
                    pre_sell_balance = self.executor.get_balance()
                print(f"  ⏳ Resolution deferred to next window balance sync")
                self._pending_phantom = {
                    "pre_sell_balance": pre_sell_balance,
                    "expected_revenue": remaining_shares * 0.99,
                    "cost": original_cost,
                    "exit_revenue": self._exit_revenue,
                    "shares": remaining_shares,
                    "side": self._trade_side,
                    "token_id": self._trade_token_id,
                    "window_ts": self._current_window,
                    "opening_price": self._opening_price,
                }
                return

        if won is None:
            return

        self._record_resolution(
            won=won,
            original_cost=original_cost,
            remaining_shares=remaining_shares,
            resolution_method=resolution_method,
            claim_revenue=claim_revenue,
            claim_result=claim_result,
        )

    def _record_resolution(
        self, won: bool, original_cost: float, remaining_shares: float,
        resolution_method: str, claim_revenue: float, claim_result: str = "not_attempted",
    ):
        """Apply win/loss to stats, print result, alert Telegram, log to tracker."""
        if won:
            resolution_payout = 0.0
            if claim_revenue > 0:
                total_received = self._exit_revenue + claim_revenue
            else:
                resolution_payout = remaining_shares * 1.0
                total_received = self._exit_revenue + resolution_payout
                self.stats.bankroll += resolution_payout
                self._unclaimed_winnings += resolution_payout
            profit = total_received - original_cost
            self._record_trend_result(profit)
            partial_note = f" (partial exit ${self._exit_revenue:.2f})" if self._exit_revenue > 0 else ""
            claimed_note = (
                f" (claimed ${claim_revenue:.2f})"
                if claim_revenue > 0
                else f" (unclaimed payout ${resolution_payout:.2f})"
            )
            print(f"  ✅ WIN{partial_note}{claimed_note} +${profit:.2f} [{resolution_method}] | "
                  f"P&L: ${self.stats.total_pnl:+.2f} | "
                  f"Bank: ${self.stats.bankroll:.2f}")
            self.telegram.strategy_result_alert(
                strategy="Trend follow",
                profit=profit,
                strategy_pnl=self._strategy_pnl.get("trend", 0.0),
                total_pnl=self.stats.total_pnl,
            )
        else:
            net_loss = original_cost - self._exit_revenue
            profit = -net_loss
            self._record_trend_result(profit)
            partial_note = f" (partial exit ${self._exit_revenue:.2f})" if self._exit_revenue > 0 else ""
            print(f"  ❌ LOSS{partial_note} -${net_loss:.2f} [{resolution_method}] | "
                  f"P&L: ${self.stats.total_pnl:+.2f} | "
                  f"Bank: ${self.stats.bankroll:.2f}")
            self.telegram.strategy_result_alert(
                strategy="Trend follow",
                profit=profit,
                strategy_pnl=self._strategy_pnl.get("trend", 0.0),
                total_pnl=self.stats.total_pnl,
            )

        btc_price, _ = self.price_feed.get_price()
        self.tracker.log_trade_resolve(
            btc_final_price=btc_price,
            opening_price=self._opening_price,
            won=won,
            profit=profit,
            exit_revenue=self._exit_revenue,
            resolution_method=resolution_method,
            claim_result=claim_result,
        )

    # ── Hourly + shutdown ───────────────────────────────────────────

    def _check_hourly_summary(self):
        current_hour = int(time.time() // 3600)
        if current_hour != self._last_hour_check:
            self._last_hour_check = current_hour
            h = self.stats.hourly.to_dict()
            o = self.stats.to_dict()

            # Sync real balance for accuracy
            if not self.dry_run and self.executor._initialized:
                real_bal = self.executor.get_balance()
                if real_bal > 0:
                    self.stats.bankroll = real_bal
                    self._last_real_balance = real_bal
                    o["bankroll"] = real_bal

            real_pnl = self.stats.bankroll - self._session_start_balance

            print(f"\n{'═' * 55}")
            print(f"  📊 HOURLY SUMMARY")
            print(f"  This hour: {h['trades']} trades | "
                  f"{h['wins']}W/{h['losses']}L | "
                  f"P&L: ${h['pnl']:+.2f}")
            if h['trades'] > 0:
                print(f"  Avg edge: {h['avg_edge']*100:.1f}%")
            print(f"  Windows: {h['windows_seen']} seen, "
                  f"{h['windows_skipped']} skipped")
            print(f"  Overall: {o['total_trades']} trades | "
                  f"P&L: ${o['pnl']:+.2f} | Bank: ${o['bankroll']:.2f}")
            print(f"  💰 Real P&L (balance): ${real_pnl:+.2f} "
                  f"(${self._session_start_balance:.2f} → ${self.stats.bankroll:.2f})")
            if self._unclaimed_winnings > 0:
                print(f"  💰 Unclaimed: ${self._unclaimed_winnings:.2f}")
            print(f"{'═' * 55}\n")
            self.telegram.hourly_summary(h, o)
            self.stats.hourly.reset()

    def _handle_shutdown(self, signum, frame):
        print(f"\n\n🛑 Shutting down...")
        self._running = False
        self.price_feed.stop()
        self.poly_feed.stop()
        if self.executor._initialized:
            self.executor.cancel_all()

        # Final real balance sync
        if not self.dry_run and self.executor._initialized:
            real_bal = self.executor.get_balance()
            if real_bal > 0:
                self.stats.bankroll = real_bal
                self._last_real_balance = real_bal

        real_pnl = self.stats.bankroll - self._session_start_balance
        o = self.stats.to_dict()
        print(f"\n{'═' * 55}")
        print(f"  FINAL: {o['total_trades']} trades | "
              f"{o['wins']}W/{o['losses']}L | "
              f"WR: {o['win_rate']:.1f}%")
        print(f"  Tracked P&L: ${o['pnl']:+.2f} | Bank: ${o['bankroll']:.2f}")
        print(f"  💰 Real P&L: ${real_pnl:+.2f} "
              f"(${self._session_start_balance:.2f} → ${self.stats.bankroll:.2f})")
        if self._unclaimed_winnings > 0:
            print(f"  💰 Unclaimed: ${self._unclaimed_winnings:.2f}")
        print(f"{'═' * 55}")

        self.telegram.status_update(o)

        self.tracker.log_session(
            start_time=self._session_start_time,
            end_time=time.time(),
            start_balance=self._session_start_balance,
            end_balance=self.stats.bankroll,
            tracked_pnl=o["pnl"],
            trades=o["total_trades"],
            wins=o["wins"],
            losses=o["losses"],
            avg_entry_price=self.stats.hourly.avg_edge,   # proxy via hourly stats
            avg_edge=self.stats.hourly.avg_edge,
            avg_delta=self.stats.hourly.avg_delta,
        )

        time.sleep(1)
        sys.exit(0)


if __name__ == "__main__":
    bot = PolyBot()
    bot.start()
