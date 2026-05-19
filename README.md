# PolyBot

Local Polymarket bot for BTC 5-minute Up/Down markets.

Current strategy: **Binance 50ms latency spike**.

## Strategy

The bot watches Binance BTCUSDT trade prints in fixed 50ms buckets. Each bucket
records BTC price movement, filled quantity, notional, and trade count. The
latest bucket is compared against rolling baseline statistics from the previous
few hours.

On startup, if the bot does not yet have the full multi-hour baseline, it waits
until `LATENCY_WARMUP_MINUTES` minutes of Binance buckets have accumulated.
After that warmup it uses the most recent 10 minutes as the comparison baseline
until enough data exists for the longer `LATENCY_BASELINE_HOURS` window.

The bot also tracks Binance-to-Polymarket signal delay. When a Binance spike
appears, it records how long the UP Polymarket price takes to move in the same
direction and keeps a rolling average. Binance-triggered buys and sells use
that average as a short synchronization wait before executing.

Entry:

- Binance BTC price suddenly moves upward inside a 50ms bucket.
- Binance filled quantity in that bucket is also unusually high.
- The Binance signal occurs within `ENTRY_WINDOW_SECONDS` from the start of
  the 5-minute market. If the signal is inside the window, the latency sync
  wait may finish just after the window and still buy.
- The Polymarket BTC UP ask is live and no higher than `MAX_BUY_PRICE`.

Exit:

- Binance BTC price suddenly moves downward inside a 50ms bucket.
- Binance filled quantity is elevated.
- The live Polymarket BTC UP bid is high enough to satisfy `MIN_PROFIT_PCT`.
- If Binance has not produced a sudden down signal, the bot trails the best
  unrealized Polymarket profit. It holds while profit keeps rising and the
  live bid is above $0.50, then sells when profit retreats by
  `PROFIT_RETREAT_PCT`. The same retreat rule also applies when the live bid is
  at or below $0.50 after a prior profit peak.

There are no Brownian probability, Kelly, Kronos, Chan, cheap scalp, panic
rebound, relative overreaction, or multi-strategy model signals in the current
runner.

## Main Files

```text
bot.py                Single-strategy bot runner
strategy.py           Binance 50ms price/volume spike detector
executor.py           Polymarket CLOB execution and balance verification
market.py             Current BTC 5-minute market discovery
price_feed.py         Binance BTCUSDT trade WebSocket
polymarket_ws.py      Polymarket best bid/ask WebSocket
telegram_notifier.py  Optional mobile alerts
proxy.py              Optional Tor proxy helper
```

## Configuration

Start from `.env.example` and set real credentials in `.env`.

Important settings:

```env
DRY_RUN=true
TRADE_AMOUNT=5.00
MAX_BUY_PRICE=0.60
MIN_PROFIT_PCT=0.06
PROFIT_RETREAT_PCT=0.35
DRY_RUN_BALANCE=100
ENTRY_WINDOW_SECONDS=240
STRATEGY_PEAK_LOSS_DROP=0
STRATEGY_COOLING_DOWN_PERIOD=5400

LATENCY_BUCKET_MS=50
LATENCY_BASELINE_HOURS=3
LATENCY_MIN_BASELINE_BUCKETS=300
LATENCY_WARMUP_MINUTES=10
LATENCY_MIN_WARMUP_BUCKETS=300
LATENCY_ENTRY_PRICE_Z=3.0
LATENCY_ENTRY_VOLUME_Z=3.0
LATENCY_EXIT_PRICE_Z=2.5
LATENCY_EXIT_VOLUME_Z=2.5
LATENCY_ENTRY_MIN_PRICE_MOVE_PCT=0.015
LATENCY_EXIT_MIN_PRICE_MOVE_PCT=0.010
CHANNEL_LATENCY_DEFAULT_MS=300
CHANNEL_LATENCY_MAX_WAIT_MS=1500
CHANNEL_LATENCY_POLY_SYNC_MIN_MOVE=0.005
```

## Run

```bash
python bot.py
```

Use `DRY_RUN=true` until the new thresholds have been watched live and tuned.
