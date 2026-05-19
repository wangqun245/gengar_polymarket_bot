# Setup

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Configure `.env`

Use `.env.example` as the template. Keep real credentials only in `.env`.

Required live-trading fields:

```env
PRIVATE_KEY=0x...
SAFE_ADDRESS=0x...
SIGNATURE_TYPE=2
DRY_RUN=false
```

Strategy fields:

```env
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

## 3. Run

```bash
python bot.py
```

The bot will warm up the Binance baseline before signals can fire. With the
default settings it waits 10 minutes, then uses those 10 minutes as the
temporary comparison baseline until enough data exists for the longer
`LATENCY_BASELINE_HOURS` window.

## 4. Validation

Syntax check:

```bash
python -c "import ast, pathlib; [ast.parse(pathlib.Path(p).read_text()) for p in ['bot.py','strategy.py','executor.py','market.py','price_feed.py','polymarket_ws.py']]; print('ok')"
```

Start with `DRY_RUN=true`, watch the terminal output through active BTC moves,
then tune the z-score and minimum movement thresholds before live trading.
