"""Market discovery for Polymarket 5-minute crypto Up/Down markets.

The market slug is deterministic: {asset}-updown-5m-{window_ts}
where window_ts = now - (now % 300), i.e. the start of the current 5-min window.
"""

import time
import json
import urllib.request
from dataclasses import dataclass
from typing import Optional


GAMMA_API = "https://gamma-api.polymarket.com"
PERIOD_SECONDS = {5: 300, 15: 900}


@dataclass
class MarketWindow:
    slug: str
    condition_id: str
    token_id_up: str
    token_id_down: str
    window_start: int
    window_end: int
    opening_price: Optional[float] = None
    up_price: float = 0.50
    down_price: float = 0.50

    @property
    def seconds_remaining(self) -> float:
        return max(0, self.window_end - time.time())


def current_window_ts(period_minutes: int = 5) -> int:
    """Calculate the Unix timestamp for the start of the current window."""
    period = PERIOD_SECONDS[period_minutes]
    now = int(time.time())
    return now - (now % period)


def next_window_ts(period_minutes: int = 5) -> int:
    """Calculate when the next window opens."""
    period = PERIOD_SECONDS[period_minutes]
    return current_window_ts(period_minutes) + period


def market_slug(period_minutes: int = 5, window_ts: int = None, asset: str = "btc") -> str:
    """Generate the deterministic market slug."""
    ts = window_ts or current_window_ts(period_minutes)
    asset = str(asset or "btc").lower()
    return f"{asset}-updown-{period_minutes}m-{ts}"


def fetch_market_by_slug(slug: str) -> Optional[dict]:
    """Fetch market data from Gamma API by slug."""
    try:
        url = f"{GAMMA_API}/events?slug={slug}"
        req = urllib.request.Request(url, headers={"User-Agent": "PolyBot/1.0"})
        resp = urllib.request.urlopen(req, timeout=10)
        data = json.loads(resp.read().decode())
        if data and len(data) > 0:
            return data[0]
        return None
    except Exception as e:
        print(f"[market] Failed to fetch {slug}: {e}")
        return None


def _json_field(value, default=None):
    if default is None:
        default = []
    if isinstance(value, str) and value:
        try:
            return json.loads(value)
        except Exception:
            return default
    return value if value not in ("", None) else default


def _normalize_outcome(value: str) -> str:
    value = str(value or "").strip().lower()
    if value in {"up", "higher", "yes"}:
        return "UP"
    if value in {"down", "lower", "no"}:
        return "DOWN"
    return ""


def extract_winning_outcome(event_data: dict) -> Optional[str]:
    """Return the resolved winning side ("UP"/"DOWN") from Gamma event data.

    Gamma fields have changed over time, so this checks explicit winner fields
    first and then falls back to resolved outcome prices where the winner trades
    near $1 and the loser near $0.
    """
    markets = event_data.get("markets", []) if event_data else []
    market = markets[0] if markets else {}

    for source in (event_data, market):
        for key in (
            "winningOutcome", "winning_outcome", "winner",
            "resolvedOutcome", "resolved_outcome", "result",
        ):
            winner = _normalize_outcome(source.get(key, ""))
            if winner:
                return winner

    outcomes = _json_field(market.get("outcomes", []), [])
    outcome_prices = _json_field(market.get("outcomePrices", []), [])
    if not isinstance(outcomes, list) or not isinstance(outcome_prices, list):
        return None

    resolved = bool(
        market.get("resolved")
        or market.get("closed")
        or event_data.get("resolved")
        or event_data.get("closed")
    )
    best = ("", 0.0)
    for i, outcome in enumerate(outcomes):
        if i >= len(outcome_prices):
            continue
        try:
            price = float(outcome_prices[i])
        except Exception:
            continue
        side = _normalize_outcome(outcome)
        if side and price > best[1]:
            best = (side, price)

    if best[0] and (best[1] >= 0.95 or resolved and best[1] >= 0.90):
        return best[0]
    return None


def get_market_winner(period_minutes: int = 5, window_ts: int = None, asset: str = "btc") -> Optional[str]:
    """Fetch the Polymarket final winner for a crypto Up/Down window."""
    event = fetch_market_by_slug(market_slug(period_minutes, window_ts, asset=asset))
    if not event:
        return None
    return extract_winning_outcome(event)


def extract_token_ids(event_data: dict) -> tuple[str, str]:
    """Extract UP and DOWN token IDs from event data.
    
    Returns (token_id_up, token_id_down).
    
    Polymarket 5-min markets have a single market with:
      outcomes: ["Up", "Down"]
      clobTokenIds: ["<up_token>", "<down_token>"]  (JSON string)
    """
    markets = event_data.get("markets", [])
    if len(markets) < 1:
        raise ValueError("No markets found in event data")
    
    market = markets[0]
    
    # clobTokenIds comes as a JSON string — parse it
    clob_tokens = market.get("clobTokenIds", [])
    if isinstance(clob_tokens, str):
        clob_tokens = json.loads(clob_tokens)
    
    if len(clob_tokens) < 2:
        raise ValueError(f"Expected 2 token IDs, got {len(clob_tokens)}")
    
    # outcomes: ["Up", "Down"] — tokens are in same order
    outcomes = market.get("outcomes", "")
    if isinstance(outcomes, str):
        outcomes = json.loads(outcomes)
    
    # Map tokens to Up/Down based on outcomes order
    token_up = None
    token_down = None
    
    for i, outcome in enumerate(outcomes):
        if outcome.lower() == "up":
            token_up = clob_tokens[i]
        elif outcome.lower() == "down":
            token_down = clob_tokens[i]
    
    # Fallback: assume index 0 = Up, index 1 = Down
    if token_up is None:
        token_up = clob_tokens[0]
    if token_down is None:
        token_down = clob_tokens[1]
    
    return token_up, token_down


def get_current_market(period_minutes: int = 5, asset: str = "btc") -> Optional[MarketWindow]:
    """Get the current active 5-min crypto market with all required info."""
    wts = current_window_ts(period_minutes)
    slug = market_slug(period_minutes, wts, asset=asset)
    period = PERIOD_SECONDS[period_minutes]
    
    event = fetch_market_by_slug(slug)
    if not event:
        # Try the next window (market might already be created for upcoming)
        return None
    
    try:
        token_up, token_down = extract_token_ids(event)
    except ValueError as e:
        print(f"[market] {e}")
        return None
    
    condition_id = event.get("markets", [{}])[0].get("conditionId", "")
    
    # Parse outcome prices if available
    market_data = event.get("markets", [{}])[0]
    outcome_prices = market_data.get("outcomePrices", "")
    if isinstance(outcome_prices, str) and outcome_prices:
        outcome_prices = json.loads(outcome_prices)
    
    outcomes = market_data.get("outcomes", "")
    if isinstance(outcomes, str) and outcomes:
        outcomes = json.loads(outcomes)
    
    up_price = 0.50
    down_price = 0.50
    if isinstance(outcomes, list) and isinstance(outcome_prices, list):
        for i, outcome in enumerate(outcomes):
            if i < len(outcome_prices):
                if outcome.lower() == "up":
                    up_price = float(outcome_prices[i])
                elif outcome.lower() == "down":
                    down_price = float(outcome_prices[i])
    
    return MarketWindow(
        slug=slug,
        condition_id=condition_id,
        token_id_up=token_up,
        token_id_down=token_down,
        window_start=wts,
        window_end=wts + period,
        up_price=up_price,
        down_price=down_price,
    )


if __name__ == "__main__":
    # Quick test
    print(f"Current window ts: {current_window_ts()}")
    print(f"Market slug: {market_slug()}")
    print(f"Next window in: {next_window_ts() - time.time():.0f}s")
    
    market = get_current_market()
    if market:
        print(f"\nActive market: {market.slug}")
        print(f"Token UP:   {market.token_id_up[:20]}...")
        print(f"Token DOWN: {market.token_id_down[:20]}...")
        print(f"Closes in:  {market.seconds_remaining:.0f}s")
    else:
        print("\nNo active market found (may be between windows)")
