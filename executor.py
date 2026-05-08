"""Order executor for Polymarket CLOB v2.

This module keeps the bot-facing Executor API stable while using
py-clob-client-v2 underneath. The v2 client signs current Exchange V2 orders
and retries once if the CLOB reports an order-version mismatch.
"""

import os
import time
from dataclasses import dataclass
from typing import Optional

from py_clob_client_v2 import (
    ApiCreds,
    AssetType,
    BalanceAllowanceParams,
    ClobClient,
    MarketOrderArgs,
    OrderArgs,
    OrderPayload,
    OrderType,
    PartialCreateOrderOptions,
    SignatureTypeV2,
)
from py_clob_client_v2.constants import POLYGON


FILLED = "FILLED"
PARTIAL = "PARTIAL"
REJECTED = "REJECTED"
FAILED = "FAILED"

MIN_SHARES = 1.0
MIN_AMOUNT_USD = 1.0
MAX_BUY_PRICE = 0.90
POLY_MIN_NOTIONAL = 5.0

DEFAULT_TICK_SIZE = "0.01"
DEFAULT_NEG_RISK = False


@dataclass
class OrderResult:
    success: bool
    order_id: str = ""
    status: str = FAILED
    side: str = ""
    price: float = 0.0
    amount_usd: float = 0.0
    shares: float = 0.0
    shares_remaining: float = 0.0
    token_id: str = ""
    error: str = ""
    dry_run: bool = True


def calculate_order_size(price: float, max_usd: float) -> tuple[float, float]:
    """Return whole shares and clean USD spend for a limit buy."""
    if price <= 0 or max_usd <= 0:
        return 0.0, 0.0

    price_cents = round(price * 100)
    max_usd_cents = int(max_usd * 100)
    if price_cents <= 0:
        return 0.0, 0.0

    max_shares = max_usd_cents // price_cents
    min_notional_cents = int(POLY_MIN_NOTIONAL * 100)
    min_notional_shares = (min_notional_cents + price_cents - 1) // price_cents
    min_required_shares = max(int(MIN_SHARES), min_notional_shares)

    # Whole-share rounding can turn a $5 Kelly budget into a sub-$5 notional
    # at high prices. Round up to the smallest valid Polymarket notional.
    shares = int(max(max_shares, min_required_shares))
    spend = shares * price_cents / 100.0
    return float(shares), spend


def _load_api_creds() -> Optional[ApiCreds]:
    key = os.getenv("CLOB_API_KEY", "")
    secret = os.getenv("CLOB_SECRET", "")
    passphrase = os.getenv("CLOB_PASS_PHRASE", "")
    if key and secret and passphrase:
        return ApiCreds(api_key=key, api_secret=secret, api_passphrase=passphrase)
    return None


def _signature_type(value: int) -> SignatureTypeV2:
    try:
        return SignatureTypeV2(int(value))
    except Exception:
        return SignatureTypeV2.EOA


def _order_options() -> PartialCreateOrderOptions:
    return PartialCreateOrderOptions(
        tick_size=DEFAULT_TICK_SIZE,
        neg_risk=DEFAULT_NEG_RISK,
    )


class Executor:
    def __init__(
        self,
        private_key: str,
        safe_address: str = "",
        dry_run: bool = True,
        signature_type: int = 0,
        funder_address: str = "",
    ):
        self.dry_run = dry_run
        self.private_key = private_key
        self.funder_address = funder_address or safe_address
        self.signature_type = _signature_type(signature_type)
        self.client: Optional[ClobClient] = None
        self._initialized = False

    def initialize(self) -> bool:
        try:
            self.client = ClobClient(
                host="https://clob.polymarket.com",
                key=self.private_key,
                chain_id=POLYGON,
                creds=_load_api_creds(),
                funder=self.funder_address or None,
                signature_type=self.signature_type,
            )
            if self.client.creds is None:
                self.client.set_api_creds(self.client.create_or_derive_api_key())

            self._initialized = True
            print(f"[executor] Initialized ({'DRY RUN' if self.dry_run else 'LIVE'})")
            print(f"[executor] Max buy price: ${MAX_BUY_PRICE:.2f}")
            print(f"[executor] Address: {self.client.get_address()}")
            print(f"[executor] Funder: {self.funder_address or self.client.get_address()}")
            print(f"[executor] Signature type: {int(self.signature_type)} ({self.signature_type.name})")

            try:
                self.client.update_balance_allowance(
                    BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
                )
            except Exception as e:
                print(f"[executor] Balance cache update warning: {e}")
            return True
        except Exception as e:
            print(f"[executor] Init failed: {e}")
            return False

    def get_balance(self) -> float:
        if not self._initialized:
            return 0.0
        try:
            params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            bal = self.client.get_balance_allowance(params)
            return float(bal.get("balance", 0)) / 1e6
        except Exception as e:
            print(f"[executor] Balance check failed: {e}")
            return 0.0

    def get_market_price(self, token_id: str, side: str, amount_usd: float) -> float:
        if not self._initialized:
            return 0.0
        try:
            price = self.client.calculate_market_price(
                token_id=token_id,
                side=side,
                amount=amount_usd,
                order_type=OrderType.GTC,
            )
            return float(price) if price else 0.0
        except Exception as e:
            err = str(e).lower()
            if "no match" not in err and "none" not in err:
                print(f"[executor] Price check failed: {e}")
            return 0.0

    def buy(self, token_id: str, amount_usd: float, price: float = 0.0) -> OrderResult:
        """Buy via v2 resting limit order with integer shares."""
        amount_usd = round(float(amount_usd), 2)
        if amount_usd < MIN_AMOUNT_USD:
            return OrderResult(
                success=False, status=REJECTED,
                error=f"Amount ${amount_usd:.2f} below min", side="BUY",
            )

        if self.dry_run:
            sim_price = 0.55
            return OrderResult(
                success=True, order_id=f"DRY-{int(time.time())}",
                status=FILLED, side="BUY", price=sim_price,
                amount_usd=amount_usd, shares=amount_usd / sim_price,
                token_id=token_id[:16] + "...", dry_run=True,
            )

        if not self._initialized:
            return OrderResult(success=False, status=FAILED, error="Not initialized")

        if price > 0:
            market_price = round(price, 2)
        else:
            market_price = self.get_market_price(token_id, "BUY", amount_usd)
            if market_price <= 0:
                return OrderResult(
                    success=False, status=FAILED,
                    error="Could not get market price", side="BUY",
                    token_id=token_id[:16] + "...",
                )
            market_price = round(market_price, 2)

        if market_price > MAX_BUY_PRICE:
            return OrderResult(
                success=False, status=REJECTED,
                error=f"Price ${market_price:.3f} > cap ${MAX_BUY_PRICE:.2f}",
                side="BUY", price=market_price, token_id=token_id[:16] + "...",
            )

        shares, clean_amount = calculate_order_size(market_price, amount_usd)
        if shares < 1 or clean_amount <= 0:
            return OrderResult(
                success=False, status=REJECTED,
                error=f"Can't afford 1 share at ${market_price:.3f} within ${amount_usd:.2f}",
                side="BUY", price=market_price, token_id=token_id[:16] + "...",
            )
        if clean_amount < POLY_MIN_NOTIONAL:
            return OrderResult(
                success=False, status=REJECTED,
                error=f"Amount ${clean_amount:.2f} < ${POLY_MIN_NOTIONAL:.0f} min",
                side="BUY", price=market_price, token_id=token_id[:16] + "...",
            )

        print(f"  [order] Market price: ${market_price:.3f}/share "
              f"-> {int(shares)} shares for ${clean_amount:.2f}")

        balance_before = self.get_balance()
        try:
            result = self.client.create_and_post_order(
                order_args=OrderArgs(
                    token_id=token_id,
                    price=market_price,
                    size=float(int(shares)),
                    side="BUY",
                ),
                options=_order_options(),
                order_type=OrderType.GTC,
            )

            order_id = result.get("orderID", "")
            if not order_id:
                return OrderResult(
                    success=False, status=REJECTED,
                    error=f"No orderID: {result}", side="BUY", price=market_price,
                    token_id=token_id[:16] + "...",
                )

            time.sleep(5)
            return self._verify_buy_via_balance(
                order_id, market_price, float(shares), token_id, balance_before,
            )
        except Exception as e:
            time.sleep(3)
            balance_after = self.get_balance()
            spent = balance_before - balance_after if balance_before > 0 else 0
            if spent > 1.0:
                actual_shares = spent / market_price if market_price > 0 else 0
                print(f"  [order] Ghost buy: balance dropped ${spent:.2f} despite error")
                return OrderResult(
                    success=True, order_id="ghost-buy",
                    status=FILLED, side="BUY", price=market_price,
                    amount_usd=spent, shares=actual_shares,
                    token_id=token_id[:16] + "...", dry_run=False,
                )
            return OrderResult(
                success=False, status=FAILED, error=str(e),
                side="BUY", price=market_price, token_id=token_id[:16] + "...",
            )

    def _verify_buy_via_balance(
        self, order_id: str, price: float, shares: float,
        token_id: str, balance_before: float,
    ) -> OrderResult:
        for attempt in range(3):
            balance_after = self.get_balance()
            spent = balance_before - balance_after if balance_before > 0 else 0
            if spent > 0.50:
                actual_shares = spent / price if price > 0 else shares
                suffix = f" (attempt {attempt + 1})" if attempt > 0 else ""
                print(f"  [order] Balance verified{suffix}: spent ${spent:.2f} "
                      f"(~{actual_shares:.0f} shares @ ${price:.3f})")
                return OrderResult(
                    success=True, order_id=order_id, status=FILLED,
                    side="BUY", price=price,
                    amount_usd=spent, shares=actual_shares,
                    token_id=token_id[:16] + "...", dry_run=False,
                )

            fill = self._check_order(order_id)
            if fill:
                matched = self._extract_fill(fill, price)
                if matched:
                    suffix = f" (attempt {attempt + 1})" if attempt > 0 else ""
                    print(f"  [order] Order API verified{suffix}: "
                          f"{matched[2]:.0f} shares @ ${matched[0]:.3f}")
                    return OrderResult(
                        success=True, order_id=order_id, status=FILLED,
                        side="BUY", price=matched[0],
                        amount_usd=matched[1], shares=matched[2],
                        token_id=token_id[:16] + "...", dry_run=False,
                    )

            if attempt < 2:
                time.sleep(3)

        print("  [order] Buy unverified after 14s - NOT cancelling")
        return OrderResult(
            success=False, order_id=order_id, status=FAILED,
            error="UNVERIFIED_BUY",
            side="BUY", price=price, amount_usd=shares * price,
            shares=shares, token_id=token_id[:16] + "...",
        )

    def sell(self, token_id: str, shares: float, price: float = 0.0) -> OrderResult:
        """Sell shares via v2 market order, balance verified."""
        sell_shares = int(shares)
        if sell_shares < 1:
            return OrderResult(
                success=False, status=REJECTED,
                error="Less than 1 share", side="SELL",
            )

        if self.dry_run:
            sim_price = price if price > 0 else 0.90
            revenue = sell_shares * sim_price
            return OrderResult(
                success=True, order_id=f"DRY-SELL-{int(time.time())}",
                status=FILLED, side="SELL", price=sim_price,
                amount_usd=revenue, shares=float(sell_shares),
                shares_remaining=0.0,
                token_id=token_id[:16] + "...", dry_run=True,
            )

        if not self._initialized:
            return OrderResult(success=False, status=FAILED, error="Not initialized")

        if price <= 0:
            notional = float(sell_shares) * 0.50
            price = self.get_market_price(token_id, "SELL", notional)
            if price <= 0:
                return OrderResult(
                    success=False, status=FAILED,
                    error="Could not get sell price", side="SELL",
                    token_id=token_id[:16] + "...",
                )

        sell_amount = round(sell_shares * price, 2)
        if sell_amount < POLY_MIN_NOTIONAL:
            return OrderResult(
                success=False, status=REJECTED,
                error=f"Notional ${sell_amount:.2f} < ${POLY_MIN_NOTIONAL:.0f} min - hold to resolution",
                side="SELL", price=price, shares=float(sell_shares),
                shares_remaining=float(sell_shares),
                token_id=token_id[:16] + "...",
            )

        print(f"  [order] Sell: {sell_shares} shares @ ${price:.3f} = ${sell_amount:.2f}")
        balance_before = self.get_balance()

        try:
            result = self.client.create_and_post_market_order(
                order_args=MarketOrderArgs(
                    token_id=token_id,
                    amount=float(sell_shares),
                    side="SELL",
                    price=round(price, 2),
                    order_type=OrderType.GTC,
                ),
                options=_order_options(),
                order_type=OrderType.GTC,
            )
            order_id = result.get("orderID", "")
            time.sleep(2)

            balance_after = self.get_balance()
            received = balance_after - balance_before
            if received > 0.10:
                shares_sold = received / price if price > 0 else 0
                shares_left = max(0, sell_shares - shares_sold)
                status = FILLED if shares_left < 1 else PARTIAL
                if status == PARTIAL:
                    print(f"  [order] Partial fill: sold ~{shares_sold:.0f} of {sell_shares}, "
                          f"~{shares_left:.0f} remaining")
                return OrderResult(
                    success=True, order_id=order_id or "balance-verified",
                    status=status, side="SELL", price=price,
                    amount_usd=received, shares=shares_sold,
                    shares_remaining=shares_left,
                    token_id=token_id[:16] + "...", dry_run=False,
                )

            if order_id:
                fill = self._check_order(order_id)
                if fill:
                    matched = self._extract_fill(fill, price)
                    if matched:
                        return OrderResult(
                            success=True, order_id=order_id, status=FILLED,
                            side="SELL", price=matched[0],
                            amount_usd=matched[1], shares=matched[2],
                            shares_remaining=max(0, sell_shares - matched[2]),
                            token_id=token_id[:16] + "...", dry_run=False,
                        )
                self.cancel_order(order_id)

            return OrderResult(
                success=False, order_id=order_id or "", status=FAILED,
                error="Sell not verified (no balance change)",
                side="SELL", price=price, token_id=token_id[:16] + "...",
            )
        except Exception as e:
            time.sleep(1)
            balance_after = self.get_balance()
            received = balance_after - balance_before
            if received > 0.10:
                shares_sold = received / price if price > 0 else 0
                shares_left = max(0, sell_shares - shares_sold)
                print(f"  [order] Ghost sell: got ${received:.2f} despite error")
                return OrderResult(
                    success=True, order_id="ghost-sell",
                    status=PARTIAL if shares_left >= 1 else FILLED,
                    side="SELL", price=price,
                    amount_usd=received, shares=shares_sold,
                    shares_remaining=shares_left,
                    token_id=token_id[:16] + "...", dry_run=False,
                )

            return OrderResult(
                success=False, status=FAILED, error=str(e),
                side="SELL", price=price, token_id=token_id[:16] + "...",
            )

    def _extract_fill(self, fill: dict, fallback_price: float) -> Optional[tuple]:
        size_matched = float(
            fill.get("size_matched", 0) if isinstance(fill, dict)
            else getattr(fill, "size_matched", 0)
        )
        if size_matched <= 0:
            return None
        fill_price = float(
            fill.get("price", fallback_price) if isinstance(fill, dict)
            else getattr(fill, "price", fallback_price)
        )
        return (fill_price, size_matched * fill_price, size_matched)

    def _check_order(self, order_id: str) -> Optional[dict]:
        if not self._initialized:
            return None
        try:
            return self.client.get_order(order_id)
        except Exception as e:
            print(f"[executor] Order check failed: {e}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        if self.dry_run or not self._initialized:
            return True
        try:
            self.client.cancel_order(OrderPayload(orderID=order_id))
            return True
        except Exception as e:
            print(f"[executor] Cancel failed: {e}")
            return False

    def cancel_all(self) -> bool:
        if self.dry_run or not self._initialized:
            return True
        try:
            self.client.cancel_all()
            return True
        except Exception as e:
            print(f"[executor] Cancel all failed: {e}")
            return False
