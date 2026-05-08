#!/usr/bin/env python3
"""Fast CLOB auth/config diagnostic.

By default, this script does not place an order. It derives or loads the same
API credentials the bot will use, prints the signer/funder/API-key
relationship, checks balance, and optionally builds a local signed order to
expose the exact order signer before a live POST.
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from executor import _load_manual_api_creds, _signature_type, _order_options  # noqa: E402
from market import get_current_market  # noqa: E402
from py_clob_client_v2 import (  # noqa: E402
    AssetType,
    BalanceAllowanceParams,
    ClobClient,
    OrderArgs,
)
from py_clob_client_v2.constants import POLYGON  # noqa: E402


def env(name: str) -> str:
    return os.getenv(name, "").strip()


def redacted(value: str) -> str:
    if not value:
        return "<empty>"
    if len(value) <= 10:
        return "***"
    return f"{value[:6]}...{value[-4:]}"


def same_address(a: str, b: str) -> bool:
    return bool(a and b and a.lower() == b.lower())


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Polymarket CLOB auth config")
    parser.add_argument(
        "--env-file",
        default=str(ROOT / ".env"),
        help="Path to .env file. Default: project .env",
    )
    parser.add_argument(
        "--build-order",
        action="store_true",
        help="Build a local signed order using the current market token. Does not POST.",
    )
    parser.add_argument(
        "--post-smoke-test",
        action="store_true",
        help=(
            "Place a post-only $5.00 limit order at $0.01, then cancel it. "
            "This is live and should only be used with DRY_RUN=true after review."
        ),
    )
    args = parser.parse_args()

    load_dotenv(args.env_file)

    private_key = env("PRIVATE_KEY")
    safe_address = env("SAFE_ADDRESS")
    funder = env("FUNDER_ADDRESS") or safe_address
    sig_type = _signature_type(int(env("SIGNATURE_TYPE") or "2"))
    creds_mode = (env("CLOB_CREDS_MODE") or "auto").lower()

    if not private_key:
        print("ERROR: PRIVATE_KEY is empty")
        return 2

    print("== .env summary ==")
    print(f"CLOB_CREDS_MODE: {creds_mode}")
    print(f"SIGNATURE_TYPE: {int(sig_type)} ({sig_type.name})")
    print(f"SAFE_ADDRESS: {safe_address or '<empty>'}")
    print(f"FUNDER_ADDRESS: {funder or '<empty>'}")
    print(f"CLOB_API_KEY: {redacted(env('CLOB_API_KEY'))}")
    print(f"CLOB_SECRET: {redacted(env('CLOB_SECRET'))}")
    print(f"CLOB_PASS_PHRASE: {redacted(env('CLOB_PASS_PHRASE'))}")

    manual_creds = _load_manual_api_creds()
    client = ClobClient(
        host="https://clob.polymarket.com",
        key=private_key,
        chain_id=POLYGON,
        creds=manual_creds,
        funder=funder or None,
        signature_type=sig_type,
    )

    signer = client.get_address()
    if client.creds is None:
        print("\nDeriving CLOB API credentials from PRIVATE_KEY...")
        client.set_api_creds(client.create_or_derive_api_key())
    else:
        print("\nUsing manual CLOB credentials from .env...")

    api_key = client.creds.api_key

    print("\n== resolved CLOB auth ==")
    print(f"wallet signer: {signer}")
    print(f"order funder/maker: {funder or signer}")
    print(f"CLOB API key/owner: {api_key}")

    if api_key.startswith("0x") and not same_address(api_key, signer):
        print("\nPROBLEM FOUND:")
        print("The order signer and CLOB API key owner do not match.")
        print("This causes: 'the order signer address has to be the address of the API KEY'")
        print("\nFix:")
        print("1. Set CLOB_CREDS_MODE=auto")
        print("2. Clear CLOB_API_KEY, CLOB_SECRET, and CLOB_PASS_PHRASE")
        print("3. Restart the bot so it derives CLOB creds from PRIVATE_KEY")
        return 1

    print("\nOK: CLOB credentials were resolved.")

    try:
        balance = client.get_balance_allowance(
            BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        )
        raw_balance = float(balance.get("balance", 0))
        print(f"USDC balance: ${raw_balance / 1e6:.2f}")
    except Exception as exc:
        print(f"Balance check failed: {exc}")

    if args.build_order or args.post_smoke_test:
        market = get_current_market()
        if market is None:
            print("Could not fetch current market; try again in a few seconds.")
            return 1

        token_id = market.token_id_down or market.token_id_up
        signed_order = client.create_order(
            OrderArgs(token_id=token_id, price=0.01, size=500.0, side="BUY"),
            options=_order_options(),
        )
        print("\n== local signed order ==")
        print(f"order signer: {signed_order.signer}")
        print(f"order maker: {signed_order.maker}")
        print(f"order signature type: {signed_order.signatureType}")
        print(f"payload owner would be: {api_key}")
        if int(signed_order.signatureType) == 3 and not same_address(
            signed_order.signer, signed_order.maker
        ):
            print("\nPROBLEM FOUND:")
            print("SIGNATURE_TYPE=3 built an order whose signer is not the funded maker.")
            print("Your account is behaving like a Safe/proxy wallet, not a 1271 deposit wallet.")
            print("Set SIGNATURE_TYPE=2 and rerun this script.")
            return 1
        if api_key.startswith("0x") and not same_address(api_key, signed_order.signer):
            print("PROBLEM FOUND: order signer != payload owner")
            return 1
        print("OK: local order signer matches payload owner.")

        if args.post_smoke_test:
            from py_clob_client_v2 import OrderPayload, OrderType

            dry_run = env("DRY_RUN").lower() != "false"
            if not dry_run:
                print("\nRefusing smoke test while DRY_RUN=false.")
                print("Set DRY_RUN=true for this one-off auth smoke test.")
                return 2

            print("\n== live CLOB POST smoke test ==")
            print("Posting post-only BUY 500 shares @ $0.01 ($5.00), then cancelling.")
            response = client.post_order(
                signed_order,
                order_type=OrderType.GTC,
                post_only=True,
            )
            order_id = response.get("orderID", "")
            print(f"POST succeeded: {response}")
            if order_id:
                cancel_response = client.cancel_order(OrderPayload(orderID=order_id))
                print(f"Cancel response: {cancel_response}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
