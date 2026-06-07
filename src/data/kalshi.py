import base64
import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx
import pandas as pd
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from src.config.settings import settings, PROJECT_ROOT

BASE_URL = "https://external-api.kalshi.com/trade-api/v2"


def _load_private_key(pem_str: str):
    return serialization.load_pem_private_key(
        pem_str.encode() if isinstance(pem_str, str) else pem_str,
        password=None,
    )


def _sign_request(method: str, path: str, secret_key: str) -> tuple[str, str]:
    timestamp_ms = str(int(time.time() * 1000))
    path_without_query = path.split("?")[0]
    message = f"{timestamp_ms}{method}{path_without_query}".encode("utf-8")

    private_key = _load_private_key(secret_key)
    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    sig_b64 = base64.b64encode(signature).decode("utf-8")
    return timestamp_ms, sig_b64


class KalshiClient:
    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("KALSHI_API_KEY", "")
        self.secret_key = secret_key or self._load_secret_key()
        self.client = httpx.Client(base_url=BASE_URL, timeout=30)
        self._rate_limit_block = []

    def _load_secret_key(self) -> str:
        key_file = os.getenv("KALSHI_SECRET_KEY_FILE", "")
        if key_file:
            path = Path(key_file)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            if path.exists():
                return path.read_text().strip()
        return os.getenv("KALSHI_SECRET_KEY", "")

    def _rate_limit(self):
        now = time.time()
        window = settings.kalshi.rate_limit
        self._rate_limit_block = [t for t in self._rate_limit_block if now - t < window]
        if len(self._rate_limit_block) >= window:
            wait = self._rate_limit_block[0] + window - now
            if wait > 0:
                time.sleep(wait)
        self._rate_limit_block.append(now)

    def _request(self, method: str, path: str, params: dict = None, body: str = "") -> dict:
        self._rate_limit()
        request_path = path
        signing_path = f"/trade-api/v2{path}"
        timestamp_ms, signature = _sign_request(method, signing_path, self.secret_key)
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "sports-betting-ai/0.1.0",
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
        }
        resp = self.client.request(
            method, request_path, headers=headers, params=params, content=body
        )
        if resp.status_code == 429:
            time.sleep(5)
            return self._request(method, path, params, body)
        if resp.status_code >= 400:
            print(f"  Kalshi API error {resp.status_code}: {resp.text[:500]}", flush=True)
        resp.raise_for_status()
        return resp.json() if resp.content else {}

    # --- Market Data ---

    def list_events(self, status: str = "open", limit: int = 100,
                    series_ticker: str = None, ticker_prefix: str = None) -> pd.DataFrame:
        params = {"status": status, "limit": limit}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if ticker_prefix:
            params["ticker_prefix"] = ticker_prefix
        data = self._request("GET", "/events", params=params)
        return pd.DataFrame(data.get("events", []))

    def list_markets(self, event_ticker: str = None, limit: int = 100,
                     series_ticker: str = None, ticker_prefix: str = None) -> pd.DataFrame:
        params = {"limit": limit}
        if event_ticker:
            params["event_ticker"] = event_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker
        if ticker_prefix:
            params["ticker_prefix"] = ticker_prefix
        data = self._request("GET", "/markets", params=params)
        return pd.DataFrame(data.get("markets", []))

    # --- Trading (V2) ---

    def create_order(
        self,
        ticker: str,
        side: str,
        yes_price: int,
        count: str,
        order_type: str = "limit",
        client_order_id: str = None,
    ) -> dict:
        price_str = f"{yes_price / 100:.4f}"
        count_str = f"{float(count):.2f}"
        oid = client_order_id or str(uuid.uuid4())

        # V2 API: side="bid" = buy YES, side="ask" = sell YES (:= buy NO)
        # Map old "yes"/"no" side to V2 "bid"/"ask"
        if side == "yes":
            v2_side = "bid"
        elif side == "no":
            # Buying NO = selling YES at 1 - yes_price
            v2_side = "ask"
        else:
            v2_side = side  # pass through if already bid/ask

        body = json.dumps({
            "ticker": ticker,
            "client_order_id": oid,
            "side": v2_side,
            "count": count_str,
            "price": price_str,
            "time_in_force": "good_till_canceled",
            "self_trade_prevention_type": "taker_at_cross",
        })
        return self._request("POST", "/portfolio/events/orders", body=body)

    def cancel_order(self, order_id: str) -> dict:
        # V2: DELETE /portfolio/orders/{order_id}
        # Some old orders had different ID formats, try both
        for fmt in [f"/portfolio/orders/{order_id}",
                    f"/portfolio/events/orders/{order_id}"]:
            try:
                return self._request("DELETE", fmt)
            except Exception:
                continue
        return {"error": "failed to cancel"}

    def get_positions(self) -> pd.DataFrame:
        data = self._request("GET", "/portfolio/positions")
        return pd.DataFrame(data.get("positions", []))

    def get_balance(self) -> float:
        data = self._request("GET", "/portfolio/balance")
        raw = data.get("balance_dollars", data.get("balance", "0"))
        if isinstance(raw, str):
            return float(raw)
        # If it's an int, might be cents (Kalshi native) or dollars
        if isinstance(raw, (int, float)):
            return float(raw) / 100.0
        return 0.0

    # --- Event discovery ---

    def _fetch_categories(self) -> list[str]:
        """Discover available event categories."""
        events = self.list_events(status="open", limit=200)
        cats = events["category"].dropna().unique().tolist() if not events.empty else []
        return cats or ["Politics", "Elections", "Finance", "Climate",
                        "Weather", "Sports", "Entertainment", "Technology"]

    def find_no_side_opportunities(self) -> pd.DataFrame:
        """Find markets with YES price <= 20¢ for NO-side buying (V2)."""
        from src.config.settings import settings

        max_yes_price = settings.kalshi.max_yes_price
        blocked_categories = {"Sports", "Entertainment"}

        events = self.list_events(status="open", limit=30)
        if events.empty:
            return pd.DataFrame()

        # Filter by category (no sports/entertainment for safe compounder)
        events = events[~events["category"].isin(blocked_categories)]

        opportunities = []
        for _, event in events.iterrows():
            event_ticker = event["event_ticker"]
            category = event.get("category", "")
            markets = self.list_markets(event_ticker=event_ticker)
            if markets.empty:
                continue
            for _, m in markets.iterrows():
                try:
                    raw = m.get("yes_ask_dollars", m.get("yes_bid_dollars", "1.0000"))
                    yes_price = float(raw) if isinstance(raw, str) else float(raw)
                except (ValueError, TypeError):
                    continue
                if 0 < yes_price <= max_yes_price:
                    opportunities.append({
                        "ticker": m.get("ticker", ""),
                        "title": event.get("title", ""),
                        "yes_price": yes_price,
                        "category": category,
                        "close_time": m.get("close_time", ""),
                    })

        return pd.DataFrame(opportunities) if opportunities else pd.DataFrame()

    # --- Sports-Specific ---

    def list_sports_events(self, sport: str = None, limit: int = 200) -> pd.DataFrame:
        params = {"limit": limit, "sport": sport} if sport else {"limit": limit}
        events = self._request("GET", "/events", params={"status": "open", **params})
        return pd.DataFrame(events.get("events", []))

    def list_sports_markets_for_events(self, event_tickers: list[str]) -> pd.DataFrame:
        all_markets = []
        for ticker in event_tickers:
            markets = self.list_markets(event_ticker=ticker)
            if not markets.empty:
                markets["event_ticker"] = ticker
                all_markets.append(markets)
        return pd.concat(all_markets, ignore_index=True) if all_markets else pd.DataFrame()

    def get_candles(self, ticker: str, tick_interval: str = "1h", limit: int = 100) -> pd.DataFrame:
        data = self._request("GET", f"/markets/{ticker}/candles",
                             params={"tick_interval": tick_interval, "limit": limit})
        return pd.DataFrame(data.get("candles", []))
