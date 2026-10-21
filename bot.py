#!/usr/bin/env python3
"""
Kraken swarm bot: one independent async worker per coin.

Spot, long-only. PAPER TRADING BY DEFAULT.

Install:
    pip install aiohttp

Run paper mode:
    python bot.py

Run live mode:
    LIVE=1 KRAKEN_API_KEY=... KRAKEN_API_SECRET=... python bot.py

On Windows PowerShell:
    $env:LIVE="1"
    $env:KRAKEN_API_KEY="..."
    $env:KRAKEN_API_SECRET="..."
    python bot.py
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import signal
import time
import urllib.parse
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import aiohttp

# ------------------------------ config ------------------------------

def env(name, default, cast=float):
    return cast(os.getenv(name, default))

LIVE = False
QUOTE = os.getenv("QUOTE", "USD")

N_COINS = env("N_COINS", 30, int)
TOTAL_ALLOC = env("TOTAL_ALLOC", 1000)
MAX_OPEN = env("MAX_OPEN", 10, int)
DAILY_LOSS_LIMIT = env("DAILY_LOSS_LIMIT", 0.03)

TIMEFRAME_MINUTES = env("TIMEFRAME_MINUTES", 5, int)
FEE = env("FEE", 0.0040)
SLIPPAGE = env("SLIPPAGE", 0.0005)

STOP_ATR, TP_ATR, TRAIL_ATR = 2.0, 3.5, 2.0
STATE_FILE = os.getenv("STATE_FILE", "state.json")

REST_URL = "https://api.kraken.com/0"
WS_PUBLIC_URL = "wss://ws.kraken.com/v2"

STABLES = {
    "USD", "USDT", "USDC", "DAI", "PYUSD", "EUR", "GBP", "AUD",
    "TUSD", "EURT", "USDG", "RLUSD", "USDP", "BUSD", "FDUSD",
}

# Kraken REST legacy asset code -> Kraken WebSocket v2 asset code.
WS_SYMBOL_RENAMES = {
    "XBT": "BTC",
    "XXBT": "BTC",
    "XDG": "DOGE",
    "XXDG": "DOGE",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("swarm")


def to_ws_v2_symbol(symbol: str) -> str:
    """
    Convert a Kraken REST wsname such as 'XDG/USD' or 'XXBT/ZUSD'
    into a WebSocket v2 symbol such as 'DOGE/USD' or 'BTC/USD'.
    """
    if "/" not in symbol:
        return symbol

    base, quote = symbol.split("/", 1)

    base = WS_SYMBOL_RENAMES.get(base, base)
    quote = WS_SYMBOL_RENAMES.get(quote, quote)

    return f"{base}/{quote}"


# ------------------------------ indicators ------------------------------

def ema(values, period):
    if len(values) < period:
        return []

    seed = sum(values[:period]) / period
    result = [seed]
    previous = seed
    multiplier = 2 / (period + 1)

    for value in values[period:]:
        previous = value * multiplier + previous * (1 - multiplier)
        result.append(previous)

    return result


def rsi(closes, period=14):
    if len(closes) <= period:
        return 50.0

    gains = 0.0
    losses = 0.0

    for i in range(1, period + 1):
        change = closes[i] - closes[i - 1]
        if change > 0:
            gains += change
        else:
            losses -= change

    average_gain = gains / period
    average_loss = losses / period

    for i in range(period + 1, len(closes)):
        change = closes[i] - closes[i - 1]

        gain = max(change, 0.0)
        loss = max(-change, 0.0)

        average_gain = (average_gain * (period - 1) + gain) / period
        average_loss = (average_gain * (period - 1) + loss) / period

    if average_loss == 0:
        return 100.0

    relative_strength = average_gain / average_loss
    return 100 - 100 / (1 + relative_strength)


def atr(candles, period=14):
    if len(candles) < period + 1:
        return 0.0

    true_ranges = []

    for previous, current in zip(candles, candles[1:]):
        high = current[2]
        low = current[3]
        previous_close = previous[4]

        true_range = max(
            high - low,
            abs(high - previous_close),
            abs(low - previous_close),
        )
        true_ranges.append(true_range)

    value = sum(true_ranges[:period]) / period

    for true_range in true_ranges[period:]:
        value = (value * (period - 1) + true_range) / period

    return value


# ------------------------------ persistence / risk ------------------------------

@dataclass
class Position:
    qty: float
    entry: float
    stop: float
    tp: float
    atr: float
    fee_paid: float


class RiskManager:
    def __init__(self):
        self.day = datetime.now(timezone.utc).date()
        self.pnl_today = 0.0
        self.total_pnl = 0.0
        self.open: dict[str, Position] = {}

    def roll_day(self):
        today = datetime.now(timezone.utc).date()

        if today != self.day:
            self.day = today
            self.pnl_today = 0.0

    def can_open(self):
        self.roll_day()

        if self.pnl_today <= -DAILY_LOSS_LIMIT * TOTAL_ALLOC:
            return False

        return len(self.open) < MAX_OPEN

    def record(self, pnl):
        self.roll_day()
        self.pnl_today += pnl
        self.total_pnl += pnl

    def save(self):
        payload = {
            "open": {
                symbol: asdict(position)
                for symbol, position in self.open.items()
            },
            "total_pnl": self.total_pnl,
        }

        temporary = STATE_FILE + ".tmp"

        with open(temporary, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

        os.replace(temporary, STATE_FILE)

    def load(self):
        if not os.path.exists(STATE_FILE):
            return

        try:
            with open(STATE_FILE, "r", encoding="utf-8") as file:
                payload = json.load(file)
        except (OSError, json.JSONDecodeError) as error:
            log.warning("could not load state file: %s", error)
            return

        self.open = {
            symbol: Position(**raw)
            for symbol, raw in payload.get("open", {}).items()
        }
        self.total_pnl = payload.get("total_pnl", 0.0)

        log.info("restored %d open positions", len(self.open))


# ------------------------------ Kraken REST/WS client ------------------------------

class KrakenClient:
    def __init__(self, session: aiohttp.ClientSession):
        self.session = session
        self.api_key = os.getenv("KRAKEN_API_KEY", "")
        self.api_secret = os.getenv("KRAKEN_API_SECRET", "")
        self.markets: dict[str, dict] = {}

    async def public_get(self, path, params=None):
        url = f"{REST_URL}{path}"

        async with self.session.get(url, params=params or {}) as response:
            payload = await response.json()

        if payload.get("error"):
            raise RuntimeError(f"Kraken {path}: {payload['error']}")

        return payload.get("result", {})

    async def private_post(self, path, payload):
        if not LIVE:
            raise RuntimeError("private_post called while in paper mode")

        if not self.api_key or not self.api_secret:
            raise RuntimeError("missing KRAKEN_API_KEY or KRAKEN_API_SECRET")

        nonce = str(int(time.time() * 1000))
        payload = dict(payload)
        payload["nonce"] = nonce

        body = urllib.parse.urlencode(payload)

        sha256_hash = hashlib.sha256(
            (nonce + body).encode("utf-8")
        ).digest()

        signature = hmac.new(
            base64.b64decode(self.api_secret),
            path.encode("utf-8") + sha256_hash,
            hashlib.sha512,
        ).digest()

        headers = {
            "API-Key": self.api_key,
            "API-Sign": base64.b64encode(signature).decode("utf-8"),
            "Content-Type": "application/x-www-form-urlencoded",
        }

        url = f"{REST_URL}{path}"

        async with self.session.post(url, data=body, headers=headers) as response:
            result = await response.json()

        if result.get("error"):
            raise RuntimeError(f"Kraken {path}: {result['error']}")

        return result.get("result", {})

    async def load_markets(self):
        result = await self.public_get("/public/AssetPairs")
        self.markets = {}

        accepted_quotes = {QUOTE, f"Z{QUOTE}"}
        log.info("looking for quote assets: %s", sorted(accepted_quotes))

        for wsname, market in result.items():
            status = market.get("status")
            if status is not None and status != "online":
                continue

            base = market.get("base")
            quote = market.get("quote")
            altname = market.get("altname") or wsname

            if not base or not quote:
                continue

            if quote not in accepted_quotes:
                continue

            if base in STABLES:
                continue

            base_normalized = base.removeprefix("X")
            quote_normalized = quote.removeprefix("Z")

            if quote_normalized != QUOTE:
                continue

            symbol = altname.replace("/", "")
            rest_wsname = market.get("wsname") or altname

            self.markets[symbol] = {
                "wsname": rest_wsname,
                "ws_v2_symbol": to_ws_v2_symbol(rest_wsname),
                "altname": altname,
                "rest_symbol": symbol,
                "base": base_normalized,
                "quote": quote_normalized,
                "lot_decimals": int(market.get("lot_decimals", 8)),
                "ordermin": float(market.get("ordermin", 0) or 0),
            }

        log.info(
            "loaded %d eligible %s spot markets; examples: %s",
            len(self.markets),
            QUOTE,
            list(self.markets)[:10],
        )

        if not self.markets:
            sample = list(result.items())[:5]
            log.error("No markets matched. Raw AssetPairs sample: %s", sample)

    async def pick_universe(self):
        tickers = await self.public_get("/public/Ticker")
        candidates = []

        log.info("requesting tickers for %d markets", len(self.markets))

        for symbol, market in self.markets.items():
            ticker = tickers.get(symbol)

            if not ticker:
                ticker = tickers.get(market["altname"])

            if not ticker:
                continue

            try:
                base_volume_24h = float(ticker["v"][1])
                last_price = float(ticker["c"][0])
            except (KeyError, IndexError, TypeError, ValueError):
                continue

            quote_volume = base_volume_24h * last_price

            if quote_volume > 0:
                candidates.append((quote_volume, symbol))

        candidates.sort(reverse=True)
        selected = [symbol for _, symbol in candidates[:N_COINS]]

        log.info(
            "selected %d markets: %s",
            len(selected),
            selected,
        )

        return selected

    async def fetch_candles(self, symbol, interval_minutes, limit=300):
        result = await self.public_get(
            "/public/OHLC",
            {
                "pair": symbol,
                "interval": interval_minutes,
            },
        )

        candles = None

        for key, value in result.items():
            if key == "last":
                continue

            if isinstance(value, list):
                candles = value
                break

        if not candles:
            return []

        parsed = []

        for row in candles:
            try:
                parsed.append([
                    int(row[0]),
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    float(row[4]),
                    float(row[6]),
                ])
            except (IndexError, TypeError, ValueError):
                continue

        return parsed[-limit:]

    async def market_order(self, symbol, side, qty, reference_price):
        if not LIVE:
            slippage = SLIPPAGE if side == "buy" else -SLIPPAGE
            fill_price = reference_price * (1 + slippage)
            fee = fill_price * qty * FEE
            return fill_price, qty, fee

        market = self.markets[symbol]
        decimals = market["lot_decimals"]
        amount = f"{qty:.{decimals}f}"

        if float(amount) < market["ordermin"]:
            log.warning(
                "%s quantity %s is below exchange minimum %s",
                symbol,
                amount,
                market["ordermin"],
            )
            return None

        result = await self.private_post(
            "/private/AddOrder",
            {
                "pair": symbol,
                "type": side,
                "ordertype": "market",
                "volume": amount,
            },
        )

        txids = result.get("txid") or []
        txid = txids[0] if txids else None

        if not txid:
            log.warning("%s order accepted without txid: %s", symbol, result)
            fee = reference_price * qty * FEE
            return reference_price, qty, fee

        for _ in range(10):
            await asyncio.sleep(1.0)

            try:
                orders = await self.private_post(
                    "/private/QueryOrders",
                    {"txid": txid},
                )

                order = orders.get(txid)

                if not order:
                    continue

                status = order.get("status")

                if status in ("closed", "canceled"):
                    filled = float(order.get("vol_exec", qty))
                    price = float(order.get("price", reference_price))
                    fee = float(order.get("fee", 0) or 0)

                    if fee == 0:
                        fee = price * filled * FEE

                    return price, filled, fee

            except Exception as error:
                log.warning("%s order query failed: %s", symbol, error)

        log.warning(
            "%s could not confirm execution; assuming reference-price fill",
            symbol,
        )
        fee = reference_price * qty * FEE
        return reference_price, qty, fee


# ------------------------------ broker ------------------------------

class Broker:
    def __init__(self, client: KrakenClient):
        self.client = client

    async def market(self, symbol, side, qty, reference_price):
        return await self.client.market_order(
            symbol,
            side,
            qty,
            reference_price,
        )


# ------------------------------ coin worker ------------------------------

class CoinWorker:
    def __init__(
        self,
        client: KrakenClient,
        broker: Broker,
        risk: RiskManager,
        symbol: str,
        budget: float,
        stop_event: asyncio.Event,
    ):
        self.client = client
        self.broker = broker
        self.risk = risk
        self.symbol = symbol

        market = client.markets[symbol]

        self.rest_wsname = market["wsname"]
        self.wsname = market["ws_v2_symbol"]
        self.budget = budget
        self.stop_event = stop_event

        self.candles: dict[int, list] = {}
        self.last_eval = 0
        self.last_data = time.time()

        # If no real OHLC update arrives for this long, reconnect.
        self.stale_timeout = max(120, TIMEFRAME_MINUTES * 60 * 3)

    async def run(self):
        backoff = 1

        while not self.stop_event.is_set():
            try:
                await self.seed_candles()
                await self.stream_candles()

                backoff = 1

            except asyncio.CancelledError:
                raise

            except Exception as error:
                log.warning(
                    "[%s] worker error: %s; reconnecting in %ss",
                    self.symbol,
                    error,
                    backoff,
                )

                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    async def seed_candles(self):
        candles = await self.client.fetch_candles(
            self.symbol,
            TIMEFRAME_MINUTES,
            limit=300,
        )

        self.candles = {candle[0]: candle for candle in candles}
        self.last_data = time.time()

        log.info(
            "[%s] seeded %d candles; REST=%s WS=%s",
            self.symbol,
            len(self.candles),
            self.rest_wsname,
            self.wsname,
        )

    async def stream_candles(self):
        timeout = aiohttp.ClientTimeout(
            total=None,
            sock_connect=20,
            sock_read=90,
        )

        async with self.client.session.ws_connect(
            WS_PUBLIC_URL,
            timeout=timeout,
        ) as websocket:
            await websocket.send_json({
                "method": "subscribe",
                "params": {
                    "channel": "ohlc",
                    "symbol": [self.wsname],
                    "interval": TIMEFRAME_MINUTES,
                    "snapshot": True,
                },
            })

            self.last_data = time.time()
            self.last_status_log = time.time()

            reader_task = asyncio.create_task(
                self.read_websocket(websocket)
            )
            watchdog_task = asyncio.create_task(
                self.watchdog()
            )

            done, pending = await asyncio.wait(
                {reader_task, watchdog_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()

            # Re-raise the actual failure instead of silently hanging.
            for task in done:
                result = task.result()

                if result is not None:
                    raise RuntimeError(result)

            raise RuntimeError("websocket worker exited unexpectedly")

    async def read_websocket(self, websocket):
        message_count = 0

        async for message in websocket:
            if self.stop_event.is_set():
                return None

            if message.type != aiohttp.WSMsgType.TEXT:
                if message.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.ERROR,
                ):
                    raise RuntimeError(f"websocket closed: {message}")

                continue

            message_count += 1

            if message_count <= 5:
                log.info(
                    "[%s] WS message %d: %s",
                    self.symbol,
                    message_count,
                    message.data[:300],
                )

            payload = json.loads(message.data)

            self.handle_websocket_message(payload)
            await self.tick()

        return None

    async def watchdog(self):
        while not self.stop_event.is_set():
            await asyncio.sleep(5)

            if time.time() - self.last_data > self.stale_timeout:
                return (
                    f"no OHLC data for {self.stale_timeout}s; "
                    f"last message age={time.time() - self.last_data:.1f}s"
                )

            if time.time() - self.last_status_log >= 300:
                self.last_status_log = time.time()

                latest = list(self.candles.values())[-1] if self.candles else None

                log.info(
                    "[%s] alive: candles=%d last_ohlc_age=%.1fs price=%.8f",
                    self.symbol,
                    len(self.candles),
                    time.time() - self.last_data,
                    latest[4] if latest else 0.0,
                )

        return None

    def handle_websocket_message(self, payload):
        if not isinstance(payload, dict):
            return

        if payload.get("channel") == "heartbeat":
            return

        if payload.get("method") == "subscribe":
            if payload.get("success") is False:
                raise RuntimeError(
                    payload.get(
                        "error",
                        "Kraken WebSocket subscription rejected",
                    )
                )

            if payload.get("success") is True:
                log.info(
                    "[%s] WebSocket subscribed: %s",
                    self.symbol,
                    self.wsname,
                )

            return

        if payload.get("channel") != "ohlc":
            return

        updated = False

        for candle in payload.get("data", []):
            if candle.get("symbol") != self.wsname:
                continue

            try:
                # Kraken WS v2 OHLC timestamps are milliseconds.
                # REST OHLC timestamps are seconds.
                timestamp = int(candle["timestamp"])

                if timestamp > 10_000_000_000:
                    timestamp //= 1000

                row = [
                    timestamp,
                    float(candle["open"]),
                    float(candle["high"]),
                    float(candle["low"]),
                    float(candle["close"]),
                    float(candle["volume"]),
                ]

            except (KeyError, TypeError, ValueError):
                continue

            self.candles[timestamp] = row
            updated = True

        if updated:
            self.last_data = time.time()

    async def tick(self):
        if len(self.candles) < 60:
            return

        candles = [
            self.candles[timestamp]
            for timestamp in sorted(self.candles)
        ][-300:]

        self.candles = {candle[0]: candle for candle in candles}

        price = candles[-1][4]

        await self.manage_position(price)

        closed_candles = candles[:-1]

        if (
            len(closed_candles) > 60
            and closed_candles[-1][0] != self.last_eval
        ):
            self.last_eval = closed_candles[-1][0]
            await self.evaluate(closed_candles, price)

    async def manage_position(self, price):
        position = self.risk.open.get(self.symbol)

        if not position:
            return

        if price > position.entry + position.atr:
            trailing_stop = price - TRAIL_ATR * position.atr
            position.stop = max(position.stop, trailing_stop)

        if price <= position.stop:
            await self.sell(price, "stop")

        elif price >= position.tp:
            await self.sell(price, "take-profit")

    async def evaluate(self, closed_candles, price):
        closes = [candle[4] for candle in closed_candles]

        fast_series = ema(closes, 9)
        slow_series = ema(closes, 21)
        trend_series = ema(closes, 50)

        if (
            len(fast_series) < 2
            or len(slow_series) < 2
            or not trend_series
        ):
            return

        fast_previous = fast_series[-2]
        fast_current = fast_series[-1]

        slow_previous = slow_series[-2]
        slow_current = slow_series[-1]

        trend = trend_series[-1]

        rsi_value = rsi(closes)
        atr_value = atr(closed_candles)

        crossed_up = (
            fast_previous <= slow_previous
            and fast_current > slow_current
        )

        crossed_down = (
            fast_previous >= slow_previous
            and fast_current < slow_current
        )

        position = self.risk.open.get(self.symbol)

        if position and crossed_down:
            await self.sell(price, "trend-exit")

        elif (
            not position
            and crossed_up
            and 50 < rsi_value < 70
            and closes[-1] > trend
        ):
            if atr_value <= 0:
                return

            expected_move_ratio = (TP_ATR * atr_value) / price

            if expected_move_ratio < 3 * FEE:
                return

            await self.buy(price, atr_value)

    async def buy(self, price, atr_value):
        if self.symbol in self.risk.open:
            return

        if not self.risk.can_open():
            return

        quantity = self.budget / price

        fill = await self.broker.market(
            self.symbol,
            "buy",
            quantity,
            price,
        )

        if not fill:
            return

        fill_price, filled_qty, fee = fill

        self.risk.open[self.symbol] = Position(
            qty=filled_qty,
            entry=fill_price,
            stop=fill_price - STOP_ATR * atr_value,
            tp=fill_price + TP_ATR * atr_value,
            atr=atr_value,
            fee_paid=fee,
        )

        self.risk.save()

        log.info(
            "[BUY ] %-12s qty=%.8f @ %.8f",
            self.symbol,
            filled_qty,
            fill_price,
        )

    async def sell(self, price, reason):
        position = self.risk.open.get(self.symbol)

        if not position:
            return

        fill = await self.broker.market(
            self.symbol,
            "sell",
            position.qty,
            price,
        )

        if not fill:
            return

        fill_price, filled_qty, fee = fill

        pnl = (
            (fill_price - position.entry) * filled_qty
            - fee
            - position.fee_paid
        )

        self.risk.record(pnl)
        self.risk.open.pop(self.symbol, None)
        self.risk.save()

        log.info(
            "[SELL] %-12s @ %.8f %-11s pnl=%+.4f | day=%+.2f total=%+.2f",
            self.symbol,
            fill_price,
            reason,
            pnl,
            self.risk.pnl_today,
            self.risk.total_pnl,
        )


# ------------------------------ main ------------------------------

async def main():
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    if os.name != "nt":
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop_event.set)

    log.info("mode: %s", "LIVE (real money)" if LIVE else "PAPER")

    timeout = aiohttp.ClientTimeout(total=30)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        client = KrakenClient(session)

        if LIVE and not (client.api_key and client.api_secret):
            raise RuntimeError(
                "LIVE=1 requires KRAKEN_API_KEY and KRAKEN_API_SECRET"
            )

        await client.load_markets()

        risk = RiskManager()
        risk.load()

        symbols = await client.pick_universe()

        if not symbols:
            raise RuntimeError("no eligible markets found")

        budget = TOTAL_ALLOC / len(symbols)

        log.info(
            "launching %d parallel workers, %.2f %s each: %s",
            len(symbols),
            budget,
            QUOTE,
            symbols,
        )

        broker = Broker(client)

        workers = [
            CoinWorker(
                client=client,
                broker=broker,
                risk=risk,
                symbol=symbol,
                budget=budget,
                stop_event=stop_event,
            )
            for symbol in symbols
        ]

        tasks = [
            asyncio.create_task(worker.run())
            for worker in workers
        ]

        try:
            await stop_event.wait()

        except KeyboardInterrupt:
            log.info("Ctrl+C received")

        finally:
            stop_event.set()

            log.info(
                "shutting down; open positions remain saved in %s",
                STATE_FILE,
            )

            for task in tasks:
                task.cancel()

            await asyncio.gather(*tasks, return_exceptions=True)
            risk.save()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
