"""
KronosTraderBot — Trades based on Kronos OHLCV forecasts stored by kronosbot.

Reads next-day predicted close prices from the `kronos_predictions` table
(written nightly by kronosbot at 22:05 UTC) and takes long/short/flat
positions based on the expected percentage move.

Tickers are loaded dynamically from the DB at startup — whatever kronosbot
predicted last night is what this bot trades. No hardcoded list.

Not backtestable: relies on live DB predictions, not historical yfinance data.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import ClassVar

from tradingbot.utils.botclass import Bot
from tradingbot.utils.db import KronosPrediction, get_db_session
from tradingbot.utils.runner import run_bot

logger = logging.getLogger(__name__)

_FALLBACK_TICKERS = ["SPY", "QQQ", "GLD", "BTC-USD"]

# Forecasts older than this are ignored. kronosbot writes every weeknight, so anything
# older means its CronJob has been failing — and a week-old prediction for a date that
# is merely still in the future is not a signal. It also ages out the rows written
# before the Space's business-day fix, whose target dates land on weekends.
MAX_PREDICTION_AGE_DAYS = 4


def _utc_naive_now() -> datetime:
    """Current UTC time as a naive datetime.

    kronos_predictions.{target_date,prediction_made_at} are bare DateTime columns
    holding naive UTC. Filtering them with an aware datetime makes Postgres coerce
    the comparison through the session time zone — the same shape of bug that froze
    stock_insider_trades for six months.
    """
    return datetime.now(UTC).replace(tzinfo=None)


def _prediction_window() -> tuple[datetime, datetime]:
    """Return (earliest usable target_date, earliest acceptable prediction_made_at)."""
    now = _utc_naive_now()
    tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return tomorrow, now - timedelta(days=MAX_PREDICTION_AGE_DAYS)


def _load_predicted_tickers() -> list[str]:
    """Return all symbols carrying a fresh Kronos prediction for tomorrow or later."""
    try:
        tomorrow, min_made_at = _prediction_window()
        with get_db_session() as session:
            rows = (
                session.query(KronosPrediction.symbol)
                .filter(
                    KronosPrediction.target_date >= tomorrow,
                    KronosPrediction.prediction_made_at >= min_made_at,
                )
                .distinct()
                .all()
            )
        tickers = [r.symbol for r in rows]
        if not tickers:
            logger.warning(
                f"KronosTraderBot: no predictions newer than {MAX_PREDICTION_AGE_DAYS}d — "
                f"is the kronosbot CronJob running? Falling back to defaults (bot will hold)."
            )
            return list(_FALLBACK_TICKERS)
        logger.info(f"KronosTraderBot: loaded {len(tickers)} tickers from DB: {tickers}")
        return tickers
    except Exception as exc:
        logger.warning(f"KronosTraderBot: could not load tickers from DB ({exc}), falling back to defaults")
        return list(_FALLBACK_TICKERS)


class KronosTraderBot(Bot):
    # Measured over the 9.5k forecasts stored between Apr and Aug 2026, anchored to the
    # last bar the model actually saw: next-bar MAE was ~2.5% of price against ~0.6% for
    # assuming no change, direction was right 32% of the time, and predicted-vs-realised
    # return correlation was 0.02. The 2%/1% thresholds this bot shipped with sat *below*
    # that noise floor, so nearly every symbol tripped one on sampling noise alone — 25
    # orders on the 2026-08-24 run. Thresholds are parked above the measured noise until a
    # re-measurement shows Kronos beating the naive baseline; see
    # docs/guides/kronos-forecasting.md#forecast-quality for how to re-run that check.
    param_grid: ClassVar[dict] = {
        "buy_threshold": [0.05, 0.075, 0.10],
        "sell_threshold": [0.05, 0.075, 0.10],
    }

    def __init__(self, buy_threshold: float = 0.05, sell_threshold: float = 0.05, **kwargs):
        """
        Args:
            buy_threshold:  Minimum predicted upside (fraction) to trigger a buy. Default 5%.
            sell_threshold: Minimum predicted downside (fraction) to trigger a sell. Default 5%.

        Defaults sit deliberately above the forecast's measured error — see the class
        comment. Lowering them re-enables trading on noise.
        """
        tickers = _load_predicted_tickers()
        super().__init__(
            "KronosTraderBot",
            tickers=tickers,
            interval="1d",
            period="1y",
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            **kwargs,
        )
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self._pred_cache: dict[str, float | None] = {}  # one DB query per ticker per run

    def _get_predicted_close(self, symbol: str) -> float | None:
        """Return the predicted close for the next forecast bar of *symbol*, or None.

        Only forecasts made within MAX_PREDICTION_AGE_DAYS count; a surviving row from
        an abandoned run is otherwise indistinguishable from last night's.
        """
        try:
            tomorrow, min_made_at = _prediction_window()
            with get_db_session() as session:
                pred = (
                    session.query(KronosPrediction)
                    .filter(
                        KronosPrediction.symbol == symbol,
                        KronosPrediction.target_date >= tomorrow,
                        KronosPrediction.prediction_made_at >= min_made_at,
                    )
                    .order_by(
                        KronosPrediction.target_date.asc(),
                        KronosPrediction.prediction_made_at.desc(),
                    )
                    .first()
                )
                # Extract value inside the session — ORM objects detach on session close
                return float(pred.predicted_close) if pred is not None else None
        except Exception as exc:
            logger.warning(f"KronosTraderBot: DB query failed for {symbol}: {exc}")
            return None

    def decisionFunction(self, row) -> int:
        symbol = self._current_ticker

        # Cache per symbol — one DB query per ticker, not one per historical row
        if symbol not in self._pred_cache:
            self._pred_cache[symbol] = self._get_predicted_close(symbol)
        predicted_close = self._pred_cache[symbol]

        if predicted_close is None:
            return 0

        current_close = row["close"]
        if not current_close or current_close <= 0:
            return 0

        pct_change = (predicted_close - current_close) / current_close

        # Only log on the most recent bar to avoid 252 log lines per ticker
        if self.data is not None and row.name == self.data.index[-1]:
            logger.info(
                f"KronosTraderBot: {symbol} current={current_close:.2f} "
                f"predicted={predicted_close:.2f} ({pct_change:+.2%})"
            )

        if pct_change > self.buy_threshold:
            return 1  # buy
        if pct_change < -self.sell_threshold:
            return -1  # sell
        return 0  # hold


if __name__ == "__main__":
    run_bot(KronosTraderBot)
