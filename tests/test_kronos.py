"""Tests for the Kronos forecast pipeline.

Covers the two defects that made the stored forecasts unusable:
  - target dates stepped by calendar days, so ~29% of rows landed on weekends
  - forecast bars written for days that had already closed (stale input history)
plus the horizon_days bookkeeping and KronosTraderBot's freshness filtering.
"""

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from kronos_space.timeline import future_timestamps


def _weekday_index(start: str, periods: int) -> pd.Series:
    """Business-day bar timestamps, the shape yfinance returns for an equity."""
    return pd.Series(pd.bdate_range(start=start, periods=periods))


def _calendar_index(start: str, periods: int) -> pd.Series:
    """Calendar-day bar timestamps, the shape yfinance returns for crypto."""
    return pd.Series(pd.date_range(start=start, periods=periods, freq="D"))


class TestFutureTimestamps:
    def test_daily_equity_bars_skip_the_weekend(self):
        # Last bar Friday 2026-08-21 -> next five *trading* days, no Sat/Sun.
        ts = _weekday_index("2026-06-01", 60)
        assert ts.iloc[-1] == pd.Timestamp("2026-08-21")

        out = future_timestamps(ts, 5)

        assert list(out) == [
            pd.Timestamp("2026-08-24"),
            pd.Timestamp("2026-08-25"),
            pd.Timestamp("2026-08-26"),
            pd.Timestamp("2026-08-27"),
            pd.Timestamp("2026-08-28"),
        ]
        assert not any(t.dayofweek >= 5 for t in out)

    def test_daily_equity_bars_from_midweek(self):
        ts = _weekday_index("2026-06-01", 61)
        assert ts.iloc[-1] == pd.Timestamp("2026-08-24")  # Monday

        out = future_timestamps(ts, 5)

        # Tue-Fri then the following Monday — the Saturday the old code emitted is gone.
        assert list(out) == [
            pd.Timestamp("2026-08-25"),
            pd.Timestamp("2026-08-26"),
            pd.Timestamp("2026-08-27"),
            pd.Timestamp("2026-08-28"),
            pd.Timestamp("2026-08-31"),
        ]

    def test_crypto_keeps_calendar_days(self):
        # 24/7 instruments do print weekend bars, so weekends must be preserved.
        ts = _calendar_index("2026-06-01", 85)
        assert ts.iloc[-1] == pd.Timestamp("2026-08-24")

        out = future_timestamps(ts, 5)

        assert list(out) == [pd.Timestamp("2026-08-24") + timedelta(days=n) for n in range(1, 6)]
        assert any(t.dayofweek >= 5 for t in out)

    def test_intraday_bars_step_by_the_observed_interval(self):
        ts = pd.Series(pd.date_range("2026-08-24 09:30", periods=120, freq="1min"))
        assert ts.iloc[-1] == pd.Timestamp("2026-08-24 11:29")

        out = future_timestamps(ts, 3)

        assert list(out) == [
            pd.Timestamp("2026-08-24 11:30"),
            pd.Timestamp("2026-08-24 11:31"),
            pd.Timestamp("2026-08-24 11:32"),
        ]

    def test_unsorted_input_is_handled(self):
        ts = _weekday_index("2026-06-01", 60).sample(frac=1.0, random_state=0)

        assert list(future_timestamps(ts, 1)) == [pd.Timestamp("2026-08-24")]

    def test_rejects_uninferable_input(self):
        with pytest.raises(ValueError, match="infer the bar frequency"):
            future_timestamps(pd.Series([pd.Timestamp("2026-08-24")]), 5)

    def test_rejects_non_positive_horizon(self):
        with pytest.raises(ValueError, match="horizon must be positive"):
            future_timestamps(_weekday_index("2026-06-01", 60), 0)


class TestKronosbotRowBuilding:
    """kronosbot.main() turns a forecast frame into KronosPrediction rows.

    The loop is exercised through a stand-in that mirrors it exactly, so the test
    does not need a live Space or a database session.
    """

    @staticmethod
    def _build(pred_df: pd.DataFrame, made_at: datetime) -> list[tuple[datetime, int]]:
        """Return (target_date, horizon_days) for the rows kronosbot would write."""
        rows = []
        for step, (_, row) in enumerate(pred_df.iterrows(), start=1):
            target = row["target_date"].to_pydatetime()
            if target <= made_at:
                continue
            rows.append((target, step))
        return rows

    def test_horizon_days_counts_steps_not_the_configured_horizon(self):
        made_at = datetime(2026, 8, 24, 22, 5)
        pred_df = pd.DataFrame({"target_date": pd.bdate_range("2026-08-25", periods=5)})

        rows = self._build(pred_df, made_at)

        # Previously every row was written with horizon_days=5, so nothing downstream
        # could tell a next-day forecast from a five-day-out one.
        assert [h for _, h in rows] == [1, 2, 3, 4, 5]

    def test_already_closed_bars_are_dropped_and_keep_their_step_number(self):
        # The European listings lag a session: input ends Friday, so the Space forecasts
        # from Friday while kronosbot runs Monday night.
        made_at = datetime(2026, 8, 24, 22, 5)
        pred_df = pd.DataFrame({"target_date": pd.bdate_range("2026-08-24", periods=5)})

        rows = self._build(pred_df, made_at)

        assert [t.date().isoformat() for t, _ in rows] == [
            "2026-08-25",
            "2026-08-26",
            "2026-08-27",
            "2026-08-28",
        ]
        # Monday was step 1 and got dropped; the survivors stay honest about how many
        # steps ahead of the model's last input bar they are.
        assert [h for _, h in rows] == [2, 3, 4, 5]

    def test_fresh_input_drops_nothing(self):
        made_at = datetime(2026, 8, 24, 22, 5)
        pred_df = pd.DataFrame({"target_date": pd.bdate_range("2026-08-25", periods=5)})

        assert len(self._build(pred_df, made_at)) == 5


class TestKronosTraderBotWindow:
    def test_window_is_naive_utc(self):
        from tradingbot.kronostraderbot import _prediction_window

        tomorrow, min_made_at = _prediction_window()

        # Aware datetimes would make Postgres coerce the comparison against these
        # bare DateTime columns through the session time zone.
        assert tomorrow.tzinfo is None
        assert min_made_at.tzinfo is None

    def test_window_bounds(self):
        from tradingbot.kronostraderbot import MAX_PREDICTION_AGE_DAYS, _prediction_window

        now = datetime.now(UTC).replace(tzinfo=None)
        tomorrow, min_made_at = _prediction_window()

        assert tomorrow > now
        assert (tomorrow - now) <= timedelta(days=1)
        assert tomorrow.hour == tomorrow.minute == tomorrow.second == 0
        assert abs((now - min_made_at) - timedelta(days=MAX_PREDICTION_AGE_DAYS)) < timedelta(seconds=5)

    def test_thresholds_sit_above_the_measured_noise_floor(self):
        import inspect

        from tradingbot.kronostraderbot import KronosTraderBot

        # Next-bar MAE measured at ~2.5% of price (Apr-Aug 2026). Defaults below that
        # mean the bot trades sampling noise, which is what the 2%/1% defaults did.
        measured_next_bar_mae = 0.025

        grid = KronosTraderBot.param_grid
        assert min(grid["buy_threshold"]) > measured_next_bar_mae
        assert min(grid["sell_threshold"]) > measured_next_bar_mae

        # Signature defaults, not an instance — constructing one queries the DB.
        params = inspect.signature(KronosTraderBot.__init__).parameters
        assert params["buy_threshold"].default > measured_next_bar_mae
        assert params["sell_threshold"].default > measured_next_bar_mae
