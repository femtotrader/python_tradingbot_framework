"""Future-bar timestamp generation for the Kronos Space.

Split out of app.py so it can be unit-tested without importing torch or the
Kronos model weights (app.py loads both at import time).
"""

from __future__ import annotations

import pandas as pd

# yfinance daily bars have a median consecutive delta of exactly 1 day (Mon->Tue
# and friends outnumber the Fri->Mon 3-day gaps). Allow some slack so a feed with
# slightly ragged timestamps still classifies as daily.
_DAILY_MIN = pd.Timedelta(hours=20)
_DAILY_MAX = pd.Timedelta(hours=36)


def future_timestamps(x_timestamp: pd.Series, horizon: int) -> pd.DatetimeIndex:
    """Return the next *horizon* bar timestamps following the last observed bar.

    Daily bars on an exchange-traded instrument are stepped by **business days**:
    the input series itself never lands on a weekend, so stepping by calendar days
    emits target dates for Saturdays and Sundays on which the instrument never
    prints a bar. 24/7 instruments (crypto) do carry weekend bars and keep the
    plain calendar-day stepping, as do every non-daily interval.

    Known limitation: business days are not exchange holidays — a forecast bar can
    still land on Thanksgiving. Intraday bars on an exchange-traded instrument step
    through the overnight session for the same reason. Both would need a real
    exchange calendar; neither affects the daily equity path this Space serves.

    Args:
        x_timestamp: Observed bar timestamps (any order; duplicates tolerated).
        horizon:     Number of future bars to generate.

    Raises:
        ValueError: If fewer than two timestamps are supplied (frequency is
            un-inferable) or *horizon* is not positive.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be positive, got {horizon}")

    ts = pd.to_datetime(pd.Series(x_timestamp)).sort_values().reset_index(drop=True)
    deltas = ts.diff().dropna()
    if deltas.empty:
        raise ValueError("need at least two timestamps to infer the bar frequency")

    freq = deltas.median()
    last_ts = ts.iloc[-1]

    is_daily = _DAILY_MIN <= freq <= _DAILY_MAX
    trades_weekends = bool((ts.dt.dayofweek >= 5).any())
    if is_daily and not trades_weekends:
        return pd.bdate_range(start=last_ts + pd.Timedelta(days=1), periods=horizon)

    return pd.date_range(start=last_ts + freq, periods=horizon, freq=freq)
