"""
Portfolio optimisation helpers (e.g. Sharpe-ratio-based weights) as pure functions.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, risk_models


def sharpe_compute_weights(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute Sharpe-optimal weights from a wide-format price DataFrame.

    Expects:
        - df: index = dates, columns = symbols, values = close prices.

    Mirrors the logic currently used by SharpePortfolioOptWeeklyBot, but without
    any Bot or I/O dependencies.
    """
    if df is None or df.empty:
        return {}

    # Calculate expected returns and sample covariance
    mu = expected_returns.mean_historical_return(df)
    S = risk_models.sample_cov(df)

    # Optimize for maximal Sharpe ratio with 20% max weight per asset
    ef = EfficientFrontier(mu, S, weight_bounds=(0, 0.2))
    ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    # Sort dict descending by value and remove zero weights
    cleaned_weights = dict(
        sorted(cleaned_weights.items(), key=lambda item: item[1], reverse=True)
    )
    cleaned_weights = {k: v for k, v in cleaned_weights.items() if v != 0}

    if not cleaned_weights:
        return {}

    # Normalize weights to sum to 1.0
    total_weight = sum(cleaned_weights.values())
    if total_weight == 0:
        return {}

    return {k: v / total_weight for k, v in cleaned_weights.items()}

