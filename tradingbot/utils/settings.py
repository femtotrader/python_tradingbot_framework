"""Configuration objects for the trading bot system.

These small dataclasses centralize tunable parameters so that code which
depends on them can be more self-documenting and easier to test.
"""

from __future__ import annotations

from dataclasses import dataclass

from .constants import (
    FRESHNESS_TOLERANCE_MINUTES,
    MIN_ASSET_VALUE_USD,
    PRICE_CACHE_MAXSIZE,
    PRICE_CACHE_TTL,
)


@dataclass(frozen=True)
class DataConfig:
    """Configuration for data freshness and caching."""

    freshness_tolerance_minutes: int = FRESHNESS_TOLERANCE_MINUTES
    price_cache_maxsize: int = PRICE_CACHE_MAXSIZE
    price_cache_ttl: int = PRICE_CACHE_TTL


@dataclass(frozen=True)
class PortfolioConfig:
    """Configuration for portfolio management thresholds."""

    min_asset_value_usd: float = MIN_ASSET_VALUE_USD


DATA_CONFIG = DataConfig()
PORTFOLIO_CONFIG = PortfolioConfig()

