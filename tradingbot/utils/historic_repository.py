"""Repository for historic OHLCV data stored in the database.

This module centralizes all direct database access for the `historic_data`
table so that higher-level services (like `DataService`) can remain focused on
data fetching, merging, and cleaning logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

import pandas as pd
from sqlalchemy.dialects.postgresql import insert

from .db import HistoricData, get_db_session


@dataclass
class HistoricDataRepository:
    """Repository abstraction for the `historic_data` table.

    Responsibilities:
    - Query historic OHLCV rows for a symbol or range.
    - Fetch the latest timestamp for a given symbol.
    - Perform bulk inserts with proper duplicate handling.
    """

    def get_latest_timestamp(self, symbol: str) -> Optional[datetime]:
        """Return the latest timestamp stored for a symbol, or None if none exists."""
        if not symbol:
            raise ValueError("symbol must be a non-empty string")

        with get_db_session() as session:
            latest = (
                session.query(HistoricData.timestamp)
                .filter_by(symbol=symbol)
                .order_by(HistoricData.timestamp.desc())
                .first()
            )
            return latest[0] if latest else None

    def get_range(
        self,
        symbol: str,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Load historic data for a symbol in an optional [start_date, end_date] range."""
        if not symbol:
            raise ValueError("symbol must be a non-empty string")

        with get_db_session() as session:
            query = session.query(HistoricData).filter_by(symbol=symbol)

            if start_date is not None:
                query = query.filter(HistoricData.timestamp >= start_date)
            if end_date is not None:
                query = query.filter(HistoricData.timestamp <= end_date)

            query = query.order_by(HistoricData.timestamp)
            results = query.all()

            if not results:
                return pd.DataFrame()

            # Build row dicts while session is open to avoid DetachedInstanceError
            rows = [
                {
                    "symbol": r.symbol,
                    "timestamp": r.timestamp,
                    "open": r.open,
                    "high": r.high,
                    "low": r.low,
                    "close": r.close,
                    "volume": r.volume,
                }
                for r in results
            ]

        return pd.DataFrame(rows)

    def bulk_insert_ohlcv(self, rows: Iterable[dict]) -> None:
        """Bulk insert OHLCV rows using ON CONFLICT DO NOTHING semantics.

        Each row must contain keys:
        - symbol
        - timestamp
        - open, high, low, close, volume
        """
        rows = list(rows)
        if not rows:
            return

        stmt = (
            insert(HistoricData)
            .values(rows)
            .on_conflict_do_nothing(index_elements=["symbol", "timestamp"])
        )
        with get_db_session() as session:
            session.execute(stmt)

