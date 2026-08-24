from unittest.mock import MagicMock, patch

import pytest

from tradingbot.livetrade.collective2 import Collective2Broker


@pytest.fixture
def broker():
    return Collective2Broker(api_key="test_api", system_id="155809898")


def _response(payload):
    return MagicMock(status_code=200, json=lambda: payload, raise_for_status=lambda: None)


def _position(full_symbol, quantity, exchange_symbol=None):
    """One row of GetStrategyOpenPositions, in the shape C2 API v4 actually returns."""
    return {
        "StrategyName": "AdaptiveMeanReversionBot",
        "StrategyId": 155809898,
        "Currency": "USD",
        "Quantity": quantity,
        "AvgPx": 100.0,
        "C2Symbol": {"FullSymbol": full_symbol, "SymbolType": "stock"},
        "ExchangeSymbol": {"Symbol": exchange_symbol or full_symbol, "Currency": "USD"},
    }


def test_c2_get_positions_reads_the_nested_symbol(broker):
    """
    The ticker lives in C2Symbol.FullSymbol; there is no top-level "Symbol" key.

    Reading pos["Symbol"] returned None for every row, so a book of 113 QQQ and
    152 TQQQ collapsed to {None: 152}. sync() reconciles to target state, so
    invisible holdings read as "not held" and the copier re-bought the whole
    target on top of the existing book.
    """
    with patch.object(broker.client, "get") as mock_get:
        mock_get.return_value = _response({"Results": [_position("QQQ", 113), _position("TQQQ", 152)]})
        assert broker.get_positions() == {"QQQ": 113.0, "TQQQ": 152.0}


def test_c2_get_positions_falls_back_to_exchange_symbol(broker):
    with patch.object(broker.client, "get") as mock_get:
        row = _position("QQQ", 10)
        row["C2Symbol"] = {}  # no FullSymbol -> ExchangeSymbol.Symbol must carry it
        mock_get.return_value = _response({"Results": [row]})
        assert broker.get_positions() == {"QQQ": 10.0}


def test_c2_get_positions_sums_lots_for_one_ticker(broker):
    """Assignment would keep only the last lot and under-report the holding."""
    with patch.object(broker.client, "get") as mock_get:
        mock_get.return_value = _response({"Results": [_position("QQQ", 100), _position("QQQ", 13)]})
        assert broker.get_positions() == {"QQQ": 113.0}


def test_c2_get_positions_preserves_short_sign(broker):
    with patch.object(broker.client, "get") as mock_get:
        mock_get.return_value = _response({"Results": [_position("QQQ", -25)]})
        assert broker.get_positions() == {"QQQ": -25.0}


def test_c2_get_positions_raises_on_unresolvable_symbol(broker):
    """
    An unreadable position must abort the sync, not silently shrink the book.

    Skipping the row would under-report holdings and the copier would re-buy
    them; sync() catches the raise and aborts, which is the safe outcome.
    """
    with patch.object(broker.client, "get") as mock_get:
        row = _position("QQQ", 10)
        row["C2Symbol"] = {}
        row["ExchangeSymbol"] = {}
        mock_get.return_value = _response({"Results": [row]})
        with pytest.raises(ValueError, match="no resolvable symbol"):
            broker.get_positions()


def test_c2_get_positions_propagates_transport_failure(broker):
    """A failed positions call must not look like a flat account."""
    with patch.object(broker.client, "get") as mock_get:
        mock_get.side_effect = RuntimeError("C2 503")
        with pytest.raises(RuntimeError):
            broker.get_positions()


def test_c2_get_positions_empty_book(broker):
    with patch.object(broker.client, "get") as mock_get:
        mock_get.return_value = _response({"Results": []})
        assert broker.get_positions() == {}
