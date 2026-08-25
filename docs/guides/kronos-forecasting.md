# Kronos Financial Forecasting Service

This guide covers the Kronos integration: a foundation model for financial K-line (candlestick) forecasting that predicts future OHLCV prices.

## What is Kronos?

**Kronos** ([NeoQuasar/Kronos-mini](https://huggingface.co/NeoQuasar/Kronos-mini)) is a 4.1M-parameter decoder-only Transformer pre-trained on 12+ billion K-line records from 45+ global exchanges. It forecasts open, high, low, close, and volume for future time periods given historical OHLCV data.

Key facts:

- **Trained on**: 12B+ K-lines across stocks, forex, crypto, commodities from multiple exchanges
- **Architecture**: Specialized financial foundation model (not a generic time-series adapter)
- **Models**: Available in 4 sizes (mini: 4.1M, small: 24.7M, base: 102M, large: 499M parameters)
- **Output**: Predicted OHLCV DataFrame matching input structure
- **Paper**: [Kronos: An Event-Driven Architecture for Autonomous Systems](https://arxiv.org/abs/2508.02739)

## Architecture

The framework splits Kronos inference across two systems to work within hardware constraints:

```
┌──────────────────────────────────────────┐
│ K8s Pod (kronosbot cronjob)              │
│  ├─ Fetch active symbols from DB         │
│  ├─ Call KronosClient (lightweight)      │
│  └─ Upsert predictions to PostgreSQL     │
└────────────┬─────────────────────────────┘
             │
             │ HTTP POST /predict
             │
┌────────────▼─────────────────────────────┐
│ HF Space (guestros/kronos-trading-api)   │
│ Docker Container (CPU-only, 16GB RAM)    │
│  ├─ FastAPI server                       │
│  ├─ Kronos-mini + tokenizer loaded       │
│  ├─ Torch CPU inference                  │
│  └─ Pre-built model weights              │
└──────────────────────────────────────────┘
```

**Why split?**

- K8s pod has strict memory limits (2Gi) — torch + Kronos would exceed this
- HF Spaces free tier offers 16GB RAM and CPU, enough for Kronos-mini
- HTTP separation allows reusability — any service can call the Space

## Deployment

### 1. HF Space (Already Done)

The Space is deployed at `https://huggingface.co/spaces/guestros/kronos-trading-api`.

Source of truth is `kronos_space/` in this repo; the Space is updated by pushing that
directory to the Space's git remote, which triggers a rebuild.

**Contents:**

- `Dockerfile`: Python 3.11, CPU-only PyTorch, FastAPI
- `app.py`: Two endpoints: `GET /health`, `POST /predict`
- `timeline.py`: future-bar timestamp generation (business days vs calendar days)
- `requirements.txt`: torch, transformers, fastapi, huggingface_hub, pandas
- Model weights pre-baked at build time (instant startup after restart)

After a push, confirm the new build is live via `/health` — it echoes the sampling
config, and a Space still running the old `sample_count=1` is otherwise
indistinguishable from a fixed one.

**Build status**: Watch logs at [guestros/kronos-trading-api](https://huggingface.co/spaces/guestros/kronos-trading-api)

### 2. K8s Cronjob (kronosbot)

Scheduled at `22:05 UTC Mon-Fri` (right after market close) via Helm values:

```yaml
- name: kronosbot
  schedule: "5 22 * * 1-5"  # 10:05 PM UTC, Monday-Friday
```

**Lifecycle:**

1. Wake the Space: `HfApi.restart_space("guestros/kronos-trading-api")`
2. Wait for Kronos-mini to load: Poll `/health` with 30s retry intervals (max 3 min wait)
3. Predict: Loop over active tickers, call `KronosClient.predict()` for each
4. Store: Upsert `KronosPrediction` rows to Postgres (deduplicated on symbol+target_date)
5. Pause: `HfApi.pause_space(...)` to save HF quota

**Total runtime**: ~2-3 minutes (includes cold start + all predictions)

### 3. Environment Variables

Set in Kubernetes secret or helm values:

**Required:**

- `KRONOS_SPACE_URL`: Base URL of the Space (e.g. `https://guestros-kronos-trading-api.hf.space`)

**For Space lifecycle control (optional but recommended):**

- `HF_TOKEN`: HuggingFace write token (stored in `tradingbot-secrets`)
- `HF_SPACE_REPO`: Space repo ID (default: `guestros/kronos-trading-api`)

**Optional tuning (kronosbot, K8s side):**

- `KRONOS_HORIZON`: Days ahead to forecast (default: 5)
- `KRONOS_EXTRA_SYMBOLS`: Comma-separated tickers to always forecast (default: `SPY,QQQ,GLD`)

**Optional tuning (the Space itself — set in the HF Space settings, not in helm):**

- `KRONOS_SAMPLE_COUNT`: Sampled paths averaged per forecast (default: 20). See
  [Forecast quality](#forecast-quality) — this is the single most important knob.
- `KRONOS_TEMPERATURE`: Sampling temperature `T` (default: 1.0)
- `KRONOS_TOP_P`: Nucleus sampling cutoff (default: 0.9)

`GET /health` echoes all three, so you can confirm what is actually deployed:

```bash
curl -s https://guestros-kronos-trading-api.hf.space/health
# {"status":"ok","model":"NeoQuasar/Kronos-mini","sample_count":20,"temperature":1.0,"top_p":0.9}
```

Example helm patch:

```bash
kubectl patch secret tradingbot-secrets -n tradingbots-2025 \
  --type=merge \
  -p '{"stringData":{"HF_TOKEN":"hf_your_write_token_here"}}'
```

## Using Kronos Predictions in Bots

### Direct API Usage

```python
from tradingbot.utils.kronos_client import KronosClient


class MyBot(Bot):
    def decisionFunction(self, row):
        client = KronosClient()
        pred = client.predict(self.symbol, horizon=5)

        if pred is not None:
            next_close = pred.iloc[0]["close"]
            current = row["close"]

            if next_close > current * 1.03:
                return 1  # Buy if model forecasts 3%+ upside

        return 0
```

### Query Predictions from Database

Other bots can query the `kronos_predictions` table:

```python
from tradingbot.utils.db import get_db_session
from tradingbot.utils.db import KronosPrediction
from datetime import datetime, timedelta


def get_kronos_signal(symbol, days_ahead=1):
    with get_db_session() as session:
        target_date = datetime.utcnow() + timedelta(days=days_ahead)

        pred = (
            session.query(KronosPrediction)
            .filter_by(symbol=symbol)
            .filter(
                KronosPrediction.target_date >= target_date.replace(hour=0, minute=0, second=0),
                KronosPrediction.target_date < target_date.replace(hour=23, minute=59, second=59),
            )
            .order_by(KronosPrediction.prediction_made_at.desc())
            .first()
        )

        if pred:
            # Compare predicted close to current close
            return pred.predicted_close
        return None
```

### LangChain Tool Integration

Use Kronos as a tool in AI flows:

```python
from tradingbot.utils.aitools import run_ai_with_tools
from tradingbot.utils.kronos_client import kronos_forecast

decision = run_ai_with_tools(
    system_prompt="Analyze the symbol and decide buy/hold/sell.",
    user_message="Should we buy SPY right now?",
    extra_tools=[kronos_forecast],  # Add Kronos
)
# AI can now call kronos_forecast("SPY") as a tool when reasoning
```

## Monitoring & Troubleshooting

### Check Space Status

```bash
python -c "
from tradingbot.utils.kronos_client import KronosClient
client = KronosClient()
print('Space healthy:', client.is_healthy())
"
```

### Manual Prediction Test

```bash
python -c "
from tradingbot.utils.kronos_client import KronosClient
client = KronosClient()
pred = client.predict('SPY', horizon=5)
if pred is not None:
    print(pred)
else:
    print('Prediction failed (Space unreachable or data insufficient)')
"
```

### Check Cronjob Logs

```bash
# Most recent job
kubectl logs -l batch.kubernetes.io/job-name=tradingbot-kronos-xxx -n tradingbots-2025 --tail=100

# All kronosbot jobs
kubectl logs -l app=tradingbot,cronjob=kronosbot -n tradingbots-2025
```

### Database Query

```bash
# Latest predictions
kubectl exec -n tradingbots-2025 $(kubectl get pods -n tradingbots-2025 -l app=psql -o name | head -1) -- \
  psql -U postgres -c "
    SELECT symbol, target_date, predicted_close, prediction_made_at
    FROM kronos_predictions
    ORDER BY prediction_made_at DESC
    LIMIT 20;
  "
```

### Common Issues

**"Space did not become healthy within retry window"**

→ Space is still building (Dockerfile running). Wait 5-10 min for Docker build to complete. Check build logs at [huggingface.co/spaces/guestros/kronos-trading-api](https://huggingface.co/spaces/guestros/kronos-trading-api).

**"Request timeout (>120s)"**

→ Space is cold (just woke up). kronosbot retries up to 6 times with 30s delays. If it times out, increase `_PREDICT_TIMEOUT` in `tradingbot/kronosbot.py`.

**"Insufficient data (<50 rows)"**

→ yfinance returned less than 50 bars for the symbol. Use a longer `period` parameter (default "2y") or check if the ticker is valid.

**"KRONOS_SPACE_URL not set"**

→ Set `KRONOS_SPACE_URL` environment variable in helm values or K8s secret.

## Forecast quality

**Before trusting these predictions, re-run this check.** As of 2026-08-25 the stored
forecasts had **no measurable edge**, and the two defects responsible were only fixed
that day — the numbers below are the "before" baseline, not the current state.

Measured over the 9,462 rows written between 2026-04-14 and 2026-08-24, each prediction
anchored to the last real bar the model actually saw:

| step ahead | n | Kronos MAE | naive "no change" MAE | direction hit | corr(pred, actual) |
|---|---|---|---|---|---|
| 1 | 506 | 2.47% | **0.57%** | 31.6% | 0.02 |
| 2 | 198 | 3.27% | **1.28%** | 51.0% | 0.27 |
| 3 | 294 | 3.60% | **1.58%** | 50.7% | 0.27 |
| 4 | 259 | 4.13% | **1.74%** | 50.6% | 0.11 |

Next-day error was **4.3x worse than assuming the price does not move**, direction was
below a coin flip, and predicted returns were uncorrelated with realised ones. It held
for every liquid symbol individually (SPY 1.13% vs 0.21% naive, GLD 2.11% vs 0.20%).

Two causes, both now fixed:

1. **`sample_count=1`** in the Space. Kronos is autoregressive and *generative* —
   `KronosPredictor.predict` averages `sample_count` sampled paths. One sample is a
   random draw from the predictive distribution, not an estimate of its mean, and the
   ~2.5% next-bar MAE was simply that sampler's dispersion. Now 20 by default.
2. **Calendar-day target dates.** The Space inferred its step frequency from the median
   bar delta (1 day for daily bars) and stepped with `pd.date_range`, so forecasts landed
   on Saturdays and Sundays: **2,761 of 9,462 rows (29%) targeted a non-trading day.**
   `kronos_space/timeline.py` now steps by business days for daily bars on instruments
   whose own history contains no weekend bars (crypto keeps calendar days).

A third issue affected symbols whose yfinance history lags a session (the European
listings — `*.DE`, `*.AS`): the Space forecast forward from a stale last bar, so the
leading rows targeted days that had already closed. `kronosbot` now drops those before
writing and logs which symbols were affected.

### Re-running the measurement

Against the in-cluster Postgres (needs `historic_data` populated for the same symbols):

```sql
WITH g AS (SELECT symbol, prediction_made_at, min(target_date) AS first_tgt
           FROM kronos_predictions GROUP BY 1,2),
a AS (SELECT g.*,
        (SELECT h.timestamp FROM historic_data h WHERE h.symbol=g.symbol AND h.interval='1d'
          AND h.timestamp < g.first_tgt ORDER BY h.timestamp DESC LIMIT 1) AS anchor_ts,
        (SELECT h.close FROM historic_data h WHERE h.symbol=g.symbol AND h.interval='1d'
          AND h.timestamp < g.first_tgt ORDER BY h.timestamp DESC LIMIT 1) AS anchor
      FROM g),
p AS (SELECT k.symbol, k.predicted_close, a.anchor,
        (SELECT h.close FROM historic_data h WHERE h.symbol=k.symbol AND h.interval='1d'
          AND h.timestamp = date(k.target_date)) AS actual,
        row_number() OVER (PARTITION BY k.symbol, k.prediction_made_at ORDER BY k.target_date) AS step
      FROM kronos_predictions k
      JOIN a ON a.symbol=k.symbol AND a.prediction_made_at=k.prediction_made_at
      WHERE a.anchor > 0 AND date(a.first_tgt) - date(a.anchor_ts) <= 4)
SELECT step, count(*) n,
  round(avg(abs(predicted_close-actual)/anchor)::numeric*100,3) AS kronos_mae_pct,
  round(avg(abs(actual-anchor)/anchor)::numeric*100,3)          AS naive_mae_pct,
  round(avg(CASE WHEN sign(predicted_close-anchor)=sign(actual-anchor) THEN 1.0 ELSE 0.0 END)::numeric*100,1) AS dir_hit_pct,
  round(corr(predicted_close/anchor-1, actual/anchor-1)::numeric,3) AS corr
FROM p WHERE actual IS NOT NULL GROUP BY step ORDER BY step;
```

**Anchor to the last bar before `min(target_date)`, not to the run date.** Anchoring to
the run date silently compares both the forecast and the actual against a stale price for
any symbol whose `historic_data` lags, which inflates the correlation to ~0.9 and makes
the model look like it beats the baseline. It does not.

### Effect on KronosTraderBot

`KronosTraderBot` shipped with `buy_threshold=0.02 / sell_threshold=0.01` — both *below*
the 2.5% noise floor, so nearly every symbol tripped one on sampling noise alone (25
orders on the 2026-08-24 run, 71% of the book parked in cash). Defaults are now 5%/5%,
above the measured error, so the bot stays close to flat. **Lower them only after re-running
the query above shows Kronos beating the naive column.**

## Performance Characteristics

### Latency

Measured on the 2026-08-24 run (76 symbols, 6m02s wall clock, `sample_count=1`):

| Operation | Time |
|-----------|------|
| Space restart → `/health` ok | ~2m15s |
| Warm inference per symbol (`sample_count=1`) | ~0.5s |
| DataService fetch per symbol | ~0.5-2s |
| DB upsert (380 predictions) | ~2s |

`sample_count=20` batches its samples into one forward pass rather than running 20
sequentially, so the per-symbol cost rises but not 20-fold. The client timeout is 300s
per symbol (`_PREDICT_TIMEOUT` in `tradingbot/utils/kronos_client.py`) and the CronJob's
`activeDeadlineSeconds` is 1800 — **check the run duration after changing
`KRONOS_SAMPLE_COUNT`**, since 76 symbols must still fit inside that 30-minute deadline.

### Space Quota

- Cronjob runs once per day (~6 min runtime at `sample_count=1`)
- Space paused after each run
- Free HF tier quota should easily cover this

### Memory

- K8s pod: stays under 500 MB (no torch)
- HF Space: ~5GB (torch + Kronos-mini loaded)

## Cost & Limits

**HF Spaces:**

- **Free tier**: CPU-only, 2 vCPU, 16GB RAM, paused after 48h inactivity
- **Pro tier** ($15/month): Always running, same resources
- **Cost for this setup**: Free (Space only runs ~2-3 min/day)

**API calls:**

- No rate limits on the Space itself (it's your own)
- HTTP client has default 120s timeout (configurable)

## Advanced Configuration

### Use a Larger Model

To use Kronos-small (24.7M params, better accuracy):

1. Edit `kronos_space/app.py`: change `MODEL_KEY = "kronos-mini"` → `"kronos-small"`
2. Rebuild Space: git push to the Space repo or trigger rebuild manually
3. Expect longer inference times (~60-90s per symbol on CPU)

### Add Custom Tickers

In helm values, set `KRONOS_EXTRA_SYMBOLS`:

```yaml
env:
  - name: KRONOS_EXTRA_SYMBOLS
    value: "SPY,QQQ,GLD,BTC/USD,EUR/USD"
```

kronosbot will predict all active bot tickers plus these extras.

### Change Forecast Horizon

In helm values:

```yaml
env:
  - name: KRONOS_HORIZON
    value: "10"  # Predict 10 days ahead instead of 5
```

## References

- **Kronos Paper**: [arxiv.org/abs/2508.02739](https://arxiv.org/abs/2508.02739)
- **HF Model Hub**: [NeoQuasar/Kronos-mini](https://huggingface.co/NeoQuasar/Kronos-mini), [Kronos-small](https://huggingface.co/NeoQuasar/Kronos-small)
- **GitHub**: [shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos)
- **KronosClient API**: [api/kronos-client.md](../api/kronos-client.md)
