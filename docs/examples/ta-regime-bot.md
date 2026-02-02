# TA Regime-Adaptive Bot: Mathematical and Quant Concepts

The **TARegimeAdaptiveBot** is a single-asset bot that uses **only historic OHLCV and technical indicators** (no Fear & Greed or other external APIs). It classifies the market into two regimes—**trend** vs **mean reversion**—using a Hurst-style proxy, then applies regime-specific rules (ADX/MACD/EMA in trend; RSI/Bollinger BBP and optional z-score in mean reversion). All logic lives in `tradingbot.utils.ta_regime`; the bot itself only fetches data and delegates.

This page explains the mathematical and quantitative concepts behind the strategy and links to primary sources.

---

## 1. Regime: Trend vs Mean Reversion

Markets alternate between periods where:

- **Trend (persistence)**: Returns tend to follow the same direction; momentum strategies work better.
- **Mean reversion**: Prices tend to revert to a local mean; oversold/overbought signals work better.

Choosing the wrong strategy for the current regime can hurt performance. The bot therefore **classifies the regime from historic returns** and then applies the matching rule set.

---

## 2. Hurst Exponent and Long-Memory

The **Hurst exponent** \(H\) is a measure of long-range dependence in a time series. It was developed by Harold Edwin Hurst for hydrology (e.g. Nile River flows) and is widely used in finance to distinguish trending from mean-reverting behavior.

### Interpretation

- **\(H = 0.5\)**: Random walk (no long-term memory); past returns do not predict future direction.
- **\(H > 0.5\)** (up to 1): **Persistence** — positive autocorrelation; highs tend to follow highs, lows tend to follow lows → suitable for **trend-following**.
- **\(H < 0.5\)** (down to 0): **Mean reversion** — negative autocorrelation; highs tend to be followed by lows → suitable for **mean-reversion** strategies.

Research shows that **dynamically selecting** strategy (trend vs mean reversion) based on a Hurst-type classification can improve returns, though with higher variability. See for example:

- **Macrosynergy**: [Detecting trends and mean reversion with the Hurst exponent](https://macrosynergy.com/research/detecting-trends-and-mean-reversion-with-the-hurst-exponent/) — application to strategy selection.
- **CFA Institute**: [Rescaled Range Analysis: Detecting Persistence, Randomness, or Mean Reversion](https://blogs.cfainstitute.org/investor/2013/01/30/rescaled-range-analysis-a-method-for-detecting-persistence-randomness-or-mean-reversion-in-financial-markets/) — R/S method and interpretation.
- **Wikipedia**: [Hurst exponent](https://en.wikipedia.org/wiki/Hurst_exponent) — definition and estimation methods.

### How the Hurst exponent is usually estimated: R/S analysis

The classical approach is **rescaled range (R/S) analysis**:

1. For a window of returns, compute the **range** \(R\) of cumulative deviations from the mean over sub-intervals of length \(\tau\).
2. Compute the **standard deviation** \(S\) of returns over the same sub-intervals.
3. The ratio \(R/S\) scales with \(\tau^H\). Regressing \(\log(R/S)\) on \(\log(\tau)\) yields an estimate of \(H\).

This is described in the CFA link above and in many quant finance texts.

### What this codebase uses: a simple proxy (lag-1 autocorrelation)

Computing the full R/S Hurst is more involved and sensitive to sample size. The bot uses a **lightweight proxy** that captures the same idea:

- **Lag-1 autocorrelation** \(\rho_1\) of returns over a rolling window:
  - \(\rho_1 > 0\) → persistence (trend-like).
  - \(\rho_1 < 0\) → mean reversion.
- The proxy maps \(\rho_1 \in [-1, 1]\) to a value in \([0, 1]\) via \(\frac{1 + \rho_1}{2}\), so that **0.5** corresponds to no autocorrelation (random walk), **> 0.5** to trend, **< 0.5** to mean reversion. A threshold (e.g. 0.5) then classifies the regime.

So the *concept* is Hurst (trend vs mean reversion); the *implementation* is autocorrelation-based for simplicity and stability on typical bar counts.

---

## 3. Variance Ratio (alternative Hurst-style proxy)

The **variance ratio** is another way to detect persistence vs mean reversion:

\[
VR(n) = \frac{\operatorname{Var}(r_t + r_{t-1} + \cdots + r_{t-n+1})}{n \cdot \operatorname{Var}(r_t)}
\]

- **\(VR > 1\)**: Persistence (trend-like).
- **\(VR < 1\)**: Mean reversion.

It is related to the Hurst exponent and is often used in empirical finance. The bot does not implement it; the lag-1 autocorrelation proxy is used instead. For variance-ratio tests and applications, see the CFA and Macrosynergy references above.

---

## 4. Z-Score for Mean Reversion

In **mean-reversion** regimes, a common idea is to treat price (or an indicator) as reverting to a local mean. The **z-score** measures how many standard deviations the current value is from its recent mean:

\[
z = \frac{x_t - \mu_w}{\sigma_w}
\]

where \(\mu_w\) and \(\sigma_w\) are the mean and standard deviation over a rolling window of length \(w\). Large negative \(z\) (e.g. \(z < -2\)) suggests oversold; large positive \(z\) suggests overbought. The bot optionally uses this to **strengthen** mean-reversion entries (e.g. only buy when RSI/BBP are oversold *and* z-score is sufficiently negative).

---

## 5. Technical Indicators Used (trend vs mean reversion)

Once the regime is classified, the bot uses only **historic TA** from the `ta` library (no external APIs):

- **Trend regime**: ADX (strength of trend), MACD vs signal (direction), and optionally EMA fast vs slow (alignment). Buy when ADX > threshold, MACD > signal (and EMAs aligned if required); sell when MACD < signal (and EMAs aligned for sell).
- **Mean-reversion regime**: RSI (oversold/overbought), Bollinger Band position BBP (low/high), and optionally the z-score of price. Buy when RSI and BBP are in oversold territory (and z below a negative threshold if enabled); sell when overbought (and z above a positive threshold if enabled).

Evidence that **combining** indicators (e.g. RSI, Bollinger Bands, ADX, MACD) improves over single-indicator rules is discussed in multi-indicator and algorithmic trading studies (e.g. [Enhancing Trading Strategies: Multi-indicator Analysis](https://link.springer.com/article/10.1007/s10614-024-10669-3); [Empirical Study of Technical Indicators](https://escholarship.org/uc/item/5tq0q6cq)).

---

## 6. Further Reading: Hilbert Transform and Ehlers MESA

**John Ehlers** introduced cycle-oriented tools that use the **Hilbert transform** to estimate dominant cycle period and to build adaptive moving averages (e.g. MESA Adaptive Moving Average, MAMA). The idea is to reduce lag and adapt to market rhythm. These are more advanced than the current bot’s logic; they are mentioned here for context and possible future extension.

- **Traders.com**: [On Lag, Signal Processing, and the Hilbert Transform](https://traders.com/documentation/feedbk_docs/2000/03/Abstracts_new/Ehlers/Ehlers.html).
- **MESA Adaptive Moving Average**: [September 2001](https://traders.com/documentation/feedbk_docs/2001/09/Abstracts_new/Ehlers/ehlers.html) — adaptive alpha from detected cycle.

---

## 7. Further Reading: Entropy and Regularity

**Approximate entropy** and **sample entropy** measure regularity/unpredictability in a time series. Lower entropy can indicate more structure (e.g. during stress); they have been used in finance as irregularity measures. The bot does not implement them; they are listed as optional “advanced” concepts.

- **Approximate entropy in finance**: [Approximate Entropy as an Irregularity Measure for Financial Data](https://ideas.repec.org/a/taf/emetrv/v27y2008i4-6p329-362.html).
- **Wikipedia**: [Approximate entropy](https://en.wikipedia.org/wiki/Approximate_entropy).

---

## 8. Summary and Code Entry Points

| Concept            | Role in bot                         | Implemented in                         |
|--------------------|-------------------------------------|----------------------------------------|
| Hurst-style regime | Trend vs mean reversion             | `hurst_proxy_from_returns` (ACF proxy) |
| Regime classification | Threshold on proxy                | `classify_ta_regime`                   |
| Z-score            | Optional mean-reversion filter      | `rolling_zscore`                       |
| TA rules           | ADX/MACD/EMA (trend); RSI/BBP (MR)  | `ta_regime_decision`                   |

- **Utils**: `tradingbot.utils.ta_regime` — pure functions, no Bot/db.
- **Bot**: `tradingbot.ta_regime_bot.TARegimeAdaptiveBot` — fetches data, calls `ta_regime_decision(row, self.data, **self._ta_params)`.

For a minimal code example, see [Example Bots](example-bots.md). For backtesting and hyperparameter tuning, see [Hyperparameter Tuning](../api/hyperparameter-tuning.md) and [Backtest](../api/backtest.md).
