# Final research (2026-10-10/11)

Run locally by Claude after the step-4 indicator search. Learning period 2016-01..2022-11; holdout 2023-01..2026-08 opened once.

- `shape_results.csv` - 64 pre-registered contract shapes on fixed random entries (best: delta 0.8, ~90 DTE).
- `calib_*.csv`, `iv_calib.csv` - model vs real option prices (ORATS 2024-01-03, HistoricalOptionData 2026-03-19). Old IV model (HV30 x SPY VRP) overpriced IV by 14-49%; new model IV = 1.2 x (0.5 HV60 + 0.5 HV252).
- `grid_iv2.csv` (645 single indicators), `combos_*.csv` (720 trigger x filter), `confluence_iv2.csv` (387 indicator pairs): none positive with t > 2 for buying options.
- `credit_*` - SPY bull put credit spreads priced with real daily VIX (+ assumed skew). Pre-committed config (delta 0.30, width 5%, exit 7 DTE, stop 2x credit, SPY > MA200) passed the holdout: 166 trades, 80% wins, +4.0% average per trade on capital at risk.

## Round 2 (2026-10-11): broader search
- `pead_learn.csv` - post-earnings drift: no drift in S&P 500 2016-2022 (stock 10-day return ~+0.7% after both up and down reactions).
- `feature_buckets.csv` - 45 features, extreme buckets: apparent winners were driven by 2020 crash rebound; cross-sectional versions negative.
- `ml_walkforward.csv` - LightGBM walk-forward 2018-2022: calls improve ~1.5 pts over baseline but stay negative.
- `spy_long_*`, `spycall_*` - buying SPY calls when SPY > MA200 (delta 0.6, ~60 DTE, +30% / -30% / 10 days). All 18 neighbouring shapes positive in learning; pre-committed config on holdout: 78 trades, 50% wins, avg win +35.6%, +5.3% per trade (t 1.4), 3 of 4 years positive.
