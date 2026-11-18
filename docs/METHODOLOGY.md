# Industry momentum and portfolio selection
George Dharma

Python research comparing equal industry weights, top-10 momentum and a
lower-turnover retention rule. The question is whether historical relative
strength improves selection after trading costs and standard equity exposures.

## Sizing the momentum allocation

The sizing study uses the top-ten strategy selected in 2000–2009 and revisits January 2010 through December 2025. The four portfolios allocate 0%, 25%, 50% or 100% to momentum. The balance goes to equal industry weights. Each portfolio trades its combined weights in one account and charges 10 basis points on each dollar bought or sold.

The 25% allocation compounds at 13.19% a year after costs, against 12.71% for the equal-industry portfolio. It has 1.73% annual tracking error. A 95% interval for annual arithmetic active return runs from −0.28 to +1.15 percentage points and includes zero. The interval uses 5,000 circular block-bootstrap draws in 12-month blocks. The sizing comparison uses a period already examined in the strategy study and is exploratory. [Allocation results](../outputs/allocation_summary.csv) · [Monthly portfolios](../outputs/allocation_monthly.csv) · [Industry contributions](../outputs/active_industries.csv)

## Running this version

See [RUNNING.md](RUNNING.md). Open [the notebook](../notebooks/analysis.ipynb) for the worked calculation.

## Research design

The design in `data/design.json` was written before evaluating performance.
Development: 1970–1999. Selection: 2000–2009. Final test: 2010–2025.
Signals compound returns from t−12 through t−2, skipping t−1. A full signal
history and the previous month's return determine eligibility. No industry is
selected using its holding-month return. Ties use alphabetic industry codes.

- Equal industries: equal target weights across all eligible industry baskets.
- Top 10 momentum: the ten highest signals, with 10% target weights.
- Buffered momentum: retain prior holdings ranked 15 or better, then fill to
  ten from the highest remaining ranks. All ten target weights are 10%.

Each evaluation block starts in cash, pays the initial entry cost and uses older
data only to warm signals. Each month rebalances drifted weights, solves costs
funded by the portfolio, then earns the holding-month return. Costs are charged
on purchases and sales separately. The base rate is 10 bp per dollar traded;
0, 25 and 50 bp are additional scenarios. There is no terminal liquidation.

Selection uses validation net information ratio relative to equal industries.
The top-10 rule wins that comparison. `outputs/selection_lock.json` records the
choice, inputs and hashes before final-test evaluation. The final stage refuses
to use changed source or design files. Its `final_results_examined: false`
records the state when that selection record was created; it is intentionally
not overwritten after evaluation. The supplied final results are now known.
Redesigning after seeing them requires a new experiment and fresh evaluation.

To reproduce the stages separately:

```bash
python run_analysis.py --stage develop
python run_analysis.py --stage test
```

## Data and sources

The published study used files from the September 2026 CRSP database. A full run downloads the four publisher archives into `data/source` and reuses them on subsequent runs. The [source record](../data/source/provenance.json) gives the original URLs, member names and SHA-256 checksums. Run `python fetch_data.py --exact` to verify the vintage.

1. [49 industry portfolios](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_49_ind_port.html).
   The first table supplies value-weighted monthly industry returns. Stocks are
   assigned using SIC industry codes at the end of June. This project's benchmark
   is equal across industries, not equal across individual stocks.
2. [French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).
   Historical returns may be revised. The recorded checksums identify the published vintage. The CIZ release method compounds
   daily returns with dividends reinvested on ex-dates. Missing sentinels are
   converted to missing observations, never to invented zero returns.
3. [Momentum factor](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_mom_factor.html).
   The stock momentum factor is used with the market, size and value factors
   for explanatory regression. It is not the industry's own trading result.
4. [Two Sigma investment process](https://www.twosigma.com/businesses/investment-management/).
   Adapted the sequence from data preparation to signal, portfolio and execution.
   This is an independent analysis, not a reproduction of a firm's strategy.
5. [AQR momentum implementation research](https://www.aqr.com/insights/research/working-paper/implementing-momentum-what-have-we-learned).
   Used to motivate explicit turnover and cost comparisons, not to import cost
   estimates from a different portfolio.
6. [Man AHL on overfitting](https://www.man.com/insights/overfitting-and-its-impact-on-the-investor).
   Used to separate selection and final evaluation and disclose the fixed set
   of compared rules.

These are historical research baskets, not directly tradable tickers. Costs
cover allocation changes between baskets; they exclude within-basket stock
maintenance, actual spreads, taxes, fund fees and market impact. Capacity and
live implementation cannot be estimated from these monthly research returns.

## Files produced by a full local run

| Exhibit / file | Generated by | Contents |
|---|---|---|
| `summary.csv` | `analysis.develop`, `analysis.final_test` | Development, validation and final metrics |
| `monthly.csv` | `engine.backtest` | Gross/net returns, exact costs, drift-based turnover and wealth |
| `holdings.csv` | `engine.backtest` | Every final-test industry weight and trade |
| `signal_diagnostics.csv` | `engine.predictive_diagnostics` | Monthly rank IC and top-minus-bottom spread |
| `sample.csv` | `analysis.final_test` | December 2025 inputs, ranks and target allocation |
| `costs.csv` | `analysis.final_test` | Recomputed cost scenarios |
| `factors.csv` | `engine.hac_ols` | Four-factor coefficients and Newey–West intervals |
| `contribution.csv` | `analysis.final_test` | Industry contributions to arithmetic active return |
| `subperiods.csv`, `nearby.csv` | `analysis.final_test` | Prespecified robustness diagnostics |
| `annual.csv` | `analysis.final_test` | Calendar-year results and turnover |
| `industry_returns.csv`, `factor_returns.csv` | `analysis.final_test` | Clean decimal-return inputs used by the companion |
| `wealth.png`, `drawdowns.png`, `cost_sensitivity.png`, `signal_quality.png` | `analysis.charts` | Report figures |

`outputs/results.json` connects the tables and readable documents. It is not the
recruiter-facing report. The separate Excel companion has live sample signal and
cost arithmetic, cumulative wealth and drawdown formulas. Full strategy runs,
chronological selection and statistical inference remain in Python.

## Statistical conventions

Sharpe uses the monthly source risk-free return and square-root-of-12 scaling.
Information ratio uses monthly active returns versus the equal-industry
benchmark. Maximum drawdown uses month-end wealth and includes initial wealth.
Factor regression explains net returns minus cash using Mkt−RF, SMB, HML and
Mom. Alpha is annualised arithmetically by multiplying the monthly intercept by
twelve. Newey–West covariance uses Bartlett weights, six lags and n/(n−k)
correction; 95% intervals use 1.96 standard errors.

Industry contributions add to arithmetic average active return before costs.
They are not contributions to CAGR. The top-minus-bottom spread is a gross
ordering diagnostic, not a self-financing long–short strategy with borrowing.
The 8/10/12-industry comparisons are robustness checks, not new selected rules.
