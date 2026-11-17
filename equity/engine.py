"""Data parsing, lagged signals, cost-funded portfolios and statistical summaries."""

from pathlib import Path
import io
import re
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import spearmanr, norm


def read_monthly(path):
    """Read the first monthly block, not subsequent equal-weight/annual tables."""
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    first = next(i for i, s in enumerate(lines) if re.match(r"^\s*\d{6},", s))
    header = next(
        lines[i] for i in range(first - 1, -1, -1) if lines[i].lstrip().startswith(",")
    )
    end = first
    while end < len(lines) and re.match(r"^\s*\d{6},", lines[end]):
        end += 1
    frame = pd.read_csv(
        io.StringIO("\n".join([header] + lines[first:end])), index_col=0
    )
    frame.columns = frame.columns.str.strip()
    frame.index = pd.PeriodIndex(frame.index.astype(str).str.strip(), freq="M")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("Monthly dates must be unique and increasing.")
    if not frame.index.equals(
        pd.period_range(frame.index[0], frame.index[-1], freq="M")
    ):
        raise ValueError("Missing calendar months in source.")
    frame = frame.astype(float).replace({-99.99: np.nan, -999.0: np.nan}) / 100
    if (frame < -1).any().any():
        raise ValueError("Invalid return below -100%.")
    return frame


def signals(returns, lookback=11, shift=2):
    """Compound past returns with an explicit lag and require prior-month eligibility."""
    if lookback < 1 or shift < 1:
        raise ValueError("Lookback and shift must be positive.")
    # The holding month t uses t-12,...,t-2 for the default 11/2 rule.
    score = (1 + returns.shift(shift)).rolling(lookback, min_periods=lookback).apply(
        np.prod, raw=True
    ) - 1
    return score.where(returns.shift(1).notna())


def ranked_codes(score):
    """Rank eligible industries from strongest to weakest, breaking ties alphabetically."""
    return sorted(score.dropna().index, key=lambda code: (-score[code], code))


def target_weights(score, policy, previous=None, count=10, buffer_rank=15):
    """Apply the allocation rule and return fully invested long-only target weights."""
    ordered = ranked_codes(score)
    if len(ordered) < count:
        raise ValueError("Too few eligible industries.")
    if policy == "Equal industries":
        chosen = ordered
    elif policy == "Top 10 momentum":
        chosen = ordered[:count]
    elif policy == "Buffered momentum":
        if buffer_rank < count:
            raise ValueError("Buffer rank cannot be below holding count.")
        retained = set(previous or []).intersection(ordered[:buffer_rank])
        chosen = [x for x in ordered if x in retained]
        chosen += [x for x in ordered if x not in retained][: count - len(chosen)]
    else:
        raise ValueError("Unknown allocation policy.")
    w = pd.Series(0.0, index=score.index)
    w.loc[chosen] = 1 / len(chosen)
    return w


def rebalance(target, pretrade, cost_bps):
    """Solve C = c*sum(abs((1-C)*target - pretrade)) in units of starting NAV.

    The target sums to one after costs. Pretrade risky weights sum to zero on
    cash-funded entry and one thereafter. Costs are financed by the portfolio.
    """
    w = np.asarray(target, dtype=float)
    p = np.asarray(pretrade, dtype=float)
    if (
        w.shape != p.shape
        or np.any(w < 0)
        or np.any(p < 0)
        or not np.isclose(w.sum(), 1)
    ):
        raise ValueError(
            "Require matching long-only weights and a fully invested target."
        )
    if not (np.isclose(p.sum(), 0) or np.isclose(p.sum(), 1)):
        raise ValueError(
            "Pretrade weights must represent all cash or a fully invested book."
        )
    if not 0 <= cost_bps <= 100:
        raise ValueError("Cost must be from 0 to 100 bp per side.")
    c = cost_bps / 10000
    total = lambda C: np.abs((1 - C) * w - p).sum()
    C = 0.0 if c == 0 else brentq(lambda x: x - c * total(x), 0, 0.1, xtol=1e-15)
    trades = (1 - C) * w - p
    buys = float(np.maximum(trades, 0).sum())
    sales = float(np.maximum(-trades, 0).sum())
    return {
        "cost": float(C),
        "buys": buys,
        "sales": sales,
        "gross_traded": buys + sales,
        "turnover": max(buys, sales),
        "trades": trades,
    }


def backtest(returns, score, start, end, policy, cost_bps=10, count=10, buffer_rank=15):
    """Each evaluation block starts in cash; old observations only warm signals."""
    dates = returns.loc[start:end].index
    if len(dates) == 0:
        raise ValueError("Empty evaluation block.")
    pre = pd.Series(0.0, index=returns.columns)
    previous = []
    records = []
    holdings = []
    net_wealth = 1.0
    gross_wealth = 1.0
    for month in dates:
        w = target_weights(score.loc[month], policy, previous, count, buffer_rank)
        realized = returns.loc[month]
        if realized[w > 0].isna().any():
            raise ValueError(f"Missing realised return for a held industry in {month}.")
        trade = rebalance(w.values, pre.values, cost_bps)
        gross = float((w * realized.fillna(0)).sum())
        net = (1 - trade["cost"]) * (1 + gross) - 1
        if gross <= -1:
            raise ValueError("Portfolio wealth exhausted.")
        net_wealth *= 1 + net
        gross_wealth *= 1 + gross
        row = {
            "month": str(month),
            "policy": policy,
            "gross_return": gross,
            "cost_fraction": trade["cost"],
            "net_return": net,
            "buys": trade["buys"],
            "sales": trade["sales"],
            "turnover": trade["turnover"],
            "gross_traded": trade["gross_traded"],
            "net_wealth": net_wealth,
            "gross_wealth": gross_wealth,
            "max_target_weight": float(w.max()),
            "holdings": int((w > 0).sum()),
            "eligible": int(score.loc[month].notna().sum()),
        }
        records.append(row)
        for i, code in enumerate(w.index):
            holdings.append(
                {
                    "month": str(month),
                    "policy": policy,
                    "industry": code,
                    "signal": (
                        None
                        if pd.isna(score.loc[month, code])
                        else float(score.loc[month, code])
                    ),
                    "pretrade_weight": float(pre[code]),
                    "target_weight": float(w[code]),
                    "trade_fraction": float(trade["trades"][i]),
                    "return": (
                        None if pd.isna(realized[code]) else float(realized[code])
                    ),
                }
            )
        pre = w * (1 + realized.fillna(0)) / (1 + gross)
        previous = list(w[w > 0].index)
    return pd.DataFrame(records), pd.DataFrame(holdings)


def hac_ols(y, x, lags=6):
    """OLS with Bartlett/Newey-West covariance and n/(n-k) correction."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    X = np.column_stack([np.ones(len(y)), x])
    n, k = X.shape
    if n <= k or not np.isfinite(y).all() or not np.isfinite(X).all():
        raise ValueError("Invalid regression data.")
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    z = X * e[:, None]
    meat = z.T @ z
    for lag in range(1, min(lags, n - 1) + 1):
        cross = z[lag:].T @ z[:-lag]
        meat += (1 - lag / (lags + 1)) * (cross + cross.T)
    bread = np.linalg.inv(X.T @ X)
    cov = bread @ meat @ bread * n / (n - k)
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    stat = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    return {
        "coefficients": beta.tolist(),
        "se": se.tolist(),
        "t": stat.tolist(),
        "ci_low": (beta - 1.96 * se).tolist(),
        "ci_high": (beta + 1.96 * se).tolist(),
        "p_normal": (2 * norm.sf(np.abs(stat))).tolist(),
        "r_squared": float(1 - e @ e / np.sum((y - y.mean()) ** 2)),
    }


def mean_hac(y, lags=6):
    y = np.asarray(y, dtype=float)
    r = hac_ols(y, np.empty((len(y), 0)), lags)
    return {
        k: r[k][0] for k in ["coefficients", "se", "t", "ci_low", "ci_high", "p_normal"]
    }


def metrics(frame, rf, benchmark=None):
    """Summarise growth, risk, drawdown, turnover and relative performance."""
    r = frame.net_return.to_numpy()
    g = frame.gross_return.to_numpy()
    n = len(r)
    excess = r - np.asarray(rf, dtype=float)
    wealth = np.cumprod(1 + r)
    peaks = np.maximum.accumulate(np.r_[1.0, wealth])[1:]
    dd = wealth / peaks - 1
    ans = {
        "months": n,
        "cagr": float(wealth[-1] ** (12 / n) - 1),
        "gross_cagr": float(np.prod(1 + g) ** (12 / n) - 1),
        "volatility": float(r.std(ddof=1) * np.sqrt(12)),
        "sharpe": float(excess.mean() / excess.std(ddof=1) * np.sqrt(12)),
        "max_drawdown": float(dd.min()),
        "final_wealth": float(wealth[-1]),
        "annual_turnover": float(frame.turnover.mean() * 12),
        "annual_cost_fraction": float(frame.cost_fraction.mean() * 12),
        "max_weight": float(frame.max_target_weight.max()),
        "average_holdings": float(frame.holdings.mean()),
    }
    if benchmark is not None:
        active = r - np.asarray(benchmark, dtype=float)
        std = active.std(ddof=1)
        ans.update(
            {
                "annual_active_mean": float(active.mean() * 12),
                "tracking_error": float(std * np.sqrt(12)),
                "information_ratio": (
                    float(active.mean() / std * np.sqrt(12)) if std > 0 else 0.0
                ),
                "active_hac": mean_hac(active) if std > 0 else None,
            }
        )
    return ans


def predictive_diagnostics(returns, score, start, end, count=10):
    rows = []
    for month in returns.loc[start:end].index:
        order = ranked_codes(score.loc[month])
        valid = [x for x in order if pd.notna(returns.loc[month, x])]
        rho = float(
            spearmanr(score.loc[month, valid], returns.loc[month, valid]).statistic
        )
        top = float(returns.loc[month, order[:count]].mean())
        bottom = float(returns.loc[month, order[-count:]].mean())
        rows.append(
            {
                "month": str(month),
                "rank_ic": rho,
                "top_return": top,
                "bottom_return": bottom,
                "spread": top - bottom,
                "industries": len(valid),
            }
        )
    return pd.DataFrame(rows)
