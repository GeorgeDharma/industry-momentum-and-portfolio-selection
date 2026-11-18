"""Size the selected industry momentum allocation alongside an equal-industry core."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import load
from .engine import rebalance, target_weights


def block_interval(active, trials=5000, block=12, seed=642):
    """Return a 95% interval for annualised arithmetic active return."""
    active = np.asarray(active)
    months = len(active)
    if block < 1 or months < block:
        raise ValueError("The block must be positive and no longer than the sample.")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, months, size=(trials, int(np.ceil(months / block))))
    indices = ((starts[:, :, None] + np.arange(block)) % months).reshape(trials, -1)
    annual_means = active[indices[:, :months]].mean(axis=1) * 12
    return np.quantile(annual_means, [0.025, 0.975])


def run(root):
    root = Path(root)
    out = root / "outputs"
    config, returns, _, scores, _ = load(root)
    selection = json.loads((out / "selection_lock.json").read_text(encoding="utf-8"))[
        "selected_policy"
    ]
    months = returns.loc[config["test_start"] : config["test_end"]].index
    allocations = (0, 0.25, 0.5, 1.0)
    prior_weights = {a: pd.Series(0.0, index=returns.columns) for a in allocations}
    prior_holdings, ledger_rows, industry_rows = [], [], []

    for month in months:
        core = target_weights(scores.loc[month], "Equal industries")
        momentum = target_weights(
            scores.loc[month],
            selection,
            prior_holdings,
            config["holding_count"],
            config["buffer_rank"],
        )
        prior_holdings = list(momentum[momentum > 0].index)
        observed = returns.loc[month]
        if observed[core > 0].isna().any():
            raise ValueError(f"A selected industry has a missing return in {month}.")
        moves = observed.fillna(0)
        for allocation in allocations:
            weights = (1 - allocation) * core + allocation * momentum
            trade = rebalance(weights, prior_weights[allocation], config["cost_bps"])
            gross = float(weights @ moves)
            net = (1 - trade["cost"]) * (1 + gross) - 1
            ledger_rows.append(
                dict(
                    month=str(month),
                    sleeve_weight=allocation,
                    gross=gross,
                    net=net,
                    cost=trade["cost"],
                    gross_trade=trade["gross_traded"],
                    one_way_turnover=trade["turnover"],
                    largest_weight=weights.max(),
                    effective_industries=1 / (weights @ weights),
                )
            )
            prior_weights[allocation] = weights * (1 + moves) / (1 + gross)
        for name in returns.columns:
            excess_weight = momentum[name] - core[name]
            industry_rows.append(
                dict(industry=name, contribution=excess_weight * moves[name])
            )

    ledger = pd.DataFrame(ledger_rows)
    core_result = ledger[ledger.sleeve_weight == 0].reset_index(drop=True)
    summaries = []
    for allocation in allocations:
        portfolio = ledger[ledger.sleeve_weight == allocation].reset_index(drop=True)
        active = (portfolio.net - core_result.net).to_numpy()
        lower, upper = block_interval(active)
        wealth = np.r_[1.0, (1 + portfolio.net).cumprod()]
        drawdown = wealth / np.maximum.accumulate(wealth) - 1
        summaries.append(
            dict(
                sleeve_weight=allocation,
                months=len(portfolio),
                annual_growth=wealth[-1] ** (12 / len(portfolio)) - 1,
                max_drawdown=drawdown.min(),
                active_arithmetic=active.mean() * 12,
                tracking_error=active.std(ddof=1) * np.sqrt(12),
                block_low=lower,
                block_high=upper,
                annual_oneway_turnover=portfolio.one_way_turnover.mean() * 12,
                annual_cost_bp=portfolio.cost.mean() * 12 * 10000,
                largest_target=portfolio.largest_weight.max(),
                effective_industries=portfolio.effective_industries.mean(),
            )
        )

    attribution = (
        pd.DataFrame(industry_rows)
        .groupby("industry")
        .contribution.mean()
        .mul(12)
        .sort_values(ascending=False)
        .rename("annual_gross_active")
        .reset_index()
    )
    full_return = ledger[ledger.sleeve_weight == 1].gross.to_numpy()
    np.testing.assert_allclose(
        attribution.annual_gross_active.sum(),
        (full_return - core_result.gross).mean() * 12,
        atol=1e-10,
    )
    summary = pd.DataFrame(summaries)
    for name, table in [
        ("allocation_summary", summary),
        ("allocation_monthly", ledger),
        ("active_industries", attribution),
    ]:
        table.to_csv(out / f"{name}.csv", index=False, float_format="%.12g")
    return summary, ledger, attribution
