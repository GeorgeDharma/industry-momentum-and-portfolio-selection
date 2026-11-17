"""A fixed research design with selection completed before final evaluation."""

from pathlib import Path
import json, hashlib, re
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .engine import (
    read_monthly,
    signals,
    backtest,
    metrics,
    predictive_diagnostics,
    mean_hac,
    hac_ols,
    ranked_codes,
)


def load(root, end=None):
    """Load archived monthly returns, design settings, lagged scores and industry names."""
    root = Path(root)
    cfg = json.loads((root / "data/design.json").read_text(encoding="utf-8"))
    source = root / "data/source"
    returns = read_monthly(source / "industry.csv")
    factors = read_monthly(source / "factors.csv").join(
        read_monthly(source / "momentum.csv"), how="inner"
    )
    if end is not None:
        returns = returns.loc[:end]
        factors = factors.loc[:end]
    score = signals(returns, cfg["lookback_months"], cfg["shift_months"])
    names = {}
    for line in (source / "definitions.csv").read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*\d{1,2}\s+(\w+)\s+(.+)$", line)
        if m:
            names[m.group(1)] = m.group(2).strip()
    return cfg, returns, factors, score, names


def fingerprint(root):
    """Hash the research design and source tables used in the selection record."""
    root = Path(root)
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            root / "data/design.json",
            *sorted((root / "data/source").glob("*.csv")),
        ]
    }
    return hashes


def evaluate(
    cfg, returns, factors, score, start, end, cost=None, count=None, buffer=None
):
    """Run every fixed allocation policy over one stated date range."""
    rows = {}
    hold = {}
    summary = []
    for policy in cfg["policies"]:
        rows[policy], hold[policy] = backtest(
            returns,
            score,
            start,
            end,
            policy,
            cfg["cost_bps"] if cost is None else cost,
            cfg["holding_count"] if count is None else count,
            cfg["buffer_rank"] if buffer is None else buffer,
        )
    for policy in cfg["policies"]:
        frame = rows[policy]
        rf = factors.loc[start:end, "RF"]
        summary.append(
            {
                "policy": policy,
                **metrics(frame, rf, rows["Equal industries"].net_return),
            }
        )
    return rows, hold, summary


def develop(root):
    """Use development and validation data to select a rule before final evaluation."""
    root = Path(root)
    out = root / "outputs"
    out.mkdir(exist_ok=True)
    # Truncate the panel before any research calculation, including signals.
    cfg = json.loads((root / "data/design.json").read_text(encoding="utf-8"))
    cfg, returns, factors, score, names = load(root, cfg["validation_end"])
    summaries = []
    for block, start, end in [
        ("Development", cfg["start"], cfg["development_end"]),
        ("Validation", cfg["validation_start"], cfg["validation_end"]),
    ]:
        rows, hold, summary = evaluate(cfg, returns, factors, score, start, end)
        summaries.extend({"block": block, **x} for x in summary)
    candidates = [
        x
        for x in summaries
        if x["block"] == "Validation" and x["policy"] != "Equal industries"
    ]
    winner = max(
        candidates,
        key=lambda x: (x["information_ratio"], x["policy"] == "Top 10 momentum"),
    )
    lock = {
        "selected_policy": winner["policy"],
        "selection_rule": cfg["selection_rule"],
        "validation_information_ratios": {
            x["policy"]: x["information_ratio"] for x in candidates
        },
        "input_sha256": fingerprint(root),
        "final_results_examined": False,
    }
    (out / "selection_lock.json").write_text(json.dumps(lock, indent=2))
    (out / "development_validation.json").write_text(json.dumps(summaries, indent=2))
    pd.DataFrame(summaries).drop(columns=["active_hac"]).to_csv(
        out / "development_validation.csv", index=False
    )
    return lock, summaries


def final_test(root):
    """Verify the selected experiment and evaluate its final period without retuning."""
    root = Path(root)
    out = root / "outputs"
    lock = json.loads((out / "selection_lock.json").read_text(encoding="utf-8"))
    if lock["input_sha256"] != fingerprint(root):
        raise ValueError(
            "Design or source data changed after selection. Start a new experiment."
        )
    cfg, returns, factors, score, names = load(root)
    start, end = cfg["test_start"], cfg["test_end"]
    selected = lock["selected_policy"]
    rows, hold, summary = evaluate(cfg, returns, factors, score, start, end)
    all_summary = json.loads(
        (out / "development_validation.json").read_text(encoding="utf-8")
    ) + [{"block": "Final test", **x} for x in summary]
    diagnostics = predictive_diagnostics(
        returns, score, start, end, cfg["holding_count"]
    )
    ic = mean_hac(diagnostics.rank_ic)
    spread = mean_hac(diagnostics.spread)
    factor_rows = []
    f = factors.loc[start:end]
    for policy in cfg["policies"]:
        fit = hac_ols(
            rows[policy].net_return.to_numpy() - f.RF.to_numpy(),
            f[["Mkt-RF", "SMB", "HML", "Mom"]].to_numpy(),
            6,
        )
        for i, name in enumerate(
            ["Monthly alpha", "Market", "Size", "Value", "Stock momentum"]
        ):
            factor_rows.append(
                {
                    "policy": policy,
                    "exposure": name,
                    "coefficient": fit["coefficients"][i],
                    "se": fit["se"][i],
                    "ci_low": fit["ci_low"][i],
                    "ci_high": fit["ci_high"][i],
                    "t": fit["t"][i],
                    "r_squared": fit["r_squared"],
                }
            )
    cost_rows = []
    for cost in cfg["cost_scenarios_bps"]:
        _, _, stats = evaluate(cfg, returns, factors, score, start, end, cost=cost)
        cost_rows.extend({"cost_bps": cost, **x} for x in stats)
    nearby = []
    for count in cfg["nearby_parameters"]["holding_counts"]:
        offsets = (
            cfg["nearby_parameters"]["buffer_offsets"]
            if selected == "Buffered momentum"
            else [0]
        )
        for offset in offsets:
            fr, _ = backtest(
                returns,
                score,
                start,
                end,
                selected,
                cfg["cost_bps"],
                count,
                count + offset,
            )
            nearby.append(
                {
                    "holdings": count,
                    "exit_rank": count + offset,
                    "policy": selected,
                    **metrics(fr, f.RF, rows["Equal industries"].net_return),
                }
            )
    subperiods = []
    annual = []
    for policy in cfg["policies"]:
        frame = rows[policy]
        for lo, hi in cfg["subperiods"]:
            mask = frame.month.between(lo, hi)
            subperiods.append(
                {
                    "period": lo + " to " + hi,
                    "policy": policy,
                    **metrics(
                        frame.loc[mask],
                        factors.loc[lo:hi, "RF"],
                        rows["Equal industries"].loc[mask, "net_return"],
                    ),
                }
            )
        for year, group in frame.groupby(frame.month.str[:4]):
            annual.append(
                {
                    "year": int(year),
                    "policy": policy,
                    "net_return": float(np.prod(1 + group.net_return) - 1),
                    "gross_return": float(np.prod(1 + group.gross_return) - 1),
                    "turnover": float(group.turnover.sum()),
                }
            )
    hh = hold[selected].copy()
    bb = hold["Equal industries"].copy()
    hh["gross_contribution"] = hh.target_weight * hh["return"]
    hh["benchmark_weight"] = bb.target_weight.to_numpy()
    hh["active_contribution"] = (hh.target_weight - hh.benchmark_weight) * hh["return"]
    contribution = (
        hh.groupby("industry")
        .agg(
            average_weight=("target_weight", "mean"),
            months_held=("target_weight", lambda x: int((x > 0).sum())),
            mean_contribution=("gross_contribution", "mean"),
            mean_active_contribution=("active_contribution", "mean"),
        )
        .reset_index()
    )
    contribution["annual_active_contribution"] = (
        12 * contribution.mean_active_contribution
    )
    contribution["name"] = contribution.industry.map(names)
    contribution = contribution.sort_values(
        "annual_active_contribution", ascending=False
    )
    sample_month = end
    sample = hh[hh.month == sample_month].copy()
    prior = (
        hold[selected][hold[selected].month == str(pd.Period(end, freq="M") - 1)]
        .set_index("industry")
        .target_weight
    )
    order = ranked_codes(score.loc[end])
    ranks = {code: i + 1 for i, code in enumerate(order)}
    sample["rank"] = sample.industry.map(ranks)
    sample["name"] = sample.industry.map(names)
    sample["previously_held"] = sample.industry.map(prior).gt(0).astype(int)
    lag_dates = [str(pd.Period(end, freq="M") - i) for i in range(12, 1, -1)]
    for month in lag_dates:
        sample[month] = sample.industry.map(returns.loc[month])
    sample = sample.sort_values("industry")
    missing = returns.loc["1969-01":end].isna()
    audit = {
        "source_header": (root / "data/source/industry.csv")
        .read_text(encoding="utf-8")
        .splitlines()[0],
        "first_month": str(returns.index[0]),
        "last_month": str(returns.index[-1]),
        "source_months": len(returns),
        "industry_count": len(returns.columns),
        "study_months": len(returns.loc[cfg["start"] : end]),
        "missing_1969_2025": int(missing.sum().sum()),
        "first_all_industries_month": str(returns.dropna().index[0]),
        "final_min_eligible": int(rows[selected].eligible.min()),
        "final_max_eligible": int(rows[selected].eligible.max()),
    }
    result = {
        "design": cfg,
        "selection": lock,
        "audit": audit,
        "industry_names": names,
        "summary": all_summary,
        "ic": ic,
        "spread": spread,
        "signal_diagnostics": diagnostics.to_dict("records"),
        "monthly": pd.concat(rows.values(), ignore_index=True).to_dict("records"),
        "factors": factor_rows,
        "costs": cost_rows,
        "nearby": nearby,
        "subperiods": subperiods,
        "annual": annual,
        "contribution": contribution.to_dict("records"),
        "sample": sample.to_dict("records"),
        "sample_months": lag_dates,
        "sample_month": sample_month,
        "sample_rebalance": rows[selected].iloc[-1].to_dict(),
    }
    for key in [
        "summary",
        "monthly",
        "factors",
        "costs",
        "nearby",
        "subperiods",
        "annual",
        "contribution",
        "sample",
        "signal_diagnostics",
    ]:
        df = pd.DataFrame(result[key])
        df.drop(columns=["active_hac"], errors="ignore").to_csv(
            out / (key + ".csv"), index=False
        )
    pd.concat(hold.values(), ignore_index=True).to_csv(
        out / "holdings.csv", index=False
    )
    returns.loc["1969-01":end].to_csv(out / "industry_returns.csv")
    factors.loc["1969-01":end].to_csv(out / "factor_returns.csv")
    (out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    charts(result, out)
    return result


def charts(result, out):
    family = "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": "#242B33",
            "axes.labelcolor": "#242B33",
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
        }
    )
    colors = {
        "Equal industries": "#626B73",
        "Top 10 momentum": "#A85E3B",
        "Buffered momentum": "#B79E8D",
    }
    data = pd.DataFrame(result["monthly"])
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for policy in result["design"]["policies"]:
        d = data[data.policy == policy]
        dates = pd.to_datetime(d.month)
        ax.plot(dates, d.net_wealth, label=policy, color=colors[policy], linewidth=1.7)
    ax.set_ylabel("Value of $1 after modelled costs")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.16)
    fig.tight_layout()
    fig.savefig(out / "wealth.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 3.3))
    for policy in result["design"]["policies"]:
        d = data[data.policy == policy]
        w = d.net_wealth.to_numpy()
        dd = w / np.maximum.accumulate(np.r_[1, w])[1:] - 1
        ax.plot(
            pd.to_datetime(d.month),
            100 * dd,
            label=policy,
            color=colors[policy],
            linewidth=1.5,
        )
    ax.set_ylabel("Decline from previous peak (%)")
    ax.legend(frameon=False, fontsize=10, loc="lower left")
    ax.grid(axis="y", alpha=0.16)
    fig.tight_layout()
    fig.savefig(out / "drawdowns.png", dpi=180)
    plt.close(fig)
    costs = pd.DataFrame(result["costs"])
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for policy in result["design"]["policies"]:
        d = costs[costs.policy == policy]
        ax.plot(
            d.cost_bps, 100 * d.cagr, marker="o", label=policy, color=colors[policy]
        )
    ax.set_xlabel("Cost per dollar bought or sold (basis points)")
    ax.set_ylabel("Compound annual return (%)")
    ax.legend(frameon=False, fontsize=10)
    ax.grid(axis="y", alpha=0.16)
    fig.tight_layout()
    fig.savefig(out / "cost_sensitivity.png", dpi=180)
    plt.close(fig)
    d = pd.DataFrame(result["signal_diagnostics"])
    fig, ax = plt.subplots(figsize=(9, 3.1))
    ax.plot(
        pd.to_datetime(d.month),
        d.rank_ic.rolling(12).mean(),
        color="#A85E3B",
        label="Trailing 12-month average",
    )
    ax.axhline(0, color="#626B73", linewidth=0.8)
    ax.axhline(
        d.rank_ic.mean(), color="#242B33", linestyle="--", label="Final-test mean"
    )
    ax.set_ylabel("Rank correlation")
    ax.legend(frameon=False, fontsize=10)
    ax.grid(axis="y", alpha=0.16)
    fig.tight_layout()
    fig.savefig(out / "signal_quality.png", dpi=180)
    plt.close(fig)
