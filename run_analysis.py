from pathlib import Path
from fetch_data import ensure_inputs
from equity.allocation_figure import draw
import argparse
from equity.analysis import develop, final_test
from equity.allocation import run as allocation_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["develop", "test", "all"], default="all")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    ensure_inputs(root)
    if args.stage in ["develop", "all"]:
        lock, summary = develop(root)
        print("Validation selection:", lock["selected_policy"])
        print("Validation information ratios:", lock["validation_information_ratios"])
    if args.stage in ["test", "all"]:
        r = final_test(root)
        allocation_test(root)
        draw(root / "outputs")
        for s in r["summary"]:
            if s["block"] == "Final test":
                print(
                    s["policy"],
                    "CAGR",
                    round(s["cagr"], 6),
                    "Sharpe",
                    round(s["sharpe"], 4),
                    "drawdown",
                    round(s["max_drawdown"], 4),
                    "turnover",
                    round(s["annual_turnover"], 3),
                )
