"""Command-line entry point.

    python -m imaging_pdm evaluate     # leave-one-site-out, with and without the spatial layer
    python -m imaging_pdm alerts       # current maintenance work list: who fails next, and why
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .features import BASE_FEATURES, SPATIAL_FEATURES, attach_spatial, build_dataset, site_failure_rates
from .fleet import generate_fleet
from .model import RootCauseModel, TimingModel, leave_one_site_out, summarize, ALERT_HORIZON_DAYS


def cmd_evaluate(args) -> None:
    data = build_dataset(generate_fleet(seed=args.seed))
    base, spatial = leave_one_site_out(data, spatial=False), leave_one_site_out(data, spatial=True)
    per_site = pd.DataFrame([r.__dict__ for r in base + spatial])
    pd.set_option("display.width", 200)
    print("Mean over 7 held-out sites\n")
    print(summarize(base + spatial).rename(index={False: "telemetry only", True: "+ spatial layer"}).to_string())
    print("\nPer site (c-index / alert recall)\n")
    pivot = per_site.pivot(index="site", columns="spatial", values=["c_index", "alert_recall"]).round(3)
    print(pivot.to_string())
    if args.plot:
        _plot(per_site, data, Path(args.plot))


def _plot(per_site: pd.DataFrame, data: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.25]})
    rates = site_failure_rates(data)
    sc = ax1.scatter(rates["lon"], rates["lat"], s=60 + 140 * rates["cooling"], c=rates["cooling"], cmap="viridis")
    for site, r in rates.iterrows():
        dx = -52 if site == "Bengaluru" else 6  # keep neighboring labels apart
        ax1.annotate(site, (r["lon"], r["lat"]), xytext=(dx, 4), textcoords="offset points", fontsize=8)
    ax1.set_title("Cooling failures per machine-year by site", loc="left", fontsize=10)
    ax1.set_xlabel("longitude")
    ax1.set_ylabel("latitude")
    fig.colorbar(sc, ax=ax1, shrink=0.8)

    p = per_site.pivot(index="site", columns="spatial", values="c_index").sort_values(True)
    y = range(len(p))
    ax2.barh([i + 0.2 for i in y], p[False], height=0.4, color="#9ca3af", label="telemetry only")
    ax2.barh([i - 0.2 for i in y], p[True], height=0.4, color="#2563eb", label="+ spatial layer")
    ax2.set_yticks(list(y), p.index)
    ax2.set_xlim(0.5, 1.0)
    ax2.set_title("Failure-timing concordance at an unseen site", loc="left", fontsize=10)
    ax2.legend(frameon=False, fontsize=8, loc="lower right")
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"\nsaved {out}")


def cmd_alerts(args) -> None:
    data = build_dataset(generate_fleet(seed=args.seed))
    data = attach_spatial(data, site_failure_rates(data))
    features = BASE_FEATURES + SPATIAL_FEATURES
    timing = TimingModel(features).fit(data.iloc[::2])
    near = data[(data["event"] == 1) & (data["days_to_failure"] <= ALERT_HORIZON_DAYS)]
    cause = RootCauseModel(features).fit(near)

    today = data[data["date"] == data["date"].max()].copy()
    today["predicted_days_to_failure"] = timing.predict_days(today).round(0)
    today["likely_root_cause"] = cause.predict(today)
    cols = ["machine_id", "site", "modality", "predicted_days_to_failure", "likely_root_cause", "days_since_service"]
    print(f"Maintenance work list for {today['date'].max():%Y-%m-%d} (top {args.top})\n")
    print(today.sort_values("predicted_days_to_failure")[cols].head(args.top).to_string(index=False))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="imaging_pdm")
    p.add_argument("--seed", type=int, default=3)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--plot", default="docs/results.png", help="write a figure here ('' to skip)")
    e.set_defaults(func=cmd_evaluate)
    a = sub.add_parser("alerts")
    a.add_argument("--top", type=int, default=10)
    a.set_defaults(func=cmd_alerts)
    args = p.parse_args(argv)
    args.func(args)
