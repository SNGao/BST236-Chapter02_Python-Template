"""Exploratory data analysis for the Boston Housing dataset.

Loads the processed CSV produced by src/00_rawdata_process.py and prints
basic summaries + saves a few key plots.

Usage:
  python sandbox/boston_eda.py
  python sandbox/boston_eda.py --data data/boston.csv --outdir src/output/eda
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compute_summary(df: pd.DataFrame) -> dict:
    summary: dict = {}

    summary["shape"] = {"rows": int(df.shape[0]), "cols": int(df.shape[1])}
    summary["dtypes"] = df.dtypes.astype(str).to_dict()

    missing_by_col = df.isna().sum().astype(int)
    summary["missing"] = {
        "total": int(missing_by_col.sum()),
        "by_col": missing_by_col.to_dict(),
        "rows_with_any_missing": int(df.isna().any(axis=1).sum()),
    }

    if "MEDV" in df.columns:
        medv = df["MEDV"].astype(float)
        summary["MEDV"] = {
            "min": float(medv.min()),
            "p25": float(medv.quantile(0.25)),
            "median": float(medv.median()),
            "mean": float(medv.mean()),
            "p75": float(medv.quantile(0.75)),
            "max": float(medv.max()),
            "std": float(medv.std(ddof=1)),
            "capped_at_50_count": int((medv >= 50).sum()),
        }

        corr = df.corr(numeric_only=True)["MEDV"].sort_values(ascending=False)
        summary["corr_with_MEDV"] = {
            "top": corr.head(8).round(3).to_dict(),
            "bottom": corr.tail(8).round(3).to_dict(),
        }

    # predictor-predictor correlations (absolute)
    predictors = [c for c in df.columns if c != "MEDV"]
    cm = df[predictors].corr(numeric_only=True)
    abs_vals = cm.abs().to_numpy(copy=True)
    np.fill_diagonal(abs_vals, 0.0)
    abs_cm = pd.DataFrame(abs_vals, index=cm.index, columns=cm.columns)

    stacked = abs_cm.stack().sort_values(ascending=False)
    seen: set[tuple[str, str]] = set()
    top_pairs: list[dict] = []
    for (a, b), abs_corr in stacked.items():
        key = tuple(sorted((str(a), str(b))))
        if key in seen:
            continue
        seen.add(key)
        corr_val = float(cm.loc[a, b])
        top_pairs.append({"pair": list(key), "corr": corr_val, "abs_corr": float(abs_corr)})
        if len(top_pairs) >= 12:
            break

    summary["top_abs_predictor_pairs"] = top_pairs

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        quantiles = df[numeric_cols].quantile([0.01, 0.05, 0.5, 0.95, 0.99]).T
        quantiles.columns = ["p01", "p05", "p50", "p95", "p99"]
        summary["numeric_quantiles"] = quantiles.round(6).to_dict(orient="index")

        skew = df[numeric_cols].skew(numeric_only=True)
        summary["numeric_skew"] = skew.round(6).to_dict()

    return summary


def save_plots(df: pd.DataFrame, outdir: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    _safe_mkdir(outdir)

    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, cmap="vlag", center=0, square=True)
    plt.title("Correlation heatmap (numeric columns)")
    plt.tight_layout()
    plt.savefig(outdir / "corr_heatmap.png", dpi=200)
    plt.close()

    if "MEDV" in df.columns:
        # MEDV histogram
        plt.figure(figsize=(10, 6))
        sns.histplot(df["MEDV"], bins=30, kde=True)
        plt.title("MEDV distribution")
        plt.tight_layout()
        plt.savefig(outdir / "medv_hist.png", dpi=200)
        plt.close()

        # Key scatter plots
        for x in ["RM", "LSTAT", "PTRATIO", "NOX"]:
            if x not in df.columns:
                continue
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=df, x=x, y="MEDV", alpha=0.75, s=25)
            sns.regplot(data=df, x=x, y="MEDV", scatter=False, color="black", line_kws={"lw": 1})
            plt.title(f"MEDV vs {x}")
            plt.tight_layout()
            plt.savefig(outdir / f"scatter_{x.lower()}_medv.png", dpi=200)
            plt.close()

        # CHAS group boxplot
        if "CHAS" in df.columns:
            plt.figure(figsize=(8, 6))
            sns.boxplot(data=df, x="CHAS", y="MEDV")
            plt.title("MEDV by CHAS")
            plt.tight_layout()
            plt.savefig(outdir / "box_medv_by_chas.png", dpi=200)
            plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="EDA for the Boston Housing dataset")
    parser.add_argument("--data", default="data/boston.csv", help="Path to processed CSV")
    parser.add_argument("--outdir", default="src/output/eda", help="Directory to write plots/summary")
    args = parser.parse_args()

    data_path = Path(args.data)
    outdir = Path(args.outdir)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find {data_path}. If starting from raw text, run: python src/00_rawdata_process.py"
        )

    df = pd.read_csv(data_path)

    summary = compute_summary(df)

    _safe_mkdir(outdir)
    with open(outdir / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print a human-readable subset
    print("Dataset:", data_path)
    print("Shape:", summary["shape"])
    print("Missing total:", summary["missing"]["total"])

    if "MEDV" in summary:
        print("\nMEDV:")
        for k in ["min", "p25", "median", "mean", "p75", "max", "std", "capped_at_50_count"]:
            print(f"  {k}: {summary['MEDV'][k]}")

        print("\nCorrelations with MEDV (top):")
        for k, v in list(summary["corr_with_MEDV"]["top"].items()):
            print(f"  {k}: {v}")

        print("\nCorrelations with MEDV (bottom):")
        for k, v in list(summary["corr_with_MEDV"]["bottom"].items()):
            print(f"  {k}: {v}")

    print("\nTop absolute predictor-predictor correlations:")
    for item in summary["top_abs_predictor_pairs"]:
        a, b = item["pair"]
        print(f"  {a} vs {b}: corr={item['corr']:.3f} (|corr|={item['abs_corr']:.3f})")

    # Quick scale/outlier view for a few commonly-discussed variables
    key_cols = [c for c in ["CRIM", "RM", "LSTAT", "TAX", "NOX", "DIS"] if c in df.columns]
    if key_cols and "numeric_quantiles" in summary:
        print("\nKey column quantiles (p01/p50/p99):")
        for c in key_cols:
            q = summary["numeric_quantiles"][c]
            print(f"  {c}: p01={q['p01']}, p50={q['p50']}, p99={q['p99']}")

    save_plots(df, outdir)
    print(f"\nWrote summary + plots to: {outdir}")


if __name__ == "__main__":
    main()
