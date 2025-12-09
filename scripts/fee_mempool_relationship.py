#!/usr/bin/env python3
"""
Analyze the relationship between transaction fee_rate and mempool congestion
(`mempool_size`) using the same SQLite dataset referenced in simple-correlation.ipynb.

The script:
 1) Samples blocks from the database (to preserve block structure)
 2) Loads minimal columns needed for the analysis
 3) Cleans and optionally downsamples rows
 4) Computes correlation stats (raw, log, and block-level medians)
 5) Produces binned summaries and diagnostic plots
"""

import argparse
import os
import sqlite3
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_DB_PATH = "/home/armin/datalake/data-samples/11-24-2025-15m-data-lake.db"


def load_sampled_transactions(db_path: str, block_sample: int) -> pd.DataFrame:
    """Load a block-sampled slice of the mempool_transactions table."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"SQLite DB not found at {db_path}")

    with sqlite3.connect(db_path) as conn:
        blocks_query = """
            SELECT DISTINCT conf_block_hash
            FROM mempool_transactions
            WHERE conf_block_hash IS NOT NULL
            ORDER BY RANDOM()
            LIMIT ?
        """
        sampled_blocks = pd.read_sql_query(blocks_query, conn, params=(block_sample,))
        block_ids = sampled_blocks["conf_block_hash"].tolist()

        if not block_ids:
            return pd.DataFrame()

        placeholders = ",".join(["?"] * len(block_ids))
        tx_query = f"""
            SELECT
                conf_block_hash,
                found_at,
                fee_rate,
                mempool_size,
                mempool_tx_count
            FROM mempool_transactions
            WHERE conf_block_hash IN ({placeholders})
        """
        txs = pd.read_sql_query(tx_query, conn, params=block_ids)

    return txs


def clean_and_sample(
    df: pd.DataFrame, row_limit: int, seed: int
) -> pd.DataFrame:
    """Filter to valid rows and optional downsample for speed."""
    if df.empty:
        return df

    df = df.copy()
    df = df[df["fee_rate"].notna() & df["mempool_size"].notna()]
    df = df[(df["fee_rate"] > 0) & (df["mempool_size"] > 0)]

    if row_limit and len(df) > row_limit:
        df = df.sample(row_limit, random_state=seed)

    df["log_fee_rate"] = np.log1p(df["fee_rate"])
    df["log_mempool_size"] = np.log1p(df["mempool_size"])
    return df


def correlation_summary(
    df: pd.DataFrame, block_level: pd.DataFrame
) -> Dict[str, float]:
    """Compute Pearson/Spearman correlations for raw, log, and block-level data."""
    metrics = {
        "pearson_raw": df["fee_rate"].corr(df["mempool_size"], method="pearson"),
        "spearman_raw": df["fee_rate"].corr(df["mempool_size"], method="spearman"),
        "pearson_log": df["log_fee_rate"].corr(df["log_mempool_size"], method="pearson"),
        "spearman_log": df["log_fee_rate"].corr(
            df["log_mempool_size"], method="spearman"
        ),
    }

    if not block_level.empty:
        metrics["pearson_block_medians"] = block_level["fee_rate_median"].corr(
            block_level["mempool_size_median"], method="pearson"
        )
        metrics["spearman_block_medians"] = block_level["fee_rate_median"].corr(
            block_level["mempool_size_median"], method="spearman"
        )

    return metrics


def build_block_level(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate by block to smooth per-tx noise."""
    if df.empty:
        return df

    grouped = (
        df.groupby("conf_block_hash")
        .agg(
            fee_rate_median=("fee_rate", "median"),
            fee_rate_mean=("fee_rate", "mean"),
            mempool_size_median=("mempool_size", "median"),
            mempool_size_mean=("mempool_size", "mean"),
            mempool_tx_count_median=("mempool_tx_count", "median"),
            n_txs=("fee_rate", "size"),
        )
        .reset_index()
    )
    return grouped


def build_mempool_bins(df: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Quantile-bin mempool_size and summarize fee_rate stats."""
    if df.empty:
        return df

    df = df.copy()
    df["mempool_bin"] = pd.qcut(
        df["mempool_size"], q=bins, labels=False, duplicates="drop"
    )

    bin_stats = (
        df.groupby("mempool_bin")
        .agg(
            mempool_min=("mempool_size", "min"),
            mempool_max=("mempool_size", "max"),
            mempool_mean=("mempool_size", "mean"),
            fee_rate_mean=("fee_rate", "mean"),
            fee_rate_median=("fee_rate", "median"),
            count=("fee_rate", "size"),
        )
        .reset_index()
    )
    bin_stats["mempool_midpoint"] = (bin_stats["mempool_min"] + bin_stats["mempool_max"]) / 2
    return bin_stats


def plot_hexbin(df: pd.DataFrame, path: str) -> None:
    """
    Hexbin scatter of fee_rate vs mempool_size using log1p-transformed values.
    Using the transformed columns avoids axis-scaling quirks that can lead to
    empty renders when the raw ranges are very wide.
    """
    if df.empty:
        return

    plot_df = df[["log_mempool_size", "log_fee_rate"]].replace([np.inf, -np.inf], np.nan)
    plot_df = plot_df.dropna()
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    hb = ax.hexbin(
        plot_df["log_mempool_size"],
        plot_df["log_fee_rate"],
        gridsize=80,
        bins="log",
        cmap="viridis",
        mincnt=1,
    )
    ax.set_xlabel("log1p(mempool_size)", fontweight="bold")
    ax.set_ylabel("log1p(fee_rate)", fontweight="bold")
    ax.set_title("fee_rate vs mempool_size (hexbin on log1p values)")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("log10(count)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_binned(bin_stats: pd.DataFrame, path: str) -> None:
    """Line plot of fee_rate stats across mempool_size quantile bins."""
    if bin_stats.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        bin_stats["mempool_midpoint"],
        bin_stats["fee_rate_median"],
        marker="o",
        label="Median fee_rate",
    )
    ax.plot(
        bin_stats["mempool_midpoint"],
        bin_stats["fee_rate_mean"],
        marker="o",
        linestyle="--",
        label="Mean fee_rate",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("mempool_size bin midpoint (transactions)", fontweight="bold")
    ax.set_ylabel("fee_rate (sat/vByte)", fontweight="bold")
    ax.set_title("fee_rate vs mempool_size (quantile bins, log-log)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_block_level(block_df: pd.DataFrame, path: str) -> None:
    """Scatter of block-level medians to show trend without per-tx noise."""
    if block_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        block_df["mempool_size_median"],
        block_df["fee_rate_median"],
        s=12,
        alpha=0.6,
        edgecolors="none",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Block median mempool_size", fontweight="bold")
    ax.set_ylabel("Block median fee_rate", fontweight="bold")
    ax.set_title("fee_rate vs mempool_size (block-level medians)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def summarize(
    df: pd.DataFrame, block_df: pd.DataFrame, corr: Dict[str, float], bin_stats: pd.DataFrame
) -> None:
    """Print a concise textual summary for CLI use."""
    print("\n=== DATA ===")
    print(f"Rows (clean): {len(df):,}")
    print(f"Blocks: {df['conf_block_hash'].nunique():,}")
    print(f"fee_rate >0 mean/median: {df['fee_rate'].mean():.2f} / {df['fee_rate'].median():.2f}")
    print(f"mempool_size mean/median: {df['mempool_size'].mean():.2f} / {df['mempool_size'].median():.2f}")

    print("\n=== CORRELATIONS ===")
    for k, v in corr.items():
        print(f"{k:24s}: {v: .4f}")

    if not block_df.empty:
        print("\nBlock-level rows:", len(block_df))

    if not bin_stats.empty:
        print("\n=== BINNED STATS (mempool_size quantiles) ===")
        printable = bin_stats[
            ["mempool_bin", "mempool_min", "mempool_max", "fee_rate_mean", "fee_rate_median", "count"]
        ].copy()
        printable["mempool_min"] = printable["mempool_min"].round(0)
        printable["mempool_max"] = printable["mempool_max"].round(0)
        printable["fee_rate_mean"] = printable["fee_rate_mean"].round(2)
        printable["fee_rate_median"] = printable["fee_rate_median"].round(2)
        print(printable.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Examine how fee_rate relates to mempool congestion (mempool_size).",
    )
    parser.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help="Path to the SQLite database (default matches simple-correlation.ipynb).",
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=2000,
        help="Number of random blocks to sample (preserves block structure).",
    )
    parser.add_argument(
        "--row-limit",
        type=int,
        default=1_000_000,
        help="Optional row cap after cleaning to keep runtime reasonable.",
    )
    parser.add_argument(
        "--plot-sample",
        type=int,
        default=400_000,
        help="Max rows to use for plots (hexbin).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Number of quantile bins for mempool_size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--output-dir",
        default="plots/fee_mempool",
        help="Directory to write plots and CSV summaries.",
    )

    args = parser.parse_args()
    np.random.seed(args.seed)

    print(f"\nLoading data from {args.db_path}")
    txs = load_sampled_transactions(args.db_path, args.blocks)
    if txs.empty:
        print("No data loaded; check DB path or sampling parameters.")
        return

    print(f"Loaded {len(txs):,} transactions from {txs['conf_block_hash'].nunique():,} blocks")
    txs = clean_and_sample(txs, args.row_limit, args.seed)
    print(f"After cleaning/downsampling: {len(txs):,} rows")

    block_df = build_block_level(txs)
    bin_stats = build_mempool_bins(txs, bins=args.bins)
    corr = correlation_summary(txs, block_df)

    os.makedirs(args.output_dir, exist_ok=True)
    plot_hexbin(
        txs.sample(min(args.plot_sample, len(txs)), random_state=args.seed),
        os.path.join(args.output_dir, "fee_vs_mempool_hexbin.png"),
    )
    plot_binned(bin_stats, os.path.join(args.output_dir, "fee_vs_mempool_binned.png"))
    plot_block_level(
        block_df, os.path.join(args.output_dir, "fee_vs_mempool_block_level.png")
    )

    bin_stats.to_csv(
        os.path.join(args.output_dir, "fee_vs_mempool_binned.csv"), index=False
    )
    block_df.to_csv(
        os.path.join(args.output_dir, "fee_vs_mempool_block_level.csv"), index=False
    )

    summarize(txs, block_df, corr, bin_stats)
    print(f"\nArtifacts written to {args.output_dir}")


if __name__ == "__main__":
    main()

