#!/usr/bin/env python3
"""
Plot a time series of mempool tx count aggregated per block across the full dataset.

For each distinct block (conf_block_hash), the script computes:
- first_seen: earliest found_at for transactions in that block
- avg_mempool_tx_count: average mempool_tx_count across txs in the block
- tx_count: number of txs in the block sample

It then plots the block-level time series ordered by first_seen.

Usage:
  python scripts/plot_block_timeseries.py \
    --db-path /home/armin/datalake/data-samples/11-24-2025-15m-data-lake.db \
    --output block_timeseries.png

Optional:
  --limit-blocks 0     # 0 means all blocks; set to cap the number plotted
  --dpi 150            # output resolution
"""

from __future__ import annotations

import argparse
import os
import sqlite3

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot block-level mempool metrics over time.")
    parser.add_argument(
        "--db-path",
        default=os.path.expanduser("/home/armin/datalake/data-samples/11-24-2025-15m-data-lake.db"),
        help="SQLite database path.",
    )
    parser.add_argument(
        "--output",
        default="block_timeseries.png",
        help="Output plot filename (PNG).",
    )
    parser.add_argument(
        "--limit-blocks",
        type=int,
        default=0,
        help="Max number of blocks to plot (ordered by first_seen). 0 = all.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output image DPI.",
    )
    return parser.parse_args()


def load_block_metrics(db_path: str, limit_blocks: int) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        limit_clause = "" if limit_blocks <= 0 else f"LIMIT {limit_blocks}"
        query = f"""
        SELECT
            conf_block_hash,
            MIN(found_at) AS first_seen,
            AVG(mempool_tx_count) AS avg_mempool_tx_count,
            COUNT(*) AS tx_count
        FROM mempool_transactions
        WHERE conf_block_hash IS NOT NULL
        GROUP BY conf_block_hash
        ORDER BY first_seen
        {limit_clause}
        """
        df = pd.read_sql_query(query, conn)
    df["first_seen"] = pd.to_datetime(df["first_seen"])
    return df


def plot_timeseries(df: pd.DataFrame, output: str, dpi: int) -> None:
    fig, ax1 = plt.subplots(figsize=(14, 6))

    ax1.plot(df["first_seen"], df["avg_mempool_tx_count"], label="avg_mempool_tx_count", color="tab:blue", alpha=0.8)
    ax1.set_ylabel("Avg mempool_tx_count", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax1.set_title("Block-level mempool tx count over time (ordered by first_seen)")
    ax1.set_xlabel("Block time (first_seen)")
    fig.tight_layout()
    fig.savefig(output, dpi=dpi)
    print(f"Saved plot to {output}")


def main() -> None:
    args = parse_args()
    print(f"DB: {args.db_path}")
    print(f"Output: {args.output}")
    print(f"Limit blocks: {args.limit_blocks if args.limit_blocks > 0 else 'all'}")

    df = load_block_metrics(args.db_path, args.limit_blocks)
    print(f"Loaded {len(df):,} blocks")
    print(df[["conf_block_hash", "first_seen", "avg_mempool_tx_count", "tx_count"]].head())

    plot_timeseries(df, args.output, args.dpi)


if __name__ == "__main__":
    main()

