#!/usr/bin/env python3
"""
Figure 2: Bitcoin transaction fees ordered by size (log scale),
showing a sharp transition between near-zero fees and the upper tail.

Narrative: Most transactions pay very low fees; a small upper tail pays
substantial fees and accounts for a large share of total fee revenue
(priority-queue interpretation: only the most impatient pay for priority).

Data: mempool_space_data.db (transactions table: fee, size).
X-axis: transaction index when ordered by size (smallest to largest).
Y-axis: fee in satoshis, log scale.
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
DB_PATH = Path(__file__).parent.parent / "mempool_space_data.db"
OUTPUT_PATH = Path(__file__).parent.parent / "plots" / "fees_ordered_by_size.png"

# Define "upper tail" as top share of transactions by fee (drives most revenue)
UPPER_TAIL_PERCENTILE = 10  # top 10% by fee


def main():
    print(f"Loading data from {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)

    # Load fee and size; order by size in SQL for efficiency
    query = """
    SELECT fee, size
    FROM transactions
    WHERE size IS NOT NULL AND size > 0
    ORDER BY size ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    n = len(df)
    print(f"Loaded {n:,} transactions")

    # Index when ordered by size (1-based for plot)
    df["order_by_size"] = np.arange(1, n + 1, dtype=np.int64)

    # Log-scale fee: use log10(fee + 1) so zero fees plot at 0
    df["log_fee"] = np.log10(df["fee"].astype(np.float64) + 1)

    # Classify into near-zero vs upper tail (by fee) for visual emphasis
    fee_threshold = np.percentile(df["fee"], 100 - UPPER_TAIL_PERCENTILE)
    df["upper_tail"] = df["fee"] >= fee_threshold

    total_fee_revenue = df["fee"].sum()
    tail_fee_revenue = df.loc[df["upper_tail"], "fee"].sum()
    tail_tx_share_pct = 100 * df["upper_tail"].sum() / n
    tail_revenue_share_pct = 100 * tail_fee_revenue / total_fee_revenue if total_fee_revenue > 0 else 0

    # Optional: subsample for huge datasets to keep plot fast and file small
    max_points = 500_000
    if n > max_points:
        step = max(1, n // max_points)
        plot_df = df.iloc[::step].copy()
        print(f"Subsampling to {len(plot_df):,} points for plotting")
    else:
        plot_df = df

    fig, ax = plt.subplots(figsize=(11, 6))

    # Plot near-zero-fee majority first (background)
    low = plot_df[~plot_df["upper_tail"]]
    ax.scatter(
        low["order_by_size"],
        low["log_fee"],
        s=0.4,
        alpha=0.4,
        c="#64748b",
        label="Near-zero / low fees (majority)",
        rasterized=True,
    )

    # Plot upper tail on top so it stands out
    high = plot_df[plot_df["upper_tail"]]
    ax.scatter(
        high["order_by_size"],
        high["log_fee"],
        s=0.6,
        alpha=0.7,
        c="#ea580c",
        label=f"Upper tail (top {UPPER_TAIL_PERCENTILE}% by fee)",
        rasterized=True,
    )

    ax.set_xlabel("Transaction order by size (smallest → largest)", fontsize=12)
    ax.set_ylabel("Log₁₀(Fee + 1) [satoshis]", fontsize=12)
    ax.set_title(
        "Bitcoin transaction fees ordered by size (log scale)\n"
        "Sharp transition between near-zero fees and the fee-paying upper tail",
        fontsize=13,
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Key statistic: small share of tx, large share of revenue (priority-queue idea)
    textstr = (
        f"Top {UPPER_TAIL_PERCENTILE}% of transactions by fee:\n"
        f"  → {tail_revenue_share_pct:.0f}% of total fee revenue"
    )
    props = dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.9)
    ax.text(
        0.02,
        0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=props,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved figure to {OUTPUT_PATH}")

    # Stats
    print(f"\nStatistics:")
    print(f"  Fee range: {df['fee'].min():,} - {df['fee'].max():,} satoshis")
    print(f"  Size range: {df['size'].min():,} - {df['size'].max():,} bytes")
    zero_pct = 100 * (df["fee"] == 0).sum() / len(df)
    print(f"  Zero-fee share: {zero_pct:.2f}%")
    print(f"  Top {UPPER_TAIL_PERCENTILE}% by fee: {tail_tx_share_pct:.1f}% of tx, {tail_revenue_share_pct:.1f}% of fee revenue")


if __name__ == "__main__":
    main()
