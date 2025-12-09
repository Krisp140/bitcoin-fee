#!/usr/bin/env python3
"""
Lasso-based feature selection to check colinearity between mempool congestion
(`rho_t`) and transaction-level predictors from phase3_fee_model_new_data.ipynb.

Default settings:
- Samples random blocks from the same SQLite DB used in the notebook
- Merges Phase 2 congestion (`rho_t`) from the pickle artifact
- Runs a standardized LassoCV and prints coefficients and simple correlations

Usage:
  python scripts/lasso_congestion.py \
    --db-path /home/armin/datalake/data-samples/11-24-2025-15m-data-lake.db \
    --phase2-path /home/kristian/notebooks/phase2_derived_features.pkl

Optional flags:
  --blocks 2000        Number of distinct blocks to sample
  --max-rows 500000    Downsample after merge to limit memory
  --random-seed 42
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lasso feature selection for mempool congestion.")
    parser.add_argument(
        "--db-path",
        default=os.path.expanduser("/home/armin/datalake/data-samples/11-24-2025-15m-data-lake.db"),
        help="SQLite database path (same as the notebook).",
    )
    parser.add_argument(
        "--phase2-path",
        default="/home/kristian/notebooks/phase2_derived_features.pkl",
        help="Pickle with Phase 2 features containing `tx_id` and `rho_t`.",
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=2000,
        help="Number of distinct blocks to sample (maintains block structure).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=500_000,
        help="Maximum merged rows to keep (downsampled for memory).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    return parser.parse_args()


def sample_transactions(db_path: str, blocks: int, rng: np.random.Generator) -> pd.DataFrame:
    """Sample transactions from random blocks to mirror notebook sampling."""
    with sqlite3.connect(db_path) as conn:
        blocks_query = """
        SELECT DISTINCT conf_block_hash
        FROM mempool_transactions
        WHERE conf_block_hash IS NOT NULL
        ORDER BY RANDOM()
        LIMIT ?
        """
        sampled_blocks = pd.read_sql_query(blocks_query, conn, params=(blocks,))["conf_block_hash"].tolist()
        placeholders = ",".join(["?"] * len(sampled_blocks))
        tx_query = f"""
        SELECT
            tx_id,
            weight,
            size,
            absolute_fee,
            fee_rate,
            mempool_size,
            mempool_tx_count,
            waittime
        FROM mempool_transactions
        WHERE conf_block_hash IN ({placeholders})
        """
        txs = pd.read_sql_query(tx_query, conn, params=sampled_blocks)
    return txs


def load_phase2_features(path: str) -> pd.DataFrame:
    phase2 = pd.read_pickle(path)[["tx_id", "rho_t"]]
    return phase2


def run_lasso(df: pd.DataFrame, feature_cols: List[str], target: str) -> None:
    X = df[feature_cols].values
    y = df[target].values

    pipe = make_pipeline(
        StandardScaler(),
        LassoCV(
            cv=5,
            n_alphas=50,
            random_state=42,
            n_jobs=-1,
        ),
    )
    pipe.fit(X, y)

    lasso = pipe.named_steps["lassocv"]
    coefs = lasso.coef_
    alpha = lasso.alpha_

    print(f"\nSelected alpha (lambda): {alpha:.4f}")
    print("\nLasso coefficients (standardized features):")
    rows = []
    for feat, coef in zip(feature_cols, coefs):
        rows.append((feat, coef, abs(coef)))
    rows.sort(key=lambda t: t[2], reverse=True)

    print(f"{'feature':<20} {'coef':>12} {'|coef|':>12}")
    print("-" * 46)
    for feat, coef, ab in rows:
        print(f"{feat:<20} {coef:>12.4f} {ab:>12.4f}")

    print("\nZero/near-zero coefficients imply the feature was deemed redundant/noisy under L1.")


def correlations(df: pd.DataFrame, cols: List[str], target: str) -> None:
    print("\nPearson correlations:")
    corr = df[cols + [target]].corr()
    for col in cols:
        print(f"{col:<20} vs {target:<8}: {corr.loc[col, target]: .4f}")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.random_seed)

    print(f"DB: {args.db_path}")
    print(f"Phase 2 features: {args.phase2_path}")
    print(f"Blocks sampled: {args.blocks}")
    print(f"Max rows after merge: {args.max_rows}")

    txs = sample_transactions(args.db_path, args.blocks, rng)
    print(f"Loaded {len(txs):,} transactions before merge")

    phase2 = load_phase2_features(args.phase2_path)
    merged = txs.merge(phase2, on="tx_id", how="inner")
    merged = merged.dropna()

    if args.max_rows and len(merged) > args.max_rows:
        merged = merged.sample(args.max_rows, random_state=args.random_seed)
        print(f"Downsampled to {len(merged):,} rows for analysis")
    else:
        print(f"Rows after merge/dropna: {len(merged):,}")

    feature_cols = [
        "weight",          # proxy for vsize*4
        "size",            # bytes
        "absolute_fee",    # sats
        "fee_rate",        # sat/vbyte (expected)
        "mempool_size",    # aggregate mempool vsize
        "mempool_tx_count",
        "waittime",        # observed wait time
    ]
    target = "rho_t"

    missing = [c for c in feature_cols + [target] if c not in merged.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    run_lasso(merged, feature_cols, target)
    correlations(merged, feature_cols, target)

    print("\nDone.")


if __name__ == "__main__":
    main()

