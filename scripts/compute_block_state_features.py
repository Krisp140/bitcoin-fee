#!/usr/bin/env python3
"""
Compute per-block state features from mempool_space_data.db transactions.

Aggregates fee-rate percentiles, block weight, inter-block timing, and EMA
features, storing results in a `block_state_features` table for use in the
fee estimation pipeline.

Usage:
    python scripts/compute_block_state_features.py
    python scripts/compute_block_state_features.py --stats
    python scripts/compute_block_state_features.py --source-db /path/to/source.db
"""

import argparse
import os
import sqlite3
import sys
import time

import numpy as np
import pandas as pd

# Defaults
MEMPOOL_SPACE_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mempool_space_data.db')
MEMPOOL_SPACE_DB = os.path.normpath(MEMPOOL_SPACE_DB)

SOURCE_DB = '/home/armin/datalake/data-samples/11-24-2025-15m-data-lake.db'

TABLE_NAME = 'block_state_features'

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    block_hash TEXT PRIMARY KEY,
    block_height INTEGER,
    block_time INTEGER,
    block_tx_count INTEGER,
    block_total_weight INTEGER,
    block_median_feerate REAL,
    block_p10_feerate REAL,
    block_p90_feerate REAL,
    time_since_last_block INTEGER,
    gap_to_prev_block INTEGER,
    ema_feerate_3block REAL,
    tx_arrival_rate_10m REAL
);
"""


def load_transactions(db_path: str) -> pd.DataFrame:
    """Load transaction-level data needed for block aggregation."""
    print(f"Loading transactions from {db_path} ...")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT block_hash, block_height, block_time, fee_rate, weight "
        "FROM transactions",
        conn,
    )
    conn.close()
    print(f"  Loaded {len(df):,} transactions across {df['block_hash'].nunique():,} blocks")
    return df


def compute_block_aggregates(tx_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-block fee-rate percentiles and weight totals."""
    print("Computing per-block aggregates ...")

    # Separate coinbase (fee_rate == 0) from real txs for fee stats
    real_tx = tx_df[tx_df['fee_rate'] > 0].copy()

    fee_agg = real_tx.groupby('block_hash')['fee_rate'].agg(
        block_median_feerate='median',
        block_p10_feerate=lambda x: np.percentile(x, 10),
        block_p90_feerate=lambda x: np.percentile(x, 90),
        block_tx_count='count',
    )

    weight_agg = tx_df.groupby('block_hash')['weight'].sum().rename('block_total_weight')

    # Block metadata (one row per block)
    meta = tx_df.groupby('block_hash').agg(
        block_height=('block_height', 'first'),
        block_time=('block_time', 'first'),
    )

    blocks = meta.join(fee_agg).join(weight_agg)
    blocks = blocks.sort_values('block_height').reset_index()

    print(f"  {len(blocks):,} blocks aggregated")
    return blocks


def compute_interblock_features(blocks: pd.DataFrame) -> pd.DataFrame:
    """Add time_since_last_block, gap_to_prev_block, and EMA features."""
    print("Computing inter-block features ...")

    blocks = blocks.sort_values('block_height').copy()

    # Time since previous block (seconds), clamped >= 0
    blocks['time_since_last_block'] = blocks['block_time'].diff().clip(lower=0).astype('Int64')

    # Height gap (detects missing blocks in dataset)
    blocks['gap_to_prev_block'] = blocks['block_height'].diff().astype('Int64')

    # Exponential moving average of median fee-rate (span=3 blocks)
    blocks['ema_feerate_3block'] = (
        blocks['block_median_feerate'].ewm(span=3, adjust=False).mean()
    )

    # First row has no previous block
    blocks.loc[blocks.index[0], 'time_since_last_block'] = pd.NA
    blocks.loc[blocks.index[0], 'gap_to_prev_block'] = pd.NA

    print("  Added time_since_last_block, gap_to_prev_block, ema_feerate_3block")
    return blocks


def compute_tx_arrival_rate(blocks: pd.DataFrame, source_db: str) -> pd.DataFrame:
    """
    Count transactions arriving in the 10 minutes before each block.

    Uses `found_at` from the source database.  Skipped if the source DB
    is not accessible.
    """
    if not os.path.exists(source_db):
        print(f"  Source DB not found ({source_db}), skipping tx_arrival_rate_10m")
        blocks['tx_arrival_rate_10m'] = pd.NA
        return blocks

    print(f"Computing tx arrival rates from {source_db} ...")
    conn = sqlite3.connect(f"file:{source_db}?mode=ro", uri=True)

    rates = []
    block_times = blocks[['block_hash', 'block_time']].dropna(subset=['block_time'])

    # Process in batches to avoid huge single queries
    BATCH = 500
    for start in range(0, len(block_times), BATCH):
        batch = block_times.iloc[start:start + BATCH]
        results = {}
        for _, row in batch.iterrows():
            bt = int(row['block_time'])
            # found_at is stored as ISO datetime; block_time is unix timestamp
            window_start = pd.Timestamp(bt - 600, unit='s').strftime('%Y-%m-%d %H:%M:%S')
            window_end = pd.Timestamp(bt, unit='s').strftime('%Y-%m-%d %H:%M:%S')
            cnt = conn.execute(
                "SELECT COUNT(*) FROM mempool_transactions "
                "WHERE found_at BETWEEN ? AND ?",
                (window_start, window_end),
            ).fetchone()[0]
            results[row['block_hash']] = cnt
        rates.append(pd.Series(results))

    conn.close()

    if rates:
        all_rates = pd.concat(rates)
        blocks['tx_arrival_rate_10m'] = blocks['block_hash'].map(all_rates)
    else:
        blocks['tx_arrival_rate_10m'] = pd.NA

    non_null = blocks['tx_arrival_rate_10m'].notna().sum()
    print(f"  Computed arrival rates for {non_null:,} blocks")
    return blocks


def write_to_db(blocks: pd.DataFrame, db_path: str) -> None:
    """Write block_state_features table (INSERT OR REPLACE for idempotency)."""
    print(f"Writing {len(blocks):,} rows to {TABLE_NAME} in {db_path} ...")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(CREATE_TABLE_SQL)

    cols = [
        'block_hash', 'block_height', 'block_time',
        'block_tx_count', 'block_total_weight',
        'block_median_feerate', 'block_p10_feerate', 'block_p90_feerate',
        'time_since_last_block', 'gap_to_prev_block',
        'ema_feerate_3block', 'tx_arrival_rate_10m',
    ]
    placeholders = ', '.join(['?'] * len(cols))
    col_names = ', '.join(cols)

    rows = blocks[cols].values.tolist()
    # Convert numpy/pandas types to native Python for sqlite
    clean_rows = []
    for row in rows:
        clean_rows.append(tuple(
            None if pd.isna(v) else (int(v) if isinstance(v, (np.integer,)) else
                                     float(v) if isinstance(v, (np.floating,)) else v)
            for v in row
        ))

    conn.executemany(
        f"INSERT OR REPLACE INTO {TABLE_NAME} ({col_names}) VALUES ({placeholders})",
        clean_rows,
    )
    conn.commit()
    conn.close()
    print(f"  Done.")


def print_stats(db_path: str) -> None:
    """Print summary statistics from the block_state_features table."""
    conn = sqlite3.connect(db_path)

    # Check table exists
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    if TABLE_NAME not in tables:
        print(f"Table '{TABLE_NAME}' does not exist in {db_path}")
        conn.close()
        return

    n = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
    if n == 0:
        print("Table is empty.")
        conn.close()
        return

    df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)
    conn.close()

    print("\n" + "=" * 60)
    print(f"BLOCK STATE FEATURES — {n:,} blocks")
    print("=" * 60)
    print(f"Height range: {df['block_height'].min()} – {df['block_height'].max()}")
    print(f"Time range:   {pd.Timestamp(df['block_time'].min(), unit='s')} – "
          f"{pd.Timestamp(df['block_time'].max(), unit='s')}")
    print()
    summary_cols = [
        'block_tx_count', 'block_total_weight',
        'block_median_feerate', 'block_p10_feerate', 'block_p90_feerate',
        'time_since_last_block', 'gap_to_prev_block',
        'ema_feerate_3block', 'tx_arrival_rate_10m',
    ]
    print(df[summary_cols].describe().round(2).to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-block state features from mempool_space_data.db"
    )
    parser.add_argument('--db', type=str, default=MEMPOOL_SPACE_DB,
                        help=f"mempool_space_data.db path (default: {MEMPOOL_SPACE_DB})")
    parser.add_argument('--source-db', type=str, default=SOURCE_DB,
                        help=f"Source DB for tx_arrival_rate (default: {SOURCE_DB})")
    parser.add_argument('--stats', action='store_true',
                        help="Print stats and exit")
    parser.add_argument('--skip-arrival-rate', action='store_true',
                        help="Skip tx_arrival_rate_10m computation (faster)")
    args = parser.parse_args()

    if args.stats:
        print_stats(args.db)
        return

    t0 = time.time()

    # Step 1: Load transactions
    tx_df = load_transactions(args.db)

    # Step 2: Per-block aggregates
    blocks = compute_block_aggregates(tx_df)
    del tx_df  # free memory

    # Step 3: Inter-block features
    blocks = compute_interblock_features(blocks)

    # Step 4: Tx arrival rate (optional, cross-DB)
    if not args.skip_arrival_rate:
        blocks = compute_tx_arrival_rate(blocks, args.source_db)
    else:
        blocks['tx_arrival_rate_10m'] = pd.NA
        print("  Skipped tx_arrival_rate_10m (--skip-arrival-rate)")

    # Step 5: Write to DB
    write_to_db(blocks, args.db)

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")
    print_stats(args.db)


if __name__ == '__main__':
    main()
