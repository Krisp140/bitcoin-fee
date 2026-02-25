#!/usr/bin/env python3
"""
Periodically poll mempool.space API to capture live mempool state snapshots.

Stores mempool depth, fee-rate histogram percentiles, recommended fees, and
projected next-block statistics in a `mempool_snapshots` table.

Usage:
    # Poll every 60s for 1 hour
    python scripts/scrape_mempool_snapshots.py --interval 60 --duration 1

    # Poll indefinitely (Ctrl+C to stop)
    python scripts/scrape_mempool_snapshots.py --interval 60

    # Show collected stats
    python scripts/scrape_mempool_snapshots.py --stats
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone

import requests

# Defaults
MEMPOOL_API_BASE = "https://mempool.space/api"
OUTPUT_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mempool_space_data.db')
OUTPUT_DB = os.path.normpath(OUTPUT_DB)

TABLE_NAME = 'mempool_snapshots'

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER NOT NULL UNIQUE,
    -- /api/mempool
    mempool_count INTEGER,
    mempool_vsize INTEGER,
    mempool_total_fee INTEGER,
    fee_histogram TEXT,
    -- Derived percentiles from histogram
    mempool_min_feerate REAL,
    mempool_p50_feerate REAL,
    mempool_p90_feerate REAL,
    -- /api/v1/fees/recommended
    rec_fastest_fee REAL,
    rec_half_hour_fee REAL,
    rec_hour_fee REAL,
    rec_economy_fee REAL,
    rec_minimum_fee REAL,
    -- /api/v1/fees/mempool-blocks (first projected block)
    next_block_median_fee REAL,
    next_block_n_tx INTEGER,
    next_block_vsize INTEGER
);
"""


def init_db(db_path: str) -> sqlite3.Connection:
    """Create the snapshots table if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(CREATE_TABLE_SQL)
    conn.commit()
    return conn


def fetch_json(url: str, max_retries: int = 3, timeout: int = 15) -> dict | list | None:
    """GET a JSON endpoint with simple retry logic."""
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt == max_retries:
                print(f"  WARN: {url} failed after {max_retries} attempts: {e}")
                return None
            time.sleep(1 * attempt)  # linear backoff


def percentiles_from_histogram(fee_histogram: list) -> tuple:
    """
    Compute min, p50, p90 fee-rates from the mempool.space fee histogram.

    The histogram is a list of [feerate, vbytes] pairs,
    ordered from highest to lowest fee-rate bucket.
    """
    if not fee_histogram:
        return (None, None, None)

    # Each entry is [feerate, vbytes]
    pairs = []
    for entry in fee_histogram:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            pairs.append((float(entry[0]), float(entry[1])))

    if not pairs:
        return (None, None, None)

    # Sort ascending by fee-rate for percentile computation
    pairs.sort(key=lambda x: x[0])

    total_vbytes = sum(vb for _, vb in pairs)
    if total_vbytes == 0:
        return (None, None, None)

    min_fr = pairs[0][0]
    cumulative = 0
    p50_fr = None
    p90_fr = None

    for fr, vb in pairs:
        cumulative += vb
        frac = cumulative / total_vbytes
        if p50_fr is None and frac >= 0.50:
            p50_fr = fr
        if p90_fr is None and frac >= 0.90:
            p90_fr = fr

    return (min_fr, p50_fr, p90_fr)


def collect_snapshot() -> dict | None:
    """Collect one snapshot from the three API endpoints."""
    ts = int(time.time())

    # 1. /api/mempool
    mempool_data = fetch_json(f"{MEMPOOL_API_BASE}/mempool")
    if mempool_data is None:
        return None

    fee_histogram = mempool_data.get('fee_histogram', [])
    min_fr, p50_fr, p90_fr = percentiles_from_histogram(fee_histogram)

    # 2. /api/v1/fees/recommended
    rec = fetch_json(f"{MEMPOOL_API_BASE}/v1/fees/recommended") or {}

    # 3. /api/v1/fees/mempool-blocks
    mempool_blocks = fetch_json(f"{MEMPOOL_API_BASE}/v1/fees/mempool-blocks")
    first_block = {}
    if mempool_blocks and isinstance(mempool_blocks, list) and len(mempool_blocks) > 0:
        first_block = mempool_blocks[0]

    return {
        'timestamp': ts,
        'mempool_count': mempool_data.get('count'),
        'mempool_vsize': mempool_data.get('vsize'),
        'mempool_total_fee': mempool_data.get('total_fee'),
        'fee_histogram': json.dumps(fee_histogram),
        'mempool_min_feerate': min_fr,
        'mempool_p50_feerate': p50_fr,
        'mempool_p90_feerate': p90_fr,
        'rec_fastest_fee': rec.get('fastestFee'),
        'rec_half_hour_fee': rec.get('halfHourFee'),
        'rec_hour_fee': rec.get('hourFee'),
        'rec_economy_fee': rec.get('economyFee'),
        'rec_minimum_fee': rec.get('minimumFee'),
        'next_block_median_fee': first_block.get('medianFee'),
        'next_block_n_tx': first_block.get('nTx'),
        'next_block_vsize': first_block.get('blockVSize'),
    }


def insert_snapshot(conn: sqlite3.Connection, snapshot: dict) -> bool:
    """Insert a snapshot row, returning True on success."""
    cols = list(snapshot.keys())
    placeholders = ', '.join(['?'] * len(cols))
    col_names = ', '.join(cols)
    try:
        conn.execute(
            f"INSERT OR IGNORE INTO {TABLE_NAME} ({col_names}) VALUES ({placeholders})",
            [snapshot[c] for c in cols],
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # Duplicate timestamp
        return False


def print_stats(db_path: str) -> None:
    """Print summary of collected snapshots."""
    conn = sqlite3.connect(db_path)
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    if TABLE_NAME not in tables:
        print(f"Table '{TABLE_NAME}' does not exist in {db_path}")
        conn.close()
        return

    n = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
    if n == 0:
        print("No snapshots collected yet.")
        conn.close()
        return

    row = conn.execute(
        f"SELECT MIN(timestamp), MAX(timestamp) FROM {TABLE_NAME}"
    ).fetchone()
    ts_min, ts_max = row

    latest = conn.execute(
        f"SELECT mempool_count, mempool_vsize, mempool_p50_feerate, "
        f"rec_fastest_fee, next_block_median_fee "
        f"FROM {TABLE_NAME} ORDER BY timestamp DESC LIMIT 1"
    ).fetchone()

    conn.close()

    span = timedelta(seconds=ts_max - ts_min)
    print("\n" + "=" * 60)
    print(f"MEMPOOL SNAPSHOTS — {n:,} snapshots")
    print("=" * 60)
    print(f"Time range: {datetime.fromtimestamp(ts_min)} – "
          f"{datetime.fromtimestamp(ts_max)} UTC")
    print(f"Duration:   {span}")
    print(f"\nLatest snapshot:")
    print(f"  Mempool count:      {latest[0]:,}")
    print(f"  Mempool vsize:      {latest[1]:,} vB")
    print(f"  Mempool p50 fee:    {latest[2]} sat/vB")
    print(f"  Recommended fast:   {latest[3]} sat/vB")
    print(f"  Next block median:  {latest[4]} sat/vB")


def main():
    parser = argparse.ArgumentParser(
        description="Poll mempool.space API for live mempool snapshots"
    )
    parser.add_argument('--interval', type=int, default=60,
                        help="Seconds between snapshots (default: 60)")
    parser.add_argument('--duration', type=float, default=None,
                        help="Hours to run (default: indefinite, Ctrl+C to stop)")
    parser.add_argument('--output-db', type=str, default=OUTPUT_DB,
                        help=f"Output database path (default: {OUTPUT_DB})")
    parser.add_argument('--stats', action='store_true',
                        help="Print collected snapshot stats and exit")
    args = parser.parse_args()

    if args.stats:
        print_stats(args.output_db)
        return

    conn = init_db(args.output_db)

    end_time = None
    if args.duration is not None:
        end_time = time.time() + args.duration * 3600

    print(f"Scraping mempool snapshots every {args.interval}s")
    if args.duration:
        print(f"  Duration: {args.duration}h (until {datetime.fromtimestamp(end_time)} UTC)")
    else:
        print(f"  Duration: indefinite (Ctrl+C to stop)")
    print(f"  Output:   {args.output_db}")
    print()

    count = 0
    try:
        while True:
            if end_time and time.time() >= end_time:
                print(f"\nDuration reached. Collected {count} snapshots.")
                break

            snapshot = collect_snapshot()
            if snapshot and insert_snapshot(conn, snapshot):
                count += 1
                ts = datetime.fromtimestamp(snapshot['timestamp']).strftime('%H:%M:%S')
                mc = snapshot['mempool_count'] or 0
                p50 = snapshot['mempool_p50_feerate']
                p50_str = f"{p50:.1f}" if p50 is not None else "N/A"
                print(f"  [{ts}] #{count:>4d}  mempool={mc:>6,}  p50_fee={p50_str:>6} sat/vB")
            else:
                print(f"  [{datetime.now(timezone.utc).strftime('%H:%M:%S')}] snapshot failed or duplicate")

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print(f"\nStopped. Collected {count} snapshots total.")

    conn.close()
    print_stats(args.output_db)


if __name__ == '__main__':
    main()
