#!/usr/bin/env python3
"""
Fetch transaction weight/size/fee data from mempool.space API for cross-checking.

Usage:
    python fetch_weights_mempool_space.py --txids <txid1,txid2,...>
    python fetch_weights_mempool_space.py --from-db <db_path> --sample <n>
    python fetch_weights_mempool_space.py --from-file <txids.txt>
"""

import argparse
import requests
import sqlite3
import time
import sys
from typing import List, Dict, Any, Optional
import pandas as pd

# mempool.space API base URL (can also use self-hosted instance)
MEMPOOL_API_BASE = "https://mempool.space/api"

# Rate limiting - mempool.space allows ~10 req/s for public API
RATE_LIMIT_DELAY = 0.15  # seconds between requests


def fetch_transaction(txid: str, base_url: str = MEMPOOL_API_BASE) -> Dict[str, Any]:
    """
    Fetch transaction data from mempool.space API.
    
    Returns dict with: txid, weight, size, fee, fee_rate (sat/vB), status, etc.
    """
    url = f"{base_url}/tx/{txid}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        # Extract relevant fields
        return {
            "txid": txid,
            "weight": data.get("weight"),
            "size": data.get("size"),  # raw size in bytes
            "vsize": data.get("weight", 0) // 4 if data.get("weight") else None,  # virtual size
            "fee": data.get("fee"),  # absolute fee in satoshis
            "fee_rate": round(data.get("fee", 0) / (data.get("weight", 1) / 4), 4) if data.get("weight") else None,
            "version": data.get("version"),
            "locktime": data.get("locktime"),
            "vin_count": len(data.get("vin", [])),
            "vout_count": len(data.get("vout", [])),
            "confirmed": data.get("status", {}).get("confirmed", False),
            "block_height": data.get("status", {}).get("block_height"),
            "block_time": data.get("status", {}).get("block_time"),
            "error": None
        }
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return {"txid": txid, "error": "Transaction not found"}
        return {"txid": txid, "error": f"HTTP {e.response.status_code}"}
    except Exception as e:
        return {"txid": txid, "error": str(e)}


def fetch_transactions_batch(txids: List[str], base_url: str = MEMPOOL_API_BASE, 
                             verbose: bool = True) -> List[Dict[str, Any]]:
    """Fetch multiple transactions with rate limiting."""
    results = []
    total = len(txids)
    errors = 0
    
    for i, txid in enumerate(txids):
        result = fetch_transaction(txid, base_url)
        results.append(result)
        
        if result.get("error"):
            errors += 1
        
        if verbose and (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{total} (errors: {errors})", end="\r")
        
        # Rate limiting
        if i < total - 1:
            time.sleep(RATE_LIMIT_DELAY)
    
    if verbose:
        print(f"  Completed: {total}/{total} (errors: {errors})     ")
    
    return results


def get_sample_txids_from_db(db_path: str, n_samples: int = 100, 
                              random: bool = True, table: str = "mempool_transactions") -> List[str]:
    """Get transaction IDs from database."""
    conn = sqlite3.connect(db_path)
    
    if random:
        query = f"""
            SELECT tx_id FROM {table}
            WHERE tx_id IS NOT NULL
            ORDER BY RANDOM()
            LIMIT {n_samples}
        """
    else:
        query = f"""
            SELECT tx_id FROM {table}
            WHERE tx_id IS NOT NULL
            LIMIT {n_samples}
        """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df['tx_id'].tolist()


def get_db_data_for_txids(db_path: str, txids: List[str], 
                          table: str = "mempool_transactions") -> pd.DataFrame:
    """Get weight/size/fee data from database for comparison."""
    conn = sqlite3.connect(db_path)
    
    placeholders = ','.join(['?' for _ in txids])
    query = f"""
        SELECT tx_id, weight, size, fee_rate, absolute_fee
        FROM {table}
        WHERE tx_id IN ({placeholders})
    """
    
    df = pd.read_sql_query(query, conn, params=txids)
    conn.close()
    return df


def compare_data(api_df: pd.DataFrame, db_df: pd.DataFrame) -> pd.DataFrame:
    """Compare API data with database data."""
    # Merge on txid
    merged = api_df.merge(
        db_df,
        left_on='txid',
        right_on='tx_id',
        how='inner',
        suffixes=('_api', '_db')
    )
    
    if merged.empty:
        print("No matching transactions found")
        return merged
    
    # Calculate differences
    merged['weight_diff'] = merged['weight_api'] - merged['weight_db']
    merged['size_diff'] = merged['size_api'] - merged['size_db']
    merged['fee_rate_diff'] = merged['fee_rate_api'] - merged['fee_rate_db']
    
    # Percentage differences
    merged['weight_pct_diff'] = (merged['weight_diff'] / merged['weight_db'] * 100).round(4)
    merged['fee_rate_pct_diff'] = (merged['fee_rate_diff'] / merged['fee_rate_db'] * 100).round(4)
    
    return merged


def print_comparison_summary(merged: pd.DataFrame):
    """Print summary of comparison results."""
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY: mempool.space API vs Database")
    print("=" * 70)
    print(f"Matched transactions: {len(merged)}")
    
    # Weight comparison
    print("\n--- WEIGHT ---")
    print(f"  Exact matches:    {(merged['weight_diff'] == 0).sum()}")
    print(f"  Mismatches:       {(merged['weight_diff'] != 0).sum()}")
    if (merged['weight_diff'] != 0).any():
        print(f"  Mean difference:  {merged['weight_diff'].mean():.2f}")
        print(f"  Max difference:   {merged['weight_diff'].abs().max():.0f}")
        print(f"  Mean % diff:      {merged['weight_pct_diff'].mean():.4f}%")
    
    # Size comparison
    print("\n--- SIZE ---")
    print(f"  Exact matches:    {(merged['size_diff'] == 0).sum()}")
    print(f"  Mismatches:       {(merged['size_diff'] != 0).sum()}")
    if (merged['size_diff'] != 0).any():
        print(f"  Mean difference:  {merged['size_diff'].mean():.2f}")
        print(f"  Max difference:   {merged['size_diff'].abs().max():.0f}")
    
    # Fee rate comparison
    print("\n--- FEE RATE (sat/vB) ---")
    print(f"  Mean API:         {merged['fee_rate_api'].mean():.4f}")
    print(f"  Mean DB:          {merged['fee_rate_db'].mean():.4f}")
    print(f"  Mean difference:  {merged['fee_rate_diff'].mean():.4f}")
    print(f"  Max abs diff:     {merged['fee_rate_diff'].abs().max():.4f}")
    close_matches = (merged['fee_rate_diff'].abs() < 0.01).sum()
    print(f"  Close (<0.01):    {close_matches}")
    
    # Show sample mismatches
    weight_mismatches = merged[merged['weight_diff'] != 0]
    if not weight_mismatches.empty:
        print("\n--- SAMPLE WEIGHT MISMATCHES (up to 10) ---")
        cols = ['txid', 'weight_api', 'weight_db', 'weight_diff', 'weight_pct_diff']
        print(weight_mismatches[cols].head(10).to_string(index=False))
    
    fee_mismatches = merged[merged['fee_rate_diff'].abs() > 0.1]
    if not fee_mismatches.empty:
        print("\n--- SAMPLE FEE RATE MISMATCHES >0.1 sat/vB (up to 10) ---")
        cols = ['txid', 'fee_rate_api', 'fee_rate_db', 'fee_rate_diff']
        print(fee_mismatches[cols].head(10).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Fetch tx weights from mempool.space API for cross-checking"
    )
    
    # Input modes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--txids", type=str,
                       help="Comma-separated list of transaction IDs")
    group.add_argument("--from-db", type=str, metavar="DB_PATH",
                       help="Sample transactions from database")
    group.add_argument("--from-file", type=str,
                       help="Read txids from file (one per line)")
    
    # Options
    parser.add_argument("--sample", type=int, default=50,
                        help="Number of transactions to sample from DB (default: 50)")
    parser.add_argument("--table", type=str, default="mempool_transactions",
                        help="Database table name (default: mempool_transactions)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare API data with database (requires --from-db)")
    parser.add_argument("--output", "-o", type=str,
                        help="Save results to CSV file")
    parser.add_argument("--api-url", type=str, default=MEMPOOL_API_BASE,
                        help="mempool.space API base URL")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress progress output")
    
    args = parser.parse_args()
    
    # Get transaction IDs
    txids = []
    db_path = None
    
    if args.txids:
        txids = [t.strip() for t in args.txids.split(',') if t.strip()]
    elif args.from_db:
        db_path = args.from_db
        print(f"Sampling {args.sample} transactions from database ({args.table})...")
        txids = get_sample_txids_from_db(db_path, args.sample, table=args.table)
    elif args.from_file:
        with open(args.from_file, 'r') as f:
            txids = [line.strip() for line in f if line.strip()]
    
    if not txids:
        print("No transaction IDs provided")
        return
    
    print(f"Fetching {len(txids)} transactions from mempool.space API...")
    results = fetch_transactions_batch(txids, args.api_url, verbose=not args.quiet)
    
    # Convert to DataFrame
    api_df = pd.DataFrame(results)
    valid = api_df[api_df['error'].isna()].copy()
    errors = api_df[api_df['error'].notna()]
    
    print(f"\n✓ Successfully fetched: {len(valid)}/{len(txids)}")
    if len(errors) > 0:
        print(f"✗ Errors: {len(errors)}")
        print(f"  Sample errors: {errors['error'].value_counts().head(3).to_dict()}")
    
    # Display results
    if not valid.empty:
        print("\n" + "=" * 70)
        print("DATA FROM MEMPOOL.SPACE API")
        print("=" * 70)
        
        print(f"\nWeight statistics:")
        print(f"  Mean:   {valid['weight'].mean():.2f}")
        print(f"  Std:    {valid['weight'].std():.2f}")
        print(f"  Min:    {valid['weight'].min():.0f}")
        print(f"  Max:    {valid['weight'].max():.0f}")
        
        print(f"\nFee rate statistics (sat/vB):")
        print(f"  Mean:   {valid['fee_rate'].mean():.4f}")
        print(f"  Std:    {valid['fee_rate'].std():.4f}")
        print(f"  Min:    {valid['fee_rate'].min():.4f}")
        print(f"  Max:    {valid['fee_rate'].max():.4f}")
        
        print(f"\n--- Sample data (first 15) ---")
        display_cols = ['txid', 'weight', 'size', 'vsize', 'fee', 'fee_rate', 'confirmed']
        print(valid[display_cols].head(15).to_string(index=False))
    
    # Compare with database if requested
    if args.compare and db_path:
        print("\nFetching comparison data from database...")
        db_df = get_db_data_for_txids(db_path, txids, table=args.table)
        
        if not db_df.empty:
            merged = compare_data(valid, db_df)
            if not merged.empty:
                print_comparison_summary(merged)
                
                # Save comparison results
                if args.output:
                    merged.to_csv(args.output, index=False)
                    print(f"\n✓ Comparison saved to {args.output}")
        else:
            print("No matching data found in database")
    elif args.output:
        api_df.to_csv(args.output, index=False)
        print(f"\n✓ Saved to {args.output}")


if __name__ == "__main__":
    main()
