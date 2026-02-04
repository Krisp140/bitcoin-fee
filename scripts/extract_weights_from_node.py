#!/usr/bin/env python3
"""
Extract transaction weight values directly from Bitcoin node for cross-checking.

Usage:
    python extract_weights_from_node.py --txids <txid1,txid2,...>
    python extract_weights_from_node.py --block <block_hash_or_height>
    python extract_weights_from_node.py --recent <n_blocks>
    python extract_weights_from_node.py --compare-db <db_path> --sample <n_txs>
"""

import asyncio
import argparse
import os
import sqlite3
from typing import Optional, List, Dict, Any
import pandas as pd

# Try to import the Bitcoin RPC library
try:
    from bitcoinrpc.authproxy import AuthServiceProxy
    USE_SYNC_RPC = True
except ImportError:
    USE_SYNC_RPC = False

try:
    from bitcoinrpc import BitcoinRPC
    USE_ASYNC_RPC = True
except ImportError:
    USE_ASYNC_RPC = False


# Default RPC configuration - modify these or use environment variables
DEFAULT_RPC_USER = os.getenv("BITCOIN_RPC_USER", "bitcoin")
DEFAULT_RPC_PASSWORD = os.getenv("BITCOIN_RPC_PASSWORD", "")
DEFAULT_RPC_HOST = os.getenv("BITCOIN_RPC_HOST", "127.0.0.1")
DEFAULT_RPC_PORT = os.getenv("BITCOIN_RPC_PORT", "8332")


async def connect_to_rpc_async(rpc_user: str, rpc_password: str, rpc_host: str, rpc_port: str):
    """Connect to Bitcoin node using async RPC."""
    host = f"http://{rpc_host}:{rpc_port}"
    try:
        rpc = BitcoinRPC.from_config(host, (rpc_user, rpc_password), timeout=30)
        # Test the connection
        test_result = await rpc.getblockchaininfo()
        print(f"✓ Connected to bitcoind. Block height: {test_result['blocks']}")
        return rpc
    except Exception as e:
        print(f"Failed to connect to bitcoind: {e}")
        print(f"Connection details: {host}")
        print(f"Username: {rpc_user}")
        raise


def connect_to_rpc_sync(rpc_user: str, rpc_password: str, rpc_host: str, rpc_port: str):
    """Connect to Bitcoin node using sync RPC (python-bitcoinrpc)."""
    url = f"http://{rpc_user}:{rpc_password}@{rpc_host}:{rpc_port}"
    try:
        rpc = AuthServiceProxy(url, timeout=30)
        # Test the connection
        info = rpc.getblockchaininfo()
        print(f"✓ Connected to bitcoind. Block height: {info['blocks']}")
        return rpc
    except Exception as e:
        print(f"Failed to connect to bitcoind: {e}")
        raise


async def get_transaction_weight_async(rpc, txid: str) -> Dict[str, Any]:
    """Get transaction weight from node using async RPC."""
    try:
        # Get raw transaction with verbose output
        tx = await rpc.getrawtransaction(txid, True)
        return {
            "txid": txid,
            "weight": tx.get("weight"),
            "vsize": tx.get("vsize"),
            "size": tx.get("size"),
            "version": tx.get("version"),
            "locktime": tx.get("locktime"),
            "vin_count": len(tx.get("vin", [])),
            "vout_count": len(tx.get("vout", [])),
            "fee": tx.get("fee"),  # May be None for confirmed txs without -txindex
            "error": None
        }
    except Exception as e:
        return {
            "txid": txid,
            "weight": None,
            "vsize": None,
            "size": None,
            "error": str(e)
        }


def get_transaction_weight_sync(rpc, txid: str) -> Dict[str, Any]:
    """Get transaction weight from node using sync RPC."""
    try:
        # Get raw transaction with verbose output
        tx = rpc.getrawtransaction(txid, True)
        return {
            "txid": txid,
            "weight": tx.get("weight"),
            "vsize": tx.get("vsize"),
            "size": tx.get("size"),
            "version": tx.get("version"),
            "locktime": tx.get("locktime"),
            "vin_count": len(tx.get("vin", [])),
            "vout_count": len(tx.get("vout", [])),
            "fee": tx.get("fee"),
            "error": None
        }
    except Exception as e:
        return {
            "txid": txid,
            "weight": None,
            "vsize": None,
            "size": None,
            "error": str(e)
        }


async def get_block_transactions_async(rpc, block_id: str) -> List[Dict[str, Any]]:
    """Get all transaction weights from a block using async RPC."""
    try:
        # Check if block_id is a height or hash
        if block_id.isdigit():
            block_hash = await rpc.getblockhash(int(block_id))
        else:
            block_hash = block_id
        
        # Get block with full transaction details (verbosity=2)
        block = await rpc.getblock(block_hash, 2)
        
        results = []
        for tx in block.get("tx", []):
            results.append({
                "txid": tx.get("txid"),
                "weight": tx.get("weight"),
                "vsize": tx.get("vsize"),
                "size": tx.get("size"),
                "block_hash": block_hash,
                "block_height": block.get("height"),
                "error": None
            })
        
        print(f"✓ Retrieved {len(results)} transactions from block {block.get('height')}")
        return results
    except Exception as e:
        print(f"Error getting block {block_id}: {e}")
        return []


def get_block_transactions_sync(rpc, block_id: str) -> List[Dict[str, Any]]:
    """Get all transaction weights from a block using sync RPC."""
    try:
        # Check if block_id is a height or hash
        if block_id.isdigit():
            block_hash = rpc.getblockhash(int(block_id))
        else:
            block_hash = block_id
        
        # Get block with full transaction details (verbosity=2)
        block = rpc.getblock(block_hash, 2)
        
        results = []
        for tx in block.get("tx", []):
            results.append({
                "txid": tx.get("txid"),
                "weight": tx.get("weight"),
                "vsize": tx.get("vsize"),
                "size": tx.get("size"),
                "block_hash": block_hash,
                "block_height": block.get("height"),
                "error": None
            })
        
        print(f"✓ Retrieved {len(results)} transactions from block {block.get('height')}")
        return results
    except Exception as e:
        print(f"Error getting block {block_id}: {e}")
        return []


async def get_recent_blocks_async(rpc, n_blocks: int = 5) -> List[Dict[str, Any]]:
    """Get transaction weights from recent blocks using async RPC."""
    info = await rpc.getblockchaininfo()
    current_height = info['blocks']
    
    all_results = []
    for height in range(current_height, current_height - n_blocks, -1):
        results = await get_block_transactions_async(rpc, str(height))
        all_results.extend(results)
    
    return all_results


def get_recent_blocks_sync(rpc, n_blocks: int = 5) -> List[Dict[str, Any]]:
    """Get transaction weights from recent blocks using sync RPC."""
    info = rpc.getblockchaininfo()
    current_height = info['blocks']
    
    all_results = []
    for height in range(current_height, current_height - n_blocks, -1):
        results = get_block_transactions_sync(rpc, str(height))
        all_results.extend(results)
    
    return all_results


def compare_with_database(db_path: str, node_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Compare node weights with database weights."""
    if not node_data:
        print("No node data to compare")
        return pd.DataFrame()
    
    node_df = pd.DataFrame(node_data)
    node_df = node_df[node_df['weight'].notna()].copy()
    
    if node_df.empty:
        print("No valid weight data from node")
        return pd.DataFrame()
    
    # Load database weights
    txids = node_df['txid'].tolist()
    placeholders = ','.join(['?' for _ in txids])
    
    conn = sqlite3.connect(db_path)
    query = f"""
        SELECT tx_id, weight, size, fee_rate
        FROM transactions
        WHERE tx_id IN ({placeholders})
    """
    db_df = pd.read_sql_query(query, conn, params=txids)
    conn.close()
    
    if db_df.empty:
        print("No matching transactions found in database")
        return node_df
    
    # Merge and compare
    merged = node_df.merge(
        db_df, 
        left_on='txid', 
        right_on='tx_id', 
        how='inner',
        suffixes=('_node', '_db')
    )
    
    if merged.empty:
        print("No overlapping transactions between node and database")
        return node_df
    
    # Calculate differences
    merged['weight_diff'] = merged['weight_node'] - merged['weight_db']
    merged['weight_pct_diff'] = (merged['weight_diff'] / merged['weight_db'] * 100).round(4)
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("WEIGHT COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Matched transactions: {len(merged)}")
    print(f"\nWeight differences:")
    print(f"  Mean diff: {merged['weight_diff'].mean():.2f}")
    print(f"  Std diff:  {merged['weight_diff'].std():.2f}")
    print(f"  Min diff:  {merged['weight_diff'].min():.2f}")
    print(f"  Max diff:  {merged['weight_diff'].max():.2f}")
    print(f"  Exact matches: {(merged['weight_diff'] == 0).sum()}")
    print(f"  Mismatches: {(merged['weight_diff'] != 0).sum()}")
    
    # Show mismatches if any
    mismatches = merged[merged['weight_diff'] != 0]
    if not mismatches.empty:
        print(f"\n--- Sample mismatches (up to 10) ---")
        display_cols = ['txid', 'weight_node', 'weight_db', 'weight_diff', 'weight_pct_diff']
        print(mismatches[display_cols].head(10).to_string(index=False))
    
    return merged


def get_sample_txids_from_db(db_path: str, n_samples: int = 100) -> List[str]:
    """Get a random sample of transaction IDs from the database."""
    conn = sqlite3.connect(db_path)
    query = f"""
        SELECT tx_id FROM transactions
        ORDER BY RANDOM()
        LIMIT {n_samples}
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df['tx_id'].tolist()


async def main_async(args):
    """Main async entry point."""
    rpc = await connect_to_rpc_async(
        args.rpc_user, args.rpc_password, args.rpc_host, args.rpc_port
    )
    
    results = []
    
    if args.txids:
        txids = [t.strip() for t in args.txids.split(',')]
        print(f"\nFetching weight for {len(txids)} transaction(s)...")
        for txid in txids:
            result = await get_transaction_weight_async(rpc, txid)
            results.append(result)
            
    elif args.block:
        print(f"\nFetching transactions from block {args.block}...")
        results = await get_block_transactions_async(rpc, args.block)
        
    elif args.recent:
        print(f"\nFetching transactions from {args.recent} recent block(s)...")
        results = await get_recent_blocks_async(rpc, args.recent)
        
    elif args.compare_db:
        if not os.path.exists(args.compare_db):
            print(f"Database not found: {args.compare_db}")
            return
        
        # Get sample txids from database
        sample_size = args.sample or 100
        print(f"\nSampling {sample_size} transactions from database...")
        txids = get_sample_txids_from_db(args.compare_db, sample_size)
        
        print(f"Fetching weights from node...")
        for txid in txids:
            result = await get_transaction_weight_async(rpc, txid)
            results.append(result)
            if len(results) % 50 == 0:
                print(f"  Processed {len(results)}/{len(txids)}")
        
        # Compare with database
        compare_with_database(args.compare_db, results)
        return
    
    # Display results
    if results:
        df = pd.DataFrame(results)
        print("\n" + "=" * 60)
        print("TRANSACTION WEIGHT DATA FROM NODE")
        print("=" * 60)
        
        # Show summary
        valid = df[df['weight'].notna()]
        print(f"Valid transactions: {len(valid)}/{len(df)}")
        
        if not valid.empty:
            print(f"\nWeight statistics:")
            print(f"  Mean:   {valid['weight'].mean():.2f}")
            print(f"  Std:    {valid['weight'].std():.2f}")
            print(f"  Min:    {valid['weight'].min():.0f}")
            print(f"  Max:    {valid['weight'].max():.0f}")
            print(f"  Median: {valid['weight'].median():.0f}")
            
            # Verify weight = 4 * vsize for SegWit transactions
            if 'vsize' in valid.columns:
                valid['expected_weight'] = valid['vsize'] * 4
                valid['weight_vs_vsize'] = valid['weight'] - valid['expected_weight']
                print(f"\nWeight vs 4*vsize check:")
                print(f"  All match: {(valid['weight_vs_vsize'] == 0).all()}")
                if (valid['weight_vs_vsize'] != 0).any():
                    print(f"  Mismatches: {(valid['weight_vs_vsize'] != 0).sum()}")
            
            print(f"\n--- Sample data (first 20) ---")
            display_cols = ['txid', 'weight', 'vsize', 'size']
            display_cols = [c for c in display_cols if c in df.columns]
            print(valid[display_cols].head(20).to_string(index=False))
        
        # Save to CSV if output specified
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"\n✓ Saved to {args.output}")
        
        return df


def main_sync(args):
    """Main sync entry point."""
    rpc = connect_to_rpc_sync(
        args.rpc_user, args.rpc_password, args.rpc_host, args.rpc_port
    )
    
    results = []
    
    if args.txids:
        txids = [t.strip() for t in args.txids.split(',')]
        print(f"\nFetching weight for {len(txids)} transaction(s)...")
        for txid in txids:
            result = get_transaction_weight_sync(rpc, txid)
            results.append(result)
            
    elif args.block:
        print(f"\nFetching transactions from block {args.block}...")
        results = get_block_transactions_sync(rpc, args.block)
        
    elif args.recent:
        print(f"\nFetching transactions from {args.recent} recent block(s)...")
        results = get_recent_blocks_sync(rpc, args.recent)
        
    elif args.compare_db:
        if not os.path.exists(args.compare_db):
            print(f"Database not found: {args.compare_db}")
            return
        
        # Get sample txids from database
        sample_size = args.sample or 100
        print(f"\nSampling {sample_size} transactions from database...")
        txids = get_sample_txids_from_db(args.compare_db, sample_size)
        
        print(f"Fetching weights from node...")
        for txid in txids:
            result = get_transaction_weight_sync(rpc, txid)
            results.append(result)
            if len(results) % 50 == 0:
                print(f"  Processed {len(results)}/{len(txids)}")
        
        # Compare with database
        compare_with_database(args.compare_db, results)
        return
    
    # Display results
    if results:
        df = pd.DataFrame(results)
        print("\n" + "=" * 60)
        print("TRANSACTION WEIGHT DATA FROM NODE")
        print("=" * 60)
        
        valid = df[df['weight'].notna()]
        print(f"Valid transactions: {len(valid)}/{len(df)}")
        
        if not valid.empty:
            print(f"\nWeight statistics:")
            print(f"  Mean:   {valid['weight'].mean():.2f}")
            print(f"  Std:    {valid['weight'].std():.2f}")
            print(f"  Min:    {valid['weight'].min():.0f}")
            print(f"  Max:    {valid['weight'].max():.0f}")
            print(f"  Median: {valid['weight'].median():.0f}")
            
            print(f"\n--- Sample data (first 20) ---")
            display_cols = ['txid', 'weight', 'vsize', 'size']
            display_cols = [c for c in display_cols if c in df.columns]
            print(valid[display_cols].head(20).to_string(index=False))
        
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"\n✓ Saved to {args.output}")
        
        return df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract transaction weights from Bitcoin node for cross-checking"
    )
    
    # RPC connection settings
    parser.add_argument("--rpc-user", default=DEFAULT_RPC_USER,
                        help="Bitcoin RPC username")
    parser.add_argument("--rpc-password", default=DEFAULT_RPC_PASSWORD,
                        help="Bitcoin RPC password")
    parser.add_argument("--rpc-host", default=DEFAULT_RPC_HOST,
                        help="Bitcoin RPC host")
    parser.add_argument("--rpc-port", default=DEFAULT_RPC_PORT,
                        help="Bitcoin RPC port")
    
    # Query modes (mutually exclusive)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--txids", type=str,
                       help="Comma-separated list of transaction IDs")
    group.add_argument("--block", type=str,
                       help="Block hash or height to fetch")
    group.add_argument("--recent", type=int, metavar="N",
                       help="Fetch transactions from N most recent blocks")
    group.add_argument("--compare-db", type=str, metavar="DB_PATH",
                       help="Compare node weights with database")
    
    # Additional options
    parser.add_argument("--sample", type=int, default=100,
                        help="Number of transactions to sample when comparing (default: 100)")
    parser.add_argument("--output", "-o", type=str,
                        help="Save results to CSV file")
    parser.add_argument("--sync", action="store_true",
                        help="Use sync RPC (python-bitcoinrpc) instead of async")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Determine which RPC library to use
    use_sync = args.sync or not USE_ASYNC_RPC
    
    if use_sync and not USE_SYNC_RPC:
        print("Error: No RPC library available.")
        print("Install one of:")
        print("  pip install python-bitcoinrpc  (for sync)")
        print("  pip install bitcoinrpc         (for async)")
        exit(1)
    
    if use_sync:
        print("Using sync RPC (python-bitcoinrpc)")
        main_sync(args)
    else:
        print("Using async RPC (bitcoinrpc)")
        asyncio.run(main_async(args))
