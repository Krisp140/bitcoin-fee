#!/usr/bin/env python3
"""
Async fetch of transaction data from mempool.space API by block.
Stores accurate weight/size/fee data in a local SQLite database.

Features:
- Async concurrent fetching for ~5-10x speedup
- Resumable - tracks progress and skips completed blocks
- Designed for overnight runs on all blocks

Usage:
    # Fetch ALL remaining blocks from your database (default)
    python fetch_mempool_space_blocks.py --source-db /path/to/source.db
    
    # Limit concurrent requests (default: 5)
    python fetch_mempool_space_blocks.py --source-db /path/to/source.db --concurrency 10
    
    # Check progress
    python fetch_mempool_space_blocks.py --stats
"""

import argparse
import asyncio
import aiohttp
import sqlite3
import time
import sys
import os
import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta
from dataclasses import dataclass

# Configuration
MEMPOOL_API_BASE = "https://mempool.space/api"
OUTPUT_DB = "mempool_space_data.db"
SOURCE_DB = "/home/armin/datalake/data-samples/11-24-2025-15m-data-lake.db"

# Rate limiting - mempool.space allows ~10 req/s
# With concurrency, we need to be careful
DEFAULT_CONCURRENCY = 2  # Concurrent block fetches
RATE_LIMIT_DELAY = 0.25  # Delay between requests within a block
REQUEST_TIMEOUT = 60
BATCH_SIZE = 25  # mempool.space returns 25 txs per request

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class FetchStats:
    """Track fetching statistics."""
    blocks_completed: int = 0
    blocks_failed: int = 0
    txs_fetched: int = 0
    start_time: float = 0
    
    def elapsed(self) -> float:
        return time.time() - self.start_time
    
    def rate(self) -> float:
        elapsed = self.elapsed()
        if elapsed > 0:
            return self.blocks_completed / elapsed * 3600  # blocks per hour
        return 0
    
    def eta(self, remaining: int) -> str:
        rate = self.rate()
        if rate > 0:
            hours = remaining / rate
            return str(timedelta(hours=hours)).split('.')[0]
        return "unknown"


def init_output_db(db_path: str) -> sqlite3.Connection:
    """Initialize the output database with proper schema."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            txid TEXT PRIMARY KEY,
            weight INTEGER,
            size INTEGER,
            vsize INTEGER,
            fee INTEGER,
            fee_rate REAL,
            version INTEGER,
            locktime INTEGER,
            vin_count INTEGER,
            vout_count INTEGER,
            block_hash TEXT,
            block_height INTEGER,
            block_time INTEGER,
            fetched_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS blocks_fetched (
            block_hash TEXT PRIMARY KEY,
            block_height INTEGER,
            tx_count INTEGER,
            fetched_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tx_block ON transactions(block_hash)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tx_fee_rate ON transactions(fee_rate)")
    
    conn.commit()
    return conn


def get_blocks_to_fetch(source_db: str, output_conn: sqlite3.Connection) -> List[str]:
    """Get list of ALL block hashes that need to be fetched."""
    source_conn = sqlite3.connect(source_db)
    source_blocks = set(row[0] for row in source_conn.execute(
        "SELECT DISTINCT conf_block_hash FROM mempool_transactions WHERE conf_block_hash IS NOT NULL"
    ).fetchall())
    source_conn.close()
    
    fetched = set(row[0] for row in output_conn.execute(
        "SELECT block_hash FROM blocks_fetched"
    ).fetchall())
    
    remaining = list(source_blocks - fetched)
    logger.info(f"Source blocks: {len(source_blocks):,}, Already fetched: {len(fetched):,}, Remaining: {len(remaining):,}")
    
    return remaining


async def fetch_block_info(session: aiohttp.ClientSession, block_hash: str) -> Optional[Dict]:
    """Fetch block metadata."""
    url = f"{MEMPOOL_API_BASE}/block/{block_hash}"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as resp:
            resp.raise_for_status()
            return await resp.json()
    except Exception as e:
        logger.warning(f"Error fetching block info {block_hash[:16]}...: {e}")
        return None


async def fetch_block_transactions(session: aiohttp.ClientSession, block_hash: str, 
                                    start_index: int = 0) -> List[Dict]:
    """Fetch a batch of transactions from a block."""
    url = f"{MEMPOOL_API_BASE}/block/{block_hash}/txs/{start_index}"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)) as resp:
            resp.raise_for_status()
            return await resp.json()
    except Exception as e:
        logger.warning(f"Error fetching txs from {block_hash[:12]}... at {start_index}: {e}")
        return []


async def fetch_all_block_transactions(session: aiohttp.ClientSession, block_hash: str, 
                                        tx_count: int) -> List[Dict]:
    """Fetch all transactions from a block with pagination."""
    all_txs = []
    fetched = 0
    retries = 0
    max_retries = 3
    
    while fetched < tx_count:
        txs = await fetch_block_transactions(session, block_hash, fetched)
        
        if not txs:
            retries += 1
            if retries >= max_retries:
                logger.error(f"Failed to fetch txs from {block_hash[:12]}... after {max_retries} retries")
                break
            await asyncio.sleep(1)  # Wait before retry
            continue
        
        retries = 0
        all_txs.extend(txs)
        fetched += len(txs)
        
        # Rate limiting within block
        await asyncio.sleep(RATE_LIMIT_DELAY)
    
    return all_txs


def process_transaction(tx: Dict, block_hash: str, block_height: int, block_time: int) -> Dict:
    """Extract relevant fields from transaction data."""
    weight = tx.get("weight", 0)
    fee = tx.get("fee", 0)
    vsize = weight // 4 if weight else 0
    
    return {
        "txid": tx.get("txid"),
        "weight": weight,
        "size": tx.get("size"),
        "vsize": vsize,
        "fee": fee,
        "fee_rate": round(fee / vsize, 6) if vsize else 0,
        "version": tx.get("version"),
        "locktime": tx.get("locktime"),
        "vin_count": len(tx.get("vin", [])),
        "vout_count": len(tx.get("vout", [])),
        "block_hash": block_hash,
        "block_height": block_height,
        "block_time": block_time
    }


def save_transactions(conn: sqlite3.Connection, transactions: List[Dict]):
    """Bulk insert transactions into database."""
    if not transactions:
        return
    
    conn.executemany("""
        INSERT OR REPLACE INTO transactions 
        (txid, weight, size, vsize, fee, fee_rate, version, locktime, 
         vin_count, vout_count, block_hash, block_height, block_time)
        VALUES (:txid, :weight, :size, :vsize, :fee, :fee_rate, :version, 
                :locktime, :vin_count, :vout_count, :block_hash, :block_height, :block_time)
    """, transactions)
    conn.commit()


def mark_block_complete(conn: sqlite3.Connection, block_hash: str, 
                        block_height: int, tx_count: int):
    """Mark a block as fully fetched."""
    conn.execute("""
        INSERT OR REPLACE INTO blocks_fetched (block_hash, block_height, tx_count)
        VALUES (?, ?, ?)
    """, (block_hash, block_height, tx_count))
    conn.commit()


async def fetch_single_block(session: aiohttp.ClientSession, block_hash: str,
                              output_conn: sqlite3.Connection, stats: FetchStats) -> bool:
    """Fetch a single block and all its transactions."""
    try:
        # Get block info
        block_info = await fetch_block_info(session, block_hash)
        if not block_info:
            stats.blocks_failed += 1
            return False
        
        block_height = block_info.get("height", 0)
        block_time = block_info.get("timestamp", 0)
        tx_count = block_info.get("tx_count", 0)
        
        # Fetch all transactions
        raw_txs = await fetch_all_block_transactions(session, block_hash, tx_count)
        
        if len(raw_txs) < tx_count * 0.9:  # Allow 10% tolerance
            logger.warning(f"Block {block_height}: only got {len(raw_txs)}/{tx_count} txs")
        
        # Process and save
        transactions = [
            process_transaction(tx, block_hash, block_height, block_time)
            for tx in raw_txs
        ]
        
        save_transactions(output_conn, transactions)
        mark_block_complete(output_conn, block_hash, block_height, tx_count)
        
        stats.blocks_completed += 1
        stats.txs_fetched += len(transactions)
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing block {block_hash[:16]}...: {e}")
        stats.blocks_failed += 1
        return False


async def fetch_blocks_async(block_hashes: List[str], output_conn: sqlite3.Connection,
                              concurrency: int = DEFAULT_CONCURRENCY) -> FetchStats:
    """Fetch all blocks with concurrent async requests."""
    stats = FetchStats(start_time=time.time())
    total_blocks = len(block_hashes)
    
    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(concurrency)
    
    async def fetch_with_semaphore(session: aiohttp.ClientSession, block_hash: str):
        async with semaphore:
            result = await fetch_single_block(session, block_hash, output_conn, stats)
            
            # Progress logging every 10 blocks
            if stats.blocks_completed % 10 == 0:
                remaining = total_blocks - stats.blocks_completed - stats.blocks_failed
                eta = stats.eta(remaining)
                logger.info(
                    f"Progress: {stats.blocks_completed}/{total_blocks} blocks "
                    f"({stats.txs_fetched:,} txs) | "
                    f"Rate: {stats.rate():.1f} blocks/hr | "
                    f"ETA: {eta}"
                )
            
            return result
    
    # Create session with connection pooling
    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Process all blocks
        tasks = [fetch_with_semaphore(session, bh) for bh in block_hashes]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    return stats


def print_db_stats(conn: sqlite3.Connection):
    """Print statistics about the output database."""
    stats = conn.execute("""
        SELECT 
            COUNT(*) as tx_count,
            COUNT(DISTINCT block_hash) as block_count,
            AVG(weight) as avg_weight,
            AVG(fee_rate) as avg_fee_rate,
            MIN(block_height) as min_height,
            MAX(block_height) as max_height
        FROM transactions
    """).fetchone()
    
    blocks_fetched = conn.execute("SELECT COUNT(*) FROM blocks_fetched").fetchone()[0]
    
    # Get source block count for progress
    try:
        source_conn = sqlite3.connect(SOURCE_DB)
        total_source = source_conn.execute(
            "SELECT COUNT(DISTINCT conf_block_hash) FROM mempool_transactions WHERE conf_block_hash IS NOT NULL"
        ).fetchone()[0]
        source_conn.close()
    except:
        total_source = None
    
    print("\n" + "=" * 60)
    print("OUTPUT DATABASE STATISTICS")
    print("=" * 60)
    print(f"Transactions:     {stats[0]:,}")
    print(f"Blocks fetched:   {blocks_fetched:,}" + (f" / {total_source:,}" if total_source else ""))
    if total_source:
        print(f"Progress:         {100*blocks_fetched/total_source:.1f}%")
    print(f"Avg weight:       {stats[2]:.2f}" if stats[2] else "Avg weight:       N/A")
    print(f"Avg fee rate:     {stats[3]:.4f} sat/vB" if stats[3] else "Avg fee rate:     N/A")
    print(f"Block range:      {stats[4]} - {stats[5]}" if stats[4] else "Block range:      N/A")


def estimate_time(n_blocks: int, concurrency: int) -> str:
    """Estimate time to fetch blocks."""
    # Rough estimate: ~2500 txs/block, 100 requests/block, with concurrency
    requests_per_block = 100  # avg
    seconds_per_block = (requests_per_block * RATE_LIMIT_DELAY) / concurrency
    total_seconds = n_blocks * seconds_per_block
    
    hours = total_seconds / 3600
    if hours < 1:
        return f"{total_seconds/60:.0f} minutes"
    return f"{hours:.1f} hours"


async def main_async():
    parser = argparse.ArgumentParser(
        description="Async fetch of transaction data from mempool.space"
    )
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--source-db", type=str, default=SOURCE_DB,
                       help=f"Source database (default: {SOURCE_DB})")
    group.add_argument("--stats", action="store_true",
                       help="Show statistics of output database")
    
    parser.add_argument("--output", "-o", type=str, default=OUTPUT_DB,
                        help=f"Output database path (default: {OUTPUT_DB})")
    parser.add_argument("--concurrency", "-c", type=int, default=DEFAULT_CONCURRENCY,
                        help=f"Number of concurrent block fetches (default: {DEFAULT_CONCURRENCY})")
    parser.add_argument("--limit", type=int,
                        help="Limit number of blocks to fetch (default: all remaining)")
    
    args = parser.parse_args()
    
    # Initialize output database
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", args.output)
    output_path = os.path.normpath(output_path)
    logger.info(f"Output database: {output_path}")
    output_conn = init_output_db(output_path)
    
    if args.stats:
        print_db_stats(output_conn)
        return
    
    # Get blocks to fetch
    if not os.path.exists(args.source_db):
        logger.error(f"Source database not found: {args.source_db}")
        return
    
    block_hashes = get_blocks_to_fetch(args.source_db, output_conn)
    
    if args.limit:
        block_hashes = block_hashes[:args.limit]
    
    if not block_hashes:
        logger.info("No blocks to fetch - all done!")
        print_db_stats(output_conn)
        return
    
    # Estimate time
    est_time = estimate_time(len(block_hashes), args.concurrency)
    logger.info(f"Blocks to fetch: {len(block_hashes):,}")
    logger.info(f"Concurrency: {args.concurrency}")
    logger.info(f"Estimated time: {est_time}")
    logger.info("-" * 40)
    
    # Fetch blocks
    try:
        stats = await fetch_blocks_async(block_hashes, output_conn, args.concurrency)
        
        logger.info("=" * 60)
        logger.info("FETCH COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Blocks completed: {stats.blocks_completed:,}")
        logger.info(f"Blocks failed:    {stats.blocks_failed:,}")
        logger.info(f"Transactions:     {stats.txs_fetched:,}")
        logger.info(f"Total time:       {timedelta(seconds=int(stats.elapsed()))}")
        logger.info(f"Average rate:     {stats.rate():.1f} blocks/hour")
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted! Progress has been saved - restart to resume")
    
    print_db_stats(output_conn)
    output_conn.close()


def main():
    """Entry point."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
