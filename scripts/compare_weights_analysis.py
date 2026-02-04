#!/usr/bin/env python3
"""
Extensive comparison of weight/size/fee data between source database and mempool.space data.
Read-only analysis - does not modify any databases.

Usage:
    python compare_weights_analysis.py
    python compare_weights_analysis.py --output comparison_report.csv
    python compare_weights_analysis.py --plot
"""

import argparse
import sqlite3
import os
import sys
import pandas as pd
import numpy as np
from typing import Tuple

# Paths
MEMPOOL_SPACE_DB = "/home/kristian/notebooks/mempool_space_data.db"
SOURCE_DB = "/home/armin/datalake/data-samples/11-24-2025-15m-data-lake.db"


def load_comparison_data(source_db: str, mempool_db: str, 
                          sample_size: int = None) -> pd.DataFrame:
    """Load and join data from both databases for comparison."""
    
    print("Loading data from both databases...")
    
    mempool_conn = sqlite3.connect(mempool_db)
    mempool_conn.execute(f"ATTACH DATABASE '{source_db}' AS source")
    
    query = """
        SELECT 
            s.tx_id,
            s.weight as weight_db,
            s.size as size_db,
            s.fee_rate as fee_rate_db,
            s.absolute_fee as fee_db,
            s.waittime,
            s.mempool_tx_count,
            s.min_respend_blocks,
            s.conf_block_hash,
            m.weight as weight_api,
            m.size as size_api,
            m.vsize as vsize_api,
            m.fee as fee_api,
            m.fee_rate as fee_rate_api,
            m.block_height
        FROM source.mempool_transactions s
        INNER JOIN transactions m ON s.tx_id = m.txid
    """
    
    if sample_size:
        query += f" ORDER BY RANDOM() LIMIT {sample_size}"
    
    df = pd.read_sql_query(query, mempool_conn)
    mempool_conn.execute("DETACH DATABASE source")
    mempool_conn.close()
    
    print(f"Loaded {len(df):,} matched transactions")
    return df


def compute_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived comparison metrics."""
    
    # Weight differences
    df['weight_diff'] = df['weight_api'] - df['weight_db']
    df['weight_pct_diff'] = (df['weight_diff'] / df['weight_db'] * 100)
    df['weight_ratio'] = df['weight_api'] / df['weight_db']
    
    # Size differences
    df['size_diff'] = df['size_api'] - df['size_db']
    df['size_pct_diff'] = (df['size_diff'] / df['size_db'] * 100)
    
    # Fee rate differences
    df['fee_rate_diff'] = df['fee_rate_api'] - df['fee_rate_db']
    df['fee_rate_pct_diff'] = (df['fee_rate_diff'] / df['fee_rate_db'].replace(0, np.nan) * 100)
    
    # What the fee rate SHOULD be with correct vsize
    # fee_rate = fee / vsize
    df['fee_rate_corrected'] = df['fee_api'] / df['vsize_api']
    
    # Check if DB is using size*4 as weight (the suspected bug)
    df['weight_db_from_size'] = df['size_db'] * 4
    df['weight_matches_size_x4'] = (df['weight_db'] == df['weight_db_from_size'])
    
    # Witness data estimation
    # For SegWit: weight = base_size*4 + witness_size = base_size*3 + total_size
    # If DB has base_size and API has total_size:
    # witness_size = size_api - size_db
    df['estimated_witness_size'] = df['size_api'] - df['size_db']
    df['is_segwit'] = df['estimated_witness_size'] > 0
    
    # Categorize transactions
    df['weight_match'] = df['weight_diff'] == 0
    df['is_legacy'] = ~df['is_segwit']
    
    return df


def print_summary_statistics(df: pd.DataFrame):
    """Print comprehensive summary statistics."""
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE WEIGHT/SIZE COMPARISON ANALYSIS")
    print("=" * 80)
    
    total = len(df)
    
    # Overall match statistics
    print(f"\n{'='*40}")
    print("1. OVERALL MATCH STATISTICS")
    print(f"{'='*40}")
    
    weight_exact = (df['weight_diff'] == 0).sum()
    weight_mismatch = (df['weight_diff'] != 0).sum()
    
    print(f"Total transactions compared:  {total:,}")
    print(f"Weight exact matches:         {weight_exact:,} ({100*weight_exact/total:.2f}%)")
    print(f"Weight mismatches:            {weight_mismatch:,} ({100*weight_mismatch/total:.2f}%)")
    
    # SegWit vs Legacy breakdown
    print(f"\n{'='*40}")
    print("2. SEGWIT vs LEGACY BREAKDOWN")
    print(f"{'='*40}")
    
    segwit = df['is_segwit'].sum()
    legacy = (~df['is_segwit']).sum()
    
    print(f"SegWit transactions:          {segwit:,} ({100*segwit/total:.2f}%)")
    print(f"Legacy transactions:          {legacy:,} ({100*legacy/total:.2f}%)")
    
    # Check match rates by type
    segwit_matches = (df[df['is_segwit']]['weight_diff'] == 0).sum()
    legacy_matches = (df[~df['is_segwit']]['weight_diff'] == 0).sum()
    
    print(f"\nSegWit weight matches:        {segwit_matches:,} ({100*segwit_matches/segwit:.2f}%)" if segwit > 0 else "")
    print(f"Legacy weight matches:        {legacy_matches:,} ({100*legacy_matches/legacy:.2f}%)" if legacy > 0 else "")
    
    # Bug hypothesis check
    print(f"\n{'='*40}")
    print("3. BUG HYPOTHESIS: weight_db = size_db * 4")
    print(f"{'='*40}")
    
    matches_size_x4 = df['weight_matches_size_x4'].sum()
    print(f"Transactions where weight_db = size_db * 4:")
    print(f"  Count: {matches_size_x4:,} ({100*matches_size_x4/total:.2f}%)")
    
    # Among mismatches, how many follow this pattern?
    mismatches = df[df['weight_diff'] != 0]
    if len(mismatches) > 0:
        mismatch_follows_pattern = mismatches['weight_matches_size_x4'].sum()
        print(f"\nAmong weight mismatches:")
        print(f"  Following size*4 pattern: {mismatch_follows_pattern:,} ({100*mismatch_follows_pattern/len(mismatches):.2f}%)")
    
    # Weight difference statistics
    print(f"\n{'='*40}")
    print("4. WEIGHT DIFFERENCE STATISTICS")
    print(f"{'='*40}")
    
    mm = df[df['weight_diff'] != 0]
    if len(mm) > 0:
        print(f"\nAmong {len(mm):,} mismatched transactions:")
        print(f"  Mean difference:    {mm['weight_diff'].mean():+.2f} WU")
        print(f"  Median difference:  {mm['weight_diff'].median():+.2f} WU")
        print(f"  Std deviation:      {mm['weight_diff'].std():.2f} WU")
        print(f"  Min difference:     {mm['weight_diff'].min():.0f} WU")
        print(f"  Max difference:     {mm['weight_diff'].max():.0f} WU")
        
        print(f"\n  Percentage differences:")
        print(f"    Mean % diff:      {mm['weight_pct_diff'].mean():+.2f}%")
        print(f"    Median % diff:    {mm['weight_pct_diff'].median():+.2f}%")
        
        print(f"\n  Weight ratio (API/DB):")
        print(f"    Mean ratio:       {mm['weight_ratio'].mean():.4f}")
        print(f"    Median ratio:     {mm['weight_ratio'].median():.4f}")
    
    # Distribution of percentage differences
    print(f"\n{'='*40}")
    print("5. MISMATCH SEVERITY DISTRIBUTION")
    print(f"{'='*40}")
    
    if len(mm) > 0:
        bins = [
            (0, 5, "0-5%"),
            (5, 10, "5-10%"),
            (10, 20, "10-20%"),
            (20, 30, "20-30%"),
            (30, 50, "30-50%"),
            (50, 100, "50-100%"),
            (100, float('inf'), ">100%")
        ]
        
        print(f"\n  {'Range':<15} {'Count':>10} {'Percentage':>12}")
        print(f"  {'-'*37}")
        for low, high, label in bins:
            count = ((mm['weight_pct_diff'].abs() >= low) & (mm['weight_pct_diff'].abs() < high)).sum()
            pct = 100 * count / len(mm)
            print(f"  {label:<15} {count:>10,} {pct:>11.2f}%")
    
    # Fee rate impact
    print(f"\n{'='*40}")
    print("6. FEE RATE IMPACT ANALYSIS")
    print(f"{'='*40}")
    
    print(f"\n  All transactions:")
    print(f"    Mean fee_rate (DB):        {df['fee_rate_db'].mean():.4f} sat/vB")
    print(f"    Mean fee_rate (API):       {df['fee_rate_api'].mean():.4f} sat/vB")
    print(f"    Mean difference:           {df['fee_rate_diff'].mean():+.4f} sat/vB")
    
    if len(mm) > 0:
        print(f"\n  Among mismatched transactions:")
        print(f"    Mean fee_rate (DB):        {mm['fee_rate_db'].mean():.4f} sat/vB")
        print(f"    Mean fee_rate (API):       {mm['fee_rate_api'].mean():.4f} sat/vB")
        print(f"    Mean difference:           {mm['fee_rate_diff'].mean():+.4f} sat/vB")
    
    # Fee rate difference distribution
    print(f"\n  Fee rate differences (all):")
    print(f"    Mean:    {df['fee_rate_diff'].mean():+.4f} sat/vB")
    print(f"    Median:  {df['fee_rate_diff'].median():+.4f} sat/vB")
    print(f"    Std:     {df['fee_rate_diff'].std():.4f} sat/vB")
    
    # Witness size analysis
    print(f"\n{'='*40}")
    print("7. WITNESS DATA ANALYSIS (SegWit only)")
    print(f"{'='*40}")
    
    segwit_df = df[df['is_segwit']]
    if len(segwit_df) > 0:
        print(f"\n  Estimated witness sizes:")
        print(f"    Mean:    {segwit_df['estimated_witness_size'].mean():.2f} bytes")
        print(f"    Median:  {segwit_df['estimated_witness_size'].median():.2f} bytes")
        print(f"    Min:     {segwit_df['estimated_witness_size'].min():.0f} bytes")
        print(f"    Max:     {segwit_df['estimated_witness_size'].max():.0f} bytes")
        
        # Witness as percentage of total size
        segwit_df = segwit_df.copy()
        segwit_df['witness_pct'] = segwit_df['estimated_witness_size'] / segwit_df['size_api'] * 100
        print(f"\n  Witness as % of total size:")
        print(f"    Mean:    {segwit_df['witness_pct'].mean():.2f}%")
        print(f"    Median:  {segwit_df['witness_pct'].median():.2f}%")
    
    # Sample transactions
    print(f"\n{'='*40}")
    print("8. SAMPLE MISMATCHED TRANSACTIONS")
    print(f"{'='*40}")
    
    if len(mm) > 0:
        print(f"\n  10 largest weight differences (absolute):")
        sample = mm.nlargest(10, 'weight_diff')[
            ['tx_id', 'weight_db', 'weight_api', 'weight_diff', 'size_db', 'size_api']
        ].copy()
        sample.columns = ['txid', 'wt_db', 'wt_api', 'wt_diff', 'sz_db', 'sz_api']
        sample['txid'] = sample['txid'].str[:16] + '...'
        print(sample.to_string(index=False))
        
        print(f"\n  10 largest percentage differences:")
        sample = mm.nlargest(10, 'weight_pct_diff')[
            ['tx_id', 'weight_db', 'weight_api', 'weight_pct_diff', 'is_segwit']
        ].copy()
        sample.columns = ['txid', 'wt_db', 'wt_api', 'pct_diff', 'segwit']
        sample['txid'] = sample['txid'].str[:16] + '...'
        print(sample.to_string(index=False))


def create_plots(df: pd.DataFrame, output_dir: str = "plots"):
    """Create visualization plots."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not available for plotting")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Weight Comparison Analysis: Database vs mempool.space API', fontsize=14)
    
    mm = df[df['weight_diff'] != 0]
    
    # 1. Weight difference distribution
    ax = axes[0, 0]
    if len(mm) > 0:
        ax.hist(mm['weight_diff'].clip(-500, 500), bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Weight Difference (API - DB)')
    ax.set_ylabel('Count')
    ax.set_title('Weight Difference Distribution')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # 2. Percentage difference distribution
    ax = axes[0, 1]
    if len(mm) > 0:
        ax.hist(mm['weight_pct_diff'].clip(-100, 100), bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Weight % Difference')
    ax.set_ylabel('Count')
    ax.set_title('Weight Percentage Difference Distribution')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # 3. DB weight vs API weight scatter
    ax = axes[0, 2]
    sample = df.sample(min(5000, len(df)))
    ax.scatter(sample['weight_db'], sample['weight_api'], alpha=0.3, s=5)
    max_w = max(sample['weight_db'].max(), sample['weight_api'].max())
    ax.plot([0, max_w], [0, max_w], 'r--', label='y=x')
    ax.set_xlabel('Weight (Database)')
    ax.set_ylabel('Weight (API)')
    ax.set_title('Weight: DB vs API')
    ax.legend()
    
    # 4. Fee rate comparison
    ax = axes[1, 0]
    ax.scatter(sample['fee_rate_db'], sample['fee_rate_api'], alpha=0.3, s=5)
    max_fr = min(sample['fee_rate_db'].quantile(0.99), sample['fee_rate_api'].quantile(0.99))
    ax.plot([0, max_fr], [0, max_fr], 'r--', label='y=x')
    ax.set_xlabel('Fee Rate (Database) sat/vB')
    ax.set_ylabel('Fee Rate (API) sat/vB')
    ax.set_title('Fee Rate: DB vs API')
    ax.set_xlim(0, max_fr)
    ax.set_ylim(0, max_fr)
    ax.legend()
    
    # 5. Weight ratio by SegWit status
    ax = axes[1, 1]
    if len(mm) > 0:
        segwit_ratios = mm[mm['is_segwit']]['weight_ratio']
        legacy_ratios = mm[~mm['is_segwit']]['weight_ratio']
        data_to_plot = []
        labels = []
        if len(segwit_ratios) > 0:
            data_to_plot.append(segwit_ratios.clip(0.5, 2))
            labels.append(f'SegWit\n(n={len(segwit_ratios):,})')
        if len(legacy_ratios) > 0:
            data_to_plot.append(legacy_ratios.clip(0.5, 2))
            labels.append(f'Legacy\n(n={len(legacy_ratios):,})')
        if data_to_plot:
            ax.boxplot(data_to_plot, labels=labels)
    ax.set_ylabel('Weight Ratio (API/DB)')
    ax.set_title('Weight Ratio by Transaction Type')
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.7)
    
    # 6. Estimated witness size distribution
    ax = axes[1, 2]
    segwit_df = df[df['is_segwit']]
    if len(segwit_df) > 0:
        ax.hist(segwit_df['estimated_witness_size'].clip(0, 500), bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Estimated Witness Size (bytes)')
    ax.set_ylabel('Count')
    ax.set_title('Witness Size Distribution (SegWit)')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'weight_comparison_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Extensive comparison of weight data (read-only)"
    )
    
    parser.add_argument("--mempool-db", type=str, default=MEMPOOL_SPACE_DB,
                        help=f"mempool.space database (default: {MEMPOOL_SPACE_DB})")
    parser.add_argument("--source-db", type=str, default=SOURCE_DB,
                        help=f"Source database (default: {SOURCE_DB})")
    parser.add_argument("--output", "-o", type=str,
                        help="Export comparison data to CSV")
    parser.add_argument("--plot", action="store_true",
                        help="Generate comparison plots")
    parser.add_argument("--sample", type=int,
                        help="Sample size for faster analysis")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.mempool_db):
        print(f"Error: mempool.space database not found: {args.mempool_db}")
        return
    
    if not os.path.exists(args.source_db):
        print(f"Error: Source database not found: {args.source_db}")
        return
    
    # Load and compare data
    df = load_comparison_data(args.source_db, args.mempool_db, args.sample)
    
    if df.empty:
        print("No matching transactions found")
        return
    
    # Compute derived metrics
    df = compute_derived_metrics(df)
    
    # Print summary
    print_summary_statistics(df)
    
    # Generate plots
    if args.plot:
        create_plots(df)
    
    # Export if requested
    if args.output:
        print(f"\nExporting comparison data to {args.output}...")
        df.to_csv(args.output, index=False)
        print(f"✓ Exported {len(df):,} rows")


if __name__ == "__main__":
    main()
