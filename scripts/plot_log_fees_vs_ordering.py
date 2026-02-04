#!/usr/bin/env python3
"""
Generate figure: log fees (vertical) vs ordering (horizontal)
Data: all transactions from mempool_space_data.db
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
DB_PATH = Path(__file__).parent.parent / "mempool_space_data.db"
OUTPUT_PATH = Path(__file__).parent.parent / "plots" / "log_fees_vs_ordering.png"


def tie_aware_percentile(df, group_col="block_hash", r_col="fee_rate"):
    """
    Compute tie-aware percentile rank within each group.
    
    Formula: p_it = (#{j: r_jt < r_it} + 0.5 * #{j: r_jt = r_it}) / N_t
    
    This defines priority where higher p_it = higher priority (higher fee rate).
    """
    def _within_group(g):
        r = g[r_col].to_numpy()
        N = len(g)
        
        if N == 0:
            return pd.Series(dtype=float)
        if N == 1:
            return pd.Series([0.5], index=g.index)
            
        # Count occurrences of each unique fee rate
        vc = pd.Series(r).value_counts().sort_index()
        
        # Cumulative count of values strictly less than each value
        cum_less = vc.cumsum().shift(1, fill_value=0)
        
        # Map each fee rate to its "strictly less than" count
        less = pd.Series(r).map(cum_less).to_numpy()
        
        # Map each fee rate to its count (ties)
        eq = pd.Series(r).map(vc).to_numpy()
        
        # Compute percentile: (less + 0.5 * eq) / N
        p_midrank = (less + 0.5 * eq) / N
        
        return pd.Series(p_midrank, index=g.index)
    
    return df.groupby(group_col, group_keys=False).apply(_within_group)


def main():
    print(f"Loading data from {DB_PATH}")
    
    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    
    # Load all transactions
    query = """
    SELECT txid, fee, fee_rate, block_hash, block_height
    FROM transactions
    WHERE fee > 0 AND fee_rate > 0
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Loaded {len(df):,} transactions from {df['block_hash'].nunique()} blocks")
    
    # Compute priority percentile (ordering) within each block
    print("Computing priority percentiles...")
    df['p'] = tie_aware_percentile(df)
    
    # Compute log fee
    df['log_fee_rate'] = np.log10(df['fee_rate'])
    
    # Create figure
    print("Creating figure...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use hexbin for efficiency with large datasets
    hb = ax.hexbin(
        df['p'], 
        df['log_fee_rate'],
        gridsize=100,
        cmap='viridis',
        mincnt=1,
        bins='log'
    )
    
    # Add colorbar
    cb = fig.colorbar(hb, ax=ax, label='Count (log scale)')
    
    # Labels and title
    ax.set_xlabel('Ordering (Priority Percentile)', fontsize=12)
    ax.set_ylabel('Log₁₀(Fee Rate in sat/vB)', fontsize=12)
    ax.set_title(f'Transaction Fees vs Priority Ordering\n(n={len(df):,} transactions from {df["block_hash"].nunique()} blocks)', 
                 fontsize=14)
    
    # Set x-axis limits
    ax.set_xlim(0, 1)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    ax.text(0.02, 0.98, 
            'Higher priority (right) = higher fee rate\nOrdering within each block',
            transform=ax.transAxes, 
            verticalalignment='top',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save figure
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight')
    print(f"✓ Saved figure to {OUTPUT_PATH}")
    
    # Also show some statistics
    print(f"\nStatistics:")
    print(f"  Fee range: {df['fee_rate'].min():,.0f} - {df['fee_rate'].max():,.0f} satoshis")
    print(f"  Fee rate range: {df['fee_rate'].min():.2f} - {df['fee_rate'].max():.2f} sat/vB")
    print(f"  Priority percentile range: [{df['p'].min():.4f}, {df['p'].max():.4f}]")
    
    plt.close()


if __name__ == "__main__":
    main()
