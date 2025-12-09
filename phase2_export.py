#!/usr/bin/env python3
"""
Phase 2 Export Script
=====================
Generates derived features for Phase 3 from transaction data.
Memory-optimized version without visualization overhead.
"""

import os
import gc
import sqlite3
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
DB_PATH = '/home/armin/datalake/data-samples/11-24-2025-15m-data-lake.db'
OUTPUT_PATH = '/home/kristian/notebooks/phase2_derived_features.pkl'
EPOCH_DURATION_MINUTES = 30
Q_MAX = 10  # Number of quantile bins
BLOCK_LIMIT = None  # Set to e.g. 5000 to limit blocks loaded, None for all

# ============================================================================
# STEP 1: LOAD DATA (Memory-efficient)
# ============================================================================
def load_data():
    print("=" * 60)
    print("STEP 1: Loading data from SQLite...")
    print("=" * 60)
    
    conn = sqlite3.connect(DB_PATH)
    
    # Get block hashes
    if BLOCK_LIMIT:
        blocks_query = f"""
            SELECT DISTINCT conf_block_hash 
            FROM mempool_transactions 
            WHERE conf_block_hash IS NOT NULL
            ORDER BY RANDOM()
            LIMIT {BLOCK_LIMIT}
        """
    else:
        blocks_query = """
            SELECT DISTINCT conf_block_hash 
            FROM mempool_transactions 
            WHERE conf_block_hash IS NOT NULL
        """
    
    print("  Fetching block hashes...")
    sampled_blocks = pd.read_sql_query(blocks_query, conn)['conf_block_hash'].tolist()
    print(f"  Found {len(sampled_blocks):,} blocks")
    
    # Load transactions in chunks to manage memory
    print("  Loading transactions...")
    placeholders = ','.join(['?' for _ in sampled_blocks])
    query = f"""
        SELECT tx_id, found_at, mined_at, pruned_at, waittime,
               min_respend_blocks, child_txid, rbf_fee_total, 
               mempool_size, mempool_tx_count
        FROM mempool_transactions 
        WHERE conf_block_hash IN ({placeholders})
    """
    
    txs = pd.read_sql_query(query, conn, params=sampled_blocks)
    conn.close()
    
    print(f"  ✓ Loaded {len(txs):,} transactions")
    print(f"  Memory usage: {txs.memory_usage(deep=True).sum() / 1e9:.2f} GB")
    
    return txs


# ============================================================================
# STEP 2: CREATE EPOCHS
# ============================================================================
def create_epochs(df):
    print("\n" + "=" * 60)
    print("STEP 2: Creating epochs...")
    print("=" * 60)
    
    # Convert found_at to timestamp
    if df['found_at'].dtype == 'object' or pd.api.types.is_datetime64_any_dtype(df['found_at']):
        df['found_at_ts'] = pd.to_datetime(df['found_at']).astype('int64') // 10**9
    else:
        df['found_at_ts'] = df['found_at']
    
    start_time = df['found_at_ts'].min()
    end_time = df['found_at_ts'].max()
    
    epoch_seconds = EPOCH_DURATION_MINUTES * 60
    num_epochs = int(np.ceil((end_time - start_time) / epoch_seconds))
    
    bins = np.linspace(start_time, start_time + (num_epochs * epoch_seconds), num_epochs + 1)
    df['epoch'] = pd.cut(df['found_at_ts'], bins=bins, labels=False, include_lowest=True)
    
    print(f"  ✓ Created {num_epochs} epochs of {EPOCH_DURATION_MINUTES} minutes each")
    return df


# ============================================================================
# STEP 3: USE MEMPOOL SIZE AS CONGESTION (rho_t)
# ============================================================================
def compute_congestion(df):
    print("\n" + "=" * 60)
    print("STEP 3: Using mempool_tx_count as congestion (rho_t)...")
    print("=" * 60)
    
    # Use mempool_size directly as rho_t
    df['rho_t'] = df['mempool_tx_count']
    
    print(f"  ✓ Using mempool_tx_count directly as rho_t")
    print(f"  Mean rho_t: {df['rho_t'].mean():,.0f}")
    print(f"  Non-null values: {df['rho_t'].notna().sum():,}")
    return df


# ============================================================================
# STEP 4: COMPUTE TIME_COST
# ============================================================================
def compute_time_cost(df):
    print("\n" + "=" * 60)
    print("STEP 4: Computing time_cost...")
    print("=" * 60)
    
    df['respend_delay'] = df['min_respend_blocks']
    valid_respend = (df['respend_delay'] >= 0) & df['respend_delay'].notna()
    
    df['time_cost'] = np.nan
    df.loc[valid_respend, 'time_cost'] = 1 / (df.loc[valid_respend, 'respend_delay'] + 1e-6)
    
    print(f"  ✓ Computed time_cost for {valid_respend.sum():,} transactions")
    print(f"  Mean time_cost: {df['time_cost'].mean():.6f}")
    return df


# ============================================================================
# STEP 5: TRAIN MODEL AND GENERATE PREDICTIONS
# ============================================================================
def train_and_predict(df):
    print("\n" + "=" * 60)
    print("STEP 5: Training model and generating predictions...")
    print("=" * 60)
    
    # Create binary features
    df['has_child'] = df['child_txid'].notna().astype(int)
    df['rbf_flag'] = df['rbf_fee_total'].notna().astype(int)
    
    # Log-transform features
    df['log_rho_t'] = np.log1p(df['rho_t'])
    df['log_time_cost'] = df['time_cost']  # Already transformed in original
    df['log_waittime'] = np.log1p(df['waittime'])
    
    # Prepare training data
    feature_cols = ['log_rho_t', 'log_time_cost', 'has_child', 'rbf_flag']
    cols_needed = feature_cols + ['log_waittime']
    
    df_clean = df[cols_needed].dropna()
    print(f"  Training on all {len(df_clean):,} clean samples")
    
    X = df_clean[feature_cols]
    y = df_clean['log_waittime']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"  Training Random Forest on {len(X_train):,} samples...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    y_pred_test = model.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)
    print(f"  ✓ Model trained. Test R² = {r2:.4f}")
    
    # Clean up training data
    del df_clean, X, y, X_train, X_test, y_train, y_test
    gc.collect()
    
    # Generate predictions for full dataset
    print("  Generating predictions for full dataset...")
    valid_features = df[feature_cols].notna().all(axis=1)
    df['W_hat'] = np.nan
    
    # Predict in chunks to manage memory
    chunk_size = 500_000
    valid_indices = df.index[valid_features]
    
    for i in range(0, len(valid_indices), chunk_size):
        chunk_idx = valid_indices[i:i+chunk_size]
        X_chunk = df.loc[chunk_idx, feature_cols]
        y_pred_log = model.predict(X_chunk)
        df.loc[chunk_idx, 'W_hat'] = np.expm1(y_pred_log)
        
        if i % 1_000_000 == 0:
            print(f"    Predicted {i:,}/{len(valid_indices):,}...")
    
    print(f"  ✓ Generated {valid_features.sum():,} predictions")
    print(f"  Mean W_hat: {df['W_hat'].mean():.2f} seconds")
    
    del model
    gc.collect()
    
    return df


# ============================================================================
# STEP 6: COMPUTE QUANTILES AND TAIL DISTRIBUTION
# ============================================================================
def compute_quantiles_and_tail(df):
    print("\n" + "=" * 60)
    print("STEP 6: Computing quantiles and tail distribution...")
    print("=" * 60)
    
    # Assign quantiles per epoch (memory-efficient approach)
    df['time_cost_quantile'] = np.nan
    
    unique_epochs = df['epoch'].dropna().unique()
    print(f"  Processing {len(unique_epochs)} epochs...")
    
    for i, epoch in enumerate(unique_epochs):
        if i % 500 == 0:
            print(f"    Epoch {i}/{len(unique_epochs)}...")
        
        epoch_mask = df['epoch'] == epoch
        tc_values = df.loc[epoch_mask, 'time_cost']
        
        if tc_values.notna().sum() < Q_MAX:
            df.loc[epoch_mask, 'time_cost_quantile'] = 1.0
            continue
        
        try:
            df.loc[epoch_mask, 'time_cost_quantile'] = pd.qcut(
                tc_values, q=Q_MAX, labels=range(1, Q_MAX + 1), duplicates='drop'
            ).astype(float)
        except ValueError:
            df.loc[epoch_mask, 'time_cost_quantile'] = 1.0
    
    gc.collect()
    
    # Compute F_tq (tail distribution)
    print("  Computing F_tq...")
    df['F_tq'] = np.nan
    
    for epoch in unique_epochs:
        epoch_mask = df['epoch'] == epoch
        epoch_data = df.loc[epoch_mask]
        total_count = epoch_data['time_cost_quantile'].notna().sum()
        
        if total_count == 0:
            continue
        
        for q in range(1, Q_MAX + 1):
            q_mask = epoch_mask & (df['time_cost_quantile'] == q)
            upper_tail = (epoch_data['time_cost_quantile'] >= q).sum()
            F_tq = upper_tail / total_count
            df.loc[q_mask, 'F_tq'] = F_tq
    
    gc.collect()
    
    print(f"  ✓ Computed quantiles and F_tq")
    return df


# ============================================================================
# STEP 7: EXPORT
# ============================================================================
def export_results(df):
    print("\n" + "=" * 60)
    print("STEP 7: Exporting results...")
    print("=" * 60)
    
    derived_cols = [
        'tx_id',
        'W_hat',
        'rho_t',
        'time_cost',
        'time_cost_quantile',
        'F_tq',
        'epoch',
    ]
    
    result = df[[col for col in derived_cols if col in df.columns]].copy()
    result.to_pickle(OUTPUT_PATH)
    
    print(f"  ✓ Saved to: {OUTPUT_PATH}")
    print(f"  Rows: {len(result):,}")
    print(f"  Columns: {list(result.columns)}")
    
    # Validation
    print("\n  Validation:")
    print(f"    W_hat non-null: {result['W_hat'].notna().sum():,}")
    print(f"    F_tq non-null: {result['F_tq'].notna().sum():,}")
    
    return result


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "=" * 60)
    print("PHASE 2 EXPORT SCRIPT")
    print("=" * 60)
    print(f"Database: {DB_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print("=" * 60)
    
    # Run pipeline
    df = load_data()
    df = create_epochs(df)
    df = compute_congestion(df)
    df = compute_time_cost(df)
    df = train_and_predict(df)
    df = compute_quantiles_and_tail(df)
    result = export_results(df)
    
    print("\n" + "=" * 60)
    print("✓ EXPORT COMPLETE!")
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    main()
