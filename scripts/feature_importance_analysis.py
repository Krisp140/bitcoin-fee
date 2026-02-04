#!/usr/bin/env python3
"""
Feature Importance Analysis
===========================
Analyzes the importance of various features for predicting fee_rate using Random Forest.
"""

import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
DB_PATH = '/home/armin/datalake/data-samples/11-24-2025-15m-data-lake.db'
PHASE2_PATH = '/home/kristian/notebooks/phase2_derived_features.pkl'
BLOCK_LIMIT = 2000  # Sampling limit

def load_data():
    """Loads transaction data from SQLite and merges with Phase 2 features."""
    print("Loading data...")
    conn = sqlite3.connect(DB_PATH)
    
    # Get block hashes
    if BLOCK_LIMIT:
        print(f"Sampling {BLOCK_LIMIT} blocks...")
        blocks_query = f"""
            SELECT DISTINCT conf_block_hash 
            FROM mempool_transactions 
            WHERE conf_block_hash IS NOT NULL
            ORDER BY RANDOM()
            LIMIT {BLOCK_LIMIT}
        """
        sampled_blocks = pd.read_sql_query(blocks_query, conn)['conf_block_hash'].tolist()
        placeholders = ','.join(['?' for _ in sampled_blocks])
        where_clause = f"WHERE conf_block_hash IN ({placeholders})"
        params = sampled_blocks
    else:
        where_clause = "WHERE conf_block_hash IS NOT NULL"
        params = []
        
    query = f"""
        SELECT 
            tx_id,
            fee_rate,
            size,
            weight,
            mempool_size,
            mempool_tx_count,
            min_respend_blocks,
            waittime,
            rbf_fee_total,
            total_output_amount,
            version,
            child_txid
        FROM mempool_transactions 
        {where_clause}
    """
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    print(f"Loaded {len(df):,} transactions from SQL.")
    
    # Load Phase 2 features
    print("Loading Phase 2 derived features...")
    try:
        df_phase2 = pd.read_pickle(PHASE2_PATH)
        print(f"Loaded {len(df_phase2):,} Phase 2 records.")
        
        phase2_cols = ['tx_id', 'W_hat', 'rho_t', 'time_cost', 'F_tq']
        phase2_cols = [c for c in phase2_cols if c in df_phase2.columns]
        
        df = df.merge(df_phase2[phase2_cols], on='tx_id', how='left')
        print(f"Merged data. Shape: {df.shape}")
        
    except Exception as e:
        print(f"Error loading Phase 2 features: {e}")
        return df
                
    return df

def preprocess_and_train(df):
    """Preprocesses data and trains Random Forest."""
    print("Preprocessing...")
    
    # Fill NA
    df['rbf_fee_total'] = df['rbf_fee_total'].fillna(0)
    df['min_respend_blocks'] = df['min_respend_blocks'].fillna(-1)
    
    # Phase 2 Defaults
    if 'W_hat' in df.columns:
        df['W_hat'] = df['W_hat'].fillna(df['W_hat'].mean())
    if 'rho_t' in df.columns:
        df['rho_t'] = df['rho_t'].fillna(0)
    if 'time_cost' in df.columns:
        df['time_cost'] = df['time_cost'].fillna(0)
    if 'F_tq' in df.columns:
        df['F_tq'] = df['F_tq'].fillna(0)
    
    # Drops
    df = df.dropna(subset=['fee_rate', 'size', 'weight', 'mempool_size'])
    
    # Transformations
    df['log_fee_rate'] = np.log1p(df['fee_rate'])
    df['log_mempool_size'] = np.log1p(df['mempool_size'])
    df['log_waittime'] = np.log1p(df['waittime'])
    df['has_child_tx'] = df['child_txid'].apply(lambda x: 1 if x and str(x).strip() else 0)
    
    # Feature List
    feature_cols = [
        'size', 'weight', 'log_mempool_size', 'mempool_tx_count', 
        'min_respend_blocks', 'rbf_fee_total', 'total_output_amount',
        'log_waittime', 'has_child_tx',
        'W_hat', 'rho_t', 'time_cost', 'F_tq'
    ]
    
    # Filter for existing columns
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"Analyzing features: {feature_cols}")
    
    X = df[feature_cols]
    y = df['log_fee_rate']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest
    print("Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=50, # Sufficient for importance analysis
        max_depth=20,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_test)
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    
    # Feature Importance
    importances = rf.feature_importances_
    feature_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})
    feature_imp = feature_imp.sort_values(by='Importance', ascending=False)
    
    print("\n" + "="*40)
    print("FEATURE IMPORTANCE RANKING")
    print("="*40)
    print(feature_imp.to_string(index=False))
    print("="*40)
    
    return feature_imp

if __name__ == "__main__":
    df = load_data()
    preprocess_and_train(df)
