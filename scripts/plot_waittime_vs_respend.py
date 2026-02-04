import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Database path
DB_PATH = os.path.expanduser('/home/armin/datalake/data-samples/11-24-2025-15m-data-lake.db')

def plot_waittime_vs_respend():
    print(f"Connecting to database at {DB_PATH}...")
    try:
        conn = sqlite3.connect(DB_PATH)
    except sqlite3.OperationalError:
        print(f"Error: Could not connect to database at {DB_PATH}")
        return

    # Query to get waittime and min_respend_blocks
    # We filter for records where min_respend_blocks is not null to get only respent transactions
    query = """
    SELECT 
        waittime,
        min_respend_blocks
    FROM mempool_transactions 
    WHERE min_respend_blocks IS NOT NULL
      AND waittime IS NOT NULL
    """
    
    print("Executing query...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"Loaded {len(df):,} transactions.")
    
    if len(df) == 0:
        print("No data found matching criteria.")
        return

    # Exclude respends over 144 blocks (~1 day) for cleaner visualization
    df = df[df['min_respend_blocks'] <= 144]
    print(f"After filtering to ≤144 blocks: {len(df):,} transactions")
    
    # Convert waittime to minutes for better readability
    df['waittime_minutes'] = df['waittime'] / 60
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Use hexbin for dense data
    # waittime on Y, respend time (blocks) on X
    plt.hexbin(df['min_respend_blocks'], df['waittime_minutes'], 
               gridsize=50, cmap='viridis', bins='log', mincnt=1)
    
    plt.colorbar(label='Count (log scale)')
    
    plt.xlabel('Time to Respend (blocks)')
    plt.ylabel('Waittime (minutes)')
    plt.title('Waittime vs Time to Respend (Respend ≤ 144 blocks / 1 day)')
    plt.grid(True, alpha=0.3)
    
    output_file = 'plots/waittime_vs_respend.png'
    os.makedirs('plots', exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    
    # Also calculate correlation
    corr = df['waittime'].corr(df['min_respend_blocks'])
    print(f"Correlation between Waittime and Min Respend Blocks: {corr:.4f}")

if __name__ == "__main__":
    plot_waittime_vs_respend()
