#!/usr/bin/env python3
"""
Generate Two-Stage Estimation Pipeline Diagram
==============================================
Creates a visual flowchart showing the data flow from raw data through
Phase 2 feature engineering to Phase 3 fee prediction.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(1, 1, figsize=(18, 12))  # Increased size
ax.set_xlim(0, 18)
ax.set_ylim(0, 12)
ax.axis('off')

# Color scheme
colors = {
    'input': '#FF6B6B',      # Red - inputs
    'process': '#4ECDC4',     # Teal - processing
    'output': '#45B7D1',      # Blue - outputs
    'stage1': '#E0F7FA',      # Light cyan - Stage 1 background
    'stage2': '#FFEBEE',      # Light pink - Stage 2 background
    'arrow': '#2C3E50',       # Dark blue - arrows
    'text': '#2C3E50'         # Dark text
}

# Define box styles
input_box_style = dict(boxstyle="round,pad=0.2", facecolor=colors['input'], edgecolor='#C62828', linewidth=2)
process_box_style = dict(boxstyle="round,pad=0.2", facecolor=colors['process'], edgecolor='#00695C', linewidth=2)
output_box_style = dict(boxstyle="round,pad=0.2", facecolor=colors['output'], edgecolor='#0277BD', linewidth=2)

# Font sizes
title_font = {'size': 20, 'weight': 'bold', 'color': colors['text']}
stage_font = {'size': 16, 'weight': 'bold', 'color': colors['text']}
box_font = {'size': 11, 'weight': 'bold', 'color': 'white'}
small_font = {'size': 10, 'color': 'white'}
var_font = {'size': 9, 'color': colors['text']}

def draw_box(ax, x_center, y_center, width, height, text_title, text_sub, style):
    """Draw a centered box with title and subtitle."""
    # FancyBboxPatch takes (x,y) as bottom-left corner
    x = x_center - width / 2
    y = y_center - height / 2
    
    # Draw box
    patch = FancyBboxPatch((x, y), width, height, **style)
    ax.add_patch(patch)
    
    # Draw text
    ax.text(x_center, y_center + 0.15, text_title, ha='center', va='center', **box_font)
    ax.text(x_center, y_center - 0.15, text_sub, ha='center', va='center', **small_font)
    
    return patch

# ============================================================================
# STAGE 1 BACKGROUND (Left Side)
# ============================================================================
# Background for Stage 1
stage1_bg = FancyBboxPatch((0.5, 0.5), 8, 10.5, 
                           boxstyle="round,pad=0.2", 
                           facecolor=colors['stage1'], 
                           edgecolor='gray', 
                           linewidth=2, 
                           alpha=0.5,
                           zorder=0)
ax.add_patch(stage1_bg)
ax.text(4.5, 10.6, 'STAGE 1: Feature Engineering (Phase 2)', 
        ha='center', **stage_font)

# ============================================================================
# STAGE 2 BACKGROUND (Right Side)
# ============================================================================
# Background for Stage 2
stage2_bg = FancyBboxPatch((9.5, 0.5), 8, 10.5, 
                           boxstyle="round,pad=0.2", 
                           facecolor=colors['stage2'], 
                           edgecolor='gray', 
                           linewidth=2, 
                           alpha=0.5,
                           zorder=0)
ax.add_patch(stage2_bg)
ax.text(13.5, 10.6, 'STAGE 2: Fee Prediction (Phase 3)', 
        ha='center', **stage_font)

# ============================================================================
# STAGE 1 CONTENT
# ============================================================================

# 1. Input Box (Top)
draw_box(ax, 4.5, 9.5, 6, 1.0, 'Raw Transaction Data', 'SQLite Database (15M transactions)', input_box_style)

# Input variables (below box)
input_vars = [
    'tx_id, found_at, mined_at',
    'waittime, min_respend_blocks',
    'mempool_size, mempool_tx_count'
]
y_pos = 8.7
for var in input_vars:
    ax.text(4.5, y_pos, f'• {var}', ha='center', **var_font)
    y_pos -= 0.25

# 2. Process Step 1: Create Epochs (Left)
draw_box(ax, 2.5, 7.0, 3.0, 0.8, '1. Create Epochs', '30-min windows', process_box_style)

# 3. Process Step 2: Compute Congestion (Right)
draw_box(ax, 6.5, 7.0, 3.0, 0.8, '2. Compute ρ_t', 'Mempool Congestion', process_box_style)

# 4. Process Step 3: Compute Time Cost (Left)
draw_box(ax, 2.5, 5.0, 3.0, 0.8, '3. Compute Time Cost', 'Impatience Proxy', process_box_style)

# 5. Process Step 4: Train Wait Model (Right)
draw_box(ax, 6.5, 5.0, 3.0, 0.8, '4. Train W_hat Model', 'Random Forest', process_box_style)

# 6. Process Step 5: Quantiles (Center)
draw_box(ax, 4.5, 3.0, 4.0, 0.8, '5. Compute Quantiles', 'F_tq (Tail Dist)', process_box_style)

# 7. Output Box (Bottom)
draw_box(ax, 4.5, 1.2, 6, 0.8, 'Phase 2 Features', 'W_hat, ρ_t, time_cost, F_tq', output_box_style)

# Stage 1 Arrows
# Input -> Steps 1 & 2
ax.add_patch(FancyArrowPatch((4.5, 8.8), (2.5, 7.5), arrowstyle='->', lw=2, color=colors['arrow'], connectionstyle="arc3,rad=0.2"))
ax.add_patch(FancyArrowPatch((4.5, 8.8), (6.5, 7.5), arrowstyle='->', lw=2, color=colors['arrow'], connectionstyle="arc3,rad=-0.2"))

# Step 1 -> Step 3
ax.add_patch(FancyArrowPatch((2.5, 6.6), (2.5, 5.4), arrowstyle='->', lw=2, color=colors['arrow']))
# Step 2 -> Step 4
ax.add_patch(FancyArrowPatch((6.5, 6.6), (6.5, 5.4), arrowstyle='->', lw=2, color=colors['arrow']))

# Steps 3 & 4 -> Step 5
ax.add_patch(FancyArrowPatch((2.5, 4.6), (4.5, 3.4), arrowstyle='->', lw=2, color=colors['arrow'], connectionstyle="arc3,rad=0.2"))
ax.add_patch(FancyArrowPatch((6.5, 4.6), (4.5, 3.4), arrowstyle='->', lw=2, color=colors['arrow'], connectionstyle="arc3,rad=-0.2"))

# Step 5 -> Output
ax.add_patch(FancyArrowPatch((4.5, 2.6), (4.5, 1.6), arrowstyle='->', lw=2, color=colors['arrow']))


# ============================================================================
# TRANSITION (Merge)
# ============================================================================
# Arrow from Stage 1 Output to Stage 2 Input
ax.add_patch(FancyArrowPatch((7.6, 1.2), (10.4, 9.5), 
                             arrowstyle='simple,head_length=0.8,head_width=0.8,tail_width=0.2', 
                             color=colors['arrow'], 
                             connectionstyle="arc3,rad=-0.3", zorder=5))
ax.text(9.0, 5.5, 'Merge Phase 2\nFeatures', ha='center', fontsize=10, weight='bold', 
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray'))


# ============================================================================
# STAGE 2 CONTENT
# ============================================================================

# 1. Input Box (Top)
draw_box(ax, 13.5, 9.5, 6, 1.0, 'Enhanced Dataset', 'Raw Data + Phase 2 Features', input_box_style)

# Additional variables
s2_vars = [
    'Exchange Flags (From/To)',
    'Blockspace Utilization',
    'Transaction Value (V_it)'
]
y_pos = 8.7
for var in s2_vars:
    ax.text(13.5, y_pos, f'• {var}', ha='center', **var_font)
    y_pos -= 0.25

# 2. Process Step 1: Feature Engineering (Center)
draw_box(ax, 13.5, 7.0, 4.0, 0.8, '1. Structural Features', 'Log transforms, interactions', process_box_style)

# 3. Process Step 2: Riemann Sum (Center)
draw_box(ax, 13.5, 5.0, 4.0, 0.8, '2. Compute Riemann Sum', 'Competition term', process_box_style)

# 4. Process Step 3: Train Models (Center)
draw_box(ax, 13.5, 3.0, 5.0, 0.8, '3. Train Fee Models', 'OLS, Quantile, Gamma GLM, Spline', process_box_style)

# 5. Output Box (Bottom)
draw_box(ax, 13.5, 1.2, 6, 0.8, 'Fee Predictions & Analysis', 'Structural Coefficients, Diagnostics', output_box_style)

# Stage 2 Arrows
# Input -> Step 1
ax.add_patch(FancyArrowPatch((13.5, 8.8), (13.5, 7.4), arrowstyle='->', lw=2, color=colors['arrow']))
# Step 1 -> Step 2
ax.add_patch(FancyArrowPatch((13.5, 6.6), (13.5, 5.4), arrowstyle='->', lw=2, color=colors['arrow']))
# Step 2 -> Step 3
ax.add_patch(FancyArrowPatch((13.5, 4.6), (13.5, 3.4), arrowstyle='->', lw=2, color=colors['arrow']))
# Step 3 -> Output
ax.add_patch(FancyArrowPatch((13.5, 2.6), (13.5, 1.6), arrowstyle='->', lw=2, color=colors['arrow']))

# ============================================================================
# TITLE & LEGEND
# ============================================================================
ax.text(9, 11.5, 'Two-Stage Estimation Pipeline', ha='center', **title_font)

# Legend (Bottom Left)
leg_x, leg_y = 1.0, 0.5

# Input
#draw_box(ax, leg_x, leg_y, 1.2, 0.4, '', '', input_box_style)
#ax.text(leg_x + 0.7, leg_y, 'Input', ha='left', va='center', fontsize=10, weight='bold')

# Process
#draw_box(ax, leg_x + 2.5, leg_y, 1.2, 0.4, '', '', process_box_style)
#ax.text(leg_x + 3.2, leg_y, 'Process', ha='left', va='center', fontsize=10, weight='bold')

# Output
#draw_box(ax, leg_x + 5.0, leg_y, 1.2, 0.4, '', '', output_box_style)
#ax.text(leg_x + 5.7, leg_y, 'Output', ha='left', va='center', fontsize=10, weight='bold')

# Save
output_path = '/home/kristian/notebooks/plots/pipeline_diagram.png'
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Saved improved diagram to: {output_path}")

# Also save as PDF
output_path_pdf = '/home/kristian/notebooks/plots/pipeline_diagram.pdf'
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
print(f"✓ Saved PDF to: {output_path_pdf}")

plt.close()
