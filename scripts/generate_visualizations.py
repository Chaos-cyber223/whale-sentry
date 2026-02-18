#!/usr/bin/env python
"""Generate visualization images for README."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 11

# Load data
print("Loading data...")
swaps = pd.read_parquet('data/processed/swaps_3days_clean.parquet')
sandwich_candidates = pd.read_parquet('data/results/sandwich_3days.parquet')

# Convert timestamps
swaps['datetime'] = pd.to_datetime(swaps['timestamp'], unit='s')
sandwich_candidates['victim_timestamp_dt'] = pd.to_datetime(
    sandwich_candidates['victim_timestamp'], unit='s'
)

print(f"Loaded {len(swaps):,} swaps")
print(f"Loaded {len(sandwich_candidates):,} sandwich candidates")

# Create output directory
Path('images').mkdir(exist_ok=True)

# ============================================================
# 1. Confidence Distribution
# ============================================================
print("\nGenerating confidence distribution plot...")
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Histogram
axes[0].hist(sandwich_candidates['confidence_score'], bins=50, 
             color='steelblue', edgecolor='black', alpha=0.7)
axes[0].axvline(0.7, color='red', linestyle='--', linewidth=2, 
                label='High Confidence Threshold')
axes[0].set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Count', fontsize=12, fontweight='bold')
axes[0].set_title('Confidence Score Distribution', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Cumulative distribution
sorted_conf = np.sort(sandwich_candidates['confidence_score'])
cumulative = np.arange(1, len(sorted_conf) + 1) / len(sorted_conf) * 100
axes[1].plot(sorted_conf, cumulative, linewidth=2, color='darkgreen')
axes[1].axvline(0.7, color='red', linestyle='--', linewidth=2, 
                label='High Confidence Threshold')
axes[1].set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Cumulative Percentage (%)', fontsize=12, fontweight='bold')
axes[1].set_title('Cumulative Confidence Distribution', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('images/confidence_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: images/confidence_distribution.png")

# ============================================================
# 2. Top Attackers
# ============================================================
print("\nGenerating top attackers plot...")
sandwich_candidates['profit_estimate_usd'] = pd.to_numeric(
    sandwich_candidates['profit_estimate_usd'], errors='coerce'
).fillna(0)

attacker_stats = sandwich_candidates.groupby('attacker').agg({
    'victim_tx': 'count',
    'profit_estimate_usd': 'sum',
    'confidence_score': 'mean'
}).rename(columns={
    'victim_tx': 'attack_count',
    'profit_estimate_usd': 'total_profit',
    'confidence_score': 'avg_confidence'
}).sort_values('total_profit', ascending=False)

top_10 = attacker_stats.head(10)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Profit bar chart
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top_10)))
axes[0].barh(range(len(top_10)), top_10['total_profit'], color=colors, edgecolor='black')
axes[0].set_yticks(range(len(top_10)))
axes[0].set_yticklabels([f"{addr[:6]}...{addr[-4:]}" for addr in top_10.index])
axes[0].set_xlabel('Total Estimated Profit (USD)', fontsize=12, fontweight='bold')
axes[0].set_title('Top 10 Attackers by Profit', fontsize=14, fontweight='bold')
axes[0].grid(axis='x', alpha=0.3)
axes[0].invert_yaxis()

# Add value labels
for i, v in enumerate(top_10['total_profit']):
    axes[0].text(v, i, f' ${v:,.0f}', va='center', fontsize=10)

# Attack count vs profit scatter
axes[1].scatter(top_10['attack_count'], top_10['total_profit'], 
                s=200, alpha=0.6, c=top_10['avg_confidence'], 
                cmap='RdYlGn', edgecolors='black', linewidth=1.5)
axes[1].set_xlabel('Number of Attacks', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Total Profit (USD)', fontsize=12, fontweight='bold')
axes[1].set_title('Attack Frequency vs Profit', fontsize=14, fontweight='bold')
axes[1].grid(alpha=0.3)

# Add colorbar
cbar = plt.colorbar(axes[1].collections[0], ax=axes[1])
cbar.set_label('Avg Confidence', fontsize=11)

plt.tight_layout()
plt.savefig('images/top_attackers.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: images/top_attackers.png")

# ============================================================
# 3. Detection Timeline
# ============================================================
print("\nGenerating detection timeline plot...")
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Hourly detection count
sandwich_candidates['hour'] = sandwich_candidates['victim_timestamp_dt'].dt.floor('h')
hourly_counts = sandwich_candidates.groupby('hour').size()

axes[0].plot(hourly_counts.index, hourly_counts.values, 
             linewidth=2, color='darkblue', marker='o', markersize=4)
axes[0].fill_between(hourly_counts.index, hourly_counts.values, alpha=0.3, color='lightblue')
axes[0].set_xlabel('Time', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Detections per Hour', fontsize=12, fontweight='bold')
axes[0].set_title('Sandwich Attack Detection Timeline', fontsize=14, fontweight='bold')
axes[0].grid(alpha=0.3)

# Confidence over time
axes[1].scatter(sandwich_candidates['victim_timestamp_dt'], 
                sandwich_candidates['confidence_score'],
                alpha=0.4, s=20, c=sandwich_candidates['profit_estimate_usd'],
                cmap='YlOrRd', edgecolors='none')
axes[1].axhline(0.7, color='red', linestyle='--', linewidth=2, 
                label='High Confidence Threshold', alpha=0.7)
axes[1].set_xlabel('Time', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Confidence Score', fontsize=12, fontweight='bold')
axes[1].set_title('Detection Confidence Over Time', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

# Add colorbar
cbar = plt.colorbar(axes[1].collections[0], ax=axes[1])
cbar.set_label('Estimated Profit (USD)', fontsize=11)

plt.tight_layout()
plt.savefig('images/detection_timeline.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: images/detection_timeline.png")

# ============================================================
# Summary Statistics
# ============================================================
print("\n" + "="*60)
print("VISUALIZATION SUMMARY")
print("="*60)
print(f"Total Swaps:              {len(swaps):,}")
print(f"Sandwich Candidates:      {len(sandwich_candidates):,}")
print(f"Detection Rate:           {len(sandwich_candidates)/len(swaps)*100:.2f}%")
print(f"Unique Attackers:         {sandwich_candidates['attacker'].nunique()}")
print(f"High Confidence (≥0.7):   {len(sandwich_candidates[sandwich_candidates['confidence_score'] >= 0.7]):,}")
print(f"Total Estimated Profit:   ${sandwich_candidates['profit_estimate_usd'].sum():,.2f}")
print("="*60)
print("\n✅ All visualizations generated successfully!")
print("\nGenerated files:")
print("  - images/confidence_distribution.png")
print("  - images/top_attackers.png")
print("  - images/detection_timeline.png")
