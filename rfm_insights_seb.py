#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
    print("✓ SHAP library loaded successfully")
except (ImportError, Exception) as e:
    print(f"Note: SHAP not available ({type(e).__name__}). Using Permutation Importance instead.")
    SHAP_AVAILABLE = False
    from sklearn.inspection import permutation_importance

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

CLUSTER_COLORS = {0: '#e74c3c', 1: '#2ecc71', 2: '#3498db', 3: '#9b59b6', 4: '#f39c12'}


class Config:
    INPUT_FILE = "/Users/mazinbashir/.gemini/antigravity/scratch/bank_rfm_analysis/rfm_cleaned_output.csv"
    OUTPUT_DIR = "/Users/mazinbashir/.gemini/antigravity/scratch/bank_rfm_analysis"
    OPTIMAL_K = 3
    RANDOM_STATE = 42
    FEATURES_LOG = ['Recency_Log', 'Frequency_Log', 'Monetary_Log']
    FEATURES_RAW = ['Recency', 'Frequency', 'Monetary']


def prepare_data():
    print("=" * 70)
    print("STEP 1: DATA PREPARATION")
    print("=" * 70)
    df = pd.read_csv(Config.INPUT_FILE)
    print(f"✓ Loaded {len(df):,} customers")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[Config.FEATURES_LOG])
    kmeans = KMeans(n_clusters=Config.OPTIMAL_K, random_state=Config.RANDOM_STATE, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    print(f"✓ Applied K-Means with k={Config.OPTIMAL_K}")
    print(f"✓ Cluster distribution:\n{df['Cluster'].value_counts().sort_index().to_string()}")
    print()
    return df, X_scaled, scaler, kmeans


def create_segment_profiles(df: pd.DataFrame) -> tuple:
    print("=" * 70)
    print("STEP 2: SEGMENT PROFILING & BUSINESS MAPPING")
    print("=" * 70)
    profile = df.groupby('Cluster')[Config.FEATURES_RAW].mean()
    cluster_stats = df.groupby('Cluster').agg({'CustomerID': 'count', 'Monetary': 'sum'}).rename(columns={'CustomerID': 'Customer_Count', 'Monetary': 'Total_Revenue'})
    profile = profile.join(cluster_stats)
    profile['Revenue_Share'] = (profile['Total_Revenue'] / profile['Total_Revenue'].sum() * 100).round(2)
    r_median, f_median, m_median = df['Recency'].median(), df['Frequency'].median(), df['Monetary'].median()
    print(f"\n[Reference Medians]\n  Recency: {r_median:.0f} days | Frequency: {f_median:.1f} | Monetary: £{m_median:.2f}")
    labels, descriptions, banking_actions = {}, {}, {}
    for cluster in profile.index:
        r, f, m = profile.loc[cluster, 'Recency'], profile.loc[cluster, 'Frequency'], profile.loc[cluster, 'Monetary']
        if r <= r_median and f >= f_median and m >= m_median:
            labels[cluster], descriptions[cluster], banking_actions[cluster] = 'VIP / Whale Customers', 'High-value, highly engaged customers with recent activity', 'Private Banking services, Premium credit products, Wealth management'
        elif r <= r_median and f >= f_median:
            labels[cluster], descriptions[cluster], banking_actions[cluster] = 'Loyal Retailers', 'Consistent engagement with regular transaction patterns', 'Loyalty rewards, Cross-sell insurance, Upgrade debit to credit'
        elif r <= r_median:
            labels[cluster], descriptions[cluster], banking_actions[cluster] = 'Promising Newcomers', 'Recent customers with growth potential', 'Onboarding campaigns, First-purchase bonuses, App adoption'
        elif r > r_median and f >= f_median:
            labels[cluster], descriptions[cluster], banking_actions[cluster] = 'At-Risk / Churn', 'Previously active customers showing disengagement', 'Urgent retention campaigns, Win-back offers, Personal outreach'
        elif r > r_median and f < f_median and m < m_median:
            labels[cluster], descriptions[cluster], banking_actions[cluster] = 'Hibernating', 'Low engagement across all metrics, dormant accounts', 'Re-activation emails, Dormancy prevention, Consider exit'
        else:
            labels[cluster], descriptions[cluster], banking_actions[cluster] = 'Occasional Buyers', 'Sporadic activity with moderate value', 'Targeted promotions, Engagement campaigns, Product education'
    profile['Segment_Label'], profile['Description'], profile['Banking_Action'] = profile.index.map(labels), profile.index.map(descriptions), profile.index.map(banking_actions)
    print(f"\n{'='*70}\nSEGMENT PROFILES WITH BUSINESS MAPPING\n{'='*70}")
    for cluster in sorted(profile.index):
        row = profile.loc[cluster]
        print(f"\n  CLUSTER {cluster}: {row['Segment_Label']}\n  {'─'*40}\n  Customers: {row['Customer_Count']:,} ({row['Customer_Count']/len(df)*100:.1f}%)\n  Recency: {row['Recency']:.0f} days | Frequency: {row['Frequency']:.1f} | Monetary: £{row['Monetary']:,.2f}\n  Revenue Share: {row['Revenue_Share']:.1f}%\n  Action: {row['Banking_Action']}")
    print()
    return profile, labels


def perform_shap_analysis(df: pd.DataFrame, X_scaled: np.ndarray) -> dict:
    print("=" * 70)
    print("STEP 3: FEATURE IMPORTANCE ANALYSIS")
    print("=" * 70)
    X_features, y_labels = df[Config.FEATURES_LOG].copy(), df['Cluster'].values
    print("\n[Training Surrogate Model for Explainability]")
    surrogate = RandomForestClassifier(n_estimators=100, random_state=Config.RANDOM_STATE, max_depth=10)
    surrogate.fit(X_features, y_labels)
    print(f"  ● Surrogate model accuracy: {surrogate.score(X_features, y_labels):.4f}")
    shap_values, explainer, use_shap = None, None, SHAP_AVAILABLE
    if use_shap:
        try:
            explainer = shap.TreeExplainer(surrogate)
            shap_values = explainer.shap_values(X_features)
            mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0) if isinstance(shap_values, list) else np.abs(shap_values).mean(axis=0)
            feature_importance = {k: float(v) for k, v in zip(Config.FEATURES_LOG, mean_shap)}
            method_used = "SHAP"
        except Exception:
            use_shap = False
    if not use_shap or shap_values is None:
        feature_importance = {k: float(v) for k, v in zip(Config.FEATURES_LOG, surrogate.feature_importances_)}
        method_used = "Random Forest Gini Importance"
    total = sum(feature_importance.values())
    feature_importance_pct = {k: v/total*100 for k, v in feature_importance.items()}
    print(f"\n[Global Feature Importance]\n  Method: {method_used}")
    for f, imp in sorted(feature_importance_pct.items(), key=lambda x: x[1], reverse=True):
        print(f"  {f.replace('_Log', '').upper():<12} {imp:>5.1f}% {'█' * int(imp / 2)}")
    print()
    return {'explainer': explainer, 'shap_values': shap_values, 'feature_importance': feature_importance_pct, 'surrogate_model': surrogate, 'method': method_used}


def create_3d_scatter_plot(df: pd.DataFrame, labels: dict, save_path: str):
    print("=" * 70 + "\nSTEP 4A: 3D CLUSTER VISUALIZATION\n" + "=" * 70)
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    for cluster in sorted(df['Cluster'].unique()):
        data = df[df['Cluster'] == cluster]
        ax.scatter(data['Recency_Log'], data['Frequency_Log'], data['Monetary_Log'], c=CLUSTER_COLORS.get(cluster, '#95a5a6'), label=f"{cluster}: {labels[cluster]}", alpha=0.6, s=50, edgecolors='white', linewidth=0.5)
    ax.set_xlabel('\n\nRecency (Log-scaled)', fontweight='bold')
    ax.set_ylabel('\n\nFrequency (Log-scaled)', fontweight='bold')
    ax.set_zlabel('\n\nMonetary (Log-scaled)', fontweight='bold')
    ax.set_title('3D Customer Segmentation Visualization\nRFM-Based K-Means Clustering Analysis', fontweight='bold', fontsize=14, pad=20)
    ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), framealpha=0.95, fontsize=10)
    ax.view_init(elev=20, azim=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ 3D scatter plot saved to:\n  {save_path}\n")


def create_snake_plot(df: pd.DataFrame, labels: dict, save_path: str):
    print("=" * 70 + "\nSTEP 4B: SNAKE PLOT (SEGMENT COMPARISON)\n" + "=" * 70)
    cluster_means = df.groupby('Cluster')[Config.FEATURES_RAW].mean()
    cluster_means_scaled = (cluster_means - cluster_means.mean()) / cluster_means.std()
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax1 = axes[0]
    ax1.set_facecolor('#f8f9fa')
    for cluster in sorted(cluster_means_scaled.index):
        ax1.plot(range(3), cluster_means_scaled.loc[cluster].values, marker='o', markersize=12, linewidth=3, color=CLUSTER_COLORS.get(cluster, '#95a5a6'), label=f"Cluster {cluster}: {labels[cluster]}", markeredgecolor='white', markeredgewidth=2)
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(['Recency\n(lower = better)', 'Frequency\n(higher = better)', 'Monetary\n(higher = better)'])
    ax1.set_ylabel('Standardized Score (Z-Score)', fontweight='bold')
    ax1.set_title('Snake Plot: Segment RFM Comparison', fontweight='bold', fontsize=13)
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.legend(loc='upper right', framealpha=0.95, fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(-2.5, 2.5)
    ax2 = axes[1]
    ax2.set_facecolor('#f8f9fa')
    x, width = np.arange(3), 0.25
    for i, cluster in enumerate(sorted(cluster_means.index)):
        vals = cluster_means.loc[cluster].values
        vals_norm = (vals - cluster_means.min().values) / (cluster_means.max().values - cluster_means.min().values) * 100
        ax2.bar(x + width*i, vals_norm, width, label=labels[cluster], color=CLUSTER_COLORS.get(cluster, '#95a5a6'), edgecolor='white')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(['Recency', 'Frequency', 'Monetary'])
    ax2.set_ylabel('Normalized Scale (0-100)', fontweight='bold')
    ax2.set_title('Segment Comparison: Absolute RFM Values', fontweight='bold', fontsize=13)
    ax2.legend(loc='upper right', framealpha=0.95, fontsize=9)
    fig.suptitle('Customer Segment Analysis\nData-Driven Insights Using RFM Clustering', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Snake plot saved to:\n  {save_path}\n")


def create_shap_visualization(shap_results: dict, df: pd.DataFrame, save_path: str):
    print("=" * 70 + "\nSTEP 4C: SHAP FEATURE IMPORTANCE VISUALIZATION\n" + "=" * 70)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_facecolor('#f8f9fa')
    importance = shap_results['feature_importance']
    features = [f.replace('_Log', '').upper() for f in importance.keys()]
    values = list(importance.values())
    sorted_pairs = sorted(zip(features, values), key=lambda x: x[1], reverse=True)
    features_sorted, values_sorted = zip(*sorted_pairs)
    bars = ax1.barh(features_sorted, values_sorted, color=['#2ecc71', '#3498db', '#e74c3c'], edgecolor='white', linewidth=2, height=0.6)
    for bar, val in zip(bars, values_sorted):
        ax1.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', va='center', fontweight='bold', fontsize=12)
    ax1.set_xlabel('Importance (%)', fontweight='bold')
    ax1.set_title('Feature Importance Analysis\nWhich Features Drive Segmentation?', fontweight='bold', fontsize=13)
    ax1.set_xlim(0, max(values_sorted) * 1.2)
    ax1.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ SHAP visualization saved to:\n  {save_path}\n")


def create_executive_summary_dashboard(df: pd.DataFrame, profile: pd.DataFrame, labels: dict, shap_results: dict, save_path: str):
    print("=" * 70 + "\nSTEP 4D: EXECUTIVE SUMMARY DASHBOARD\n" + "=" * 70)
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    segment_counts = df['Cluster'].value_counts().sort_index()
    colors = [CLUSTER_COLORS.get(i, '#95a5a6') for i in segment_counts.index]
    ax1.pie(segment_counts, labels=[f"{labels[i]}\n({c:,})" for i, c in segment_counts.items()], autopct='%1.1f%%', colors=colors, explode=[0.05]*len(segment_counts), shadow=True)
    ax1.set_title('Customer Segment Distribution', fontweight='bold', fontsize=13)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.pie(profile['Revenue_Share'], labels=[f"{labels[i]}\n(£{profile.loc[i, 'Total_Revenue']:,.0f})" for i in profile.index], autopct='%1.1f%%', colors=colors, explode=[0.05]*len(profile), shadow=True)
    ax2.set_title('Revenue Share by Segment', fontweight='bold', fontsize=13)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor('#f8f9fa')
    imp = shap_results['feature_importance']
    sorted_imp = sorted([(f.replace('_Log', '').upper(), v) for f, v in imp.items()], key=lambda x: x[1], reverse=True)
    ax3.barh([x[0] for x in sorted_imp], [x[1] for x in sorted_imp], color=['#2ecc71', '#3498db', '#e74c3c'], edgecolor='white')
    ax3.set_xlabel('Importance (%)', fontweight='bold')
    ax3.set_title('Feature Importance', fontweight='bold', fontsize=13)
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.set_facecolor('#f8f9fa')
    cluster_means = df.groupby('Cluster')[Config.FEATURES_RAW].mean()
    cluster_means_scaled = (cluster_means - cluster_means.mean()) / cluster_means.std()
    for cluster in sorted(cluster_means_scaled.index):
        ax4.plot(range(3), cluster_means_scaled.loc[cluster].values, marker='o', markersize=14, linewidth=3, color=CLUSTER_COLORS.get(cluster, '#95a5a6'), label=labels[cluster], markeredgecolor='white', markeredgewidth=2)
    ax4.set_xticks(range(3))
    ax4.set_xticklabels(['Recency', 'Frequency', 'Monetary'], fontsize=12, fontweight='bold')
    ax4.set_ylabel('Standardized Score', fontweight='bold')
    ax4.set_title('Snake Plot: Segment RFM Profiles', fontweight='bold', fontsize=13)
    ax4.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax4.legend(loc='upper right', framealpha=0.95)
    ax4.set_ylim(-2.5, 2.5)
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    action_text = "RECOMMENDED ACTIONS\n" + "═"*25 + "\n\n"
    for c in sorted(profile.index):
        action_text += f"● {labels[c]}:\n  {profile.loc[c, 'Banking_Action']}\n\n"
    ax5.text(0.05, 0.95, action_text, transform=ax5.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1', edgecolor='#2c3e50', alpha=0.9))
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')
    table_data = [[f"Cluster {c}", labels[c], f"{profile.loc[c, 'Customer_Count']:,}", f"£{profile.loc[c, 'Monetary']:.2f}", f"{profile.loc[c, 'Frequency']:.1f}", f"{profile.loc[c, 'Recency']:.0f}", f"{profile.loc[c, 'Revenue_Share']:.1f}%"] for c in sorted(profile.index)]
    table = ax6.table(cellText=table_data, colLabels=['Cluster', 'Segment', 'Customers', 'Avg Monetary', 'Avg Frequency', 'Avg Recency', 'Revenue Share'], loc='center', cellLoc='center', colWidths=[0.1, 0.25, 0.1, 0.12, 0.12, 0.11, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    for key in table.get_celld():
        cell = table.get_celld()[key]
        if key[0] == 0:
            cell.set_text_props(fontweight='bold', color='white')
            cell.set_facecolor('#2c3e50')
    ax6.set_title('Segment Summary Table', fontweight='bold', fontsize=13, pad=20)
    fig.suptitle('Customer Segmentation Analysis\nExecutive Dashboard', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Executive dashboard saved to:\n  {save_path}\n")


def main():
    print("\n" + "=" * 70 + "\nDATA-DRIVEN CUSTOMER INSIGHTS\nSEB Data Analyst Trainee Application\n" + "=" * 70 + "\n")
    df, X_scaled, scaler, kmeans = prepare_data()
    profile, labels = create_segment_profiles(df)
    shap_results = perform_shap_analysis(df, X_scaled)
    create_3d_scatter_plot(df, labels, f"{Config.OUTPUT_DIR}/3d_cluster_visualization.png")
    create_snake_plot(df, labels, f"{Config.OUTPUT_DIR}/snake_plot_comparison.png")
    create_shap_visualization(shap_results, df, f"{Config.OUTPUT_DIR}/shap_feature_importance.png")
    create_executive_summary_dashboard(df, profile, labels, shap_results, f"{Config.OUTPUT_DIR}/executive_dashboard.png")
    df['Segment_Label'] = df['Cluster'].map(labels)
    df.to_csv(f"{Config.OUTPUT_DIR}/rfm_final_segmented.csv", index=False)
    print("=" * 70 + "\nANALYSIS COMPLETE\n" + "=" * 70)
    print(f"\nFILES GENERATED:\n• 3d_cluster_visualization.png\n• snake_plot_comparison.png\n• shap_feature_importance.png\n• executive_dashboard.png\n• rfm_final_segmented.csv\n")
    return df, profile, shap_results


if __name__ == "__main__":
    df_final, profiles, shap_analysis = main()
