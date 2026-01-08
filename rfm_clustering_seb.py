#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import warnings

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12


class Config:
    INPUT_FILE = "/Users/mazinbashir/.gemini/antigravity/scratch/bank_rfm_analysis/rfm_cleaned_output.csv"
    OUTPUT_FILE = "/Users/mazinbashir/.gemini/antigravity/scratch/bank_rfm_analysis/rfm_segmented.csv"
    PLOT_FILE = "/Users/mazinbashir/.gemini/antigravity/scratch/bank_rfm_analysis/clustering_analysis.png"
    K_RANGE = range(2, 11)
    RANDOM_STATE = 42
    N_INIT = 10
    MAX_ITER = 300
    FEATURES = ['Recency_Log', 'Frequency_Log', 'Monetary_Log']


def load_rfm_data(filepath: str) -> pd.DataFrame:
    print("=" * 70)
    print("STEP 1: LOADING PRE-PROCESSED RFM DATA")
    print("=" * 70)
    
    df = pd.read_csv(filepath)
    
    print(f"✓ Loaded {len(df):,} customer records")
    print(f"✓ Columns: {list(df.columns)}")
    print(f"\n[Feature Statistics for Clustering]")
    print(df[Config.FEATURES].describe().round(4).to_string())
    print()
    
    return df


def scale_features(df: pd.DataFrame, features: list) -> tuple:
    print("=" * 70)
    print("STEP 2: FEATURE SCALING (StandardScaler)")
    print("=" * 70)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    
    print("\n[Scaling Transformation]")
    print(f"{'Feature':<20} {'Mean (pre)':<12} {'Std (pre)':<12} {'Mean (post)':<12} {'Std (post)':<12}")
    print("-" * 68)
    
    for i, feature in enumerate(features):
        pre_mean = df[feature].mean()
        pre_std = df[feature].std()
        post_mean = X_scaled[:, i].mean()
        post_std = X_scaled[:, i].std()
        print(f"{feature:<20} {pre_mean:<12.4f} {pre_std:<12.4f} {post_mean:<12.4f} {post_std:<12.4f}")
    
    print("\n✓ Features scaled to zero mean and unit variance")
    print()
    
    return X_scaled, scaler


def find_optimal_k(X_scaled: np.ndarray, k_range: range) -> dict:
    print("=" * 70)
    print("STEP 3: HYPERPARAMETER TUNING")
    print("=" * 70)
    
    results = {
        'k_values': list(k_range),
        'sse': [],
        'silhouette': []
    }
    
    print(f"\n[Evaluating k from {min(k_range)} to {max(k_range)}]")
    print(f"{'k':<5} {'SSE (Inertia)':<18} {'Silhouette Score':<18}")
    print("-" * 41)
    
    for k in k_range:
        kmeans = KMeans(
            n_clusters=k,
            random_state=Config.RANDOM_STATE,
            n_init=Config.N_INIT,
            max_iter=Config.MAX_ITER
        )
        kmeans.fit(X_scaled)
        
        sse = kmeans.inertia_
        silhouette = silhouette_score(X_scaled, kmeans.labels_)
        
        results['sse'].append(sse)
        results['silhouette'].append(silhouette)
        
        print(f"{k:<5} {sse:<18,.2f} {silhouette:<18.4f}")
    
    optimal_idx = np.argmax(results['silhouette'])
    optimal_k = results['k_values'][optimal_idx]
    
    print(f"\n[OPTIMAL K SELECTION]")
    print(f"  → Based on Silhouette Score: k = {optimal_k}")
    print(f"  → Silhouette Score: {results['silhouette'][optimal_idx]:.4f}")
    print(f"  → SSE at optimal k: {results['sse'][optimal_idx]:,.2f}")
    print()
    
    results['optimal_k'] = optimal_k
    
    return results


def plot_clustering_analysis(results: dict, save_path: str):
    print("=" * 70)
    print("STEP 4: VISUALIZATION")
    print("=" * 70)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    optimal_k = results['optimal_k']
    optimal_idx = results['k_values'].index(optimal_k)
    
    primary_color = '#1e3a5f'
    accent_color = '#e74c3c'
    background_color = '#f8f9fa'
    
    ax1 = axes[0]
    ax1.set_facecolor(background_color)
    
    ax1.plot(results['k_values'], results['sse'], 
             marker='o', markersize=8, linewidth=2.5,
             color=primary_color, label='SSE (Inertia)')
    
    ax1.scatter([optimal_k], [results['sse'][optimal_idx]], 
                s=200, color=accent_color, zorder=5, 
                edgecolors='white', linewidths=2,
                label=f'Optimal k={optimal_k}')
    
    ax1.axvline(x=optimal_k, color=accent_color, linestyle='--', 
                alpha=0.5, linewidth=1.5)
    
    ax1.set_xlabel('Number of Clusters (k)', fontweight='bold')
    ax1.set_ylabel('Sum of Squared Errors (SSE)', fontweight='bold')
    ax1.set_title('Elbow Method for Optimal k Selection', 
                  fontweight='bold', pad=15)
    ax1.set_xticks(results['k_values'])
    ax1.legend(loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    ax1.annotate(f'Elbow Point\nk = {optimal_k}',
                 xy=(optimal_k, results['sse'][optimal_idx]),
                 xytext=(optimal_k + 1.5, results['sse'][optimal_idx] * 1.1),
                 fontsize=10, ha='left',
                 arrowprops=dict(arrowstyle='->', color=accent_color, lw=1.5),
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                          edgecolor=accent_color, alpha=0.9))
    
    ax2 = axes[1]
    ax2.set_facecolor(background_color)
    
    bars = ax2.bar(results['k_values'], results['silhouette'],
                   color=primary_color, alpha=0.7, edgecolor='white', 
                   linewidth=1.5)
    
    bars[optimal_idx].set_color(accent_color)
    bars[optimal_idx].set_alpha(1.0)
    
    for i, (k, score) in enumerate(zip(results['k_values'], results['silhouette'])):
        ax2.text(k, score + 0.01, f'{score:.3f}', 
                 ha='center', va='bottom', fontsize=9,
                 fontweight='bold' if k == optimal_k else 'normal',
                 color=accent_color if k == optimal_k else primary_color)
    
    ax2.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, 
                linewidth=1.5, label='Good (≥0.5)')
    ax2.axhline(y=0.25, color='orange', linestyle='--', alpha=0.5, 
                linewidth=1.5, label='Fair (≥0.25)')
    
    ax2.set_xlabel('Number of Clusters (k)', fontweight='bold')
    ax2.set_ylabel('Silhouette Score', fontweight='bold')
    ax2.set_title('Silhouette Score Analysis', 
                  fontweight='bold', pad=15)
    ax2.set_xticks(results['k_values'])
    ax2.set_ylim(0, max(results['silhouette']) * 1.2)
    ax2.legend(loc='upper right', framealpha=0.9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    interpretation = f"Optimal k = {optimal_k}\nSilhouette = {results['silhouette'][optimal_idx]:.4f}"
    ax2.text(0.02, 0.98, interpretation, transform=ax2.transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                      edgecolor=accent_color, alpha=0.9),
             fontweight='bold', color=primary_color)
    
    fig.suptitle('K-Means Hyperparameter Tuning for RFM Customer Segmentation\n'
                 'RFM-Based Customer Segmentation Analysis', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✓ Visualization saved to:\n  {save_path}")
    print()


def fit_final_model(df: pd.DataFrame, X_scaled: np.ndarray, 
                    optimal_k: int) -> pd.DataFrame:
    print("=" * 70)
    print("STEP 5: FINAL MODEL FITTING")
    print("=" * 70)
    
    final_kmeans = KMeans(
        n_clusters=optimal_k,
        random_state=Config.RANDOM_STATE,
        n_init=Config.N_INIT,
        max_iter=Config.MAX_ITER
    )
    
    cluster_labels = final_kmeans.fit_predict(X_scaled)
    
    df_result = df.copy()
    df_result['Cluster'] = cluster_labels
    
    print(f"\n✓ K-Means fitted with k = {optimal_k}")
    print(f"✓ Final SSE (Inertia): {final_kmeans.inertia_:,.2f}")
    print(f"✓ Silhouette Score: {silhouette_score(X_scaled, cluster_labels):.4f}")
    
    print(f"\n[CLUSTER DISTRIBUTION]")
    cluster_counts = df_result['Cluster'].value_counts().sort_index()
    for cluster, count in cluster_counts.items():
        pct = count / len(df_result) * 100
        print(f"  Cluster {cluster}: {count:,} customers ({pct:.1f}%)")
    
    print()
    
    return df_result, final_kmeans


def profile_segments(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 70)
    print("STEP 6: SEGMENT PROFILING")
    print("=" * 70)
    
    profile = df.groupby('Cluster').agg({
        'CustomerID': 'count',
        'Recency': ['mean', 'median'],
        'Frequency': ['mean', 'median'],
        'Monetary': ['mean', 'median', 'sum']
    }).round(2)
    
    profile.columns = ['_'.join(col).strip() for col in profile.columns]
    profile = profile.rename(columns={'CustomerID_count': 'Customer_Count'})
    
    total_revenue = df['Monetary'].sum()
    profile['Revenue_Pct'] = (profile['Monetary_sum'] / total_revenue * 100).round(2)
    
    segment_names = []
    for cluster in profile.index:
        cluster_data = df[df['Cluster'] == cluster]
        avg_r = cluster_data['Recency'].mean()
        avg_f = cluster_data['Frequency'].mean()
        avg_m = cluster_data['Monetary'].mean()
        
        r_median = df['Recency'].median()
        f_median = df['Frequency'].median()
        m_median = df['Monetary'].median()
        
        if avg_r <= r_median and avg_f >= f_median and avg_m >= m_median:
            segment_names.append('Champions')
        elif avg_r <= r_median and avg_f >= f_median:
            segment_names.append('Loyal Customers')
        elif avg_r <= r_median and avg_m >= m_median:
            segment_names.append('Potential Loyalists')
        elif avg_r > r_median and avg_f >= f_median:
            segment_names.append('At Risk')
        elif avg_r > r_median and avg_m < m_median:
            segment_names.append('Hibernating')
        else:
            segment_names.append('Need Attention')
    
    profile['Segment_Name'] = segment_names
    
    print("\n[SEGMENT PROFILES]")
    print("-" * 90)
    
    display_cols = ['Segment_Name', 'Customer_Count', 'Recency_mean', 
                    'Frequency_mean', 'Monetary_mean', 'Revenue_Pct']
    
    for idx, row in profile.iterrows():
        print(f"\n  CLUSTER {idx}: {row['Segment_Name']}")
        print(f"  {'─' * 40}")
        print(f"  Customers:      {row['Customer_Count']:,} ({row['Customer_Count']/len(df)*100:.1f}%)")
        print(f"  Avg Recency:    {row['Recency_mean']:.0f} days")
        print(f"  Avg Frequency:  {row['Frequency_mean']:.1f} transactions")
        print(f"  Avg Monetary:   £{row['Monetary_mean']:,.2f}")
        print(f"  Revenue Share:  {row['Revenue_Pct']:.1f}%")
    
    print("\n" + "-" * 90)
    print()
    
    return profile


def main():
    print("\n" + "=" * 70)
    print("K-MEANS CUSTOMER SEGMENTATION")
    print("SEB Data Analyst Trainee Application")
    print("=" * 70 + "\n")
    
    df = load_rfm_data(Config.INPUT_FILE)
    X_scaled, scaler = scale_features(df, Config.FEATURES)
    tuning_results = find_optimal_k(X_scaled, Config.K_RANGE)
    plot_clustering_analysis(tuning_results, Config.PLOT_FILE)
    df_clustered, model = fit_final_model(df, X_scaled, tuning_results['optimal_k'])
    segment_profiles = profile_segments(df_clustered)
    
    print("=" * 70)
    print("STEP 7: OUTPUT GENERATION")
    print("=" * 70)
    
    df_clustered.to_csv(Config.OUTPUT_FILE, index=False)
    print(f"\n✓ Segmented customer data saved to:\n  {Config.OUTPUT_FILE}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"""
CLUSTERING RESULTS SUMMARY:
----------------------------
• Optimal k: {tuning_results['optimal_k']}
• Silhouette Score: {tuning_results['silhouette'][tuning_results['k_values'].index(tuning_results['optimal_k'])]:.4f}
• Total Customers Segmented: {len(df_clustered):,}

FILES GENERATED:
-----------------
• Visualization: {Config.PLOT_FILE}
• Segmented Data: {Config.OUTPUT_FILE}
""")
    
    return df_clustered, segment_profiles, tuning_results


if __name__ == "__main__":
    df_result, profiles, results = main()
