#!/usr/bin/env python3

import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class Config:
    INPUT_FILE = "/Users/mazinbashir/Downloads/online_retail_II.xlsx"
    OUTPUT_FILE = "/Users/mazinbashir/.gemini/antigravity/scratch/bank_rfm_analysis/rfm_cleaned_output.csv"
    OUTLIER_PERCENTILE = 99
    REFERENCE_DATE = datetime(2011, 12, 10)
    MIN_QUANTITY = 0
    MIN_UNIT_PRICE = 0


def load_data(filepath: str) -> pd.DataFrame:
    print("=" * 70)
    print("STEP 1: DATA LOADING")
    print("=" * 70)
    
    df = pd.read_excel(filepath, engine='openpyxl')
    
    print(f"✓ Loaded {len(df):,} records from dataset")
    print(f"✓ Columns: {list(df.columns)}")
    print(f"✓ Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
    print()
    
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 70)
    print("STEP 2: DATA CLEANING")
    print("=" * 70)
    
    initial_count = len(df)
    
    missing_customer_id = df['Customer ID'].isna().sum()
    print(f"\n[2.1] Missing CustomerID Analysis:")
    print(f"     - Records with missing CustomerID: {missing_customer_id:,} ({missing_customer_id/len(df)*100:.2f}%)")
    
    df = df.dropna(subset=['Customer ID'])
    print(f"     - Action: Removed {missing_customer_id:,} records")
    print(f"     - Rationale: CustomerID is required for RFM aggregation")
    
    print(f"\n[2.2] Date Format Conversion:")
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    print(f"     - InvoiceDate dtype: {df['InvoiceDate'].dtype}")
    print(f"     - Date range preserved: {df['InvoiceDate'].min().date()} to {df['InvoiceDate'].max().date()}")
    
    print(f"\n[2.3] Transaction Integrity - Cancellation Removal:")
    
    df['Invoice'] = df['Invoice'].astype(str)
    cancellations = df['Invoice'].str.startswith('C').sum()
    print(f"     - Cancelled transactions found: {cancellations:,}")
    
    df = df[~df['Invoice'].str.startswith('C')]
    print(f"     - Action: Removed {cancellations:,} cancelled transactions")
    print(f"     - Rationale: Cancellations would distort spending patterns")
    
    print(f"\n[2.4] Data Quality - Positive Values Enforcement:")
    
    invalid_quantity = (df['Quantity'] <= Config.MIN_QUANTITY).sum()
    invalid_price = (df['Price'] <= Config.MIN_UNIT_PRICE).sum()
    
    print(f"     - Records with Quantity ≤ 0: {invalid_quantity:,}")
    print(f"     - Records with Price ≤ 0: {invalid_price:,}")
    
    df = df[(df['Quantity'] > Config.MIN_QUANTITY) & (df['Price'] > Config.MIN_UNIT_PRICE)]
    
    print(f"     - Action: Removed invalid quantity/price records")
    print(f"     - Rationale: Ensures only valid sales are counted")
    
    final_count = len(df)
    print(f"\n[CLEANING SUMMARY]")
    print(f"     - Initial records: {initial_count:,}")
    print(f"     - Final records: {final_count:,}")
    print(f"     - Data retention rate: {final_count/initial_count*100:.2f}%")
    print()
    
    return df


def create_rfm_features(df: pd.DataFrame, reference_date: datetime) -> pd.DataFrame:
    print("=" * 70)
    print("STEP 3: FEATURE ENGINEERING")
    print("=" * 70)
    
    print("\n[3.1] Transaction-Level Feature:")
    df['TotalSpend'] = df['Quantity'] * df['Price']
    print(f"     - Created TotalSpend = Quantity × Price")
    print(f"     - Total revenue: £{df['TotalSpend'].sum():,.2f}")
    
    print("\n[3.2] Customer-Level RFM Aggregation:")
    print(f"     - Reference date for recency: {reference_date.date()}")
    
    rfm = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,
        'Invoice': 'nunique',
        'TotalSpend': 'sum'
    }).reset_index()
    
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    
    print(f"     - Unique customers analyzed: {len(rfm):,}")
    print(f"\n[RFM Statistics]")
    print(rfm[['Recency', 'Frequency', 'Monetary']].describe().round(2).to_string())
    print()
    
    return rfm


def prepare_for_clustering(rfm: pd.DataFrame, percentile: int = 99) -> pd.DataFrame:
    print("=" * 70)
    print("STEP 4: STATISTICAL PREPARATION FOR CLUSTERING")
    print("=" * 70)
    
    rfm_processed = rfm.copy()
    features = ['Recency', 'Frequency', 'Monetary']
    
    print(f"\n[4.1] Outlier Treatment (Capping at {percentile}th Percentile):")
    
    for feature in features:
        cap_value = rfm_processed[feature].quantile(percentile / 100)
        outliers_count = (rfm_processed[feature] > cap_value).sum()
        rfm_processed[feature] = rfm_processed[feature].clip(upper=cap_value)
        print(f"     - {feature}: Capped {outliers_count:,} outliers at {cap_value:,.2f}")
    
    print(f"\n[4.2] Log Transformation (log1p for zero-safe computation):")
    
    for feature in features:
        original_skew = rfm_processed[feature].skew()
        rfm_processed[f'{feature}_Log'] = np.log1p(rfm_processed[feature])
        transformed_skew = rfm_processed[f'{feature}_Log'].skew()
        print(f"     - {feature}: Skewness {original_skew:.2f} → {transformed_skew:.2f}")
    
    print("\n[POST-PROCESSING STATISTICS]")
    print(rfm_processed[['Recency_Log', 'Frequency_Log', 'Monetary_Log']].describe().round(4).to_string())
    print()
    
    return rfm_processed


def main():
    print("\n" + "=" * 70)
    print("RFM CUSTOMER SEGMENTATION ANALYSIS")
    print("SEB Data Analyst Trainee Application")
    print("=" * 70 + "\n")
    
    df = load_data(Config.INPUT_FILE)
    df_clean = clean_data(df)
    rfm = create_rfm_features(df_clean, Config.REFERENCE_DATE)
    rfm_final = prepare_for_clustering(rfm, Config.OUTLIER_PERCENTILE)
    
    print("=" * 70)
    print("STEP 5: OUTPUT GENERATION")
    print("=" * 70)
    
    rfm_final.to_csv(Config.OUTPUT_FILE, index=False)
    print(f"\n✓ Final RFM dataframe saved to:\n  {Config.OUTPUT_FILE}")
    
    print("\n[SAMPLE OUTPUT - TOP 10 CUSTOMERS BY MONETARY VALUE]")
    print("-" * 70)
    top_customers = rfm_final.nlargest(10, 'Monetary')[
        ['CustomerID', 'Recency', 'Frequency', 'Monetary', 
         'Recency_Log', 'Frequency_Log', 'Monetary_Log']
    ]
    print(top_customers.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"""
KEY INSIGHTS FOR SEB BANKING CONTEXT:
--------------------------------------
1. CUSTOMER BASE: {len(rfm_final):,} unique customers analyzed
2. TOTAL REVENUE: £{rfm['Monetary'].sum():,.2f}
3. AVG FREQUENCY: {rfm['Frequency'].mean():.1f} transactions per customer
4. AVG RECENCY: {rfm['Recency'].mean():.0f} days since last purchase
""")
    
    return rfm_final


if __name__ == "__main__":
    rfm_result = main()
