"""
Feature Engineering Tools for Amazon Sales Pipeline
Transforms raw Amazon product data into numerical features for ML models.
"""

import os
from pathlib import Path
from typing import Tuple, Dict, Any

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# 1. Cleaning helpers (same pattern as eda_tools.py:preprocess_dataframe)
# ---------------------------------------------------------------------------

def clean_price_column(series: pd.Series) -> pd.Series:
    """Convert price strings like '₹1,099' to float 1099.0."""
    return (
        series
        .astype(str)
        .str.replace('₹', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.strip()
        .pipe(pd.to_numeric, errors='coerce')
    )


def clean_rating_count(series: pd.Series) -> pd.Series:
    """Convert rating_count strings like '24,269' to float 24269.0."""
    return (
        series
        .astype(str)
        .str.replace(',', '', regex=False)
        .str.strip()
        .pipe(pd.to_numeric, errors='coerce')
    )


def clean_discount_percentage(series: pd.Series) -> pd.Series:
    """Convert discount_percentage strings like '64%' to float 64.0."""
    return (
        series
        .astype(str)
        .str.replace('%', '', regex=False)
        .str.strip()
        .pipe(pd.to_numeric, errors='coerce')
    )


# ---------------------------------------------------------------------------
# 2. Feature extraction functions
# ---------------------------------------------------------------------------

def extract_category_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Split the pipe-delimited 'category' column into main_category and
    sub_category, then Label-Encode both. Also compute category_depth.

    Returns:
        (df with new columns, metadata dict with encoder mappings)
    """
    df = df.copy()

    # Split category string
    df['main_category'] = df['category'].apply(
        lambda x: str(x).split('|')[0] if pd.notna(x) else 'Unknown'
    )
    df['sub_category'] = df['category'].apply(
        lambda x: str(x).split('|')[1]
        if pd.notna(x) and len(str(x).split('|')) > 1
        else 'Unknown'
    )
    df['category_depth'] = df['category'].apply(
        lambda x: len(str(x).split('|')) if pd.notna(x) else 0
    )

    # Label encode
    le_main = LabelEncoder()
    le_sub = LabelEncoder()

    df['main_category_encoded'] = le_main.fit_transform(df['main_category'])
    df['sub_category_encoded'] = le_sub.fit_transform(df['sub_category'])

    metadata = {
        'main_category_classes': list(le_main.classes_),
        'sub_category_classes': list(le_sub.classes_),
    }

    print(f"Encoded {len(le_main.classes_)} main categories, "
          f"{len(le_sub.classes_)} sub categories")

    return df, metadata


def extract_text_length_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract title_length and desc_length from text columns."""
    df = df.copy()
    df['title_length'] = df['product_name'].astype(str).str.len()
    df['desc_length'] = df['about_product'].astype(str).str.len()
    return df


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute discount_amount and price_ratio from price columns."""
    df = df.copy()
    df['discount_amount'] = df['actual_price'] - df['discounted_price']
    df['price_ratio'] = (
        df['discounted_price'] / df['actual_price']
    ).replace([np.inf, -np.inf], np.nan)
    return df


# ---------------------------------------------------------------------------
# 3. Main orchestrator
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    'discounted_price',
    'actual_price',
    'discount_percentage',
    'rating',
    'rating_count',
    'main_category_encoded',
    'sub_category_encoded',
    'category_depth',
    'title_length',
    'desc_length',
    'discount_amount',
    'price_ratio',
]


def engineer_features(
    input_path: str,
    output_path: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Full feature engineering pipeline.

    Args:
        input_path:  Path to input CSV (amazon.csv or clean_data.csv)
        output_path: Path to save the numerical features.csv

    Returns:
        (features DataFrame, metadata dict)
    """
    print(f"Loading data from {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Step 1: Clean numeric columns
    df['discounted_price'] = clean_price_column(df['discounted_price'])
    df['actual_price'] = clean_price_column(df['actual_price'])
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['rating_count'] = clean_rating_count(df['rating_count'])
    df['discount_percentage'] = clean_discount_percentage(df['discount_percentage'])

    # Step 2: Category features
    df, cat_metadata = extract_category_features(df)

    # Step 3: Text length features
    df = extract_text_length_features(df)

    # Step 4: Derived features
    df = compute_derived_features(df)

    # Step 5: Select only numerical columns
    features_df = df[FEATURE_COLUMNS].copy()

    # Step 6: Fill remaining NaN with median
    nan_count = features_df.isnull().sum().sum()
    features_df = features_df.fillna(features_df.median(numeric_only=True))
    print(f"Filled {nan_count} NaN values with median")

    # Step 7: Save
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)
    print(f"Saved features to {output_path}: "
          f"{len(features_df)} rows x {len(features_df.columns)} columns")

    metadata = {
        **cat_metadata,
        'feature_columns': FEATURE_COLUMNS,
        'num_rows': len(features_df),
        'num_features': len(FEATURE_COLUMNS),
    }

    return features_df, metadata


# ---------------------------------------------------------------------------
# 4. Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    INPUT_CSV = os.path.join("data", "amazon.csv")
    OUTPUT_CSV = os.path.join("data", "features", "features.csv")

    features, meta = engineer_features(INPUT_CSV, OUTPUT_CSV)
    print(f"\nFeature engineering complete!")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Shape: {features.shape}")
    print(f"\nColumns: {meta['feature_columns']}")
    print(f"Main categories: {meta['main_category_classes']}")
