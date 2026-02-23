"""
Feature Engineering Module for Scientist Crew
==============================================
מודול הנדסת פיצ'רים לחיזוי שיעור הנחה אופטימלי למוצרי אמזון.

This module provides functions for:
- Type conversions (string → numeric)
- Derived feature creation
- Text feature extraction
- Category encoding
- Product-level aggregation
- Feature validation

Author: ML Specialist
Date: 2026-02-06
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from loguru import logger
import re
import json


# ============================================================================
# Type Conversion Functions
# ============================================================================

def convert_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    המרת עמודות מחיר מstring לfloat.
    Convert price columns from string to float.

    Handles:
    - Currency symbols (₹, $, etc.)
    - Commas in numbers (1,999 → 1999)
    - Missing values

    Args:
        df: DataFrame with price columns as strings

    Returns:
        DataFrame with numeric price columns
    """
    logger.info("Converting price columns to numeric...")

    price_columns = ['actual_price', 'discounted_price']

    for col in price_columns:
        if col in df.columns:
            # Remove currency symbols and commas
            df[col] = df[col].astype(str).str.replace('₹', '', regex=False)
            df[col] = df[col].str.replace('$', '', regex=False)
            df[col] = df[col].str.replace(',', '', regex=False)
            df[col] = df[col].str.strip()

            # Convert to float
            df[col] = pd.to_numeric(df[col], errors='coerce')

            # Fill missing values with median
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

            logger.info(f"  ✓ Converted {col}: {df[col].dtype}, range: [{df[col].min():.2f}, {df[col].max():.2f}]")

    return df


def convert_rating_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    המרת עמודות דירוג מstring לnumeric.
    Convert rating columns from string to numeric.

    Handles:
    - Rating (0-5 scale)
    - Rating count (remove commas)
    - Missing values

    Args:
        df: DataFrame with rating columns as strings

    Returns:
        DataFrame with numeric rating columns
    """
    logger.info("Converting rating columns to numeric...")

    # Convert rating
    if 'rating' in df.columns:
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

        # Fill missing with median
        median_rating = df['rating'].median()
        df['rating'] = df['rating'].fillna(median_rating)

        # Clip to valid range [0, 5]
        df['rating'] = df['rating'].clip(0, 5)

        logger.info(f"  ✓ Converted rating: {df['rating'].dtype}, range: [{df['rating'].min():.2f}, {df['rating'].max():.2f}]")

    # Convert rating_count
    if 'rating_count' in df.columns:
        # Remove commas
        df['rating_count'] = df['rating_count'].astype(str).str.replace(',', '', regex=False)
        df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')

        # Fill missing with 0
        df['rating_count'] = df['rating_count'].fillna(0)

        # Ensure non-negative
        df['rating_count'] = df['rating_count'].clip(lower=0)

        logger.info(f"  ✓ Converted rating_count: {df['rating_count'].dtype}, range: [{df['rating_count'].min():.0f}, {df['rating_count'].max():.0f}]")

    return df


def convert_discount_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    המרת עמודת אחוז הנחה מstring לfloat.
    Convert discount_percentage from string to float.

    This is the TARGET VARIABLE for prediction.

    Handles:
    - Percentage symbols (%)
    - Missing values
    - Outliers

    Args:
        df: DataFrame with discount_percentage as string

    Returns:
        DataFrame with numeric discount_percentage
    """
    logger.info("Converting discount_percentage (TARGET) to numeric...")

    if 'discount_percentage' in df.columns:
        # Remove % symbol
        df['discount_percentage'] = df['discount_percentage'].astype(str).str.replace('%', '', regex=False)
        df['discount_percentage'] = df['discount_percentage'].str.strip()

        # Convert to float
        df['discount_percentage'] = pd.to_numeric(df['discount_percentage'], errors='coerce')

        # Fill missing with median
        median_discount = df['discount_percentage'].median()
        df['discount_percentage'] = df['discount_percentage'].fillna(median_discount)

        # Clip to valid range [0, 100]
        df['discount_percentage'] = df['discount_percentage'].clip(0, 100)

        logger.info(f"  ✓ Converted discount_percentage (TARGET): {df['discount_percentage'].dtype}, range: [{df['discount_percentage'].min():.2f}%, {df['discount_percentage'].max():.2f}%]")

    return df


# ============================================================================
# Derived Feature Creation
# ============================================================================

def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    יצירת פיצ'רים נגזרים מהעמודות הקיימות.
    Create derived features from existing columns.

    Creates:
    - price_level: log1p(actual_price) - normalized price scale
    - discounted_price_level: log1p(discounted_price)
    - log_rating_count: log1p(rating_count) - handle skewness
    - rating_weighted: rating × log1p(rating_count) - popularity × quality
    - is_highly_rated: 1 if rating >= 4.0 else 0
    - reviews_per_rating: rating_count / (rating + 0.1)
    - has_many_reviews: 1 if rating_count > median else 0

    Args:
        df: DataFrame with base features

    Returns:
        DataFrame with additional derived features
    """
    logger.info("Creating derived features...")

    # Price-based features
    if 'actual_price' in df.columns:
        df['price_level'] = np.log1p(df['actual_price'])
        logger.info("  ✓ Created price_level (log-transformed)")

    if 'discounted_price' in df.columns:
        df['discounted_price_level'] = np.log1p(df['discounted_price'])
        logger.info("  ✓ Created discounted_price_level (log-transformed)")

    # Rating-based features
    if 'rating_count' in df.columns:
        df['log_rating_count'] = np.log1p(df['rating_count'])
        logger.info("  ✓ Created log_rating_count")

    if 'rating' in df.columns and 'rating_count' in df.columns:
        df['rating_weighted'] = df['rating'] * np.log1p(df['rating_count'])
        logger.info("  ✓ Created rating_weighted (popularity × quality)")

    if 'rating' in df.columns:
        df['is_highly_rated'] = (df['rating'] >= 4.0).astype(int)
        logger.info("  ✓ Created is_highly_rated (threshold feature)")

    # Engagement features
    if 'rating_count' in df.columns and 'rating' in df.columns:
        df['reviews_per_rating'] = df['rating_count'] / (df['rating'] + 0.1)
        logger.info("  ✓ Created reviews_per_rating (engagement rate)")

    if 'rating_count' in df.columns:
        median_count = df['rating_count'].median()
        df['has_many_reviews'] = (df['rating_count'] > median_count).astype(int)
        logger.info(f"  ✓ Created has_many_reviews (median={median_count:.0f})")

    return df


# ============================================================================
# Text Feature Extraction
# ============================================================================

def extract_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    חילוץ פיצ'רים מעמודות טקסט.
    Extract features from text columns.

    Creates:
    - description_length: length of about_product
    - description_word_count: word count in about_product
    - has_premium_keywords: contains premium, quality, best, luxury
    - has_tech_keywords: contains wireless, smart, digital, bluetooth

    Args:
        df: DataFrame with text columns

    Returns:
        DataFrame with text-based features
    """
    logger.info("Extracting text features...")

    # Description features
    if 'about_product' in df.columns:
        df['about_product'] = df['about_product'].fillna('')

        df['description_length'] = df['about_product'].str.len()
        df['description_word_count'] = df['about_product'].str.split().str.len()

        # Premium keywords
        premium_keywords = ['premium', 'quality', 'best', 'luxury', 'high-quality', 'excellent']
        df['has_premium_keywords'] = df['about_product'].str.lower().str.contains(
            '|'.join(premium_keywords), regex=True, na=False
        ).astype(int)

        # Tech keywords
        tech_keywords = ['wireless', 'smart', 'digital', 'bluetooth', 'usb', 'electronic']
        df['has_tech_keywords'] = df['about_product'].str.lower().str.contains(
            '|'.join(tech_keywords), regex=True, na=False
        ).astype(int)

        logger.info("  ✓ Created description_length, description_word_count")
        logger.info("  ✓ Created has_premium_keywords, has_tech_keywords")

    return df


def extract_review_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    חילוץ פיצ'רים מביקורות (ברמת review).
    Extract features from reviews (review-level).

    Creates:
    - review_length: length of review_content
    - review_word_count: word count in review_content
    - review_sentiment_score: positive - negative words
    - has_positive_review: 1 if sentiment > 0

    Note: These will be aggregated to product-level later.

    Args:
        df: DataFrame with review columns

    Returns:
        DataFrame with review-based features
    """
    logger.info("Extracting review features...")

    if 'review_content' in df.columns:
        df['review_content'] = df['review_content'].fillna('')

        # Length features
        df['review_length'] = df['review_content'].str.len()
        df['review_word_count'] = df['review_content'].str.split().str.len()

        # Simple sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'best', 'awesome']
        negative_words = ['bad', 'poor', 'worst', 'terrible', 'hate', 'awful', 'disappointed']

        # Count positive/negative words
        positive_count = df['review_content'].str.lower().apply(
            lambda x: sum(word in x for word in positive_words)
        )
        negative_count = df['review_content'].str.lower().apply(
            lambda x: sum(word in x for word in negative_words)
        )

        df['review_sentiment_score'] = positive_count - negative_count
        df['has_positive_review'] = (df['review_sentiment_score'] > 0).astype(int)

        logger.info("  ✓ Created review_length, review_word_count")
        logger.info("  ✓ Created review_sentiment_score, has_positive_review")

    return df


# ============================================================================
# Category Encoding
# ============================================================================

def encode_categories(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    קידוד קטגוריות באמצעות One-Hot Encoding.
    Encode categories using One-Hot Encoding.

    Keeps top N categories, groups others as "Other".

    Args:
        df: DataFrame with category column
        top_n: Number of top categories to keep (default: 10)

    Returns:
        DataFrame with one-hot encoded category columns
    """
    logger.info(f"Encoding categories (top {top_n})...")

    if 'category' not in df.columns:
        logger.warning("  ⚠ No 'category' column found, skipping encoding")
        return df

    # Get top N categories
    category_counts = df['category'].value_counts()
    top_categories = category_counts.head(top_n).index.tolist()

    logger.info(f"  Top categories: {top_categories}")

    # Create "Other" category for rare ones
    df['category_grouped'] = df['category'].apply(
        lambda x: x if x in top_categories else 'Other'
    )

    # One-hot encoding
    category_dummies = pd.get_dummies(df['category_grouped'], prefix='category')

    # Add to dataframe
    df = pd.concat([df, category_dummies], axis=1)

    logger.info(f"  ✓ Created {len(category_dummies.columns)} category columns")

    # Drop temporary column
    df = df.drop('category_grouped', axis=1)

    return df


# ============================================================================
# Product-Level Aggregation
# ============================================================================

def aggregate_product_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    אגרגציה של features ברמת מוצר.
    Aggregate features to product-level.

    Since the dataset has multiple reviews per product, we aggregate:
    - Review features: mean, std, count
    - Product features: first value (same for all reviews)

    Args:
        df: DataFrame with review-level features

    Returns:
        DataFrame aggregated to product-level
    """
    logger.info("Aggregating features to product-level...")

    if 'product_id' not in df.columns:
        logger.warning("  ⚠ No 'product_id' column, skipping aggregation")
        return df

    # Define aggregation functions
    agg_dict = {}

    # Product features (take first - they're the same for all reviews)
    product_cols = [
        'actual_price', 'discounted_price', 'discount_percentage',
        'rating', 'rating_count',
        'price_level', 'discounted_price_level',
        'log_rating_count', 'rating_weighted', 'is_highly_rated',
        'reviews_per_rating', 'has_many_reviews',
        'description_length', 'description_word_count',
        'has_premium_keywords', 'has_tech_keywords'
    ]

    for col in product_cols:
        if col in df.columns:
            agg_dict[col] = 'first'

    # Review features (aggregate - mean, std, count)
    review_cols = ['review_length', 'review_word_count', 'review_sentiment_score', 'has_positive_review']

    for col in review_cols:
        if col in df.columns:
            agg_dict[f'{col}_mean'] = (col, 'mean')
            agg_dict[f'{col}_std'] = (col, 'std')
            agg_dict[f'{col}_count'] = (col, 'count')

    # Category features (take first)
    category_cols = [col for col in df.columns if col.startswith('category_')]
    for col in category_cols:
        agg_dict[col] = 'first'

    # Aggregate
    df_agg = df.groupby('product_id').agg(agg_dict).reset_index()

    # Flatten column names for multi-level aggregation
    df_agg.columns = [col[0] if isinstance(col, tuple) else col for col in df_agg.columns]

    # Fill NaN in std columns (single review → std=0)
    std_cols = [col for col in df_agg.columns if col.endswith('_std')]
    for col in std_cols:
        df_agg[col] = df_agg[col].fillna(0)

    logger.info(f"  ✓ Aggregated from {len(df)} reviews to {len(df_agg)} products")
    logger.info(f"  ✓ Final features: {len(df_agg.columns)} columns")

    return df_agg


# ============================================================================
# Feature Validation
# ============================================================================

def validate_features(df: pd.DataFrame, contract: Dict = None) -> Tuple[bool, str]:
    """
    ולידציה של features שנוצרו.
    Validate engineered features.

    Checks:
    - No missing values
    - Correct data types (numeric)
    - Valid value ranges
    - Target variable exists

    Args:
        df: DataFrame with engineered features
        contract: Data contract dictionary (optional)

    Returns:
        (is_valid, message)
    """
    logger.info("Validating engineered features...")

    # Check 1: No missing values
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        null_cols = null_counts[null_counts > 0].to_dict()
        msg = f"Found missing values: {null_cols}"
        logger.error(f"  ✗ {msg}")
        return False, msg

    logger.info("  ✓ No missing values")

    # Check 2: Target variable exists
    if 'discount_percentage' not in df.columns:
        msg = "Target variable 'discount_percentage' not found"
        logger.error(f"  ✗ {msg}")
        return False, msg

    logger.info("  ✓ Target variable 'discount_percentage' present")

    # Check 3: Numeric types
    non_numeric_cols = df.select_dtypes(include=['object']).columns.tolist()
    # Exclude product_id if present
    non_numeric_cols = [col for col in non_numeric_cols if col != 'product_id']

    if len(non_numeric_cols) > 0:
        msg = f"Found non-numeric columns: {non_numeric_cols}"
        logger.error(f"  ✗ {msg}")
        return False, msg

    logger.info("  ✓ All features are numeric")

    # Check 4: Valid ranges
    if 'discount_percentage' in df.columns:
        if (df['discount_percentage'] < 0).any() or (df['discount_percentage'] > 100).any():
            msg = "discount_percentage has values outside [0, 100] range"
            logger.error(f"  ✗ {msg}")
            return False, msg

    if 'rating' in df.columns:
        if (df['rating'] < 0).any() or (df['rating'] > 5).any():
            msg = "rating has values outside [0, 5] range"
            logger.error(f"  ✗ {msg}")
            return False, msg

    logger.info("  ✓ All values in valid ranges")

    # Success
    msg = f"Feature validation passed: {len(df)} rows, {len(df.columns)} columns"
    logger.success(f"✓ {msg}")

    return True, msg


# ============================================================================
# Main Feature Engineering Pipeline
# ============================================================================

def engineer_features(
    df: pd.DataFrame,
    contract: Dict = None,
    top_categories: int = 10
) -> pd.DataFrame:
    """
    פיפליין מלא של הנדסת פיצ'רים.
    Complete feature engineering pipeline.

    Steps:
    1. Type conversions (prices, ratings, discount)
    2. Derived features (logs, ratios, thresholds)
    3. Text features (description, reviews)
    4. Category encoding (one-hot)
    5. Product-level aggregation
    6. Validation

    Args:
        df: Raw DataFrame from clean_data.csv
        contract: Data contract (optional)
        top_categories: Number of top categories for encoding

    Returns:
        DataFrame with engineered features ready for ML
    """
    logger.info("="*60)
    logger.info("Starting Feature Engineering Pipeline")
    logger.info("="*60)

    # Step 1: Type conversions
    df = convert_price_columns(df)
    df = convert_rating_columns(df)
    df = convert_discount_column(df)

    # Step 2: Derived features
    df = create_derived_features(df)

    # Step 3: Text features
    df = extract_text_features(df)
    df = extract_review_features(df)

    # Step 4: Category encoding
    df = encode_categories(df, top_n=top_categories)

    # Step 5: Product-level aggregation
    df = aggregate_product_level(df)

    # Step 6: Validation
    is_valid, msg = validate_features(df, contract)
    if not is_valid:
        raise ValueError(f"Feature validation failed: {msg}")

    logger.info("="*60)
    logger.success("Feature Engineering Pipeline Complete!")
    logger.info(f"Final shape: {df.shape}")
    logger.info("="*60)

    return df


def save_features(
    df: pd.DataFrame,
    features_path: str,
    metadata_path: str = None
) -> bool:
    """
    שמירת features וmetadata.
    Save engineered features and metadata.

    Args:
        df: DataFrame with features
        features_path: Path to save features.csv
        metadata_path: Path to save metadata JSON (optional)

    Returns:
        True if successful
    """
    logger.info(f"Saving features to {features_path}...")

    # Save features CSV
    df.to_csv(features_path, index=False)
    logger.success(f"✓ Features saved: {features_path}")

    # Save metadata if path provided
    if metadata_path:
        metadata = {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "statistics": {
                "discount_percentage": {
                    "mean": float(df['discount_percentage'].mean()),
                    "std": float(df['discount_percentage'].std()),
                    "min": float(df['discount_percentage'].min()),
                    "max": float(df['discount_percentage'].max())
                }
            },
            "feature_count": len(df.columns) - 1,  # exclude target
            "target_variable": "discount_percentage"
        }

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.success(f"✓ Metadata saved: {metadata_path}")

    return True


# ============================================================================
# Self-Test
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Feature Engineering Module - Self Test")
    print("="*60 + "\n")

    # Create sample data
    sample_data = pd.DataFrame({
        'product_id': ['P1', 'P1', 'P2'],
        'actual_price': ['₹2,999', '₹2,999', '₹1,499'],
        'discounted_price': ['₹1,999', '₹1,999', '₹999'],
        'discount_percentage': ['33%', '33%', '33%'],
        'rating': ['4.5', '4.5', '4.0'],
        'rating_count': ['1,234', '1,234', '567'],
        'category': ['Electronics', 'Electronics', 'Home'],
        'about_product': ['Premium wireless headphones', 'Premium wireless headphones', 'Smart home device'],
        'review_content': ['Great product!', 'Love it!', 'Good value']
    })

    print("Sample data created (3 rows)\n")

    # Run feature engineering
    features = engineer_features(sample_data, top_categories=5)

    print(f"\nFinal features shape: {features.shape}")
    print(f"Feature columns: {features.columns.tolist()}")

    print("\n" + "="*60)
    print("Self test complete!")
    print("="*60 + "\n")
