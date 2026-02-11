"""
Unit Tests for Feature Engineering Tools
"""

import unittest
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.tools.feature_engineering import (
    clean_price_column,
    clean_rating_count,
    clean_discount_percentage,
    extract_category_features,
    extract_text_length_features,
    compute_derived_features,
    engineer_features,
    FEATURE_COLUMNS,
)


class TestCleaningHelpers(unittest.TestCase):
    """Test the cleaning helper functions."""

    def test_clean_price_column(self):
        """Test ₹ and comma removal from prices."""
        s = pd.Series(['₹1,099', '₹399', '₹10,999'])
        result = clean_price_column(s)
        self.assertEqual(list(result), [1099.0, 399.0, 10999.0])

    def test_clean_price_already_numeric(self):
        """Test that already-numeric values pass through."""
        s = pd.Series([399.0, 1099.0])
        result = clean_price_column(s)
        self.assertEqual(list(result), [399.0, 1099.0])

    def test_clean_rating_count(self):
        """Test comma removal from rating counts."""
        s = pd.Series(['24,269', '43,994'])
        result = clean_rating_count(s)
        self.assertEqual(list(result), [24269.0, 43994.0])

    def test_clean_discount_percentage(self):
        """Test % removal from discount percentages."""
        s = pd.Series(['64%', '43%', '90%'])
        result = clean_discount_percentage(s)
        self.assertEqual(list(result), [64.0, 43.0, 90.0])


class TestFeatureExtraction(unittest.TestCase):
    """Test feature extraction functions."""

    def test_extract_category_features(self):
        """Test category splitting and label encoding."""
        df = pd.DataFrame({
            'category': [
                'Computers&Accessories|Cables|USBCables',
                'Electronics|Headphones',
                'Home&Kitchen',
            ]
        })
        result, meta = extract_category_features(df)

        self.assertIn('main_category_encoded', result.columns)
        self.assertIn('sub_category_encoded', result.columns)
        self.assertIn('category_depth', result.columns)
        self.assertEqual(list(result['category_depth']), [3, 2, 1])
        self.assertEqual(len(meta['main_category_classes']), 3)

    def test_extract_text_length_features(self):
        """Test title and description length extraction."""
        df = pd.DataFrame({
            'product_name': ['Short', 'A much longer product name'],
            'about_product': ['Desc', 'Longer description here!!'],
        })
        result = extract_text_length_features(df)

        self.assertEqual(result['title_length'].iloc[0], 5)
        self.assertEqual(result['title_length'].iloc[1], 26)
        self.assertEqual(result['desc_length'].iloc[0], 4)
        self.assertEqual(result['desc_length'].iloc[1], 25)

    def test_compute_derived_features(self):
        """Test discount_amount and price_ratio computation."""
        df = pd.DataFrame({
            'actual_price': [1000.0, 500.0],
            'discounted_price': [600.0, 250.0],
        })
        result = compute_derived_features(df)

        self.assertEqual(result['discount_amount'].iloc[0], 400.0)
        self.assertEqual(result['discount_amount'].iloc[1], 250.0)
        self.assertAlmostEqual(result['price_ratio'].iloc[0], 0.6)
        self.assertAlmostEqual(result['price_ratio'].iloc[1], 0.5)


class TestFullPipeline(unittest.TestCase):
    """Test the full engineer_features pipeline."""

    def test_full_pipeline_creates_features_csv(self):
        """Test that the pipeline creates a valid features.csv."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mini CSV matching the real data format
            df = pd.DataFrame({
                'product_id': ['B001', 'B002', 'B003'],
                'product_name': ['Test Product One', 'Another Product', 'Third Item'],
                'category': [
                    'Electronics|Audio|Headphones',
                    'Home&Kitchen|Appliances',
                    'Electronics|Cables',
                ],
                'discounted_price': ['₹399', '₹1,099', '₹199'],
                'actual_price': ['₹999', '₹1,599', '₹349'],
                'discount_percentage': ['60%', '31%', '43%'],
                'rating': ['4.2', '3.8', '4.5'],
                'rating_count': ['24,269', '5,000', '10,500'],
                'about_product': ['Great product for testing', 'Nice item', 'Good cable'],
                'user_id': ['U1', 'U2', 'U3'],
                'user_name': ['User1', 'User2', 'User3'],
                'review_id': ['R1', 'R2', 'R3'],
                'review_title': ['Good', 'OK', 'Great'],
                'review_content': ['Content1', 'Content2', 'Content3'],
                'img_link': ['http://img1', 'http://img2', 'http://img3'],
                'product_link': ['http://l1', 'http://l2', 'http://l3'],
            })
            input_path = os.path.join(tmpdir, 'test_input.csv')
            output_path = os.path.join(tmpdir, 'features.csv')
            df.to_csv(input_path, index=False)

            # Run the pipeline
            features, metadata = engineer_features(input_path, output_path)

            # Assertions
            self.assertEqual(len(features), 3)
            self.assertEqual(len(features.columns), 12)
            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)

            # All columns should be numeric
            for col in features.columns:
                self.assertTrue(
                    features[col].dtype in ['float64', 'int64', 'int32', 'float32'],
                    f"Column {col} is {features[col].dtype}, expected numeric"
                )

            # Column names should match FEATURE_COLUMNS
            self.assertEqual(list(features.columns), FEATURE_COLUMNS)

            # Metadata should have category classes
            self.assertIn('main_category_classes', metadata)
            self.assertIn('feature_columns', metadata)
            self.assertEqual(metadata['num_features'], 12)


if __name__ == '__main__':
    unittest.main(verbosity=2)
