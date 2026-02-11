"""
Unit Tests for EDA Tools
Tests the generate_eda_report function to ensure it creates valid HTML output.
"""

import unittest
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.tools.eda_tools import generate_eda_report


class TestEDATools(unittest.TestCase):
    """Test cases for EDA Tools functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures - runs once before all tests."""
        cls.csv_path = os.path.join(project_root, 'data', 'amazon.csv')
        cls.expected_html_path = os.path.join(project_root, 'data', 'eda_report.html')

    def test_csv_file_exists(self):
        """Test that the input CSV file exists."""
        self.assertTrue(
            os.path.exists(self.csv_path),
            f"CSV file not found at: {self.csv_path}"
        )

    def test_generate_eda_report_creates_html_file(self):
        """Test that generate_eda_report creates an HTML file."""
        # Remove existing HTML file if it exists
        if os.path.exists(self.expected_html_path):
            os.remove(self.expected_html_path)

        # Run the EDA report generation
        result = generate_eda_report.run(self.csv_path)

        # Assert the HTML file was created
        self.assertTrue(
            os.path.exists(self.expected_html_path),
            f"HTML report was not created at: {self.expected_html_path}"
        )

    def test_html_file_not_empty(self):
        """Test that the generated HTML file is not empty."""
        # Ensure the file exists (run generation if needed)
        if not os.path.exists(self.expected_html_path):
            generate_eda_report.run(self.csv_path)

        # Check file size
        file_size = os.path.getsize(self.expected_html_path)
        self.assertGreater(
            file_size, 0,
            "HTML report file is empty"
        )

    def test_html_file_contains_valid_content(self):
        """Test that the HTML file contains expected content."""
        # Ensure the file exists
        if not os.path.exists(self.expected_html_path):
            generate_eda_report.run(self.csv_path)

        # Read and check content
        with open(self.expected_html_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for basic HTML structure
        self.assertIn('<!DOCTYPE html>', content, "Missing DOCTYPE declaration")
        self.assertIn('<html', content, "Missing html tag")
        self.assertIn('EDA Report', content, "Missing report title")
        self.assertIn('data:image/png;base64', content, "Missing embedded charts")

    def test_generate_eda_report_returns_summary(self):
        """Test that generate_eda_report returns a statistical summary."""
        result = generate_eda_report.run(self.csv_path)

        # Check that result contains expected summary sections
        self.assertIn('STATISTICAL SUMMARY', result, "Missing statistical summary")
        self.assertIn('PRICE ANALYSIS', result, "Missing price analysis")
        self.assertIn('CATEGORY ANALYSIS', result, "Missing category analysis")
        self.assertIn('RATING ANALYSIS', result, "Missing rating analysis")


if __name__ == '__main__':
    unittest.main(verbosity=2)
