"""
Smoke Test - Verify that all critical project files exist.
Covers deliverables from Week 1, 2, and 3.
"""

import unittest
import os
import sys

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)


class TestProjectStructure(unittest.TestCase):
    """Verify that all required project files are in place."""

    # -- Week 1: Analyst Crew --

    def test_main_py_exists(self):
        """main.py entry point exists in project root."""
        self.assertTrue(os.path.exists(os.path.join(PROJECT_ROOT, 'main.py')))

    def test_eda_tools_exists(self):
        """EDA tools module exists."""
        self.assertTrue(os.path.exists(os.path.join(PROJECT_ROOT, 'src', 'tools', 'eda_tools.py')))

    def test_analyst_crew_exists(self):
        """Analyst crew module exists."""
        self.assertTrue(os.path.exists(os.path.join(PROJECT_ROOT, 'src', 'crews', 'analyst_crew.py')))

    def test_test_eda_exists(self):
        """EDA unit tests exist."""
        self.assertTrue(os.path.exists(os.path.join(PROJECT_ROOT, 'tests', 'test_eda.py')))

    # -- Week 2: Feature Engineering --

    def test_features_csv_exists(self):
        """Generated features.csv exists in data/features/."""
        self.assertTrue(os.path.exists(os.path.join(PROJECT_ROOT, 'data', 'features', 'features.csv')))

    # -- Week 3: Streamlit UI --

    def test_app_py_exists(self):
        """Streamlit app exists at src/app.py."""
        self.assertTrue(os.path.exists(os.path.join(PROJECT_ROOT, 'src', 'app.py')))

    # -- Core project files --

    def test_requirements_exists(self):
        """requirements.txt exists."""
        self.assertTrue(os.path.exists(os.path.join(PROJECT_ROOT, 'requirements.txt')))

    def test_amazon_csv_exists(self):
        """Amazon dataset exists in data/."""
        self.assertTrue(os.path.exists(os.path.join(PROJECT_ROOT, 'data', 'amazon.csv')))

    def test_readme_exists(self):
        """README.md exists."""
        self.assertTrue(os.path.exists(os.path.join(PROJECT_ROOT, 'README.md')))


if __name__ == '__main__':
    unittest.main(verbosity=2)
