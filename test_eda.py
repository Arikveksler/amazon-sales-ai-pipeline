"""
Test script for EDA Tools
Run this to verify the generate_eda_report tool works correctly
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tools.eda_tools import generate_eda_report

if __name__ == "__main__":
    # Path to the CSV file
    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'amazon.csv')

    print("=" * 50)
    print("Testing EDA Report Generation")
    print("=" * 50)
    print(f"\nInput file: {csv_path}")
    print("\nGenerating report...")

    # Run the tool
    result = generate_eda_report.run(csv_path)

    print(f"\n{result}")
    print("\n" + "=" * 50)
    print("Test completed!")
    print("=" * 50)
