import pandas as pd
from pathlib import Path

# Get the path relative to this file's location
data_path = Path(__file__).parent / 'raw' / 'amazon_sales.csv'
df = pd.read_csv(data_path)
print(df.info())
print(df.head())
print(f"Total rows: {len(df)}")

