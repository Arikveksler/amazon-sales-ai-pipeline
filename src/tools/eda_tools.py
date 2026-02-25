"""
EDA Tools for CrewAI Agent
Tools for performing Exploratory Data Analysis on datasets
"""

import os
import base64
from io import BytesIO
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from crewai.tools import tool


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the Amazon dataset:
    1. Clean price columns (remove ₹ and commas, convert to float)
    2. Clean rating column (convert to numeric)
    3. Extract main category
    """
    df = df.copy()

    # Clean discounted_price: '₹1,299' -> 1299.0
    df['discounted_price'] = (
        df['discounted_price']
        .astype(str)
        .str.replace('₹', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.strip()
    )
    df['discounted_price'] = pd.to_numeric(df['discounted_price'], errors='coerce')

    # Clean actual_price: '₹1,299' -> 1299.0
    df['actual_price'] = (
        df['actual_price']
        .astype(str)
        .str.replace('₹', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.strip()
    )
    df['actual_price'] = pd.to_numeric(df['actual_price'], errors='coerce')

    # Clean rating column
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

    # Extract main category (first part before '|')
    df['main_category'] = df['category'].apply(
        lambda x: str(x).split('|')[0] if pd.notna(x) else 'Unknown'
    )

    return df


def fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string for HTML embedding."""
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig)
    return img_base64


def generate_statistical_summary(df: pd.DataFrame) -> str:
    """
    Generate a statistical summary string for the agent to understand the data.
    """
    # Price statistics
    avg_price = df['discounted_price'].mean()
    median_price = df['discounted_price'].median()
    min_price = df['discounted_price'].min()
    max_price = df['discounted_price'].max()

    # Most expensive product
    most_expensive_idx = df['discounted_price'].idxmax()
    most_expensive_product = df.loc[most_expensive_idx, 'product_name'][:100]  # Truncate long names
    most_expensive_price = df.loc[most_expensive_idx, 'discounted_price']

    # Cheapest product
    cheapest_idx = df['discounted_price'].idxmin()
    cheapest_product = df.loc[cheapest_idx, 'product_name'][:100]
    cheapest_price = df.loc[cheapest_idx, 'discounted_price']

    # Category statistics
    top_category = df['main_category'].value_counts().index[0]
    top_category_count = df['main_category'].value_counts().iloc[0]
    total_categories = df['main_category'].nunique()

    # Rating statistics
    avg_rating = df['rating'].mean()
    highest_rated_idx = df['rating'].idxmax()
    highest_rated_product = df.loc[highest_rated_idx, 'product_name'][:100]
    highest_rating = df.loc[highest_rated_idx, 'rating']

    # Discount analysis
    df['discount_amount'] = df['actual_price'] - df['discounted_price']
    df['discount_pct'] = (df['discount_amount'] / df['actual_price'] * 100).fillna(0)
    avg_discount_pct = df['discount_pct'].mean()
    max_discount_idx = df['discount_pct'].idxmax()
    max_discount_product = df.loc[max_discount_idx, 'product_name'][:100]
    max_discount_pct = df.loc[max_discount_idx, 'discount_pct']

    # Price-Rating correlation
    correlation = df['discounted_price'].corr(df['rating'])

    summary = f"""
=== STATISTICAL SUMMARY ===

DATASET OVERVIEW:
- Total products: {len(df):,}
- Total categories: {total_categories}

PRICE ANALYSIS:
- Average price: {avg_price:,.2f} INR
- Median price: {median_price:,.2f} INR
- Price range: {min_price:,.2f} - {max_price:,.2f} INR
- Most expensive product: "{most_expensive_product}" at {most_expensive_price:,.2f} INR
- Cheapest product: "{cheapest_product}" at {cheapest_price:,.2f} INR

CATEGORY ANALYSIS:
- Most popular category: "{top_category}" with {top_category_count:,} products
- This category represents {top_category_count/len(df)*100:.1f}% of all products

RATING ANALYSIS:
- Average rating: {avg_rating:.2f} out of 5
- Highest rated product: "{highest_rated_product}" with rating {highest_rating}

DISCOUNT ANALYSIS:
- Average discount: {avg_discount_pct:.1f}%
- Highest discount: "{max_discount_product}" with {max_discount_pct:.1f}% off

CORRELATION INSIGHTS:
- Price-Rating correlation: {correlation:.3f}
- {"Weak" if abs(correlation) < 0.3 else "Moderate" if abs(correlation) < 0.7 else "Strong"} {"positive" if correlation > 0 else "negative"} correlation between price and rating

=== END OF SUMMARY ===
"""
    return summary


@tool("Generate EDA Report")
def generate_eda_report(file_path: str) -> str:
    """
    Generate an EDA report with visualizations from a CSV file.
    Creates 3 charts: price distribution, top 10 categories, and price vs rating correlation.
    Saves the report as an HTML file and returns a statistical summary.

    Args:
        file_path: Path to the CSV file to analyze

    Returns:
        A string containing the HTML report path AND a statistical summary for insights generation
    """
    # Set style for all plots
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)

    # Load and preprocess the dataset
    df_raw = pd.read_csv(file_path)
    df = preprocess_dataframe(df_raw)

    # Verify preprocessing worked
    print(f"Preprocessing complete:")
    print(f"  - discounted_price dtype: {df['discounted_price'].dtype}")
    print(f"  - actual_price dtype: {df['actual_price'].dtype}")
    print(f"  - rating dtype: {df['rating'].dtype}")
    print(f"  - Sample discounted_price values: {df['discounted_price'].head(3).tolist()}")

    charts = []

    # ============================================
    # Chart 1: Price Distribution (Histogram)
    # ============================================
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    price_data = df['discounted_price'].dropna()

    sns.histplot(price_data, bins=50, kde=True, color='steelblue', ax=ax1)
    ax1.set_title('Distribution of Discounted Prices', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Price (INR)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)

    # Add statistics annotation (values are now numeric floats)
    mean_price = price_data.mean()
    median_price = price_data.median()
    std_price = price_data.std()

    stats_text = f'Mean: {mean_price:,.0f}\nMedian: {median_price:,.0f}\nStd: {std_price:,.0f}'
    ax1.text(0.95, 0.95, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    charts.append({
        'title': 'Price Distribution',
        'description': 'Histogram showing the distribution of discounted prices (in INR) with KDE curve',
        'image': fig_to_base64(fig1)
    })

    # ============================================
    # Chart 2: Top 10 Categories (Bar Chart)
    # ============================================
    fig2, ax2 = plt.subplots(figsize=(12, 6))

    category_counts = df['main_category'].value_counts().head(10)
    colors = sns.color_palette('viridis', len(category_counts))

    bars = ax2.barh(category_counts.index, category_counts.values, color=colors)
    ax2.set_title('Top 10 Product Categories', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Number of Products', fontsize=12)
    ax2.set_ylabel('Category', fontsize=12)
    ax2.invert_yaxis()

    # Add value labels on bars
    for bar, value in zip(bars, category_counts.values):
        ax2.text(value + 0.5, bar.get_y() + bar.get_height()/2,
                 f'{value:,}', va='center', fontsize=10)

    plt.tight_layout()
    charts.append({
        'title': 'Top 10 Categories',
        'description': 'Bar chart showing the most common product categories',
        'image': fig_to_base64(fig2)
    })

    # ============================================
    # Chart 3: Price vs Rating (Scatter Plot)
    # ============================================
    fig3, ax3 = plt.subplots(figsize=(10, 6))

    # Filter valid data for scatter plot (both price and rating must be numeric)
    scatter_df = df[['discounted_price', 'rating']].dropna()

    sns.scatterplot(
        data=scatter_df,
        x='discounted_price',
        y='rating',
        alpha=0.5,
        color='coral',
        ax=ax3
    )

    # Add trend line
    sns.regplot(
        data=scatter_df,
        x='discounted_price',
        y='rating',
        scatter=False,
        color='darkred',
        ax=ax3
    )

    ax3.set_title('Price vs Rating Correlation', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Discounted Price (INR)', fontsize=12)
    ax3.set_ylabel('Rating', fontsize=12)
    ax3.set_ylim(0, 5.5)

    # Calculate and display correlation
    correlation = scatter_df['discounted_price'].corr(scatter_df['rating'])
    ax3.text(0.95, 0.05, f'Correlation: {correlation:.3f}', transform=ax3.transAxes,
             fontsize=11, verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    charts.append({
        'title': 'Price vs Rating',
        'description': 'Scatter plot showing correlation between price and product rating',
        'image': fig_to_base64(fig3)
    })

    # ============================================
    # Generate HTML Report
    # ============================================
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Calculate summary statistics (all numeric now)
    total_products = len(df)
    num_categories = df['main_category'].nunique()
    avg_price = price_data.mean()
    avg_rating = df['rating'].mean()

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EDA Report - Amazon Products</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .summary h2 {{
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .stat-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-card .value {{
            font-size: 2em;
            color: #667eea;
            font-weight: bold;
        }}
        .stat-card .label {{
            color: #666;
            margin-top: 5px;
        }}
        .chart-section {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .chart-section h3 {{
            color: #333;
            margin-top: 0;
        }}
        .chart-section p {{
            color: #666;
            margin-bottom: 15px;
        }}
        .chart-section img {{
            max-width: 100%;
            height: auto;
            border-radius: 5px;
        }}
        .footer {{
            text-align: center;
            color: #666;
            margin-top: 30px;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>EDA Report</h1>
        <p>Amazon Products Dataset Analysis</p>
        <p>Generated: {timestamp}</p>
    </div>

    <div class="summary">
        <h2>Dataset Summary</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="value">{total_products:,}</div>
                <div class="label">Total Products</div>
            </div>
            <div class="stat-card">
                <div class="value">{num_categories}</div>
                <div class="label">Categories</div>
            </div>
            <div class="stat-card">
                <div class="value">{avg_price:,.0f} INR</div>
                <div class="label">Avg Price</div>
            </div>
            <div class="stat-card">
                <div class="value">{avg_rating:.2f}</div>
                <div class="label">Avg Rating</div>
            </div>
        </div>
    </div>
"""

    # Add each chart section
    for chart in charts:
        html_content += f"""
    <div class="chart-section">
        <h3>{chart['title']}</h3>
        <p>{chart['description']}</p>
        <img src="data:image/png;base64,{chart['image']}" alt="{chart['title']}">
    </div>
"""

    html_content += """
    <div class="footer">
        <p>Generated by CrewAI EDA Agent</p>
    </div>
</body>
</html>
"""

    # Save HTML file
    output_dir = os.path.dirname(file_path)
    output_path = os.path.join(output_dir, 'eda_report.html')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    # Generate statistical summary for the agent
    statistical_summary = generate_statistical_summary(df)

    # Return both the path and the summary
    result = f"""
EDA Report generated successfully!

HTML Report saved at: {output_path}

{statistical_summary}

Use the above statistics to write business insights in the insights.md file.
"""
    return result
