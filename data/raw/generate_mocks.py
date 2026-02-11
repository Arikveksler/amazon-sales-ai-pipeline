import os

# 1. יצירת תוכן דמו לדוח התובנות
markdown_content = """
# Business Insights Report (MOCK DATA)

## Executive Summary
This is a placeholder report generated to test the Streamlit UI.
The actual Analyst Crew analysis will appear here in production.

## Key Findings
* **Price Sensitivity:** High correlation between discounts and sales volume.
* **Category Performance:** 'Electronics' is the leading category.
* **Customer Satisfaction:** Average rating is 4.2 stars.

## Recommendations
1. Increase stock for high-rated items.
2. Optimize discount strategy for low-performing categories.
"""

# 2. יצירת תוכן דמו לדוח הגרפים (HTML)
html_content = """
<html>
<head><title>EDA Report Mock</title></head>
<body style="font-family: Arial; padding: 20px;">
    <h1 style="color: #2e86de;">EDA Report (Mock Visualization)</h1>
    <div style="background-color: #f1f2f6; padding: 20px; border-radius: 10px;">
        <h3>Price Distribution Chart</h3>
        <p>[Placeholder for Matplotlib/Seaborn Chart]</p>
        <div style="height: 200px; background-color: #ddd; display: flex; align-items: center; justify-content: center;">
            Chart Image Would Go Here
        </div>
    </div>
    <br>
    <div style="background-color: #f1f2f6; padding: 20px; border-radius: 10px;">
        <h3>Rating Analysis</h3>
        <p>Average Rating: 4.2 / 5.0</p>
    </div>
</body>
</html>
"""

# שמירת הקבצים בתיקייה הראשית
with open("insights.md", "w", encoding="utf-8") as f:
    f.write(markdown_content)
    print("✅ Created insights.md")

with open("eda_report.html", "w", encoding="utf-8") as f:
    f.write(html_content)
    print("✅ Created eda_report.html")