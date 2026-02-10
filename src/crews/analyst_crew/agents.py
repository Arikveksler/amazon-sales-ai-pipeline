"""
הגדרת אג'נטים של Analyst Crew - agents.py
Agent ראשון: Data Cleaner - ניקוי וטרנספורמציה של נתונים גולמיים
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from crewai import Agent
from crewai.tools import tool
from .tools import (
    clean_missing_values,
    remove_duplicates,
    handle_outliers,
    standardize_text,
    validate_data_types,
)

# טעינת משתני סביבה מ-.env
env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(env_path)


# ─────────────────────────────────────────────
# עטיפת הפונקציות ככלי CrewAI
# ─────────────────────────────────────────────

@tool("clean_missing_values_tool")
def clean_missing_values_tool(file_path: str) -> str:
    """Clean missing values (null, NaN, empty strings) in a CSV file.
    Numeric columns are filled with median, categorical with mode or 'Unknown'.
    Input: file_path - path to the CSV file."""
    import pandas as pd
    df = pd.read_csv(file_path)
    df, stats = clean_missing_values(df)
    df.to_csv(file_path, index=False)
    return f"Missing values cleaned. Stats: {stats}"


@tool("remove_duplicates_tool")
def remove_duplicates_tool(file_path: str) -> str:
    """Detect and remove duplicate rows from a CSV file.
    Input: file_path - path to the CSV file."""
    import pandas as pd
    df = pd.read_csv(file_path)
    df, stats = remove_duplicates(df)
    df.to_csv(file_path, index=False)
    return f"Duplicates removed. Stats: {stats}"


@tool("handle_outliers_tool")
def handle_outliers_tool(file_path: str) -> str:
    """Detect and handle outliers in numeric columns using IQR method with capping.
    Input: file_path - path to the CSV file."""
    import pandas as pd
    df = pd.read_csv(file_path)
    df, stats = handle_outliers(df)
    df.to_csv(file_path, index=False)
    return f"Outliers handled. Stats: {stats}"


@tool("standardize_text_tool")
def standardize_text_tool(file_path: str) -> str:
    """Standardize text columns - strip whitespace and normalize spacing.
    Input: file_path - path to the CSV file."""
    import pandas as pd
    df = pd.read_csv(file_path)
    df, stats = standardize_text(df)
    df.to_csv(file_path, index=False)
    return f"Text standardized. Stats: {stats}"


@tool("validate_data_types_tool")
def validate_data_types_tool(file_path: str) -> str:
    """Validate and fix data types in a CSV file. Converts prices to float,
    percentages to float, ratings to float, and rating_count to integer.
    Input: file_path - path to the CSV file."""
    import pandas as pd
    schema = {
        "discounted_price": "float",
        "actual_price": "float",
        "discount_percentage": "float",
        "rating": "float",
        "rating_count": "int",
    }
    df = pd.read_csv(file_path)
    # ניקוי סימנים מיוחדים לפני המרה
    for col in ["discounted_price", "actual_price"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
    if "discount_percentage" in df.columns:
        df["discount_percentage"] = df["discount_percentage"].astype(str).str.replace(r'[%]', '', regex=True)
    if "rating_count" in df.columns:
        df["rating_count"] = df["rating_count"].astype(str).str.replace(r'[,\s]', '', regex=True)
    df, stats = validate_data_types(df, schema)
    df.to_csv(file_path, index=False)
    return f"Data types validated. Stats: {stats}"


# ─────────────────────────────────────────────
# יצירת ה-Agent
# ─────────────────────────────────────────────

def create_data_cleaner_agent() -> Agent:
    """
    יצירת Data Cleaner Agent

    Agent זה אחראי על:
    - ניקוי ערכים חסרים (null, NaN, empty strings)
    - הסרת שורות כפולות
    - זיהוי וטיפול ב-outliers (IQR method)
    - תקנון טקסט (רווחים, lowercase)
    - בדיקת ותיקון סוגי נתונים (מחירים, אחוזים, דירוגים)

    Returns:
        Agent מוגדר עם כלי ניקוי נתונים
    """
    return Agent(
        role="Data Cleaning Specialist",

        goal=(
            "Clean and prepare raw Amazon sales data for analysis by handling "
            "missing values, removing duplicates, and ensuring data quality"
        ),

        backstory=(
            "You are an expert data engineer with years of experience in data "
            "preprocessing. You have a keen eye for data quality issues and know "
            "exactly how to transform messy datasets into clean, analysis-ready "
            "data. You understand the importance of maintaining data integrity "
            "while removing noise and inconsistencies."
        ),

        tools=[
            validate_data_types_tool,
            clean_missing_values_tool,
            remove_duplicates_tool,
            handle_outliers_tool,
            standardize_text_tool,
        ],

        verbose=True,

        allow_delegation=False,
    )


def create_analyst_agents() -> list:
    """
    יצירת כל האג'נטים של ה-Analyst Crew

    Returns:
        רשימת אג'נטים
    """
    data_cleaner = create_data_cleaner_agent()
    return [data_cleaner]
