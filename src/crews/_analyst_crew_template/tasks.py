"""
הגדרת משימות של Analyst Crew - tasks.py
משימה ראשונה: ניקוי נתונים גולמיים ע"י Data Cleaner Agent
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from crewai import Task, Agent

# טעינת משתני סביבה מ-.env
env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(env_path)


def create_clean_data_task(agent: Agent, raw_data_path: str, clean_data_path: str) -> Task:
    """
    יצירת משימת ניקוי נתונים

    המשימה מנחה את ה-Agent לבצע את כל שלבי הניקוי בסדר הנכון:
    1. בדיקת ותיקון סוגי נתונים (מחירים, אחוזים, דירוגים)
    2. ניקוי ערכים חסרים
    3. הסרת שורות כפולות
    4. זיהוי וטיפול ב-outliers
    5. תקנון טקסט

    Args:
        agent: ה-Data Cleaner Agent שיבצע את המשימה
        raw_data_path: נתיב לקובץ הנתונים הגולמיים
        clean_data_path: נתיב לשמירת הנתונים הנקיים

    Returns:
        Task מוגדר לניקוי נתונים
    """
    return Task(
        description=(
            f"Clean the raw Amazon sales dataset located at '{raw_data_path}'. "
            f"Follow these steps IN ORDER:\n\n"
            f"Step 1: Use the validate_data_types_tool on '{raw_data_path}' to convert "
            f"price columns (discounted_price, actual_price) from strings with currency "
            f"symbols to float, discount_percentage from string with % to float, "
            f"rating to float, and rating_count to integer.\n\n"
            f"Step 2: Use the clean_missing_values_tool on '{raw_data_path}' to handle "
            f"any null, NaN, or empty string values.\n\n"
            f"Step 3: Use the remove_duplicates_tool on '{raw_data_path}' to detect and "
            f"remove any duplicate rows.\n\n"
            f"Step 4: Use the handle_outliers_tool on '{raw_data_path}' "
            f"to detect and handle outliers in numeric columns using the IQR method.\n\n"
            f"Step 5: Use the standardize_text_tool on '{raw_data_path}' to clean up "
            f"text columns by removing extra whitespace and standardizing case.\n\n"
            f"After all steps are complete, provide a comprehensive data quality report "
            f"summarizing all changes made to the dataset."
        ),

        expected_output=(
            "A comprehensive data quality report containing:\n"
            "1. Data types validation results - which columns were converted and to what type\n"
            "2. Missing values summary - how many nulls were found and how they were filled\n"
            "3. Duplicates summary - how many duplicate rows were found and removed\n"
            "4. Outliers summary - which columns had outliers and how they were handled\n"
            "5. Text standardization summary - which columns were cleaned\n"
            "6. Final dataset shape (rows x columns)\n"
            "7. Overall data quality assessment"
        ),

        agent=agent,
    )


def create_analyst_tasks(agents: list, raw_data_path: str, clean_data_path: str) -> list:
    """
    יצירת כל המשימות של ה-Analyst Crew

    Args:
        agents: רשימת אג'נטים (agents[0] = data_cleaner)
        raw_data_path: נתיב לנתונים הגולמיים
        clean_data_path: נתיב לשמירת נתונים נקיים

    Returns:
        רשימת משימות
    """
    data_cleaner = agents[0]

    clean_task = create_clean_data_task(
        agent=data_cleaner,
        raw_data_path=raw_data_path,
        clean_data_path=clean_data_path,
    )

    return [clean_task]
