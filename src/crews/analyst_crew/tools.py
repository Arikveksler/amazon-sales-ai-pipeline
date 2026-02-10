"""
כלים לניקוי נתונים - tools.py
פונקציות עזר לעיבוד DataFrame
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
from loguru import logger


def clean_missing_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    טיפול בערכים חסרים (null, NaN, empty strings)

    Args:
        df: DataFrame לניקוי

    Returns:
        Tuple[DataFrame מעודכן, Dict של סטטיסטיקות]
    """
    stats = {
        "missing_before": df.isnull().sum().to_dict(),
        "filled_values": {}
    }

    # טיפול בעמודות מספריות - מילוי ב-median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            filled_count = df[col].isnull().sum()
            df[col] = df[col].fillna(median_val)
            stats["filled_values"][col] = {
                "count": int(filled_count),
                "method": "median",
                "value": float(median_val)
            }
            logger.info(f"מולאו {filled_count} ערכים חסרים ב-{col} עם median: {median_val:.2f}")

    # טיפול בעמודות קטגוריאליות - מילוי ב-mode או "Unknown"
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            filled_count = df[col].isnull().sum()
            if df[col].mode().size > 0:
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                stats["filled_values"][col] = {
                    "count": int(filled_count),
                    "method": "mode",
                    "value": str(mode_val)
                }
            else:
                df[col] = df[col].fillna("Unknown")
                stats["filled_values"][col] = {
                    "count": int(filled_count),
                    "method": "default",
                    "value": "Unknown"
                }
            logger.info(f"מולאו {filled_count} ערכים חסרים ב-{col}")

    return df, stats


def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    זיהוי והסרת שורות כפולות

    Args:
        df: DataFrame לניקוי

    Returns:
        Tuple[DataFrame מעודכן, Dict של סטטיסטיקות]
    """
    rows_before = len(df)
    df_cleaned = df.drop_duplicates()
    rows_after = len(df_cleaned)
    duplicates_removed = rows_before - rows_after

    stats = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "duplicates_removed": duplicates_removed,
        "percentage": round((duplicates_removed / rows_before) * 100, 2) if rows_before > 0 else 0
    }

    logger.info(f"הוסרו {duplicates_removed} שורות כפולות ({stats['percentage']}% מהנתונים)")

    return df_cleaned, stats


def handle_outliers(df: pd.DataFrame, columns: list = None) -> Tuple[pd.DataFrame, Dict]:
    """
    זיהוי וטיפול בערכים חריגים (IQR method)

    Args:
        df: DataFrame לניקוי
        columns: רשימת עמודות לבדיקה (None = כל העמודות המספריות)

    Returns:
        Tuple[DataFrame מעודכן, Dict של סטטיסטיקות]
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    stats = {}

    for col in columns:
        if col not in df.columns or df[col].dtype not in [np.int64, np.float64]:
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        outliers_count = outliers_mask.sum()

        if outliers_count > 0:
            # Capping - מגביל ל-99th percentile
            percentile_99 = df[col].quantile(0.99)
            df.loc[df[col] > percentile_99, col] = percentile_99

            stats[col] = {
                "outliers_detected": int(outliers_count),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "capped_to": float(percentile_99)
            }

            logger.info(f"זוהו {outliers_count} ערכים חריגים ב-{col}, הוגבלו ל-{percentile_99:.2f}")

    return df, stats


def standardize_text(df: pd.DataFrame, columns: list = None) -> Tuple[pd.DataFrame, Dict]:
    """
    תקנון טקסט - הסרת רווחים מיותרים ו-lowercase

    Args:
        df: DataFrame לניקוי
        columns: רשימת עמודות לבדיקה (None = כל עמודות הטקסט)

    Returns:
        Tuple[DataFrame מעודכן, Dict של סטטיסטיקות]
    """
    if columns is None:
        columns = df.select_dtypes(include=['object']).columns.tolist()

    stats = {"columns_processed": []}

    for col in columns:
        if col in df.columns and df[col].dtype == 'object':
            # הסרת רווחים מיותרים
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].str.replace(r'\s+', ' ', regex=True)

            stats["columns_processed"].append(col)
            logger.info(f"תוקנו ערכי טקסט ב-{col}")

    return df, stats


def validate_data_types(df: pd.DataFrame, schema: dict = None) -> Tuple[pd.DataFrame, Dict]:
    """
    בדיקה ותיקון של סוגי נתונים

    Args:
        df: DataFrame לבדיקה
        schema: מילון של עמודות וסוגי הנתונים הצפויים (אופציונלי)

    Returns:
        Tuple[DataFrame מעודכן, Dict של סטטיסטיקות]
    """
    stats = {
        "original_dtypes": df.dtypes.astype(str).to_dict(),
        "conversions": {}
    }

    if schema:
        for col, expected_type in schema.items():
            if col in df.columns:
                try:
                    if expected_type == "float" or expected_type == "numeric":
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        stats["conversions"][col] = f"converted to numeric"
                    elif expected_type == "int":
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                        stats["conversions"][col] = f"converted to integer"
                    elif expected_type == "string":
                        df[col] = df[col].astype(str)
                        stats["conversions"][col] = f"converted to string"
                except Exception as e:
                    logger.warning(f"לא ניתן להמיר {col} ל-{expected_type}: {str(e)}")

    stats["final_dtypes"] = df.dtypes.astype(str).to_dict()
    logger.info(f"בוצעו {len(stats['conversions'])} המרות סוג נתונים")

    return df, stats
