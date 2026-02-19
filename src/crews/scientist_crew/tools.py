"""
כלים של Scientist Crew - הנדסת תכונות ואימון מודלים
Scientist Crew tools - Feature Engineering and Model Training

מימוש: נווה (מדען נתונים - ML)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder


# =============================================================================
# שלב 1: ניקוי ערכים מספריים
# =============================================================================

def clean_numeric_column(series: pd.Series) -> pd.Series:
    """
    ניקוי עמודה מספרית - הסרת סימני מטבע, פסיקים, ותווי אחוז.

    Args:
        series: עמודת pandas לניקוי

    Returns:
        עמודה מספרית נקייה
    """
    cleaned = series.astype(str).str.replace("₹", "", regex=False)
    cleaned = cleaned.str.replace(",", "", regex=False)
    cleaned = cleaned.str.replace("%", "", regex=False)
    cleaned = cleaned.str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


# =============================================================================
# שלב 2: הנדסת תכונות (Feature Engineering)
# =============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    יצירת תכונות חדשות מהנתונים הנקיים.

    תכונות חדשות:
    - price_to_rating_ratio: יחס מחיר לדירוג
    - is_discounted: האם המוצר במבצע
    - discount_pct: אחוז הנחה (מספרי נקי)
    - category_encoded: קטגוריה מקודדת
    - popularity_level: רמת פופולריות לפי כמות ביקורות

    Args:
        df: DataFrame עם נתונים נקיים

    Returns:
        DataFrame עם תכונות חדשות
    """
    logger.info("מתחיל הנדסת תכונות...")
    features = df.copy()

    # --- ניקוי עמודות מספריות ---
    features["discounted_price_clean"] = clean_numeric_column(features["discounted_price"])
    features["actual_price_clean"] = clean_numeric_column(features["actual_price"])
    features["discount_pct"] = clean_numeric_column(features["discount_percentage"])
    features["rating_clean"] = clean_numeric_column(features["rating"])
    features["rating_count_clean"] = clean_numeric_column(features["rating_count"])

    # הסרת שורות עם ערכים חסרים בעמודות המספריות
    numeric_cols = [
        "discounted_price_clean", "actual_price_clean",
        "discount_pct", "rating_clean", "rating_count_clean",
    ]
    before_count = len(features)
    features = features.dropna(subset=numeric_cols)
    after_count = len(features)
    logger.info(f"הוסרו {before_count - after_count} שורות עם ערכים חסרים")

    # --- תכונה 1: יחס מחיר לדירוג ---
    features["price_to_rating_ratio"] = (
        features["discounted_price_clean"] / features["rating_clean"].replace(0, np.nan)
    ).fillna(0)

    # --- תכונה 2: האם המוצר במבצע ---
    features["is_discounted"] = (
        features["actual_price_clean"] > features["discounted_price_clean"]
    ).astype(int)

    # --- תכונה 3: קטגוריה מקודדת ---
    # שימוש בקטגוריה הראשית (לפני ה-| הראשון)
    features["main_category"] = features["category"].astype(str).str.split("|").str[0].str.strip()
    le = LabelEncoder()
    features["category_encoded"] = le.fit_transform(features["main_category"])

    # --- תכונה 4: רמת פופולריות ---
    features["popularity_level"] = pd.cut(
        features["rating_count_clean"],
        bins=[0, 100, 1000, 10000, float("inf")],
        labels=[0, 1, 2, 3],  # 0=נמוך, 1=בינוני, 2=גבוה, 3=מאוד גבוה
    ).astype(int)

    logger.info(f"הנדסת תכונות הושלמה. {len(features)} שורות, {len(features.columns)} עמודות")
    return features


# =============================================================================
# שלב 3: אימון מודלים
# =============================================================================

def prepare_model_data(features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    הכנת נתונים לאימון - בחירת תכונות ומשתנה מטרה.
    משתנה מטרה: discounted_price_clean (חיזוי מחיר מוצר)

    Args:
        features_df: DataFrame עם תכונות מהונדסות

    Returns:
        X (תכונות), y (מטרה)
    """
    feature_columns = [
        "actual_price_clean",
        "discount_pct",
        "rating_clean",
        "rating_count_clean",
        "price_to_rating_ratio",
        "is_discounted",
        "category_encoded",
        "popularity_level",
    ]

    X = features_df[feature_columns].copy()
    y = features_df["discounted_price_clean"].copy()

    return X, y


def train_models(X_train, y_train, X_test, y_test) -> Dict[str, Any]:
    """
    אימון שני מודלים והשוואה ביניהם.

    מודלים:
    1. LinearRegression
    2. RandomForestRegressor

    Args:
        X_train, y_train: נתוני אימון
        X_test, y_test: נתוני מבחן

    Returns:
        מילון עם תוצאות כל המודלים והמודל הטוב ביותר
    """
    results = {}

    # --- מודל 1: Linear Regression ---
    logger.info("מאמן מודל Linear Regression...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)

    results["LinearRegression"] = {
        "model": lr_model,
        "predictions": lr_pred,
        "mae": mean_absolute_error(y_test, lr_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, lr_pred)),
        "r2": r2_score(y_test, lr_pred),
    }
    logger.info(f"  MAE={results['LinearRegression']['mae']:.2f}, R²={results['LinearRegression']['r2']:.4f}")

    # --- מודל 2: Random Forest Regressor ---
    logger.info("מאמן מודל Random Forest Regressor...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    results["RandomForestRegressor"] = {
        "model": rf_model,
        "predictions": rf_pred,
        "mae": mean_absolute_error(y_test, rf_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, rf_pred)),
        "r2": r2_score(y_test, rf_pred),
    }
    logger.info(f"  MAE={results['RandomForestRegressor']['mae']:.2f}, R²={results['RandomForestRegressor']['r2']:.4f}")

    # --- בחירת המודל הטוב ביותר ---
    best_name = min(results, key=lambda k: results[k]["mae"])
    results["best_model_name"] = best_name
    results["best_model"] = results[best_name]["model"]

    logger.info(f"המודל הטוב ביותר: {best_name}")

    return results


def get_scientist_tools() -> list:
    """
    קבלת רשימת כלים למדען נתונים

    Returns:
        רשימת כלים
    """
    return []
