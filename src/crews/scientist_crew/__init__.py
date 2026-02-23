# Scientist Crew Package
from .agents import create_scientist_agents
from .tasks import create_scientist_tasks
from .tools import (
    engineer_features,
    prepare_model_data,
    train_models,
)

# מימוש: נווה (מדען נתונים - ML)

import os
import pandas as pd
import joblib
from pathlib import Path
from loguru import logger
from sklearn.model_selection import train_test_split


def run_scientist_crew(
    clean_data_path: str,
    contract_path: str,
    features_dir: str,
    models_dir: str,
    reports_dir: str,
) -> dict:
    """
    הרצת Data Scientist Crew

    תחום אחריות (נווה - מדען נתונים):
    - Feature Engineering (הנדסת תכונות)
    - אימון 2 מודלים (LinearRegression + RandomForestRegressor)
    - שמירת המודל הטוב ביותר

    תוצרים: features.csv, model.pkl

    Args:
        clean_data_path: נתיב לנתונים הנקיים
        contract_path: נתיב לחוזה הנתונים
        features_dir: תיקייה לשמירת features
        models_dir: תיקייה לשמירת מודלים
        reports_dir: תיקייה לשמירת דוחות

    Returns:
        מילון עם נתיבי התוצרים שנוצרו
    """
    logger.info("=" * 60)
    logger.info("Scientist Crew - Feature Engineering + Model Training")
    logger.info("=" * 60)

    # --- ולידציה ---
    if not os.path.exists(clean_data_path):
        raise FileNotFoundError(f"Clean data file not found: {clean_data_path}")
    if not os.path.exists(contract_path):
        raise FileNotFoundError(f"Contract file not found: {contract_path}")

    # --- יצירת תיקיות ---
    for dir_path in [features_dir, models_dir]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    # --- שלב 1: טעינת נתונים נקיים ---
    logger.info("שלב 1: טוען נתונים נקיים...")
    clean_data = pd.read_csv(clean_data_path)
    logger.info(f"  נטענו {len(clean_data)} שורות, {len(clean_data.columns)} עמודות")

    # --- שלב 2: הנדסת תכונות ---
    logger.info("שלב 2: הנדסת תכונות...")
    features_df = engineer_features(clean_data)

    features_path = os.path.join(features_dir, "features.csv")
    features_df.to_csv(features_path, index=False)
    logger.info(f"  נשמר: {features_path}")

    # --- שלב 3: אימון מודלים ---
    logger.info("שלב 3: אימון מודלים...")
    X, y = prepare_model_data(features_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    logger.info(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    results = train_models(X_train, y_train, X_test, y_test)

    # --- שלב 4: שמירת המודל הטוב ביותר ---
    logger.info("שלב 4: שמירת המודל הטוב ביותר...")
    model_path = os.path.join(models_dir, "model.pkl")
    joblib.dump(results["best_model"], model_path)
    logger.info(f"  נשמר: {model_path} ({results['best_model_name']})")

    # --- סיכום ---
    logger.info("=" * 60)
    logger.success("Scientist Crew - סיים בהצלחה!")
    logger.info(f"  מודל: {results['best_model_name']}")
    logger.info(f"  MAE: {results[results['best_model_name']]['mae']:.2f}")
    logger.info(f"  R2: {results[results['best_model_name']]['r2']:.4f}")
    logger.info("=" * 60)

    return {
        "features_path": features_path,
        "model_path": model_path,
        "model_type": results["best_model_name"],
        "metrics": {
            name: {"mae": r["mae"], "rmse": r["rmse"], "r2": r["r2"]}
            for name, r in results.items()
            if isinstance(r, dict) and "mae" in r
        },
    }
