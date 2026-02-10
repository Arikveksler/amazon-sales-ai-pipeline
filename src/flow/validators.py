"""
Pipeline Validators - src/flow/validators.py

פונקציות validation לבדיקת תקינות בין שלבי ה-Pipeline
"""

import json
from pathlib import Path
from typing import Tuple, Dict, Any

import pandas as pd
from loguru import logger


def validate_raw_data(file_path: str) -> Tuple[bool, str]:
    """
    בדיקת תקינות נתונים גולמיים

    בדיקות:
    1. הקובץ קיים
    2. הקובץ נטען תקין עם pandas
    3. הקובץ לא ריק (יש שורות ועמודות)

    Args:
        file_path: נתיב לקובץ CSV

    Returns:
        Tuple[bool, str]: (האם תקין, הודעה)

    Examples:
        >>> valid, msg = validate_raw_data("data/raw/amazon_sales.csv")
        >>> if valid:
        >>>     print("Data is valid!")
    """
    logger.info(f"בודק תקינות נתונים גולמיים: {file_path}")

    path = Path(file_path)

    # בדיקה 1: הקובץ קיים
    if not path.exists():
        error_msg = f"קובץ לא נמצא: {file_path}"
        logger.error(error_msg)
        return False, error_msg

    # בדיקה 2: הקובץ נטען תקין
    try:
        df = pd.read_csv(path)
    except Exception as e:
        error_msg = f"שגיאה בטעינת הקובץ: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

    # בדיקה 3: הקובץ לא ריק
    if df.empty:
        error_msg = f"הקובץ ריק (0 שורות)"
        logger.error(error_msg)
        return False, error_msg

    if len(df.columns) == 0:
        error_msg = f"הקובץ ללא עמודות"
        logger.error(error_msg)
        return False, error_msg

    # הצלחה
    success_msg = f"נתונים גולמיים תקינים: {len(df)} שורות, {len(df.columns)} עמודות"
    logger.success(success_msg)
    return True, success_msg


def validate_clean_data(file_path: str, contract_path: str) -> Tuple[bool, str]:
    """
    בדיקת תקינות נתונים נקיים

    בדיקות:
    1. הקובץ קיים ונטען תקין
    2. ה-schema תואם ל-dataset_contract.json
    3. אין missing values בעמודות קריטיות (מוגדרות ב-contract)
    4. כל העמודות הנדרשות קיימות

    Args:
        file_path: נתיב לקובץ clean_data.csv
        contract_path: נתיב ל-dataset_contract.json

    Returns:
        Tuple[bool, str]: (האם תקין, הודעה)
    """
    logger.info(f"בודק תקינות נתונים נקיים: {file_path}")

    # בדיקה 1: הקובץ קיים
    path = Path(file_path)
    if not path.exists():
        error_msg = f"קובץ נתונים נקיים לא נמצא: {file_path}"
        logger.error(error_msg)
        return False, error_msg

    # בדיקה 2: טעינת הנתונים
    try:
        df = pd.read_csv(path)
    except Exception as e:
        error_msg = f"שגיאה בטעינת נתונים נקיים: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

    # בדיקה 3: טעינת ה-contract
    contract_path_obj = Path(contract_path)
    if not contract_path_obj.exists():
        error_msg = f"קובץ contract לא נמצא: {contract_path}"
        logger.error(error_msg)
        return False, error_msg

    try:
        with open(contract_path_obj, 'r', encoding='utf-8') as f:
            contract = json.load(f)
    except Exception as e:
        error_msg = f"שגיאה בטעינת contract: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

    # בדיקה 4: כל העמודות הנדרשות קיימות
    required_columns = contract.get("required_columns", [])
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        error_msg = f"עמודות חסרות: {', '.join(missing_columns)}"
        logger.error(error_msg)
        return False, error_msg

    # בדיקה 5: אין missing values בעמודות קריטיות
    for col in required_columns:
        if df[col].isnull().any():
            missing_count = df[col].isnull().sum()
            error_msg = f"נמצאו {missing_count} ערכים חסרים בעמודה קריטית: {col}"
            logger.error(error_msg)
            return False, error_msg

    # הצלחה
    success_msg = f"נתונים נקיים תקינים: {len(df)} שורות, {len(df.columns)} עמודות, schema תואם"
    logger.success(success_msg)
    return True, success_msg


def validate_dataset_contract(contract_path: str) -> Tuple[bool, str]:
    """
    בדיקת תקינות dataset contract (חוזה הנתונים)

    בדיקות:
    1. הקובץ קיים
    2. הJSON תקין (ניתן לפענוח)
    3. קיימים כל השדות הנדרשים: schema, required_columns, constraints
    4. schema הוא dict תקין
    5. required_columns הוא list

    Args:
        contract_path: נתיב ל-dataset_contract.json

    Returns:
        Tuple[bool, str]: (האם תקין, הודעה)
    """
    logger.info(f"בודק תקינות dataset contract: {contract_path}")

    path = Path(contract_path)

    # בדיקה 1: הקובץ קיים
    if not path.exists():
        error_msg = f"קובץ contract לא נמצא: {contract_path}"
        logger.error(error_msg)
        return False, error_msg

    # בדיקה 2: JSON תקין
    try:
        with open(path, 'r', encoding='utf-8') as f:
            contract = json.load(f)
    except json.JSONDecodeError as e:
        error_msg = f"JSON לא תקין: {str(e)}"
        logger.error(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"שגיאה בקריאת contract: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

    # בדיקה 3: שדות חובה קיימים
    required_fields = ["schema", "required_columns", "constraints"]
    missing_fields = [field for field in required_fields if field not in contract]

    if missing_fields:
        error_msg = f"שדות חסרים ב-contract: {', '.join(missing_fields)}"
        logger.error(error_msg)
        return False, error_msg

    # בדיקה 4: schema הוא dict
    if not isinstance(contract["schema"], dict):
        error_msg = f"schema צריך להיות dict, לא {type(contract['schema'])}"
        logger.error(error_msg)
        return False, error_msg

    # בדיקה 5: required_columns הוא list
    if not isinstance(contract["required_columns"], list):
        error_msg = f"required_columns צריך להיות list, לא {type(contract['required_columns'])}"
        logger.error(error_msg)
        return False, error_msg

    # הצלחה
    success_msg = f"Contract תקין: {len(contract['schema'])} עמודות, {len(contract['required_columns'])} required"
    logger.success(success_msg)
    return True, success_msg


def validate_features(features_path: str) -> Tuple[bool, str]:
    """
    בדיקת תקינות קובץ features

    בדיקות:
    1. הקובץ קיים
    2. הקובץ נטען תקין
    3. יש לפחות 3 features (עמודות)
    4. יש לפחות 10 שורות

    Args:
        features_path: נתיב ל-features.csv

    Returns:
        Tuple[bool, str]: (האם תקין, הודעה)
    """
    logger.info(f"בודק תקינות features: {features_path}")

    path = Path(features_path)

    # בדיקה 1: הקובץ קיים
    if not path.exists():
        error_msg = f"קובץ features לא נמצא: {features_path}"
        logger.error(error_msg)
        return False, error_msg

    # בדיקה 2: טעינת הקובץ
    try:
        df = pd.read_csv(path)
    except Exception as e:
        error_msg = f"שגיאה בטעינת features: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

    # בדיקה 3: לפחות 3 features
    if len(df.columns) < 3:
        error_msg = f"מספר features לא מספיק: {len(df.columns)} (נדרש לפחות 3)"
        logger.error(error_msg)
        return False, error_msg

    # בדיקה 4: לפחות 10 שורות
    if len(df) < 10:
        error_msg = f"מספר שורות לא מספיק: {len(df)} (נדרש לפחות 10)"
        logger.error(error_msg)
        return False, error_msg

    # הצלחה
    success_msg = f"Features תקינים: {len(df)} שורות, {len(df.columns)} features"
    logger.success(success_msg)
    return True, success_msg


def validate_model_outputs(
    model_path: str,
    eval_path: str,
    card_path: str
) -> Tuple[bool, str]:
    """
    בדיקת תקינות תוצרי המודל

    בדיקות:
    1. model.pkl קיים
    2. evaluation_report.md קיים ולא ריק
    3. model_card.md קיים ומכיל את הסקשנים הנדרשים:
       - Model Purpose
       - Training Data
       - Metrics
       - Limitations

    Args:
        model_path: נתיב ל-model.pkl
        eval_path: נתיב ל-evaluation_report.md
        card_path: נתיב ל-model_card.md

    Returns:
        Tuple[bool, str]: (האם תקין, הודעה)
    """
    logger.info(f"בודק תקינות תוצרי מודל")

    # בדיקה 1: model.pkl קיים
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        error_msg = f"קובץ מודל לא נמצא: {model_path}"
        logger.error(error_msg)
        return False, error_msg
    logger.success(f"model.pkl קיים")

    # בדיקה 2: evaluation_report.md קיים ולא ריק
    eval_path_obj = Path(eval_path)
    if not eval_path_obj.exists():
        error_msg = f"דוח הערכה לא נמצא: {eval_path}"
        logger.error(error_msg)
        return False, error_msg

    try:
        with open(eval_path_obj, 'r', encoding='utf-8') as f:
            eval_content = f.read()

        if len(eval_content.strip()) < 50:
            error_msg = f"דוח הערכה קצר מדי (פחות מ-50 תווים)"
            logger.error(error_msg)
            return False, error_msg

        logger.success(f"evaluation_report.md תקין")
    except Exception as e:
        error_msg = f"שגיאה בקריאת דוח הערכה: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

    # בדיקה 3: model_card.md קיים ומכיל סקשנים נדרשים
    card_path_obj = Path(card_path)
    if not card_path_obj.exists():
        error_msg = f"model card לא נמצא: {card_path}"
        logger.error(error_msg)
        return False, error_msg

    try:
        with open(card_path_obj, 'r', encoding='utf-8') as f:
            card_content = f.read().lower()  # lowercase לבדיקה

        required_sections = [
            "model purpose",
            "training data",
            "metrics",
            "limitations"
        ]

        missing_sections = [
            section for section in required_sections
            if section not in card_content
        ]

        if missing_sections:
            error_msg = f"סקשנים חסרים ב-model_card: {', '.join(missing_sections)}"
            logger.error(error_msg)
            return False, error_msg

        logger.success(f"model_card.md תקין עם כל הסקשנים")
    except Exception as e:
        error_msg = f"שגיאה בקריאת model card: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

    # הצלחה
    success_msg = f"כל תוצרי המודל תקינים (model + evaluation + card)"
    logger.success(success_msg)
    return True, success_msg
