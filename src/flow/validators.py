"""
Validators Module
=================
פונקציות validation לבדיקת כל שלבי ה-Pipeline.

כל פונקציה מחזירה (bool, str):
- (True, הודעה) - אם הvalidation עבר
- (False, הודעה) - אם הvalidation נכשל

Author: Pipeline Lead
Date: 2026
"""

from pathlib import Path
import json
import pandas as pd
from loguru import logger
from typing import Tuple


def validate_raw_data(file_path: str) -> Tuple[bool, str]:
    """
    בדיקת תקינות נתונים גולמיים.
    מוודא שהקובץ קיים, ניתן לטעינה, ומכיל נתונים.

    Args:
        file_path: נתיב לקובץ הנתונים הגולמי

    Returns:
        (True, הודעה) אם הקובץ תקין
        (False, הודעה) אם יש בעיה
    """
    logger.info(f"🔍 Validating raw data: {file_path}")

    try:
        # המרה ל-Path object
        file_path_obj = Path(file_path)

        # בדיקת קיום הקובץ
        if not file_path_obj.exists():
            msg = f"File not found: {file_path}"
            logger.error(f"✗ {msg}")
            return False, msg

        # טעינת הנתונים
        df = pd.read_csv(file_path)

        # בדיקת שורות - האם יש לפחות שורה אחת
        if len(df) == 0:
            msg = "File is empty (no rows)"
            logger.error(f"✗ {msg}")
            return False, msg

        # בדיקת עמודות - האם יש לפחות עמודה אחת
        if len(df.columns) == 0:
            msg = "File has no columns"
            logger.error(f"✗ {msg}")
            return False, msg

        # הצלחה - הכל תקין
        msg = f"Raw data validation passed: {len(df)} rows, {len(df.columns)} columns"
        logger.success(f"✓ {msg}")
        return True, msg

    except pd.errors.EmptyDataError as e:
        msg = f"File is empty or corrupted: {str(e)}"
        logger.error(f"✗ {msg}")
        return False, msg

    except pd.errors.ParserError as e:
        msg = f"Failed to parse CSV file: {str(e)}"
        logger.error(f"✗ {msg}")
        return False, msg

    except Exception as e:
        msg = f"Error validating raw data: {str(e)}"
        logger.error(f"✗ {msg}")
        return False, msg


def validate_clean_data(file_path: str, contract_path: str) -> Tuple[bool, str]:
    """
    בדיקת תקינות נתונים מנוקים מול החוזה.
    מוודא שכל העמודות הנדרשות קיימות ואין בהן ערכים חסרים.

    Args:
        file_path: נתיב לקובץ הנתונים המנוקים (clean_data.csv)
        contract_path: נתיב לקובץ החוזה (dataset_contract.json)

    Returns:
        (True, הודעה) אם הנתונים תואמים לחוזה
        (False, הודעה) אם יש אי-התאמה
    """
    logger.info(f"🔍 Validating clean data against contract")
    logger.info(f"   Data file: {file_path}")
    logger.info(f"   Contract file: {contract_path}")

    try:
        # בדיקת קיום קובץ הנתונים המנוקים
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            msg = f"Clean data file not found: {file_path}"
            logger.error(f"✗ {msg}")
            return False, msg

        # בדיקת קיום קובץ החוזה
        contract_path_obj = Path(contract_path)
        if not contract_path_obj.exists():
            msg = f"Contract file not found: {contract_path}"
            logger.error(f"✗ {msg}")
            return False, msg

        # טעינת ה-DataFrame
        logger.info("   Loading clean data...")
        df = pd.read_csv(file_path)

        # טעינת החוזה
        logger.info("   Loading contract...")
        with open(contract_path, "r", encoding="utf-8") as f:
            contract = json.load(f)

        # קבלת העמודות הנדרשות מהחוזה
        # בדיקה אם יש required_columns, אחרת נשתמש בעמודות מה-schema
        if "required_columns" in contract:
            required_columns = contract["required_columns"]
        elif "schema" in contract and "columns" in contract["schema"]:
            required_columns = contract["schema"]["columns"]
        else:
            # אם אין הגדרה ספציפית, נניח שכל העמודות נדרשות
            msg = "Contract missing required_columns definition, skipping column check"
            logger.warning(f"⚠ {msg}")
            required_columns = []

        # בדיקת עמודות חסרות
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                msg = f"Missing required columns: {missing_cols}"
                logger.error(f"✗ {msg}")
                return False, msg
            logger.info(f"   ✓ All {len(required_columns)} required columns present")

            # בדיקת missing values בעמודות הנדרשות
            for col in required_columns:
                if col in df.columns and df[col].isna().any():
                    null_count = df[col].isna().sum()
                    msg = f"Column '{col}' has {null_count} missing values"
                    logger.error(f"✗ {msg}")
                    return False, msg

            logger.info("   ✓ No missing values in required columns")

        # הצלחה
        msg = f"Clean data validation passed: {len(df)} rows, all required columns present"
        logger.success(f"✓ {msg}")
        return True, msg

    except json.JSONDecodeError as e:
        msg = f"Invalid JSON in contract file: {str(e)}"
        logger.error(f"✗ {msg}")
        return False, msg

    except pd.errors.EmptyDataError as e:
        msg = f"Clean data file is empty: {str(e)}"
        logger.error(f"✗ {msg}")
        return False, msg

    except Exception as e:
        msg = f"Error validating clean data: {str(e)}"
        logger.error(f"✗ {msg}")
        return False, msg


def validate_dataset_contract(contract_path: str) -> Tuple[bool, str]:
    """
    בדיקת תקינות חוזה הנתונים.
    מוודא שהקובץ הוא JSON תקין ומכיל את כל השדות הנדרשים.

    Args:
        contract_path: נתיב לקובץ החוזה (dataset_contract.json)

    Returns:
        (True, הודעה) אם החוזה תקין
        (False, הודעה) אם יש בעיה בחוזה
    """
    logger.info(f"🔍 Validating dataset contract: {contract_path}")

    try:
        # בדיקת קיום הקובץ
        contract_path_obj = Path(contract_path)
        if not contract_path_obj.exists():
            msg = f"Contract file not found: {contract_path}"
            logger.error(f"✗ {msg}")
            return False, msg

        # טעינת ה-JSON
        logger.info("   Loading contract JSON...")
        with open(contract_path, "r", encoding="utf-8") as f:
            contract = json.load(f)

        # שדות חובה שצריכים להיות בחוזה
        required_fields = ["schema", "required_columns", "constraints"]

        # בדיקת שדות חובה
        missing_fields = []
        for field in required_fields:
            if field not in contract:
                missing_fields.append(field)

        if missing_fields:
            # בדיקה אם יש לפחות schema
            if "schema" not in contract:
                msg = f"Contract missing critical field: schema"
                logger.error(f"✗ {msg}")
                return False, msg
            else:
                logger.warning(f"⚠ Contract missing optional fields: {missing_fields}")

        # בדיקת required_columns אם קיים
        if "required_columns" in contract:
            required_columns = contract["required_columns"]

            # בדיקה שזה list
            if not isinstance(required_columns, list):
                msg = "required_columns must be a list"
                logger.error(f"✗ {msg}")
                return False, msg

            # בדיקה שלא ריק
            if len(required_columns) == 0:
                msg = "required_columns cannot be empty"
                logger.error(f"✗ {msg}")
                return False, msg

            logger.info(f"   ✓ required_columns: {len(required_columns)} columns defined")

        # בדיקת schema אם קיים
        if "schema" in contract:
            schema = contract["schema"]

            # בדיקה שזה dict
            if not isinstance(schema, dict):
                msg = "schema must be a dictionary"
                logger.error(f"✗ {msg}")
                return False, msg

            # בדיקה שלא ריק
            if len(schema) == 0:
                msg = "schema cannot be empty"
                logger.error(f"✗ {msg}")
                return False, msg

            logger.info(f"   ✓ schema: {len(schema)} fields defined")

        # ספירת עמודות נדרשות להודעת ההצלחה
        num_columns = 0
        if "required_columns" in contract:
            num_columns = len(contract["required_columns"])
        elif "schema" in contract and "columns" in contract["schema"]:
            num_columns = len(contract["schema"]["columns"])

        # הצלחה
        msg = f"Dataset contract validation passed: {num_columns} required columns defined"
        logger.success(f"✓ {msg}")
        return True, msg

    except json.JSONDecodeError as e:
        msg = f"Invalid JSON format: {str(e)}"
        logger.error(f"✗ {msg}")
        return False, msg

    except FileNotFoundError as e:
        msg = f"Contract file not found: {str(e)}"
        logger.error(f"✗ {msg}")
        return False, msg

    except Exception as e:
        msg = f"Error validating contract: {str(e)}"
        logger.error(f"✗ {msg}")
        return False, msg


def validate_features(features_path: str, contract_path: str = None) -> Tuple[bool, str]:
    """
    בדיקת תקינות קובץ ה-features.
    מוודא שהקובץ קיים ומכיל נתונים.

    Args:
        features_path: נתיב לקובץ ה-features (features.csv)
        contract_path: נתיב לקובץ החוזה (אופציונלי, לבדיקות נוספות)

    Returns:
        (True, הודעה) אם ה-features תקינים
        (False, הודעה) אם יש בעיה
    """
    logger.info(f"🔍 Validating features: {features_path}")

    try:
        # בדיקת קיום קובץ ה-features
        features_path_obj = Path(features_path)
        if not features_path_obj.exists():
            msg = f"Features file not found: {features_path}"
            logger.error(f"✗ {msg}")
            return False, msg

        # טעינת הקובץ
        logger.info("   Loading features file...")
        df = pd.read_csv(features_path)

        # בדיקה שיש שורות
        if len(df) == 0:
            msg = "Features file is empty (no rows)"
            logger.error(f"✗ {msg}")
            return False, msg

        # בדיקה שיש עמודות (features)
        if len(df.columns) == 0:
            msg = "Features file has no columns"
            logger.error(f"✗ {msg}")
            return False, msg

        logger.info(f"   ✓ Features loaded: {len(df)} rows, {len(df.columns)} features")

        # בדיקות אופציונליות מול החוזה
        if contract_path:
            contract_path_obj = Path(contract_path)
            if contract_path_obj.exists():
                logger.info("   Checking against contract...")
                with open(contract_path, "r", encoding="utf-8") as f:
                    contract = json.load(f)

                # בדיקה אם יש מינימום features מוגדר
                if "min_features" in contract:
                    min_features = contract["min_features"]
                    if len(df.columns) < min_features:
                        msg = f"Not enough features: {len(df.columns)} < {min_features} required"
                        logger.error(f"✗ {msg}")
                        return False, msg
                    logger.info(f"   ✓ Minimum features requirement met")

        # הצלחה
        msg = f"Features validation passed: {len(df)} rows, {len(df.columns)} features"
        logger.success(f"✓ {msg}")
        return True, msg

    except pd.errors.EmptyDataError as e:
        msg = f"Features file is empty or corrupted: {str(e)}"
        logger.error(f"✗ {msg}")
        return False, msg

    except pd.errors.ParserError as e:
        msg = f"Failed to parse features CSV: {str(e)}"
        logger.error(f"✗ {msg}")
        return False, msg

    except json.JSONDecodeError as e:
        msg = f"Invalid JSON in contract file: {str(e)}"
        logger.error(f"✗ {msg}")
        return False, msg

    except Exception as e:
        msg = f"Error validating features: {str(e)}"
        logger.error(f"✗ {msg}")
        return False, msg


def validate_model_outputs(
    model_path: str,
    eval_path: str,
    card_path: str
) -> Tuple[bool, str]:
    """
    בדיקת תקינות כל תוצרי המודל.
    מוודא שהמודל, דוח ההערכה, וכרטיס המודל נוצרו ותקינים.

    Args:
        model_path: נתיב לקובץ המודל (model.pkl)
        eval_path: נתיב לדוח ההערכה (evaluation_report.md)
        card_path: נתיב לכרטיס המודל (model_card.md)

    Returns:
        (True, הודעה) אם כל התוצרים תקינים
        (False, הודעה) אם חסר קובץ או יש בעיה
    """
    logger.info("🔍 Validating model outputs")
    logger.info(f"   Model: {model_path}")
    logger.info(f"   Evaluation: {eval_path}")
    logger.info(f"   Model Card: {card_path}")

    try:
        # בדיקה 1: קיום קובץ המודל
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            msg = f"Model file not found: {model_path}"
            logger.error(f"✗ {msg}")
            return False, msg

        # בדיקה שהקובץ לא ריק
        if model_path_obj.stat().st_size == 0:
            msg = f"Model file is empty: {model_path}"
            logger.error(f"✗ {msg}")
            return False, msg

        logger.info("   ✓ Model file exists and not empty")

        # בדיקה 2: קיום דוח ההערכה
        eval_path_obj = Path(eval_path)
        if not eval_path_obj.exists():
            msg = f"Evaluation report not found: {eval_path}"
            logger.error(f"✗ {msg}")
            return False, msg

        # קריאת דוח ההערכה ובדיקה שיש תוכן
        with open(eval_path, "r", encoding="utf-8") as f:
            eval_content = f.read()

        if len(eval_content.strip()) == 0:
            msg = "Evaluation report is empty"
            logger.error(f"✗ {msg}")
            return False, msg

        logger.info("   ✓ Evaluation report exists and has content")

        # בדיקה 3: קיום כרטיס המודל
        card_path_obj = Path(card_path)
        if not card_path_obj.exists():
            msg = f"Model card not found: {card_path}"
            logger.error(f"✗ {msg}")
            return False, msg

        # קריאת כרטיס המודל
        with open(card_path, "r", encoding="utf-8") as f:
            card_content = f.read()

        if len(card_content.strip()) == 0:
            msg = "Model card is empty"
            logger.error(f"✗ {msg}")
            return False, msg

        logger.info("   ✓ Model card exists and has content")

        # בדיקה 4: בדיקת סקשנים נדרשים בכרטיס המודל
        required_sections = ["Purpose", "Data", "Metrics", "Limitations", "Ethical"]
        missing_sections = []

        for section in required_sections:
            # בדיקה case-insensitive
            if section.lower() not in card_content.lower():
                missing_sections.append(section)

        if missing_sections:
            msg = f"Model card missing section: {missing_sections[0]}"
            logger.error(f"✗ {msg}")
            return False, msg

        logger.info("   ✓ Model card contains all required sections")

        # הצלחה - כל הקבצים קיימים ותקינים
        msg = "Model outputs validation passed: all files present and valid"
        logger.success(f"✓ {msg}")
        return True, msg

    except FileNotFoundError as e:
        msg = f"File not found: {str(e)}"
        logger.error(f"✗ {msg}")
        return False, msg

    except PermissionError as e:
        msg = f"Permission denied reading file: {str(e)}"
        logger.error(f"✗ {msg}")
        return False, msg

    except Exception as e:
        msg = f"Error validating model outputs: {str(e)}"
        logger.error(f"✗ {msg}")
        return False, msg


# =============================================================================
# מחלקת DataValidator לתאימות אחורה
# =============================================================================

class DataValidator:
    """
    מחלקה עוטפת לפונקציות ה-validation.
    שומרת על תאימות אחורה עם הקוד הקיים.
    """

    def __init__(self):
        """אתחול הValidator"""
        self.logger = logger.bind(name="DataValidator")

    def validate_raw_data(self, file_path) -> Tuple[bool, str]:
        """עוטף את הפונקציה validate_raw_data"""
        return validate_raw_data(str(file_path))

    def validate_clean_data(self, file_path, contract_path=None) -> Tuple[bool, str]:
        """עוטף את הפונקציה validate_clean_data"""
        if contract_path:
            return validate_clean_data(str(file_path), str(contract_path))
        # אם אין contract, פשוט בדוק שהקובץ קיים ותקין
        return validate_raw_data(str(file_path))

    def validate_dataset_contract(self, contract_path) -> Tuple[bool, str]:
        """עוטף את הפונקציה validate_dataset_contract"""
        return validate_dataset_contract(str(contract_path))

    def validate_features(self, features_path, contract_path=None) -> Tuple[bool, str]:
        """עוטף את הפונקציה validate_features"""
        return validate_features(str(features_path), str(contract_path) if contract_path else None)

    def validate_model_outputs(self, model_path, eval_path, card_path) -> Tuple[bool, str]:
        """עוטף את הפונקציה validate_model_outputs"""
        return validate_model_outputs(str(model_path), str(eval_path), str(card_path))


# =============================================================================
# בדיקת נתונים מול constraints
# =============================================================================

def validate_against_constraints(df: pd.DataFrame, contract: dict) -> Tuple[bool, str]:
    """
    בדיקת DataFrame מול אילוצי החוזה.
    מוודא שכל הנתונים עומדים ב-constraints שהוגדרו.

    Args:
        df: הנתונים לבדיקה
        contract: החוזה עם ה-constraints

    Returns:
        (True, הודעה) אם עובר
        (False, רשימת שגיאות) אם נכשל

    Example:
        >>> df = pd.read_csv("clean_data.csv")
        >>> contract = json.load(open("dataset_contract.json"))
        >>> is_valid, msg = validate_against_constraints(df, contract)
    """
    logger.info("🔍 Validating data against contract constraints")

    errors = []
    constraints = contract.get('constraints', {})

    if not constraints:
        logger.warning("⚠ No constraints defined in contract")
        return True, "No constraints to validate"

    for column, rules in constraints.items():
        # בדיקת קיום עמודה
        if column not in df.columns:
            if rules.get('required', False):
                errors.append(f"עמודה חובה חסרה: {column}")
            continue

        col_data = df[column]

        # בדיקת ערכים מספריים
        if rules.get('type') == 'numeric':
            try:
                # ניסיון להמיר למספרים (מטפל בפורמט מחירים עם ₹)
                numeric_data = pd.to_numeric(
                    col_data.astype(str).str.replace('[₹,]', '', regex=True),
                    errors='coerce'
                )

                # בדיקת מינימום
                if 'min' in rules:
                    below_min = numeric_data < rules['min']
                    if below_min.any():
                        count = below_min.sum()
                        errors.append(f"{column}: {count} ערכים מתחת למינימום {rules['min']}")

                # בדיקת מקסימום
                if 'max' in rules:
                    above_max = numeric_data > rules['max']
                    if above_max.any():
                        count = above_max.sum()
                        errors.append(f"{column}: {count} ערכים מעל מקסימום {rules['max']}")

            except Exception as e:
                errors.append(f"{column}: שגיאה בהמרה למספר - {str(e)}")

        # בדיקת ייחודיות
        if rules.get('unique', False):
            duplicates = col_data.duplicated().sum()
            if duplicates > 0:
                errors.append(f"{column}: {duplicates} ערכים כפולים (צריך להיות ייחודי)")

        # בדיקת ערכים חסרים
        if rules.get('required', False):
            null_count = col_data.isnull().sum()
            if null_count > 0:
                errors.append(f"{column}: {null_count} ערכים חסרים (שדה חובה)")

    # סיכום
    if errors:
        error_msg = "; ".join(errors)
        logger.error(f"✗ Validation failed: {error_msg}")
        return False, error_msg

    validated_count = len(constraints)
    logger.success(f"✓ All {validated_count} constraints validated successfully")
    return True, f"Validated {validated_count} constraints successfully"


# =============================================================================
# פונקציית עזר להרצת כל הvalidations
# =============================================================================

def validate_all(
    raw_data_path: str = None,
    clean_data_path: str = None,
    contract_path: str = None,
    features_path: str = None,
    model_path: str = None,
    eval_path: str = None,
    card_path: str = None
) -> Tuple[bool, dict]:
    """
    הרצת כל הvalidations בבת אחת.
    שימושי לבדיקה מהירה של כל ה-Pipeline.

    Args:
        raw_data_path: נתיב לנתונים גולמיים
        clean_data_path: נתיב לנתונים מנוקים
        contract_path: נתיב לחוזה
        features_path: נתיב ל-features
        model_path: נתיב למודל
        eval_path: נתיב לדוח הערכה
        card_path: נתיב לכרטיס מודל

    Returns:
        (True, results) אם כל הבדיקות עברו
        (False, results) אם יש כשלון
    """
    logger.info("=" * 50)
    logger.info("🔍 Running all validations")
    logger.info("=" * 50)

    results = {
        "raw_data": None,
        "clean_data": None,
        "contract": None,
        "features": None,
        "model_outputs": None
    }

    all_passed = True

    # בדיקת נתונים גולמיים
    if raw_data_path:
        is_valid, msg = validate_raw_data(raw_data_path)
        results["raw_data"] = {"valid": is_valid, "message": msg}
        if not is_valid:
            all_passed = False

    # בדיקת חוזה
    if contract_path:
        is_valid, msg = validate_dataset_contract(contract_path)
        results["contract"] = {"valid": is_valid, "message": msg}
        if not is_valid:
            all_passed = False

    # בדיקת נתונים מנוקים
    if clean_data_path and contract_path:
        is_valid, msg = validate_clean_data(clean_data_path, contract_path)
        results["clean_data"] = {"valid": is_valid, "message": msg}
        if not is_valid:
            all_passed = False

    # בדיקת features
    if features_path:
        is_valid, msg = validate_features(features_path, contract_path)
        results["features"] = {"valid": is_valid, "message": msg}
        if not is_valid:
            all_passed = False

    # בדיקת תוצרי מודל
    if model_path and eval_path and card_path:
        is_valid, msg = validate_model_outputs(model_path, eval_path, card_path)
        results["model_outputs"] = {"valid": is_valid, "message": msg}
        if not is_valid:
            all_passed = False

    # סיכום
    logger.info("=" * 50)
    if all_passed:
        logger.success("✓ All validations passed!")
    else:
        logger.error("✗ Some validations failed")
    logger.info("=" * 50)

    return all_passed, results


# =============================================================================
# בדיקה עצמית
# =============================================================================

if __name__ == "__main__":
    # בדיקה מהירה של הפונקציות
    project_root = Path(__file__).parent.parent.parent

    print("\n" + "=" * 60)
    print("Testing Validators Module")
    print("=" * 60 + "\n")

    # בדיקת נתונים גולמיים
    raw_path = project_root / "data" / "raw" / "amazon_sales.csv"
    if raw_path.exists():
        is_valid, msg = validate_raw_data(str(raw_path))
        status = "PASS" if is_valid else "FAIL"
        print(f"\nRaw data: [{status}] {msg}")

    # בדיקת חוזה
    contract_path = project_root / "data" / "contracts" / "dataset_contract.json"
    if contract_path.exists():
        is_valid, msg = validate_dataset_contract(str(contract_path))
        status = "PASS" if is_valid else "FAIL"
        print(f"\nContract: [{status}] {msg}")

    # בדיקת נתונים מנוקים
    clean_path = project_root / "data" / "processed" / "clean_data.csv"
    if clean_path.exists() and contract_path.exists():
        is_valid, msg = validate_clean_data(str(clean_path), str(contract_path))
        status = "PASS" if is_valid else "FAIL"
        print(f"\nClean data: [{status}] {msg}")

    print("\n" + "=" * 60)
    print("Validators test complete")
    print("=" * 60 + "\n")
