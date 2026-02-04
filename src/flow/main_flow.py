"""
Amazon Sales AI Pipeline - Main Flow
=====================================
קובץ זה מנהל את כל ה-Pipeline של הפרויקט.
הוא אחראי על הרצת ה-Crews לפי הסדר ועל ניהול ה-State.
"""

from pathlib import Path
from loguru import logger
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional


class AmazonSalesPipeline:
    """
    מחלקה ראשית לניהול ה-Pipeline.
    מריצה את כל השלבים לפי הסדר ומנהלת את ה-State.
    """

    def __init__(self):
        """
        אתחול ה-Pipeline.
        מגדיר נתיבים, מאתחל logger ו-state.
        """
        # נתיבי בסיס של הפרויקט
        self.project_root = Path(__file__).parent.parent.parent

        # נתיבי נתונים
        self.raw_data_path = self.project_root / "data" / "raw" / "amazon_sales.csv"
        self.processed_data_path = self.project_root / "data" / "processed"
        self.contracts_path = self.project_root / "data" / "contracts"

        # נתיבי outputs
        self.reports_path = self.project_root / "outputs" / "reports"
        self.models_path = self.project_root / "outputs" / "models"
        self.features_path = self.project_root / "outputs" / "features"

        # נתיב ל-state file
        self.state_file = self.project_root / "outputs" / "pipeline_state.json"

        # אתחול state
        self.state: Dict[str, Any] = {
            "pipeline_start": None,
            "pipeline_end": None,
            "status": "initialized",
            "stages": {},
            "created_files": []
        }

        # הגדרת logger
        self._setup_logger()

        logger.info("Pipeline אותחל בהצלחה")

    def _setup_logger(self):
        """
        הגדרת ה-logger עם loguru.
        שומר לוגים גם לקונסול וגם לקובץ.
        """
        log_path = self.project_root / "outputs" / "reports" / "pipeline.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # הסרת handler ברירת מחדל
        logger.remove()

        # הוספת handler לקונסול עם צבעים
        logger.add(
            sink=lambda msg: print(msg, end=""),
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
            level="INFO"
        )

        # הוספת handler לקובץ
        logger.add(
            sink=str(log_path),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            level="DEBUG",
            rotation="10 MB"
        )

    def _save_state(self, stage: str, status: str, details: Optional[Dict] = None):
        """
        שמירת state לקובץ JSON.
        מתעד את הסטטוס של כל שלב עם timestamps.

        Args:
            stage: שם השלב הנוכחי
            status: הסטטוס (running/completed/failed)
            details: פרטים נוספים אופציונליים
        """
        timestamp = datetime.now().isoformat()

        # עדכון ה-state
        if stage not in self.state["stages"]:
            self.state["stages"][stage] = {}

        self.state["stages"][stage].update({
            "status": status,
            f"{status}_at": timestamp
        })

        if details:
            self.state["stages"][stage].update(details)

        self.state["status"] = f"{stage}_{status}"

        # שמירה לקובץ
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(self.state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"לא הצלחתי לשמור state: {e}")

    def _load_raw_data(self) -> pd.DataFrame:
        """
        שלב 1: טעינת נתונים גולמיים.
        טוען את קובץ ה-CSV מתיקיית raw.

        Returns:
            DataFrame עם הנתונים הגולמיים

        Raises:
            FileNotFoundError: אם הקובץ לא נמצא
            Exception: אם יש בעיה בטעינה
        """
        logger.info("שלב 1: מתחיל טעינת נתונים גולמיים...")
        self._save_state("load_raw_data", "running")

        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"קובץ הנתונים לא נמצא: {self.raw_data_path}")

        df = pd.read_csv(self.raw_data_path)

        logger.info(f"נטענו {len(df)} שורות ו-{len(df.columns)} עמודות")
        self._save_state("load_raw_data", "completed", {
            "rows": len(df),
            "columns": len(df.columns),
            "file_path": str(self.raw_data_path)
        })

        logger.success("שלב 1: טעינת נתונים הושלמה בהצלחה")
        return df

    def _run_analyst_crew(self, raw_data: pd.DataFrame) -> Dict[str, Any]:
        """
        שלב 2: הרצת Analyst Crew.
        PLACEHOLDER - יוחלף באימפלמנטציה אמיתית.

        האנליסט אחראי על:
        - ניקוי נתונים
        - תיקוף נתונים
        - יצירת Data Contract

        Args:
            raw_data: הנתונים הגולמיים

        Returns:
            dict עם תוצאות האנליסט
        """
        logger.info("שלב 2: מתחיל הרצת Analyst Crew...")
        self._save_state("analyst_crew", "running")

        # ============================================
        # PLACEHOLDER - יוחלף ב-Crew האמיתי
        # ============================================
        logger.warning(">>> PLACEHOLDER: Running Analyst Crew (Mock) <<<")

        # יצירת נתונים מנוקים (mock)
        clean_data = raw_data.copy()

        # הסרת שורות עם ערכים חסרים (דוגמה פשוטה)
        clean_data = clean_data.dropna()

        # שמירת קובץ נתונים מנוקים
        clean_data_path = self.processed_data_path / "clean_data.csv"
        clean_data_path.parent.mkdir(parents=True, exist_ok=True)
        clean_data.to_csv(clean_data_path, index=False)
        self.state["created_files"].append(str(clean_data_path))

        # יצירת Data Contract (mock)
        contract = {
            "dataset_name": "amazon_sales_clean",
            "version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "schema": {
                "columns": list(clean_data.columns),
                "dtypes": {col: str(dtype) for col, dtype in clean_data.dtypes.items()},
                "row_count": len(clean_data)
            },
            "quality_checks": {
                "no_nulls": True,
                "validated": True
            }
        }

        contract_path = self.contracts_path / "dataset_contract.json"
        contract_path.parent.mkdir(parents=True, exist_ok=True)
        with open(contract_path, "w", encoding="utf-8") as f:
            json.dump(contract, f, ensure_ascii=False, indent=2)
        self.state["created_files"].append(str(contract_path))

        # ============================================
        # סוף PLACEHOLDER
        # ============================================

        result = {
            "clean_data_path": str(clean_data_path),
            "contract_path": str(contract_path),
            "rows_cleaned": len(clean_data),
            "rows_removed": len(raw_data) - len(clean_data)
        }

        self._save_state("analyst_crew", "completed", result)
        logger.success("שלב 2: Analyst Crew הושלם בהצלחה")

        return result

    def _validate_analyst_outputs(self) -> bool:
        """
        שלב 3: בדיקת outputs של Analyst Crew.
        מוודא שהקבצים הנדרשים נוצרו.

        Returns:
            True אם כל הקבצים קיימים

        Raises:
            FileNotFoundError: אם קובץ חסר
        """
        logger.info("שלב 3: מאמת outputs של Analyst Crew...")
        self._save_state("validate_analyst", "running")

        required_files = [
            self.processed_data_path / "clean_data.csv",
            self.contracts_path / "dataset_contract.json"
        ]

        missing_files = []
        for file_path in required_files:
            if not file_path.exists():
                missing_files.append(str(file_path))

        if missing_files:
            raise FileNotFoundError(f"קבצים חסרים: {missing_files}")

        # בדיקות נוספות
        clean_data = pd.read_csv(required_files[0])
        with open(required_files[1], "r", encoding="utf-8") as f:
            contract = json.load(f)

        validation_result = {
            "clean_data_exists": True,
            "contract_exists": True,
            "clean_data_rows": len(clean_data),
            "contract_version": contract.get("version", "unknown")
        }

        self._save_state("validate_analyst", "completed", validation_result)
        logger.success("שלב 3: אימות outputs של Analyst הושלם")

        return True

    def _run_scientist_crew(self, clean_data_path: str) -> Dict[str, Any]:
        """
        שלב 4: הרצת Scientist Crew.
        PLACEHOLDER - יוחלף באימפלמנטציה אמיתית.

        ה-Scientist אחראי על:
        - הנדסת פיצ'רים
        - אימון מודל
        - הערכת מודל

        Args:
            clean_data_path: נתיב לנתונים המנוקים

        Returns:
            dict עם תוצאות ה-Scientist
        """
        logger.info("שלב 4: מתחיל הרצת Scientist Crew...")
        self._save_state("scientist_crew", "running")

        # ============================================
        # PLACEHOLDER - יוחלף ב-Crew האמיתי
        # ============================================
        logger.warning(">>> PLACEHOLDER: Running Scientist Crew (Mock) <<<")

        import joblib

        # טעינת נתונים מנוקים
        clean_data = pd.read_csv(clean_data_path)

        # יצירת מודל mock (רק placeholder)
        mock_model = {
            "model_type": "placeholder",
            "trained_at": datetime.now().isoformat(),
            "features": list(clean_data.columns),
            "n_samples": len(clean_data)
        }

        # שמירת המודל
        model_path = self.models_path / "model.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(mock_model, model_path)
        self.state["created_files"].append(str(model_path))

        # יצירת דוח הערכה
        evaluation_report = f"""# Model Evaluation Report

## Overview
- **Model Type**: Placeholder (Mock)
- **Training Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Dataset Size**: {len(clean_data)} samples

## Features Used
{chr(10).join([f"- {col}" for col in clean_data.columns[:10]])}
{"..." if len(clean_data.columns) > 10 else ""}

## Performance Metrics (Mock)
| Metric | Value |
|--------|-------|
| Accuracy | 0.XX |
| Precision | 0.XX |
| Recall | 0.XX |
| F1-Score | 0.XX |

> **Note**: This is a placeholder report. Real metrics will be added when the Scientist Crew is implemented.

## Next Steps
1. Implement feature engineering
2. Train actual model
3. Perform hyperparameter tuning
4. Generate real evaluation metrics
"""

        report_path = self.reports_path / "evaluation_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(evaluation_report)
        self.state["created_files"].append(str(report_path))

        # ============================================
        # סוף PLACEHOLDER
        # ============================================

        result = {
            "model_path": str(model_path),
            "report_path": str(report_path),
            "model_type": "placeholder"
        }

        self._save_state("scientist_crew", "completed", result)
        logger.success("שלב 4: Scientist Crew הושלם בהצלחה")

        return result

    def _validate_scientist_outputs(self) -> bool:
        """
        שלב 5: בדיקת outputs של Scientist Crew.
        מוודא שהמודל ודוח ההערכה נוצרו.

        Returns:
            True אם כל הקבצים קיימים

        Raises:
            FileNotFoundError: אם קובץ חסר
        """
        logger.info("שלב 5: מאמת outputs של Scientist Crew...")
        self._save_state("validate_scientist", "running")

        required_files = [
            self.models_path / "model.pkl",
            self.reports_path / "evaluation_report.md"
        ]

        missing_files = []
        for file_path in required_files:
            if not file_path.exists():
                missing_files.append(str(file_path))

        if missing_files:
            raise FileNotFoundError(f"קבצים חסרים: {missing_files}")

        validation_result = {
            "model_exists": True,
            "report_exists": True
        }

        self._save_state("validate_scientist", "completed", validation_result)
        logger.success("שלב 5: אימות outputs של Scientist הושלם")

        return True

    def _finalize(self):
        """
        שלב 6: סיום ה-Pipeline.
        כותב log סופי ומעדכן את ה-state.
        """
        logger.info("שלב 6: מסיים את ה-Pipeline...")

        self.state["pipeline_end"] = datetime.now().isoformat()
        self.state["status"] = "completed"

        # חישוב זמן ריצה כולל
        if self.state["pipeline_start"]:
            start = datetime.fromisoformat(self.state["pipeline_start"])
            end = datetime.fromisoformat(self.state["pipeline_end"])
            duration = (end - start).total_seconds()
            self.state["total_duration_seconds"] = duration

        # שמירת state סופי
        self._save_state("finalize", "completed")

        # הדפסת סיכום
        logger.success("=" * 50)
        logger.success("Pipeline הושלם בהצלחה!")
        logger.success("=" * 50)
        logger.info(f"קבצים שנוצרו: {len(self.state['created_files'])}")
        for file in self.state["created_files"]:
            logger.info(f"  - {file}")
        if "total_duration_seconds" in self.state:
            logger.info(f"זמן ריצה כולל: {self.state['total_duration_seconds']:.2f} שניות")

    def run(self):
        """
        פונקציה ראשית להרצת כל ה-Pipeline.
        מריצה את כל השלבים לפי הסדר עם error handling.
        """
        logger.info("=" * 50)
        logger.info("מתחיל Amazon Sales AI Pipeline")
        logger.info("=" * 50)

        self.state["pipeline_start"] = datetime.now().isoformat()

        try:
            # שלב 1: טעינת נתונים גולמיים
            raw_data = self._load_raw_data()

            # שלב 2: הרצת Analyst Crew
            analyst_result = self._run_analyst_crew(raw_data)

            # שלב 3: אימות outputs של Analyst
            self._validate_analyst_outputs()

            # שלב 4: הרצת Scientist Crew
            scientist_result = self._run_scientist_crew(analyst_result["clean_data_path"])

            # שלב 5: אימות outputs של Scientist
            self._validate_scientist_outputs()

            # שלב 6: סיום
            self._finalize()

        except FileNotFoundError as e:
            logger.error(f"שגיאת קובץ: {e}")
            self.state["status"] = "failed"
            self.state["error"] = str(e)
            self._save_state("error", "failed", {"error_message": str(e)})
            raise

        except Exception as e:
            logger.error(f"שגיאה לא צפויה: {e}")
            self.state["status"] = "failed"
            self.state["error"] = str(e)
            self._save_state("error", "failed", {"error_message": str(e)})
            raise


# נקודת כניסה להרצה ישירה
if __name__ == "__main__":
    pipeline = AmazonSalesPipeline()
    pipeline.run()
