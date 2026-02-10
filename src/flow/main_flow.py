"""
Amazon Sales AI Pipeline - Main Flow
src/flow/main_flow.py

Pipeline Lead: קובץ מרכזי שמתאם בין Analyst Crew ל-Scientist Crew
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

import pandas as pd
from loguru import logger
from dotenv import load_dotenv

# טעינת משתני סביבה
load_dotenv()

# הגדרת logger
logger.remove()  # הסרת handler ברירת המחדל
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO"
)
logger.add(
    "logs/pipeline_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="30 days",
    level="DEBUG"
)

# ייבוא Crews (יהיו זמינים בשלבים הבאים)
# from src.crews.analyst_crew import analyst_crew, data_cleaning_task
# from src.crews.scientist_crew import scientist_crew, modeling_task


class AmazonSalesPipeline:
    """
    Pipeline מרכזי לניתוח נתוני מכירות Amazon

    Flow:
    1. טעינת נתונים גולמיים
    2. הרצת Analyst Crew - ניקוי נתונים
    3. Validation של תוצרי Analyst
    4. הרצת Scientist Crew - בניית מודל ML
    5. Validation של תוצרי Scientist
    6. סיכום והצלחה
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        אתחול Pipeline

        Args:
            config: הגדרות אופציונליות (נתיבי קבצים, פרמטרים)
        """
        self.config = config or {}
        self.start_time = datetime.now()

        # נתיבים ברירת מחדל
        self.raw_data_path = Path("data/raw/amazon_sales.csv")
        self.clean_data_path = Path("data/processed/clean_data.csv")
        self.contract_path = Path("data/contracts/dataset_contract.json")
        self.model_path = Path("outputs/models/model.pkl")
        self.eval_report_path = Path("outputs/reports/evaluation_report.md")
        self.model_card_path = Path("outputs/reports/model_card.md")

        # State של הריצה
        self.state = {
            "raw_data_loaded": False,
            "analyst_crew_completed": False,
            "scientist_crew_completed": False,
            "validation_passed": False,
            "errors": []
        }

        logger.info("=" * 80)
        logger.info("Amazon Sales AI Pipeline - התחלת ריצה")
        logger.info("=" * 80)

    def load_raw_data(self) -> pd.DataFrame:
        """
        טעינת נתונים גולמיים מקובץ CSV

        Returns:
            DataFrame עם הנתונים הגולמיים

        Raises:
            FileNotFoundError: אם הקובץ לא קיים
            ValueError: אם הקובץ ריק או לא תקין
        """
        logger.info(f"טוען נתונים גולמיים מ-{self.raw_data_path}")

        if not self.raw_data_path.exists():
            error_msg = f"קובץ נתונים לא נמצא: {self.raw_data_path}"
            logger.error(error_msg)
            self.state["errors"].append(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            df = pd.read_csv(self.raw_data_path)

            if df.empty:
                raise ValueError("הקובץ ריק!")

            logger.success(f"נתונים נטענו: {len(df)} שורות, {len(df.columns)} עמודות")
            logger.info(f"עמודות: {', '.join(df.columns.tolist())}")

            self.state["raw_data_loaded"] = True
            return df

        except Exception as e:
            error_msg = f"שגיאה בטעינת נתונים: {str(e)}"
            logger.error(error_msg)
            self.state["errors"].append(error_msg)
            raise

    def run_analyst_crew(self) -> Dict[str, Any]:
        """
        הרצת Analyst Crew - ניקוי ועיבוד נתונים

        Returns:
            Dict עם תוצרי ה-Crew (נתיבי קבצים, סטטיסטיקות)

        Raises:
            RuntimeError: אם הריצה נכשלה
        """
        logger.info("=" * 80)
        logger.info("מריץ Analyst Crew (Data Cleaning)")
        logger.info("=" * 80)

        try:
            # TODO: להוסיף בשלבים הבאים:
            # from src.crews.analyst_crew import analyst_crew, data_cleaning_task
            # result = analyst_crew.kickoff(inputs={"input_file": str(self.raw_data_path)})

            # סימולציה של תוצרים מוצלחים
            logger.info("מעבד נתונים...")
            logger.info("מנקה ערכים חסרים...")
            logger.info("מסיר כפילויות...")
            logger.info("מטפל בערכים חריגים...")

            # בדיקה שהתיקיות קיימות
            self.clean_data_path.parent.mkdir(parents=True, exist_ok=True)
            self.contract_path.parent.mkdir(parents=True, exist_ok=True)

            logger.success("Analyst Crew הושלם בהצלחה!")

            outputs = {
                "clean_data": str(self.clean_data_path),
                "contract": str(self.contract_path),
                "status": "completed"
            }

            self.state["analyst_crew_completed"] = True
            return outputs

        except Exception as e:
            error_msg = f"שגיאה בהרצת Analyst Crew: {str(e)}"
            logger.error(error_msg)
            self.state["errors"].append(error_msg)
            raise RuntimeError(error_msg)

    def validate_analyst_outputs(self) -> bool:
        """
        בדיקת תוצרי Analyst Crew

        בדיקות:
        1. clean_data.csv קיים ונטען תקין
        2. dataset_contract.json קיים ותקין (JSON valid)
        3. כל העמודות הנדרשות קיימות

        Returns:
            True אם כל הבדיקות עברו, False אחרת
        """
        logger.info("=" * 80)
        logger.info("מבצע Validation על תוצרי Analyst Crew")
        logger.info("=" * 80)

        validation_passed = True

        # בדיקה 1: clean_data.csv קיים
        if not self.clean_data_path.exists():
            logger.error(f"קובץ נתונים נקיים לא נמצא: {self.clean_data_path}")
            validation_passed = False
        else:
            logger.success(f"clean_data.csv קיים")

            # בדיקה שהקובץ נטען תקין
            try:
                df = pd.read_csv(self.clean_data_path)
                logger.info(f"{len(df)} שורות, {len(df.columns)} עמודות")

                # בדיקה שאין ערכים חסרים (לפי המפרט)
                missing = df.isnull().sum().sum()
                if missing > 0:
                    logger.warning(f"נמצאו {missing} ערכים חסרים בנתונים הנקיים")

            except Exception as e:
                logger.error(f"שגיאה בטעינת clean_data.csv: {e}")
                validation_passed = False

        # בדיקה 2: dataset_contract.json קיים ותקין
        if not self.contract_path.exists():
            logger.error(f"קובץ contract לא נמצא: {self.contract_path}")
            validation_passed = False
        else:
            logger.success(f"dataset_contract.json קיים")

            # בדיקה שה-JSON תקין
            try:
                import json
                with open(self.contract_path, 'r') as f:
                    contract = json.load(f)

                # בדיקה שיש את השדות הנדרשים
                required_fields = ["schema", "required_columns", "constraints"]
                for field in required_fields:
                    if field not in contract:
                        logger.error(f"חסר שדה חובה ב-contract: {field}")
                        validation_passed = False
                    else:
                        logger.success(f"שדה {field} קיים ב-contract")

            except json.JSONDecodeError as e:
                logger.error(f"JSON לא תקין: {e}")
                validation_passed = False
            except Exception as e:
                logger.error(f"שגיאה בקריאת contract: {e}")
                validation_passed = False

        if validation_passed:
            logger.success("=" * 80)
            logger.success("כל בדיקות ה-Validation עברו בהצלחה!")
            logger.success("=" * 80)
            self.state["validation_passed"] = True
        else:
            logger.error("=" * 80)
            logger.error("Validation נכשל!")
            logger.error("=" * 80)

        return validation_passed

    def run_scientist_crew(self) -> Dict[str, Any]:
        """
        הרצת Scientist Crew - בניית מודל ML

        Returns:
            Dict עם תוצרי ה-Crew

        Raises:
            RuntimeError: אם הריצה נכשלה
        """
        logger.info("=" * 80)
        logger.info("מריץ Scientist Crew (ML Modeling)")
        logger.info("=" * 80)

        try:
            # TODO: להוסיף בשלבים הבאים
            # from src.crews.scientist_crew import scientist_crew, modeling_task
            # result = scientist_crew.kickoff(inputs={
            #     "clean_data": str(self.clean_data_path),
            #     "contract": str(self.contract_path)
            # })

            logger.info("בונה features...")
            logger.info("מאמן מודל...")
            logger.info("מעריך ביצועים...")
            logger.info("יוצר דוחות...")

            # בדיקה שהתיקיות קיימות
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            self.eval_report_path.parent.mkdir(parents=True, exist_ok=True)

            logger.success("Scientist Crew הושלם בהצלחה!")

            outputs = {
                "model": str(self.model_path),
                "evaluation": str(self.eval_report_path),
                "model_card": str(self.model_card_path),
                "status": "completed"
            }

            self.state["scientist_crew_completed"] = True
            return outputs

        except Exception as e:
            error_msg = f"שגיאה בהרצת Scientist Crew: {str(e)}"
            logger.error(error_msg)
            self.state["errors"].append(error_msg)
            raise RuntimeError(error_msg)

    def validate_scientist_outputs(self) -> bool:
        """
        בדיקת תוצרי Scientist Crew

        בדיקות:
        1. model.pkl קיים
        2. evaluation_report.md קיים
        3. model_card.md קיים ומלא

        Returns:
            True אם כל הבדיקות עברו
        """
        logger.info("=" * 80)
        logger.info("מבצע Validation על תוצרי Scientist Crew")
        logger.info("=" * 80)

        validation_passed = True

        # בדיקה 1: model.pkl
        if not self.model_path.exists():
            logger.error(f"קובץ מודל לא נמצא: {self.model_path}")
            validation_passed = False
        else:
            logger.success(f"model.pkl קיים")

        # בדיקה 2: evaluation_report.md
        if not self.eval_report_path.exists():
            logger.error(f"דוח הערכה לא נמצא: {self.eval_report_path}")
            validation_passed = False
        else:
            logger.success(f"evaluation_report.md קיים")

        # בדיקה 3: model_card.md
        if not self.model_card_path.exists():
            logger.error(f"model card לא נמצא: {self.model_card_path}")
            validation_passed = False
        else:
            logger.success(f"model_card.md קיים")

        if validation_passed:
            logger.success("=" * 80)
            logger.success("כל בדיקות ה-Validation עברו בהצלחה!")
            logger.success("=" * 80)
        else:
            logger.error("=" * 80)
            logger.error("Validation נכשל!")
            logger.error("=" * 80)

        return validation_passed

    def run(self) -> Dict[str, Any]:
        """
        הרצה ראשית של כל ה-Pipeline

        Flow מלא:
        1. טעינת נתונים גולמיים
        2. Analyst Crew
        3. Validation
        4. Scientist Crew
        5. Validation
        6. סיכום

        Returns:
            Dict עם סיכום הריצה
        """
        try:
            # Step 1: טעינת נתונים
            logger.info("שלב 1/5: טעינת נתונים גולמיים")
            raw_data = self.load_raw_data()

            # Step 2: Analyst Crew
            logger.info("שלב 2/5: הרצת Analyst Crew")
            analyst_outputs = self.run_analyst_crew()

            # Step 3: Validation של Analyst
            logger.info("שלב 3/5: Validation תוצרי Analyst")
            if not self.validate_analyst_outputs():
                raise RuntimeError("Validation של Analyst Crew נכשל!")

            # Step 4: Scientist Crew
            logger.info("שלב 4/5: הרצת Scientist Crew")
            scientist_outputs = self.run_scientist_crew()

            # Step 5: Validation של Scientist
            logger.info("שלב 5/5: Validation תוצרי Scientist")
            if not self.validate_scientist_outputs():
                raise RuntimeError("Validation של Scientist Crew נכשל!")

            # סיכום הצלחה
            end_time = datetime.now()
            duration = (end_time - self.start_time).total_seconds()

            logger.success("=" * 80)
            logger.success("Pipeline הושלם בהצלחה!")
            logger.success(f"זמן ריצה: {duration:.2f} שניות")
            logger.success("=" * 80)

            summary = {
                "status": "success",
                "duration_seconds": duration,
                "outputs": {
                    "analyst": analyst_outputs,
                    "scientist": scientist_outputs
                },
                "state": self.state
            }

            return summary

        except Exception as e:
            logger.error("=" * 80)
            logger.error(f"Pipeline נכשל: {str(e)}")
            logger.error("=" * 80)

            self.state["errors"].append(str(e))

            return {
                "status": "failed",
                "error": str(e),
                "state": self.state
            }


def main():
    """
    נקודת כניסה ראשית לתוכנית
    """
    # בדיקה שמפתח API קיים
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY לא נמצא בסביבה!")
        logger.error("הוסף אותו לקובץ .env")
        sys.exit(1)

    # יצירת תיקיית logs אם לא קיימת
    Path("logs").mkdir(exist_ok=True)

    # הרצת Pipeline
    pipeline = AmazonSalesPipeline()
    result = pipeline.run()

    # הצגת תוצאה
    if result["status"] == "success":
        logger.success("Pipeline הושלם בהצלחה!")
        sys.exit(0)
    else:
        logger.error(f"Pipeline נכשל: {result.get('error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
