"""
הגדרות גלובליות לפרויקט Amazon Sales AI Pipeline
Global settings for the Amazon Sales AI Pipeline project
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# טעינת משתני סביבה מקובץ .env
load_dotenv()


class Settings:
    """
    מחלקת הגדרות מרכזית לפרויקט
    Central settings class for the project
    """

    # נתיבי תיקיות בסיסיות
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    SRC_DIR = BASE_DIR / "src"
    DATA_DIR = BASE_DIR / "data"
    OUTPUTS_DIR = BASE_DIR / "outputs"

    # נתיבי נתונים
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    CONTRACTS_DIR = DATA_DIR / "contracts"

    # נתיבי תוצרים
    REPORTS_DIR = OUTPUTS_DIR / "reports"
    MODELS_DIR = OUTPUTS_DIR / "models"
    FEATURES_DIR = OUTPUTS_DIR / "features"

    # קבצים ספציפיים
    RAW_DATA_FILE = RAW_DATA_DIR / "amazon_sales.csv"
    CLEAN_DATA_FILE = PROCESSED_DATA_DIR / "clean_data.csv"
    DATASET_CONTRACT_FILE = CONTRACTS_DIR / "dataset_contract.json"

    # קבצי תוצרים
    EDA_REPORT_FILE = REPORTS_DIR / "eda_report.html"
    INSIGHTS_FILE = REPORTS_DIR / "insights.md"
    EVALUATION_REPORT_FILE = REPORTS_DIR / "evaluation_report.md"
    MODEL_CARD_FILE = REPORTS_DIR / "model_card.md"
    MODEL_FILE = MODELS_DIR / "model.pkl"
    FEATURES_FILE = FEATURES_DIR / "features.csv"

    # הגדרות API (אם יש)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    # הגדרות logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = BASE_DIR / "logs" / "pipeline.log"

    # הגדרות retry
    MAX_RETRIES = 3
    RETRY_DELAY = 5  # שניות

    @classmethod
    def ensure_directories(cls):
        """
        יצירת כל התיקיות הנדרשות אם לא קיימות
        Creates all required directories if they don't exist
        """
        directories = [
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.CONTRACTS_DIR,
            cls.REPORTS_DIR,
            cls.MODELS_DIR,
            cls.FEATURES_DIR,
            cls.LOG_FILE.parent,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_all_paths(cls) -> dict:
        """
        מחזיר מילון של כל הנתיבים החשובים
        Returns a dictionary of all important paths
        """
        return {
            "base_dir": str(cls.BASE_DIR),
            "data_dir": str(cls.DATA_DIR),
            "outputs_dir": str(cls.OUTPUTS_DIR),
            "raw_data_file": str(cls.RAW_DATA_FILE),
            "clean_data_file": str(cls.CLEAN_DATA_FILE),
            "contract_file": str(cls.DATASET_CONTRACT_FILE),
            "model_file": str(cls.MODEL_FILE),
        }


# יצירת instance גלובלי
settings = Settings()
