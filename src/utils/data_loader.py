"""
טעינת נתונים
Data loading utilities

פונקציות לטעינת נתונים מסוגים שונים
Functions for loading data from various sources
"""

from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
from loguru import logger


class DataLoader:
    """
    מחלקה לטעינת נתונים
    Class for loading data
    """

    def __init__(self):
        """אתחול ה-DataLoader"""
        self.logger = logger.bind(name="DataLoader")

    def load_csv(
        self,
        file_path: Path,
        encoding: str = "utf-8",
        **kwargs,
    ) -> Optional[pd.DataFrame]:
        """
        טעינת קובץ CSV

        Args:
            file_path: נתיב לקובץ
            encoding: קידוד הקובץ
            **kwargs: פרמטרים נוספים ל-pandas

        Returns:
            DataFrame או None אם נכשל
        """
        file_path = Path(file_path)
        self.logger.info(f"Loading CSV: {file_path}")

        if not file_path.exists():
            self.logger.error(f"File not found: {file_path}")
            return None

        try:
            df = pd.read_csv(file_path, encoding=encoding, **kwargs)
            self.logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            return df

        except Exception as e:
            self.logger.error(f"Error loading CSV: {e}")
            return None

    def load_raw_data(self, file_path: Path) -> Optional[pd.DataFrame]:
        """
        טעינת נתונים גולמיים עם ניקוי בסיסי

        Args:
            file_path: נתיב לקובץ

        Returns:
            DataFrame או None אם נכשל
        """
        df = self.load_csv(file_path)

        if df is None:
            return None

        # ניקוי בסיסי - הסרת רווחים מעמודות
        df.columns = df.columns.str.strip()

        self.logger.info("Raw data loaded with basic cleaning")
        return df

    def get_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        קבלת מידע על DataFrame

        Args:
            df: ה-DataFrame

        Returns:
            מילון עם מידע על הנתונים
        """
        return {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "memory_usage": df.memory_usage(deep=True).sum(),
        }

    def save_csv(
        self,
        df: pd.DataFrame,
        file_path: Path,
        index: bool = False,
        encoding: str = "utf-8",
    ) -> bool:
        """
        שמירת DataFrame לקובץ CSV

        Args:
            df: ה-DataFrame לשמירה
            file_path: נתיב לקובץ
            index: האם לשמור את האינדקס
            encoding: קידוד הקובץ

        Returns:
            True אם השמירה הצליחה
        """
        file_path = Path(file_path)
        self.logger.info(f"Saving CSV: {file_path}")

        try:
            # יצירת תיקייה אם לא קיימת
            file_path.parent.mkdir(parents=True, exist_ok=True)

            df.to_csv(file_path, index=index, encoding=encoding)
            self.logger.info(f"Saved {len(df)} rows to {file_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error saving CSV: {e}")
            return False
