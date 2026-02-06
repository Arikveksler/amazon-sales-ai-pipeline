"""
ניהול קבצים
File management utilities

פונקציות לניהול קבצים ותיקיות
Functions for file and directory management
"""

import json
import shutil
from pathlib import Path
from typing import Any, Optional, List
from loguru import logger


class FileManager:
    """
    מחלקה לניהול קבצים
    Class for file management
    """

    def __init__(self):
        """אתחול ה-FileManager"""
        self.logger = logger.bind(name="FileManager")

    def ensure_directory(self, path: Path) -> bool:
        """
        יצירת תיקייה אם לא קיימת

        Args:
            path: נתיב לתיקייה

        Returns:
            True אם התיקייה קיימת/נוצרה
        """
        path = Path(path)

        try:
            path.mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            self.logger.error(f"Failed to create directory {path}: {e}")
            return False

    def file_exists(self, path: Path) -> bool:
        """
        בדיקה אם קובץ קיים

        Args:
            path: נתיב לקובץ

        Returns:
            True אם הקובץ קיים
        """
        return Path(path).exists()

    def read_json(self, file_path: Path) -> Optional[dict]:
        """
        קריאת קובץ JSON

        Args:
            file_path: נתיב לקובץ

        Returns:
            תוכן הקובץ כמילון או None
        """
        file_path = Path(file_path)
        self.logger.debug(f"Reading JSON: {file_path}")

        if not file_path.exists():
            self.logger.error(f"JSON file not found: {file_path}")
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in {file_path}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error reading JSON: {e}")
            return None

    def write_json(self, data: Any, file_path: Path, indent: int = 2) -> bool:
        """
        כתיבת מילון לקובץ JSON

        Args:
            data: הנתונים לכתיבה
            file_path: נתיב לקובץ
            indent: רווחים להזחה

        Returns:
            True אם הכתיבה הצליחה
        """
        file_path = Path(file_path)
        self.logger.debug(f"Writing JSON: {file_path}")

        try:
            self.ensure_directory(file_path.parent)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=indent, ensure_ascii=False)

            return True
        except Exception as e:
            self.logger.error(f"Error writing JSON: {e}")
            return False

    def read_text(self, file_path: Path) -> Optional[str]:
        """
        קריאת קובץ טקסט

        Args:
            file_path: נתיב לקובץ

        Returns:
            תוכן הקובץ או None
        """
        file_path = Path(file_path)

        if not file_path.exists():
            self.logger.error(f"Text file not found: {file_path}")
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Error reading text file: {e}")
            return None

    def write_text(self, content: str, file_path: Path) -> bool:
        """
        כתיבת טקסט לקובץ

        Args:
            content: התוכן לכתיבה
            file_path: נתיב לקובץ

        Returns:
            True אם הכתיבה הצליחה
        """
        file_path = Path(file_path)

        try:
            self.ensure_directory(file_path.parent)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            return True
        except Exception as e:
            self.logger.error(f"Error writing text file: {e}")
            return False

    def copy_file(self, src: Path, dst: Path) -> bool:
        """
        העתקת קובץ

        Args:
            src: קובץ מקור
            dst: יעד

        Returns:
            True אם ההעתקה הצליחה
        """
        src = Path(src)
        dst = Path(dst)

        if not src.exists():
            self.logger.error(f"Source file not found: {src}")
            return False

        try:
            self.ensure_directory(dst.parent)
            shutil.copy2(src, dst)
            self.logger.debug(f"Copied {src} -> {dst}")
            return True
        except Exception as e:
            self.logger.error(f"Error copying file: {e}")
            return False

    def delete_file(self, file_path: Path) -> bool:
        """
        מחיקת קובץ

        Args:
            file_path: נתיב לקובץ

        Returns:
            True אם המחיקה הצליחה
        """
        file_path = Path(file_path)

        if not file_path.exists():
            return True  # כבר לא קיים

        try:
            file_path.unlink()
            self.logger.debug(f"Deleted: {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting file: {e}")
            return False

    def list_files(self, directory: Path, pattern: str = "*") -> List[Path]:
        """
        רשימת קבצים בתיקייה

        Args:
            directory: תיקייה לסריקה
            pattern: תבנית חיפוש (glob)

        Returns:
            רשימת נתיבים
        """
        directory = Path(directory)

        if not directory.exists():
            return []

        return list(directory.glob(pattern))

    def get_file_size(self, file_path: Path) -> int:
        """
        קבלת גודל קובץ בבייטים

        Args:
            file_path: נתיב לקובץ

        Returns:
            גודל הקובץ בבייטים או 0
        """
        file_path = Path(file_path)

        if not file_path.exists():
            return 0

        return file_path.stat().st_size
