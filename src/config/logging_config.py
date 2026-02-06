"""
הגדרות Logging לפרויקט
Logging configuration for the project
"""

import sys
from pathlib import Path
from loguru import logger
from .settings import Settings


def setup_logging(log_level: str = None, log_file: Path = None) -> None:
    """
    הגדרת הלוגינג לפרויקט
    Setup logging for the project

    Args:
        log_level: רמת הלוגינג (DEBUG, INFO, WARNING, ERROR)
        log_file: נתיב לקובץ הלוגים
    """
    # הסרת handler ברירת מחדל
    logger.remove()

    # קביעת רמת לוגינג
    level = log_level or Settings.LOG_LEVEL
    file_path = log_file or Settings.LOG_FILE

    # פורמט הלוגים
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    # הוספת handler לקונסול
    logger.add(
        sys.stdout,
        format=log_format,
        level=level,
        colorize=True,
    )

    # יצירת תיקיית לוגים אם לא קיימת
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # הוספת handler לקובץ
    logger.add(
        str(file_path),
        format=log_format,
        level=level,
        rotation="10 MB",  # סיבוב קובץ כל 10MB
        retention="7 days",  # שמירת לוגים ל-7 ימים
        compression="zip",  # דחיסת לוגים ישנים
        encoding="utf-8",
    )

    logger.info("מערכת הלוגינג הופעלה בהצלחה | Logging system initialized")


def get_logger(name: str = None):
    """
    מחזיר logger עם שם מותאם
    Returns a logger with custom name

    Args:
        name: שם הלוגר (לרוב שם המודול)

    Returns:
        logger instance
    """
    if name:
        return logger.bind(name=name)
    return logger
