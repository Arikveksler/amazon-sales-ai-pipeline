"""
Pipeline Error Handler - src/utils/error_handler.py

קובץ מרכזי לטיפול בשגיאות ב-Pipeline:
- הגדרת Exceptions מותאמים
- Decorator ל-Retry אוטומטי
- פונקציה מרכזית ללוג שגיאות
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, TypeVar, Any, Dict, Optional, Tuple, Type
from functools import wraps
from datetime import datetime

from loguru import logger

try:
    from typing import ParamSpec
except ImportError:
    from typing_extensions import ParamSpec

P = ParamSpec("P")
R = TypeVar("R")


# =============================================================================
# Custom Exceptions - שגיאות מותאמות אישית
# =============================================================================

class DataValidationError(Exception):
    """
    שגיאה הקשורה ל-Validation של נתונים (קבצים, schema וכו').

    נזרקת כאשר:
    - קובץ נתונים לא עומד בדרישות ה-schema
    - חסרות עמודות נדרשות
    - ערכים לא תקינים בנתונים

    Examples:
        >>> raise DataValidationError("Missing required column: price")
        >>> raise DataValidationError("File is empty", context={"file": "data.csv"})
    """

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        """
        אתחול שגיאת Validation.

        Args:
            message: הודעת השגיאה
            context: מידע נוסף על השגיאה (קובץ, שדה, שלב וכו')
        """
        self.message = message
        self.context = context or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.context:
            return f"DataValidationError: {self.message} | context={self.context}"
        return f"DataValidationError: {self.message}"


class CrewExecutionError(Exception):
    """
    שגיאה בהרצת Crew (Analyst/Scientist) או Task כלשהו ב-CrewAI.

    נזרקת כאשר:
    - Crew נכשל בביצוע משימה
    - Timeout בהרצת Crew
    - שגיאת LLM בתוך ה-Crew

    Examples:
        >>> raise CrewExecutionError("Analyst Crew failed to complete")
        >>> raise CrewExecutionError("Timeout", context={"crew": "scientist", "timeout": 300})
    """

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        """
        אתחול שגיאת Crew.

        Args:
            message: הודעת השגיאה
            context: מידע נוסף על השגיאה (שם Crew, משימה, וכו')
        """
        self.message = message
        self.context = context or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.context:
            return f"CrewExecutionError: {self.message} | context={self.context}"
        return f"CrewExecutionError: {self.message}"


class FlowStateError(Exception):
    """
    שגיאה הקשורה למצב ה-Flow או ל-PipelineStateManager.

    נזרקת כאשר:
    - כשל בשמירת/טעינת state
    - state לא תקין (JSON corrupted)
    - ניסיון לעבור לשלב לא חוקי

    Examples:
        >>> raise FlowStateError("Failed to save state to JSON")
        >>> raise FlowStateError("Invalid step transition", context={"from": "load", "to": "finalize"})
    """

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        """
        אתחול שגיאת State.

        Args:
            message: הודעת השגיאה
            context: מידע נוסף על השגיאה (מפתח, קובץ, מעבר וכו')
        """
        self.message = message
        self.context = context or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.context:
            return f"FlowStateError: {self.message} | context={self.context}"
        return f"FlowStateError: {self.message}"


# =============================================================================
# Decorator: retry_on_failure - ניסיונות חוזרים
# =============================================================================

def retry_on_failure(
    max_retries: int = 2,
    delay_seconds: int = 5,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator שמנסה להריץ פונקציה שוב אם היא נכשלת.

    עוטף פונקציה כלשהי (למשל הרצת Crew או טעינת קובץ).
    אם יש Exception מהסוגים שברשימת exceptions:
    - רושם log אזהרה עם מספר ניסיון
    - מחכה delay_seconds
    - מנסה שוב, עד max_retries
    אם אחרי כל הניסיונות עדיין נכשל - מעלה את ה-Exception האחרון.
    אם הפונקציה מצליחה - רושם log success עם מספר הניסיון שבו הצליחה.

    Args:
        max_retries: מספר הניסיונות המקסימלי (ברירת מחדל: 2)
        delay_seconds: זמן המתנה בשניות בין ניסיונות (ברירת מחדל: 5)
        exceptions: tuple של סוגי שגיאות לתפוס (ברירת מחדל: כל Exception)

    Returns:
        הפונקציה העטופה עם לוגיקת retry

    Examples:
        >>> @retry_on_failure(max_retries=3, delay_seconds=2, exceptions=(DataValidationError,))
        ... def load_contract_with_retry(path: str) -> dict:
        ...     import json
        ...     with open(path) as f:
        ...         return json.load(f)
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            last_exception: Optional[Exception] = None

            for attempt in range(1, max_retries + 1):
                try:
                    result = func(*args, **kwargs)

                    # אם הצליח אחרי ניסיון חוזר - רושם success
                    if attempt > 1:
                        logger.success(
                            f"{func.__name__} הצליח בניסיון {attempt}/{max_retries}"
                        )

                    return result

                except exceptions as e:
                    last_exception = e

                    # אם זה לא הניסיון האחרון - מנסה שוב
                    if attempt < max_retries:
                        logger.warning(
                            f"ניסיון {attempt}/{max_retries} נכשל עבור {func.__name__}: "
                            f"{type(e).__name__}: {str(e)}. "
                            f"מנסה שוב בעוד {delay_seconds} שניות..."
                        )
                        time.sleep(delay_seconds)
                    else:
                        # ניסיון אחרון - רושם שגיאה סופית
                        logger.error(
                            f"כל {max_retries} הניסיונות נכשלו עבור {func.__name__}: "
                            f"{type(e).__name__}: {str(e)}"
                        )

            # אם הגענו לכאן - כל הניסיונות נכשלו
            raise last_exception

        return wrapper
    return decorator


# =============================================================================
# פונקציה: log_error - רישום שגיאה מפורטת
# =============================================================================

def log_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    log_file: str = "logs/errors.log"
) -> None:
    """
    רישום שגיאה מפורטת ללוג מרכזי.

    - רושם ל-loguru ברמת ERROR
    - שומר שורה מפורטת גם לקובץ errors.log (append)
    - מוסיף timestamp, סוג השגיאה, הודעה, ו-context אם קיים
    - אם השגיאה היא Custom Exception - ממזג את ה-context שלה

    Args:
        error: השגיאה שנתפסה
        context: מידע נוסף על השגיאה (שלב, קובץ, וכו')
        log_file: נתיב לקובץ הלוג (ברירת מחדל: logs/errors.log)

    Examples:
        >>> try:
        ...     validate_data()
        ... except DataValidationError as e:
        ...     log_error(e, {"step": "validation", "file": "clean_data.csv"})
    """
    # וידוא ש-context הוא dict
    merged_context: Dict[str, Any] = context.copy() if context else {}

    # אם השגיאה היא Custom Exception - ממזג את ה-context שלה
    if isinstance(error, (DataValidationError, CrewExecutionError, FlowStateError)):
        error_context = getattr(error, "context", {})
        if error_context:
            merged_context.update(error_context)

    # בניית הודעת השגיאה
    error_type = type(error).__name__
    error_message = str(error)
    timestamp = datetime.now().isoformat()

    # רישום ל-loguru
    logger.error(f"{error_type}: {error_message}")
    if merged_context:
        logger.error(f"  context: {merged_context}")

    # שמירה לקובץ errors.log
    try:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # פורמט שורה לקובץ
        context_str = f" | context={merged_context}" if merged_context else ""
        log_line = f"[{timestamp}] {error_type}: {error_message}{context_str}\n"

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_line)

    except Exception as write_error:
        logger.exception(f"שגיאה בכתיבה לקובץ לוג {log_file}: {write_error}")
