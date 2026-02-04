"""
Error Handler Module
====================
מודול לטיפול מרכזי בשגיאות עם Custom Exceptions, retry logic, ו-error handling.

מכיל:
- Custom Exceptions לסוגי שגיאות שונים
- Decorator לניסיונות חוזרים
- פונקציות עזר לטיפול בשגיאות

Author: Pipeline Lead
Date: 2026
"""

import time
import traceback
from typing import Callable, Any, Optional, Type, Tuple
from functools import wraps
from loguru import logger


# =============================================================================
# Custom Exceptions - שגיאות מותאמות אישית
# =============================================================================

class PipelineError(Exception):
    """
    שגיאת בסיס לכל שגיאות ה-Pipeline.
    כל השגיאות האחרות יורשות ממנה.

    Example:
        >>> raise PipelineError("Something went wrong in the pipeline")
    """

    def __init__(self, message: str, step: str = None, recoverable: bool = False):
        """
        אתחול שגיאת Pipeline.

        Args:
            message: הודעת השגיאה
            step: השלב בו קרתה השגיאה
            recoverable: האם אפשר להתאושש מהשגיאה
        """
        self.message = message
        self.step = step
        self.context = step  # alias לתאימות
        self.recoverable = recoverable

        # רישום אוטומטי ל-logger
        logger.error(f"PipelineError: {self}")

        super().__init__(self.message)

    def __str__(self):
        if self.step:
            return f"[{self.step}] {self.message}"
        return self.message


class DataValidationError(PipelineError):
    """
    שגיאת validation של נתונים.
    נזרקת כאשר נתונים לא עומדים בדרישות.

    Example:
        >>> raise DataValidationError("Missing required column: price")
        >>> raise DataValidationError("Invalid data type", step="clean_data.csv")
    """

    def __init__(self, message: str, step: str = None, field: str = None):
        """
        אתחול שגיאת Validation.

        Args:
            message: תיאור בעיית ה-validation
            step: שם הקובץ או השלב שבו קרתה השגיאה
            field: השדה הבעייתי (אופציונלי)
        """
        self.field = field
        logger.error(f"DataValidationError: {message}")
        super().__init__(message, step, recoverable=False)


class CrewExecutionError(PipelineError):
    """
    שגיאת הרצת Crew.
    נזרקת כאשר Crew נכשל בביצוע המשימה.

    Example:
        >>> raise CrewExecutionError("Analyst Crew failed to complete")
        >>> raise CrewExecutionError("Timeout", crew_name="scientist_crew")
    """

    def __init__(self, message: str, crew_name: str = None):
        """
        אתחול שגיאת Crew.

        Args:
            message: תיאור הכשלון
            crew_name: שם ה-Crew שנכשל
        """
        self.crew_name = crew_name
        logger.error(f"CrewExecutionError: {message}")
        super().__init__(message, step=crew_name, recoverable=True)


class FlowStateError(PipelineError):
    """
    שגיאת ניהול State.
    נזרקת כאשר יש בעיה בשמירה/טעינה של מצב ה-Pipeline.

    Example:
        >>> raise FlowStateError("Failed to save state to JSON")
        >>> raise FlowStateError("State file corrupted", state_key="stages")
    """

    def __init__(self, message: str, state_key: str = None):
        """
        אתחול שגיאת State.

        Args:
            message: תיאור הבעיה
            state_key: המפתח הבעייתי ב-state
        """
        self.state_key = state_key
        logger.error(f"FlowStateError: {message}")
        super().__init__(message, step=state_key, recoverable=False)


# =============================================================================
# Retry Decorators - ניסיונות חוזרים
# =============================================================================

def retry_on_failure(
    max_retries: int = 3,
    delay: float = 1.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None
):
    """
    Decorator שמנסה להריץ פונקציה שוב אם היא נכשלת.

    מאפשר לציין כמה פעמים לנסות, כמה לחכות בין ניסיונות,
    ואילו שגיאות לתפוס.

    Args:
        max_retries: מספר הניסיונות המקסימלי (ברירת מחדל: 3)
        delay: זמן המתנה בשניות בין ניסיונות (ברירת מחדל: 1.0)
        exceptions: tuple של סוגי שגיאות לתפוס (ברירת מחדל: כל Exception)
        on_retry: פונקציה שתקרא בכל ניסיון נכשל (אופציונלי)

    Returns:
        הפונקציה העטופה

    Example:
        >>> @retry_on_failure(max_retries=3, delay=2.0)
        ... def fetch_data():
        ...     # קוד שעלול להיכשל
        ...     pass

        >>> @retry_on_failure(exceptions=(ConnectionError, TimeoutError))
        ... def connect_to_api():
        ...     pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except exceptions as e:
                    last_exception = e

                    # אם זה לא הניסיון האחרון
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt}/{max_retries} failed for {func.__name__}: {str(e)}. "
                            f"Retrying in {delay} seconds..."
                        )

                        # קריאה לפונקציית on_retry אם הוגדרה
                        if on_retry:
                            on_retry(attempt, e)

                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {max_retries} attempts failed for {func.__name__}: {str(e)}"
                        )

            # אם הגענו לכאן - כל הניסיונות נכשלו
            raise last_exception

        return wrapper
    return decorator


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable] = None,
):
    """
    Decorator לניסיון חוזר בעת כישלון עם backoff.

    Args:
        max_attempts: מספר ניסיונות מקסימלי
        delay: המתנה בשניות בין ניסיונות
        backoff: מכפיל ההמתנה בין ניסיונות
        exceptions: סוגי שגיאות לתפוס
        on_retry: פונקציה להרצה בכל ניסיון חוזר

    Example:
        >>> @retry(max_attempts=3, delay=2, backoff=2.0)
        ... def risky_operation():
        ...     pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_delay = delay

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)

                except exceptions as e:
                    if attempt == max_attempts:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}"
                        )
                        raise

                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {e}"
                    )

                    if on_retry:
                        on_retry(attempt, e)

                    logger.info(f"Retrying in {current_delay} seconds...")
                    time.sleep(current_delay)
                    current_delay *= backoff

        return wrapper
    return decorator


# =============================================================================
# Error Handling Functions - פונקציות טיפול בשגיאות
# =============================================================================

def handle_error(
    error: Exception,
    context: str,
    reraise: bool = True,
    extra_info: Optional[dict] = None
) -> None:
    """
    טיפול מרכזי בשגיאות.
    מרכז את הלוגים, מוסיף context, ומחליט אם לזרוק שוב.

    Args:
        error: השגיאה שנתפסה
        context: הקשר - מאיפה השגיאה (שם פונקציה, שלב, וכו')
        reraise: האם לזרוק את השגיאה שוב (ברירת מחדל: True)
        extra_info: מידע נוסף לlog (אופציונלי)

    Raises:
        Exception: השגיאה המקורית אם reraise=True

    Example:
        >>> try:
        ...     risky_operation()
        ... except Exception as e:
        ...     handle_error(e, context="data_loading", reraise=False)
    """
    # בניית הודעת השגיאה
    error_type = type(error).__name__
    error_message = str(error)

    # הודעה בסיסית
    log_message = f"[{context}] {error_type}: {error_message}"

    # הוספת מידע נוסף
    if extra_info:
        info_str = ", ".join(f"{k}={v}" for k, v in extra_info.items())
        log_message += f" | Extra: {info_str}"

    # רישום השגיאה
    logger.error(log_message)

    # רישום ה-traceback למצב debug
    logger.debug(f"Traceback:\n{traceback.format_exc()}")

    # זריקה מחדש אם נדרש
    if reraise:
        raise error


def log_and_raise(
    exception_class: Type[Exception],
    message: str,
    context: Optional[str] = None
) -> None:
    """
    רושם שגיאה ל-log וזורק אותה - בפעולה אחת.
    שימושי כשרוצים ליצור exception חדש.

    Args:
        exception_class: סוג השגיאה לזרוק
        message: הודעת השגיאה
        context: הקשר נוסף (אופציונלי)

    Raises:
        exception_class: השגיאה שנוצרה

    Example:
        >>> log_and_raise(DataValidationError, "File is empty", context="clean_data.csv")
        >>> log_and_raise(ValueError, "Invalid parameter value")
    """
    # בניית הודעה מלאה
    full_message = message if not context else f"[{context}] {message}"

    # רישום
    logger.error(f"{exception_class.__name__}: {full_message}")

    # יצירה וזריקה - תומך גם ב-PipelineError וגם ב-Exception רגיל
    if issubclass(exception_class, PipelineError):
        raise exception_class(message, step=context)
    else:
        raise exception_class(full_message)


def handle_pipeline_error(error: PipelineError, state_manager: Any = None) -> None:
    """
    טיפול ספציפי בשגיאות Pipeline.
    מתעד את השגיאה ומעדכן את ה-state אם יש state_manager.

    Args:
        error: שגיאת Pipeline
        state_manager: מנהל ה-state (אופציונלי)

    Example:
        >>> try:
        ...     run_pipeline()
        ... except PipelineError as e:
        ...     handle_pipeline_error(e, state_manager)
    """
    error_type = type(error).__name__

    logger.error("=" * 50)
    logger.error(f"Pipeline Error: {error_type}")
    logger.error(f"Message: {error.message}")

    if hasattr(error, "step") and error.step:
        logger.error(f"Step: {error.step}")

    if hasattr(error, "field") and error.field:
        logger.error(f"Field: {error.field}")

    if hasattr(error, "crew_name") and error.crew_name:
        logger.error(f"Crew: {error.crew_name}")

    logger.error("=" * 50)

    # אם זו שגיאה ניתנת להתאוששות
    if error.recoverable:
        logger.info("This error may be recoverable. Consider retrying.")

    # עדכון state אם יש state_manager
    if state_manager:
        try:
            stage = error.step or "unknown"
            if hasattr(state_manager, 'add_error'):
                state_manager.add_error(stage, str(error))
            elif hasattr(state_manager, 'record_error'):
                state_manager.record_error(error_type, str(error), stage)
            state_manager.save_state()
        except Exception as e:
            logger.warning(f"Failed to update state with error: {e}")


def safe_execute(
    func: Callable,
    *args,
    default: Any = None,
    log_errors: bool = True,
    **kwargs
) -> Any:
    """
    מריץ פונקציה בצורה בטוחה - מחזיר default אם נכשלת.

    Args:
        func: הפונקציה להרצה
        *args: ארגומנטים לפונקציה
        default: ערך ברירת מחדל אם נכשל (ברירת מחדל: None)
        log_errors: האם לרשום שגיאות (ברירת מחדל: True)
        **kwargs: ארגומנטים נוספים לפונקציה

    Returns:
        תוצאת הפונקציה או default

    Example:
        >>> result = safe_execute(int, "not_a_number", default=0)
        >>> print(result)  # 0

        >>> data = safe_execute(load_json, "file.json", default={})
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger.warning(f"safe_execute: {func.__name__} failed with {type(e).__name__}: {e}")
        return default


# =============================================================================
# Context Manager - ניהול הקשר
# =============================================================================

class ErrorContext:
    """
    Context Manager לניהול שגיאות בבלוק קוד.

    Example:
        >>> with ErrorContext("loading data", reraise=False):
        ...     load_data()  # אם נכשל - רק ידווח, לא יזרוק

        >>> with ErrorContext("critical operation", reraise=True):
        ...     critical_work()  # אם נכשל - ידווח ויזרוק
    """

    def __init__(
        self,
        context: str,
        reraise: bool = True,
        exception_types: Tuple[Type[Exception], ...] = (Exception,)
    ):
        """
        אתחול context manager.

        Args:
            context: תיאור ההקשר
            reraise: האם לזרוק שגיאות (ברירת מחדל: True)
            exception_types: סוגי שגיאות לתפוס
        """
        self.context = context
        self.reraise = reraise
        self.exception_types = exception_types
        self.error: Optional[Exception] = None

    def __enter__(self):
        logger.debug(f"Entering context: {self.context}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None and isinstance(exc_val, self.exception_types):
            self.error = exc_val
            handle_error(exc_val, self.context, reraise=False)

            if self.reraise:
                return False  # תזרוק את השגיאה
            return True  # תבלע את השגיאה

        return False


# =============================================================================
# בדיקה עצמית
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Testing Error Handler Module")
    print("=" * 60 + "\n")

    # בדיקת Custom Exceptions
    print("1. Testing Custom Exceptions:")
    try:
        raise DataValidationError("Test validation error", step="test_file.csv")
    except DataValidationError as e:
        print(f"   Caught: {type(e).__name__}")

    # בדיקת retry decorator
    print("\n2. Testing retry_on_failure decorator:")

    attempt_count = 0

    @retry_on_failure(max_retries=3, delay=0.1)
    def failing_function():
        global attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ValueError("Not yet!")
        return "Success!"

    try:
        result = failing_function()
        print(f"   Result: {result} (after {attempt_count} attempts)")
    except Exception as e:
        print(f"   Failed: {e}")

    # בדיקת safe_execute
    print("\n3. Testing safe_execute:")
    result = safe_execute(int, "not_a_number", default=42, log_errors=False)
    print(f"   safe_execute result: {result}")

    # בדיקת ErrorContext
    print("\n4. Testing ErrorContext:")
    with ErrorContext("test context", reraise=False):
        raise ValueError("Test error in context")
    print("   ErrorContext handled the error")

    print("\n" + "=" * 60)
    print("Error Handler test complete")
    print("=" * 60 + "\n")
