"""
Pipeline State Manager - src/flow/state_manager.py

ניהול מצב ריצת ה-Pipeline
"""

import json
import uuid
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from loguru import logger


class PipelineStateManager:
    """
    מנהל מצב ריצת Pipeline

    שומר state ב-JSON, כולל:
    - איזה שלבים הושלמו
    - timestamps של כל שלב
    - נתיבי קבצים שנוצרו
    - שגיאות שהיו
    - metadata נוסף

    Examples:
        >>> manager = PipelineStateManager()
        >>> manager.update_step("load_data", "completed", {"rows": 1465})
        >>> last_step = manager.get_last_completed_step()
        >>> print(last_step)
        "load_data"
    """

    def __init__(self, state_file: str = "pipeline_state.json"):
        """
        אתחול State Manager

        Args:
            state_file: נתיב לקובץ JSON שישמור את ה-state
        """
        self.state_file = Path(state_file)
        self.state: Dict[str, Any] = self._initialize_state()

        # אם יש state קיים - טען אותו
        if self.state_file.exists():
            logger.info(f"טוען state קיים מ-{self.state_file}")
            self.load_state()
        else:
            logger.info(f"יוצר state חדש")
            self.save_state()

    def _initialize_state(self) -> Dict[str, Any]:
        """
        יצירת state ריק התחלתי

        Returns:
            Dict עם המבנה הבסיסי של state
        """
        return {
            "run_id": str(uuid.uuid4()),
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "status": "initialized",  # initialized, running, completed, failed
            "steps": {},
            "outputs": {},
            "errors": [],
            "metadata": {}
        }

    def save_state(self) -> None:
        """
        שמירת state לקובץ JSON

        מעדכן אוטומטית את last_updated לזמן הנוכחי

        Raises:
            IOError: אם נכשלה השמירה
        """
        try:
            # עדכון timestamp
            self.state["last_updated"] = datetime.now().isoformat()

            # יצירת תיקייה אם לא קיימת
            self.state_file.parent.mkdir(parents=True, exist_ok=True)

            # שמירה לקובץ עם indent יפה
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)

            logger.debug(f"State נשמר ל-{self.state_file}")

        except Exception as e:
            error_msg = f"שגיאה בשמירת state: {str(e)}"
            logger.error(error_msg)
            raise IOError(error_msg)

    def load_state(self) -> Dict[str, Any]:
        """
        טעינת state מקובץ JSON

        Returns:
            Dict עם ה-state שנטען

        Raises:
            FileNotFoundError: אם הקובץ לא קיים
            json.JSONDecodeError: אם ה-JSON לא תקין
        """
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                self.state = json.load(f)

            logger.debug(f"State נטען מ-{self.state_file}")
            logger.info(f"Run ID: {self.state['run_id']}")
            logger.info(f"סטטוס: {self.state['status']}")

            return self.state

        except FileNotFoundError:
            error_msg = f"קובץ state לא נמצא: {self.state_file}"
            logger.error(error_msg)
            raise
        except json.JSONDecodeError as e:
            error_msg = f"JSON לא תקין בקובץ state: {str(e)}"
            logger.error(error_msg)
            raise
        except Exception as e:
            error_msg = f"שגיאה בטעינת state: {str(e)}"
            logger.error(error_msg)
            raise

    def update_step(
        self,
        step_name: str,
        status: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        עדכון סטטוס של שלב בודד

        Args:
            step_name: שם השלב (load_data, analyst_crew, scientist_crew, etc.)
            status: סטטוס השלב (running, completed, failed)
            details: מידע נוסף על השלב (אופציונלי)

        Examples:
            >>> manager.update_step("load_data", "completed", {"rows": 1465})
            >>> manager.update_step("analyst_crew", "running")
            >>> manager.update_step("validation", "failed", {"error": "missing columns"})
        """
        logger.info(f"מעדכן שלב: {step_name} -> {status}")

        # יצירת entry לשלב
        step_entry = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }

        # עדכון ב-state
        self.state["steps"][step_name] = step_entry

        # עדכון סטטוס כללי של ה-Pipeline
        if status == "running":
            self.state["status"] = "running"
        elif status == "failed":
            self.state["status"] = "failed"

        # שמירה אוטומטית
        self.save_state()

        logger.success(f"שלב {step_name} עודכן ל-{status}")

    def get_last_completed_step(self) -> Optional[str]:
        """
        קבלת השלב האחרון שהושלם בהצלחה

        Returns:
            שם השלב האחרון שהושלם, או None אם אין

        Examples:
            >>> last = manager.get_last_completed_step()
            >>> if last:
            >>>     print(f"Last completed: {last}")
        """
        completed_steps = [
            step_name
            for step_name, step_data in self.state["steps"].items()
            if step_data["status"] == "completed"
        ]

        if not completed_steps:
            logger.debug("אין שלבים שהושלמו")
            return None

        # השלב האחרון הוא זה עם ה-timestamp האחרון
        last_step = max(
            completed_steps,
            key=lambda s: self.state["steps"][s]["timestamp"]
        )

        logger.info(f"שלב אחרון שהושלם: {last_step}")
        return last_step

    def add_output(self, output_name: str, output_path: str) -> None:
        """
        הוספת תוצר (output) ל-state

        Args:
            output_name: שם התוצר (clean_data, model, report, etc.)
            output_path: נתיב מלא לקובץ

        Examples:
            >>> manager.add_output("clean_data", "data/processed/clean_data.csv")
            >>> manager.add_output("model", "outputs/models/model.pkl")
        """
        self.state["outputs"][output_name] = {
            "path": output_path,
            "created_at": datetime.now().isoformat()
        }

        self.save_state()
        logger.info(f"תוצר נוסף: {output_name} -> {output_path}")

    def add_error(self, error_message: str, step_name: Optional[str] = None) -> None:
        """
        הוספת שגיאה ל-state

        Args:
            error_message: תיאור השגיאה
            step_name: שם השלב בו התרחשה השגיאה (אופציונלי)

        Examples:
            >>> manager.add_error("Validation failed: missing columns", "validation")
        """
        error_entry = {
            "message": error_message,
            "timestamp": datetime.now().isoformat(),
            "step": step_name
        }

        self.state["errors"].append(error_entry)
        self.save_state()

        logger.error(f"שגיאה נוספה ל-state: {error_message}")

    def mark_completed(self) -> None:
        """
        סימון ה-Pipeline כהושלם בהצלחה

        מעדכן את הסטטוס ל-completed ושומר
        """
        self.state["status"] = "completed"
        self.state["completed_at"] = datetime.now().isoformat()
        self.save_state()

        logger.success("Pipeline הושלם בהצלחה!")

    def mark_failed(self, reason: str) -> None:
        """
        סימון ה-Pipeline כנכשל

        Args:
            reason: סיבת הכשל
        """
        self.state["status"] = "failed"
        self.state["failed_at"] = datetime.now().isoformat()
        self.state["failure_reason"] = reason
        self.add_error(reason)
        self.save_state()

        logger.error(f"Pipeline נכשל: {reason}")

    def reset_state(self) -> None:
        """
        איפוס מלא של ה-state

        יוצר state חדש לגמרי (run_id חדש)
        שימושי כשרוצים להתחיל מחדש לגמרי

        Warning:
            פעולה זו מוחקת את כל ההיסטוריה!
        """
        logger.warning("מאפס state לגמרי!")

        self.state = self._initialize_state()
        self.save_state()

        logger.info("State אופס בהצלחה")

    def get_state(self) -> Dict[str, Any]:
        """
        קבלת ה-state המלא

        Returns:
            Dict עם כל ה-state
        """
        return self.state.copy()

    def get_summary(self) -> str:
        """
        קבלת סיכום טקסטואלי של ה-state

        Returns:
            str עם סיכום נחמד וקריא
        """
        summary_lines = [
            "=" * 60,
            "Pipeline State Summary",
            "=" * 60,
            f"Run ID: {self.state['run_id']}",
            f"Created: {self.state['created_at']}",
            f"Last Updated: {self.state['last_updated']}",
            f"Status: {self.state['status']}",
            "",
            f"Completed Steps: {len([s for s in self.state['steps'].values() if s['status'] == 'completed'])}",
            f"Running Steps: {len([s for s in self.state['steps'].values() if s['status'] == 'running'])}",
            f"Failed Steps: {len([s for s in self.state['steps'].values() if s['status'] == 'failed'])}",
            "",
            f"Outputs: {len(self.state['outputs'])}",
            f"Errors: {len(self.state['errors'])}",
            "=" * 60
        ]

        return "\n".join(summary_lines)

    def __repr__(self) -> str:
        """String representation"""
        return f"PipelineStateManager(run_id={self.state['run_id']}, status={self.state['status']})"


# Alias לתאימות אחורה
StateManager = PipelineStateManager
