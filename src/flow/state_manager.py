"""
State Manager Module
====================
מחלקה לניהול מצב הריצה של ה-Pipeline, עם שמירה ל-JSON.

שומר:
- איזה שלבים הושלמו
- נתיבי קבצים שנוצרו (artifacts)
- timestamps
- שגיאות

Author: Pipeline Lead
Date: 2026
"""

import json
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from loguru import logger


class StateManager:
    """
    מנהל מצב הריצה של Pipeline.
    שומר ל-JSON את כל השלבים, קבצים, שגיאות.

    Example:
        >>> sm = StateManager("outputs/pipeline_state.json")
        >>> sm.update_stage("load_data", "completed", "Loaded 1465 rows")
        >>> sm.add_artifact("clean_data", "data/processed/clean_data.csv")
        >>> sm.save_state()
    """

    def __init__(self, state_file: str = "outputs/pipeline_state.json"):
        """
        אתחול מנהל המצב.

        Args:
            state_file: נתיב לקובץ ה-state (ברירת מחדל: outputs/pipeline_state.json)
        """
        self.logger = logger.bind(name="StateManager")
        self.state_file = Path(state_file)

        # אתחול state חדש
        self.initialize_state()

        self.logger.info(f"StateManager initialized | pipeline_id: {self.state['pipeline_id']}")

    def initialize_state(self) -> None:
        """
        יצירת state חדש עם ערכי התחלה.
        נקרא אוטומטית באתחול.
        """
        self.state: Dict[str, Any] = {
            "pipeline_id": self._generate_pipeline_id(),
            "start_time": datetime.now().isoformat(),
            "last_update": datetime.now().isoformat(),
            "status": "running",
            "stages": {},
            "artifacts": {},
            "errors": []
        }

    def _generate_pipeline_id(self) -> str:
        """
        יצירת מזהה ייחודי ל-Pipeline.

        Returns:
            מזהה בפורמט: YYYYMMDD_HHMMSS_XXXX
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = str(uuid.uuid4())[:4]
        return f"{timestamp}_{short_uuid}"

    def load_state(self) -> Dict[str, Any]:
        """
        טעינת state מקובץ JSON.
        אם הקובץ לא קיים - מחזיר את ה-state הנוכחי.

        Returns:
            ה-state שנטען

        Example:
            >>> sm = StateManager()
            >>> state = sm.load_state()
            >>> print(state["status"])
        """
        if not self.state_file.exists():
            self.logger.warning(f"State file not found: {self.state_file}")
            return self.state

        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                self.state = json.load(f)
            self.logger.info(f"State loaded from: {self.state_file}")
            return self.state

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse state JSON: {e}")
            return self.state

        except Exception as e:
            self.logger.error(f"Failed to load state: {e}")
            return self.state

    def save_state(self) -> None:
        """
        שמירת state לקובץ JSON.
        יוצר את התיקייה אוטומטית אם לא קיימת.

        Example:
            >>> sm = StateManager()
            >>> sm.update_stage("step1", "completed")
            >>> sm.save_state()
        """
        try:
            # עדכון last_update
            self.state["last_update"] = datetime.now().isoformat()

            # יצירת תיקייה אם לא קיימת
            self.state_file.parent.mkdir(parents=True, exist_ok=True)

            # שמירה
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)

            self.logger.debug(f"State saved to: {self.state_file}")

        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")

    def update_stage(self, stage_name: str, status: str, message: str = "") -> None:
        """
        עדכון סטטוס של שלב.

        Args:
            stage_name: שם השלב
            status: סטטוס (pending/in_progress/completed/failed/skipped)
            message: הודעה נוספת (אופציונלי)

        Example:
            >>> sm.update_stage("load_data", "completed", "Loaded 1465 rows")
            >>> sm.update_stage("analyst_crew", "in_progress")
        """
        self.state["stages"][stage_name] = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "message": message
        }
        self.logger.debug(f"Stage '{stage_name}' -> {status}")

    def update_step(self, step_name: str, status: str, metadata: Optional[Dict] = None) -> None:
        """
        עדכון סטטוס שלב (alias ל-update_stage לתאימות).

        Args:
            step_name: שם השלב
            status: סטטוס
            metadata: מידע נוסף (אופציונלי)
        """
        message = str(metadata) if metadata else ""
        self.update_stage(step_name, status, message)

    def add_artifact(self, artifact_name: str, file_path: str) -> None:
        """
        הוספת artifact שנוצר.

        Args:
            artifact_name: שם ה-artifact
            file_path: נתיב לקובץ

        Example:
            >>> sm.add_artifact("clean_data", "data/processed/clean_data.csv")
            >>> sm.add_artifact("model", "outputs/models/model.pkl")
        """
        self.state["artifacts"][artifact_name] = {
            "path": file_path,
            "created_at": datetime.now().isoformat(),
            "exists": Path(file_path).exists()
        }
        self.logger.debug(f"Artifact added: {artifact_name} -> {file_path}")

    def add_error(self, stage: str, error_message: str) -> None:
        """
        רישום שגיאה.

        Args:
            stage: השלב בו קרתה השגיאה
            error_message: הודעת השגיאה

        Example:
            >>> sm.add_error("analyst_crew", "Crew failed with timeout")
        """
        error_record = {
            "stage": stage,
            "message": error_message,
            "timestamp": datetime.now().isoformat()
        }
        self.state["errors"].append(error_record)
        self.logger.error(f"Error recorded in '{stage}': {error_message}")

    def record_error(self, error_type: str, message: str, step: Optional[str] = None) -> None:
        """
        תיעוד שגיאה (alias ל-add_error לתאימות).

        Args:
            error_type: סוג השגיאה
            message: הודעת השגיאה
            step: שלב בו קרתה השגיאה
        """
        stage = step or "unknown"
        full_message = f"{error_type}: {message}"
        self.add_error(stage, full_message)

    def get_stage_status(self, stage_name: str) -> Optional[str]:
        """
        קבלת סטטוס של שלב.

        Args:
            stage_name: שם השלב

        Returns:
            הסטטוס או None אם לא קיים

        Example:
            >>> status = sm.get_stage_status("load_data")
            >>> print(status)  # "completed"
        """
        stage = self.state["stages"].get(stage_name)
        return stage["status"] if stage else None

    def get_step_status(self, step_name: str) -> Optional[str]:
        """
        קבלת סטטוס שלב (alias ל-get_stage_status).
        """
        return self.get_stage_status(step_name)

    def is_stage_complete(self, stage_name: str) -> bool:
        """
        בדיקה אם שלב הושלם.

        Args:
            stage_name: שם השלב

        Returns:
            True אם השלב הושלם

        Example:
            >>> if sm.is_stage_complete("load_data"):
            ...     print("Data loaded!")
        """
        return self.get_stage_status(stage_name) == "completed"

    def is_step_completed(self, step_name: str) -> bool:
        """
        בדיקה אם שלב הושלם (alias ל-is_stage_complete).
        """
        return self.is_stage_complete(step_name)

    def get_artifact_path(self, name: str) -> Optional[str]:
        """
        קבלת נתיב artifact.

        Args:
            name: שם ה-artifact

        Returns:
            נתיב לקובץ או None
        """
        artifact = self.state["artifacts"].get(name)
        return artifact["path"] if artifact else None

    def get_summary(self) -> Dict[str, Any]:
        """
        סיכום המצב.

        Returns:
            dict עם סיכום הריצה

        Example:
            >>> summary = sm.get_summary()
            >>> print(summary["steps"])
        """
        # חישוב משך הריצה
        start = datetime.fromisoformat(self.state["start_time"])
        now = datetime.now()
        duration = (now - start).total_seconds()

        return {
            "pipeline_id": self.state["pipeline_id"],
            "status": self.state["status"],
            "start_time": self.state["start_time"],
            "last_update": self.state["last_update"],
            "duration_seconds": duration,
            "steps": {
                name: stage["status"]
                for name, stage in self.state["stages"].items()
            },
            "artifacts": list(self.state["artifacts"].keys()),
            "errors": [e["message"] for e in self.state["errors"]]
        }

    def complete(self, success: bool = True) -> None:
        """
        סימון הריצה כהושלמה.

        Args:
            success: האם הריצה הצליחה

        Example:
            >>> sm.complete(success=True)
        """
        self.state["status"] = "completed" if success else "failed"
        self.state["end_time"] = datetime.now().isoformat()
        self.save_state()
        self.logger.info(f"Pipeline completed with status: {self.state['status']}")

    def reset(self) -> None:
        """
        איפוס המצב לריצה חדשה.

        Example:
            >>> sm.reset()  # מתחיל ריצה חדשה
        """
        self.initialize_state()
        self.logger.info("State reset for new run")


# =============================================================================
# Alias לתאימות אחורה
# =============================================================================

PipelineStateManager = StateManager


# =============================================================================
# בדיקה עצמית
# =============================================================================

if __name__ == "__main__":
    import tempfile
    import os

    print("\n" + "=" * 60)
    print("Testing State Manager Module")
    print("=" * 60 + "\n")

    # יצירת קובץ זמני לבדיקה
    temp_dir = tempfile.mkdtemp()
    state_file = os.path.join(temp_dir, "test_state.json")

    # בדיקת אתחול
    print("1. Testing initialization:")
    sm = StateManager(state_file)
    print(f"   Pipeline ID: {sm.state['pipeline_id']}")

    # בדיקת עדכון שלבים
    print("\n2. Testing stage updates:")
    sm.update_stage("load_data", "completed", "Loaded 100 rows")
    sm.update_stage("analyst_crew", "in_progress")
    print(f"   load_data status: {sm.get_stage_status('load_data')}")
    print(f"   analyst_crew status: {sm.get_stage_status('analyst_crew')}")

    # בדיקת artifacts
    print("\n3. Testing artifacts:")
    sm.add_artifact("clean_data", "data/processed/clean_data.csv")
    print(f"   Added artifact: clean_data")

    # בדיקת שגיאות
    print("\n4. Testing error recording:")
    sm.add_error("test_stage", "This is a test error")
    print(f"   Errors: {len(sm.state['errors'])}")

    # בדיקת שמירה וטעינה
    print("\n5. Testing save/load:")
    sm.save_state()
    print(f"   Saved to: {state_file}")

    sm2 = StateManager(state_file)
    sm2.load_state()
    print(f"   Loaded pipeline_id: {sm2.state['pipeline_id']}")

    # בדיקת סיכום
    print("\n6. Testing summary:")
    summary = sm.get_summary()
    print(f"   Status: {summary['status']}")
    print(f"   Steps: {summary['steps']}")
    print(f"   Artifacts: {summary['artifacts']}")

    # ניקוי
    os.remove(state_file)
    os.rmdir(temp_dir)

    print("\n" + "=" * 60)
    print("State Manager test complete")
    print("=" * 60 + "\n")
