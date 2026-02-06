"""
בדיקות עבור ה-Flow הראשי
Tests for the main Flow

בדיקות:
1. Happy Path - הריצה המושלמת
2. Missing Data - מה קורה אם הקובץ לא קיים
3. Invalid Contract - מה קורה אם ה-contract שבור
4. Crew Failure - מה קורה אם ה-Crew נכשל
"""

import pytest
import json
import tempfile
from pathlib import Path
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.flow.main_flow import AmazonSalesPipeline
from src.flow.state_manager import StateManager
from src.config.settings import Settings


class TestAmazonSalesPipeline:
    """בדיקות עבור ה-Pipeline הראשי"""

    @pytest.fixture
    def temp_dir(self):
        """יצירת תיקייה זמנית לבדיקות"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_data(self, temp_dir):
        """יצירת נתונים לדוגמה"""
        # יצירת CSV לדוגמה
        data = {
            "product_id": ["P001", "P002", "P003"],
            "product_name": ["Product A", "Product B", "Product C"],
            "category": ["Electronics", "Books", "Clothing"],
            "discounted_price": [100.0, 50.0, 75.0],
            "actual_price": [120.0, 60.0, 90.0],
            "rating": [4.5, 3.8, 4.2],
        }
        df = pd.DataFrame(data)

        # שמירת הקובץ
        raw_data_path = temp_dir / "raw" / "amazon_sales.csv"
        raw_data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw_data_path, index=False)

        return raw_data_path

    @pytest.fixture
    def sample_contract(self, temp_dir):
        """יצירת חוזה לדוגמה"""
        contract = {
            "schema": {
                "product_id": "string",
                "product_name": "string",
                "category": "string",
                "discounted_price": "float",
                "actual_price": "float",
                "rating": "float",
            },
            "required_columns": ["product_id", "category", "rating"],
            "constraints": {
                "rating": {"min": 0, "max": 5},
                "discounted_price": {"min": 0},
            },
        }

        contract_path = temp_dir / "contracts" / "dataset_contract.json"
        contract_path.parent.mkdir(parents=True, exist_ok=True)
        with open(contract_path, "w") as f:
            json.dump(contract, f)

        return contract_path

    def test_pipeline_initialization(self):
        """
        בדיקה: אתחול Pipeline
        Test: Pipeline initialization
        """
        pipeline = AmazonSalesPipeline()
        assert pipeline is not None
        assert pipeline.validator is not None
        assert pipeline.state_manager is not None

    def test_state_manager_operations(self, temp_dir):
        """
        בדיקה: פעולות State Manager
        Test: State Manager operations
        """
        state_file = temp_dir / "test_state.json"
        sm = StateManager(state_file)

        # בדיקת עדכון שלב
        sm.update_step("test_step", "in_progress")
        assert sm.get_step_status("test_step") == "in_progress"

        # בדיקת הוספת artifact
        sm.add_artifact("test_artifact", str(temp_dir / "test.csv"))
        assert sm.get_artifact_path("test_artifact") is not None

        # בדיקת שמירה וטעינה
        sm.save_state()
        assert state_file.exists()

        new_sm = StateManager(state_file)
        assert new_sm.load_state()

    def test_missing_raw_data(self, temp_dir):
        """
        בדיקה: מה קורה כשהנתונים הגולמיים חסרים
        Test: What happens when raw data is missing

        צריך להחזיר שגיאה ברורה
        """
        from src.flow.validators import DataValidator

        validator = DataValidator()
        non_existent_path = temp_dir / "non_existent.csv"

        is_valid, message = validator.validate_raw_data(non_existent_path)

        assert is_valid is False
        assert "not found" in message.lower()

    def test_valid_raw_data(self, sample_data):
        """
        בדיקה: validation של נתונים תקינים
        Test: Validation of valid data
        """
        from src.flow.validators import DataValidator

        validator = DataValidator()
        is_valid, message = validator.validate_raw_data(sample_data)

        assert is_valid is True
        assert "rows" in message.lower()

    def test_invalid_contract(self, temp_dir):
        """
        בדיקה: מה קורה כשה-contract לא תקין
        Test: What happens when contract is invalid
        """
        from src.flow.validators import DataValidator

        # יצירת JSON לא תקין
        invalid_contract_path = temp_dir / "invalid_contract.json"
        with open(invalid_contract_path, "w") as f:
            f.write("{ invalid json }")

        validator = DataValidator()
        is_valid, message = validator.validate_dataset_contract(invalid_contract_path)

        assert is_valid is False
        assert "json" in message.lower()

    def test_missing_contract_fields(self, temp_dir):
        """
        בדיקה: חוזה עם שדות חסרים
        Test: Contract with missing fields
        """
        from src.flow.validators import DataValidator

        # יצירת חוזה עם שדות חסרים
        incomplete_contract = {"schema": {"field1": "string"}}
        contract_path = temp_dir / "incomplete_contract.json"
        with open(contract_path, "w") as f:
            json.dump(incomplete_contract, f)

        validator = DataValidator()
        is_valid, message = validator.validate_dataset_contract(contract_path)

        assert is_valid is False
        assert "missing" in message.lower()

    def test_valid_contract(self, sample_contract):
        """
        בדיקה: חוזה תקין
        Test: Valid contract
        """
        from src.flow.validators import DataValidator

        validator = DataValidator()
        is_valid, message = validator.validate_dataset_contract(sample_contract)

        assert is_valid is True

    def test_data_against_contract(self, sample_data, sample_contract, temp_dir):
        """
        בדיקה: נתונים מול חוזה
        Test: Data validation against contract
        """
        from src.flow.validators import DataValidator

        # יצירת clean data
        clean_data_path = temp_dir / "clean_data.csv"
        df = pd.read_csv(sample_data)
        df.to_csv(clean_data_path, index=False)

        validator = DataValidator()
        is_valid, message = validator.validate_clean_data(
            clean_data_path, sample_contract
        )

        assert is_valid is True

    def test_error_handler_custom_exceptions(self):
        """
        בדיקה: Custom Exceptions
        Test: Custom Exceptions work correctly
        """
        from src.utils.error_handler import (
            PipelineError,
            DataValidationError,
            CrewExecutionError,
        )

        # בדיקת PipelineError
        error = PipelineError("Test error", step="test_step")
        assert "test_step" in str(error)

        # בדיקת DataValidationError
        validation_error = DataValidationError("Invalid data", field="rating")
        assert validation_error.field == "rating"

        # בדיקת CrewExecutionError
        crew_error = CrewExecutionError("Crew failed", crew_name="analyst")
        assert crew_error.recoverable is True


class TestValidators:
    """בדיקות עבור ה-Validators"""

    @pytest.fixture
    def temp_dir(self):
        """יצירת תיקייה זמנית"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_validate_empty_file(self, temp_dir):
        """
        בדיקה: קובץ ריק
        Test: Empty file validation
        """
        from src.flow.validators import DataValidator

        empty_file = temp_dir / "empty.csv"
        empty_file.touch()

        validator = DataValidator()
        is_valid, message = validator.validate_raw_data(empty_file)

        assert is_valid is False

    def test_validate_model_outputs_missing(self, temp_dir):
        """
        בדיקה: תוצרי מודל חסרים
        Test: Missing model outputs
        """
        from src.flow.validators import DataValidator

        validator = DataValidator()
        is_valid, message = validator.validate_model_outputs(
            model_path=temp_dir / "model.pkl",
            eval_path=temp_dir / "eval.md",
            card_path=temp_dir / "card.md",
        )

        assert is_valid is False
        assert "not found" in message.lower()


# הרצת הבדיקות
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
