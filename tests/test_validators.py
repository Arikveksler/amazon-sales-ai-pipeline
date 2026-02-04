"""
בדיקות עבור Validators
Tests for Validators module
"""

import pytest
import json
import tempfile
from pathlib import Path
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.flow.validators import DataValidator


class TestDataValidator:
    """בדיקות עבור DataValidator"""

    @pytest.fixture
    def validator(self):
        """יצירת validator לבדיקות"""
        return DataValidator()

    @pytest.fixture
    def temp_dir(self):
        """יצירת תיקייה זמנית"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_validate_raw_data_file_not_exists(self, validator, temp_dir):
        """בדיקה: קובץ לא קיים"""
        is_valid, msg = validator.validate_raw_data(temp_dir / "not_exists.csv")
        assert is_valid is False
        assert "not found" in msg.lower()

    def test_validate_raw_data_wrong_extension(self, validator, temp_dir):
        """בדיקה: סיומת קובץ שגויה"""
        txt_file = temp_dir / "data.txt"
        txt_file.write_text("some data")

        is_valid, msg = validator.validate_raw_data(txt_file)
        assert is_valid is False
        assert "csv" in msg.lower()

    def test_validate_raw_data_empty(self, validator, temp_dir):
        """בדיקה: קובץ CSV ריק"""
        empty_csv = temp_dir / "empty.csv"
        empty_csv.touch()

        is_valid, msg = validator.validate_raw_data(empty_csv)
        assert is_valid is False

    def test_validate_raw_data_valid(self, validator, temp_dir):
        """בדיקה: קובץ CSV תקין"""
        valid_csv = temp_dir / "valid.csv"
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
        df.to_csv(valid_csv, index=False)

        is_valid, msg = validator.validate_raw_data(valid_csv)
        assert is_valid is True
        assert "3 rows" in msg

    def test_validate_contract_not_exists(self, validator, temp_dir):
        """בדיקה: קובץ חוזה לא קיים"""
        is_valid, msg = validator.validate_dataset_contract(
            temp_dir / "not_exists.json"
        )
        assert is_valid is False

    def test_validate_contract_invalid_json(self, validator, temp_dir):
        """בדיקה: JSON לא תקין"""
        invalid_json = temp_dir / "invalid.json"
        invalid_json.write_text("{ not valid json }")

        is_valid, msg = validator.validate_dataset_contract(invalid_json)
        assert is_valid is False
        assert "json" in msg.lower()

    def test_validate_contract_missing_fields(self, validator, temp_dir):
        """בדיקה: חוזה חסר שדות"""
        incomplete = temp_dir / "incomplete.json"
        with open(incomplete, "w") as f:
            json.dump({"only_one_field": True}, f)

        is_valid, msg = validator.validate_dataset_contract(incomplete)
        assert is_valid is False
        assert "missing" in msg.lower()

    def test_validate_contract_valid(self, validator, temp_dir):
        """בדיקה: חוזה תקין"""
        valid_contract = temp_dir / "valid.json"
        contract = {
            "schema": {"field1": "string", "field2": "int"},
            "required_columns": ["field1"],
            "constraints": {},
        }
        with open(valid_contract, "w") as f:
            json.dump(contract, f)

        is_valid, msg = validator.validate_dataset_contract(valid_contract)
        assert is_valid is True

    def test_validate_clean_data_against_contract(self, validator, temp_dir):
        """בדיקה: נתונים מול חוזה"""
        # יצירת חוזה
        contract_path = temp_dir / "contract.json"
        contract = {
            "schema": {"id": "int", "name": "string", "value": "float"},
            "required_columns": ["id", "name"],
            "constraints": {"value": {"min": 0, "max": 100}},
        }
        with open(contract_path, "w") as f:
            json.dump(contract, f)

        # יצירת נתונים תקינים
        data_path = temp_dir / "data.csv"
        df = pd.DataFrame(
            {"id": [1, 2, 3], "name": ["a", "b", "c"], "value": [10.0, 20.0, 30.0]}
        )
        df.to_csv(data_path, index=False)

        is_valid, msg = validator.validate_clean_data(data_path, contract_path)
        assert is_valid is True

    def test_validate_clean_data_missing_required_column(self, validator, temp_dir):
        """בדיקה: עמודה נדרשת חסרה"""
        # יצירת חוזה
        contract_path = temp_dir / "contract.json"
        contract = {
            "schema": {"id": "int", "name": "string"},
            "required_columns": ["id", "name", "missing_col"],
            "constraints": {},
        }
        with open(contract_path, "w") as f:
            json.dump(contract, f)

        # יצירת נתונים ללא העמודה הנדרשת
        data_path = temp_dir / "data.csv"
        df = pd.DataFrame({"id": [1, 2], "name": ["a", "b"]})
        df.to_csv(data_path, index=False)

        is_valid, msg = validator.validate_clean_data(data_path, contract_path)
        assert is_valid is False
        assert "missing" in msg.lower()

    def test_validate_features(self, validator, temp_dir):
        """בדיקה: קובץ features"""
        features_path = temp_dir / "features.csv"
        df = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "target": [0, 1, 0]}
        )
        df.to_csv(features_path, index=False)

        is_valid, msg = validator.validate_features(features_path)
        assert is_valid is True
        assert "3 features" in msg

    def test_validate_model_card_missing_sections(self, validator, temp_dir):
        """בדיקה: Model Card חסר סקשנים"""
        card_path = temp_dir / "model_card.md"
        card_path.write_text("# Model\nSome content without required sections")

        is_valid, msg = validator._validate_model_card(card_path)
        assert is_valid is False
        assert "missing" in msg.lower()

    def test_validate_model_card_valid(self, validator, temp_dir):
        """בדיקה: Model Card תקין"""
        card_path = temp_dir / "model_card.md"
        content = """
# Model Card

## Model Purpose
This model predicts sales.

## Training Data
Amazon sales dataset.

## Metrics
- Accuracy: 85%
- F1: 0.82

## Limitations
- Only works for specific categories
"""
        card_path.write_text(content)

        is_valid, msg = validator._validate_model_card(card_path)
        assert is_valid is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
