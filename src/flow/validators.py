"""
Pipeline Validators - src/flow/validators.py

Data validation module with DataValidator class for validating
data against dataset contracts in the Pipeline.
"""

import json
from pathlib import Path
from typing import Tuple, Dict, Any, List

import pandas as pd
from loguru import logger


class DataValidationError(Exception):
    """Custom exception for data validation failures."""
    pass


class DataValidator:
    """
    Centralized data validator for the Amazon Sales Pipeline.

    Validates data against dataset contracts, checking schema,
    constraints, and required columns.

    Attributes:
        contract: The loaded dataset contract dictionary

    Examples:
        >>> validator = DataValidator()
        >>> is_valid, msg = validator.validate_raw_data("data/raw/amazon.csv")
        >>> if is_valid:
        ...     print("Data is valid!")
    """

    def __init__(self, contract_path: str = None):
        """
        Initialize the DataValidator.

        Args:
            contract_path: Optional path to the dataset contract JSON.
                          If provided, loads the contract immediately.
        """
        self.contract = None
        if contract_path:
            self.load_contract(contract_path)

    def load_contract(self, contract_path: str) -> dict:
        """
        Load and parse the dataset contract JSON.

        Args:
            contract_path: Path to the dataset_contract.json file

        Returns:
            The parsed contract dictionary

        Raises:
            FileNotFoundError: If contract file doesn't exist
            json.JSONDecodeError: If JSON is invalid
        """
        path = Path(contract_path)

        if not path.exists():
            raise FileNotFoundError(f"Contract file not found: {contract_path}")

        with open(path, 'r', encoding='utf-8') as f:
            self.contract = json.load(f)

        logger.info(f"Contract loaded: {self.contract.get('dataset_name', 'unknown')}")
        return self.contract

    def validate_raw_data(self, file_path) -> Tuple[bool, str]:
        """
        Validate raw CSV file existence and structure.

        Checks:
        1. File exists
        2. File has .csv extension
        3. File is not empty
        4. File can be loaded with pandas

        Args:
            file_path: Path to the raw CSV file

        Returns:
            Tuple[bool, str]: (is_valid, message)
        """
        logger.info(f"Validating raw data: {file_path}")

        path = Path(file_path)

        # Check 1: File exists
        if not path.exists():
            error_msg = f"File not found: {file_path}"
            logger.error(error_msg)
            return False, error_msg

        # Check 2: CSV extension
        if path.suffix.lower() != '.csv':
            error_msg = f"File must be CSV format, got: {path.suffix}"
            logger.error(error_msg)
            return False, error_msg

        # Check 3: File not empty
        if path.stat().st_size == 0:
            error_msg = f"File is empty: {file_path}"
            logger.error(error_msg)
            return False, error_msg

        # Check 4: Can be loaded with pandas
        try:
            df = pd.read_csv(path)
        except Exception as e:
            error_msg = f"Error loading CSV: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

        # Check DataFrame is not empty
        if df.empty:
            error_msg = f"CSV file has no data rows"
            logger.error(error_msg)
            return False, error_msg

        # Success
        success_msg = f"Raw data valid: {len(df)} rows, {len(df.columns)} columns"
        logger.success(success_msg)
        return True, success_msg

    def validate_dataset_contract(self, contract_path) -> Tuple[bool, str]:
        """
        Validate the dataset contract JSON structure.

        Checks:
        1. File exists
        2. Valid JSON format
        3. Required fields present: schema, required_columns, constraints
        4. schema is a dict
        5. required_columns is a list

        Args:
            contract_path: Path to the dataset_contract.json file

        Returns:
            Tuple[bool, str]: (is_valid, message)
        """
        logger.info(f"Validating dataset contract: {contract_path}")

        path = Path(contract_path)

        # Check 1: File exists
        if not path.exists():
            error_msg = f"Contract file not found: {contract_path}"
            logger.error(error_msg)
            return False, error_msg

        # Check 2: Valid JSON
        try:
            with open(path, 'r', encoding='utf-8') as f:
                contract = json.load(f)
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON format: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Error reading contract: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

        # Check 3: Required fields present
        required_fields = ["schema", "required_columns", "constraints"]
        missing_fields = [field for field in required_fields if field not in contract]

        if missing_fields:
            error_msg = f"Missing required fields: {', '.join(missing_fields)}"
            logger.error(error_msg)
            return False, error_msg

        # Check 4: schema is dict
        if not isinstance(contract["schema"], dict):
            error_msg = f"schema must be a dict, got {type(contract['schema']).__name__}"
            logger.error(error_msg)
            return False, error_msg

        # Check 5: required_columns is list
        if not isinstance(contract["required_columns"], list):
            error_msg = f"required_columns must be a list, got {type(contract['required_columns']).__name__}"
            logger.error(error_msg)
            return False, error_msg

        # Success
        success_msg = f"Contract valid: {len(contract.get('required_columns', []))} required columns"
        logger.success(success_msg)
        return True, success_msg

    def validate_clean_data(self, file_path, contract_path) -> Tuple[bool, str]:
        """
        Validate cleaned data against the dataset contract.

        Checks:
        1. Data file exists and loads
        2. Contract file is valid
        3. All required columns present
        4. No null values in required columns
        5. Constraints are satisfied (if applicable)

        Args:
            file_path: Path to the clean_data.csv file
            contract_path: Path to the dataset_contract.json file

        Returns:
            Tuple[bool, str]: (is_valid, message)
        """
        logger.info(f"Validating clean data: {file_path}")

        path = Path(file_path)

        # Check 1: Data file exists
        if not path.exists():
            error_msg = f"Clean data file not found: {file_path}"
            logger.error(error_msg)
            return False, error_msg

        # Load data
        try:
            df = pd.read_csv(path)
        except Exception as e:
            error_msg = f"Error loading clean data: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

        # Check 2: Load contract
        contract_path_obj = Path(contract_path)
        if not contract_path_obj.exists():
            error_msg = f"Contract file not found: {contract_path}"
            logger.error(error_msg)
            return False, error_msg

        try:
            with open(contract_path_obj, 'r', encoding='utf-8') as f:
                contract = json.load(f)
        except Exception as e:
            error_msg = f"Error loading contract: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

        # Check 3: Required columns present
        required_columns = contract.get("required_columns", [])
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            error_msg = f"Missing required columns: {', '.join(missing_columns)}"
            logger.error(error_msg)
            return False, error_msg

        # Check 4: No null values in required columns
        for col in required_columns:
            if df[col].isnull().any():
                null_count = df[col].isnull().sum()
                error_msg = f"Null values in required column '{col}': {null_count} nulls"
                logger.error(error_msg)
                return False, error_msg

        # Check 5: Validate constraints
        constraints = contract.get("constraints", {})
        if constraints:
            is_valid, errors = self.validate_constraints(df, constraints)
            if not is_valid:
                error_msg = f"Constraint violations: {'; '.join(errors)}"
                logger.error(error_msg)
                return False, error_msg

        # Success
        success_msg = f"Clean data valid: {len(df)} rows, schema matches contract"
        logger.success(success_msg)
        return True, success_msg

    def validate_constraints(self, df: pd.DataFrame, constraints: dict) -> Tuple[bool, List[str]]:
        """
        Validate DataFrame against numeric constraints.

        Checks min/max values and nullable rules for each column
        defined in the constraints dict.

        Args:
            df: The DataFrame to validate
            constraints: Dict mapping column names to constraint rules
                        e.g. {"rating": {"min": 0, "max": 5, "nullable": false}}

        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_errors)
        """
        errors = []

        for col, rules in constraints.items():
            if col not in df.columns:
                continue  # Skip columns not in DataFrame

            # Convert column to numeric for comparison (handle string types)
            try:
                numeric_col = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                numeric_col = df[col]

            # Check min constraint
            min_val = rules.get('min')
            if min_val is not None:
                violations = numeric_col[numeric_col < min_val]
                if len(violations) > 0:
                    errors.append(f"{col}: {len(violations)} values below min ({min_val})")

            # Check max constraint
            max_val = rules.get('max')
            if max_val is not None:
                violations = numeric_col[numeric_col > max_val]
                if len(violations) > 0:
                    errors.append(f"{col}: {len(violations)} values above max ({max_val})")

            # Check nullable constraint
            if not rules.get('nullable', True):
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    errors.append(f"{col}: {null_count} null values (not nullable)")

        return len(errors) == 0, errors

    def validate_features(self, features_path) -> Tuple[bool, str]:
        """
        Validate the features CSV file.

        Checks:
        1. File exists
        2. File loads correctly
        3. Has at least 3 features (columns)
        4. Has at least 10 rows

        Args:
            features_path: Path to the features.csv file

        Returns:
            Tuple[bool, str]: (is_valid, message)
        """
        logger.info(f"Validating features: {features_path}")

        path = Path(features_path)

        # Check 1: File exists
        if not path.exists():
            error_msg = f"Features file not found: {features_path}"
            logger.error(error_msg)
            return False, error_msg

        # Check 2: Load file
        try:
            df = pd.read_csv(path)
        except Exception as e:
            error_msg = f"Error loading features: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

        # Check 3: At least 3 features
        if len(df.columns) < 3:
            error_msg = f"Insufficient features: {len(df.columns)} (minimum 3 required)"
            logger.error(error_msg)
            return False, error_msg

        # Check 4: At least 10 rows
        if len(df) < 10:
            error_msg = f"Insufficient rows: {len(df)} (minimum 10 required)"
            logger.error(error_msg)
            return False, error_msg

        # Success
        success_msg = f"Features valid: {len(df)} rows, {len(df.columns)} features"
        logger.success(success_msg)
        return True, success_msg

    def validate_model_outputs(
        self,
        model_path: str,
        eval_path: str,
        card_path: str
    ) -> Tuple[bool, str]:
        """
        Validate all model output artifacts.

        Checks:
        1. model.pkl exists
        2. evaluation_report.md exists and is not empty
        3. model_card.md exists and has required sections

        Args:
            model_path: Path to model.pkl
            eval_path: Path to evaluation_report.md
            card_path: Path to model_card.md

        Returns:
            Tuple[bool, str]: (is_valid, message)
        """
        logger.info("Validating model outputs")

        # Check 1: model.pkl exists
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            error_msg = f"Model file not found: {model_path}"
            logger.error(error_msg)
            return False, error_msg
        logger.success("model.pkl exists")

        # Check 2: evaluation_report.md exists and not empty
        eval_path_obj = Path(eval_path)
        if not eval_path_obj.exists():
            error_msg = f"Evaluation report not found: {eval_path}"
            logger.error(error_msg)
            return False, error_msg

        try:
            with open(eval_path_obj, 'r', encoding='utf-8') as f:
                eval_content = f.read()

            if len(eval_content.strip()) < 50:
                error_msg = "Evaluation report too short (less than 50 characters)"
                logger.error(error_msg)
                return False, error_msg

            logger.success("evaluation_report.md valid")
        except Exception as e:
            error_msg = f"Error reading evaluation report: {str(e)}"
            logger.error(error_msg)
            return False, error_msg

        # Check 3: model_card.md validation
        is_valid, msg = self._validate_model_card(card_path)
        if not is_valid:
            return False, msg

        # Success
        success_msg = "All model outputs valid (model + evaluation + card)"
        logger.success(success_msg)
        return True, success_msg

    def _validate_model_card(self, card_path) -> Tuple[bool, str]:
        """
        Validate the model card markdown file.

        Required sections:
        - Model Purpose
        - Training Data
        - Metrics
        - Limitations

        Args:
            card_path: Path to model_card.md

        Returns:
            Tuple[bool, str]: (is_valid, message)
        """
        logger.info(f"Validating model card: {card_path}")

        card_path_obj = Path(card_path)

        # Check file exists
        if not card_path_obj.exists():
            error_msg = f"Model card not found: {card_path}"
            logger.error(error_msg)
            return False, error_msg

        try:
            with open(card_path_obj, 'r', encoding='utf-8') as f:
                card_content = f.read().lower()  # Lowercase for case-insensitive matching

            required_sections = [
                "model purpose",
                "training data",
                "metrics",
                "limitations"
            ]

            missing_sections = [
                section for section in required_sections
                if section not in card_content
            ]

            if missing_sections:
                error_msg = f"Missing sections in model card: {', '.join(missing_sections)}"
                logger.error(error_msg)
                return False, error_msg

            logger.success("model_card.md valid with all required sections")
            return True, "Model card valid"

        except Exception as e:
            error_msg = f"Error reading model card: {str(e)}"
            logger.error(error_msg)
            return False, error_msg


# Backward compatibility: Keep standalone functions for existing code
def validate_raw_data(file_path: str) -> Tuple[bool, str]:
    """Legacy function for backward compatibility."""
    validator = DataValidator()
    return validator.validate_raw_data(file_path)


def validate_clean_data(file_path: str, contract_path: str) -> Tuple[bool, str]:
    """Legacy function for backward compatibility."""
    validator = DataValidator()
    return validator.validate_clean_data(file_path, contract_path)


def validate_dataset_contract(contract_path: str) -> Tuple[bool, str]:
    """Legacy function for backward compatibility."""
    validator = DataValidator()
    return validator.validate_dataset_contract(contract_path)


def validate_features(features_path: str) -> Tuple[bool, str]:
    """Legacy function for backward compatibility."""
    validator = DataValidator()
    return validator.validate_features(features_path)


def validate_model_outputs(
    model_path: str,
    eval_path: str,
    card_path: str
) -> Tuple[bool, str]:
    """Legacy function for backward compatibility."""
    validator = DataValidator()
    return validator.validate_model_outputs(model_path, eval_path, card_path)
