"""
Model Training Module for Scientist Crew
=========================================
מודול אימון מודלים לחיזוי שיעור הנחה אופטימלי.

This module provides functions for:
- Train/test split
- Model training (Random Forest, XGBoost, Linear Regression)
- Hyperparameter tuning with GridSearchCV
- Model evaluation
- Model selection and saving

Author: ML Specialist
Date: 2026-02-06
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any
from loguru import logger
import joblib
from datetime import datetime
import time

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    logger.warning("XGBoost not available, will skip XGBoost training")
    XGBOOST_AVAILABLE = False


# ============================================================================
# Train/Test Split
# ============================================================================

def prepare_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    חלוקת נתונים לtrain/test.
    Prepare train/test split.

    Uses stratified split by price category if available.

    Args:
        X: Features DataFrame
        y: Target Series (discount_percentage)
        test_size: Test set size (default: 0.2)
        random_state: Random seed (default: 42)

    Returns:
        X_train, X_test, y_train, y_test
    """
    logger.info(f"Splitting data: {1-test_size:.0%} train, {test_size:.0%} test")

    # Check if we can stratify by actual_price
    stratify = None
    if 'actual_price' in X.columns:
        # Create price buckets for stratification
        price_bins = [0, 500, 1000, 2000, 5000, np.inf]
        price_labels = ['<500', '500-1K', '1K-2K', '2K-5K', '>5K']
        stratify_col = pd.cut(X['actual_price'], bins=price_bins, labels=price_labels)

        logger.info(f"  Using stratification by price category")
        logger.info(f"  Price distribution: {stratify_col.value_counts().to_dict()}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_col
        )
    else:
        logger.info("  No stratification (actual_price not found)")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )

    logger.info(f"  ✓ Train set: {len(X_train)} samples")
    logger.info(f"  ✓ Test set: {len(X_test)} samples")

    return X_train, X_test, y_train, y_test


# ============================================================================
# Model Training Functions
# ============================================================================

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tune_hyperparameters: bool = True,
    cv: int = 5
) -> Tuple[Any, Dict]:
    """
    אימון Random Forest Regressor.
    Train Random Forest Regressor.

    Args:
        X_train: Training features
        y_train: Training target
        tune_hyperparameters: Whether to use GridSearchCV (default: True)
        cv: Cross-validation folds (default: 5)

    Returns:
        (model, training_info)
    """
    logger.info("Training Random Forest Regressor...")
    start_time = time.time()

    if tune_hyperparameters:
        logger.info("  Using GridSearchCV for hyperparameter tuning")

        # Hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }

        # GridSearchCV
        rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)

        grid_search = GridSearchCV(
            estimator=rf_base,
            param_grid=param_grid,
            cv=cv,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        cv_score = grid_search.best_score_

        logger.info(f"  ✓ Best parameters: {best_params}")
        logger.info(f"  ✓ CV R² score: {cv_score:.4f}")

    else:
        logger.info("  Using default hyperparameters")

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        best_params = model.get_params()
        cv_score = None

    training_time = time.time() - start_time

    # Training info
    training_info = {
        'model_type': 'Random Forest Regressor',
        'hyperparameters': best_params,
        'cv_score': cv_score,
        'training_time_seconds': training_time,
        'n_features': X_train.shape[1],
        'n_samples': len(X_train)
    }

    logger.success(f"✓ Random Forest trained in {training_time:.2f} seconds")

    return model, training_info


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tune_hyperparameters: bool = True,
    cv: int = 5
) -> Tuple[Any, Dict]:
    """
    אימון XGBoost Regressor.
    Train XGBoost Regressor.

    Args:
        X_train: Training features
        y_train: Training target
        tune_hyperparameters: Whether to use GridSearchCV (default: True)
        cv: Cross-validation folds (default: 5)

    Returns:
        (model, training_info)
    """
    if not XGBOOST_AVAILABLE:
        logger.error("XGBoost not available, skipping")
        return None, None

    logger.info("Training XGBoost Regressor...")
    start_time = time.time()

    if tune_hyperparameters:
        logger.info("  Using GridSearchCV for hyperparameter tuning")

        # Hyperparameter grid (smaller for speed)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [4, 6],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }

        # GridSearchCV
        xgb_base = XGBRegressor(
            random_state=42,
            n_jobs=-1,
            objective='reg:squarederror'
        )

        grid_search = GridSearchCV(
            estimator=xgb_base,
            param_grid=param_grid,
            cv=cv,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        cv_score = grid_search.best_score_

        logger.info(f"  ✓ Best parameters: {best_params}")
        logger.info(f"  ✓ CV R² score: {cv_score:.4f}")

    else:
        logger.info("  Using default hyperparameters")

        model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
            objective='reg:squarederror'
        )

        model.fit(X_train, y_train)

        best_params = model.get_params()
        cv_score = None

    training_time = time.time() - start_time

    # Training info
    training_info = {
        'model_type': 'XGBoost Regressor',
        'hyperparameters': best_params,
        'cv_score': cv_score,
        'training_time_seconds': training_time,
        'n_features': X_train.shape[1],
        'n_samples': len(X_train)
    }

    logger.success(f"✓ XGBoost trained in {training_time:.2f} seconds")

    return model, training_info


def train_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Tuple[Any, Dict]:
    """
    אימון Linear Regression (baseline).
    Train Linear Regression (baseline).

    Args:
        X_train: Training features
        y_train: Training target

    Returns:
        (model, training_info)
    """
    logger.info("Training Linear Regression (baseline)...")
    start_time = time.time()

    model = LinearRegression(n_jobs=-1)
    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    # Training info
    training_info = {
        'model_type': 'Linear Regression',
        'hyperparameters': {},
        'cv_score': None,
        'training_time_seconds': training_time,
        'n_features': X_train.shape[1],
        'n_samples': len(X_train)
    }

    logger.success(f"✓ Linear Regression trained in {training_time:.2f} seconds")

    return model, training_info


# ============================================================================
# Model Evaluation
# ============================================================================

def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train: pd.DataFrame = None,
    y_train: pd.Series = None
) -> Dict:
    """
    הערכת מודל על test set (ואופציונלי train set).
    Evaluate model on test set (and optionally train set).

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        X_train: Training features (optional, for train metrics)
        y_train: Training target (optional, for train metrics)

    Returns:
        Dictionary with metrics
    """
    logger.info("Evaluating model...")

    # Test set predictions
    y_pred = model.predict(X_test)

    # Test metrics
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
    mae_test = mean_absolute_error(y_test, y_pred)
    r2_test = r2_score(y_test, y_pred)

    # Handle MAPE (avoid division by zero)
    try:
        mape_test = mean_absolute_percentage_error(y_test, y_pred) * 100
    except:
        mape_test = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100

    logger.info(f"  Test set metrics:")
    logger.info(f"    R²:   {r2_test:.4f}")
    logger.info(f"    RMSE: {rmse_test:.4f}%")
    logger.info(f"    MAE:  {mae_test:.4f}%")
    logger.info(f"    MAPE: {mape_test:.4f}%")

    metrics = {
        'test_r2': r2_test,
        'test_rmse': rmse_test,
        'test_mae': mae_test,
        'test_mape': mape_test
    }

    # Train set metrics (if provided)
    if X_train is not None and y_train is not None:
        y_train_pred = model.predict(X_train)

        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        mae_train = mean_absolute_error(y_train, y_train_pred)
        r2_train = r2_score(y_train, y_train_pred)

        try:
            mape_train = mean_absolute_percentage_error(y_train, y_train_pred) * 100
        except:
            mape_train = np.mean(np.abs((y_train - y_train_pred) / (y_train + 1e-10))) * 100

        logger.info(f"  Train set metrics:")
        logger.info(f"    R²:   {r2_train:.4f}")
        logger.info(f"    RMSE: {rmse_train:.4f}%")
        logger.info(f"    MAE:  {mae_train:.4f}%")

        metrics['train_r2'] = r2_train
        metrics['train_rmse'] = rmse_train
        metrics['train_mae'] = mae_train
        metrics['train_mape'] = mape_train

        # Overfitting check
        overfitting_gap = r2_train - r2_test
        logger.info(f"  Overfitting gap (R²): {overfitting_gap:.4f}")
        metrics['overfitting_gap'] = overfitting_gap

    return metrics


# ============================================================================
# Model Selection
# ============================================================================

def select_best_model(models_dict: Dict[str, Dict]) -> Tuple[str, Any, Dict]:
    """
    בחירת המודל הטוב ביותר לפי R² על test set.
    Select best model based on test R² score.

    Args:
        models_dict: Dictionary with model results
            {
                'model_name': {
                    'model': model_object,
                    'metrics': {...},
                    'training_info': {...}
                }
            }

    Returns:
        (best_model_name, best_model, best_info)
    """
    logger.info("Selecting best model based on test R² score...")

    best_name = None
    best_r2 = -np.inf
    best_model = None
    best_info = None

    for name, info in models_dict.items():
        if info is None or info['model'] is None:
            continue

        r2 = info['metrics']['test_r2']

        logger.info(f"  {name}: R² = {r2:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_name = name
            best_model = info['model']
            best_info = info

    logger.success(f"✓ Best model: {best_name} (R² = {best_r2:.4f})")

    return best_name, best_model, best_info


# ============================================================================
# Model Persistence
# ============================================================================

def save_model_with_metadata(
    model: Any,
    model_name: str,
    metrics: Dict,
    training_info: Dict,
    feature_names: List[str],
    path: str
) -> bool:
    """
    שמירת מודל עם metadata.
    Save model with metadata.

    Args:
        model: Trained model object
        model_name: Model name
        metrics: Evaluation metrics
        training_info: Training information
        feature_names: List of feature names
        path: Path to save model.pkl

    Returns:
        True if successful
    """
    logger.info(f"Saving model to {path}...")

    # Prepare metadata
    metadata = {
        'model_type': model_name,
        'task': 'discount_percentage prediction',
        'model_params': training_info['hyperparameters'],
        'features': feature_names,
        'target': 'discount_percentage',
        'train_metrics': {
            'rmse': metrics.get('train_rmse', 0.0),
            'mae': metrics.get('train_mae', 0.0),
            'r2': metrics.get('train_r2', 0.0),
            'mape': metrics.get('train_mape', 0.0)
        },
        'test_metrics': {
            'rmse': metrics['test_rmse'],
            'mae': metrics['test_mae'],
            'r2': metrics['test_r2'],
            'mape': metrics['test_mape']
        },
        'cv_score_mean': training_info.get('cv_score', 0.0),
        'training_time_seconds': training_info['training_time_seconds'],
        'trained_at': datetime.now().isoformat(),
        'n_features': len(feature_names),
        'n_samples_train': training_info['n_samples']
    }

    # Add feature importance if available
    if hasattr(model, 'feature_importances_'):
        importance_dict = dict(zip(feature_names, model.feature_importances_))
        # Sort by importance
        importance_sorted = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        metadata['feature_importance'] = importance_sorted
        logger.info("  ✓ Feature importance included")

    # Save model + metadata
    model_data = {
        'model': model,
        'metadata': metadata
    }

    joblib.dump(model_data, path)

    logger.success(f"✓ Model saved: {path}")
    logger.info(f"  Model type: {model_name}")
    logger.info(f"  Test R²: {metrics['test_r2']:.4f}")
    logger.info(f"  Features: {len(feature_names)}")

    return True


# ============================================================================
# Complete Training Pipeline
# ============================================================================

def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    tune_hyperparameters: bool = True
) -> Dict[str, Dict]:
    """
    אימון כל המודלים.
    Train all models.

    Trains:
    - Random Forest Regressor
    - XGBoost Regressor
    - Linear Regression (baseline)

    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        tune_hyperparameters: Whether to tune hyperparameters

    Returns:
        Dictionary with all model results
    """
    logger.info("="*60)
    logger.info("Training All Models")
    logger.info("="*60)

    models_dict = {}

    # 1. Random Forest
    try:
        rf_model, rf_train_info = train_random_forest(
            X_train, y_train,
            tune_hyperparameters=tune_hyperparameters
        )
        rf_metrics = evaluate_model(rf_model, X_test, y_test, X_train, y_train)

        models_dict['Random Forest'] = {
            'model': rf_model,
            'training_info': rf_train_info,
            'metrics': rf_metrics
        }
    except Exception as e:
        logger.error(f"Random Forest training failed: {e}")
        models_dict['Random Forest'] = None

    # 2. XGBoost
    try:
        xgb_model, xgb_train_info = train_xgboost(
            X_train, y_train,
            tune_hyperparameters=tune_hyperparameters
        )

        if xgb_model is not None:
            xgb_metrics = evaluate_model(xgb_model, X_test, y_test, X_train, y_train)

            models_dict['XGBoost'] = {
                'model': xgb_model,
                'training_info': xgb_train_info,
                'metrics': xgb_metrics
            }
        else:
            models_dict['XGBoost'] = None

    except Exception as e:
        logger.error(f"XGBoost training failed: {e}")
        models_dict['XGBoost'] = None

    # 3. Linear Regression (baseline)
    try:
        lr_model, lr_train_info = train_baseline(X_train, y_train)
        lr_metrics = evaluate_model(lr_model, X_test, y_test, X_train, y_train)

        models_dict['Linear Regression'] = {
            'model': lr_model,
            'training_info': lr_train_info,
            'metrics': lr_metrics
        }
    except Exception as e:
        logger.error(f"Linear Regression training failed: {e}")
        models_dict['Linear Regression'] = None

    logger.info("="*60)
    logger.success("All Models Trained!")
    logger.info("="*60)

    return models_dict


# ============================================================================
# Self-Test
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Model Training Module - Self Test")
    print("="*60 + "\n")

    # Create sample data
    np.random.seed(42)
    n_samples = 200

    X = pd.DataFrame({
        'actual_price': np.random.uniform(500, 5000, n_samples),
        'rating': np.random.uniform(3.0, 5.0, n_samples),
        'log_rating_count': np.random.uniform(2, 8, n_samples)
    })

    # Target: discount based on price and rating
    y = pd.Series(
        20 + (5000 - X['actual_price']) / 100 + X['rating'] * 2 + np.random.normal(0, 3, n_samples)
    )
    y = y.clip(0, 100)

    print(f"Sample data: {len(X)} samples, {len(X.columns)} features\n")

    # Train/test split
    X_train, X_test, y_train, y_test = prepare_train_test_split(X, y)

    # Train models (no tuning for speed)
    models = train_all_models(X_train, y_train, X_test, y_test, tune_hyperparameters=False)

    # Select best
    best_name, best_model, best_info = select_best_model(models)

    print(f"\nBest model: {best_name}")
    print(f"Test R²: {best_info['metrics']['test_r2']:.4f}")

    print("\n" + "="*60)
    print("Self test complete!")
    print("="*60 + "\n")
