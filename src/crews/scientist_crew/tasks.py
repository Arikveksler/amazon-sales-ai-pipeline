"""
הגדרת משימות של Scientist Crew
Scientist Crew tasks definition

Defines 4 sequential tasks for ML pipeline:
1. Feature Engineering - Create ML-ready features
2. Model Training - Train multiple models with hyperparameter tuning
3. Model Evaluation - Evaluate and compare models
4. Model Card Creation - Document model for responsible AI

Author: ML Specialist
Date: 2026-02-06
"""

from typing import List
from crewai import Task


def create_scientist_tasks(
    agents: List,
    clean_data_path: str,
    contract_path: str,
    features_dir: str,
    models_dir: str,
    reports_dir: str
) -> List[Task]:
    """
    יצירת משימות של ה-Scientist Crew.
    Create Scientist Crew tasks.

    Args:
        agents: List of 4 agents [feature_engineer, model_trainer, model_evaluator, documentation_expert]
        clean_data_path: Path to clean_data.csv
        contract_path: Path to dataset_contract.json
        features_dir: Directory for features output
        models_dir: Directory for model output
        reports_dir: Directory for reports output

    Returns:
        List of 4 Task objects with sequential dependencies
    """

    # Unpack agents
    feature_engineer, model_trainer, model_evaluator, documentation_expert = agents

    # ========================================================================
    # Task 1: Feature Engineering
    # ========================================================================

    feature_engineering_task = Task(
        description=f"""
        Load clean data from {clean_data_path} and validate against {contract_path}.

        Perform the following feature engineering steps:

        1. **Type Conversions**: Convert string columns to numeric types
           - actual_price, discounted_price: remove ₹ symbol and commas → float
           - rating: convert to float (0-5 range)
           - rating_count: remove commas → int
           - discount_percentage: remove % symbol → float (TARGET VARIABLE)

        2. **Derived Features**: Create the following features:
           - price_level = log1p(actual_price)
           - discounted_price_level = log1p(discounted_price)
           - log_rating_count = log1p(rating_count)
           - rating_weighted = rating × log1p(rating_count)
           - is_highly_rated = 1 if rating >= 4.0 else 0
           - reviews_per_rating = rating_count / (rating + 0.1)
           - has_many_reviews = 1 if rating_count > median else 0

        3. **Text Features**: Extract from about_product and review_content:
           - description_length, description_word_count
           - has_premium_keywords, has_tech_keywords
           - review_length, review_word_count, review_sentiment_score

        4. **Category Encoding**: One-hot encoding for top 10 categories

        5. **Product-Level Aggregation**: Group by product_id and aggregate review features

        6. **Validation**: Ensure no missing values, all numeric types, valid ranges

        Save engineered features to {features_dir}/features.csv
        Save feature metadata to {features_dir}/feature_metadata.json

        IMPORTANT: discount_percentage is the TARGET VARIABLE - do not use it as a feature!

        Expected features: 25-30 columns (features + target)
        Expected rows: ~1463 products (after aggregation)
        """,
        expected_output=f"""
        CSV file with engineered features (features.csv) containing:
        - All numeric features (25-30 feature columns)
        - No missing values
        - Proper data types (float64, int64)
        - Target variable column (discount_percentage)
        - Product-level aggregation (grouped from review-level)

        JSON metadata file (feature_metadata.json) with:
        - Feature names and descriptions
        - Data types
        - Basic statistics (mean, std, min, max)
        - Feature count and target variable name
        """,
        agent=feature_engineer,
        output_file=f"{features_dir}/features.csv"
    )

    # ========================================================================
    # Task 2: Model Training
    # ========================================================================

    model_training_task = Task(
        description=f"""
        Load engineered features from {features_dir}/features.csv

        Train the following models for discount percentage prediction:

        1. **Random Forest Regressor** with GridSearchCV hyperparameter tuning:
           - Tune: n_estimators=[100, 200], max_depth=[10, 20, None], min_samples_split=[2, 5]
           - Use 5-fold cross-validation
           - Scoring metric: R² score

        2. **XGBoost Regressor** with GridSearchCV hyperparameter tuning:
           - Tune: n_estimators=[100, 200], max_depth=[4, 6], learning_rate=[0.05, 0.1]
           - Use 5-fold cross-validation
           - Scoring metric: R² score

        3. **Linear Regression** (baseline, no tuning)

        For each model:
        - Use 80/20 train/test split (stratified by price buckets, random_state=42)
        - Track training time and metrics (RMSE, MAE, R², MAPE)
        - Record best hyperparameters from GridSearchCV

        Select the best model based on test set R² score.

        Save the best model with metadata to {models_dir}/model.pkl

        Model metadata should include:
        - Model type and hyperparameters
        - Training/test metrics (RMSE, MAE, R², MAPE)
        - Feature names used
        - Cross-validation score
        - Training timestamp
        - Feature importance (if available)
        """,
        expected_output=f"""
        Pickle file (model.pkl) containing:
        - Trained model object (best performer based on test R²)
        - Comprehensive metadata dictionary with:
          * Model type (e.g., "XGBoost Regressor")
          * Hyperparameters
          * Train/test metrics
          * Feature names list
          * Cross-validation score
          * Training time and timestamp

        The model must be loadable with joblib.load() and ready for predictions.
        Metadata must be accessible via model_data['metadata'] after loading.
        """,
        agent=model_trainer,
        context=[feature_engineering_task],
        output_file=f"{models_dir}/model.pkl"
    )

    # ========================================================================
    # Task 3: Model Evaluation
    # ========================================================================

    model_evaluation_task = Task(
        description=f"""
        Load the trained models and test data from {models_dir}/model.pkl and {features_dir}/features.csv

        Evaluate all trained models (Random Forest, XGBoost, Linear Regression):

        1. **Calculate Metrics** for each model:
           - R² Score (train and test)
           - RMSE (Root Mean Squared Error)
           - MAE (Mean Absolute Error)
           - MAPE (Mean Absolute Percentage Error)
           - Training time

        2. **Model Comparison**: Create a comparison table with all models and their metrics

        3. **Feature Importance Analysis**: Extract and rank top 15 most important features

        4. **Best Model Selection**: Identify the best model (highest test R²)

        5. **Business Insights**:
           - Optimal discount ranges by category
           - Model strengths and weaknesses
           - Overfitting analysis (train vs test performance)

        6. **Recommendations**:
           - Deployment strategy
           - Monitoring plan
           - Suggestions for improvement

        Create a comprehensive evaluation report in Markdown format.
        Save to {reports_dir}/evaluation_report.md

        Report must include these sections:
        - Overview
        - Models Compared (table)
        - Best Model Performance
        - Feature Importance Analysis (table with top 15)
        - Model Strengths
        - Model Weaknesses & Limitations
        - Business Recommendations
        - Recommendations for Improvement
        - Conclusion
        """,
        expected_output=f"""
        Markdown report (evaluation_report.md) with sections:

        # Model Evaluation Report - Optimal Discount Prediction

        ## 1. Overview
        ## 2. Models Compared (with comparison table)
        ## 3. Best Model Performance (hyperparameters + metrics table)
        ## 4. Feature Importance Analysis (top 15 features table)
        ## 5. Model Strengths
        ## 6. Model Weaknesses & Limitations
        ## 7. Business Recommendations
        ## 8. Recommendations for Improvement
        ## 9. Conclusion

        Must be well-formatted, data-driven, and actionable.
        Include specific numbers, not generic statements.
        """,
        agent=model_evaluator,
        context=[model_training_task],
        output_file=f"{reports_dir}/evaluation_report.md"
    )

    # ========================================================================
    # Task 4: Model Card Creation
    # ========================================================================

    model_card_task = Task(
        description=f"""
        Create a comprehensive Model Card for the selected best model.
        Load model metadata from {models_dir}/model.pkl
        Reference evaluation report from {reports_dir}/evaluation_report.md

        Create Model Card with these REQUIRED sections (validation will check for these):

        1. **Purpose** ✓ REQUIRED:
           - What does this model do? (optimal discount prediction)
           - Intended use cases (dynamic pricing, promotional planning, BI)
           - Out-of-scope uses (what NOT to use it for)

        2. **Data** ✓ REQUIRED:
           - Training data description (size, features, date range)
           - Feature categories (pricing, ratings, text, categories)
           - Target variable (discount_percentage)
           - Preprocessing steps
           - Train/test split details

        3. **Metrics** ✓ REQUIRED:
           - Test set performance (R², RMSE, MAE, MAPE)
           - Training set performance (optional)
           - Cross-validation score
           - Interpretation of metrics (what do the numbers mean?)

        4. **Limitations** ✓ REQUIRED:
           - Known limitations (temporal validity, category coverage, sample size)
           - Edge cases (brand-new products, extreme prices, niche categories)
           - Prediction accuracy constraints

        5. **Ethical Considerations** ✓ REQUIRED:
           - Fairness (potential biases: category, rating, price discrimination)
           - Transparency (model interpretability, feature importance)
           - Responsible use guidelines (dos and don'ts)
           - Risk of misuse (price manipulation, anti-competitive behavior)
           - Privacy (PII handling, compliance)

        Additional sections:
        - Model Details (architecture, hyperparameters)
        - Recommendations for Use (deployment, validation, guardrails)
        - Contact & Support

        Save to {reports_dir}/model_card.md
        """,
        expected_output=f"""
        Markdown document (model_card.md) with:

        # Model Card: Amazon Optimal Discount Prediction Model

        ## Purpose ✓ (REQUIRED - must contain this word)
        [Clear description of model purpose and use cases]

        ## Data ✓ (REQUIRED - must contain this word)
        [Training data details, features, preprocessing]

        ## Metrics ✓ (REQUIRED - must contain this word)
        [Performance metrics with specific values]

        ## Limitations ✓ (REQUIRED - must contain this word)
        [Known limitations, edge cases, constraints]

        ## Ethical Considerations ✓ (REQUIRED - must contain these words)
        [Fairness, bias, responsible use guidelines, risks, privacy]

        ## Model Details
        [Architecture, hyperparameters, training procedure]

        ## Recommendations for Use
        [Deployment guidelines, guardrails, monitoring]

        ## Contact & Support
        [Owner, update schedule, support contact]

        Must be comprehensive, transparent, and follow responsible AI practices.
        CRITICAL: Must include all 5 required sections with exact keywords (case-insensitive).
        """,
        agent=documentation_expert,
        context=[model_evaluation_task],
        output_file=f"{reports_dir}/model_card.md"
    )

    return [
        feature_engineering_task,
        model_training_task,
        model_evaluation_task,
        model_card_task
    ]
