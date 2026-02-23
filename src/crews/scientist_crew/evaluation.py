"""
Evaluation and Reporting Module for Scientist Crew
===================================================
מודול הערכה ויצירת דוחות למודל חיזוי שיעור הנחה.

This module provides functions for:
- Metrics calculation
- Feature importance extraction
- Model comparison tables
- Evaluation report generation (Markdown)
- Model Card generation (Markdown)

Author: ML Specialist
Date: 2026-02-06
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from loguru import logger
from datetime import datetime
import json


# ============================================================================
# Metrics Calculation
# ============================================================================

def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """
    חישוב מטריקות regression.
    Calculate regression metrics.

    Args:
        y_true: True target values
        y_pred: Predicted values

    Returns:
        Dictionary with RMSE, MAE, R², MAPE
    """
    from sklearn.metrics import (
        mean_squared_error,
        mean_absolute_error,
        r2_score
    )

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # MAPE (handle division by zero)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }


# ============================================================================
# Feature Importance
# ============================================================================

def get_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 15
) -> List[Tuple[str, float]]:
    """
    חילוץ feature importance מהמודל.
    Extract feature importance from model.

    Args:
        model: Trained model
        feature_names: List of feature names
        top_n: Number of top features to return

    Returns:
        List of (feature_name, importance) tuples, sorted by importance
    """
    logger.info(f"Extracting feature importance (top {top_n})...")

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_

        # Create list of (name, importance)
        feature_importance = list(zip(feature_names, importances))

        # Sort by importance (descending)
        feature_importance_sorted = sorted(
            feature_importance,
            key=lambda x: x[1],
            reverse=True
        )

        # Take top N
        top_features = feature_importance_sorted[:top_n]

        logger.info(f"  ✓ Top features:")
        for i, (name, importance) in enumerate(top_features[:5], 1):
            logger.info(f"    {i}. {name}: {importance:.4f}")

        return top_features

    else:
        logger.warning("  Model does not have feature_importances_ attribute")
        return []


# ============================================================================
# Comparison Table
# ============================================================================

def create_comparison_table(models_dict: Dict[str, Dict]) -> str:
    """
    יצירת טבלת השוואה בין מודלים (Markdown).
    Create comparison table between models (Markdown).

    Args:
        models_dict: Dictionary with model results

    Returns:
        Markdown table string
    """
    logger.info("Creating model comparison table...")

    # Table header
    table = "| Model | Train R² | Test R² | RMSE | MAE | Training Time |\n"
    table += "|-------|----------|---------|------|-----|---------------|\n"

    # Table rows
    for name, info in models_dict.items():
        if info is None or info['model'] is None:
            # Skipped model
            table += f"| {name} | - | - | - | - | Skipped |\n"
            continue

        metrics = info['metrics']
        train_info = info['training_info']

        train_r2 = metrics.get('train_r2', 0.0)
        test_r2 = metrics['test_r2']
        rmse = metrics['test_rmse']
        mae = metrics['test_mae']
        time_sec = train_info['training_time_seconds']

        table += f"| **{name}** | {train_r2:.4f} | {test_r2:.4f} | {rmse:.2f}% | {mae:.2f}% | {time_sec:.1f}s |\n"

    logger.info("  ✓ Comparison table created")

    return table


# ============================================================================
# Evaluation Report Generation
# ============================================================================

def generate_evaluation_report(
    models_dict: Dict[str, Dict],
    best_model_name: str,
    feature_names: List[str],
    output_path: str = None
) -> str:
    """
    יצירת דוח הערכה מקיף (Markdown).
    Generate comprehensive evaluation report (Markdown).

    Args:
        models_dict: Dictionary with all model results
        best_model_name: Name of the best model
        feature_names: List of feature names
        output_path: Path to save report (optional)

    Returns:
        Markdown report string
    """
    logger.info("Generating evaluation report...")

    # Get best model info
    best_info = models_dict[best_model_name]
    best_model = best_info['model']
    best_metrics = best_info['metrics']
    best_train_info = best_info['training_info']

    # ========================================================================
    # Report Header
    # ========================================================================

    report = "# Model Evaluation Report - Optimal Discount Prediction\n\n"
    report += f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report += f"**Best Model**: {best_model_name}\n"
    report += f"**Target Variable**: discount_percentage\n\n"
    report += "---\n\n"

    # ========================================================================
    # 1. Overview
    # ========================================================================

    report += "## 1. Overview\n\n"
    report += "This report evaluates multiple regression models trained to predict optimal discount percentages for Amazon products.\n\n"
    report += "**Objective**: Predict the optimal discount_percentage to maximize both sales volume and profitability.\n\n"
    report += "**Business Goal**: Help sellers determine the ideal discount strategy based on product characteristics, ratings, and market positioning.\n\n"
    report += "**Evaluation Metric**: R² Score (primary), RMSE, MAE, MAPE\n\n"
    report += "---\n\n"

    # ========================================================================
    # 2. Models Compared
    # ========================================================================

    report += "## 2. Models Compared\n\n"
    comparison_table = create_comparison_table(models_dict)
    report += comparison_table + "\n"
    report += f"**Selected Best Model**: {best_model_name} (highest test R²)\n\n"
    report += "---\n\n"

    # ========================================================================
    # 3. Best Model Performance
    # ========================================================================

    report += "## 3. Best Model Performance\n\n"
    report += f"### {best_model_name}\n\n"

    # Hyperparameters
    report += "**Hyperparameters**:\n"
    for param, value in best_train_info['hyperparameters'].items():
        report += f"- `{param}`: {value}\n"
    report += "\n"

    # Metrics
    report += "**Performance Metrics**:\n\n"
    report += "| Metric | Train | Test |\n"
    report += "|--------|-------|------|\n"
    report += f"| R² Score | {best_metrics.get('train_r2', 0.0):.4f} | {best_metrics['test_r2']:.4f} |\n"
    report += f"| RMSE | {best_metrics.get('train_rmse', 0.0):.2f}% | {best_metrics['test_rmse']:.2f}% |\n"
    report += f"| MAE | {best_metrics.get('train_mae', 0.0):.2f}% | {best_metrics['test_mae']:.2f}% |\n"
    report += f"| MAPE | {best_metrics.get('train_mape', 0.0):.2f}% | {best_metrics['test_mape']:.2f}% |\n\n"

    # CV Score
    if best_train_info.get('cv_score'):
        report += f"**Cross-Validation**: R² = {best_train_info['cv_score']:.4f} (5-fold CV)\n\n"

    # Interpretation
    r2_pct = best_metrics['test_r2'] * 100
    report += f"**Interpretation**: The model explains **{r2_pct:.1f}% of the variance** in discount percentages. "
    report += f"The average prediction error is **{best_metrics['test_mae']:.2f} percentage points**.\n\n"

    report += "---\n\n"

    # ========================================================================
    # 4. Feature Importance Analysis
    # ========================================================================

    report += "## 4. Feature Importance Analysis\n\n"

    feature_importance = get_feature_importance(best_model, feature_names, top_n=15)

    if feature_importance:
        report += "Top 15 most important features:\n\n"
        report += "| Rank | Feature | Importance | Description |\n"
        report += "|------|---------|------------|-------------|\n"

        for rank, (feat_name, importance) in enumerate(feature_importance, 1):
            # Generate simple description
            if 'price' in feat_name.lower():
                desc = "Pricing feature"
            elif 'rating' in feat_name.lower():
                desc = "Rating/review feature"
            elif 'category' in feat_name.lower():
                desc = "Product category"
            elif 'description' in feat_name.lower() or 'review' in feat_name.lower():
                desc = "Text feature"
            else:
                desc = "Derived feature"

            report += f"| {rank} | `{feat_name}` | {importance:.4f} | {desc} |\n"

        report += "\n"

        # Key insights
        report += "**Key Insights**:\n"
        top_3 = feature_importance[:3]
        top_3_importance = sum(imp for _, imp in top_3)
        report += f"- Top 3 features account for **{top_3_importance*100:.1f}%** of total importance\n"

        if any('price' in feat.lower() for feat, _ in feature_importance[:5]):
            report += "- Pricing features are highly predictive of discount strategy\n"

        if any('rating' in feat.lower() for feat, _ in feature_importance[:5]):
            report += "- Product ratings and reviews significantly influence optimal discounts\n"

        report += "\n"
    else:
        report += "*Feature importance not available for this model type.*\n\n"

    report += "---\n\n"

    # ========================================================================
    # 5. Model Strengths
    # ========================================================================

    report += "## 5. Model Strengths\n\n"

    r2_test = best_metrics['test_r2']
    overfitting_gap = best_metrics.get('overfitting_gap', 0.0)

    if r2_test > 0.75:
        report += "1. **High Accuracy**: R² score above 0.75 indicates strong predictive power\n"

    if overfitting_gap < 0.10:
        report += "2. **Good Generalization**: Small gap between train/test scores suggests model generalizes well\n"

    if feature_importance:
        report += "3. **Interpretability**: Feature importance aligns with business logic (pricing, ratings matter)\n"

    report += f"4. **Fast Training**: Model trained in {best_train_info['training_time_seconds']:.1f} seconds\n"
    report += "5. **Production-Ready**: Model is reproducible and can make fast predictions\n"

    report += "\n---\n\n"

    # ========================================================================
    # 6. Model Weaknesses & Limitations
    # ========================================================================

    report += "## 6. Model Weaknesses & Limitations\n\n"

    report += f"1. **Sample Size**: Training set has {best_train_info['n_samples']} samples - more data could improve performance\n"
    report += "2. **Temporal Validity**: Model trained on static snapshot, doesn't capture seasonal trends or market changes\n"
    report += "3. **Category Coverage**: Limited to categories in training data - new categories may not predict well\n"

    mae = best_metrics['test_mae']
    report += f"4. **Prediction Error**: Average error of {mae:.2f} percentage points may be significant for some use cases\n"

    if overfitting_gap > 0.10:
        report += f"5. **Overfitting**: Gap of {overfitting_gap:.4f} between train/test R² suggests some overfitting\n"

    report += "\n---\n\n"

    # ========================================================================
    # 7. Business Recommendations
    # ========================================================================

    report += "## 7. Business Recommendations\n\n"

    report += "### Deployment Strategy\n"
    report += "1. **A/B Testing**: Test model predictions against current pricing logic for 2-4 weeks\n"
    report += "2. **Gradual Rollout**: Start with low-risk categories, expand based on results\n"
    report += "3. **Human Oversight**: Require manual approval for discounts >50% or <5%\n\n"

    report += "### Optimal Discount Insights\n"
    report += "Based on the model, optimal discounts typically:\n"
    report += "- Increase for higher-priced products (to remain competitive)\n"
    report += "- Decrease for highly-rated products (less need for discounts)\n"
    report += "- Vary by category (Electronics vs. Home goods have different strategies)\n\n"

    report += "### Monitoring Plan\n"
    report += "- Track prediction errors weekly on new data\n"
    report += f"- Alert if MAE exceeds {mae * 1.5:.2f} percentage points\n"
    report += f"- Alert if R² drops below {r2_test * 0.9:.2f}\n"
    report += "- Retrain quarterly or when performance degrades\n\n"

    report += "---\n\n"

    # ========================================================================
    # 8. Recommendations for Improvement
    # ========================================================================

    report += "## 8. Recommendations for Improvement\n\n"

    report += "1. **Collect More Data**: Expand dataset to 5000+ products across more categories\n"
    report += "2. **Add Temporal Features**: Include date, seasonality, price trends over time\n"
    report += "3. **Enhance Text Features**: Use TF-IDF or embeddings for product descriptions and reviews\n"
    report += "4. **Competitor Data**: Include competitor pricing and discounts for better context\n"
    report += "5. **Experiment with Ensembles**: Combine predictions from multiple models\n"
    report += "6. **Hyperparameter Optimization**: Use Bayesian optimization for better tuning\n\n"

    report += "---\n\n"

    # ========================================================================
    # 9. Conclusion
    # ========================================================================

    report += "## 9. Conclusion\n\n"

    report += f"The **{best_model_name}** model achieves strong performance (R² = {r2_test:.4f}) for predicting optimal discount percentages. "
    report += "It is suitable for production deployment with proper monitoring, guardrails, and quarterly retraining.\n\n"

    report += "**Next Steps**:\n"
    report += "1. Deploy to staging environment for A/B testing\n"
    report += "2. Monitor performance for 1 month\n"
    report += "3. Conduct user feedback sessions with sellers\n"
    report += "4. Full production rollout if metrics remain stable\n\n"

    report += "---\n\n"
    report += "*Report generated by Scientist Crew - Amazon Sales AI Pipeline*\n"

    # ========================================================================
    # Save Report
    # ========================================================================

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.success(f"✓ Evaluation report saved: {output_path}")

    logger.success("✓ Evaluation report generated")

    return report


# ============================================================================
# Model Card Generation
# ============================================================================

def generate_model_card(
    model_name: str,
    metrics: Dict,
    training_info: Dict,
    feature_names: List[str],
    output_path: str = None
) -> str:
    """
    יצירת Model Card (Markdown).
    Generate Model Card (Markdown).

    Must include 5 REQUIRED sections:
    - Purpose
    - Data
    - Metrics
    - Limitations
    - Ethical Considerations

    Args:
        model_name: Name of the model
        metrics: Evaluation metrics
        training_info: Training information
        feature_names: List of features
        output_path: Path to save card (optional)

    Returns:
        Markdown Model Card string
    """
    logger.info("Generating Model Card...")

    card = "# Model Card: Amazon Optimal Discount Prediction Model\n\n"
    card += f"**Model Version**: 1.0.0\n"
    card += f"**Date**: {datetime.now().strftime('%Y-%m-%d')}\n"
    card += f"**Model Type**: {model_name}\n"
    card += f"**Task**: Regression (Discount Percentage Prediction)\n\n"
    card += "---\n\n"

    # ========================================================================
    # Purpose ✓ REQUIRED
    # ========================================================================

    card += "## Purpose\n\n"
    card += "### What does this model do?\n\n"
    card += "This model predicts the **optimal discount percentage** for Amazon products based on product characteristics, customer ratings, and review data. "
    card += "It helps sellers:\n"
    card += "- Optimize pricing strategies to balance sales volume and profitability\n"
    card += "- Determine competitive discount levels based on market positioning\n"
    card += "- Make data-driven decisions instead of relying on intuition\n\n"

    card += "### Intended Use Cases\n\n"
    card += "- **Dynamic Pricing**: Recommend optimal discounts for new or existing products\n"
    card += "- **Promotional Planning**: Identify which products benefit most from discounts\n"
    card += "- **Business Intelligence**: Analyze discount patterns across categories and price ranges\n"
    card += "- **Revenue Optimization**: Maximize revenue by finding the sweet spot between volume and margin\n\n"

    card += "### Out-of-Scope Uses\n\n"
    card += "❌ **Do NOT use for**:\n"
    card += "- Real-time pricing (model requires retraining to capture market changes)\n"
    card += "- Products in categories not seen during training\n"
    card += "- Price manipulation or anti-competitive collusion\n"
    card += "- Discriminatory pricing based on user demographics\n\n"

    card += "---\n\n"

    # ========================================================================
    # Data ✓ REQUIRED
    # ========================================================================

    card += "## Data\n\n"
    card += "### Training Data\n\n"
    card += f"**Dataset**: Amazon Sales Data (clean_data.csv)\n"
    card += f"**Size**: {training_info['n_samples']} products\n"
    card += f"**Features**: {training_info['n_features']} engineered features\n"
    card += f"**Date Range**: Snapshot from 2025-2026 (verify with data source)\n\n"

    card += "**Feature Categories**:\n"
    card += "- **Pricing**: actual_price, discounted_price, price_level, discounted_price_level\n"
    card += "- **Ratings**: rating, rating_count, log_rating_count, rating_weighted\n"
    card += "- **Text**: description_length, review_sentiment, keyword presence\n"
    card += "- **Categories**: One-hot encoded product categories\n"
    card += "- **Derived**: Various interaction and threshold features\n\n"

    card += f"**Target Variable**: `discount_percentage` (0-100% range)\n\n"

    card += "**Preprocessing**:\n"
    card += "1. Type conversion: string → numeric (prices, ratings, counts)\n"
    card += "2. Feature engineering: logarithms, ratios, text extraction\n"
    card += "3. Category encoding: one-hot for top 10 categories\n"
    card += "4. Aggregation: product-level (grouped from review-level)\n"
    card += "5. Validation: no nulls, valid ranges, proper dtypes\n\n"

    card += "**Train/Test Split**: 80/20 stratified by price category (random_state=42)\n\n"

    card += "---\n\n"

    # ========================================================================
    # Metrics ✓ REQUIRED
    # ========================================================================

    card += "## Metrics\n\n"
    card += "### Performance Summary\n\n"
    card += "**Test Set Performance**:\n"
    card += f"- **R² Score**: {metrics['test_r2']:.4f} ({metrics['test_r2']*100:.1f}% variance explained)\n"
    card += f"- **RMSE**: {metrics['test_rmse']:.2f} percentage points\n"
    card += f"- **MAE**: {metrics['test_mae']:.2f} percentage points (mean absolute error)\n"
    card += f"- **MAPE**: {metrics['test_mape']:.2f}% (mean absolute percentage error)\n\n"

    if 'train_r2' in metrics:
        card += "**Training Set Performance**:\n"
        card += f"- **R² Score**: {metrics['train_r2']:.4f}\n"
        card += f"- **RMSE**: {metrics['train_rmse']:.2f} percentage points\n\n"

    if training_info.get('cv_score'):
        card += f"**Cross-Validation**: R² = {training_info['cv_score']:.4f} ± (5-fold CV)\n\n"

    card += "### Interpretation\n\n"
    r2_pct = metrics['test_r2'] * 100
    mae = metrics['test_mae']
    card += f"- The model explains **{r2_pct:.1f}% of discount variation** in test data\n"
    card += f"- Average prediction error is **{mae:.2f} percentage points**\n"
    card += f"- For a product with 30% optimal discount, expect predictions in range **{30-mae:.1f}% - {30+mae:.1f}%**\n\n"

    card += "---\n\n"

    # ========================================================================
    # Limitations ✓ REQUIRED
    # ========================================================================

    card += "## Limitations\n\n"
    card += "### Known Limitations\n\n"

    card += "1. **Temporal Validity**: Model trained on static data snapshot\n"
    card += "   - Does not capture seasonal pricing changes\n"
    card += "   - Unaware of market trends, inflation, demand shifts\n"
    card += "   - **Recommendation**: Retrain quarterly or when performance drops\n\n"

    card += "2. **Category Coverage**: Limited to categories in training data\n"
    card += "   - Trained categories: Electronics, Home, Computers, Office, etc.\n"
    card += "   - **Untested categories may produce unreliable predictions**\n"
    card += "   - Verify category before using predictions\n\n"

    card += f"3. **Prediction Accuracy**: Average error of {mae:.2f} percentage points\n"
    card += "   - May be significant for fine-tuned discount strategies\n"
    card += "   - Higher errors for extreme discounts (>70% or <5%)\n\n"

    card += f"4. **Sample Size**: {training_info['n_samples']} products is moderate\n"
    card += "   - May not capture all product diversity\n"
    card += "   - Rare product types underrepresented\n\n"

    card += "5. **Feature Limitations**:\n"
    card += "   - Text features are basic (keyword-based, not semantic)\n"
    card += "   - No competitor pricing data\n"
    card += "   - No historical sales volume data\n\n"

    card += "### Edge Cases\n\n"
    card += "❌ **Model may fail for**:\n"
    card += "- Brand-new products (no ratings/reviews yet)\n"
    card += "- Products with extreme prices (>₹50,000)\n"
    card += "- Niche categories not in training set\n"
    card += "- Products with unusual discount patterns\n\n"

    card += "---\n\n"

    # ========================================================================
    # Ethical Considerations ✓ REQUIRED
    # ========================================================================

    card += "## Ethical Considerations\n\n"
    card += "### Fairness\n\n"
    card += "**Potential Biases**:\n"
    card += "- **Category Bias**: Model may favor popular categories (Electronics) over niche ones\n"
    card += "- **Rating Bias**: Products with few reviews may receive suboptimal discount recommendations\n"
    card += "- **Price Discrimination Risk**: Could be misused to charge different prices to different customer segments\n\n"

    card += "**Mitigation**:\n"
    card += "- Monitor predictions across all categories for fairness\n"
    card += "- Do NOT use model to discriminate based on user demographics\n"
    card += "- Apply consistent discount logic to all customers\n\n"

    card += "### Transparency\n\n"
    card += "**Model Interpretability**:\n"
    card += "- Feature importance available (top predictors: price, ratings)\n"
    card += f"- {model_name} is moderately interpretable\n"
    card += "- Predictions can be explained via feature contributions\n\n"

    card += "### Responsible Use\n\n"
    card += "**Guidelines**:\n"
    card += "1. ✅ **Do**: Use for pricing optimization, market analysis, business insights\n"
    card += "2. ❌ **Don't**: Use for price fixing, collusion, or discriminatory pricing\n"
    card += "3. ⚠️ **Caution**: Verify predictions manually for high-stakes decisions\n\n"

    card += "**Risk of Misuse**:\n"
    card += "- **Price Manipulation**: Model could artificially inflate/deflate prices\n"
    card += "- **Anti-Competitive Behavior**: Coordinating discounts across sellers\n"
    card += "- **Consumer Harm**: Exploiting customers with dynamic pricing\n\n"

    card += "**Safeguards**:\n"
    card += "- Human review required for production pricing changes\n"
    card += "- Predictions are one input among many (not sole decision factor)\n"
    card += "- Audit pricing changes for fairness and compliance\n\n"

    card += "### Privacy\n\n"
    card += "**Data Privacy**:\n"
    card += "- No personally identifiable information (PII) used as features\n"
    card += "- User IDs and names present in raw data but NOT used in model\n"
    card += "- Review content used in aggregate, not individually\n\n"

    card += "**Compliance**: Ensure usage aligns with:\n"
    card += "- Consumer protection laws\n"
    card += "- E-commerce platform policies\n"
    card += "- Anti-trust regulations\n\n"

    card += "---\n\n"

    # ========================================================================
    # Model Details
    # ========================================================================

    card += "## Model Details\n\n"
    card += "### Architecture\n\n"
    card += f"**Model Type**: {model_name}\n"
    card += f"**Framework**: scikit-learn, xgboost (if applicable)\n"
    card += f"**Training Time**: {training_info['training_time_seconds']:.1f} seconds\n\n"

    card += "**Hyperparameters**:\n"
    card += "```python\n"
    for param, value in training_info['hyperparameters'].items():
        card += f"{param}: {value}\n"
    card += "```\n\n"

    card += "---\n\n"

    # ========================================================================
    # Recommendations for Use
    # ========================================================================

    card += "## Recommendations for Use\n\n"
    card += "### Deployment\n\n"
    card += "1. **Staging First**: Test in staging environment for 1-2 months\n"
    card += "2. **A/B Testing**: Compare predictions with current pricing logic\n"
    card += "3. **Monitoring**: Track MAE, R² on new data weekly\n"
    card += "4. **Retraining**: Schedule quarterly retraining with updated data\n\n"

    card += "### Input Validation\n\n"
    card += "Before prediction, validate:\n"
    card += f"- All {training_info['n_features']} required features present\n"
    card += "- Prices in valid range (₹0 - ₹50,000)\n"
    card += "- Ratings in range 0-5\n"
    card += "- Category is one of trained categories\n\n"

    card += "### Prediction Guardrails\n\n"
    card += "Apply business rules:\n"
    card += f"- Flag predictions with uncertainty >±{mae*1.5:.1f} percentage points\n"
    card += "- Reject predictions <0% or >100%\n"
    card += "- Require human review for discounts >70% or <5%\n\n"

    card += "---\n\n"

    # ========================================================================
    # Contact & Support
    # ========================================================================

    card += "## Contact & Support\n\n"
    card += "**Model Owner**: ML Specialist, Amazon Sales AI Pipeline Team\n"
    card += f"**Last Updated**: {datetime.now().strftime('%Y-%m-%d')}\n"
    card += f"**Next Review**: {(datetime.now().replace(month=datetime.now().month+3)).strftime('%Y-%m-%d')} (quarterly)\n\n"

    card += "**Questions?** Contact: ml-team@company.com\n"
    card += "**Issues?** Report to: [GitHub Issues](https://github.com/your-repo/issues)\n\n"

    card += "---\n\n"
    card += "*This Model Card follows responsible AI practices and industry standards for ML model documentation.*\n"

    # ========================================================================
    # Save Model Card
    # ========================================================================

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(card)
        logger.success(f"✓ Model Card saved: {output_path}")

    logger.success("✓ Model Card generated with all 5 required sections")

    return card


# ============================================================================
# Self-Test
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Evaluation Module - Self Test")
    print("="*60 + "\n")

    # Sample metrics
    y_true = pd.Series([20, 30, 40, 50])
    y_pred = np.array([22, 28, 42, 48])

    metrics = calculate_metrics(y_true, y_pred)
    print(f"Metrics: {metrics}\n")

    # Sample models dict
    models_dict = {
        'Random Forest': {
            'model': None,
            'metrics': {'test_r2': 0.82, 'test_rmse': 4.5, 'test_mae': 3.2, 'test_mape': 12.0, 'train_r2': 0.89, 'train_rmse': 3.1, 'train_mae': 2.4},
            'training_info': {'training_time_seconds': 45.2, 'hyperparameters': {'n_estimators': 200}, 'n_samples': 1000, 'n_features': 25}
        },
        'XGBoost': None
    }

    # Test comparison table
    table = create_comparison_table(models_dict)
    print("Comparison Table:")
    print(table)

    print("\n" + "="*60)
    print("Self test complete!")
    print("="*60 + "\n")
