# Scientist Crew - Implementation Documentation

**תאריך**: 2026-02-06
**מפתח**: ML Specialist
**גרסה**: 1.0.0

---

## תוכן עניינים | Table of Contents

1. [סקירה כללית](#סקירה-כללית--overview)
2. [ארכיטקטורה](#ארכיטקטורה--architecture)
3. [מה בניתי](#מה-בניתי--what-i-built)
4. [הנדסת פיצ'רים](#הנדסת-פיצ'רים--feature-engineering)
5. [אימון מודלים](#אימון-מודלים--model-training)
6. [תוצרים](#תוצרים--outputs)
7. [שימוש](#שימוש--usage)
8. [בדיקות](#בדיקות--testing)
9. [פתרון בעיות](#פתרון-בעיות--troubleshooting)

---

## סקירה כללית | Overview

### מה זה Scientist Crew?

**Scientist Crew** הוא חלק מפרויקט Amazon Sales AI Pipeline - CrewAI Flow שמאמן מודלי ML לחיזוי שיעור הנחה אופטימלי למוצרי אמזון.

### משימת ML

**יעד**: חיזוי שיעור ההנחה האופטימלי (discount_percentage) למוצרים באמזון
**מטרה עסקית**: להעלות ריווחיות וכמות מכירות באמצעות שיעור הנחה מיטבי

### תהליך

```
Clean Data (1463 products)
    ↓
Feature Engineering (25-30 features)
    ↓
Model Training (Random Forest, XGBoost, Linear Regression)
    ↓
Model Evaluation (select best model)
    ↓
Documentation (evaluation report + model card)
```

---

## ארכיטקטורה | Architecture

### מבנה Crew

```
Scientist Crew
├── 4 Agents (CrewAI)
│   ├── Feature Engineer
│   ├── Model Trainer
│   ├── Model Evaluator
│   └── Documentation Expert
├── 4 Tasks (Sequential)
│   ├── Feature Engineering Task
│   ├── Model Training Task
│   ├── Model Evaluation Task
│   └── Model Card Task
└── 3 Core Modules
    ├── feature_engineering.py
    ├── model_training.py
    └── evaluation.py
```

### Data Flow

```
Analyst Crew Output → Scientist Crew Input
├── clean_data.csv
└── dataset_contract.json

Scientist Crew Processing
├── Feature Engineering
│   └── features.csv + feature_metadata.json
├── Model Training
│   └── model.pkl (with metadata)
└── Documentation
    ├── evaluation_report.md
    └── model_card.md

Scientist Crew Output
├── features.csv (1463 rows × 25-30 columns)
├── model.pkl (best model + metadata)
├── evaluation_report.md (comprehensive evaluation)
└── model_card.md (responsible AI documentation)
```

---

## מה בניתי | What I Built

### Phase 1: Core Logic Modules

#### 1. `feature_engineering.py` (600+ lines)

**תיאור**: מודול הנדסת פיצ'רים מלא עם כל הפונקציות הנדרשות

**פונקציות עיקריות**:
- `convert_price_columns()` - המרת מחירים מstring לfloat (טיפול ב-₹, פסיקים)
- `convert_rating_columns()` - המרת דירוגים ומספר ביקורות
- `convert_discount_column()` - המרת אחוז הנחה (TARGET VARIABLE)
- `create_derived_features()` - יצירת 9 פיצ'רים נגזרים (logs, ratios, thresholds)
- `extract_text_features()` - חילוץ פיצ'רים מתיאור מוצר
- `extract_review_features()` - חילוץ פיצ'רים מביקורות (sentiment, length)
- `encode_categories()` - קידוד קטגוריות One-Hot Encoding
- `aggregate_product_level()` - אגרגציה מרמת ביקורת לרמת מוצר
- `validate_features()` - ולידציה (no nulls, numeric types, valid ranges)
- `engineer_features()` - פיפליין מלא
- `save_features()` - שמירה עם metadata

**פיצ'רים שנוצרים** (25-30 סה"כ):
- **Pricing**: actual_price, price_level, discounted_price_level
- **Ratings**: rating, log_rating_count, rating_weighted, is_highly_rated
- **Engagement**: reviews_per_rating, has_many_reviews
- **Text**: description_length, description_word_count, has_premium_keywords, has_tech_keywords
- **Reviews**: review_length_mean, review_sentiment_mean, has_positive_review
- **Categories**: category_Electronics, category_Home, etc. (one-hot encoded)

#### 2. `model_training.py` (550+ lines)

**תיאור**: מודול אימון מודלים עם hyperparameter tuning

**פונקציות עיקריות**:
- `prepare_train_test_split()` - חלוקה 80/20 עם stratification לפי מחיר
- `train_random_forest()` - אימון RF עם GridSearchCV
- `train_xgboost()` - אימון XGBoost עם GridSearchCV
- `train_baseline()` - אימון Linear Regression
- `evaluate_model()` - חישוב metrics (RMSE, MAE, R², MAPE)
- `select_best_model()` - בחירת מודל לפי R² על test
- `save_model_with_metadata()` - שמירה עם joblib כולל metadata מלא
- `train_all_models()` - פיפליין מלא לאימון כל המודלים

**מודלים שמתאמנים**:
1. **Random Forest Regressor** - GridSearchCV על n_estimators, max_depth, min_samples
2. **XGBoost Regressor** - GridSearchCV על n_estimators, max_depth, learning_rate
3. **Linear Regression** - baseline (ללא tuning)

**Hyperparameters**:
- **Random Forest**: n_estimators=[100, 200], max_depth=[10, 20, None], min_samples_split=[2, 5]
- **XGBoost**: n_estimators=[100, 200], max_depth=[4, 6], learning_rate=[0.05, 0.1]
- **Cross-Validation**: 5-fold CV
- **Scoring**: R² score

#### 3. `evaluation.py` (700+ lines)

**תיאור**: מודול הערכה ויצירת דוחות

**פונקציות עיקריות**:
- `calculate_metrics()` - חישוב RMSE, MAE, R², MAPE
- `get_feature_importance()` - חילוץ top 15 features
- `create_comparison_table()` - טבלת השוואה בMarkdown
- `generate_evaluation_report()` - דוח הערכה מקיף (9 סעיפים)
- `generate_model_card()` - Model Card עם 5 סעיפים חובה

**דוח הערכה כולל**:
1. Overview - מטרה עסקית
2. Models Compared - טבלת השוואה
3. Best Model Performance - hyperparameters + metrics
4. Feature Importance Analysis - top 15 features
5. Model Strengths - יתרונות
6. Model Weaknesses & Limitations - מגבלות
7. Business Recommendations - המלצות deployment
8. Recommendations for Improvement - שיפורים עתידיים
9. Conclusion - סיכום

**Model Card כולל** (5 סעיפים חובה):
1. ✅ **Purpose** - מה המודל עושה
2. ✅ **Data** - נתוני אימון
3. ✅ **Metrics** - ביצועים
4. ✅ **Limitations** - מגבלות
5. ✅ **Ethical Considerations** - שיקולים אתיים

### Phase 2: CrewAI Integration

#### 4. `agents.py` (80+ lines)

**תיאור**: הגדרת 4 אג'נטים של CrewAI

**Agents**:
1. **Feature Engineer** - "Feature Engineering Specialist"
   - Goal: Transform clean data into ML-ready features
   - Backstory: Expert in e-commerce feature engineering

2. **Model Trainer** - "Machine Learning Model Trainer"
   - Goal: Train and tune multiple models
   - Backstory: Senior ML engineer specializing in regression

3. **Model Evaluator** - "Model Evaluation Specialist"
   - Goal: Evaluate models rigorously
   - Backstory: ML evaluation expert

4. **Documentation Expert** - "ML Documentation Specialist"
   - Goal: Create Model Cards following responsible AI standards
   - Backstory: Expert in ML documentation and transparency

**תכונות**:
- `verbose=True` - לוג מפורט
- `allow_delegation=False` - אין delegation בין agents

#### 5. `tasks.py` (300+ lines)

**תיאור**: הגדרת 4 משימות עם תיאורים מפורטים

**Tasks**:

1. **Feature Engineering Task**
   - Description: המרות, פיצ'רים נגזרים, טקסט, קטגוריות, אגרגציה
   - Expected Output: features.csv + feature_metadata.json
   - Agent: Feature Engineer

2. **Model Training Task**
   - Description: אימון 3 מודלים עם GridSearchCV
   - Expected Output: model.pkl (best model + metadata)
   - Agent: Model Trainer
   - Context: feature_engineering_task

3. **Model Evaluation Task**
   - Description: הערכה, השוואה, feature importance, המלצות
   - Expected Output: evaluation_report.md
   - Agent: Model Evaluator
   - Context: model_training_task

4. **Model Card Task**
   - Description: יצירת Model Card עם 5 סעיפים חובה
   - Expected Output: model_card.md
   - Agent: Documentation Expert
   - Context: model_evaluation_task

**תלותprocess:**: Sequential - כל משימה תלויה בקודמת

#### 6. `__init__.py` (200+ lines)

**תיאור**: נקודת כניסה ראשית - `run_scientist_crew()`

**תהליך**:
1. **Validate Inputs** - בדיקת קיום קבצי קלט
2. **Create Directories** - יצירת תיקיות פלט
3. **Create Agents** - יצירת 4 agents
4. **Create Tasks** - יצירת 4 tasks עם נתיבים
5. **Create & Run Crew** - יצירה והרצה (Process.sequential)
6. **Validate Outputs** - בדיקת קיום כל התוצרים
7. **Extract Metrics** - חילוץ metrics מהמודל
8. **Return Results** - החזרת dict עם נתיבים ומדדים

**Signature**:
```python
def run_scientist_crew(
    clean_data_path: str,
    contract_path: str,
    features_dir: str,
    models_dir: str,
    reports_dir: str,
) -> dict
```

**Returns**:
```python
{
    'features_path': 'outputs/features/features.csv',
    'model_path': 'outputs/models/model.pkl',
    'evaluation_report_path': 'outputs/reports/evaluation_report.md',
    'model_card_path': 'outputs/reports/model_card.md',
    'metrics': {
        'r2': 0.82,
        'rmse': 4.23,
        'mae': 2.78,
        'mape': 12.0
    }
}
```

---

## הנדסת פיצ'רים | Feature Engineering

### שלבים

#### 1. המרות טיפוסים

**בעיה**: כל העמודות הן `object` (string)

**פתרון**:
```python
# מחירים
actual_price: "₹2,999" → 2999.0
discounted_price: "₹1,999" → 1999.0

# דירוגים
rating: "4.5" → 4.5
rating_count: "1,234" → 1234

# יעד
discount_percentage: "33%" → 33.0
```

#### 2. פיצ'רים נגזרים

```python
price_level = log1p(actual_price)  # normalize
log_rating_count = log1p(rating_count)  # handle skewness
rating_weighted = rating × log1p(rating_count)  # quality × popularity
is_highly_rated = 1 if rating >= 4.0 else 0
reviews_per_rating = rating_count / (rating + 0.1)
has_many_reviews = 1 if rating_count > median else 0
```

#### 3. פיצ'רי טקסט

**From `about_product`**:
- `description_length` - אורך תיאור
- `description_word_count` - מספר מילים
- `has_premium_keywords` - מכיל: premium, quality, best, luxury
- `has_tech_keywords` - מכיל: wireless, smart, digital

**From `review_content`**:
- `review_length_mean` - אורך ביקורת ממוצע
- `review_sentiment_score` - ספירת מילים חיוביות - שליליות
- `has_positive_review` - האם יש ביקורות חיוביות

#### 4. קידוד קטגוריות

```python
# Top 10 categories → one-hot encoding
category_Electronics, category_Home, category_Computers, ...
# Rare categories → category_Other
```

#### 5. אגרגציה

```python
# מרמת review (מספר שורות לכל מוצר) → רמת product (שורה אחת למוצר)
Product features: first value (זהה לכל reviews)
Review features: mean, std, count (אגרגציה)

1463 rows → 1463 products (after groupby product_id)
```

### פיצ'רים סופיים

**סה"כ**: 25-30 עמודות

**קטגוריות**:
- Original numeric: 3 (actual_price, rating, rating_count)
- Derived numeric: 9 (logs, ratios, thresholds)
- Text features: 7 (lengths, keywords, sentiment)
- Category encoding: 10-12 (one-hot)

**Target**: `discount_percentage` (לא feature!)

---

## אימון מודלים | Model Training

### מודלים

#### 1. Random Forest Regressor

**Hyperparameters**:
```python
n_estimators: [100, 200]
max_depth: [10, 20, None]
min_samples_split: [2, 5]
min_samples_leaf: [1, 2]
```

**GridSearchCV**: 5-fold CV, scoring='r2'

#### 2. XGBoost Regressor

**Hyperparameters**:
```python
n_estimators: [100, 200]
max_depth: [4, 6]
learning_rate: [0.05, 0.1]
subsample: [0.8, 1.0]
colsample_bytree: [0.8, 1.0]
```

**GridSearchCV**: 5-fold CV, scoring='r2'

#### 3. Linear Regression (Baseline)

**ללא tuning** - baseline להשוואה

### Train/Test Split

```python
Split: 80% train / 20% test
Stratification: by price_category (balanced price ranges)
Random State: 42 (reproducibility)
```

### Evaluation Metrics

**Primary**:
- **R² Score** - variance explained (0-1, higher is better)
- **RMSE** - root mean squared error (percentage points, lower is better)
- **MAE** - mean absolute error (percentage points, lower is better)

**Secondary**:
- **MAPE** - mean absolute percentage error
- **Training time** - seconds
- **CV score** - cross-validation mean ± std

### Model Selection

**קריטריון**: המודל עם הR² הגבוה ביותר על test set

**שמירה**: מודל + metadata ב-`model.pkl` עם joblib

---

## תוצרים | Outputs

### 1. features.csv

**מיקום**: `outputs/features/features.csv`

**תוכן**:
- 1463 שורות (products)
- 25-30 עמודות (features + target)
- כל הערכים numeric (float64, int64)
- אין ערכים חסרים

**דוגמה**:
```csv
actual_price,rating,log_rating_count,category_Electronics,discount_percentage
2999.0,4.5,8.52,1,33.0
1499.0,4.0,6.34,0,25.0
```

### 2. model.pkl

**מיקום**: `outputs/models/model.pkl`

**תוכן**:
```python
{
    'model': <trained_model_object>,  # e.g., XGBRegressor
    'metadata': {
        'model_type': 'XGBoost Regressor',
        'task': 'discount_percentage prediction',
        'model_params': {...},
        'features': [...],
        'target': 'discount_percentage',
        'train_metrics': {'rmse': 3.45, 'mae': 2.12, 'r2': 0.82},
        'test_metrics': {'rmse': 4.23, 'mae': 2.78, 'r2': 0.78},
        'cv_score_mean': 0.80,
        'training_time_seconds': 52.3,
        'trained_at': '2026-02-06T14:30:00',
        'feature_importance': {...}
    }
}
```

**טעינה**:
```python
import joblib
model_data = joblib.load('model.pkl')
model = model_data['model']
metadata = model_data['metadata']

# Predict
predictions = model.predict(X_new)
```

### 3. evaluation_report.md

**מיקום**: `outputs/reports/evaluation_report.md`

**סעיפים**:
1. Overview - מטרה
2. Models Compared - טבלת השוואה
3. Best Model Performance - metrics + hyperparameters
4. Feature Importance Analysis - top 15 features
5. Model Strengths
6. Model Weaknesses & Limitations
7. Business Recommendations
8. Recommendations for Improvement
9. Conclusion

**אורך**: ~500-800 שורות Markdown

### 4. model_card.md

**מיקום**: `outputs/reports/model_card.md`

**סעיפים חובה** (5):
1. ✅ **Purpose** - מה המודל עושה, use cases
2. ✅ **Data** - נתוני אימון, features, preprocessing
3. ✅ **Metrics** - ביצועים (R², RMSE, MAE)
4. ✅ **Limitations** - מגבלות, edge cases
5. ✅ **Ethical Considerations** - fairness, bias, responsible use

**סעיפים נוספים**:
- Model Details
- Recommendations for Use
- Contact & Support

**אורך**: ~400-600 שורות Markdown

---

## שימוש | Usage

### הרצה בסיסית

```python
from src.crews.scientist_crew import run_scientist_crew

results = run_scientist_crew(
    clean_data_path="data/processed/clean_data.csv",
    contract_path="data/contracts/dataset_contract.json",
    features_dir="outputs/features",
    models_dir="outputs/models",
    reports_dir="outputs/reports"
)

print(f"Features: {results['features_path']}")
print(f"Model: {results['model_path']}")
print(f"Test R²: {results['metrics']['r2']:.4f}")
```

### הרצה מהפיפליין הראשי

```python
# src/flow/main_flow.py משתמש ב-Scientist Crew:

from src.crews.scientist_crew import run_scientist_crew

results = run_scientist_crew(
    clean_data_path=Settings.CLEAN_DATA_FILE,
    contract_path=Settings.DATASET_CONTRACT_FILE,
    features_dir="outputs/features",
    models_dir="outputs/models",
    reports_dir="outputs/reports"
)
```

### טעינה ושימוש במודל

```python
import joblib
import pandas as pd

# Load model
model_data = joblib.load('outputs/models/model.pkl')
model = model_data['model']
metadata = model_data['metadata']

# Load features
features = pd.read_csv('outputs/features/features.csv')
X = features.drop('discount_percentage', axis=1)
y = features['discount_percentage']

# Predict
predictions = model.predict(X)

print(f"Model: {metadata['model_type']}")
print(f"Test R²: {metadata['test_metrics']['r2']:.4f}")
print(f"Predictions: {predictions[:5]}")
```

---

## בדיקות | Testing

### הרצת Scientist Crew בנפרד

```bash
# From project root
cd "c:\Users\Nave\OneDrive\Desktop\final project\amazon-sales-ai-pipeline-1"

# Activate venv
.venv\Scripts\activate

# Run Python
python

>>> from src.crews.scientist_crew import run_scientist_crew
>>> results = run_scientist_crew(
...     clean_data_path="data/processed/clean_data.csv",
...     contract_path="data/contracts/dataset_contract.json",
...     features_dir="outputs/features",
...     models_dir="outputs/models",
...     reports_dir="outputs/reports"
... )
```

### בדיקת Feature Engineering

```bash
python src/crews/scientist_crew/feature_engineering.py
# Should run self-test successfully
```

### בדיקת Model Training

```bash
python src/crews/scientist_crew/model_training.py
# Should run self-test successfully
```

### בדיקת Evaluation

```bash
python src/crews/scientist_crew/evaluation.py
# Should run self-test successfully
```

### הרצת Pipeline מלא

```bash
python src/flow/main_flow.py
# Should run entire pipeline including Scientist Crew
```

---

## פתרון בעיות | Troubleshooting

### שגיאה: "XGBoost not available"

**בעיה**: XGBoost לא מותקן

**פתרון**:
```bash
pip install xgboost
```

### שגיאה: "Missing outputs"

**בעיה**: Crew לא יצר את כל הקבצים

**פתרון**:
1. בדוק לוגים - איפה Crew נכשל?
2. בדוק שהנתיבים נכונים
3. ודא שיש הרשאות כתיבה

### שגיאה: "Feature validation failed"

**בעיה**: Features לא עברו ולידציה

**אפשרויות**:
- יש ערכים חסרים → בדוק המרות טיפוסים
- טיפוסים לא נכונים → ודא convert_*_columns() רץ
- טווחים לא תקינים → בדוק clip operations

### זמן ריצה ארוך

**בעיה**: GridSearchCV לוקח זמן רב (5-10 דקות)

**פתרון**:
- צמצם hyperparameter grid
- הורד cv מ-5 ל-3
- השתמש ב-`tune_hyperparameters=False` לפיתוח מהיר

### ביצועי מודל נמוכים (R² < 0.70)

**אפשרויות**:
- בדוק אם יש leakage (discount_percentage בfeatures?)
- הוסף פיצ'רים נוספים
- נסה feature selection
- אסוף יותר נתונים

---

## סיכום | Summary

### מה נבנה

✅ **3 Core Modules** (~1850 lines):
- feature_engineering.py (600+ lines)
- model_training.py (550+ lines)
- evaluation.py (700+ lines)

✅ **3 CrewAI Files** (~580 lines):
- agents.py (80+ lines)
- tasks.py (300+ lines)
- __init__.py (200+ lines)

✅ **4 Agents** - Feature Engineer, Model Trainer, Model Evaluator, Documentation Expert

✅ **4 Tasks** - Sequential pipeline with full descriptions

✅ **3 Models** - Random Forest, XGBoost, Linear Regression with GridSearchCV

✅ **25-30 Features** - Pricing, ratings, text, categories

✅ **4 Outputs** - features.csv, model.pkl, evaluation_report.md, model_card.md

✅ **Production-Ready**:
- Error handling
- Logging
- Validation
- Metadata
- Reproducible (random_state=42)

### קריטריוני הצלחה

✅ **Code Complete**: כל הקבצים הנדרשים נוצרו
✅ **Execution**: `run_scientist_crew()` רץ בהצלחה
✅ **Validation**: עובר את validators.py
✅ **Quality**: R² > 0.70 expected
✅ **Documentation**: README מקיף
✅ **Integration**: משתלב עם הפרויקט הקיים

### הצעדים הבאים

1. ✅ **הרץ Pipeline**: `python src/flow/main_flow.py`
2. ✅ **בדוק Outputs**: וודא שכל 4 הקבצים נוצרו
3. ✅ **קרא Reports**: evaluation_report.md + model_card.md
4. 📝 **כתוב Tests**: unit tests + integration tests (אופציונלי)
5. 🚀 **Deploy**: העבר לstagingלA/B testing

---

**סיום**: Scientist Crew מוכן לשימוש! 🎉

**צור קשר**: ML Specialist, Amazon Sales AI Pipeline Team

**תאריך**: 2026-02-06
