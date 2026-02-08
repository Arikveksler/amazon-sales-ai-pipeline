# תוכנית עבודה - מירב (ולידציה ויציבות)

## תפקיד כללי
ולידציה ויציבות - אחראית על חוזה הנתונים, הערכת מודל, ומנגנוני Fail Gracefully.

## Branch: `feature/meirav`

---

## שבוע 1: Dataset Contract

### משימות
| # | משימה | תלוי ב- | תוצר | Status |
|---|--------|---------|------|--------|
| 1 | הגדרת dataset_contract.json | - | dataset_contract.json | ⬜ |
| 2 | הגדרת constraints | - | constraints in JSON | ⬜ |

### מבנה החוזה
```json
{
  "dataset_name": "amazon_sales_clean",
  "version": "1.0.0",
  "created_at": "2024-XX-XX",
  "created_by": "Analyst Crew",
  "description": "Amazon product sales data for price prediction",
  "source": "Kaggle Amazon Sales Dataset",

  "required_columns": [
    "product_id",
    "product_name",
    "category",
    "discounted_price",
    "actual_price",
    "rating"
  ],

  "constraints": {
    "discounted_price": {
      "type": "numeric",
      "min": 0,
      "required": true
    },
    "actual_price": {
      "type": "numeric",
      "min": 0,
      "required": true
    },
    "rating": {
      "type": "numeric",
      "min": 0,
      "max": 5,
      "required": true
    },
    "discount_percentage": {
      "type": "numeric",
      "min": 0,
      "max": 100
    },
    "product_id": {
      "type": "string",
      "unique": true,
      "required": true
    },
    "category": {
      "type": "categorical",
      "required": true
    }
  },

  "schema": {
    "columns": ["..."],
    "dtypes": {"...": "..."},
    "row_count": 1463
  },

  "quality_checks": {
    "no_nulls": true,
    "validated": true
  },

  "min_features": 5
}
```

### חשוב לדעת
- הקובץ `validators.py` (שורה 202) מצפה לשדות `required_columns` ו-`constraints`
- חייב להוסיף אותם לחוזה הקיים!

### תוצרים
- [ ] `data/contracts/dataset_contract.json` מעודכן

---

## שבוע 2: הערכת מודל ותיעוד

### משימות
| # | משימה | תלוי ב- | תוצר | Status |
|---|--------|---------|------|--------|
| 1 | Evaluation Report | model (נווה) | evaluation_report.md | ⬜ |
| 2 | Model Card | model (נווה) | model_card.md | ⬜ |

### דוח הערכה (evaluation_report.md)
```markdown
# Model Evaluation Report

## 1. Overview
- **Model Type**: [Linear Regression / Random Forest]
- **Target Variable**: discounted_price
- **Training Date**: [תאריך]
- **Dataset Size**: 1,463 samples

## 2. Performance Metrics
| Metric | Linear Regression | Random Forest | Winner |
|--------|-------------------|---------------|--------|
| MAE    | X.XX             | X.XX          | RF/LR  |
| RMSE   | X.XX             | X.XX          | RF/LR  |
| R²     | X.XX             | X.XX          | RF/LR  |
| MAPE   | X.XX%            | X.XX%         | RF/LR  |

## 3. Feature Importance (Top 5)
1. actual_price - XX%
2. category - XX%
3. rating - XX%
4. discount_percentage - XX%
5. rating_count - XX%

## 4. Cross-Validation Results
- 5-Fold CV Mean Score: X.XX
- Standard Deviation: X.XX

## 5. Recommendations
[המלצה על איזה מודל לבחור ולמה]
```

### כרטיס מודל (model_card.md)
**חשוב**: הפונקציה `validate_model_outputs()` בודקת שהסקשנים הבאים קיימים:
- Purpose
- Data
- Metrics
- Limitations
- Ethical

```markdown
# Model Card: Amazon Sales Price Predictor

## Purpose
### Model Details
- **Name**: Amazon Sales Price Predictor
- **Version**: 1.0
- **Type**: Regression
- **Framework**: Scikit-learn

### Intended Use
- **Primary use**: Predict discounted prices for Amazon products
- **Users**: Business analysts, pricing teams
- **Out-of-scope**: Real-time production pricing decisions

## Data
### Training Data
- **Source**: Kaggle Amazon Sales Dataset
- **Size**: ~1,463 records
- **Features**: Product category, actual price, ratings, etc.

### Data Processing
- Removed rows with null values
- Categorical encoding for category field

## Metrics
### Performance Results
| Metric | Value |
|--------|-------|
| MAE    | X.XX  |
| RMSE   | X.XX  |
| R²     | X.XX  |

## Limitations
- Limited to product categories present in training data
- May not generalize to new product types
- Small dataset size (1,463 samples)
- Does not account for seasonal trends

## Ethical Considerations
### Bias Concerns
- Model trained on specific Amazon product categories
- May have bias towards certain price ranges

### Privacy
- No personal user data used in predictions

### Fairness
- Equal treatment across all product categories

### Transparency
- All features and logic fully documented
```

### תוצרים
- [ ] `outputs/reports/evaluation_report.md`
- [ ] `outputs/reports/model_card.md`

---

## שבוע 3: Fail Gracefully

### משימות
| # | משימה | תלוי ב- | תוצר | Status |
|---|--------|---------|------|--------|
| 1 | validate_against_constraints | contract | validators.py | ⬜ |
| 2 | UserFriendlyErrors | - | error_handler.py | ⬜ |
| 3 | _fail_gracefully method | flow (אריק) | main_flow.py | ⬜ |

### פונקציית ולידציה מול constraints
```python
# להוסיף ל-src/flow/validators.py

def validate_against_constraints(df: pd.DataFrame, contract: dict) -> Tuple[bool, str]:
    """בדיקת DataFrame מול אילוצי החוזה."""
    logger.info("🔍 Validating data against contract constraints")

    errors = []
    constraints = contract.get('constraints', {})

    if not constraints:
        logger.warning("⚠ No constraints defined in contract")
        return True, "No constraints to validate"

    for column, rules in constraints.items():
        if column not in df.columns:
            if rules.get('required', False):
                errors.append(f"עמודה חובה חסרה: {column}")
            continue

        col_data = df[column]

        # בדיקת ערכים מספריים
        if rules.get('type') == 'numeric':
            try:
                numeric_data = pd.to_numeric(
                    col_data.astype(str).str.replace('[₹,]', '', regex=True),
                    errors='coerce'
                )

                if 'min' in rules:
                    below_min = numeric_data < rules['min']
                    if below_min.any():
                        errors.append(f"{column}: {below_min.sum()} ערכים מתחת למינימום")

                if 'max' in rules:
                    above_max = numeric_data > rules['max']
                    if above_max.any():
                        errors.append(f"{column}: {above_max.sum()} ערכים מעל מקסימום")
            except Exception as e:
                errors.append(f"{column}: שגיאה בהמרה - {str(e)}")

        # בדיקת ייחודיות
        if rules.get('unique', False):
            duplicates = col_data.duplicated().sum()
            if duplicates > 0:
                errors.append(f"{column}: {duplicates} ערכים כפולים")

        # בדיקת ערכים חסרים
        if rules.get('required', False):
            null_count = col_data.isnull().sum()
            if null_count > 0:
                errors.append(f"{column}: {null_count} ערכים חסרים")

    if errors:
        return False, "; ".join(errors)

    return True, f"Validated {len(constraints)} constraints"
```

### הודעות שגיאה ידידותיות
```python
# להוסיף ל-src/utils/error_handler.py

class UserFriendlyErrors:
    """הודעות שגיאה ידידותיות בעברית."""

    MESSAGES = {
        'file_not_found': "הקובץ '{file}' לא נמצא. אנא ודא שהקובץ קיים ב-{path}",
        'invalid_data': "הנתונים לא תקינים: {reason}",
        'contract_violation': "הנתונים לא עומדים בחוזה: {violations}",
        'model_training_failed': "אימון המודל נכשל: {reason}",
        'validation_failed': "הולידציה נכשלה בשלב '{stage}': {details}",
        'crew_failed': "צוות {crew} נכשל בביצוע המשימה: {reason}",
        'missing_columns': "עמודות חסרות בנתונים: {columns}",
        'null_values': "נמצאו ערכים חסרים בעמודות: {columns}"
    }

    @classmethod
    def get(cls, error_type: str, **kwargs) -> str:
        template = cls.MESSAGES.get(error_type, "שגיאה לא ידועה")
        try:
            return template.format(**kwargs)
        except KeyError:
            return template
```

### מתודת _fail_gracefully
```python
# להוסיף ל-src/flow/main_flow.py

def _fail_gracefully(self, stage: str, error: Exception, user_message: str) -> dict:
    """טיפול בכשלון בצורה ברורה למשתמש."""
    logger.error("=" * 50)
    logger.error(f"❌ Pipeline נכשל בשלב: {stage}")
    logger.error(f"📋 הודעה: {user_message}")
    logger.error(f"🔍 פרטים: {str(error)}")
    logger.error("=" * 50)

    self.state["status"] = "failed"
    self.state["error"] = {
        "stage": stage,
        "message": user_message,
        "details": str(error),
        "timestamp": datetime.now().isoformat()
    }
    self._save_state(stage, "failed", {"error": user_message})

    return {
        "status": "failed",
        "stage": stage,
        "message": user_message
    }
```

### תוצרים
- [ ] `src/flow/validators.py` - validate_against_constraints
- [ ] `src/utils/error_handler.py` - UserFriendlyErrors
- [ ] `src/flow/main_flow.py` - _fail_gracefully (בתיאום עם אריק)

---

## שבוע 4: QA סופי

### משימות
| # | משימה | תלוי ב- | תוצר | Status |
|---|--------|---------|------|--------|
| 1 | QA Checklist | all | בדיקות מערכת | ⬜ |
| 2 | בדיקות E2E | working system | passing tests | ⬜ |
| 3 | איסוף Artifacts | all | כל הקבצים ב-Repo | ⬜ |
| 4 | דוח QA סופי | tests | QA_report.md | ⬜ |

### Checklist QA
```
□ Pipeline
  □ הרצה מההתחלה לסוף בלי שגיאות
  □ כל השלבים מתועדים ב-logs
  □ State נשמר ל-JSON

□ קבצי נתונים
  □ data/raw/amazon_sales.csv קיים
  □ data/processed/clean_data.csv נוצר
  □ data/contracts/dataset_contract.json תקין
  □ data/features/features.csv נוצר

□ תוצרי מודל
  □ outputs/models/model.pkl קיים ולא ריק
  □ outputs/reports/evaluation_report.md מלא
  □ outputs/reports/model_card.md עם כל הסקשנים

□ ממשק משתמש
  □ Streamlit רץ בלי שגיאות
  □ גרפים מוצגים
  □ Prediction עובד

□ Fail Gracefully
  □ הודעה ברורה כשחסר קובץ
  □ הודעה ברורה כשנתונים לא תקינים
  □ State נשמר גם בכשלון

□ Git
  □ README מעודכן
  □ אין קבצים רגישים
  □ requirements.txt מלא
```

### פקודות בדיקה
```bash
# בדיקת Pipeline
python -c "from src.flow.main_flow import AmazonSalesPipeline; p = AmazonSalesPipeline(); p.run()"

# בדיקת Fail Gracefully
python -c "
from src.flow.main_flow import AmazonSalesPipeline
p = AmazonSalesPipeline()
p.raw_data_path = p.project_root / 'nonexistent.csv'
result = p.run()
print('OK!' if result.get('status') == 'failed' else 'PROBLEM!')
"

# בדיקות pytest
pytest tests/ -v

# Streamlit
streamlit run app/streamlit_app.py
```

### תוצרים
- [ ] QA checklist מלא
- [ ] כל הבדיקות עוברות
- [ ] כל ה-Artifacts ב-Repo

---

## קבצים באחריותי

| קובץ | תיאור |
|------|-------|
| `data/contracts/dataset_contract.json` | חוזה נתונים |
| `outputs/reports/evaluation_report.md` | דוח הערכה |
| `outputs/reports/model_card.md` | כרטיס מודל |
| `src/flow/validators.py` | פונקציות ולידציה |
| `src/utils/error_handler.py` | טיפול בשגיאות |

---

## נקודות ממשק

### מקבל מ:
- **נווה**: model.pkl למטריקות
- **אריק**: main_flow.py לאינטגרציה

### נותן ל:
- **כולם**: dataset_contract.json
- **אריק**: error handling code
- **אחיאב**: model_card לאתיקה

---

## תלויות בחברי צוות

```
┌─────────────────┐     ┌─────────────────┐
│   נווה (4)      │     │   אריק (1)      │
│   מאמן מודל     │     │   main_flow.py  │
│   model.pkl     │     │                 │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────────────────────────────┐
│              מירב (5)                   │
│  שבוע 1: contract (לא תלוי!)           │
│  שבוע 2: eval + card (תלוי בנווה!)     │
│  שבוע 3: Fail gracefully (תלוי באריק!) │
└─────────────────────────────────────────┘
```

### המלצות לתיאום
1. **שבוע 1**: התחילי מיד! לא תלוי באף אחד
2. **שבוע 2**: תאמי עם נווה - צריכה את המודל שלו
3. **שבוע 3**: תאמי עם אריק - צריכה לשלב קוד ל-Flow
