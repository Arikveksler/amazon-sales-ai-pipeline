# תוכנית עבודה - נווה (מדען נתונים)

## תפקיד כללי
מדען נתונים (ML) - אחראי על בחירת Dataset, אימון מודלים, ו-Prediction.

## Branch: `feature/nave`

---

## שבוע 1: בחירת Dataset

### משימות
| # | משימה | תלוי ב- | תוצר | Status |
|---|--------|---------|------|--------|
| 1 | בחירת Dataset | - | amazon_sales.csv | ⬜ |
| 2 | מחקר מקדים | Dataset | data understanding | ⬜ |
| 3 | העלאה ל-Repo | Repo (אריק) | data/raw/ | ⬜ |

### Dataset נבחר
- **שם**: Amazon Sales Dataset
- **מקור**: Kaggle
- **גודל**: ~1,465 שורות
- **עמודות עיקריות**:
  - product_id
  - product_name
  - category
  - discounted_price
  - actual_price
  - discount_percentage
  - rating
  - rating_count

### בדיקות מקדימות
```python
import pandas as pd

df = pd.read_csv('data/raw/amazon_sales.csv')

# בדיקות בסיסיות
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Nulls:\n{df.isnull().sum()}")
print(f"Dtypes:\n{df.dtypes}")
```

### תוצרים
- [ ] `data/raw/amazon_sales.csv`

---

## שבוע 2: אימון מודלים

### משימות
| # | משימה | תלוי ב- | תוצר | Status |
|---|--------|---------|------|--------|
| 1 | אימון Linear Regression | features (אוהד) | model_lr.pkl | ⬜ |
| 2 | אימון Random Forest | features (אוהד) | model_rf.pkl | ⬜ |
| 3 | השוואת מודלים | models | comparison.json | ⬜ |
| 4 | שמירת המודל הטוב | comparison | model.pkl | ⬜ |

### קוד אימון
```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import pandas as pd
import numpy as np
import json

# טעינת נתונים
df = pd.read_csv('data/features/features.csv')

# הפרדת X ו-y
X = df.drop(['discounted_price', 'product_id', 'product_name'], axis=1, errors='ignore')
y = df['discounted_price']

# פיצול
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# אימון מודל 1: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# אימון מודל 2: Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# חישוב מטריקות
def calc_metrics(y_true, y_pred):
    return {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred)
    }

lr_metrics = calc_metrics(y_test, lr_pred)
rf_metrics = calc_metrics(y_test, rf_pred)

# השוואה
comparison = {
    'linear_regression': lr_metrics,
    'random_forest': rf_metrics,
    'winner': 'random_forest' if rf_metrics['r2'] > lr_metrics['r2'] else 'linear_regression'
}

# שמירה
with open('outputs/models/model_comparison.json', 'w') as f:
    json.dump(comparison, f, indent=2)

# שמירת המודל הטוב
best_model = rf_model if comparison['winner'] == 'random_forest' else lr_model
joblib.dump(best_model, 'outputs/models/model.pkl')
```

### תוצרים
- [ ] `outputs/models/model.pkl`
- [ ] `outputs/models/model_comparison.json`

---

## שבוע 3: Prediction API

### משימות
| # | משימה | תלוי ב- | תוצר | Status |
|---|--------|---------|------|--------|
| 1 | Prediction API | model.pkl | predict function | ⬜ |
| 2 | אינטגרציה ל-UI | UI (אוהד) | prediction in app | ⬜ |

### פונקציית Prediction
```python
# src/crews/scientist_crew/tools.py
import joblib
import pandas as pd
from pathlib import Path

def predict_price(features: dict) -> float:
    """
    חיזוי מחיר לאחר הנחה.

    Args:
        features: dict עם הפיצ'רים הנדרשים
            - actual_price: מחיר מקורי
            - rating: דירוג
            - category: קטגוריה
            - ...

    Returns:
        float: מחיר חזוי
    """
    model_path = Path(__file__).parent.parent.parent.parent / 'outputs' / 'models' / 'model.pkl'
    model = joblib.load(model_path)

    # המרה ל-DataFrame
    df = pd.DataFrame([features])

    # וידוא שכל הפיצ'רים קיימים
    # (להוסיף encoding אם צריך)

    prediction = model.predict(df)[0]
    return float(prediction)


def get_model_info() -> dict:
    """מידע על המודל לתצוגה ב-UI."""
    import json
    comparison_path = Path(__file__).parent.parent.parent.parent / 'outputs' / 'models' / 'model_comparison.json'

    with open(comparison_path, 'r') as f:
        comparison = json.load(f)

    return {
        'model_type': comparison['winner'],
        'metrics': comparison[comparison['winner']]
    }
```

### אינטגרציה ל-Streamlit
```python
# לתת לאוהד להוסיף ל-streamlit_app.py
from src.crews.scientist_crew.tools import predict_price, get_model_info

# בתוך ה-tab של Prediction
with tab_predict:
    st.subheader("🔮 Price Prediction")

    col1, col2 = st.columns(2)
    with col1:
        actual_price = st.number_input("Actual Price (₹)", min_value=0, value=1000)
        rating = st.slider("Rating", 1.0, 5.0, 4.0, 0.1)
    with col2:
        category = st.selectbox("Category", ["Electronics", "Fashion", "Home"])

    if st.button("Predict Price"):
        features = {
            'actual_price': actual_price,
            'rating': rating,
            'category': category
        }
        result = predict_price(features)
        st.success(f"💰 Predicted Discounted Price: ₹{result:,.2f}")
```

### תוצרים
- [ ] פונקציית predict_price עובדת
- [ ] אינטגרציה ל-UI

---

## שבוע 4: תיעוד טכני

### משימות
| # | משימה | תלוי ב- | תוצר | Status |
|---|--------|---------|------|--------|
| 1 | תיעוד טכני | model card (מירב) | README tech section | ⬜ |
| 2 | Review model card | מירב | approved card | ⬜ |

### סקשן טכני ל-README
```markdown
## Technical Details

### Model Architecture
- **Type**: Random Forest Regressor
- **Features**: 10 engineered features
- **Target**: discounted_price

### Training
- **Dataset Size**: 1,463 samples
- **Train/Test Split**: 80/20
- **Cross-Validation**: 5-fold

### Performance
| Metric | Value |
|--------|-------|
| MAE    | X.XX  |
| RMSE   | X.XX  |
| R²     | X.XX  |

### Usage
```python
from src.crews.scientist_crew.tools import predict_price

result = predict_price({
    'actual_price': 1000,
    'rating': 4.5,
    'category': 'Electronics'
})
print(f"Predicted price: ₹{result}")
```
```

### תוצרים
- [ ] סקשן טכני ב-README
- [ ] Model card reviewed

---

## קבצים באחריותי

| קובץ | תיאור |
|------|-------|
| `data/raw/amazon_sales.csv` | Dataset מקורי |
| `src/crews/scientist_crew/agents.py` | Scientist agent |
| `src/crews/scientist_crew/tools.py` | Prediction tools |
| `outputs/models/model.pkl` | מודל מאומן |
| `outputs/models/model_comparison.json` | השוואת מודלים |

---

## נקודות ממשק

### מקבל מ:
- **אוהד**: features.csv
- **מירב**: Dataset contract

### נותן ל:
- **אוהד**: model.pkl ל-UI
- **מירב**: מטריקות להערכה
