# שבוע 3: ממשק ואינטגרציה

## מטרת השבוע
בניית ממשק Streamlit, אינטגרציה של כל החלקים, ומנגנוני Fail Gracefully.

---

## משימות מפורטות

| משימה | אחראי | תלוי ב- | תוצר | Deadline |
|-------|-------|---------|------|----------|
| Flow Orchestration | אריק | all crews | flow.py | יום 2 |
| Fail Gracefully | מירב | flow | error handling | יום 3 |
| Streamlit UI | אוהד | flow | streamlit_app.py | יום 4 |
| Prediction ב-UI | נווה | UI + model | prediction API | יום 4 |
| תובנות ב-UI | אחיאב | UI | UI text | יום 4 |
| Full Integration | אריק | all | working system | יום 5 |

---

## דיאגרמת תלויות

```
    ┌────────────┐
    │ אריק:      │
    │ Flow       │──────────────────────┐
    └─────┬──────┘                      │
          │                             │
          ▼                             ▼
    ┌────────────┐  ┌────────────┐  ┌────────────┐
    │ אוהד:      │  │ נווה:      │  │ מירב:      │
    │ UI         │◄─│ Prediction │  │ Error      │
    └─────┬──────┘  └────────────┘  │ Handling   │
          │                         └─────┬──────┘
          │                               │
          ▼                               ▼
    ┌────────────┐              ┌────────────────┐
    │ אחיאב:     │              │ אריק:          │
    │ UI text    │              │ Integration    │
    └────────────┘              └────────────────┘
```

---

## לוח זמנים יומי

### יום 1 (ראשון)
| שעה | אריק | אוהד | אחיאב | נווה | מירב |
|-----|------|------|-------|------|------|
| בוקר | Pull from develop | Pull | Pull | Pull | Pull |
| צהריים | תכנון Flow | תכנון UI | הכנת תובנות | תכנון API | תכנון errors |
| ערב | - | wireframes | - | - | - |

### יום 2 (שני)
| שעה | אריק | אוהד | אחיאב | נווה | מירב |
|-----|------|------|-------|------|------|
| בוקר | Flow Orchestration | Pull | - | Pull | Pull |
| צהריים | המשך Flow | התחלת UI | - | predict function | validate_constraints |
| ערב | Push flow.py | - | - | - | - |

### יום 3 (שלישי)
| שעה | אריק | אוהד | אחיאב | נווה | מירב |
|-----|------|------|-------|------|------|
| בוקר | Pull | המשך UI | Pull | Pull | Error handling |
| צהריים | Review | גרפים | - | - | _fail_gracefully |
| ערב | - | - | - | - | Push errors |

### יום 4 (רביעי)
| שעה | אריק | אוהד | אחיאב | נווה | מירב |
|-----|------|------|-------|------|------|
| בוקר | Pull | Pull | Pull | Pull | Pull |
| צהריים | Integration | סיום UI | תובנות ב-UI | Prediction ב-UI | בדיקות |
| ערב | - | Push UI | Push text | Push API | - |

### יום 5 (חמישי)
| שעה | אריק | אוהד | אחיאב | נווה | מירב |
|-----|------|------|-------|------|------|
| בוקר | Full Integration | בדיקות | בדיקות | בדיקות | בדיקות |
| צהריים | בדיקות E2E | Fix bugs | Fix text | Fix API | Fix errors |
| ערב | Merge to main | - | - | - | - |

---

## תוצרים צפויים בסוף השבוע

### קבצים חדשים/מעודכנים
- [ ] `src/flow/main_flow.py` - Flow מלא ועובד
- [ ] `app/streamlit_app.py` - UI עובד
- [ ] `src/flow/validators.py` - validate_against_constraints
- [ ] `src/utils/error_handler.py` - UserFriendlyErrors

### אינטגרציה
- [ ] Pipeline רץ מקצה לקצה
- [ ] UI מציג נתונים וגרפים
- [ ] Prediction עובד
- [ ] שגיאות מוצגות בצורה ברורה

---

## דגשים טכניים

### Flow Orchestration (אריק)
```python
# main_flow.py - הרצת כל השלבים
def run(self):
    raw_data = self._load_raw_data()
    analyst_result = self._run_analyst_crew(raw_data)
    self._validate_analyst_outputs()
    scientist_result = self._run_scientist_crew(...)
    self._validate_scientist_outputs()
    self._finalize()
```

### Fail Gracefully (מירב)
```python
# הוספה ל-main_flow.py
def _fail_gracefully(self, stage: str, error: Exception, user_message: str):
    logger.error(f"❌ Pipeline נכשל בשלב: {stage}")
    logger.error(f"📋 הודעה: {user_message}")
    self.state["status"] = "failed"
    self._save_state(stage, "failed", {"error": user_message})
    return {"status": "failed", "message": user_message}
```

### Streamlit UI (אוהד)
```python
# app/streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Amazon Sales AI Pipeline")

# הצגת נתונים
data = pd.read_csv("data/processed/clean_data.csv")
st.dataframe(data)

# גרפים
fig = px.histogram(data, x="rating")
st.plotly_chart(fig)

# Prediction
if st.button("Predict"):
    result = predict(input_data)
    st.success(f"Predicted price: {result}")
```

### Prediction API (נווה)
```python
# src/crews/scientist_crew/tools.py
def predict_price(features: dict) -> float:
    model = joblib.load('outputs/models/model.pkl')
    df = pd.DataFrame([features])
    return model.predict(df)[0]
```

---

## פקודות Git לשבוע

### יום 1 - Pull
```bash
git checkout feature/YOUR_NAME
git fetch origin
git merge origin/develop
```

### יום 5 - Merge to main
```bash
# אריק מבצע
git checkout develop
git merge feature/arik
git merge feature/ohad
git merge feature/achiav
git merge feature/nave
git merge feature/meirav
git push origin develop

# Merge to main
git checkout main
git merge develop
git push origin main
```

---

## בדיקות נדרשות

```bash
# בדיקת Pipeline
python -c "from src.flow.main_flow import AmazonSalesPipeline; p = AmazonSalesPipeline(); p.run()"

# בדיקת Streamlit
streamlit run app/streamlit_app.py

# בדיקת Fail Gracefully
python -c "
from src.flow.main_flow import AmazonSalesPipeline
p = AmazonSalesPipeline()
p.raw_data_path = p.project_root / 'nonexistent.csv'
result = p.run()
print('OK!' if result.get('status') == 'failed' else 'PROBLEM!')
"
```
