# תוכנית עבודה - אוהד (אנליסט וויזואליזציה)

## תפקיד כללי
אנליסט וויזואליזציה - אחראי על EDA, Feature Engineering, וממשק Streamlit.

## Branch: `feature/ohad`

---

## שבוע 1: EDA

### משימות
| # | משימה | תלוי ב- | תוצר | Status |
|---|--------|---------|------|--------|
| 1 | סוכן EDA | clean_data (אריק) | eda_report.html | ⬜ |
| 2 | גרפים בסיסיים | EDA | visualizations | ⬜ |

### קוד EDA Agent
```python
# src/crews/analyst_crew/agents.py
from crewai import Agent

def create_eda_agent():
    return Agent(
        role="EDA Analyst",
        goal="Perform exploratory data analysis on Amazon sales data",
        backstory="Expert data analyst specializing in retail patterns"
    )
```

### גרפים נדרשים
- [ ] Distribution של מחירים
- [ ] Correlation heatmap
- [ ] Category breakdown
- [ ] Rating distribution

### תוצרים
- [ ] `outputs/reports/eda_report.html`

---

## שבוע 2: Feature Engineering

### משימות
| # | משימה | תלוי ב- | תוצר | Status |
|---|--------|---------|------|--------|
| 1 | Feature Engineering | clean_data | features.csv | ⬜ |
| 2 | תיעוד הפיצ'רים | features | features_doc.md | ⬜ |

### פיצ'רים ליצור
```python
# הוספת פיצ'רים חדשים
df['price_ratio'] = df['discounted_price'] / df['actual_price']
df['discount_amount'] = df['actual_price'] - df['discounted_price']
df['is_high_rated'] = (df['rating'] >= 4.0).astype(int)

# המרת קטגוריות
df = pd.get_dummies(df, columns=['category'])

# ניקוי מחירים (הסרת ₹)
df['discounted_price'] = df['discounted_price'].str.replace('[₹,]', '', regex=True).astype(float)
```

### תוצרים
- [ ] `data/features/features.csv`
- [ ] תיעוד הפיצ'רים

---

## שבוע 3: Streamlit UI

### משימות
| # | משימה | תלוי ב- | תוצר | Status |
|---|--------|---------|------|--------|
| 1 | Streamlit UI | flow (אריק) | streamlit_app.py | ⬜ |
| 2 | הצגת גרפים | UI | charts in app | ⬜ |
| 3 | שילוב prediction | model (נווה) | prediction UI | ⬜ |

### מבנה ה-UI
```python
# app/streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.set_page_config(page_title="Amazon Sales AI", layout="wide")

st.title("🛒 Amazon Sales AI Pipeline")

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Data", "📈 Analysis", "🔮 Predict"])

with tab1:
    data = pd.read_csv("data/processed/clean_data.csv")
    st.dataframe(data)

with tab2:
    fig = px.histogram(data, x="rating", title="Rating Distribution")
    st.plotly_chart(fig)

with tab3:
    st.subheader("Price Prediction")
    # Input fields
    price = st.number_input("Actual Price", min_value=0)
    category = st.selectbox("Category", options=categories)

    if st.button("Predict"):
        model = joblib.load("outputs/models/model.pkl")
        # prediction logic
        st.success(f"Predicted Price: ₹{result:.2f}")
```

### תוצרים
- [ ] `app/streamlit_app.py` עובד
- [ ] גרפים אינטראקטיביים
- [ ] Prediction form

---

## שבוע 4: סרטון דמו

### משימות
| # | משימה | תלוי ב- | תוצר | Status |
|---|--------|---------|------|--------|
| 1 | עיצוב סופי UI | working system | polished UI | ⬜ |
| 2 | סרטון דמו (5 דק') | UI complete | video file | ⬜ |

### מבנה הסרטון (5 דקות)
| זמן | תוכן |
|-----|------|
| 0:00-0:30 | פתיחה - הצגת הפרויקט |
| 0:30-1:30 | הרצת Pipeline בטרמינל |
| 1:30-3:30 | סיור ב-Streamlit UI |
| 3:30-4:30 | הדגמת Prediction |
| 4:30-5:00 | סיכום ותודות |

### Tips להקלטה
- השתמש ב-OBS או Loom
- הכן script מראש
- בדוק שהקול ברור
- הראה את כל הפיצ'רים

### תוצרים
- [ ] UI מעוצב סופי
- [ ] סרטון דמו (עד 5 דקות)

---

## קבצים באחריותי

| קובץ | תיאור |
|------|-------|
| `src/crews/analyst_crew/agents.py` | EDA agent |
| `data/features/features.csv` | פיצ'רים |
| `outputs/reports/eda_report.html` | דוח EDA |
| `app/streamlit_app.py` | ממשק משתמש |

---

## נקודות ממשק

### מקבל מ:
- **אריק**: clean_data.csv
- **נווה**: model.pkl לשילוב ב-UI

### נותן ל:
- **נווה**: features.csv לאימון
- **כולם**: UI להדגמה
