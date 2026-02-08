# תפקידי צוות הפרויקט

---

## אריק (1) - מוביל Pipeline ותשתית

### תחומי אחריות
- הקמת ה-Repository והסביבה
- ארכיטקטורת ה-Pipeline הראשית
- תזמון הסוכנים (Orchestration)
- אינטגרציה בין כל החלקים
- ניהול ה-Git Workflow

### קבצים עיקריים
- `src/flow/main_flow.py`
- `src/flow/state_manager.py`
- `requirements.txt`
- `.gitignore`

### נקודות ממשק
- **מקבל מ:** כולם (קוד לאינטגרציה)
- **נותן ל:** כולם (תשתית בסיסית)

### Branch: `feature/arik`

---

## אוהד (2) - אנליסט וויזואליזציה

### תחומי אחריות
- סוכן EDA (ניתוח חקירתי)
- הנדסת תכונות (Feature Engineering)
- בניית ממשק Streamlit
- גרפים וויזואליזציות

### קבצים עיקריים
- `src/crews/analyst_crew/agents.py` (EDA agent)
- `data/features/features.csv`
- `outputs/reports/eda_report.html`
- `app/streamlit_app.py`

### נקודות ממשק
- **מקבל מ:** נתונים גולמיים, מודל מאומן (נווה)
- **נותן ל:** features.csv לנווה, UI לכולם

### Branch: `feature/ohad`

---

## אחיאב (3) - עסקי וסטנדרטים

### תחומי אחריות
- תובנות עסקיות
- הסבר הלוגיקה מאחורי בחירות
- תיעוד README
- מצגת עסקית

### קבצים עיקריים
- `outputs/reports/insights.md`
- `README.md`
- `docs/presentation.pptx`
- הסברים בתוך ה-UI

### נקודות ממשק
- **מקבל מ:** תוצאות מכולם
- **נותן ל:** תיעוד לכולם, תובנות ל-UI

### Branch: `feature/achiav`

---

## נווה (4) - מדען נתונים (ML)

### תחומי אחריות
- בחירת מערך הנתונים
- אימון מודלים (לפחות 2)
- Hyperparameter tuning
- שמירת המודל

### קבצים עיקריים
- `src/crews/scientist_crew/agents.py`
- `outputs/models/model.pkl`
- `outputs/models/model_comparison.json`

### נקודות ממשק
- **מקבל מ:** features.csv (אוהד), contract (מירב)
- **נותן ל:** model.pkl לאוהד (UI), מטריקות למירב

### Branch: `feature/nave`

---

## מירב (5) - ולידציה ויציבות

### תחומי אחריות
- חוזה הנתונים (Dataset Contract)
- הערכת מודל ואתיקה
- מנגנוני Fail Gracefully
- בדיקות ו-QA

### קבצים עיקריים
- `data/contracts/dataset_contract.json`
- `outputs/reports/evaluation_report.md`
- `outputs/reports/model_card.md`
- `src/flow/validators.py`
- `src/utils/error_handler.py`

### נקודות ממשק
- **מקבל מ:** מודל מאומן (נווה), flow (אריק)
- **נותן ל:** contract לכולם, error handling לאריק

### Branch: `feature/meirav`

---

## דיאגרמת תלויות

```
     ┌─────────┐
     │ אריק (1)│ ◄── מוביל
     │ תשתית  │
     └────┬────┘
          │
    ┌─────┴─────┐
    ▼           ▼
┌───────┐   ┌───────┐
│אוהד(2)│   │נווה(4)│
│ EDA   │──▶│ Model │
│UI     │   │       │
└───┬───┘   └───┬───┘
    │           │
    │   ┌───────┴───────┐
    │   ▼               ▼
    │ ┌───────┐    ┌───────┐
    └▶│אחיאב(3│    │מירב(5)│
      │ תיעוד │    │ QA    │
      └───────┘    └───────┘
```
