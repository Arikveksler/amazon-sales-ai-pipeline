# Amazon Sales AI Pipeline

CrewAI Flow לניתוח וחיזוי מכירות אמזון | CrewAI Flow for Amazon Sales Analysis & Prediction

## תיאור הפרויקט | Project Description

פרויקט זה משלב שתי Crews של CrewAI לניתוח נתוני מכירות אמזון:
1. **Analyst Crew** - ניקוי נתונים, EDA, ויצירת תובנות
2. **Scientist Crew** - Feature Engineering, אימון מודל, והערכה

## ארכיטקטורה | Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Raw Data      │────▶│  Analyst Crew   │────▶│ Clean Data +    │
│ amazon_sales.csv│     │                 │     │ Contract        │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                        ┌────────────────────────────────▼────────┐
                        │              Validation                  │
                        └────────────────────────────────┬────────┘
                                                         │
┌─────────────────┐     ┌─────────────────┐     ┌───────▼─────────┐
│   Model +       │◀────│ Scientist Crew  │◀────│ Clean Data +    │
│   Reports       │     │                 │     │ Contract        │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## התקנה | Installation

```bash
# Clone the repository
git clone <repository-url>
cd amazon-sales-ai-pipeline

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Mac/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## הגדרת סביבה | Environment Setup

צור קובץ `.env` בשורש הפרויקט:

```env
OPENAI_API_KEY=your_api_key_here
LOG_LEVEL=INFO
```

## הרצה | Running

```bash
# Run the main pipeline
python src/flow/main_flow.py
```

## מבנה התיקיות | Project Structure

```
amazon-sales-ai-pipeline/
├── data/                    # נתונים
│   ├── raw/                # נתונים גולמיים
│   ├── processed/          # נתונים מעובדים
│   └── contracts/          # חוזי נתונים
├── src/                    # קוד מקור
│   ├── config/            # הגדרות
│   ├── crews/             # Crews
│   │   ├── analyst_crew/  # Data Analyst Crew
│   │   └── scientist_crew/ # Data Scientist Crew
│   ├── flow/              # Flow הראשי
│   └── utils/             # כלי עזר
├── outputs/               # תוצרים
│   ├── reports/          # דוחות
│   ├── models/           # מודלים
│   └── features/         # Features
├── tests/                # בדיקות
└── docs/                 # תיעוד
```

## תוצרים | Outputs

- `clean_data.csv` - נתונים נקיים
- `eda_report.html` - דוח EDA
- `insights.md` - תובנות
- `dataset_contract.json` - חוזה נתונים
- `features.csv` - Features
- `model.pkl` - מודל מאומן
- `evaluation_report.md` - דוח הערכה
- `model_card.md` - כרטיס מודל

## הרצת בדיקות | Running Tests

```bash
pytest tests/ -v
```

## הצוות | Team

- **Pipeline Lead**: [שם]
- **EDA Specialist**: [שם]
- **ML Specialist**: [שם]
- **UI Developer**: [שם]
- **Business & Docs**: [שם]

## רישיון | License

MIT
