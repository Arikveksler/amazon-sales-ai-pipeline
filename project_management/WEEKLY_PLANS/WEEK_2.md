# שבוע 2: צוות מדעני נתונים

## מטרת השבוע
הנדסת פיצ'רים, אימון מודלים, הערכה ותיעוד.

---

## משימות מפורטות

| משימה | אחראי | תלוי ב- | תוצר | Deadline |
|-------|-------|---------|------|----------|
| Feature Engineering | אוהד | clean_data | features.csv | יום 2 |
| Handoff mechanism | אריק | features | handoff code | יום 3 |
| אימון מודלים (2+) | נווה | features | model.pkl | יום 4 |
| Evaluation Report | מירב | model | evaluation_report.md | יום 5 |
| Model Card | מירב | model | model_card.md | יום 5 |
| מדדי הצלחה + אתיקה | אחיאב | eval report | README section | יום 5 |

---

## דיאגרמת תלויות

```
    ┌────────────┐
    │ אוהד:      │
    │ features   │──────┐
    └────────────┘      │
                        ▼
    ┌────────────┐  ┌────────────┐
    │ מירב:      │◄─│ נווה:      │
    │ eval+card  │  │ model.pkl  │
    └────────────┘  └────────────┘
           │               │
           ▼               ▼
    ┌────────────────────────┐
    │ אריק: Handoff          │
    └────────────────────────┘
           │
           ▼
    ┌────────────┐
    │ אחיאב:     │
    │ Ethics     │
    └────────────┘
```

---

## לוח זמנים יומי

### יום 1 (ראשון)
| שעה | אריק | אוהד | אחיאב | נווה | מירב |
|-----|------|------|-------|------|------|
| בוקר | Pull from develop | Pull | Pull | Pull | Pull |
| צהריים | תכנון handoff | Feature Engineering | - | הכנה לאימון | הכנת templates |
| ערב | - | המשך עבודה | - | - | - |

### יום 2 (שני)
| שעה | אריק | אוהד | אחיאב | נווה | מירב |
|-----|------|------|-------|------|------|
| בוקר | Pull | סיום features | - | Pull | Pull |
| צהריים | - | Push features.csv | - | Pull features | - |
| ערב | - | - | - | התחלת אימון | - |

### יום 3 (שלישי)
| שעה | אריק | אוהד | אחיאב | נווה | מירב |
|-----|------|------|-------|------|------|
| בוקר | Pull features | - | - | אימון מודל 1 | Pull |
| צהריים | Handoff mechanism | תמיכה | - | אימון מודל 2 | model_card template |
| ערב | Push handoff | - | - | השוואה | - |

### יום 4 (רביעי)
| שעה | אריק | אוהד | אחיאב | נווה | מירב |
|-----|------|------|-------|------|------|
| בוקר | Pull | Pull | Pull | סיום אימון | Pull |
| צהריים | Review code | - | הכנה לאתיקה | Push model.pkl | Pull model |
| ערב | - | - | - | - | התחלת הערכה |

### יום 5 (חמישי)
| שעה | אריק | אוהד | אחיאב | נווה | מירב |
|-----|------|------|-------|------|------|
| בוקר | Pull | Pull | Pull model | - | סיום דוחות |
| צהריים | Merge to develop | - | כתיבת אתיקה | Review | Push reports |
| ערב | - | - | Push README | - | - |

---

## תוצרים צפויים בסוף השבוע

### קבצים חדשים
- [ ] `data/features/features.csv`
- [ ] `outputs/models/model.pkl`
- [ ] `outputs/models/model_comparison.json`
- [ ] `outputs/reports/evaluation_report.md`
- [ ] `outputs/reports/model_card.md`

### עדכונים
- [ ] `README.md` - סקשן מדדי הצלחה ואתיקה

---

## דגשים טכניים

### Feature Engineering (אוהד)
```python
# דוגמה לפיצ'רים נדרשים
- המרת קטגוריות (OneHotEncoding)
- ניקוי מחירים (הסרת סימני מטבע ₹)
- נרמול מספרים
- יצירת פיצ'רים חדשים (discount_percentage, price_ratio)
```

### אימון מודלים (נווה)
```python
# מודלים נדרשים (לפחות 2)
1. Linear Regression
2. Random Forest

# מטריקות להשוואה
- MAE, RMSE, R², MAPE
```

### Model Card (מירב)
```markdown
# סקשנים חובה (לפי validators.py)
- Purpose
- Data
- Metrics
- Limitations
- Ethical Considerations
```

---

## פקודות Git לשבוע

### יום 1 - Pull
```bash
git checkout feature/YOUR_NAME
git fetch origin
git merge origin/develop
```

### יום 3 - אוהד Push features
```bash
git add data/features/features.csv
git commit -m "Add engineered features"
git push origin feature/ohad
```

### יום 4 - נווה Push model
```bash
git add outputs/models/
git commit -m "Add trained models and comparison"
git push origin feature/nave
```

### יום 5 - Merge
```bash
git add .
git commit -m "Week 2 deliverables"
git push origin feature/YOUR_NAME
# צור Pull Request ל-develop
```
