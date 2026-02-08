# שבוע 4: פינישים והגשה

## מטרת השבוע
סיום כל המשימות, בדיקות סופיות, והכנת חומרי ההגשה.

---

## משימות מפורטות

| משימה | אחראי | תלוי ב- | תוצר | Deadline |
|-------|-------|---------|------|----------|
| Testing + Bug fixes | אריק | integration | passing tests | יום 2 |
| Code Review | אריק | - | clean code | יום 3 |
| סרטון דמו (5 דק') | אוהד | working system | video file | יום 4 |
| מצגת עסקית (10-12) | אחיאב | all | presentation.pptx | יום 4 |
| תיעוד טכני | נווה | model card | README final | יום 4 |
| בדיקות סופיות | מירב | all | QA report | יום 4 |
| איסוף Artifacts | מירב | all | all files in repo | יום 5 |
| Final Merge to Main | אריק | all | release | יום 5 |

---

## דיאגרמת תלויות

```
    ┌─────────────────────────────────────────┐
    │           אריק: Final Integration        │
    │           Testing + Merge to Main        │
    └──────────────────┬──────────────────────┘
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
┌────────────┐  ┌────────────┐  ┌────────────┐
│ אוהד:      │  │ אחיאב:     │  │ מירב:      │
│ Demo Video │  │ מצגת       │  │ QA Final   │
└────────────┘  └────────────┘  └────────────┘
       │               │               │
       └───────────────┼───────────────┘
                       ▼
              ┌────────────────┐
              │ נווה:          │
              │ Technical Docs │
              └────────────────┘
```

---

## לוח זמנים יומי

### יום 1 (ראשון)
| שעה | אריק | אוהד | אחיאב | נווה | מירב |
|-----|------|------|-------|------|------|
| בוקר | Pull from develop | Pull | Pull | Pull | Pull |
| צהריים | Run all tests | בדיקת UI | איסוף תובנות | בדיקת model | בדיקת validators |
| ערב | Fix critical bugs | Fix UI bugs | רשימת נקודות למצגת | Model card review | QA checklist |

### יום 2 (שני)
| שעה | אריק | אוהד | אחיאב | נווה | מירב |
|-----|------|------|-------|------|------|
| בוקר | Code cleanup | UI polish | התחלת מצגת | README טכני | בדיקות E2E |
| צהריים | Integration tests | גרפים סופיים | מצגת המשך | תיעוד מודל | Fix bugs found |
| ערב | Push fixes | Push UI | שמירת טיוטה | Push docs | Push fixes |

### יום 3 (שלישי)
| שעה | אריק | אוהד | אחיאב | נווה | מירב |
|-----|------|------|-------|------|------|
| בוקר | Code review | תכנון סרטון | סיום מצגת | Review מירב | Review final |
| צהריים | Final fixes | הקלטת דמו | עיצוב מצגת | עזרה לאוהד | Artifacts check |
| ערב | Merge to develop | עריכת סרטון | PDF export | עזרה לאחיאב | Final checklist |

### יום 4 (רביעי)
| שעה | אריק | אוהד | אחיאב | נווה | מירב |
|-----|------|------|-------|------|------|
| בוקר | Final testing | גימור סרטון | גימור מצגת | README final | QA report |
| צהריים | PR to main | Upload video | Upload pptx | Final review | Final review |
| ערב | Merge to main | - | - | - | - |

### יום 5 (חמישי) - הגשה
| שעה | כולם |
|-----|------|
| בוקר | בדיקה אחרונה שהכל ב-Repo |
| צהריים | **הגשה!** |

---

## Checklist הגשה

### תוצרים טכניים (חובה)
- [ ] `data/raw/amazon_sales.csv` - נתונים גולמיים
- [ ] `data/processed/clean_data.csv` - נתונים מנוקים
- [ ] `data/contracts/dataset_contract.json` - חוזה נתונים
- [ ] `data/features/features.csv` - פיצ'רים מהונדסים
- [ ] `outputs/models/model.pkl` - מודל מאומן
- [ ] `outputs/reports/eda_report.html` - דוח EDA
- [ ] `outputs/reports/evaluation_report.md` - דוח הערכה
- [ ] `outputs/reports/model_card.md` - כרטיס מודל
- [ ] `outputs/reports/insights.md` - תובנות עסקיות
- [ ] `app/streamlit_app.py` - ממשק משתמש עובד

### תיעוד (חובה)
- [ ] `README.md` - תיעוד מלא ומעודכן
- [ ] `requirements.txt` - כל התלויות

### חומרי הגשה
- [ ] מצגת עסקית (10-12 שקפים)
- [ ] סרטון דמו (עד 5 דקות)

---

## בדיקות QA סופיות (מירב)

### Checklist בדיקות
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
  □ מערכת מחזירה הודעה ברורה כשחסר קובץ
  □ מערכת מחזירה הודעה ברורה כשנתונים לא תקינים
  □ State נשמר גם בכשלון

□ Git ותיעוד
  □ README מעודכן
  □ אין קבצים רגישים (.env, credentials)
  □ requirements.txt מלא
  □ כל ה-branches ממוזגים ל-develop
```

---

## פקודות בדיקה סופיות

```bash
# התקנה נקייה
pip install -r requirements.txt

# הרצת Pipeline
python -c "
from src.flow.main_flow import AmazonSalesPipeline
p = AmazonSalesPipeline()
result = p.run()
print('Status:', result.get('status') if isinstance(result, dict) else 'completed')
"

# הרצת Streamlit
streamlit run app/streamlit_app.py

# הרצת בדיקות
pytest tests/ -v

# בדיקת Fail Gracefully
python -c "
from src.flow.main_flow import AmazonSalesPipeline
p = AmazonSalesPipeline()
p.raw_data_path = p.project_root / 'nonexistent.csv'
result = p.run()
print('Fail gracefully works!' if result.get('status') == 'failed' else 'PROBLEM!')
"
```

---

## מבנה המצגת (אחיאב)

| # | שקף | תוכן |
|---|-----|------|
| 1 | פתיחה | שם הפרויקט + צוות |
| 2 | הבעיה | הבעיה העסקית שפותרים |
| 3 | הפתרון | סקירת הפתרון שלנו |
| 4 | ארכיטקטורה | דיאגרמת Pipeline |
| 5 | נתונים | מערך הנתונים והפיצ'רים |
| 6 | עיבוד | שלבי ה-Pipeline |
| 7 | מודל | המודל שנבחר ולמה |
| 8 | תוצאות | ביצועים ומטריקות |
| 9 | דמו | Screenshot של ה-UI |
| 10 | מגבלות | מה לא עובד / שיפורים עתידיים |
| 11 | סיכום | נקודות מפתח |
| 12 | שאלות | Q&A |

---

## Tips לסרטון דמו (אוהד)

1. **פתיחה (30 שניות)**
   - הצגת הפרויקט
   - מה המטרה

2. **Pipeline (1 דקה)**
   - הרצת ה-Pipeline בטרמינל
   - הצגת הלוגים

3. **UI (2 דקות)**
   - ניווט ב-Streamlit
   - הצגת גרפים
   - הדגמת Prediction

4. **תוצרים (1 דקה)**
   - הצגת הקבצים שנוצרו
   - סקירת הדוחות

5. **סיום (30 שניות)**
   - סיכום
   - תודות

---

## Git - Final Merge

```bash
# אריק מבצע (יום 4 ערב)
git checkout develop
git pull origin develop

# וידוא שכל ה-PRs ממוזגים
git merge feature/arik
git merge feature/ohad
git merge feature/achiav
git merge feature/nave
git merge feature/meirav

# פתרון קונפליקטים אם יש
git push origin develop

# Merge to main
git checkout main
git merge develop
git tag -a v1.0.0 -m "Final Release"
git push origin main --tags
```
