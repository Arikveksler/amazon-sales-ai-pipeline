# שבוע 1: תשתית וצוות אנליסטים

## מטרת השבוע
הקמת תשתית הפרויקט, בחירת Dataset, וביצוע ניתוח ראשוני.

---

## משימות מפורטות

| משימה | אחראי | תלוי ב- | תוצר | Deadline |
|-------|-------|---------|------|----------|
| הקמת Repo + branches | אריק | - | GitHub repo | יום 1 |
| סביבת Python + requirements | אריק | Repo | requirements.txt | יום 2 |
| בחירת Dataset | נווה | Repo | amazon_sales.csv | יום 2 |
| Dataset Contract | מירב | - | dataset_contract.json | יום 3 |
| סוכן ניקוי נתונים | אריק | Dataset | clean_data.csv | יום 4 |
| סוכן EDA | אוהד | clean_data | eda_report.html | יום 5 |
| תובנות עסקיות | אחיאב | EDA | insights.md | יום 5 |

---

## דיאגרמת תלויות

```
                    ┌──────────────┐
                    │ אריק: Repo   │
                    └──────┬───────┘
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
    ┌────────────┐  ┌────────────┐  ┌────────────┐
    │ אוהד: EDA  │  │ מירב:      │  │ נווה:      │
    │            │  │ Contract   │  │ Dataset    │
    └────────────┘  └────────────┘  └────────────┘
           │               │               │
           └───────────────┼───────────────┘
                           ▼
                    ┌──────────────┐
                    │ אחיאב:       │
                    │ Insights     │
                    └──────────────┘
```

---

## לוח זמנים יומי

### יום 1 (ראשון)
| שעה | אריק | אוהד | אחיאב | נווה | מירב |
|-----|------|------|-------|------|------|
| בוקר | הקמת Repo | - | - | חיפוש Datasets | - |
| צהריים | יצירת branches | - | - | בחירה מקדימה | - |
| ערב | Push structure | - | - | העלאת Dataset | - |

### יום 2 (שני)
| שעה | אריק | אוהד | אחיאב | נווה | מירב |
|-----|------|------|-------|------|------|
| בוקר | requirements.txt | Pull | Pull | Pull | Pull |
| צהריים | .gitignore | הכנה ל-EDA | קריאת נתונים | מחקר מקדים | הכנת template |
| ערב | Push | - | - | Push dataset | - |

### יום 3 (שלישי)
| שעה | אריק | אוהד | אחיאב | נווה | מירב |
|-----|------|------|-------|------|------|
| בוקר | Pull | Pull | Pull | Pull | כתיבת contract |
| צהריים | סוכן ניקוי | תכנון EDA | תכנון תובנות | עזרה לאריק | constraints |
| ערב | - | - | - | - | Push contract |

### יום 4 (רביעי)
| שעה | אריק | אוהד | אחיאב | נווה | מירב |
|-----|------|------|-------|------|------|
| בוקר | סוכן ניקוי | Pull | Pull | Pull | Pull |
| צהריים | בדיקות | התחלת EDA | - | בדיקת נתונים | בדיקת contract |
| ערב | Push clean_data | - | - | - | - |

### יום 5 (חמישי)
| שעה | אריק | אוהד | אחיאב | נווה | מירב |
|-----|------|------|-------|------|------|
| בוקר | Pull | סיום EDA | Pull | - | - |
| צהריים | Code review | Push eda_report | תובנות עסקיות | Review | Review |
| ערב | Merge to develop | - | Push insights | - | - |

---

## תוצרים צפויים בסוף השבוע

### קבצים חדשים
- [ ] `data/raw/amazon_sales.csv`
- [ ] `data/processed/clean_data.csv`
- [ ] `data/contracts/dataset_contract.json`
- [ ] `outputs/reports/eda_report.html`
- [ ] `outputs/reports/insights.md`
- [ ] `requirements.txt`
- [ ] `.gitignore`

### Git
- [ ] Repo מוקם
- [ ] Branch `develop` נוצר
- [ ] 5 branches אישיים נוצרו
- [ ] כל התוצרים ממוזגים ל-develop

---

## פקודות Git לשבוע

### אריק (יום 1)
```bash
# יצירת repo ו-branches
git init
git remote add origin <URL>
git checkout -b develop
git push -u origin develop

# יצירת branches לכולם
for name in arik ohad achiav nave meirav; do
  git checkout -b feature/$name
  git push -u origin feature/$name
  git checkout develop
done
```

### כולם (יום 2)
```bash
git fetch origin
git checkout feature/YOUR_NAME
git merge origin/develop
```

### כולם (יום 5)
```bash
git add .
git commit -m "Week 1 deliverables"
git push origin feature/YOUR_NAME
# צור Pull Request ל-develop
```
