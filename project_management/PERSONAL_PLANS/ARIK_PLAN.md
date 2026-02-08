# תוכנית עבודה - אריק (מוביל Pipeline)

## תפקיד כללי
מוביל Pipeline ותשתית - אחראי על הארכיטקטורה, האינטגרציה, וניהול ה-Git.

## Branch: `feature/arik`

---

## שבוע 1: תשתית

### משימות
| # | משימה | תלוי ב- | תוצר | Status |
|---|--------|---------|------|--------|
| 1 | הקמת Repo | - | GitHub repo | ⬜ |
| 2 | יצירת branches לכולם | Repo | 5 branches | ⬜ |
| 3 | הגדרת requirements.txt | - | requirements.txt | ⬜ |
| 4 | סוכן ניקוי נתונים | Dataset (נווה) | clean_data.csv | ⬜ |

### פקודות Git
```bash
# יום 1: יצירת repo ו-branches
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

### תוצרים
- [ ] GitHub repo פעיל
- [ ] 6 branches (main, develop, 5 feature branches)
- [ ] requirements.txt
- [ ] .gitignore
- [ ] data/processed/clean_data.csv

---

## שבוע 2: Handoff

### משימות
| # | משימה | תלוי ב- | תוצר | Status |
|---|--------|---------|------|--------|
| 1 | Handoff mechanism | features.csv (אוהד) | handoff code | ⬜ |
| 2 | Git workflow docs | - | GIT_WORKFLOW.md | ⬜ |
| 3 | תמיכה בצוות | - | - | ⬜ |

### דגשים טכניים
```python
# מנגנון Handoff בין Crews
class HandoffManager:
    def __init__(self):
        self.state = {}

    def save_analyst_output(self, clean_data_path, contract_path):
        self.state['analyst'] = {
            'clean_data': clean_data_path,
            'contract': contract_path,
            'completed_at': datetime.now().isoformat()
        }

    def get_for_scientist(self):
        return self.state.get('analyst', {})
```

### תוצרים
- [ ] Handoff mechanism עובד
- [ ] תיעוד Git workflow

---

## שבוע 3: אינטגרציה

### משימות
| # | משימה | תלוי ב- | תוצר | Status |
|---|--------|---------|------|--------|
| 1 | Flow Orchestration | all crews | flow.py complete | ⬜ |
| 2 | Integration of error handling | מירב | integrated flow | ⬜ |
| 3 | Full system test | all | working pipeline | ⬜ |

### קוד Flow
```python
# src/flow/main_flow.py
def run(self):
    try:
        raw_data = self._load_raw_data()
        analyst_result = self._run_analyst_crew(raw_data)
        self._validate_analyst_outputs()
        scientist_result = self._run_scientist_crew(analyst_result["clean_data_path"])
        self._validate_scientist_outputs()
        self._finalize()
    except Exception as e:
        return self._fail_gracefully(...)
```

### תוצרים
- [ ] Pipeline רץ מקצה לקצה
- [ ] Error handling משולב
- [ ] First merge to main

---

## שבוע 4: סיום

### משימות
| # | משימה | תלוי ב- | תוצר | Status |
|---|--------|---------|------|--------|
| 1 | Testing + Bug fixes | integration | passing tests | ⬜ |
| 2 | Code Review | - | clean code | ⬜ |
| 3 | Final Merge to Main | all | release | ⬜ |

### Final Checklist
- [ ] All tests pass
- [ ] All PRs merged
- [ ] main branch is clean
- [ ] No merge conflicts
- [ ] Tag v1.0.0 created

### פקודות Final Merge
```bash
git checkout develop
git pull origin develop
git checkout main
git merge develop
git tag -a v1.0.0 -m "Final Release"
git push origin main --tags
```

---

## קבצים באחריותי

| קובץ | תיאור |
|------|-------|
| `src/flow/main_flow.py` | Flow ראשי |
| `src/flow/state_manager.py` | ניהול State |
| `requirements.txt` | תלויות |
| `.gitignore` | קבצים להתעלם |

---

## נקודות ממשק

### מקבל מ:
- **כולם**: קוד לאינטגרציה
- **מירב**: Error handling code

### נותן ל:
- **כולם**: תשתית בסיסית
- **כולם**: Git branches
