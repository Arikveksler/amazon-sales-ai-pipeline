# Git Workflow - הנחיות עבודה

## מבנה ה-Branches

```
main (protected - לא לעבוד כאן!)
  │
  └── develop (integration branch)
        │
        ├── feature/arik      ← אריק עובד כאן
        ├── feature/ohad      ← אוהד עובד כאן
        ├── feature/achiav    ← אחיאב עובד כאן
        ├── feature/nave      ← נווה עובד כאן
        └── feature/meirav    ← מירב עובדת כאן
```

---

## הקמת ה-Branch שלך (פעם אחת)

```bash
# עדכון מ-remote
git fetch origin

# יצירת branch חדש מ-develop
git checkout develop
git pull origin develop
git checkout -b feature/YOUR_NAME
git push -u origin feature/YOUR_NAME
```

---

## עבודה יומית

### תחילת יום עבודה - קבלת עדכונים

```bash
# עדכון ה-branch שלך מ-develop
git checkout feature/YOUR_NAME
git fetch origin
git merge origin/develop

# אם יש קונפליקטים - פתור אותם ואז:
git add .
git commit -m "Merge develop into feature/YOUR_NAME"
```

### סיום יום עבודה - שליחת עדכונים

```bash
# שמירת השינויים
git add .
git commit -m "תיאור השינויים"
git push origin feature/YOUR_NAME
```

---

## לוח זמני Push/Pull

| יום | פעולה | מי עושה | מאיפה → לאיפה |
|-----|-------|---------|---------------|
| **שבוע 1** | | | |
| יום 3 | Pull | כולם | develop → branches |
| יום 5 | Push | כולם | branches → develop |
| **שבוע 2** | | | |
| יום 1 | Pull | כולם | develop → branches |
| יום 3 | Push | אוהד | features.csv → develop |
| יום 4 | Pull | נווה | develop → feature/nave |
| יום 5 | Push | כולם | branches → develop |
| **שבוע 3** | | | |
| יום 1 | Pull | כולם | develop → branches |
| יום 3 | Push | כולם | branches → develop |
| יום 5 | Merge | אריק | develop → main |
| **שבוע 4** | | | |
| יום 1 | Pull | כולם | main → branches |
| יום 3 | Final PR | אריק | develop → main |

---

## מיזוג ל-develop (Pull Request)

כשסיימת משימה גדולה:

```bash
# ודא שה-branch שלך מעודכן
git checkout feature/YOUR_NAME
git fetch origin
git merge origin/develop
git push origin feature/YOUR_NAME

# צור Pull Request ב-GitHub
# Title: [YOUR_NAME] תיאור קצר
# Reviewer: אריק
```

---

## כללי זהב

1. **לעולם אל תעבוד על main!**
2. **Pull לפני Push** - תמיד עדכן לפני שאתה שולח
3. **Commit קטנים** - עדיף הרבה commits קטנים מאשר אחד גדול
4. **הודעות ברורות** - כתוב מה עשית, לא "fix"
5. **פתור קונפליקטים מיד** - אל תדחה לאחר כך

---

## פתרון בעיות נפוצות

### קונפליקט במיזוג

```bash
# אחרי git merge origin/develop אם יש קונפליקט:
# 1. פתח את הקבצים עם הקונפליקט
# 2. מחק את הסימנים <<<<<<< ======= >>>>>>>
# 3. השאר את הקוד הנכון
# 4. שמור ואז:
git add .
git commit -m "Resolve merge conflict"
git push
```

### ביטול שינויים

```bash
# ביטול שינויים בקובץ (לפני commit)
git checkout -- filename.py

# ביטול commit אחרון (בלי למחוק שינויים)
git reset --soft HEAD~1
```

### עדכון branch מ-main

```bash
# אם צריך לעדכן מ-main (נדיר)
git fetch origin
git merge origin/main
```

---

## פקודות שימושיות

```bash
# צפייה בסטטוס
git status

# צפייה ב-branches
git branch -a

# צפייה בהיסטוריה
git log --oneline -10

# צפייה בשינויים
git diff

# החלפת branch
git checkout feature/NAME
```
