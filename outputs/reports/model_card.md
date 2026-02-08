# Model Card: Amazon Sales Price Predictor

---

## Purpose

### Model Details
- **Name**: Amazon Sales Price Predictor
- **Version**: 1.0
- **Type**: Regression (Price Prediction)
- **Framework**: Scikit-learn
- **Created by**: Data Scientist Crew
- **Date**: [יתעדכן לאחר אימון]

### Intended Use
- **Primary use**: חיזוי מחיר לאחר הנחה למוצרי Amazon
- **Target users**: צוותי תמחור, אנליסטים עסקיים, מנהלי קטגוריות
- **Use cases**:
  - קביעת מחירי הנחה אופטימליים
  - ניתוח תחרותיות מחירים
  - תכנון מבצעים

### Out-of-Scope Uses
- **לא מיועד ל:**
  - החלטות תמחור אוטומטיות בזמן אמת ללא פיקוח אנושי
  - קטגוריות מוצרים שלא נכללו בנתוני האימון
  - שווקים מחוץ להודו (הנתונים מבוססים על Amazon India)

---

## Data

### Training Data
- **Source**: Kaggle Amazon Sales Dataset
- **Size**: ~1,463 רשומות
- **Time period**: [יתעדכן]
- **Geographic scope**: הודו (Amazon India)

### Features Used
| Feature | Type | Description |
|---------|------|-------------|
| actual_price | Numeric | מחיר מקורי (₹) |
| category | Categorical | קטגוריית המוצר |
| rating | Numeric | דירוג ממוצע (1-5) |
| rating_count | Numeric | מספר דירוגים |
| discount_percentage | Numeric | אחוז הנחה |

### Target Variable
- **discounted_price**: מחיר לאחר הנחה (₹)

### Data Processing
1. הסרת שורות עם ערכים חסרים
2. ניקוי פורמט מחירים (הסרת סימני ₹ ופסיקים)
3. המרת קטגוריות ל-One-Hot Encoding
4. נרמול ערכים מספריים

### Data Splits
- **Training**: 80%
- **Testing**: 20%
- **Validation**: Cross-validation (5-fold)

---

## Metrics

### Performance Results

> **הערה**: הטבלה תתעדכן לאחר אימון המודל על ידי נווה

| Metric | Linear Regression | Random Forest | Selected Model |
|--------|-------------------|---------------|----------------|
| MAE    | [TBD]            | [TBD]         | [TBD]          |
| RMSE   | [TBD]            | [TBD]         | [TBD]          |
| R²     | [TBD]            | [TBD]         | [TBD]          |
| MAPE   | [TBD]            | [TBD]         | [TBD]          |

### Metrics Explanation
- **MAE (Mean Absolute Error)**: שגיאה ממוצעת במונחי מחיר (₹)
- **RMSE (Root Mean Squared Error)**: שורש השגיאה הריבועית הממוצעת
- **R² (Coefficient of Determination)**: אחוז השונות שהמודל מסביר
- **MAPE (Mean Absolute Percentage Error)**: שגיאה ממוצעת באחוזים

### Success Criteria
| Metric | Target | Status |
|--------|--------|--------|
| R²     | > 0.70 | [TBD]  |
| MAPE   | < 15%  | [TBD]  |

---

## Limitations

### Known Limitations

1. **גודל מדגם מוגבל**
   - רק ~1,463 רשומות
   - עלול לגרום ל-overfitting
   - ייצוג לא מספק של כל הקטגוריות

2. **הטיית נתונים**
   - הנתונים מ-Amazon India בלבד
   - לא בהכרח מייצגים שווקים אחרים
   - ייתכן ריכוז בקטגוריות מסוימות

3. **תלות בזמן**
   - הנתונים מתקופה ספציפית
   - לא לוקחים בחשבון מגמות עונתיות
   - מחירים עשויים להשתנות לאורך זמן

4. **פיצ'רים חסרים**
   - אין מידע על מלאי
   - אין מידע על מתחרים
   - אין נתוני עלות לספק

### When Not to Use
- למוצרים חדשים ללא היסטוריה
- לקטגוריות שלא נראו באימון
- להחלטות קריטיות ללא בדיקה אנושית

---

## Ethical Considerations

### Bias and Fairness

#### Potential Biases
- **הטיה לקטגוריות נפוצות**: המודל עשוי לבצע טוב יותר על קטגוריות עם יותר נתונים
- **הטיית טווח מחירים**: ביצועים טובים יותר בטווחי מחירים נפוצים
- **הטיה גיאוגרפית**: מאומן על נתוני הודו בלבד

#### Mitigation Steps
- ניטור ביצועים לפי קטגוריה
- בדיקה ידנית של חיזויים קיצוניים
- עדכון תקופתי של המודל

### Privacy

- **אין שימוש בנתונים אישיים**: המודל לא משתמש במידע מזהה של משתמשים
- **נתוני מוצרים בלבד**: כל הפיצ'רים מבוססים על מאפייני המוצר
- **תוכן ביקורות לא נכלל**: טקסט הביקורות לא משמש לחיזוי

### Transparency

- **קוד פתוח**: כל הקוד זמין ב-Repository
- **תיעוד מלא**: כל ההחלטות מתועדות
- **Feature Importance**: חשיבות הפיצ'רים תפורסם

### Human Oversight

- **המלצה בלבד**: המודל מספק המלצות, לא החלטות סופיות
- **בדיקה אנושית**: מומלץ לבדוק חיזויים לפני שימוש
- **אחריות**: משתמשי המודל אחראים להחלטות הסופיות

---

## Recommendations

### Best Practices for Use
1. השתמש בחיזוי כנקודת התחלה, לא כהחלטה סופית
2. בדוק חיזויים קיצוניים (גבוהים או נמוכים מאוד)
3. עדכן את המודל תקופתית עם נתונים חדשים
4. נטר ביצועים לאורך זמן

### Future Improvements
- [ ] הרחבת מערך הנתונים
- [ ] הוספת פיצ'רים (עונתיות, מתחרים)
- [ ] ניסוי מודלים מתקדמים יותר (XGBoost, Neural Networks)
- [ ] הוספת Confidence Intervals לחיזויים

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0     | [TBD] | Initial release |

---

## Contact

- **Created by**: צוות מדעני הנתונים
- **Reviewed by**: מירב (QA)
- **Questions**: [Repository Issues]
