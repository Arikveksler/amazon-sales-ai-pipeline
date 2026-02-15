# Feature Registry

Reference for all features produced by `src/tools/feature_engineering.py`.

**Output file:** `data/features/features.csv`
**Source data:** `data/amazon.csv` (1,465 products, 16 raw columns)
**Result:** 1,465 rows x 12 numerical columns

---

## 1. Cleaned Numeric Features

These columns exist in the raw CSV but are stored as strings with currency symbols, commas, or percent signs. The pipeline strips those characters and converts to float.

| # | Column | Type | Raw Example | Logic |
|---|--------|------|-------------|-------|
| 1 | `discounted_price` | float64 | `₹1,099` | Remove `₹` and `,`, convert to float. Function: `clean_price_column()` |
| 2 | `actual_price` | float64 | `₹1,599` | Same as above. Function: `clean_price_column()` |
| 3 | `discount_percentage` | float64 | `64%` | Remove `%`, convert to float. Function: `clean_discount_percentage()` |
| 4 | `rating` | float64 | `4.2` | Convert string to float via `pd.to_numeric()` |
| 5 | `rating_count` | float64 | `24,269` | Remove `,`, convert to float. Function: `clean_rating_count()` |

---

## 2. Category-Encoded Features

The raw `category` column is a pipe-delimited string (e.g. `Electronics|Headphones|InEar`). These features extract structure from it.

| # | Column | Type | Logic |
|---|--------|------|-------|
| 6 | `main_category_encoded` | int64 | First segment before `\|`, then Label Encoded. 9 classes: Car&Motorbike, Computers&Accessories, Electronics, Health&PersonalCare, Home&Kitchen, HomeImprovement, MusicalInstruments, OfficeProducts, Toys&Games |
| 7 | `sub_category_encoded` | int64 | Second segment after the first `\|` (or `'Unknown'` if none), then Label Encoded. 29 classes. |
| 8 | `category_depth` | int64 | Number of `\|`-separated segments in the full category string. Represents how specific the product categorization is. |

**Why Label Encoding instead of One-Hot:**
9 main + 29 sub-categories = 38 One-Hot columns for only 1,465 rows. Label Encoding produces just 2 columns and works well with tree-based models (Random Forest, XGBoost).

Function: `extract_category_features()` using `sklearn.preprocessing.LabelEncoder`.

---

## 3. Text-Derived Features

Numerical proxies extracted from free-text columns.

| # | Column | Type | Logic |
|---|--------|------|-------|
| 9 | `title_length` | int64 | Character count of `product_name`. Longer titles often indicate more detailed or feature-rich products. |
| 10 | `desc_length` | int64 | Character count of `about_product`. Longer descriptions may correlate with higher-value products. |

Function: `extract_text_length_features()`

---

## 4. Computed Features

Derived from the cleaned price columns.

| # | Column | Type | Logic |
|---|--------|------|-------|
| 11 | `discount_amount` | float64 | `actual_price - discounted_price`. The absolute savings in INR. |
| 12 | `price_ratio` | float64 | `discounted_price / actual_price`. Values between 0 and 1 — lower means a bigger discount. |

Function: `compute_derived_features()`

---

## NaN Handling

After all transformations, any remaining NaN values (from unparseable strings or division by zero) are filled with the **column median**. In practice, the Amazon dataset has only 3 NaN values across all 12 columns.

---

## How to Regenerate

```bash
python src/tools/feature_engineering.py
```

## How to Validate

```bash
python -m unittest tests.test_feature_engineering -v
```
