# Chocolate-Sales-Revenue-Predictor

## Overview

This project builds an end-to-end machine learning pipeline to predict transaction-level revenue in a multi-store chocolate retail business. By joining five relational data tables — sales transactions, product catalogue, store metadata, customer demographics, and a calendar dimension — the project demonstrates how raw business data is transformed into actionable predictive models.

The pipeline covers data merging, cleaning, exploratory analysis, feature engineering, multi-model training, and evaluation. The final model enables the business to estimate expected revenue for any combination of product, store, customer profile, and date — supporting decisions in inventory planning, pricing strategy, and store expansion.

---

## Business Problem

A chocolate retail business operates across multiple store types and geographies. Revenue per transaction varies significantly across products, stores, seasons, and customer segments, making it difficult to forecast performance or identify underperforming areas.

**Questions this project answers:**

- Which products, brands, and store types generate the most revenue and profit?
- How does revenue vary by month, quarter, and day of week?
- Which features are the strongest predictors of transaction revenue?
- How accurately can a machine learning model predict revenue for a new transaction?

---

## Dataset

The dataset is a relational schema of five CSV files that are merged prior to analysis.

| File | Description | Key Columns |
|---|---|---|
| `sales.csv` | One row per transaction | order_id, product_id, store_id, customer_id, revenue, profit, cost, quantity |
| `products.csv` | Product catalogue | product_id, product_name, brand, category, cocoa_percent, weight_g |
| `stores.csv` | Store metadata | store_id, store_type, country |
| `customers.csv` | Customer demographics | customer_id, customer attributes |
| `calendar.csv` | Date dimension table | date, year, month, quarter, day_of_week |

**Merged DataFrame:** One wide analytical table containing all transaction, product, store, customer, and calendar attributes joined on their respective keys.

**Target Variable:** `revenue` (continuous numeric — transaction revenue in pounds)

---

## Technologies Used

| Category | Libraries |
|---|---|
| Data manipulation | Python, Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Machine learning | Scikit-Learn |
| Notebook | Jupyter Notebook |
| Model persistence | joblib |

---

## Project Workflow

### 1. Data Loading and Schema Merging

The five source tables are loaded and joined using LEFT JOINs to preserve every sales transaction. The calendar dimension is merged on the order date to attach pre-computed temporal features.

### 2. Data Cleaning

| Issue | Treatment |
|---|---|
| Rows where all product fields are null | Dropped |
| Categorical nulls (product_name, brand, category) | Filled with 'Unknown' |
| Numeric nulls (cocoa_percent, weight_g) | Imputed with column median |
| Duplicate rows | Detected and removed |
| Invalid revenue values (null or negative) | Removed |
| order_date stored as string | Converted to datetime64 |

### 3. Exploratory Data Analysis

EDA examines revenue distributions, product and store performance, seasonal trends, and correlation between numeric features. Every chart is followed by a written observation explaining what the data shows and why it matters for the business.

Key EDA charts:

- Revenue histogram and boxplot (distribution shape and outliers)
- Revenue by product category and top brands
- Revenue and profit by store type and country
- Monthly revenue trend by year (seasonality)
- Quantity vs revenue scatter plot (price consistency)
- Correlation heatmap (multicollinearity check)
- Feature importance from Random Forest

### 4. Feature Engineering

| Feature | Type | Rationale |
|---|---|---|
| year, month, day, quarter | Temporal decomposition | Captures seasonality and growth trends |
| day_of_week | Temporal | Day-level purchase behaviour |
| is_weekend | Binary flag | Weekend vs weekday traffic patterns |
| is_month_start, is_month_end | Binary flags | Payday and end-of-month purchase spikes |
| Brand frequency capping | Categorical reduction | Groups rare brands as 'Other' to reduce noise |
| Label encoding | Categorical encoding | Converts string categories to integer codes for ML |

Columns dropped before modelling to prevent data leakage: `profit`, `cost`, `profit_margin`, `revenue_per_unit`, `order_date`, `product_id`, `store_id`, `customer_id`, `order_id`.

### 5. Model Building

Four regression models are trained and compared:

| Model | Description |
|---|---|
| Linear Regression | Baseline; assumes linear relationships between features and revenue |
| Decision Tree | Non-linear; splits data on feature thresholds; prone to overfitting |
| Random Forest | Ensemble of 100 trees trained on bootstrap samples; robust and accurate |
| Gradient Boosting | Sequential ensemble; each tree corrects previous errors; typically highest accuracy |

### 6. Model Evaluation

---

## Key Findings

1. **Seasonality drives revenue.** Month and quarter consistently rank as the most important features, confirming predictable seasonal peaks aligned with gifting occasions (Valentine's Day, Easter, Christmas).

2. **Product mix matters as much as volume.** Category and brand are strong predictors — what is sold influences revenue at least as much as how much is sold.

3. **Store type creates structural revenue differences.** Different channels (flagship, franchise, online) produce different average revenue per transaction. Expanding higher-performing channels is a direct growth lever.

4. **Gradient Boosting outperforms all other models.** Sequential boosting captures complex feature interactions that linear models and single trees cannot learn.

5. **Quantity alone does not determine revenue.** Price per unit varies significantly across transactions, driven by product tier, store type, and promotional activity. Revenue optimisation requires managing both dimensions simultaneously.

---

## Visualisations

| Chart | What It Shows |
|---|---|
| Revenue Histogram and KDE | Distribution shape and presence of skew |
| Revenue Boxplot | Outlier transactions |
| Revenue by Category and Brand | Top-performing product segments |
| Revenue and Profit by Country | Geographic performance distribution |
| Monthly Trend by Year | Seasonal patterns and year-on-year growth |
| Scatter — Quantity vs Revenue | Price consistency across transactions |
| Correlation Heatmap | Linear relationships and multicollinearity |
| Actual vs Predicted | Model prediction accuracy |
| Residual Plot | Systematic errors and model bias |
| R2 Comparison Bar Chart | Ranking of all four models |
| RMSE Comparison Bar Chart | Error in pounds for all four models |
| Feature Importance | What the model relies on most |

---

## Future Improvements

- **Hyperparameter tuning:** Use Optuna or GridSearchCV to optimise Gradient Boosting parameters for further accuracy gains.
- **XGBoost or LightGBM:** These gradient boosting libraries are typically faster and more accurate than sklearn's implementation on tabular retail data.
- **Lagged and rolling features:** Previous-period revenue (last month, last quarter same year) are powerful forecasting signals — requires time-sorted train/test splits.
- **SHAP explainability:** Per-transaction feature attribution to explain individual predictions to business stakeholders.
- **Model deployment:** Wrap the trained model in a FastAPI REST endpoint or Streamlit dashboard for real-time revenue scoring by the operations team.
- **Automated retraining pipeline:** Schedule monthly retraining as new transaction data accumulates.

