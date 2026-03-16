# 🛒 E-Commerce Purchase Prediction — Databricks Capstone

A production-grade end-to-end Data & AI system built on Databricks Lakehouse Platform to predict whether an e-commerce user will make a purchase, using 42 million real-world behavioral events.

---

## 🎯 Problem Statement

**Business Question:** Given a user's browsing and interaction behavior on an e-commerce platform, can we predict whether they will make a purchase?

- **Task:** Binary Classification (will purchase = 1, will not = 0)
- **Dataset:** [E-Commerce Behavior Data](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store) — October 2019 (42M events)
- **Impact:** Enables targeted marketing, personalized recommendations, and revenue optimization

---

## 🏗️ Architecture

```
Raw CSV (42M rows)
      ↓
┌─────────────────────────────────────────────────────┐
│              MEDALLION ARCHITECTURE                  │
│                                                      │
│  [Bronze Layer]  →  [Silver Layer]  →  [Gold Layer]  │
│  Raw Delta table    Cleaned + dedup   Aggregations   │
│  42,448,764 rows    42,344,170 rows   Feature tables │
└─────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────┐
│              ML PIPELINE                             │
│                                                      │
│  Feature Engineering → Model Training → MLflow      │
│  user_features table   RandomForest    Experiment   │
│  3,021,434 users       AUC: 0.926      Tracking     │
└─────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────┐
│              BATCH INFERENCE                         │
│                                                      │
│  Load Model → Score 3M users → Gold Predictions     │
│  43,392 likely buyers identified                     │
└─────────────────────────────────────────────────────┘
      ↓
 Databricks Job Orchestration (automated pipeline)
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | Kaggle — Multi-Category Store |
| File | 2019-Oct.csv |
| Total Events | 42,448,764 |
| Event Types | view, cart, purchase |
| Time Period | October 2019 |
| Storage | Databricks Unity Catalog Volume |

---

## 🔧 Tech Stack

| Component | Technology |
|---|---|
| Platform | Databricks (Serverless) |
| Storage | Delta Lake (ACID transactions) |
| Processing | Apache Spark / PySpark |
| ML Library | Scikit-learn |
| Experiment Tracking | MLflow |
| Orchestration | Databricks Workflows (Jobs) |
| Language | Python, SQL |

---

## 📁 Project Structure

```
ecommerce-databricks-capstone/
├── notebooks/
│   ├── NB1_Bronze.py          # Raw ingestion → Delta table
│   ├── NB2_Silver.py          # Cleaning + deduplication
│   ├── NB3_Gold.py            # Aggregations + feature engineering
│   ├── NB4_ML_MLflow.py       # ML training + MLflow tracking
│   └── NB5_Job_Orchestration.py  # Pipeline verification
├── README.md
└── architecture_diagram.png
```

---

## 🥉 Bronze Layer — Raw Ingestion

- Ingests raw CSV from Databricks Volume
- Writes as Delta table with schema enforcement
- Adds `ingestion_ts` audit column
- Demonstrates time travel with `DESCRIBE HISTORY`
- Optimized with `OPTIMIZE + ZORDER BY (user_id, event_type)`

**Table:** `workspace.bronze.events` — 42,448,764 rows

---

## 🥈 Silver Layer — Cleaned Data

Transformations applied:
- Filter out records with `price <= 0`
- Remove duplicates by `user_session + event_time + product_id`
- Add derived columns: `event_date`, `event_hour`, `price_category`
- Optimized with `ZORDER BY (user_id, event_date)`

**Table:** `workspace.silver.events` — 42,344,170 rows

---

## 🥇 Gold Layer — Business & ML Ready

Three Gold tables created:

| Table | Description | Rows |
|---|---|---|
| `gold.daily_revenue` | Revenue by date | 31 days |
| `gold.product_performance` | Views, conversions by product | ~500K products |
| `gold.user_features` | ML feature table per user | 3,021,434 users |

**Feature Engineering:**
- `total_events` — total interactions per user
- `views` — view count
- `cart_adds` — add to cart count
- `total_spent` — total purchase value
- `avg_price` — average item price
- `unique_products` — product diversity
- `active_days` — engagement span

---

## 🤖 Machine Learning Pipeline

### Problem
Binary classification: predict if a user will purchase (`purchases > 0`)

### Models Trained & Compared

| Model | AUC | F1 Score | Accuracy |
|---|---|---|---|
| **Random Forest** ✅ | **0.9260** | **0.8540** | **0.8929** |
| Logistic Regression | 0.8360 | 0.9472 | 0.9494 |

### MLflow Tracking
- Experiment: `/Day-12-MLflow-Basics`
- 2 runs logged with parameters, metrics, and models
- Best model: Random Forest (AUC 0.926)

### Batch Inference Results
- Scored all 3,021,434 users
- **43,392 users** identified as "Likely to Purchase"
- Results stored in `workspace.gold.user_purchase_predictions`

---

## ⚙️ Databricks Job Orchestration

Automated pipeline: **Ecommerce Capstone Pipeline**

```
[bronze] → [silver] → [gold] → [ml_training]
  ✅           ✅         ✅          ✅
```

Each task depends on the previous ensuring data quality at every layer.

---

## 🚀 Setup Instructions

### Prerequisites
- Databricks workspace (Free Edition or above)
- Kaggle account for dataset download

### Step 1 — Download Dataset
```python
import os
os.environ["KAGGLE_USERNAME"] = "your_username"
os.environ["KAGGLE_KEY"] = "your_key"
```
```bash
kaggle datasets download -d mkechinov/ecommerce-behavior-data-from-multi-category-store
```

### Step 2 — Create Schemas
```python
spark.sql("CREATE SCHEMA IF NOT EXISTS workspace.bronze")
spark.sql("CREATE SCHEMA IF NOT EXISTS workspace.silver")
spark.sql("CREATE SCHEMA IF NOT EXISTS workspace.gold")
```

### Step 3 — Run Notebooks in Order
1. `NB1_Bronze.py` — ingest raw data
2. `NB2_Silver.py` — clean and transform
3. `NB3_Gold.py` — build feature tables
4. `NB4_ML_MLflow.py` — train model and log to MLflow
5. `NB5_Job_Orchestration.py` — verify full pipeline

### Step 4 — Set Up Databricks Job
- Go to Workflows → Create Job
- Add 4 tasks in sequence: bronze → silver → gold → ml_training
- Click Run now

---

## 📈 Key Business Insights

1. **Purchase rate is 11.5%** — 1 in 9 users makes a purchase
2. **43,392 high-intent users** identified for targeted campaigns
3. **Top conversion driver:** number of active days on platform
4. **Revenue pattern:** consistent daily peaks in October 2019
5. **Model ROI:** targeting the 43K predicted buyers vs 3M total users = 98.6% reduction in marketing spend waste

---

## 🏆 Results Summary

| Metric | Value |
|---|---|
| Total events processed | 42,448,764 |
| Users scored | 3,021,434 |
| Likely buyers identified | 43,392 (1.4%) |
| Best model AUC | 0.926 |
| Pipeline layers | 5 (Bronze→Silver→Gold→ML→Inference) |
| Delta tables created | 6 |
| MLflow runs logged | 2 |

---

## 👩‍💻 Author

**Savitha** | Data Analyst  
[LinkedIn](https://www.linkedin.com/in/savita-m-82b2b6111/)) | [GitHub](https://github.com/Savita-insights)

---

*Built as part of the Build With Databricks: Hands-On Project Challenge*
