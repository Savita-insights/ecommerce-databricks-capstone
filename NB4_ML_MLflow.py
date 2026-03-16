# Databricks notebook source
# MAGIC %md
# MAGIC # NB4 – ML Training + MLflow Experiment Tracking
# MAGIC **Problem:** Predict whether a user will make a purchase (binary classification)
# MAGIC **Data:** workspace.gold.user_features
# MAGIC **Models:** Random Forest vs Logistic Regression

# COMMAND ----------

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

# Use folder-based experiment path (proven to work in your workspace)
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Users/malisavitackd@gmail.com/Capstone-Purchase-Prediction")
print("✅ MLflow experiment set")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 – Load Gold Feature Table

# COMMAND ----------

pdf = spark.table("workspace.gold.user_features").fillna(0).toPandas()

print(f"Total users: {len(pdf):,}")
print(f"Columns: {list(pdf.columns)}")
pdf.head(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 – Define Target Label
# MAGIC Users with purchases > 0 = buyer (1), else non-buyer (0)

# COMMAND ----------

feature_cols = ["total_events", "views", "cart_adds",
                "total_spent", "avg_price",
                "unique_products", "active_days"]

pdf["label"] = (pdf["purchases"] > 0).astype(int)

print("Class distribution:")
print(pdf["label"].value_counts())
print(f"\nBuyer rate: {pdf['label'].mean()*100:.1f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 – Train/Test Split

# COMMAND ----------

X = pdf[feature_cols]
y = pdf["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"Buyers in test: {y_test.sum():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4 – MLflow Run 1: Random Forest

# COMMAND ----------

with mlflow.start_run(run_name="RandomForest_100trees"):

    # Log parameters
    mlflow.log_param("model_type",   "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth",    5)
    mlflow.log_param("features",     str(feature_cols))
    mlflow.log_param("train_size",   len(X_train))
    mlflow.log_param("test_size",    len(X_test))

    # Train
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=42
    )
    rf.fit(X_train, y_train)

    # Evaluate
    y_pred_rf      = rf.predict(X_test)
    y_pred_prob_rf = rf.predict_proba(X_test)[:, 1]

    auc_rf = roc_auc_score(y_test, y_pred_prob_rf)
    f1_rf  = f1_score(y_test, y_pred_rf, average="weighted")
    acc_rf = accuracy_score(y_test, y_pred_rf)

    # Log metrics
    mlflow.log_metric("auc",      auc_rf)
    mlflow.log_metric("f1_score", f1_rf)
    mlflow.log_metric("accuracy", acc_rf)

    # Log model
    mlflow.sklearn.log_model(rf, "rf_model")

    rf_run_id = mlflow.active_run().info.run_id
    print(f"✅ Random Forest logged!")
    print(f"   Run ID:   {rf_run_id}")
    print(f"   AUC:      {auc_rf:.4f}")
    print(f"   F1 Score: {f1_rf:.4f}")
    print(f"   Accuracy: {acc_rf:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5 – MLflow Run 2: Logistic Regression (baseline comparison)

# COMMAND ----------

with mlflow.start_run(run_name="LogisticRegression_baseline"):

    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter",   100)
    mlflow.log_param("features",   str(feature_cols))
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size",  len(X_test))

    lr = LogisticRegression(max_iter=100, random_state=42)
    lr.fit(X_train, y_train)

    y_pred_lr      = lr.predict(X_test)
    y_pred_prob_lr = lr.predict_proba(X_test)[:, 1]

    auc_lr = roc_auc_score(y_test, y_pred_prob_lr)
    f1_lr  = f1_score(y_test, y_pred_lr, average="weighted")
    acc_lr = accuracy_score(y_test, y_pred_lr)

    mlflow.log_metric("auc",      auc_lr)
    mlflow.log_metric("f1_score", f1_lr)
    mlflow.log_metric("accuracy", acc_lr)
    mlflow.sklearn.log_model(lr, "lr_model")

    lr_run_id = mlflow.active_run().info.run_id
    print(f"✅ Logistic Regression logged!")
    print(f"   Run ID:   {lr_run_id}")
    print(f"   AUC:      {auc_lr:.4f}")
    print(f"   F1 Score: {f1_lr:.4f}")
    print(f"   Accuracy: {acc_lr:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 – Model Comparison

# COMMAND ----------

print("=" * 45)
print("        MODEL COMPARISON SUMMARY")
print("=" * 45)
print(f"{'Metric':<12} {'RandomForest':>14} {'LogisticReg':>14}")
print("-" * 45)
print(f"{'AUC':<12} {auc_rf:>14.4f} {auc_lr:>14.4f}")
print(f"{'F1 Score':<12} {f1_rf:>14.4f} {f1_lr:>14.4f}")
print(f"{'Accuracy':<12} {acc_rf:>14.4f} {acc_lr:>14.4f}")
print("=" * 45)

best_model    = rf if auc_rf >= auc_lr else lr
best_run_id   = rf_run_id if auc_rf >= auc_lr else lr_run_id
best_name     = "RandomForest" if auc_rf >= auc_lr else "LogisticRegression"
best_model_uri = f"runs:/{best_run_id}/{'rf_model' if auc_rf >= auc_lr else 'lr_model'}"

print(f"\n🏆 Winner: {best_name}")
print(f"   Run ID: {best_run_id}")
print(f"   URI:    {best_model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7 – Batch Inference → Write to Gold Layer

# COMMAND ----------

# Load best model
loaded_model = mlflow.sklearn.load_model(best_model_uri)
print(f"✅ Best model loaded: {best_name}")

# Score all users
scoring_pdf = spark.table("workspace.gold.user_features").fillna(0).toPandas()

scoring_pdf["will_purchase"]        = loaded_model.predict(scoring_pdf[feature_cols])
scoring_pdf["purchase_probability"] = loaded_model.predict_proba(
                                          scoring_pdf[feature_cols])[:, 1].round(4)
scoring_pdf["purchase_label"]       = scoring_pdf["will_purchase"].apply(
    lambda x: "Likely to Purchase" if x == 1 else "Not Likely to Purchase"
)

print(f"Scored {len(scoring_pdf):,} users")

# COMMAND ----------

# Write predictions to Gold Delta table
predictions_spark = spark.createDataFrame(
    scoring_pdf[[
        "user_id", "total_events", "total_spent", "avg_price",
        "will_purchase", "purchase_probability", "purchase_label"
    ]]
)

predictions_spark.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.gold.user_purchase_predictions")

print("✅ Predictions written to workspace.gold.user_purchase_predictions")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   purchase_label,
# MAGIC   COUNT(*)                             AS user_count,
# MAGIC   ROUND(AVG(total_spent), 2)           AS avg_spend,
# MAGIC   ROUND(AVG(purchase_probability), 4)  AS avg_score
# MAGIC FROM workspace.gold.user_purchase_predictions
# MAGIC GROUP BY purchase_label
# MAGIC ORDER BY avg_spend DESC

# COMMAND ----------

print("🎯 ML Pipeline Complete!")
print(f"   Best Model:  {best_name}")
print(f"   AUC:         {max(auc_rf, auc_lr):.4f}")
print(f"   Predictions: workspace.gold.user_purchase_predictions")
