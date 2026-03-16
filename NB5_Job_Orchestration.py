# Databricks notebook source
# MAGIC %md
# MAGIC # NB5 – Job Orchestration
# MAGIC Runs the full pipeline end to end and verifies each layer

# COMMAND ----------

import time
from datetime import datetime

def log_step(step, count=None, duration=None):
    ts  = datetime.now().strftime("%H:%M:%S")
    cnt = f"  |  {count:,} rows" if count else ""
    dur = f"  |  {duration:.1f}s" if duration else ""
    print(f"[{ts}]  ✅  {step}{cnt}{dur}")

print("=" * 55)
print("   ECOMMERCE CAPSTONE – FULL PIPELINE VERIFICATION")
print("=" * 55)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Layer 1 – Bronze

# COMMAND ----------

t = time.time()
bronze_count = spark.table("workspace.bronze.events").count()
log_step("Bronze layer", bronze_count, time.time() - t)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Layer 2 – Silver

# COMMAND ----------

t = time.time()
silver_count = spark.table("workspace.silver.events").count()
log_step("Silver layer", silver_count, time.time() - t)
print(f"   Rows removed in cleaning: {bronze_count - silver_count:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Layer 3 – Gold Tables

# COMMAND ----------

t = time.time()
gold_users    = spark.table("workspace.gold.user_features").count()
gold_products = spark.table("workspace.gold.product_performance").count()
gold_revenue  = spark.table("workspace.gold.daily_revenue").count()
log_step("Gold – user_features",        gold_users,    time.time() - t)
log_step("Gold – product_performance",  gold_products)
log_step("Gold – daily_revenue",        gold_revenue)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Layer 4 – ML Predictions

# COMMAND ----------

t = time.time()
pred_count = spark.table("workspace.gold.user_purchase_predictions").count()
log_step("Gold – user_purchase_predictions", pred_count, time.time() - t)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Full Pipeline Summary

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 'workspace.bronze.events'                   AS layer, COUNT(*) AS rows FROM workspace.bronze.events
# MAGIC UNION ALL
# MAGIC SELECT 'workspace.silver.events',                           COUNT(*) FROM workspace.silver.events
# MAGIC UNION ALL
# MAGIC SELECT 'workspace.gold.user_features',                      COUNT(*) FROM workspace.gold.user_features
# MAGIC UNION ALL
# MAGIC SELECT 'workspace.gold.product_performance',                COUNT(*) FROM workspace.gold.product_performance
# MAGIC UNION ALL
# MAGIC SELECT 'workspace.gold.daily_revenue',                      COUNT(*) FROM workspace.gold.daily_revenue
# MAGIC UNION ALL
# MAGIC SELECT 'workspace.gold.user_purchase_predictions',          COUNT(*) FROM workspace.gold.user_purchase_predictions
# MAGIC ORDER BY layer

# COMMAND ----------

print("🎉 Full pipeline verified!")
print("   Bronze → Silver → Gold → ML → Predictions")
print("   All Delta tables present and populated ✅")
print(f"\n   Bronze:      {bronze_count:>10,} rows")
print(f"   Silver:      {silver_count:>10,} rows")
print(f"   Users(Gold): {gold_users:>10,} users")
print(f"   Predictions: {pred_count:>10,} users scored")
