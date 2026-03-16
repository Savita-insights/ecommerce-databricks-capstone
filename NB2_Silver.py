# Databricks notebook source
# MAGIC %md
# MAGIC # NB2 – Silver Layer
# MAGIC Clean, deduplicate, standardize → optimized Delta table

# COMMAND ----------

from pyspark.sql import functions as F

# Load from Bronze
bronze_df = spark.table("workspace.bronze.events")
print(f"Bronze rows: {bronze_df.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Checks

# COMMAND ----------

# Check nulls before cleaning
print("Null counts before cleaning:")
from pyspark.sql.functions import col, sum as spark_sum
null_counts = bronze_df.select([
    spark_sum(col(c).isNull().cast("int")).alias(c)
    for c in bronze_df.columns
])
null_counts.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Silver Transformations

# COMMAND ----------

silver_df = bronze_df \
    .filter(F.col("price") > 0) \
    .filter(F.col("user_id").isNotNull()) \
    .filter(F.col("event_type").isNotNull()) \
    .dropDuplicates(["user_session", "event_time", "product_id"]) \
    .withColumn("event_date", F.to_date(F.col("event_time"))) \
    .withColumn("event_hour", F.hour(F.col("event_time"))) \
    .withColumn("price_category",
        F.when(F.col("price") < 100, "budget")
         .when(F.col("price") < 1000, "mid_range")
         .otherwise("premium")
    ) \
    .drop("ingestion_ts")

print(f"Silver rows after cleaning: {silver_df.count():,}")

# COMMAND ----------

# Write Silver Delta table
silver_df.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.silver.events")

print("✅ Silver table created: workspace.silver.events")

# COMMAND ----------

# Optimize Silver
spark.sql("OPTIMIZE workspace.silver.events ZORDER BY (user_id, event_date)")
print("✅ Silver table optimized")

# COMMAND ----------

# Time travel check
spark.sql("DESCRIBE HISTORY workspace.silver.events").select(
    "version", "timestamp", "operation"
).show(5)

# COMMAND ----------

# Business insight from Silver
print("Event type distribution:")
spark.table("workspace.silver.events") \
    .groupBy("event_type") \
    .count() \
    .orderBy("count", ascending=False) \
    .show()
