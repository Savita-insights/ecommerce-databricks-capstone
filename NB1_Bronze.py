# Databricks notebook source
# MAGIC %md
# MAGIC # NB1 – Bronze Layer
# MAGIC Raw data ingestion from CSV → Delta table with schema enforcement and time travel

# COMMAND ----------

from pyspark.sql import functions as F

# Read raw CSV
raw_df = spark.read.csv(
    "/Volumes/workspace/ecommerce/ecommerce_data/2019-Oct.csv",
    header=True,
    inferSchema=True
)

print(f"✅ Raw data loaded: {raw_df.count():,} rows")
raw_df.printSchema()

# COMMAND ----------

# Add ingestion timestamp (audit column)
bronze_df = raw_df.withColumn("ingestion_ts", F.current_timestamp())

# Write as Delta table - Bronze layer
bronze_df.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.bronze.events")

print("✅ Bronze table created: workspace.bronze.events")

# COMMAND ----------

# Verify schema enforcement - this should FAIL (proves enforcement works)
try:
    wrong_df = spark.createDataFrame([("a","b","c")], ["x","y","z"])
    wrong_df.write.format("delta").mode("append").saveAsTable("workspace.bronze.events")
except Exception as e:
    print("✅ Schema enforcement working - bad write correctly rejected")
    print(f"   Error: {str(e)[:80]}")

# COMMAND ----------

# Show time travel history
spark.sql("DESCRIBE HISTORY workspace.bronze.events").select(
    "version", "timestamp", "operation", "operationMetrics"
).show(5, truncate=False)

# COMMAND ----------

# Optimize bronze table
spark.sql("OPTIMIZE workspace.bronze.events ZORDER BY (user_id, event_type)")
print("✅ Bronze table optimized with ZORDER")

# COMMAND ----------

# Final count and sample
print(f"Bronze row count: {spark.table('workspace.bronze.events').count():,}")
spark.table("workspace.bronze.events").show(5)
