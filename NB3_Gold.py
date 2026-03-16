# Databricks notebook source
# MAGIC %md
# MAGIC # NB3 – Gold Layer
# MAGIC Business aggregations + ML feature table

# COMMAND ----------

from pyspark.sql import functions as F

silver_df = spark.table("workspace.silver.events")
print(f"Silver rows loaded: {silver_df.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gold Table 1 – Daily Revenue Summary (Business Analytics)

# COMMAND ----------

daily_revenue = silver_df \
    .filter(F.col("event_type") == "purchase") \
    .groupBy("event_date") \
    .agg(
        F.count("*").alias("total_purchases"),
        F.sum("price").alias("total_revenue"),
        F.avg("price").alias("avg_order_value"),
        F.countDistinct("user_id").alias("unique_buyers")
    ) \
    .orderBy("event_date")

daily_revenue.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.gold.daily_revenue")

print("✅ Gold table created: workspace.gold.daily_revenue")
daily_revenue.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gold Table 2 – Product Performance

# COMMAND ----------

product_stats = silver_df \
    .groupBy("product_id", "brand", "price_category") \
    .agg(
        F.count("*").alias("total_views"),
        F.sum(F.when(F.col("event_type") == "purchase", 1).otherwise(0)).alias("purchases"),
        F.sum(F.when(F.col("event_type") == "add_to_cart", 1).otherwise(0)).alias("cart_adds"),
        F.avg("price").alias("avg_price")
    ) \
    .withColumn("conversion_rate",
        F.round(F.col("purchases") / F.col("total_views") * 100, 2)
    )

product_stats.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.gold.product_performance")

print("✅ Gold table created: workspace.gold.product_performance")
product_stats.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gold Table 3 – User Feature Table (for ML)

# COMMAND ----------

user_features = silver_df \
    .groupBy("user_id") \
    .agg(
        F.count("*").alias("total_events"),
        F.sum(F.when(F.col("event_type") == "purchase", 1).otherwise(0)).alias("purchases"),
        F.sum(F.when(F.col("event_type") == "view", 1).otherwise(0)).alias("views"),
        F.sum(F.when(F.col("event_type") == "add_to_cart", 1).otherwise(0)).alias("cart_adds"),
        F.sum(F.when(F.col("event_type") == "purchase", F.col("price")).otherwise(0)).alias("total_spent"),
        F.avg("price").alias("avg_price"),
        F.countDistinct("product_id").alias("unique_products"),
        F.countDistinct("event_date").alias("active_days")
    ) \
    .fillna(0)

user_features.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.gold.user_features")

print("✅ Gold table created: workspace.gold.user_features")
print(f"Total users: {user_features.count():,}")
user_features.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Business Insights (SQL Analytics)

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Top 10 revenue days
# MAGIC SELECT event_date, total_revenue, total_purchases, unique_buyers
# MAGIC FROM workspace.gold.daily_revenue
# MAGIC ORDER BY total_revenue DESC
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Purchase funnel analysis
# MAGIC SELECT
# MAGIC   ROUND(SUM(cart_adds) * 100.0 / SUM(total_views), 2) AS view_to_cart_pct,
# MAGIC   ROUND(SUM(purchases) * 100.0 / SUM(cart_adds), 2)   AS cart_to_purchase_pct,
# MAGIC   ROUND(SUM(purchases) * 100.0 / SUM(total_views), 2) AS overall_conversion_pct
# MAGIC FROM workspace.gold.product_performance

# COMMAND ----------

print("✅ Gold layer complete — 3 tables ready:")
print("   workspace.gold.daily_revenue")
print("   workspace.gold.product_performance")
print("   workspace.gold.user_features")
