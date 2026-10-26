import duckdb
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "data/raw/yellow_tripdata_2024-01.parquet"
con = duckdb.connect()

print("COLUMNS")
con.sql(f"DESCRIBE SELECT * FROM '{PATH}'").show(max_rows=40)

print("QUALITY CHECKS")
con.sql(f"""
WITH t AS (
  SELECT *,
         (epoch(tpep_dropoff_datetime) - epoch(tpep_pickup_datetime)) / 60.0 AS minutes
  FROM '{PATH}'
)
SELECT
  count(*) AS n_rows,
  min(tpep_pickup_datetime) AS first_pickup,
  max(tpep_pickup_datetime) AS last_pickup,
  count(*) FILTER (WHERE minutes <= 0) AS non_positive_duration,
  count(*) FILTER (WHERE minutes > 180) AS over_3_hours,
  count(*) FILTER (WHERE trip_distance = 0) AS zero_distance,
  count(*) FILTER (WHERE passenger_count IS NULL) AS null_passengers,
  count(DISTINCT PULocationID) AS pickup_zones,
  round(quantile_cont(minutes, 0.5), 1) AS median_minutes,
  round(quantile_cont(minutes, 0.99), 1) AS p99_minutes
FROM t
""").show(max_width=200)