from pathlib import Path

import duckdb

RAW = "data/raw/yellow_tripdata_2024-*.parquet"
OUT_DIR = Path("data/clean")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "trips.parquet"

START, END = "2024-01-01", "2024-07-01"
MIN_MINUTES, MAX_MINUTES = 1, 180

con = duckdb.connect()
con.sql(f"""
CREATE VIEW raw AS
SELECT tpep_pickup_datetime AS pickup,
       passenger_count,
       PULocationID AS pu_zone,
       DOLocationID AS do_zone,
       (epoch(tpep_dropoff_datetime) - epoch(tpep_pickup_datetime)) / 60.0 AS minutes
FROM read_parquet('{RAW}', union_by_name=true)
""")

print("ROWS REMOVED BY EACH RULE (rules overlap, so these don't add up)")
con.sql(f"""
SELECT count(*) AS raw_rows,
  count(*) FILTER (WHERE pickup < TIMESTAMP '{START}' OR pickup >= TIMESTAMP '{END}') AS outside_dates,
  count(*) FILTER (WHERE minutes < {MIN_MINUTES}) AS too_short,
  count(*) FILTER (WHERE minutes > {MAX_MINUTES}) AS too_long,
  count(*) FILTER (WHERE pu_zone IS NULL OR do_zone IS NULL) AS missing_zone
FROM raw
""").show(max_width=200)

con.sql(f"""
CREATE TABLE clean AS
SELECT pickup, passenger_count, pu_zone, do_zone, minutes
FROM raw
WHERE pickup >= TIMESTAMP '{START}' AND pickup < TIMESTAMP '{END}'
  AND minutes BETWEEN {MIN_MINUTES} AND {MAX_MINUTES}
  AND pu_zone IS NOT NULL AND do_zone IS NOT NULL
""")
con.sql(f"COPY clean TO '{OUT}' (FORMAT PARQUET)")

print("ROWS KEPT PER MONTH")
con.sql("""
SELECT date_trunc('month', pickup) AS month,
       count(*) AS rows,
       round(avg(minutes), 1) AS mean_minutes
FROM clean GROUP BY 1 ORDER BY 1
""").show(max_width=200)
print("saved to", OUT)