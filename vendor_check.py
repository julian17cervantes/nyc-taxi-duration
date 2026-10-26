import sys

import duckdb

con = duckdb.connect()
for path in sys.argv[1:]:
    print(path)
    con.sql(f"""
    SELECT VendorID,
           count(*) AS trips,
           round(100.0 * count(*) FILTER (WHERE minutes <= 0) / count(*), 2) AS pct_non_positive,
           round(100.0 * count(*) FILTER (WHERE minutes < 1) / count(*), 2) AS pct_under_1_min,
           round(100.0 * count(*) FILTER (WHERE passenger_count IS NULL) / count(*), 2) AS pct_null_pax,
           round(quantile_cont(minutes, 0.5), 1) AS median_minutes
    FROM (
      SELECT *, (epoch(tpep_dropoff_datetime) - epoch(tpep_pickup_datetime)) / 60.0 AS minutes
      FROM '{path}'
    )
    GROUP BY VendorID ORDER BY VendorID
    """).show(max_width=200)