import duckdb

DATA = "data/clean/trips.parquet"
TRAIN_END = "2024-05-01"
VAL_END = "2024-06-01"
MIN_N = 30

con = duckdb.connect()
con.execute(f"""
CREATE TABLE trips AS
SELECT pickup, pu_zone, do_zone, minutes, hour(pickup) AS hr,
  CASE WHEN pickup < TIMESTAMP '{TRAIN_END}' THEN 'train'
       WHEN pickup < TIMESTAMP '{VAL_END}' THEN 'val'
       ELSE 'test' END AS split
FROM read_parquet('{DATA}')
""")

print("SPLITS")
con.sql("""
SELECT split, count(*) AS rows, round(avg(minutes), 1) AS mean_minutes
FROM trips GROUP BY 1 ORDER BY 1
""").show()

statements = [
    "CREATE TABLE z_global AS SELECT median(minutes) AS m FROM trips WHERE split = 'train'",
    f"""CREATE TABLE z_pu AS SELECT pu_zone, median(minutes) AS m FROM trips
        WHERE split = 'train' GROUP BY 1 HAVING count(*) >= {MIN_N}""",
    f"""CREATE TABLE z_pair AS SELECT pu_zone, do_zone, median(minutes) AS m FROM trips
        WHERE split = 'train' GROUP BY 1, 2 HAVING count(*) >= {MIN_N}""",
    f"""CREATE TABLE z_pair_hr AS SELECT pu_zone, do_zone, hr, median(minutes) AS m FROM trips
        WHERE split = 'train' GROUP BY 1, 2, 3 HAVING count(*) >= {MIN_N}""",
    """CREATE TABLE val_pred AS
       SELECT t.minutes,
              (SELECT m FROM z_global) AS p_global,
              COALESCE(a.m, b.m, (SELECT m FROM z_global)) AS p_pair,
              COALESCE(c.m, a.m, b.m, (SELECT m FROM z_global)) AS p_pair_hour,
              a.m IS NULL AS pair_missing
       FROM trips t
       LEFT JOIN z_pair a ON t.pu_zone = a.pu_zone AND t.do_zone = a.do_zone
       LEFT JOIN z_pu b ON t.pu_zone = b.pu_zone
       LEFT JOIN z_pair_hr c ON t.pu_zone = c.pu_zone AND t.do_zone = c.do_zone AND t.hr = c.hr
       WHERE t.split = 'val'""",
]
for statement in statements:
    con.execute(statement)

print("BASELINES ON VALIDATION (May)")
for column, label in [
    ("p_global", "overall median"),
    ("p_pair", "zone-pair median"),
    ("p_pair_hour", "zone-pair + hour median"),
]:
    mae, rmse = con.execute(
        f"SELECT avg(abs(minutes - {column})), sqrt(avg(pow(minutes - {column}, 2))) FROM val_pred"
    ).fetchone()
    print(f"{label:>25}: MAE {mae:.2f} min   RMSE {rmse:.2f} min")

share = con.execute("SELECT avg(pair_missing::INT) FROM val_pred").fetchone()[0]
print(f"\nvalidation trips whose zone pair had too little training data: {share:.1%}")