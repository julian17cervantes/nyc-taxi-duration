from config import TEST_MONTH, TRAIN_END, VAL_END

import json
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

DATA = "data/clean/trips.parquet"
MIN_N = 30
FEATURES = ["pu_zone", "do_zone", "hour", "dow", "passenger_count"]
ZONES = list(range(1, 266))

con = duckdb.connect()
con.execute(f"""
CREATE TABLE trips AS
SELECT pu_zone, do_zone, hour(pickup) AS hour, dayofweek(pickup) AS dow,
       passenger_count, minutes,
  CASE WHEN pickup < TIMESTAMP '{TRAIN_END}' THEN 'train'
       WHEN pickup < TIMESTAMP '{VAL_END}' THEN 'val'
       ELSE 'test' END AS split
FROM read_parquet('{DATA}')
""")

for statement in [
    "CREATE TABLE z_global AS SELECT median(minutes) AS m FROM trips WHERE split = 'train'",
    f"""CREATE TABLE z_pu AS SELECT pu_zone, median(minutes) AS m FROM trips
        WHERE split = 'train' GROUP BY 1 HAVING count(*) >= {MIN_N}""",
    f"""CREATE TABLE z_pair AS SELECT pu_zone, do_zone, median(minutes) AS m FROM trips
        WHERE split = 'train' GROUP BY 1, 2 HAVING count(*) >= {MIN_N}""",
    f"""CREATE TABLE z_pair_hr AS SELECT pu_zone, do_zone, hour, median(minutes) AS m FROM trips
        WHERE split = 'train' GROUP BY 1, 2, 3 HAVING count(*) >= {MIN_N}""",
]:
    con.execute(statement)

test = con.execute("""
SELECT t.pu_zone, t.do_zone, t.hour, t.dow, t.passenger_count, t.minutes,
       (SELECT m FROM z_global) AS p_global,
       COALESCE(a.m, b.m, (SELECT m FROM z_global)) AS p_pair,
       COALESCE(c.m, a.m, b.m, (SELECT m FROM z_global)) AS p_pair_hour
FROM trips t
LEFT JOIN z_pair a ON t.pu_zone = a.pu_zone AND t.do_zone = a.do_zone
LEFT JOIN z_pu b ON t.pu_zone = b.pu_zone
LEFT JOIN z_pair_hr c ON t.pu_zone = c.pu_zone AND t.do_zone = c.do_zone AND t.hour = c.hour
WHERE t.split = 'test'
""").df()
print(f"test rows (June): {len(test):,}")

for col in ["pu_zone", "do_zone"]:
    test[col] = pd.Categorical(test[col], categories=ZONES)

booster = lgb.Booster(model_file="models/lgbm.txt")
print("trees in model:", booster.num_trees(), "(the full run should have 1500)")
test["p_model"] = booster.predict(test[FEATURES])

actual = test["minutes"].to_numpy()

def score(column):
    error = test[column].to_numpy() - actual
    return {
        "mae": round(float(np.abs(error).mean()), 3),
        "rmse": round(float(np.sqrt((error**2).mean())), 3),
        "mean_error": round(float(error.mean()), 3),
        "median_error": round(float(np.median(error)), 3),
    }


results = {
    "test_month": TEST_MONTH,
    "test_rows": len(test),
    "overall_median": score("p_global"),
    "zone_pair_median": score("p_pair"),
    "zone_pair_hour_median": score("p_pair_hour"),
    "lightgbm": score("p_model"),
}

print("\nJUNE TEST RESULTS (minutes)")
print(f"{'predictor':>24} {'MAE':>7} {'RMSE':>7} {'mean err':>9} {'median err':>11}")
for key in ["overall_median", "zone_pair_median", "zone_pair_hour_median", "lightgbm"]:
    r = results[key]
    print(f"{key:>24} {r['mae']:>7.2f} {r['rmse']:>7.2f} {r['mean_error']:>+9.2f} {r['median_error']:>+11.2f}")

gain = 1 - results["lightgbm"]["mae"] / results["zone_pair_hour_median"]["mae"]
print(f"\nmodel MAE is {gain:.1%} lower than the zone-pair + hour baseline")

Path("results").mkdir(exist_ok=True)
Path("results/test_metrics.json").write_text(json.dumps(results, indent=2))
print("saved results/test_metrics.json")