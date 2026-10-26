YEAR = 2026

MONTHS = [f"{YEAR}-{m:02d}" for m in range(1, 7)]  # January to June
RAW_GLOB = f"data/raw/yellow_tripdata_{YEAR}-*.parquet"

START = f"{YEAR}-01-01"
TRAIN_END = f"{YEAR}-05-01"
VAL_END = f"{YEAR}-06-01"
END = f"{YEAR}-07-01"
TEST_MONTH = f"{YEAR}-06"

BASELINE_VAL_MAE = 5.15