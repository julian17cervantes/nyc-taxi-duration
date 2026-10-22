import sys
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

DATA = "data/clean/trips.parquet"
TRAIN_END = "2024-05-01"
VAL_END = "2024-06-01"
TRAIN_SAMPLE = int(sys.argv[1]) if len(sys.argv) > 1 else None
BASELINE_MAE = 4.75

FEATURES = ["pu_zone", "do_zone", "hour", "dow", "passenger_count"]
ZONES = list(range(1, 266))
PARAMS = dict(
    objective="l1",
    n_estimators=1500,
    learning_rate=0.1,
    num_leaves=127,
    min_child_samples=200,
)

con = duckdb.connect()


def load(start, end):
    return con.execute(
        f"""
        SELECT pu_zone, do_zone, hour(pickup) AS hour, dayofweek(pickup) AS dow,
               passenger_count, minutes
        FROM read_parquet('{DATA}')
        WHERE pickup >= TIMESTAMP '{start}' AND pickup < TIMESTAMP '{end}'
        """
    ).df()


def prep(df):
    for col in ["pu_zone", "do_zone"]:
        df[col] = pd.Categorical(df[col], categories=ZONES)
    return df


train = load("2024-01-01", TRAIN_END)
val = load(TRAIN_END, VAL_END)
if TRAIN_SAMPLE:
    train = train.sample(TRAIN_SAMPLE, random_state=0)
train, val = prep(train), prep(val)
print(f"training rows: {len(train):,}   validation rows: {len(val):,}")

mlflow.set_experiment("taxi-duration")
with mlflow.start_run():
    mlflow.log_params(
        {
            **PARAMS,
            "features": ",".join(FEATURES),
            "train_rows": len(train),
            "val_rows": len(val),
            "train_end": TRAIN_END,
        }
    )

    model = lgb.LGBMRegressor(**PARAMS, n_jobs=-1, verbose=-1)
    start = time.time()
    model.fit(
        train[FEATURES],
        train["minutes"],
        eval_set=[(val[FEATURES], val["minutes"])],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(100)],
    )
    train_minutes = (time.time() - start) / 60
    print(f"trained in {train_minutes:.1f} min, best round {model.best_iteration_}")

    pred = model.predict(val[FEATURES])
    error = pred - val["minutes"].to_numpy()
    metrics = {
        "val_mae": float(np.abs(error).mean()),
        "val_rmse": float(np.sqrt((error**2).mean())),
        "val_mean_error": float(error.mean()),
        "val_median_error": float(np.median(error)),
        "baseline_val_mae": BASELINE_MAE,
        "train_minutes": train_minutes,
        "best_round": int(model.best_iteration_),
    }
    mlflow.log_metrics(metrics)

    print("\nVALIDATION (May)")
    print(f"model MAE {metrics['val_mae']:.2f} min   RMSE {metrics['val_rmse']:.2f} min")
    print(f"mean error (positive = predicts too long): {metrics['val_mean_error']:+.2f} min")
    print(f"median error: {metrics['val_median_error']:+.2f} min")
    print(f"baseline MAE {BASELINE_MAE:.2f} min -> model is {1 - metrics['val_mae'] / BASELINE_MAE:+.1%} lower")

    gain = model.booster_.feature_importance(importance_type="gain")
    print("\nFEATURE IMPORTANCE (share of total gain)")
    for name, share in sorted(zip(FEATURES, gain / gain.sum()), key=lambda x: -x[1]):
        print(f"  {name:>16}: {share:.1%}")

    Path("models").mkdir(exist_ok=True)
    model.booster_.save_model("models/lgbm.txt")
    mlflow.log_artifact("models/lgbm.txt")
    print("\nsaved models/lgbm.txt")