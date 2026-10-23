import numpy as np
import pandas as pd

ZONES = list(range(1, 266))
FEATURES = ["pu_zone", "do_zone", "hour", "dow", "passenger_count"]
MIN_MINUTES = 1.0

class Predictor:
    def __init__(self, booster):
        self.booster = booster

    def predict(self, pu_zone, do_zone, pickup, passenger_count=None):
        row = pd.DataFrame(
            {
                "pu_zone": pd.Categorical([pu_zone], categories=ZONES),
                "do_zone": pd.Categorical([do_zone], categories=ZONES),
                "hour": [pickup.hour],
                "dow": [(pickup.weekday() + 1) % 7],  # Sunday = 0, like DuckDB's dayofweek
                "passenger_count": [np.nan if passenger_count is None else float(passenger_count)],
            }
        )
        minutes = float(self.booster.predict(row[FEATURES])[0])
        return max(minutes, MIN_MINUTES)