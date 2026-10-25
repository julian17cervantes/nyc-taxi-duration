# NYC Taxi Trip Duration

![Tests](https://github.com/julian17cervantes/nyc-taxi-duration/actions/workflows/tests.yml/badge.svg)

Predicts how long a NYC yellow taxi trip will take from what is known at pickup, with a time-based evaluation against non-ML baselines, and serves the model through a FastAPI endpoint.

## Results

**Data:** 20,083,587 yellow taxi trips from January to June 2024, kept from 20,332,093 raw trips (98.8%) after cleaning.

**Split by time, not at random:** train on January to April (12.9M trips), validate on May (3.7M), and test once on June (3.5M). The model and its settings were chosen on May. June was scored one time at the end.

**June test results**, in minutes of absolute error (MAE) and root mean squared error (RMSE):

| Predictor | MAE | RMSE |
| --- | --- | --- |
| Overall median | 9.15 | 14.81 |
| Median per pickup/dropoff zone pair | 5.00 | 8.63 |
| Median per zone pair and hour of day | 4.56 | 8.02 |
| **LightGBM** | **4.04** | **6.83** |

The model's MAE is 11.5% lower than the strongest baseline and 56% lower than predicting one overall median. The baselines are fit on the training months only.

**Model:** LightGBM with an absolute-error objective, trained on the pickup zone, dropoff zone, hour, day of week, and passenger count. It took 4.3 minutes on a laptop. Pickup and dropoff zones account for about 89% of the model's total gain, so treat the feature ranking as rough, since gain tends to favor high-cardinality features.

### Limitations

- The test set is one month. It shows the model holds up on the next month, not that it works all year.
- Every predictor, including the model, runs short on June (mean error -1.77 min for the model). June trips averaged 17.0 minutes versus 15.8 in the training months. I did not test why, so I can't say whether traffic, season, or something else is behind it.
- Trip distance is not a feature, on purpose. It is only known after the trip ends, and using it would leak the answer.
- There is no weather, traffic, or event data.
- Only trips of 1 to 180 minutes are covered. Shorter and longer trips were removed in cleaning (about 1.2% of rows).
- The tests cover the API layer with a fake model. They do not cover the data pipeline, which needs the full dataset.

## How it works

1. `download.py` fetches monthly yellow taxi Parquet files from the NYC TLC.
2. `clean.py` uses DuckDB to filter bad rows and build one dataset of 20M trips.
3. `baseline.py` builds the non-ML baselines on the time-based split.
4. `train.py` trains the LightGBM model and logs the run to a local MLflow database.
5. `evaluate_june.py` scores every predictor on the June test month, once.
6. `main.py` serves `POST /predict`, which validates its input (zone IDs 1 to 265, local New York times only) and returns the predicted minutes.

## Run it yourself

```bash
git clone https://github.com/julian17cervantes/nyc-taxi-duration.git
cd nyc-taxi-duration
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python download.py 2024-01 2024-02 2024-03 2024-04 2024-05 2024-06
python clean.py
python baseline.py
python train.py
python evaluate_june.py
uvicorn main:app
```

Then open http://127.0.0.1:8000/docs and try `POST /predict`:

```json
{
  "pickup_zone": 161,
  "dropoff_zone": 236,
  "pickup_time": "2024-06-05T08:30:00",
  "passenger_count": 1
}
```

The raw data is about 300 MB and the cleaned dataset about 176 MB. Neither is in the repo.

## Tests

```bash
pip install -r requirements-ci.txt
pytest
```

## Data

NYC Taxi and Limousine Commission trip record data. The TLC states that it does not create the data and makes no representation about its accuracy, so cleaning is part of this project.