# NYC Taxi Trip Duration

## Run with Docker

The image contains the code and libraries only. The trained model is not in the repo, so train it first (`python train.py`) and mount it into the container:

```bash
docker build -t nyc-taxi-duration .
docker run --rm -p 8000:8000 -v "$(pwd)/models:/app/models:ro" nyc-taxi-duration
```

Then open http://127.0.0.1:8000/docs.

![Tests](https://github.com/julian17cervantes/nyc-taxi-duration/actions/workflows/tests.yml/badge.svg)

Predicts how long a NYC yellow taxi trip will take from what is known at pickup, with a time-based evaluation against non-ML baselines, and serves the model through a FastAPI endpoint.

## Results

**Data:** 22,308,570 yellow taxi trips from January to June 2026, kept from 22,836,530 raw trips (97.7%) after cleaning.

**Split by time, not at random:** train on January to April (14.6M trips), validate on May (4.0M), and test once on June (3.7M). The model settings come from an earlier run on 2024 data and were not retuned for 2026. June was scored one time at the end.

**June 2026 test results**, in minutes of absolute error (MAE) and root mean squared error (RMSE):

| Predictor | MAE | RMSE |
| --- | --- | --- |
| Overall median | 8.97 | 13.60 |
| Median per pickup/dropoff zone pair | 4.99 | 8.01 |
| Median per zone pair and hour of day | 4.51 | 7.32 |
| **LightGBM** | **4.05** | **6.55** |

The model's MAE is 10.2% lower than the strongest baseline and about 55% lower than predicting one overall median. The baselines are fit on the training months only. The lead on the validation month (May) was larger, 14.5%, so the June figure is the one to trust.

**Earlier run, same settings, January to June 2024 data:** 4.04 min MAE versus 4.56 for the strongest baseline (11.5% lower).

**Model:** LightGBM with an absolute-error objective, trained on the pickup zone, dropoff zone, hour, day of week, and passenger count. Training took 4.6 minutes on a laptop. Pickup and dropoff zones account for about 89% of the model's total gain, so treat the feature ranking as rough, since gain tends to favor high-cardinality features.

### Limitations

- The test set is one month per year. It shows the model holds up on the next month, not that it works all year.
- Every predictor runs short on average, including the baselines (mean error -1.71 min for the model). The predictors estimate a typical trip, and trip durations have a long right tail, which pulls the average above the typical value. June's average duration (17.7 min) was close to the training months' (17.6 min), so a month-to-month shift does not explain it. I did not test this directly.
- The data changed between years. In January 2026, VendorID 7 reported 44,705 trips that all had zero duration, and the cleaning rules remove them. About 29% of January 2026 passenger counts were missing, up from 5% in January 2024. I only checked January, so I do not know how widespread this is in other months.
- Trip distance is not a feature, on purpose. It is only known after the trip ends, and using it would leak the answer.
- There is no weather, traffic, or event data.
- Only trips of 1 to 180 minutes are covered. Cleaning removed 2.3% of raw rows, mostly trips under one minute.
- The tests cover the API layer with a fake model. They do not cover the data pipeline, which needs the full dataset.

## How it works

1. `download.py` fetches monthly yellow taxi Parquet files from the NYC TLC.
2. `clean.py` uses DuckDB to filter bad rows and build one dataset of 22M trips.
3. `baseline.py` builds the non-ML baselines on the time-based split.
4. `train.py` trains the LightGBM model and logs the run to a local MLflow database.
5. `evaluate_june.py` scores every predictor on the June test month, once.
6. `main.py` serves `POST /predict`, which validates its input (zone IDs 1 to 265, local New York times only) and returns the predicted minutes.

## Run it yourself

The period is set in `config.py`: change `YEAR` there to run the whole pipeline on another year.The period is set in `config.py`: change `YEAR` there to run the whole pipeline on another year.

```bash
git clone https://github.com/julian17cervantes/nyc-taxi-duration.git
cd nyc-taxi-duration
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python download.py 2026-01 2026-02 2026-03 2026-04 2026-05 2026-06
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
  "pickup_time": "2026-06-03T08:30:00",
  "passenger_count": 1
}
```

The raw data is about 372 MB and the cleaned dataset about 196 MB. Neither is in the repo.

## Tests

```bash
pip install -r requirements-ci.txt
pytest
```

## Data

NYC Taxi and Limousine Commission trip record data. The TLC states that it does not create the data and makes no representation about its accuracy, so cleaning is part of this project.