from datetime import datetime

import numpy as np

from predictor import Predictor

TRIP = {
    "pickup_zone": 161,
    "dropoff_zone": 236,
    "pickup_time": "2024-06-05T08:30:00",
    "passenger_count": 1,
}


def test_day_of_week_matches_training_convention(booster):
    predictor = Predictor(booster)
    predictor.predict(1, 2, datetime(2024, 6, 5, 8, 30))  # a Wednesday
    assert booster.last_input["dow"].iloc[0] == 3
    predictor.predict(1, 2, datetime(2024, 6, 9, 8, 30))  # a Sunday
    assert booster.last_input["dow"].iloc[0] == 0


def test_features_are_in_training_order(booster):
    Predictor(booster).predict(10, 20, datetime(2024, 6, 5, 8, 30), 2)
    frame = booster.last_input
    assert list(frame.columns) == ["pu_zone", "do_zone", "hour", "dow", "passenger_count"]
    assert frame["pu_zone"].iloc[0] == 10
    assert frame["do_zone"].iloc[0] == 20
    assert frame["hour"].iloc[0] == 8
    assert frame["passenger_count"].iloc[0] == 2


def test_missing_passenger_count_becomes_nan(booster):
    Predictor(booster).predict(10, 20, datetime(2024, 6, 5, 8, 30))
    assert np.isnan(booster.last_input["passenger_count"].iloc[0])


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_predict(client):
    response = client.post("/predict", json=TRIP)
    assert response.status_code == 200
    assert response.json() == {"predicted_minutes": 12.3}


def test_predictions_are_at_least_one_minute(client, booster):
    booster.value = 0.2
    assert client.post("/predict", json=TRIP).json()["predicted_minutes"] == 1.0


def test_passenger_count_is_optional(client):
    trip = {k: v for k, v in TRIP.items() if k != "passenger_count"}
    assert client.post("/predict", json=trip).status_code == 200


def test_zone_out_of_range_is_rejected(client):
    assert client.post("/predict", json={**TRIP, "pickup_zone": 0}).status_code == 422
    assert client.post("/predict", json={**TRIP, "dropoff_zone": 266}).status_code == 422


def test_timezone_aware_time_is_rejected(client):
    trip = {**TRIP, "pickup_time": "2024-06-05T08:30:00+00:00"}
    assert client.post("/predict", json=trip).status_code == 422


def test_missing_field_is_rejected(client):
    trip = {k: v for k, v in TRIP.items() if k != "pickup_time"}
    assert client.post("/predict", json=trip).status_code == 422