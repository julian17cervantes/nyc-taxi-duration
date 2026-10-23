from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field, NaiveDatetime

from predictor import Predictor

class TripRequest(BaseModel):
    pickup_zone: int = Field(ge=1, le=265, description="TLC taxi zone ID")
    dropoff_zone: int = Field(ge=1, le=265, description="TLC taxi zone ID")
    pickup_time: NaiveDatetime = Field(description="Local New York time, without a timezone")
    passenger_count: Optional[int] = Field(default=None, ge=0, le=9)


@asynccontextmanager
async def lifespan(app):
    import lightgbm as lgb

    app.state.predictor = Predictor(lgb.Booster(model_file="models/lgbm.txt"))
    yield

app = FastAPI(title="NYC Taxi Trip Duration", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(trip: TripRequest, request: Request):
    minutes = request.app.state.predictor.predict(
        trip.pickup_zone, trip.dropoff_zone, trip.pickup_time, trip.passenger_count
    )
    return {"predicted_minutes": round(minutes, 1)}