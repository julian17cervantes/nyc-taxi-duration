import sys
from pathlib import Path

import requests

BASE = "https://d37ci6vzurychx.cloudfront.net/trip-data"
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

months = sys.argv[1:] or ["2024-01"]

for month in months:
    name = f"yellow_tripdata_{month}.parquet"
    path = RAW_DIR / name
    if path.exists():
        print(f"{name}: already dowloaded, skipping")
        continue
    response = requests.get(f"{BASE}/{name}", stream=True, timeout=60)
    if response.status_code != 200:
        print(f"{name}: HTTP {response.status_code}, not downloaded")
        continue
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as out:
        for piece in response.iter_content(chunk_size=1024 * 1024):
            out.write(piece)
    tmp.rename(path)
    print(f"{name}: {path.stat().st_size / 1e6:.0f} MB")