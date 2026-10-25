FROM python:3.13-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-runtime.txt .
RUN pip install --no-cache-dir -r requirements-runtime.txt

COPY main.py predictor.py ./

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]