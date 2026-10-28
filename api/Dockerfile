# ---------- Proximity Search API (FastAPI + Uvicorn) ----------
# Small official Python image (Debian slim). Python 3.12 is what the API was built and tested on.
FROM python:3.12-slim

# No .pyc files, logs printed straight away, no pip cache inside the image,
# and the folder where the API looks for locations.csv and link.txt.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/data

WORKDIR /app

# Install libraries first, in their own layer, so Docker can reuse this layer
# when only the code changes (much faster rebuilds).
COPY requirements.txt .
RUN pip install -r requirements.txt

# Now copy the application code
COPY app ./app

# Run as a normal user instead of root
RUN useradd --create-home --uid 10001 appuser
USER appuser

# The port Uvicorn listens on inside the container
EXPOSE 8000

# Docker checks this every 10 s and shows the container as healthy or unhealthy.
# python:slim has no curl, so Python itself is used to call /health.
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
