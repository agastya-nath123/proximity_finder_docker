# Proximity Search API - Docker setup

One container: the FastAPI app served by Uvicorn on Python 3.12.

## Run

Install Docker Desktop (Windows / Mac) or Docker Engine + Compose plugin (Linux), then in this folder:

```
docker compose up --build
```

- API: http://localhost:8000/search/?lat=0.5&long=0.5&cat=bank&rad=0.2
- Interactive docs (with file upload): http://localhost:8000/docs
- Health: http://localhost:8000/health

Stop with Ctrl+C, then `docker compose down`.

## Settings (.env)

- `API_PORT` host port for the API (default 8000). Change it if 8000 is already used.
- `TIE_BREAK` `id` or `euclid`, how equal road distances are ordered.

## Data

`data/` is mounted into the container read-only at `/data`.
Replace `locations.csv` or `link.txt` and run `docker compose restart api`, no rebuild needed.

## Tests (outside Docker)

```
cd api
pip install -r requirements.txt pytest httpx
python -m pytest -q tests/
```
# proximity_finder_docker
