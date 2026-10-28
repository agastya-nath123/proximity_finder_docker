import os

# Tests run outside Docker, so point the API at the shared data folder.
os.environ.setdefault("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "data"))
