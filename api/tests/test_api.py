"""
Run:  python -m pytest -q tests/        (from the project folder)

1. Correctness: 300 random queries compared against an independent brute-force
   answer (scipy all-node shortest paths + filter + sort).
2. Every way of passing the link file gives the same answer.
3. A link file with some roads removed changes the answer (link is really used).
4. Speed.
"""
import http.server
import os
import random
import socketserver
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path

from app.main import DATA_DIR, app, store

client = TestClient(app)
LINK_PATH = os.path.join(DATA_DIR, "link.txt")
LINK_TEXT = open(LINK_PATH).read()


# ---------------------------------------------------------- brute force ----
def build_matrix(link_text):
    rows = []
    for line in link_text.splitlines():
        p = line.split()
        if len(p) == 4:
            a = store.tree.query([float(p[0]), float(p[1])])[1]
            b = store.tree.query([float(p[2]), float(p[3])])[1]
            rows.append((a, b))
    a, b = np.array(rows).T
    return coo_matrix((np.ones(len(a)), (a, b)), shape=(store.n, store.n)).tocsr()


MATRIX = build_matrix(LINK_TEXT)
COORDS = store.coords
CATS = np.array(store.cat)
IDS = np.array(store.ids)
STEP = store.grid["step_lat"]
GI, GJ = np.array(store.grid["gi"]), np.array(store.grid["gj"])


def brute_force(lat, lon, cat, rad, k=10, matrix=MATRIX, tie="id"):
    d0 = np.hypot(COORDS[:, 0] - lat, COORDS[:, 1] - lon)
    sources = np.where(d0 <= d0.min() + 1e-6)[0]
    hops = shortest_path(matrix, directed=False, unweighted=True, indices=sources).min(axis=0)
    ok = (CATS == cat) & (d0 <= rad + 1e-9) & np.isfinite(hops)
    cand = np.where(ok)[0]
    qi, qj = round(lat / STEP, 6), round(lon / STEP, 6)
    tiek = np.round((GI[cand] - qi) ** 2 + (GJ[cand] - qj) ** 2, 6)
    if tie == "euclid":
        order = np.lexsort((IDS[cand], tiek, hops[cand]))
    else:
        order = np.lexsort((IDS[cand], hops[cand]))
    return IDS[cand[order]][:k].tolist(), hops[cand[order]][:k].tolist()


def random_query(rng):
    if rng.random() < 0.5:  # exactly on a grid point
        lat, lon = rng.randrange(100) / 99, rng.randrange(100) / 99
        lat, lon = round(lat, 6), round(lon, 6)
    else:  # anywhere
        lat, lon = round(rng.random(), 4), round(rng.random(), 4)
    return lat, lon, rng.choice(store.categories), round(rng.uniform(0.08, 0.6), 3)


# ------------------------------------------------------------ the tests ----
def test_matches_brute_force():
    rng = random.Random(42)
    checked = 0
    for _ in range(300):
        lat, lon, cat, rad = random_query(rng)
        expected, _ = brute_force(lat, lon, cat, rad)
        if len(expected) < 10:
            continue  # hidden tests guarantee >= 10 valid answers
        r = client.get("/search/", params=dict(lat=lat, long=lon, cat=cat, rad=rad))
        assert r.status_code == 200, r.text
        assert r.json()["ids"] == expected, (lat, lon, cat, rad)
        checked += 1
    assert checked > 200


def test_matches_brute_force_euclid_tiebreak():
    rng = random.Random(99)
    for _ in range(150):
        lat, lon, cat, rad = random_query(rng)
        expected, _ = brute_force(lat, lon, cat, rad, tie="euclid")
        if len(expected) < 10:
            continue
        r = client.get("/search/", params=dict(lat=lat, long=lon, cat=cat, rad=rad, tie="euclid"))
        assert r.json()["ids"] == expected, (lat, lon, cat, rad)


def test_road_distance_differs_from_straight_line():
    """Sanity: in many queries the road answer is NOT the Euclidean answer."""
    rng = random.Random(7)
    differ = 0
    for _ in range(50):
        lat, lon, cat, rad = random_query(rng)
        got = client.get("/search/", params=dict(lat=lat, long=lon, cat=cat, rad=rad)).json()["ids"]
        d0 = np.hypot(COORDS[:, 0] - lat, COORDS[:, 1] - lon)
        idx = np.where((CATS == cat) & (d0 <= rad))[0]
        eu = IDS[idx[np.argsort(d0[idx], kind="stable")]][:10].tolist()
        differ += set(got) != set(eu)
    assert differ > 0


def test_all_results_valid():
    r = client.get("/search/", params=dict(lat=0.3, long=0.7, cat="cafe", rad=0.15)).json()
    assert len(r["ids"]) == 10
    for row in r["results"]:
        assert row["category"] == "cafe"
        assert row["straight_distance"] <= 0.15 + 1e-6
    d = [row["road_distance"] for row in r["results"]]
    assert d == sorted(d)


Q = dict(lat=0.25, long=0.6, cat="hospital", rad=0.2)


def expected_q():
    return brute_force(Q["lat"], Q["long"], Q["cat"], Q["rad"])[0]


@pytest.mark.parametrize("link", [None, "link.txt", "C:\\Users\\prof\\Desktop\\link.txt", "/some/path/link.txt"])
def test_link_by_name_or_default(link):
    params = dict(Q) if link is None else dict(Q, link=link)
    r = client.get("/search/", params=params)
    assert r.status_code == 200 and r.json()["ids"] == expected_q()


def test_link_uploaded_file():
    with open(LINK_PATH, "rb") as f:
        r = client.post("/search/", data={k: str(v) for k, v in Q.items()}, files={"link": ("link.txt", f, "text/plain")})
    assert r.status_code == 200, r.text
    assert r.json()["ids"] == expected_q()
    assert r.json()["link"]["source"].startswith("uploaded")


def test_link_upload_with_other_field_name():
    r = client.post("/search/", data={k: str(v) for k, v in Q.items()},
                    files={"file1": ("roads.txt", LINK_TEXT.encode(), "text/plain")})
    assert r.json()["ids"] == expected_q()


def test_link_inline_text_and_json_body():
    r = client.post("/search/", json=dict(Q, link=LINK_TEXT))
    assert r.json()["ids"] == expected_q()
    assert r.json()["link"]["source"] == "inline text"


def test_link_as_url():
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=DATA_DIR, **kw)

        def log_message(self, *a):
            pass

    with socketserver.TCPServer(("127.0.0.1", 0), Quiet) as httpd:
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        r = client.get("/search/", params=dict(Q, link=f"http://127.0.0.1:{port}/link.txt"))
        httpd.shutdown()
    assert r.status_code == 200, r.text
    assert r.json()["ids"] == expected_q()


def test_link_file_with_ids_format():
    """'a b' per line (IDs) must give the same answer as the coordinate format."""
    lines = []
    for line in LINK_TEXT.splitlines():
        p = list(map(float, line.split()))
        a = store.ids[store.tree.query(p[:2])[1]]
        b = store.ids[store.tree.query(p[2:])[1]]
        lines.append(f"{a} {b}")
    r = client.post("/search/", data={k: str(v) for k, v in Q.items()},
                    files={"link": ("ids.txt", "\n".join(lines).encode(), "text/plain")})
    assert r.json()["link"]["format"] == "ids"
    assert r.json()["ids"] == expected_q()


def test_different_link_file_changes_answer():
    rng = random.Random(1)
    kept = [l for l in LINK_TEXT.splitlines() if rng.random() > 0.25]  # drop 25% of roads
    text = "\n".join(kept)
    exp = brute_force(Q["lat"], Q["long"], Q["cat"], Q["rad"], matrix=build_matrix(text))[0]
    r = client.post("/search/", json=dict(Q, link=text)).json()
    assert r["ids"] == exp
    assert r["ids"] != expected_q()


def test_trailing_slash_optional_and_aliases():
    a = client.get("/search", params=Q).json()["ids"]
    b = client.get("/search/", params=dict(latitude=Q["lat"], longitude=Q["long"],
                                           category=Q["cat"], radius=Q["rad"])).json()["ids"]
    assert a == b == expected_q()


def test_ids_only_format():
    r = client.get("/search/", params=dict(Q, format="ids"))
    assert r.json() == expected_q()


def test_errors():
    assert client.get("/search/", params=dict(lat=0.5, long=0.5)).status_code == 400
    assert client.get("/search/", params=dict(lat="x", long=0.5, cat="bank")).status_code == 400
    assert client.get("/search/", params=dict(lat=0.5, long=0.5, cat="zoo")).status_code == 400


def test_speed():
    rng = random.Random(3)
    qs = [random_query(rng) for _ in range(200)]
    t = time.perf_counter()
    for lat, lon, cat, rad in qs:
        client.get("/search/", params=dict(lat=lat, long=lon, cat=cat, rad=rad))
    per_query_ms = (time.perf_counter() - t) / len(qs) * 1000
    print(f"\naverage {per_query_ms:.2f} ms per request (including HTTP layer)")
    assert per_query_ms < 100
