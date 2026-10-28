"""
Core search engine for the "10 closest locations" API.

Idea in one paragraph
---------------------
Every location is a node on a 100 x 100 grid inside the unit square. The link
file lists which neighbouring nodes are joined by a road. "Closest" means the
shortest *road* distance (shortest path through the links), not the
straight-line distance, so we run Dijkstra's algorithm from the node nearest to
the user's position. Dijkstra settles nodes in increasing road distance, so the
first locations it reaches that (a) have the requested category and (b) lie
inside the search circle are exactly the answer, and we can stop early instead
of exploring the whole map. The radius only filters *which* locations may be
returned; a road is still allowed to leave the circle on the way there.
"""

from __future__ import annotations

import csv
import hashlib
import heapq
import math
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

EPS = 1e-9


# --------------------------------------------------------------------------- #
# Locations
# --------------------------------------------------------------------------- #
class LocationStore:
    """Holds the CSV data plus a KD-tree for "which node is nearest to (lat, lon)"."""

    def __init__(self, csv_path: str):
        ids, lats, lons, cats = [], [], [], []
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            # tolerate header case / spacing differences
            cols = {c.strip().lower(): c for c in reader.fieldnames or []}
            c_id = cols.get("id")
            c_lat = cols.get("latitude") or cols.get("lat")
            c_lon = cols.get("longitude") or cols.get("long") or cols.get("lon") or cols.get("lng")
            c_cat = cols.get("category") or cols.get("cat")
            if not all([c_id, c_lat, c_lon, c_cat]):
                raise ValueError(f"CSV must have ID, Latitude, Longitude, Category columns; got {reader.fieldnames}")
            for row in reader:
                ids.append(int(float(row[c_id])))
                lats.append(float(row[c_lat]))
                lons.append(float(row[c_lon]))
                cats.append(row[c_cat].strip().lower())

        self.n = len(ids)
        self.ids = ids                                   # node index -> location ID
        self.lat = lats
        self.lon = lons
        self.cat = cats
        self.id_to_idx = {loc_id: i for i, loc_id in enumerate(ids)}
        self.categories = sorted(set(cats))

        self.coords = np.column_stack([np.array(lats), np.array(lons)])
        self.tree = cKDTree(self.coords)

        # Detect the regular grid so road lengths can be computed exactly in
        # "grid steps" (avoids floating-point noise from the 6-decimal CSV
        # coordinates, which would otherwise break ties between equal routes
        # at random).
        self.grid = self._detect_grid()

    def _detect_grid(self) -> Optional[dict]:
        lat_u = np.unique(np.round(self.coords[:, 0], 6))
        lon_u = np.unique(np.round(self.coords[:, 1], 6))
        if len(lat_u) < 2 or len(lon_u) < 2:
            return None
        step_lat = (lat_u[-1] - lat_u[0]) / (len(lat_u) - 1)
        step_lon = (lon_u[-1] - lon_u[0]) / (len(lon_u) - 1)
        gi = (self.coords[:, 0] - lat_u[0]) / step_lat
        gj = (self.coords[:, 1] - lon_u[0]) / step_lon
        if np.abs(gi - np.rint(gi)).max() > 0.01 or np.abs(gj - np.rint(gj)).max() > 0.01:
            return None  # not a regular grid -> fall back to plain Euclidean edge lengths
        self.grid_origin = (float(lat_u[0]), float(lon_u[0]))
        return {
            "step_lat": float(step_lat),
            "step_lon": float(step_lon),
            "gi": np.rint(gi).astype(int).tolist(),
            "gj": np.rint(gj).astype(int).tolist(),
        }

    def start_nodes(self, lat: float, lon: float, tol: float = 1e-6) -> List[int]:
        """
        Node(s) the user starts from: the location nearest to (lat, lon).
        If the point is exactly equidistant from several locations (e.g. the
        centre of a grid cell such as 0.5, 0.5), all of them are used as
        starting points so the answer doesn't depend on an arbitrary pick.
        """
        d, idx = self.tree.query([lat, lon], k=min(8, self.n))
        d, idx = np.atleast_1d(d), np.atleast_1d(idx)
        return sorted(int(i) for di, i in zip(d, idx) if di <= d[0] + tol)

    def edge_length(self, a: int, b: int) -> float:
        """Length of the road a-b, in grid steps when the data is a grid."""
        if self.grid:
            di = self.grid["gi"][a] - self.grid["gi"][b]
            dj = self.grid["gj"][a] - self.grid["gj"][b]
            return math.hypot(di, dj)  # exactly 1.0 for neighbouring grid points
        return math.hypot(self.lat[a] - self.lat[b], self.lon[a] - self.lon[b])

    def unit_to_coord(self) -> float:
        """Multiply a road length by this to express it in lat/long units."""
        if self.grid and abs(self.grid["step_lat"] - self.grid["step_lon"]) < 1e-12:
            return self.grid["step_lat"]
        if self.grid:  # non-square cells; report in grid steps
            return 1.0
        return 1.0


# --------------------------------------------------------------------------- #
# Road network (built from a link file)
# --------------------------------------------------------------------------- #
@dataclass
class RoadGraph:
    adj: List[List[Tuple[int, float]]]
    n_edges: int
    n_lines: int
    n_skipped: int
    fmt: str


_SPLIT = re.compile(r"[\s,;]+")


def parse_links(text: str, store: LocationStore) -> RoadGraph:
    """
    Accepts both formats seen for this assignment:
      * 4 numbers per line:  lat_a lon_a lat_b lon_b   (the provided link.txt)
      * 2 numbers per line:  id_a id_b                 (as worded in the brief)
    Separators may be spaces, tabs, commas or semicolons. Blank lines, headers
    and comment lines are skipped. Coordinates are snapped to the nearest
    location, so tiny rounding differences don't matter.
    """
    coord_pairs: List[Tuple[float, float, float, float]] = []
    id_pairs: List[Tuple[int, int]] = []
    n_lines = n_skipped = 0

    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        n_lines += 1
        parts = [p for p in _SPLIT.split(line) if p]
        try:
            nums = [float(p) for p in parts]
        except ValueError:
            n_skipped += 1  # header or junk
            continue
        if len(nums) == 4:
            coord_pairs.append((nums[0], nums[1], nums[2], nums[3]))
        elif len(nums) == 2 and all(float(x).is_integer() for x in nums):
            id_pairs.append((int(nums[0]), int(nums[1])))
        else:
            n_skipped += 1

    adj: List[List[Tuple[int, float]]] = [[] for _ in range(store.n)]
    seen = set()

    def add(a: int, b: int):
        if a == b:
            return
        key = (a, b) if a < b else (b, a)
        if key in seen:
            return
        seen.add(key)
        w = store.edge_length(a, b)
        adj[a].append((b, w))  # roads are two-way
        adj[b].append((a, w))

    if coord_pairs:
        arr = np.array(coord_pairs, dtype=float)
        _, ia = store.tree.query(arr[:, :2])
        _, ib = store.tree.query(arr[:, 2:])
        for a, b in zip(ia.tolist(), ib.tolist()):
            add(a, b)

    for a_id, b_id in id_pairs:
        a = store.id_to_idx.get(a_id)
        b = store.id_to_idx.get(b_id)
        if a is None or b is None:
            n_skipped += 1
            continue
        add(a, b)

    fmt = "coordinates" if coord_pairs and not id_pairs else "ids" if id_pairs and not coord_pairs else "mixed"
    return RoadGraph(adj=adj, n_edges=len(seen), n_lines=n_lines, n_skipped=n_skipped, fmt=fmt)


class GraphCache:
    """Parsed road graphs keyed by a hash of the link file contents (small LRU)."""

    def __init__(self, store: LocationStore, max_items: int = 16):
        self.store = store
        self.max_items = max_items
        self._items: "OrderedDict[str, RoadGraph]" = OrderedDict()

    def get(self, text: str) -> Tuple[str, RoadGraph]:
        key = hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()
        g = self._items.get(key)
        if g is None:
            g = parse_links(text, self.store)
            self._items[key] = g
            if len(self._items) > self.max_items:
                self._items.popitem(last=False)
        else:
            self._items.move_to_end(key)
        return key, g


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #
def search(
    store: LocationStore,
    graph: RoadGraph,
    lat: float,
    lon: float,
    category: str,
    radius: float,
    k: int = 10,
    tie: str = "id",
) -> dict:
    """
    Return the k locations of `category` inside the circle of `radius` around
    (lat, lon) that have the smallest road distance from the user.

    Dijkstra with early termination: nodes come off the heap in increasing road
    distance, so once k valid locations are found we only keep going long
    enough to collect anything tied with the k-th one, then stop.

    On a grid many locations share the same road distance, so ties are
    common.  tie="id"     -> equal road distance: smaller ID first (default)
                 tie="euclid" -> equal road distance: straight-line closer first,
                                 then smaller ID
    """
    cat = category.strip().lower()
    r2 = radius * radius + EPS if math.isfinite(radius) else math.inf
    sources = store.start_nodes(lat, lon)

    adj = graph.adj
    S_lat, S_lon, S_cat = store.lat, store.lon, store.cat
    dist = [math.inf] * store.n
    heap = []
    for src in sources:
        dist[src] = 0.0
        heap.append((0.0, src))
    heapq.heapify(heap)
    found: List[Tuple[float, float, int, int]] = []  # (road, euclid^2, id, idx)
    cutoff = math.inf
    settled = 0

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue  # stale heap entry
        if d > cutoff + EPS:
            break
        settled += 1
        if S_cat[u] == cat:
            e2 = (S_lat[u] - lat) ** 2 + (S_lon[u] - lon) ** 2
            if e2 <= r2:
                found.append((d, e2, store.ids[u], u))
                if len(found) == k:
                    cutoff = d  # keep collecting only exact ties
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))

    # Round road distance before sorting so equal routes compare equal
    if store.grid:
        # straight-line tie-break measured in exact grid units (immune to CSV rounding)
        g = store.grid
        lat0, lon0 = store.grid_origin
        qi = round((lat - lat0) / g["step_lat"], 6)
        qj = round((lon - lon0) / g["step_lon"], 6)
        tie_key = lambda u: round((g["gi"][u] - qi) ** 2 + (g["gj"][u] - qj) ** 2, 6)
    else:
        tie_key = lambda u: round((S_lat[u] - lat) ** 2 + (S_lon[u] - lon) ** 2, 12)
    if tie == "euclid":
        found.sort(key=lambda t: (round(t[0], 9), tie_key(t[3]), t[2]))
    else:
        found.sort(key=lambda t: (round(t[0], 9), t[2]))
    top = found[:k]
    scale = store.unit_to_coord()

    return {
        "ids": [t[2] for t in top],
        "start_nodes": [store.ids[i] for i in sources],
        "nodes_explored": settled,
        "results": [
            {
                "rank": i + 1,
                "id": t[2],
                "latitude": S_lat[t[3]],
                "longitude": S_lon[t[3]],
                "category": S_cat[t[3]],
                "road_distance": round(t[0] * scale, 6),
                "road_steps": round(t[0], 6) if store.grid else None,
                "straight_distance": round(math.sqrt(t[1]), 6),
            }
            for i, t in enumerate(top)
        ],
    }
