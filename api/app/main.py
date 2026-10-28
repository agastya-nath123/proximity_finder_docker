"""
HTTP layer.  Route:  /search/   (also /search)   — GET or POST

Fields (names required by the brief, common aliases also accepted):
    lat   current latitude                       (also: latitude)
    long  current longitude                      (also: lon, lng, longitude)
    cat   category, e.g. "bank", "cafe"          (also: category)
    rad   search radius, straight-line           (also: radius)
    link  road linkage .txt  — any of:
            * an uploaded file (multipart/form-data, POST)
            * a URL to a .txt file (http/https, Google Drive share links ok)
            * a file name already on the server, e.g. "link.txt"
            * the file contents pasted directly as text
            * omitted -> the link.txt shipped with the assignment
    k     optional, how many results (default 10)
    tie   optional, how to order locations at the same road distance:
          "id" (smaller ID first, default) or "euclid" (straight-line closer first)
    format optional, "ids" to get a bare JSON array of IDs

Response (default):  {"ids": [10 location IDs], "results": [...details...], ...}
"""

from __future__ import annotations

import math
import os
import re
import time
import urllib.request
from collections import OrderedDict
from typing import Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile

from .engine import GraphCache, LocationStore, search

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))
LOCATIONS_CSV = os.environ.get("LOCATIONS_CSV", os.path.join(DATA_DIR, "locations.csv"))
DEFAULT_LINK = os.environ.get("DEFAULT_LINK", os.path.join(DATA_DIR, "link.txt"))
MAX_LINK_BYTES = 50 * 1024 * 1024

# ----------------------------------------------------------------- startup --
store = LocationStore(LOCATIONS_CSV)
graphs = GraphCache(store)
with open(DEFAULT_LINK, encoding="utf-8", errors="ignore") as _f:
    DEFAULT_LINK_TEXT = _f.read()
graphs.get(DEFAULT_LINK_TEXT)  # parse once at boot so the first query is fast
_url_cache: "OrderedDict[str, str]" = OrderedDict()

app = FastAPI(
    title="Nearest Locations API",
    description="Returns the 10 closest locations by road (grid) distance, filtered by category and straight-line radius.",
    version="1.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ----------------------------------------------------------------- helpers --
ALIASES = {
    "lat": ("lat", "latitude", "current_lat", "curr_lat"),
    "long": ("long", "lon", "lng", "longitude", "current_long", "curr_long"),
    "cat": ("cat", "category", "type"),
    "rad": ("rad", "radius", "r", "search_radius"),
    "link": ("link", "links", "link_file", "linkage", "linkfile", "file"),
    "k": ("k", "n", "top", "limit"),
    "format": ("format", "fmt", "output"),
    "tie": ("tie", "tiebreak", "tie_break"),
}
DEFAULT_TIE = os.environ.get("TIE_BREAK", "id").lower()


class BadRequest(Exception):
    pass


def _pick(params: dict, name: str):
    lower = {str(k).lower(): v for k, v in params.items()}
    for alias in ALIASES[name]:
        if alias in lower and lower[alias] not in (None, ""):
            return lower[alias]
    return None


def _as_float(value, name: str, required: bool = True) -> Optional[float]:
    if value is None:
        if required:
            raise BadRequest(f"Missing required field '{name}'")
        return None
    try:
        x = float(str(value).strip())
    except ValueError:
        raise BadRequest(f"Field '{name}' must be a number, got {value!r}")
    if math.isnan(x):
        raise BadRequest(f"Field '{name}' must be a number, got NaN")
    return x


def _gdrive_direct(url: str) -> str:
    m = re.search(r"drive\.google\.com/(?:file/d/|open\?id=|uc\?(?:.*&)?id=)([\w-]+)", url)
    return f"https://drive.google.com/uc?export=download&id={m.group(1)}" if m else url


def _download(url: str) -> str:
    if url in _url_cache:
        _url_cache.move_to_end(url)
        return _url_cache[url]
    req = urllib.request.Request(_gdrive_direct(url), headers={"User-Agent": "nearest-api/1.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = resp.read(MAX_LINK_BYTES + 1)
    if len(data) > MAX_LINK_BYTES:
        raise BadRequest("Link file is too large (max 50 MB)")
    text = data.decode("utf-8", errors="ignore")
    _url_cache[url] = text
    if len(_url_cache) > 16:
        _url_cache.popitem(last=False)
    return text


def _looks_like_link_data(s: str) -> bool:
    return bool(re.match(r"^\s*-?\d+(\.\d+)?([\s,;]+-?\d+(\.\d+)?){1,}", s))


async def resolve_link(value) -> Tuple[str, str, Optional[str]]:
    """Returns (link_text, description_of_source, warning_or_None)."""
    if value is None:
        return DEFAULT_LINK_TEXT, "default link.txt", None

    if isinstance(value, UploadFile):
        data = await value.read()
        if len(data) > MAX_LINK_BYTES:
            raise BadRequest("Link file is too large (max 50 MB)")
        return data.decode("utf-8", errors="ignore"), f"uploaded file '{value.filename}'", None

    s = str(value).strip()
    if not s:
        return DEFAULT_LINK_TEXT, "default link.txt", None

    if re.match(r"^https?://", s, re.I):
        try:
            return await run_in_threadpool(_download, s), f"url {s}", None
        except BadRequest:
            raise
        except Exception as e:  # network problem
            raise BadRequest(f"Could not download link file from {s}: {e}")

    # A file name / path: look it up (by base name) in the server's data folder
    if "\n" not in s and len(s) < 300 and not _looks_like_link_data(s):
        name = os.path.basename(s.replace("\\", "/"))
        path = os.path.join(DATA_DIR, name)
        if name and os.path.isfile(path):
            with open(path, encoding="utf-8", errors="ignore") as f:
                return f.read(), f"server file '{name}'", None
        return (
            DEFAULT_LINK_TEXT,
            "default link.txt",
            f"Link '{s}' is not a URL, uploaded file, or file on this server; used the default link.txt instead.",
        )

    # Otherwise treat the value itself as the file contents
    return s, "inline text", None


async def collect_params(request: Request) -> dict:
    params = dict(request.query_params)
    if request.method == "POST":
        ctype = request.headers.get("content-type", "")
        if "application/json" in ctype:
            try:
                body = await request.json()
                if isinstance(body, dict):
                    params.update(body)
            except Exception:
                raise BadRequest("Request body is not valid JSON")
        elif "multipart/form-data" in ctype or "application/x-www-form-urlencoded" in ctype:
            form = await request.form()
            uploads = []
            for key, val in form.multi_items():
                if isinstance(val, UploadFile):
                    if val.filename or (val.size or 0) > 0:  # ignore "no file chosen"
                        uploads.append((key, val))
                else:
                    params[key] = val
            # an uploaded file is the link file whatever its field name is
            if uploads:
                named = [u for k, u in uploads if k.lower() in ALIASES["link"]]
                params["link"] = named[0] if named else uploads[0][1]
        else:
            raw = (await request.body()).decode("utf-8", errors="ignore")
            if raw.strip() and _pick(params, "link") is None:
                params["link"] = raw  # raw text body = link file contents
    return params


# ------------------------------------------------------------------ routes --
SEARCH_DOC = {
    "parameters": [
        {"name": "lat", "in": "query", "required": True, "schema": {"type": "number"}, "example": 0.5},
        {"name": "long", "in": "query", "required": True, "schema": {"type": "number"}, "example": 0.5},
        {"name": "cat", "in": "query", "required": True, "schema": {"type": "string"}, "example": "bank"},
        {"name": "rad", "in": "query", "required": False, "schema": {"type": "number"}, "example": 0.2},
        {"name": "link", "in": "query", "required": False, "schema": {"type": "string"}, "example": "link.txt",
         "description": "URL, server file name, or file contents. Omit to use the provided link.txt. "
                        "To upload a file, POST multipart/form-data with field 'link'."},
        {"name": "k", "in": "query", "required": False, "schema": {"type": "integer", "default": 10}},
        {"name": "tie", "in": "query", "required": False, "schema": {"type": "string", "enum": ["id", "euclid"], "default": "id"}},
    ]
}


# POST form shown in /docs: gives a "Choose File" button for the link file
UPLOAD_DOC = {
    "summary": "Search (upload a link file)",
    "description": "Fill in the fields and choose your link .txt file. "
                   "If no file is chosen, the provided link.txt is used.",
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["lat", "long", "cat"],
                    "properties": {
                        "lat": {"type": "number", "example": 0.5},
                        "long": {"type": "number", "example": 0.5},
                        "cat": {"type": "string", "example": "bank",
                                "enum": store.categories},
                        "rad": {"type": "number", "example": 0.2},
                        "link": {"type": "string", "format": "binary",
                                 "contentMediaType": "application/octet-stream",
                                 "description": "Road linkage .txt file"},
                        "k": {"type": "integer", "default": 10},
                        "tie": {"type": "string", "enum": ["id", "euclid"], "default": "id"},
                    },
                }
            }
        },
    },
}


@app.get("/search/", summary="Search (link as name / URL / text)", openapi_extra=SEARCH_DOC)
@app.post("/search/", openapi_extra=UPLOAD_DOC)
@app.api_route("/search", methods=["GET", "POST"], include_in_schema=False)
async def search_endpoint(request: Request):
    t0 = time.perf_counter()
    try:
        params = await collect_params(request)
        lat = _as_float(_pick(params, "lat"), "lat")
        lon = _as_float(_pick(params, "long"), "long")
        rad = _as_float(_pick(params, "rad"), "rad", required=False)
        rad = math.inf if rad is None else rad
        if rad < 0:
            raise BadRequest("Field 'rad' must be >= 0")
        cat = _pick(params, "cat")
        if cat is None:
            raise BadRequest("Missing required field 'cat'")
        cat = str(cat).strip().lower()
        if cat not in store.categories:
            raise BadRequest(f"Unknown category '{cat}'. Valid categories: {store.categories}")
        k_raw = _pick(params, "k")
        k = 10 if k_raw is None else int(_as_float(k_raw, "k"))
        if not 1 <= k <= 1000:
            raise BadRequest("Field 'k' must be between 1 and 1000")

        link_text, link_source, warning = await resolve_link(_pick(params, "link"))
        _, graph = graphs.get(link_text)
        if graph.n_edges == 0:
            raise BadRequest("No usable links found in the link file. Expected lines like "
                             "'lat_a lon_a lat_b lon_b' or 'id_a id_b'.")

        tie = str(_pick(params, "tie") or DEFAULT_TIE).lower()
        if tie not in ("id", "euclid"):
            raise BadRequest("Field 'tie' must be 'id' or 'euclid'")

        result = search(store, graph, lat, lon, cat, rad, k, tie)
    except BadRequest as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    if str(_pick(params, "format") or "").lower() in ("ids", "list", "plain"):
        return JSONResponse(content=result["ids"])

    body = {
        "ids": result["ids"],
        "count": len(result["ids"]),
        "query": {"lat": lat, "long": lon, "cat": cat, "rad": None if math.isinf(rad) else rad, "k": k, "tie": tie},
        "start_nodes": result["start_nodes"],
        "link": {"source": link_source, "roads": graph.n_edges, "format": graph.fmt},
        "nodes_explored": result["nodes_explored"],
        "time_ms": round((time.perf_counter() - t0) * 1000, 2),
        "results": result["results"],
    }
    if len(result["ids"]) < k:
        body["note"] = f"Only {len(result['ids'])} locations of '{cat}' are reachable inside radius {rad}."
    if warning:
        body["warning"] = warning
    return body


@app.get("/")
def root():
    return {
        "service": "Nearest Locations API",
        "usage": "GET /search/?lat=0.5&long=0.5&cat=bank&rad=0.2&link=link.txt",
        "upload_link_file": "POST /search/ as multipart/form-data with fields lat, long, cat, rad and file field 'link'",
        "categories": store.categories,
        "locations": store.n,
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok", "locations": store.n}
