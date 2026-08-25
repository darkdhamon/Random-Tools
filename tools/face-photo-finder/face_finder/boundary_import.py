from __future__ import annotations

import argparse
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import requests
import shapefile
import truststore

from .catalog import FaceCatalog, default_catalog_path

truststore.inject_into_ssl()


US_FILES = {
    "country": "cb_2025_us_nation_5m.zip",
    "state": "cb_2025_us_state_500k.zip",
    "county": "cb_2025_us_county_500k.zip",
    "city": "cb_2025_us_place_500k.zip",
}
US_ROOT = "https://www2.census.gov/geo/tiger/GENZ2025/shp/"
CANADA_CITIES = "https://geo.statcan.gc.ca/geo_wa/rest/services/2025/lcsd000a25s_e/MapServer/0/query"
MEXICO_STATES = "https://lcidsig.inegi.org.mx/server/rest/services/Hosted/Entidades_federativas_2025/FeatureServer/0/query"
MEXICO_MUNICIPALITIES = "https://lcidsig.inegi.org.mx/server/rest/services/Hosted/Municipios_2025/FeatureServer/0/query"


def _download(url: str, destination: Path) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with destination.open("wb") as output:
            for chunk in response.iter_content(1024 * 1024):
                output.write(chunk)


def _shape_geometry(shape: shapefile.Shape) -> dict:
    points = shape.points
    parts = list(shape.parts) + [len(points)]
    rings = [[[float(x), float(y)] for x, y in points[parts[i]:parts[i + 1]]]
             for i in range(len(parts) - 1)]
    # Shapefile outer rings are clockwise. Group following counter-clockwise holes.
    polygons: list[list[list[list[float]]]] = []
    for ring in rings:
        area = sum((ring[i + 1][0] - ring[i][0]) * (ring[i + 1][1] + ring[i][1])
                   for i in range(len(ring) - 1))
        if area >= 0 or not polygons:
            polygons.append([ring])
        else:
            polygons[-1].append(ring)
    return {"type": "MultiPolygon", "coordinates": polygons}


def _import_us(catalog: FaceCatalog, cache: Path) -> int:
    imported = 0
    parent_ids: dict[str, int] = {}
    for level, filename in US_FILES.items():
        archive = cache / filename
        _download(US_ROOT + filename, archive)
        folder = cache / filename.removesuffix(".zip")
        if not folder.exists():
            with zipfile.ZipFile(archive) as source:
                source.extractall(folder)
        shp = next(folder.glob("*.shp"))
        reader = shapefile.Reader(str(shp), encoding="utf-8")
        fields = [field[0] for field in reader.fields[1:]]
        for item in reader.iterShapeRecords():
            record = dict(zip(fields, item.record))
            geoid = str(record.get("GEOID") or record.get("AFFGEOID") or "US")
            key = f"US:{level}:{geoid}"
            parent = None
            if level == "state": parent = parent_ids.get("US")
            elif level == "county": parent = parent_ids.get(str(record.get("STATEFP")))
            elif level == "city": parent = parent_ids.get(str(record.get("STATEFP")))
            location_id = catalog.upsert_imported_location(
                key, str(record.get("NAME") or "United States"), level,
                _shape_geometry(item.shape), "U.S. Census Bureau TIGER/Line 2025", parent,
            )
            imported += 1
            if level == "country": parent_ids["US"] = location_id
            elif level == "state": parent_ids[str(record.get("STATEFP"))] = location_id
    return imported


def _arcgis_features(url: str, out_fields: str) -> list[dict]:
    features, offset = [], 0
    while True:
        response = requests.get(url, params={"where": "1=1", "outFields": out_fields,
            "returnGeometry": "true", "outSR": 4326, "f": "geojson",
            "resultOffset": offset, "resultRecordCount": 1000,
            "geometryPrecision": 5, "maxAllowableOffset": .0001}, timeout=180)
        response.raise_for_status()
        page = response.json().get("features", [])
        features.extend(page)
        if len(page) < 1000: return features
        offset += len(page)


def _aggregate(features: list[dict]) -> dict:
    polygons = []
    for feature in features:
        geometry = feature["geometry"]
        polygons.extend(geometry["coordinates"] if geometry["type"] == "MultiPolygon"
                        else [geometry["coordinates"]])
    return {"type": "MultiPolygon", "coordinates": polygons}


def _import_canada(catalog: FaceCatalog) -> int:
    features = _arcgis_features(CANADA_CITIES, "*")
    if not features: return 0
    country = catalog.upsert_imported_location("CA:country:CA", "Canada", "country",
        _aggregate(features), "Statistics Canada 2025")
    provinces: dict[str, list[dict]] = defaultdict(list)
    divisions: dict[str, list[dict]] = defaultdict(list)
    for feature in features:
        p = feature["properties"]
        provinces[str(p.get("PRUID"))].append(feature)
        divisions[str(p.get("CDUID"))].append(feature)
    province_ids = {code: catalog.upsert_imported_location(f"CA:state:{code}",
        str(items[0]["properties"].get("PRNAME") or code), "state", _aggregate(items),
        "Statistics Canada 2025", country) for code, items in provinces.items()}
    division_ids = {code: catalog.upsert_imported_location(f"CA:county:{code}",
        str(items[0]["properties"].get("CDNAME") or code), "county", _aggregate(items),
        "Statistics Canada 2025", province_ids.get(str(items[0]["properties"].get("PRUID"))))
        for code, items in divisions.items()}
    for feature in features:
        p = feature["properties"]; code = str(p.get("CSDUID"))
        catalog.upsert_imported_location(f"CA:city:{code}", str(p.get("CSDNAME") or code),
            "city", feature["geometry"], "Statistics Canada 2025",
            division_ids.get(str(p.get("CDUID"))))
    return 1 + len(province_ids) + len(division_ids) + len(features)


def _import_mexico(catalog: FaceCatalog) -> int:
    states = _arcgis_features(MEXICO_STATES, "*")
    municipalities = _arcgis_features(MEXICO_MUNICIPALITIES, "*")
    country = catalog.upsert_imported_location("MX:country:MX", "Mexico", "country",
        _aggregate(states), "INEGI Marco Geoestadistico 2025")
    state_ids = {}
    for feature in states:
        p = feature["properties"]; code = str(p.get("cvegeo") or p.get("CVEGEO"))
        state_ids[code] = catalog.upsert_imported_location(f"MX:state:{code}",
            str(p.get("nomgeo") or p.get("NOMGEO") or code), "state", feature["geometry"],
            "INEGI Marco Geoestadistico 2025", country)
    for feature in municipalities:
        p = feature["properties"]; code = str(p.get("cvegeo") or p.get("CVEGEO"))
        state = code[:2]
        catalog.upsert_imported_location(f"MX:county:{code}",
            str(p.get("nomgeo") or p.get("NOMGEO") or code), "county", feature["geometry"],
            "INEGI Marco Geoestadistico 2025", state_ids.get(state))
    return 1 + len(states) + len(municipalities)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import official North American administrative boundaries.")
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "face-photo-finder" / "boundaries")
    parser.add_argument("--countries", nargs="+", choices=("us", "ca", "mx"), default=("us", "ca", "mx"))
    args = parser.parse_args()
    catalog = FaceCatalog(default_catalog_path())
    try:
        with catalog.connection:
            catalog.connection.execute("DELETE FROM photo_imported_location_matches")
            catalog.connection.execute("DELETE FROM photo_boundary_scans")
        counts = {}
        if "us" in args.countries: counts["United States"] = _import_us(catalog, args.cache)
        if "ca" in args.countries: counts["Canada"] = _import_canada(catalog)
        if "mx" in args.countries: counts["Mexico"] = _import_mexico(catalog)
        catalog.rebuild_boundary_match_cache()
    finally:
        catalog.close()
    print(json.dumps(counts, indent=2))
