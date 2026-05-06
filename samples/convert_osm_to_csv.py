
import json
from pathlib import Path
import pandas as pd

INPUT = Path("nova_amenities.geojson")
OUTPUT = Path("pois_nova_real.csv")


def extract_row(feature):
    geom = feature.get("geometry") or {}
    if geom.get("type") != "Point":
        return None
    coords = geom.get("coordinates") or []
    if len(coords) < 2:
        return None

    props = feature.get("properties") or {}
    amenity = props.get("amenity")
    if not amenity:
        return None

    name = props.get("name") or props.get("name:en")
    description = props.get("description") or props.get("note")

    raw_capacity = props.get("capacity")
    try:
        capacity = int(raw_capacity) if raw_capacity is not None else None
    except (ValueError, TypeError):
        capacity = None

    return {
        "osm_id": props.get("@id") or props.get("id") or feature.get("id"),
        "name": name,
        "description": description,
        "category": amenity,
        "amenity_type": amenity,
        "access": props.get("access"),
        "capacity": capacity,
        "rating": None,
        "latitude": coords[1],
        "longitude": coords[0],
    }


with INPUT.open("r", encoding="utf-8") as f:
    data = json.load(f)

rows = []
for feature in data.get("features", []):
    row = extract_row(feature)
    if row is not None:
        rows.append(row)

df = pd.DataFrame(rows)
df = df[df["name"].notna()].reset_index(drop=True)
df.to_csv(OUTPUT, index=False)
print(f"Wrote {len(df)} rows to {OUTPUT}")
print("Top 10 amenity types:")
print(df["amenity_type"].value_counts().head(10).to_string())
