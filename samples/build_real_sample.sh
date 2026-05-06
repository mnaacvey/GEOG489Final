#!/bin/bash
set -e
osmium extract --bbox -77.36,38.66,-77.00,38.96 --strategy complete_ways virginia-260424.osm.pbf -o nova.osm.pbf
osmium tags-filter nova.osm.pbf n/amenity -o nova_amenities.osm.pbf
osmium export --add-unique-id=type_id nova_amenities.osm.pbf -o nova_amenities.geojson
python3 convert_osm_to_csv.py
rm virginia-260424.osm.pbf nova.osm.pbf nova_amenities.osm.pbf nova_amenities.geojson