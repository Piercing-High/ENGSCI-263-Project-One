"""
Foodstuffs route map — following the course "Routing in Python" template
========================================================================

Draws the optimiser's selected delivery routes on an interactive Auckland
map, each truck's trip following the road network via OpenRouteService.

Follows the conventions from the course resource:
  - coords read as [Long, Lat], reversed to [Lat, Long] for folium
  - folium.Marker + folium.Icon for store pins
  - store colours: Four Square = green, New World = red,
                   Pak 'n Save = orange, Warehouse = black
  - ORS profile = 'driving-hgv' (heavy goods vehicle / truck)
  - client.directions(..., format='geojson', validate=False)

------------------------------------------------------------------
SETUP (one-time)
------------------------------------------------------------------
  pip install openrouteservice folium pandas numpy

  Get a free key at https://openrouteservice.org (dashboard -> request token)
  and paste it into ORSkey below.

  Run:  python "Visualising Routes/draw_routes_map.py"
  Open the generated .html in a browser.
------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import folium
import openrouteservice as ors
import time
from pathlib import Path


# ============================================================
# PATHS -- script lives in "Visualising Routes/", repo root one level up
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCATIONS_CSV = PROJECT_ROOT / "Resources" / "FoodstuffsLocations.csv"
SELECTED_XLSX = PROJECT_ROOT / "Linear Program" / "selected_routes.xlsx"


# ============================================================
# CONFIG -- edit these
# ============================================================

ORSkey = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjY4MWI3OThkNjZjZjRmOTZiYWRlODkyYzM1NDU3OTJmIiwiaCI6Im11cm11cjY0In0="

# Which scenario to map (sheet name in selected_routes.xlsx):
#   "Weekday Baseline", "Saturday Baseline",
#   "Weekday Shedding", "Saturday Shedding"
DAY_SHEET = "Saturday Shedding"

OUTPUT_HTML = PROJECT_ROOT / "Visualising Routes" / "saturday_shedding_map.html"


# ============================================================
# LOAD LOCATIONS (course template style)
# ============================================================

locations = pd.read_csv(LOCATIONS_CSV)

coords = locations[["Long", "Lat"]]          # mapping packages work with Long, Lat
coords = coords.to_numpy().tolist()          # list of [Long, Lat] pairs

# name -> index, to look up a store's coords by its Supermarket name
name_to_idx = {locations.Supermarket[i]: i for i in range(len(locations))}
WAREHOUSE_IDX = name_to_idx["Warehouse"]


# ============================================================
# INITIAL MAP + STORE MARKERS (course template style)
# ============================================================

# centre on the warehouse; folium needs [Lat, Long] so reverse
m = folium.Map(location=list(reversed(coords[WAREHOUSE_IDX])), zoom_start=11)

for i in range(len(locations)):
    if locations.Type[i] == "Four Square":
        iconCol = "green"
    elif locations.Type[i] == "New World":
        iconCol = "red"
    elif locations.Type[i] == "New World Metro":
        iconCol = "lightred"
    elif locations.Type[i] == "Pak 'n Save":
        iconCol = "orange"
    elif locations.Type[i] == "Warehouse":
        iconCol = "black"
    else:
        iconCol = "gray"
    folium.Marker(
        list(reversed(coords[i])),
        popup=locations.Supermarket[i],
        icon=folium.Icon(color=iconCol),
    ).add_to(m)


# ============================================================
# LOAD SELECTED ROUTES + DRAW ROAD-FOLLOWING PATHS
# ============================================================

client = ors.Client(key=ORSkey)

routes_df = pd.read_excel(SELECTED_XLSX, sheet_name=DAY_SHEET)


def parse_stops(route_str):
    """'Warehouse -> A -> B -> Warehouse' -> ['A', 'B']"""
    return route_str.split(" -> ")[1:-1]


# a palette so each route line gets a distinct colour
PALETTE = [
    "#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
    "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9A6324",
    "#800000", "#aaffc3", "#808000", "#000075", "#a9a9a9", "#ffd8b1",
]

for r, route_str in enumerate(routes_df["route"]):
    stops = parse_stops(route_str)

    # ordered coordinate list: warehouse -> stops -> warehouse ([Long, Lat])
    seq = [coords[WAREHOUSE_IDX]]
    seq += [coords[name_to_idx[s]] for s in stops]
    seq += [coords[WAREHOUSE_IDX]]

    colour = PALETTE[r % len(PALETTE)]

    try:
        route = client.directions(
            coordinates=seq,
            profile="driving-hgv",      # heavy goods vehicle (truck)
            format="geojson",
            validate=False,
        )
        # path is [Long, Lat]; folium PolyLine needs [Lat, Long]
        folium.PolyLine(
            locations=[list(reversed(c))
                       for c in route["features"][0]["geometry"]["coordinates"]],
            color=colour, weight=3, opacity=0.7,
            popup=f"Route {r + 1}: {' -> '.join(stops)}",
        ).add_to(m)
        time.sleep(1.6)   # respect the 40 requests/minute rate limit
    except Exception as e:
        print(f"[warn] ORS failed for route {r + 1} ({e}); drawing straight line.")
        folium.PolyLine(
            locations=[list(reversed(c)) for c in seq],
            color=colour, weight=2, opacity=0.5, dash_array="5",
        ).add_to(m)


# ============================================================
# SAVE
# ============================================================

m.save(str(OUTPUT_HTML))
print(f"\nDone. Open '{OUTPUT_HTML}' in your browser.")
print(f"Mapped {len(routes_df)} routes for scenario: {DAY_SHEET}")
