"""
Foodstuffs route map — road-following, interactive
==================================================

Draws the optimiser's selected delivery routes on an interactive Auckland
map, with each truck's trip following the actual road network (via
OpenRouteService) rather than straight lines.

Output: an HTML file you can open in a browser, pan/zoom, and screenshot
for slides.

------------------------------------------------------------------
SETUP (one-time, run these in the VS Code terminal with your venv active)
------------------------------------------------------------------
1.  pip install openrouteservice folium pandas openpyxl

2.  Get a FREE OpenRouteService API key:
      - go to https://openrouteservice.org/dev/#/signup
      - sign up, then create a token under "Dashboard -> Request a token"
      - copy the long key string

3.  Paste your key into ORS_API_KEY below.

4.  Make sure these two files are where the script expects (see PATHS):
      - FoodstuffsLocations.csv
      - selected_routes.xlsx

5.  Run:  python draw_routes_map.py
    Open the generated  weekday_routes_map.html  in your browser.
------------------------------------------------------------------

Notes
-----
- The free ORS tier allows ~2000 requests/day and 40 requests/minute.
  This script makes one request per route (40 for weekday), so it's well
  within limits, but it inserts a small delay to respect the rate cap.
- If ORS fails for a route (e.g. a temporary network issue), the script
  falls back to a straight line for that route and prints a warning, so
  you still get a complete map.
- Set DAY_SHEET to "Saturday Baseline" (or the shedding sheets) to map a
  different scenario.
"""

import time
import pandas as pd
import folium
import openrouteservice
from openrouteservice import convert
from pathlib import Path

# This script is in "Visualising Routes/", so the repo root is one level up
PROJECT_ROOT = Path(__file__).resolve().parent.parent

LOCATIONS_CSV = PROJECT_ROOT / "Resources" / "FoodstuffsLocations.csv"
SELECTED_XLSX = PROJECT_ROOT / "Linear Program" / "selected_routes.xlsx"
OUTPUT_HTML   = PROJECT_ROOT / "Visualising Routes" / "saturday_shedding_map.html"

# ============================================================
# CONFIG — edit these
# ============================================================

ORS_API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjY4MWI3OThkNjZjZjRmOTZiYWRlODkyYzM1NDU3OTJmIiwiaCI6Im11cm11cjY0In0="

# Which scenario to map. Options (sheet names in selected_routes.xlsx):
#   "Weekday Baseline", "Saturday Baseline",
#   "Weekday Shedding", "Saturday Shedding"
DAY_SHEET = "Saturday Shedding"


# ============================================================
# LOAD DATA
# ============================================================

loc = pd.read_csv(LOCATIONS_CSV)
loc["Long"] = loc["Long"].astype(float)
loc["Lat"] = loc["Lat"].astype(float)

# name -> (lon, lat)  [ORS expects lon,lat order]
coord = {row["Supermarket"]: (row["Long"], row["Lat"]) for _, row in loc.iterrows()}
WAREHOUSE = coord["Warehouse"]

routes = pd.read_excel(SELECTED_XLSX, sheet_name=DAY_SHEET)


def parse_stops(route_str):
    """'Warehouse -> A -> B -> Warehouse' -> ['A', 'B']"""
    return route_str.split(" -> ")[1:-1]


routes["stops"] = routes["route"].apply(parse_stops)


# ============================================================
# STORE-TYPE COLOURS (match the reference map)
# ============================================================

def store_type(name):
    if name.startswith("Pak"):
        return "PnS"
    if name.startswith("New World Metro"):
        return "NWM"
    if name.startswith("New World"):
        return "NW"
    if name.startswith("Four Square"):
        return "FS"
    return "WH"

TYPE_COLOUR = {
    "PnS": "red",
    "NW": "orange",
    "NWM": "beige",
    "FS": "green",
    "WH": "black",
}

# A palette to give each route line a distinct colour
ROUTE_PALETTE = [
    "#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
    "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9A6324",
    "#800000", "#aaffc3", "#808000", "#000075", "#a9a9a9", "#ffd8b1",
]


# ============================================================
# BUILD THE MAP
# ============================================================

client = openrouteservice.Client(key=ORS_API_KEY)

# Centre the map on the warehouse; folium wants (lat, lon)
fmap = folium.Map(location=[WAREHOUSE[1], WAREHOUSE[0]], zoom_start=11,
                  tiles="cartodbpositron")

# --- draw each route as a road-following line ---
for i, (_, row) in enumerate(routes.iterrows()):
    stops = row["stops"]
    # full ordered list of coordinates: warehouse -> stops -> warehouse
    seq = [WAREHOUSE] + [coord[s] for s in stops] + [WAREHOUSE]
    colour = ROUTE_PALETTE[i % len(ROUTE_PALETTE)]

    try:
        # ORS directions: pass coordinates as [lon, lat] pairs
        result = client.directions(
            coordinates=seq,
            profile="driving-car",
            format="geojson",
        )
        geom = result["features"][0]["geometry"]["coordinates"]
        # geojson is [lon, lat]; folium wants [lat, lon]
        line = [(pt[1], pt[0]) for pt in geom]
        folium.PolyLine(
            line, color=colour, weight=3, opacity=0.7,
            tooltip=f"Route {i+1}: {' -> '.join(stops)}  (${row['cost']:.0f})",
        ).add_to(fmap)
        time.sleep(1.6)  # respect ~40 req/min rate limit
    except Exception as e:
        print(f"[warn] ORS failed for route {i+1} ({e}); drawing straight line.")
        line = [(p[1], p[0]) for p in seq]
        folium.PolyLine(line, color=colour, weight=2, opacity=0.5,
                        dash_array="5").add_to(fmap)

# --- draw store markers ---
for _, r in loc.iterrows():
    if r["Supermarket"] == "Warehouse":
        continue
    t = store_type(r["Supermarket"])
    folium.CircleMarker(
        location=[r["Lat"], r["Long"]],
        radius=5,
        color="white", weight=1,
        fill=True, fill_color=TYPE_COLOUR[t], fill_opacity=0.95,
        tooltip=r["Supermarket"],
    ).add_to(fmap)

# --- warehouse marker (big star) ---
folium.Marker(
    location=[WAREHOUSE[1], WAREHOUSE[0]],
    tooltip="Mt Roskill Warehouse",
    icon=folium.Icon(color="black", icon="home", prefix="fa"),
).add_to(fmap)

# --- legend ---
legend_html = """
<div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
     background: white; padding: 10px 14px; border: 1px solid #ccc;
     border-radius: 6px; font-family: Arial; font-size: 13px;">
  <b>Store type</b><br>
  <span style="color:red;">&#9679;</span> Pak &rsquo;n Save<br>
  <span style="color:orange;">&#9679;</span> New World<br>
  <span style="color:#c9a227;">&#9679;</span> New World Metro<br>
  <span style="color:green;">&#9679;</span> Four Square<br>
  <span style="color:black;">&#9733;</span> Warehouse<br>
  <span style="color:#555;">Each coloured line = one truck's round trip</span>
</div>
"""
fmap.get_root().html.add_child(folium.Element(legend_html))

fmap.save(OUTPUT_HTML)
print(f"\nDone. Open '{OUTPUT_HTML}' in your browser.")
print(f"Mapped {len(routes)} routes for scenario: {DAY_SHEET}")
