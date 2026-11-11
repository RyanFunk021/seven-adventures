"""Pull tract/block-group inputs for St. Louis City (29510) and County (29189):
 - FEMA IHP valid registrations (block-group coded) for disasters 4665 (Jul-2022 flood), 4867 (Mar-2025 storms), 4877 (May-2025 tornado)
 - HUD CDBG activity aggregated by 2010 tract
 - Census TIGERweb 2020 tract centroids and incorporated places (for municipality / neighborhood labels)"""
import requests, pandas as pd, pathlib, json, time
G = pathlib.Path(__file__).resolve().parents[1] / "data" / "geo"

def get(url, **kw):
    for a in range(5):
        try:
            r = requests.get(url, timeout=180, **kw); r.raise_for_status(); return r
        except requests.RequestException:
            time.sleep(2 ** a)
    raise RuntimeError(url)

# FEMA
rows = []
for dn in (4665, 4867, 4877):
    skip = 0
    while True:
        r = get("https://www.fema.gov/api/open/v2/IndividualsAndHouseholdsProgramValidRegistrations",
                params={"$filter": f"disasterNumber eq {dn} and (fips eq '29510' or fips eq '29189')", "$top": 10000, "$skip": skip}).json()
        b = r["IndividualsAndHouseholdsProgramValidRegistrations"]; rows += b
        if len(b) < 10000: break
        skip += 10000
fema = pd.DataFrame(rows); fema.to_csv(G / "fema_ihp_registrations_stl.csv", index=False)
print("FEMA", fema.groupby(["disasterNumber", "fips"]).size())

# HUD CDBG by tract
B = "https://services.arcgis.com/VTyQ9soqVukalItT/ArcGIS/rest/services/CDBG_ACTIVITY_BY_TRACT/FeatureServer/0/query"
r = get(B, params={"where": "STATE='29' AND COUNTY IN ('510','189')", "outFields": "*", "returnGeometry": "false", "f": "json"}).json()
cd = pd.DataFrame([f["attributes"] for f in r["features"]]); cd.to_csv(G / "hud_cdbg_activity_by_tract.csv", index=False); print("CDBG tracts", len(cd))

# TIGERweb tracts (2020) + places
T = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Tracts_Blocks/MapServer"
lay = get(T + "?f=json").json()["layers"]; print([ (l["id"], l["name"]) for l in lay][:8])
tid = next(l["id"] for l in lay if l["name"].lower().startswith("census tracts"))
tr = get(f"{T}/{tid}/query", params={"where": "STATE='29' AND COUNTY IN ('510','189')", "outFields": "GEOID,NAME,CENTLAT,CENTLON,AREALAND",
                                      "returnGeometry": "true", "outSR": 4326, "f": "geojson"}).json()
json.dump(tr, open(G / "tracts_2020.geojson", "w")); print("tracts", len(tr["features"]))
P = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Places_CouSub_ConCity_SubMCD/MapServer"
lay = get(P + "?f=json").json()["layers"]; pid = next(l["id"] for l in lay if l["name"].lower().startswith("incorporated places"))
pl = get(f"{P}/{pid}/query", params={"where": "STATE='29'", "geometry": "-90.75,38.38,-90.10,38.90", "geometryType": "esriGeometryEnvelope", "inSR": 4326,
                                    "outFields": "GEOID,BASENAME,NAME", "returnGeometry": "true", "outSR": 4326, "f": "geojson"}).json()
json.dump(pl, open(G / "places.geojson", "w")); print("places", len(pl["features"]))
