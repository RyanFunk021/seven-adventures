"""Score every census tract in St. Louis City + County on where grant-eligible repair jobs are most likely.
Inputs: ACS 2023 5-yr summary files (www2.census.gov table-based SF), FEMA IHP registrations, HUD CDBG-by-tract,
TIGERweb tracts/places, City neighborhood boundaries.  Usage: python 07_location_scoring.py <dir with ACS .dat files>"""
import sys, json, pathlib, numpy as np, pandas as pd, shapefile
from shapely.geometry import shape, Point, mapping
from shapely.ops import transform  # noqa
from pyproj import Transformer
ACS = pathlib.Path(sys.argv[1]); R = pathlib.Path(__file__).resolve().parents[1]; G = R / "data" / "geo"; D = R / "data"

def acs(t):
    df = pd.read_csv(ACS / f"{t}.dat", sep="|", dtype=str)
    df = df[df.GEO_ID.str.match(r"1400000US29(510|189)")].copy()
    df["tract"] = df.GEO_ID.str[-11:]
    return df.set_index("tract")
b36, b07, b77, b119, b03 = (acs(t) for t in ["b25036", "b25007", "b25077", "b25119", "b25003"])
n = lambda df, c: pd.to_numeric(df[c], errors="coerce").where(lambda x: x >= 0)   # ACS jam values (-666666666 etc.) -> NaN
t = pd.DataFrame(index=b36.index)
t["owner_units"] = n(b36, "B25036_E002")
t["owner_pre1940"] = n(b36, "B25036_E012")
t["owner_pre1960"] = n(b36, "B25036_E010") + n(b36, "B25036_E011") + n(b36, "B25036_E012")
t["owner_65plus"] = n(b07, "B25007_E009") + n(b07, "B25007_E010") + n(b07, "B25007_E011")
t["median_value"] = n(b77, "B25077_E001")
t["owner_median_income"] = n(b119, "B25119_E002")
t["occupied_units"] = n(b03, "B25003_E001")
t = t[t.owner_units > 0]
t["pct_owner_pre1960"] = t.owner_pre1960 / t.owner_units
t["pct_owner_65plus"] = t.owner_65plus / t.owner_units
# senior homeowners living in pre-1960 homes (assumes age and building era are independent within a tract)
t["senior_old_homes"] = t.owner_pre1960 * t.pct_owner_65plus

# FEMA: owner-occupant registrations by tract
f = pd.read_csv(G / "fema_ihp_registrations_stl.csv", dtype={"censusGeoid": str}, low_memory=False)
f = f[(f.ownRent == "O") & f.censusGeoid.notna()].copy()
f["tract"] = f.censusGeoid.str[:11]
for c in ["rpfvl", "unmetNeedRp", "roofDamageAmount", "foundationDamageAmount", "floodDamageAmount"]:
    f[c] = pd.to_numeric(f[c], errors="coerce").fillna(0)
f["low_income"] = f.grossIncome.isin(["0", "<$15,000", "$15,000-$30,000", "$30,001-$60,000"])   # under $60K
f["senior"] = f.applicantAge.eq("65+")
f["habitability"] = f.habitabilityRepairsRequired.astype(str).str.lower().eq("true")
f["uninsured"] = ~f.homeOwnersInsurance.astype(str).str.lower().eq("true")
f["flood"] = f.floodDamage.astype(str).str.lower().eq("true")
fe = f.groupby("tract").agg(fema_owner_regs=("id", "count"), fema_verified_loss=("rpfvl", "sum"), fema_unmet_need=("unmetNeedRp", "sum"),
                            fema_habitability=("habitability", "sum"), fema_senior=("senior", "sum"), fema_flood=("flood", "sum"),
                            fema_low_income=("low_income", "sum"), fema_uninsured=("uninsured", "sum"),
                            fema_tornado_2025=("disasterNumber", lambda s: (s == 4877).sum()), fema_flood_2022=("disasterNumber", lambda s: (s == 4665).sum()))
t = t.join(fe).fillna({c: 0 for c in fe.columns})
# FEMA by income/age band overall for context
f.groupby(["disasterNumber", "grossIncome"]).size().rename("owner_regs").to_csv(D / "fema_owner_regs_by_income.csv")

# HUD CDBG housing dollars by (2010) tract
cd = pd.read_csv(G / "hud_cdbg_activity_by_tract.csv", dtype={"GEOID": str})
t = t.join(cd.set_index("GEOID")[["HSG_COUNT", "HSG_FUNDING_AMT", "TOTAL_AMT"]].rename(columns={"HSG_COUNT": "cdbg_housing_activities", "HSG_FUNDING_AMT": "cdbg_housing_funding", "TOTAL_AMT": "cdbg_total_funding"}))

# Labels: City neighborhood or County municipality, by tract centroid
tr = json.load(open(G / "tracts_2020.geojson"))
geo = {ft["properties"]["GEOID"]: shape(ft["geometry"]) for ft in tr["features"]}
to_wgs = Transformer.from_crs("ESRI:102696", "EPSG:4326", always_xy=True).transform
nb = shapefile.Reader(str(G / "Neighborhood_Boundaries"))
nbs = [(rec["NHD_NAME"], transform(to_wgs, shape(s.__geo_interface__))) for s, rec in zip(nb.shapes(), nb.records())]
pl = [(ft["properties"]["BASENAME"], shape(ft["geometry"])) for ft in json.load(open(G / "places.geojson"))["features"]]
def label(g):
    gm = geo.get(g)
    if gm is None: return ("", np.nan, np.nan)
    p = gm.representative_point()
    pool = nbs if g.startswith("29510") else pl
    name = next((nm for nm, poly in pool if poly.contains(p)), "Unincorporated St. Louis County" if g.startswith("29189") else "St. Louis City")
    return (name, p.y, p.x)
lab = pd.DataFrame([label(g) for g in t.index], index=t.index, columns=["area", "lat", "lon"])
t = t.join(lab)
t["jurisdiction"] = np.where(t.index.str.startswith("29510"), "St. Louis City", "St. Louis County")

# Scores (percentile ranks 0-100 across all 300+ tracts)
pr = lambda s: s.rank(pct=True) * 100
t["score_lead"] = pr(t.owner_pre1940)
t["score_aging"] = pr(t.senior_old_homes)
t["score_eligibility"] = pr(-t.owner_median_income.fillna(t.owner_median_income.median()))
t["score_low_value"] = pr(-t.median_value.fillna(t.median_value.median()))
t["score_disaster"] = pr(t.fema_unmet_need + 0.25 * t.fema_verified_loss)
# weights follow the ROI ranking: lead > aging/weatherization > disaster/roofs; eligibility gates everything
t["opportunity_score"] = (0.25 * t.score_lead + 0.20 * t.score_aging + 0.25 * t.score_eligibility + 0.10 * t.score_low_value + 0.20 * t.score_disaster).round(1)
t = t.sort_values("opportunity_score", ascending=False)
t.index.name = "tract"
t.round(3).to_csv(D / "tract_opportunity_scores.csv")

# Area roll-up (neighborhoods / municipalities)
a = t.groupby(["jurisdiction", "area"]).agg(tracts=("owner_units", "size"), owner_units=("owner_units", "sum"), owner_pre1940=("owner_pre1940", "sum"),
    owner_pre1960=("owner_pre1960", "sum"), senior_old_homes=("senior_old_homes", "sum"), fema_owner_regs=("fema_owner_regs", "sum"),
    fema_unmet_need=("fema_unmet_need", "sum"), fema_habitability=("fema_habitability", "sum"), fema_low_income=("fema_low_income", "sum"), fema_tornado_2025=("fema_tornado_2025", "sum"), fema_flood_2022=("fema_flood_2022", "sum"),
    cdbg_housing_funding=("cdbg_housing_funding", "sum"), owner_median_income=("owner_median_income", "median"),
    median_value=("median_value", "median"), opportunity_score=("opportunity_score", "mean")).reset_index()
a = a[a.owner_units >= 150].sort_values("opportunity_score", ascending=False)
a["rank"] = range(1, len(a) + 1)
a.round(1).to_csv(D / "area_opportunity_scores.csv", index=False)

# Simplified map file
feats = []
keep = t[["area", "jurisdiction", "opportunity_score", "owner_pre1940", "senior_old_homes", "owner_median_income", "fema_owner_regs", "fema_unmet_need", "median_value"]]
for g, gm in geo.items():
    if g not in keep.index: continue
    s = gm.simplify(0.0004, preserve_topology=True)
    gj = json.loads(json.dumps(mapping(s)), parse_float=lambda x: round(float(x), 4))
    props = {k: (None if pd.isna(v) else (round(float(v), 1) if isinstance(v, (int, float, np.floating, np.integer)) else v)) for k, v in keep.loc[g].items()}
    props["tract"] = g; feats.append({"type": "Feature", "properties": props, "geometry": gj})
json.dump({"type": "FeatureCollection", "features": feats}, open(D / "tract_map.geojson", "w"), separators=(",", ":"))
pd.set_option("display.width", 250)
print(t[["area", "jurisdiction", "opportunity_score", "owner_units", "owner_pre1940", "senior_old_homes", "owner_median_income", "fema_owner_regs", "fema_unmet_need", "cdbg_housing_funding"]].head(25).to_string())
print(a.head(25).to_string())

# Top areas for each program type ("where to find the houses" for each kind of job)
LENS = [("Lead hazard control", "owner_pre1940", "Owner-occupied homes built before 1940"),
        ("Aging-in-place", "senior_old_homes", "Est. homeowners 65+ in pre-1960 homes"),
        ("Disaster repair (2022 flood, 2025 tornado)", "fema_unmet_need", "FEMA unmet homeowner repair need ($)"),
        ("Weatherization & general repair", "owner_pre1960", "Owner-occupied homes built before 1960")]
out = []
elig = a[a.owner_median_income.fillna(0) < 80000]          # keep areas where typical owner is plausibly income-eligible
for lens, col, label in LENS:
    top = elig.sort_values(col, ascending=False).head(10)
    for i, r in enumerate(top.itertuples(), 1):
        out.append(dict(lens=lens, metric=label, lens_rank=i, area=r.area, jurisdiction=r.jurisdiction, value=round(getattr(r, col), 0),
                        owner_median_income=r.owner_median_income, key=f"{lens}|{i}"))
pd.DataFrame(out).to_csv(D / "top_areas_by_program.csv", index=False)
print(pd.DataFrame(out).to_string())
