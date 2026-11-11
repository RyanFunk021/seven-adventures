"""Classify St. Louis federal sub-awards into improvement / activity types using
keyword rules on the sub-award description, sub-awardee and assistance listing."""
import pandas as pd, re, pathlib
D = pathlib.Path(__file__).resolve().parents[1] / "data"
RULES = [  # (category, is_home_repair_market, regex on DESC|AWARDEE)
    ("Lead hazard control", True, r"\bLEAD\b|LEAD-BASED|LEAD HAZARD|LEAD GRANT|BOZIKIS|LEAD SAFE"),
    ("Weatherization / energy efficiency", True, r"WEATHERI|ENERGY EFFICIEN|INSULAT"),
    ("Aging-in-place: minor repair, accessibility & safety", True,
     r"ACCESSIB|ARCHITECTURAL BARRIER|OLDER ADULT|HOME HAZARD|AGING IN PLACE|SENIOR HOME SECURITY|SAFETY AND SECURITY|RAMP|GRAB BAR"),
    ("Owner-occupied home repair / rehab", True,
     r"HOME REPAIR|HEALTHY HOME|REHAB(ILITATION)? OF PRIVATE|HOME IMPROVEMENT|MINOR REPAIR|EMERGENCY REPAIR|ROOF|FURNACE|"
     r"FORGIVABLE|HOMEOWNER.*REPAIR|REPAIR.*HOMEOWNER|EXTERIOR REPAIR|HOUSING REHAB|^REHAB\b|REHAB OF \d+ PRIVATE|OWNER.OCCUPIED|"
     r"REHABILITATING EXISTING STRUCTURE|REHABILITATION OF HOUSING|PRESERVATION COMMISSION"),
    ("Rental / multi-unit rehab", False, r"RENTAL REHAB|MULTI.?(FAMILY|UNIT)|APARTMENT|HOUSING AUTHORIT|SHELTER|HOMELESS"),
    ("New construction / development", False, r"NEW CONSTRUCTION|DEVELOPMENT OF .*HOUSING|LP$|L\.P\.|HOUSING DEVELOPMENT|AFFORDABLE HOUSING"),
    ("Vacant property: demolition, board-up, stabilization", False, r"DEMOLI|CLEARANCE|BOARD|SECURE VACANT|VACANT|NUISANCE|STABILIZ"),
    ("Energy bill / crisis assistance (non-repair)", False, r"LIHEAP|ENERGY ASSISTANCE|ECIP|CRISIS"),
    ("Down-payment assistance", False, r"DOWN ?PAYMENT|DPA"),
]
CFDA_FALLBACK = {"14.900": "Lead hazard control", "14.905": "Lead hazard control", "81.042": "Weatherization / energy efficiency",
                 "14.921": "Aging-in-place: minor repair, accessibility & safety", "93.568": "Energy bill / crisis assistance (non-repair)"}
HOME_REPAIR = {r[0] for r in RULES if r[1]}

def classify(row):
    text = f"{row['desc']} | {row['awardee']}"
    if row["cfda"].startswith(("14.900", "14.905")):
        return "Lead hazard control"
    if row["cfda"] == "81.042":
        return "Weatherization / energy efficiency"
    for cat, _, rx in RULES:
        if re.search(rx, text):
            return cat
    return CFDA_FALLBACK.get(row["cfda"], "Other CDBG/HOME (services, econ. dev., infrastructure)")

if __name__ == "__main__":
    s = pd.read_csv(D / "federal_subawards_stl.csv")
    s["cfda"] = s["Assistance Listing"].str.extract(r"'cfda_number': '([^']+)'")[0].fillna("")
    s["desc"] = s["Sub-Award Description"].fillna("").str.upper().str.strip()
    s["awardee"] = s["Sub-Awardee Name"].fillna("").str.upper()
    s["category"] = s.apply(classify, axis=1)
    s["home_repair_market"] = s["category"].isin(HOME_REPAIR)
    s["year"] = pd.to_datetime(s["Sub-Award Date"], errors="coerce").dt.year
    s = s[s["Sub-Award Amount"] > 0]
    s.drop(columns=["desc", "awardee"]).to_csv(D / "federal_subawards_stl_classified.csv", index=False)
    g = (s.groupby(["category", "home_repair_market"])["Sub-Award Amount"]
         .agg(n="count", total="sum", mean="mean", median="median").sort_values("total", ascending=False).reset_index())
    g.to_csv(D / "subawards_by_improvement_type.csv", index=False)
    pd.set_option("display.width", 200); print(g.to_string())
    top = (s[s.home_repair_market].groupby(["category", "Sub-Awardee Name"])["Sub-Award Amount"].sum()
           .sort_values(ascending=False).groupby(level=0).head(5))
    top.to_csv(D / "subawards_top_recipients_by_type.csv"); print(top.to_string())
    yr = (s[s.home_repair_market & s.year.between(2011, 2025)].groupby(["year", "category"])["Sub-Award Amount"].sum()
          .reset_index().rename(columns={"Sub-Award Amount": "amount"}))
    yr.to_csv(D / "home_repair_subawards_by_year.csv", index=False)
