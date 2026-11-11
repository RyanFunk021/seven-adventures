"""Pull federal assistance awards for home-repair-relevant programs to recipients in
St. Louis City (FIPS 29510) and St. Louis County (29189) from the USAspending.gov API,
plus sub-awards (e.g. Weatherization / LIHEAP passed through the State of Missouri)."""
import requests, pandas as pd, pathlib, time
OUT = pathlib.Path(__file__).resolve().parents[1] / "data"
API = "https://api.usaspending.gov/api/v2/search/spending_by_award/"
PROGRAMS = {
    "14.218": "CDBG Entitlement", "14.228": "CDBG State / Disaster Recovery",
    "14.239": "HOME Investment Partnerships", "14.900": "Lead Hazard Control (privately-owned housing)",
    "14.905": "Lead Hazard Reduction Demonstration", "14.913": "Healthy Homes Production",
    "14.906": "Healthy Homes Technical Studies", "14.921": "Older Adult Home Modification",
    "81.042": "Weatherization Assistance (DOE)", "93.568": "LIHEAP (incl. weatherization)",
    "10.417": "USDA Sec. 504 Very-Low-Income Housing Repair", "10.433": "USDA Housing Preservation Grants",
}
LOCS = [{"country": "USA", "state": "MO", "county": "510"}, {"country": "USA", "state": "MO", "county": "189"}]

def search(subawards: bool, loc_key: str):
    rows, page = [], 1
    fields = (["Sub-Award ID", "Sub-Awardee Name", "Sub-Award Amount", "Sub-Award Date", "Sub-Award Description",
               "Prime Recipient Name", "Prime Award ID", "Assistance Listing"] if subawards else
              ["Award ID", "Recipient Name", "Award Amount", "Total Outlays", "Description", "Start Date",
               "End Date", "CFDA Number", "Awarding Agency"])
    while True:
        body = {"filters": {"award_type_codes": ["02", "03", "04", "05"], "program_numbers": list(PROGRAMS),
                            loc_key: LOCS, "time_period": [{"start_date": "2007-10-01", "end_date": "2026-09-30"}]},
                "fields": fields, "limit": 100, "page": page, "subawards": subawards,
                "sort": fields[2], "order": "desc"}
        if subawards:
            body["filters"]["time_period"] = [{"start_date": "2010-10-01", "end_date": "2026-09-30"}]
        for attempt in range(5):
            try:
                d = requests.post(API, json=body, timeout=120).json(); break
            except requests.RequestException:
                time.sleep(2 ** attempt)
        rows += d.get("results", [])
        if not d.get("page_metadata", {}).get("hasNext"):
            return pd.DataFrame(rows)
        page += 1

if __name__ == "__main__":
    prime = pd.concat([search(False, "recipient_locations"), search(False, "place_of_performance_locations")])
    prime = prime.drop_duplicates("Award ID")
    prime["program"] = prime["CFDA Number"].map(PROGRAMS)
    prime["year"] = pd.to_datetime(prime["Start Date"], errors="coerce").dt.year
    prime = prime[prime["Award Amount"] > 0]
    prime.drop(columns=[c for c in ["internal_id", "generated_internal_id", "awarding_agency_id",
                                    "agency_slug"] if c in prime], errors="ignore").to_csv(OUT / "federal_prime_awards_stl.csv", index=False)
    summ = (prime.groupby(["CFDA Number", "program"])["Award Amount"]
            .agg(n_awards="count", total="sum", average="mean").sort_values("total", ascending=False).reset_index())
    summ.to_csv(OUT / "federal_awards_by_program.csv", index=False)
    print(summ.to_string())
    try:
        sub = pd.concat([search(True, "place_of_performance_locations"), search(True, "recipient_locations")])
        sub["Assistance Listing"] = sub["Assistance Listing"].astype(str)
        sub = sub.drop_duplicates(["Sub-Award ID", "Prime Award ID", "Sub-Award Amount"])
        sub.drop(columns=[c for c in sub if "internal" in c], errors="ignore").to_csv(OUT / "federal_subawards_stl.csv", index=False)
        print(sub.groupby("Assistance Listing")["Sub-Award Amount"].agg(["count", "sum", "mean"]))
    except Exception as e:
        print("subaward pull failed:", e)
