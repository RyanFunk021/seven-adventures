# St. Louis federally funded home repair: ROI by improvement type

Analysis for a St. Louis contractor doing federally funded home repair. It answers: which improvement types
have the highest ROI, and, where ROI isn't measured, which carry the most total value?

## Data sources
| Level | Source | What it gives |
|---|---|---|
| Federal | USAspending.gov API (prime awards + sub-awards) | All CDBG, CDBG-DR, HOME, Lead Hazard Control, Healthy Homes, Older Adult Home Mod, Weatherization, LIHEAP, USDA 504 money to St. Louis City (29510) and County (29189) |
| Federal | HUD CPD open data (CDBG/HOME activity layers, ArcGIS) | Checked. Leaves out single-family rehab, so not used for costs |
| Federal | Census/HUD American Housing Survey 2019/2021/2023 national files | 82,656 contractor-done jobs with type and cost, plus home value. Used for the regression models |
| Federal | DOE/ORNL Weatherization evaluations | Savings-to-investment ratio 1.4; benefit-cost ratio 4.1 including health and safety |
| Local | City of St. Louis CAPERs 2018–2025 (IDIS PR26) | CDBG disbursements by HUD matrix code; Healthy Home Repair per-home figures |
| Local | UMSL Home Repair Study (2023) | Repair needs and cost per repair for 583 older St. Louis homeowners |
| Literature | Gould 2009 (lead); CAPABLE / Bishai 2018 (aging in place) | Benefit per $1 |

State-level data (MO DNR weatherization, MHDC) appears here as pass-through sub-awards in USAspending.

## Pipeline
```
python scripts/01_usaspending.py                 # data/federal_*.csv
python scripts/02_classify_subawards.py          # improvement-type classification
python scripts/03_caper_pr26.py <dir of CAPER .txt (pdftotext -layout)>
python scripts/04_ahs_models.py <dir with n19.zip n21.zip n23.zip from www2.census.gov/programs-surveys/ahs>
python scripts/05_synthesis.py                   # data/final_ranking.csv
```
Model summaries are in `output/`.

## Models
* **M1 cost model** (weighted OLS on log job cost): improvement type, log home value, log income, era built, Census division, year, beds and baths. R² 0.34, n=57,830. Duan smearing is used to predict cost for a typical client home (pre-1940, $90K, $30K income, Midwest).
* **M2 hedonic value model** (weighted OLS on log owner-reported value): indicators for each repair done in the last 2 years plus controls. R² 0.43, n=80,001. A Midwest pre-1960 version is also estimated. Correlational only.
* **M3**: gradient boosting vs OLS on held-out jobs (R² 0.341 vs 0.328). The type of repair, not the home, drives most cost variation.

## Result
1. **Lead hazard control**: $17–$221 of benefit per $1, the highest measured ROI. About $54M in local and direct grants since 2008. Requires lead abatement certification.
2. **Aging-in-place modifications**: about $2.9–$5.2 per $1 (CAPABLE Medicaid savings). $17.6M in local sub-awards.
3. **Weatherization (insulation, air sealing)**: savings-to-investment ratio 1.4; 4.1 with health and safety. The most common St. Louis need and the largest total dollar need.
4. **Highest value without ROI data**: roof replacement (about $10.4K per job), flood/disaster repair (about $14.6K per job; $82M CDBG-DR through 2030), furnace (about $4.7K).

Healthy Home Repair benchmark (2025 CAPER): 415 repairs for 57 homeowners, about $22K per home.

Caveats: ROI is public benefit per grant dollar, not contractor margin. Sub-awards are program allocations, not single jobs. Classification is keyword-based.

## Where to find the houses (scripts 06–07)
```
python scripts/06_pull_location_data.py          # FEMA IHP registrations, HUD CDBG by tract, TIGERweb tracts/places
python scripts/07_location_scoring.py <dir of ACS 2023 table-based summary files: b25003 b25007 b25036 b25077 b25119 .dat>
```
Every census tract in St. Louis City and County gets a 0–100 percentile score. The weights follow the ROI ranking:

| Factor | Weight |
|---|---|
| Owner-occupied homes built before 1940 (lead) | 25% |
| Low owner median income (program eligibility) | 25% |
| Seniors in pre-1960 homes (aging in place) | 20% |
| FEMA unmet homeowner repair need from the 2022 flood and 2025 storms (disaster) | 20% |
| Low home value | 10% |

Outputs: `data/tract_opportunity_scores.csv`, `data/area_opportunity_scores.csv` (City neighborhoods and County municipalities), `data/top_areas_by_program.csv`, `data/tract_map.geojson`.

What it found:
* **North City, along the May 2025 tornado path**: Kingsway East, O'Fallon, Penrose, Greater Ville, Lewis Place, Walnut Park and Mark Twain. FEMA owner registrations there show $23M of unmet homeowner repair need from the tornado alone.
* **North County's inner-ring suburbs**: Velda Village Hills, Northwoods, Jennings, Ferguson, Bellefontaine Neighbors, Overland, Florissant and St. Ann. These have the largest count of seniors in 1950s houses (aging in place, weatherization).
* **South City's pre-1940 brick**: Dutchtown, Gravois Park, Bevo Mill, Princeton Heights, Mount Pleasant and Patch. These have the most pre-1940 owner-occupied homes (lead).
