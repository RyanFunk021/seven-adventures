"""Combine the local funding data, St. Louis need survey (UMSL 2023), AHS cost models and
published benefit-cost evidence into one ranking table by improvement type."""
import pathlib, numpy as np, pandas as pd
D = pathlib.Path(__file__).resolve().parents[1] / "data"
HOME_VALUE = 90_000   # assumed value of a typical St. Louis Healthy Home Repair client home (adjustable)

# St. Louis need survey: UMSL Community Innovation & Action Center (2023), Tables 2-4, n=583 older homeowners
umsl = pd.DataFrame([
    ("Weatherization / insulation / air sealing", 399, 1542), ("Exterior wall / tuckpointing", 250, 2254),
    ("Roof replacement", 125, 4287), ("Seal basement / foundation", 335, 1500), ("Window repair (sashes)", 140, 2900),
    ("Water / sewer piping", 319, 1175), ("Interior wall repair", 315, 1025), ("Gutters", 202, 1503),
    ("Heating equipment (furnace/boiler)", None, 4655), ("Central air conditioning", None, 2100),
    ("Water heater", None, 2044), ("Electrical outlets / wiring", 477, None)],
    columns=["improvement", "umsl_instances", "umsl_cost_each"])
umsl["umsl_total_need"] = umsl.umsl_instances * umsl.umsl_cost_each

ahs = pd.read_csv(D / "ahs_cost_by_improvement_type.csv", index_col=0)
pred = pd.read_csv(D / "model1_predicted_cost_stl_client.csv", index_col=0).pred_cost_stl_client
hed = pd.read_csv(D / "model2_hedonic_premiums.csv", index_col=0)
MAP = {"Weatherization / insulation / air sealing": "Insulation", "Exterior wall / tuckpointing": "Siding",
       "Roof replacement": "Roof replacement", "Window repair (sashes)": "Windows / doors", "Water / sewer piping": "Water pipes",
       "Interior wall repair": "Other interior repair", "Heating equipment (furnace/boiler)": "Heating equipment (furnace/boiler)",
       "Central air conditioning": "Central air conditioning", "Water heater": "Water heater",
       "Electrical outlets / wiring": "Electrical wiring / panel", "Seal basement / foundation": None, "Gutters": None}
umsl["ahs_type"] = umsl.improvement.map(MAP)
umsl["ahs_us_median_cost"] = umsl.ahs_type.map(ahs["median"])
umsl["model_cost_stl_client"] = umsl.ahs_type.map(pred)
umsl["hedonic_premium"] = umsl.ahs_type.map(hed.pct_value_premium_all)
umsl["hedonic_p"] = umsl.ahs_type.map(hed.p_all)
umsl["value_roi"] = umsl.hedonic_premium * HOME_VALUE / umsl.model_cost_stl_client
umsl.to_csv(D / "synthesis_by_improvement.csv", index=False)

# Program-level ROI evidence (benefit per $1 of cost) — published sources, see README
evidence = pd.DataFrame([
    ("Lead hazard control", 17, 221, "Gould 2009, Environ. Health Perspect. 117(7) — health, IQ/earnings, tax, crime benefits"),
    ("Aging-in-place modifications (CAPABLE model)", 2.9, 5.2,
     "Szanton et al. / Bishai et al. 2018 JAGS: ~$867/mo lower Medicaid spend x 17-mo avg follow-up (~$14.7K) vs ~$2,825-$5,000 program cost"),
    ("Weatherization (incl. health & safety)", 4.1, 4.1, "DOE/ORNL WAP Retrospective Evaluation PY2008 — benefit-cost ratio"),
    ("Weatherization (energy savings only)", 1.4, 1.5, "DOE/ORNL WAP evaluations — savings-to-investment ratio (SIR)"),
    ("Roof / furnace / major systems", None, None, "No rigorous published ROI; value is avoided damage & displacement (UMSL 2023 qualitative)"),
], columns=["program", "bcr_low", "bcr_high", "source"])
evidence.to_csv(D / "roi_evidence_benchmarks.csv", index=False)
pd.set_option("display.width", 250); print(umsl.round(3).to_string()); print(evidence.to_string())

# ---------- Final ranking: ROI where measured, total value otherwise ----------
sub = pd.read_csv(D / "federal_subawards_stl_classified.csv")
sub_tot = sub.groupby("category")["Sub-Award Amount"].sum()
prime = pd.read_csv(D / "federal_prime_awards_stl.csv")
prime_tot = prime.groupby("CFDA Number")["Award Amount"].sum()
pc = pred.to_dict()
rank = pd.DataFrame([
    dict(rank=1, improvement="Lead hazard control", bcr_low=17, bcr_high=221, roi_basis="Measured (health/earnings, Gould 2009)",
         stl_federal_funding=sub_tot.get("Lead hazard control", 0), stl_prime_grants=prime_tot.get(14.905, 0) + prime_tot.get(14.900, 0),
         typical_job=np.nan, note="HUD lead grants go straight to contractors (Bozikis, Maurice-Benjamin, United Infrastructure). Needs EPA/MO lead abatement certification."),
    dict(rank=2, improvement="Aging-in-place: accessibility, safety, minor repair", bcr_low=2.9, bcr_high=5.2, roi_basis="Measured (Medicaid savings, CAPABLE)",
         stl_federal_funding=sub_tot.get("Aging-in-place: minor repair, accessibility & safety", 0), stl_prime_grants=prime_tot.get(14.921, 0),
         typical_job=3000, note="Grab bars, railings, ramps, lighting, small fixes. Low ticket, high volume; Senior Home Security is the largest local sub-recipient."),
    dict(rank=3, improvement="Weatherization: insulation & air sealing", bcr_low=1.4, bcr_high=4.1, roi_basis="Measured (energy SIR 1.4; 4.1 with health & safety, DOE/ORNL)",
         stl_federal_funding=sub_tot.get("Weatherization / energy efficiency", 0), stl_prime_grants=np.nan,
         typical_job=pc.get("Insulation"), note="Most common St. Louis need with the largest total dollar need (UMSL). Flows via Urban League, CAASTLC, Senior Home Security."),
    dict(rank=4, improvement="Roof replacement", bcr_low=np.nan, bcr_high=np.nan, roi_basis="Not measured; highest value per job",
         stl_federal_funding=np.nan, stl_prime_grants=np.nan, typical_job=pc.get("Roof replacement"),
         note="Largest U.S. repair spend category and #2 most expensive St. Louis need. Funded inside Healthy Home Repair and CDBG-DR/tornado programs."),
    dict(rank=5, improvement="Flood / disaster repair (CDBG-DR)", bcr_low=np.nan, bcr_high=np.nan, roi_basis="Not measured; largest open funding pool",
         stl_federal_funding=np.nan, stl_prime_grants=prime_tot.get(14.228, 0), typical_job=pc.get("Disaster repair"),
         note="$82M HUD CDBG-DR to City + County for the July 2022 floods, available through 2030."),
    dict(rank=6, improvement="Heating equipment (furnace/boiler)", bcr_low=np.nan, bcr_high=np.nan, roi_basis="Not measured; part of WAP health & safety",
         stl_federal_funding=np.nan, stl_prime_grants=np.nan, typical_job=pc.get("Heating equipment (furnace/boiler)"),
         note="Most expensive single repair in the UMSL St. Louis survey ($4,655)."),
    dict(rank=7, improvement="Exterior wall / tuckpointing", bcr_low=np.nan, bcr_high=np.nan, roi_basis="Not measured; hedonic +4.3% (p=0.09)",
         stl_federal_funding=np.nan, stl_prime_grants=np.nan, typical_job=pc.get("Siding"),
         note="St. Louis brick stock: #2 total need in UMSL survey; CDBG tuckpointing programs (Harambee)."),
    dict(rank=8, improvement="Electrical wiring, panels & outlets", bcr_low=np.nan, bcr_high=np.nan, roi_basis="Hedonic +4.5% (p=0.003)",
         stl_federal_funding=np.nan, stl_prime_grants=np.nan, typical_job=pc.get("Electrical wiring / panel"),
         note="Most common St. Louis need (82% of older homeowners). Small ticket, fire-safety priority."),
])
rank["owner_occupied_repair_pool"] = sub_tot.get("Owner-occupied home repair / rehab", 0)
rank.round(0).to_csv(D / "final_ranking.csv", index=False)
print(rank[["rank", "improvement", "bcr_low", "bcr_high", "stl_federal_funding", "stl_prime_grants", "typical_job"]].to_string())
