"""Parse the IDIS PR26 voucher-level detail printed in the City of St. Louis CAPERs
(program years 2018-2025) and total CDBG disbursements by HUD matrix code.
Rows are de-duplicated on (activity, voucher, amount) because PR26 repeats draws in several lines."""
import re, sys, pathlib, pandas as pd
D = pathlib.Path(__file__).resolve().parents[1] / "data"
TXT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
FILES = {2025: "City-of-St-Louis-CDA-2025-CAPER-Final-March-30-2026-v2", 2024: "CDA-STL-2024-CAPER",
         2023: "2023-CAPER-STL-CDA-final-2", 2022: "2022-City-of-St-Louis-CAPER", 2021: "CAPER-2021-Final-3-31-22",
         2020: "2020-Caper-Final-Version-3-31-21-2", 2019: "2019-CAPER-Final-Revised-6-8-21", 2018: "2018-CAPER-Final"}
ROW = re.compile(r"^\s*(20\d\d)\s+(\d{1,4})\s+(\d{3,5})\s+(\d{6,8})\s+(.+?)\s{2,}(\d{2}[A-Z]?)\s+(LM[ACHJ]C?|SB[ARS]|URG)\b.*?\$\s?([\d,]+\.\d\d)\s*$")
MATRIX = {"14A": "Rehab: single-unit residential", "14B": "Rehab: multi-unit residential", "14F": "Energy efficiency improvements",
          "14H": "Rehab administration", "14I": "Lead-based paint abatement", "14J": "Housing services", "14G": "Acquisition for rehab",
          "14C": "Public housing modernization", "14E": "Rehab: commercial/industrial", "10": "Removal of architectural barriers",
          "04": "Clearance & demolition", "15": "Code enforcement", "12": "Construction of housing", "13B": "Homeownership assistance",
          "05Z": "Other public services", "05A": "Senior services", "05D": "Youth services", "03Z": "Other public improvements",
          "03K": "Street improvements", "03E": "Neighborhood facilities", "18A": "ED: assistance to for-profits", "19C": "CDBG nonprofit capacity",
          "21A": "General program administration", "20": "Planning", "01": "Acquisition", "02": "Disposition", "08": "Relocation", "06": "Interim assistance"}
rows = []
for py, f in FILES.items():
    for line in (TXT / f"{f}.txt").read_text(errors="ignore").splitlines():
        m = ROW.match(line)
        if m:
            plan_yr, proj, act, vouch, name, mc, nobj, amt = m.groups()
            rows.append(dict(caper_year=py, plan_year=int(plan_yr), project=int(proj), activity=int(act), voucher=vouch,
                             activity_name=re.sub(r"\s+", " ", name).strip(), matrix_code=mc, national_objective=nobj,
                             amount=float(amt.replace(",", ""))))
df = pd.DataFrame(rows).drop_duplicates(["caper_year", "activity", "voucher", "amount"])
df["matrix_desc"] = df.matrix_code.map(MATRIX).fillna("Other")
df.to_csv(D / "caper_pr26_vouchers.csv", index=False)
by = (df.groupby(["matrix_code", "matrix_desc"]).agg(total=("amount", "sum"), activities=("activity", "nunique"),
                                                    vouchers=("voucher", "count")).reset_index())
by["avg_per_activity"] = by.total / by.activities
by = by.sort_values("total", ascending=False); by.to_csv(D / "caper_cdbg_by_matrix_code.csv", index=False)
yr = df.pivot_table(index="matrix_code", columns="caper_year", values="amount", aggfunc="sum").fillna(0)
yr.to_csv(D / "caper_cdbg_by_matrix_code_year.csv")
pd.set_option("display.width", 220); print(df.groupby("caper_year").amount.agg(["count", "sum"])); print(by.head(30).to_string())
