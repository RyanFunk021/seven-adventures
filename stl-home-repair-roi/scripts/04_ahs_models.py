"""American Housing Survey (2019, 2021, 2023 national PUFs): job-level home-improvement costs
by improvement type, plus three regression models:
  M1  log(job cost) ~ improvement type + home/household controls      (what drives cost)
  M2  log(home value) ~ improvement types done in last 2 yrs + controls (hedonic value added)
  M3  gradient-boosted cost model vs. OLS (out-of-sample fit check)
Usage: python 04_ahs_models.py <dir containing n19.zip n21.zip n23.zip>"""
import sys, zipfile, pathlib, numpy as np, pandas as pd, statsmodels.formula.api as smf
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
RAW = pathlib.Path(sys.argv[1]); D = pathlib.Path(__file__).resolve().parents[1] / "data"
CPI = {2019: 255.657, 2021: 270.970, 2023: 304.702}; CPI_2024 = 313.689   # CPI-U annual averages -> 2024 dollars
TYPE = {16: "Roof replacement", 17: "Siding", 18: "Windows / doors", 20: "Insulation", 21: "Water pipes",
        22: "Plumbing fixtures", 23: "Electrical wiring / panel", 27: "Heating equipment (furnace/boiler)",
        26: "Central air conditioning", 29: "Water heater", 25: "Flooring / ceilings", 12: "Bathroom remodel",
        13: "Kitchen remodel", 15: "Porch / deck / patio", 19: "Exterior stairs / chimney", 28: "Septic tank",
        24: "Security system", 30: "Dishwasher / disposal", 31: "Other interior repair", 32: "Driveway / walkway",
        33: "Fencing / walls", 7: "Addition", 8: "Addition", 9: "Addition", 10: "Addition", 11: "Addition", 14: "Garage / carport",
        1: "Disaster repair", 2: "Disaster repair", 3: "Disaster repair", 4: "Disaster repair", 5: "Disaster repair", 6: "Disaster repair",
        34: "Yard / recreation", 35: "Yard / recreation", 36: "Yard / recreation", 37: "Yard / recreation"}
GRANT_ELIGIBLE = {"Roof replacement", "Siding", "Windows / doors", "Insulation", "Water pipes", "Plumbing fixtures",
                  "Electrical wiring / panel", "Heating equipment (furnace/boiler)", "Central air conditioning",
                  "Water heater", "Flooring / ceilings", "Bathroom remodel", "Porch / deck / patio",
                  "Exterior stairs / chimney", "Other interior repair", "Disaster repair"}
HH = ["CONTROL", "WEIGHT", "MARKETVAL", "HINCP", "YRBUILT", "UNITSIZE", "BLD", "TENURE", "DIVISION", "BEDROOMS",
      "BATHROOMS", "RATINGHS", "HHAGE"]

def clean(df):
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].str.strip("'")
    return df.apply(pd.to_numeric, errors="coerce")

frames, hhs = [], []
for yr, z in [(2019, "n19"), (2021, "n21"), (2023, "n23")]:
    zf = zipfile.ZipFile(RAW / f"{z}.zip")
    p = clean(pd.read_csv(zf.open("project.csv"), usecols=["CONTROL", "JOBTYPE", "JOBCOST", "JOBDIY", "JOBFUNDS"], dtype=str))
    h = clean(pd.read_csv(zf.open("household.csv"), usecols=HH, dtype=str))
    p["year"] = h["year"] = yr
    frames.append(p); hhs.append(h)
proj, hh = pd.concat(frames), pd.concat(hhs)
own = hh[(hh.TENURE == 1) & hh.BLD.isin([2, 3]) & (hh.MARKETVAL > 0) & (hh.HINCP > -1)].copy()
own["infl"] = CPI_2024 / own.year.map(CPI)
own["value24"] = own.MARKETVAL * own.infl
own["inc24"] = own.HINCP.clip(lower=1) * own.infl
own["era"] = pd.cut(own.YRBUILT, [0, 1939, 1959, 1979, 1999, 2100], labels=["pre-1940", "1940-59", "1960-79", "1980-99", "2000+"])
own["midwest"] = own.DIVISION.isin([3, 4])
own["stl_like"] = own.midwest & (own.YRBUILT < 1960) & (own.inc24 < 60000) & (own.value24 < 200000)

j = proj.merge(own, on=["CONTROL", "year"])
j = j[(j.JOBCOST > 0) & j.JOBTYPE.isin(TYPE)].copy()
j["type"] = j.JOBTYPE.map(TYPE)
j["cost24"] = j.JOBCOST * j.infl
j["pro"] = (j.JOBDIY == 2)                       # hired a contractor (what a grant-funded business does)
jp = j[j.pro]

def wstats(g):
    w = g.WEIGHT; x = g.cost24
    return pd.Series({"jobs_n": len(g), "wtd_mean": np.average(x, weights=w), "median": x.median(),
                      "p25": x.quantile(.25), "p75": x.quantile(.75),
                      "annual_jobs_US": w.sum() / 3, "annual_spend_US": (w * x).sum() / 3})
tab = jp.groupby("type").apply(wstats, include_groups=False)
stl = jp[jp.stl_like].groupby("type").apply(wstats, include_groups=False)[["jobs_n", "wtd_mean", "median"]]
stl.columns = ["stl_like_n", "stl_like_wtd_mean", "stl_like_median"]
tab = tab.join(stl); tab["grant_eligible"] = tab.index.isin(GRANT_ELIGIBLE)
tab.sort_values("annual_spend_US", ascending=False).round(0).to_csv(D / "ahs_cost_by_improvement_type.csv")

# ---------- M1: cost model ----------
m1d = jp[jp.type.isin(GRANT_ELIGIBLE)].copy()
m1d["lcost"] = np.log(m1d.cost24); m1d["lval"] = np.log(m1d.value24); m1d["linc"] = np.log(m1d.inc24)
m1 = smf.wls("lcost ~ C(type, Treatment('Water heater')) + lval + linc + C(era) + C(DIVISION) + C(year) + BEDROOMS + BATHROOMS",
             data=m1d.dropna(subset=["era"]), weights=m1d.dropna(subset=["era"]).WEIGHT).fit(cov_type="HC1")
open(D.parent / "output" / "model1_cost_ols.txt", "w").write(str(m1.summary()))
# predicted cost for a typical St. Louis Healthy-Home-Repair client home
prof = dict(lval=np.log(90000), linc=np.log(30000), era="pre-1940", DIVISION=4, year=2023, BEDROOMS=3, BATHROOMS=2)
smear = np.average(np.exp(m1.resid), weights=m1.model.weights)          # Duan smearing for log retransform
pred = pd.DataFrame([{**prof, "type": t} for t in sorted(GRANT_ELIGIBLE)])
pred["pred_cost_stl_client"] = np.exp(m1.predict(pred)) * smear
pred[["type", "pred_cost_stl_client"]].round(0).to_csv(D / "model1_predicted_cost_stl_client.csv", index=False)

# ---------- M2: hedonic value model ----------
done = (jp[jp.type.isin(GRANT_ELIGIBLE)].assign(v=1).pivot_table(index=["CONTROL", "year"], columns="type", values="v", aggfunc="max"))
done.columns = ["did_" + "".join(ch if ch.isalnum() else "_" for ch in c) for c in done.columns]
h2 = own.set_index(["CONTROL", "year"]).join(done).fillna({c: 0 for c in done.columns}).reset_index().dropna(subset=["era"])
h2 = h2[(h2.RATINGHS > 0)]
h2["lval"] = np.log(h2.value24); h2["linc"] = np.log(h2.inc24)
rhs = " + ".join(done.columns)
ctrl = "linc + C(era) + C(DIVISION) + C(year) + BEDROOMS + BATHROOMS + C(UNITSIZE) + C(BLD)"
m2 = smf.wls(f"lval ~ {rhs} + {ctrl}", data=h2, weights=h2.WEIGHT).fit(cov_type="HC1")
m2b = smf.wls(f"lval ~ {rhs} + {ctrl}", data=h2[h2.midwest & (h2.YRBUILT < 1960)],
              weights=h2[h2.midwest & (h2.YRBUILT < 1960)].WEIGHT).fit(cov_type="HC1")
open(D.parent / "output" / "model2_hedonic_value.txt", "w").write(str(m2.summary()) + "\n\n=== Midwest, pre-1960 homes ===\n" + str(m2b.summary()))
inv = {("did_" + "".join(ch if ch.isalnum() else "_" for ch in t)): t for t in GRANT_ELIGIBLE}
rows = []
for c in done.columns:
    rows.append(dict(type=inv[c], coef_all=m2.params[c], se_all=m2.bse[c], p_all=m2.pvalues[c],
                     coef_mw_old=m2b.params[c], se_mw_old=m2b.bse[c], p_mw_old=m2b.pvalues[c]))
hed = pd.DataFrame(rows)
hed["pct_value_premium_all"] = np.exp(hed.coef_all) - 1
hed["pct_value_premium_mw_old"] = np.exp(hed.coef_mw_old) - 1
hed.to_csv(D / "model2_hedonic_premiums.csv", index=False)

# ---------- M3: ML vs OLS out-of-sample ----------
X = pd.get_dummies(m1d.dropna(subset=["era"])[["type", "lval", "linc", "era", "DIVISION", "year", "BEDROOMS", "BATHROOMS"]],
                   columns=["type", "era", "DIVISION"], dtype=float)
y = m1d.dropna(subset=["era"]).lcost
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
gb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(Xtr, ytr)
import statsmodels.api as sm
ols = sm.OLS(ytr, sm.add_constant(Xtr)).fit()
fit = {"M1_ols_r2_in_sample": m1.rsquared, "M1_n": int(m1.nobs), "M2_r2": m2.rsquared, "M2_n": int(m2.nobs),
       "M2_midwest_old_n": int(m2b.nobs), "M3_ols_r2_test": r2_score(yte, ols.predict(sm.add_constant(Xte, has_constant="add"))),
       "M3_gbm_r2_test": r2_score(yte, gb.predict(Xte)), "jobs_total_prof": len(jp), "jobs_stl_like": int(jp.stl_like.sum())}
pd.Series(fit).rename_axis("metric").to_csv(D / "model_fit_summary.csv", header=["value"])
pd.set_option("display.width", 220)
print(tab.sort_values("annual_spend_US", ascending=False).round(0).to_string()); print(pred.round(0)[["type", "pred_cost_stl_client"]])
print(hed.round(3).to_string()); print(fit)
