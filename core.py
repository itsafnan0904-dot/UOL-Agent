"""FYP-Pulse core logic (no streamlit import).

Sections: A1 cleaning | features | A2 dashboard stats | A3 models | A4 tiers |
A5 explainability + fairness | A6/A7 helpers | build pipeline.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "fyp_pulse_train.csv"
ART = ROOT / "artifacts"
MODEL_PATH = ART / "model.joblib"
RESULTS_PATH = ART / "results.json"
CLEAN_PATH = ART / "fyp_clean.csv"
TEST_SCORED_PATH = ART / "test_scored.csv"
RANDOM_STATE = 42

# =============================================================== schema
CAMPUSES = ["Lahore", "Karachi", "Islamabad", "Faisalabad", "Multan", "Rawalpindi", "Peshawar", "Quetta"]
CAMPUS_MAP = {c.lower(): c for c in CAMPUSES}
CAMPUS_MAP.update({"lhr": "Lahore", "khi": "Karachi", "rwp": "Rawalpindi", "pindi": "Rawalpindi",
                   "isb": "Islamabad", "fsd": "Faisalabad", "pesh": "Peshawar", "psh": "Peshawar",
                   "mul": "Multan", "qta": "Quetta"})
PROGRAMS = ["BSCS", "BSSE", "BSAI", "BSIT", "BSDS"]
PROJECT_TYPES = ["Web App", "Mobile App", "AI-ML", "IoT", "Research"]
GENDER_MAP = {"m": "Male", "male": "Male", "f": "Female", "female": "Female"}
YESNO_MAP = {"yes": "Yes", "y": "Yes", "true": "Yes", "1": "Yes", "no": "No", "n": "No", "false": "No", "0": "No"}
YESNO_COLS = ["has_industry_sponsor", "extension_granted", "poor_outcome"]
CAT_COLS_ALL = ["campus", "program", "leader_gender", "project_type", "has_industry_sponsor", "extension_granted"]

RANGES = {
    "group_size": (1, 5), "team_avg_cgpa": (0, 4), "team_min_cgpa": (0, 4),
    "supervisor_experience_years": (0, 45), "supervisor_active_groups": (1, 15),
    "proposal_revisions": (0, 10), "weeks_to_proposal_approval": (0, 30),
    "supervisor_meetings_8wk": (0, 30), "github_commits_8wk": (0, 1000),
    "mid_eval_score": (0, 100), "final_defense_score": (0, 100),
}
INT_COLS = ["group_size", "supervisor_experience_years", "supervisor_active_groups", "proposal_revisions",
            "weeks_to_proposal_approval", "supervisor_meetings_8wk", "github_commits_8wk"]

# model inputs (raw, known at week 8)
NUM_INPUTS = ["group_size", "team_avg_cgpa", "team_min_cgpa", "supervisor_experience_years",
              "supervisor_active_groups", "proposal_revisions", "weeks_to_proposal_approval",
              "supervisor_meetings_8wk", "github_commits_8wk", "mid_eval_score"]
CAT_INPUTS = ["program", "project_type", "has_industry_sponsor"]
RAW_INPUTS = CAT_INPUTS + NUM_INPUTS
ENGINEERED = ["commits_per_member", "cgpa_gap", "meetings_per_load"]
TARGET = "poor_outcome_bin"

DROPPED_FEATURES = [
    ("final_defense_score", "Leakage: only known at the final defense, after week 8."),
    ("extension_granted", "Leakage: decided at the end of the project, after week 8."),
    ("poor_outcome", "Target variable (encoded as poor_outcome_bin)."),
    ("group_id", "Identifier, carries no behavioural signal."),
    ("proposal_date", "Calendar cohort, not a behavioural signal; risks learning term-specific noise."),
    ("leader_gender", "Protected attribute: excluded from the model, audited in the Fairness tab."),
    ("campus", "Context attribute: excluded from the model to avoid location bias, audited in the Fairness tab."),
]

LABELS = {
    "group_size": "group size", "team_avg_cgpa": "team average CGPA", "team_min_cgpa": "lowest member CGPA",
    "supervisor_experience_years": "supervisor experience (years)",
    "supervisor_active_groups": "supervisor's active groups", "proposal_revisions": "proposal revisions",
    "weeks_to_proposal_approval": "weeks to proposal approval",
    "supervisor_meetings_8wk": "supervisor meetings (8 wks)", "github_commits_8wk": "GitHub commits (8 wks)",
    "mid_eval_score": "mid-evaluation score", "program": "program", "project_type": "project type",
    "has_industry_sponsor": "industry sponsor",
}

ACTIONS = {
    "supervisor_meetings_8wk": "Few supervisor meetings: schedule a fixed weekly slot with the supervisor.",
    "github_commits_8wk": "Low coding activity: set weekly commit targets and a short demo each week.",
    "mid_eval_score": "Weak mid-evaluation: hold a remediation review of the mid-eval feedback and agree a recovery plan.",
    "team_avg_cgpa": "Academically weaker team: pair with a peer mentor and offer extra technical support.",
    "team_min_cgpa": "Academically weaker team member: pair with a peer mentor and offer extra technical support.",
    "proposal_revisions": "Slow proposal approval: run a scope-clarity session and a fast-track milestone plan.",
    "weeks_to_proposal_approval": "Slow proposal approval: run a scope-clarity session and a fast-track milestone plan.",
    "supervisor_active_groups": "Supervisor overloaded: reduce the supervisor's load or assign a co-supervisor.",
    "supervisor_experience_years": "Less experienced supervisor: assign a senior co-supervisor.",
    "has_industry_sponsor": "No industry sponsor: connect the group with an industry mentor.",
    "group_size": "Team size risk: agree clear role split and individual deliverables (re-balance scope to team size).",
    "project_type": "Project-type risk: review scope and technical feasibility with the supervisor; trim to a core MVP.",
    "program": "Program-level risk: check the group has the required background; offer targeted skills workshops.",
}
NO_ACTION = "No action needed, continue routine monitoring."


# =============================================================== A1 cleaning
def _snake(c: str) -> str:
    c = re.sub(r"[^0-9a-zA-Z]+", "_", str(c).strip())
    return re.sub(r"_+", "_", c).strip("_").lower()


def _parse_date(v):
    if pd.isna(v):
        return pd.NaT
    for fmt in ("%Y-%m-%d", "%d-%b-%y", "%d-%b-%Y", "%d/%m/%Y"):
        try:
            return pd.to_datetime(str(v).strip(), format=fmt)
        except (ValueError, TypeError):
            pass
    return pd.NaT


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = raw.copy()
    cols = list(df.columns)
    counts = {}  # column -> {standardized, out_of_range, rule_fixed, imputed}

    def add(col, kind, n):
        d = counts.setdefault(col, {"standardized": 0, "out_of_range": 0, "rule_fixed": 0, "imputed": 0})
        d[kind] += int(n)

    rules = []
    q = {"initial_rows": int(len(df))}

    # 1. names, whitespace, blanks
    df.columns = [_snake(c) for c in cols]
    original = df.copy()
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):  # object or pandas-3 str dtype
            df[c] = df[c].astype("string").str.strip()
            df[c] = df[c].mask(df[c].isin(["", "nan", "NaN", "NA", "None"])).astype(object)
            df[c] = df[c].where(df[c].notna(), np.nan)

    # 2. categorical standardisation
    def std(col, mapper):
        before = df[col]
        after = before.map(lambda v: mapper(str(v)) if pd.notna(v) else np.nan)
        changed = before.notna() & after.notna() & (original[col].astype(str) != after.astype(str))
        add(col, "standardized", changed.sum())
        q.setdefault("unmapped_to_nan", {})[col] = int((before.notna() & after.isna()).sum())
        df[col] = after

    std("campus", lambda v: CAMPUS_MAP.get(re.sub(r"\s+", " ", v).strip().lower(), np.nan))
    std("leader_gender", lambda v: GENDER_MAP.get(v.lower(), np.nan))
    for c in YESNO_COLS:
        std(c, lambda v: YESNO_MAP.get(v.lower(), np.nan))
    std("program", lambda v: {p.lower(): p for p in PROGRAMS}.get(v.lower(), np.nan))
    std("project_type", lambda v: {p.lower(): p for p in PROJECT_TYPES}.get(v.lower().replace("_", " "), np.nan))

    # 3. dates
    rawd = df["proposal_date"]
    iso = rawd.astype("string").str.fullmatch(r"\d{4}-\d{2}-\d{2}").fillna(False).astype(bool)
    parsed = pd.to_datetime(rawd.map(_parse_date))
    q["dates_non_iso_converted"] = int((~iso & parsed.notna()).sum())
    q["dates_unparseable"] = int((rawd.notna() & parsed.isna()).sum())
    add("proposal_date", "standardized", q["dates_non_iso_converted"])
    df["proposal_date"] = parsed

    # 4. duplicates
    n = len(df)
    keep = ~df.duplicated(keep="first")
    df, original = df[keep], original[keep]
    exact = n - len(df)
    n = len(df)
    keep = ~df["group_id"].duplicated(keep="first")
    df, original = df[keep].reset_index(drop=True), original[keep].reset_index(drop=True)
    gid = n - len(df)
    q["rows_after_exact_dedup"] = int(len(df) + gid)
    q["duplicates_removed"] = {"exact_duplicate_rows": int(exact), "duplicate_group_id": int(gid)}

    # 5. target
    miss_t = df["poor_outcome"].isna()
    q["rows_dropped_missing_target"] = int(miss_t.sum())
    df = df[~miss_t].reset_index(drop=True)
    df[TARGET] = (df["poor_outcome"] == "Yes").astype(int)

    # 6. range checks
    for c, (lo, hi) in RANGES.items():
        num = pd.to_numeric(df[c], errors="coerce")
        bad = num.notna() & ((num < lo) | (num > hi))
        if c in INT_COLS:
            bad |= num.notna() & (num != np.round(num))
        bad |= df[c].notna() & num.isna()  # non-numeric text
        add(c, "out_of_range", bad.sum())
        df[c] = num.mask(bad).astype(float)

    # 7. CGPA rules
    solo = (df["group_size"] == 1) & df["team_avg_cgpa"].notna() & (df["team_min_cgpa"] != df["team_avg_cgpa"])
    df.loc[solo, "team_min_cgpa"] = df.loc[solo, "team_avg_cgpa"]
    add("team_min_cgpa", "rule_fixed", solo.sum())
    rules.append({"rule": "Solo group (group_size = 1): team_min_cgpa must equal team_avg_cgpa",
                  "rows_broken": int(solo.sum()), "how_fixed": "team_min_cgpa set equal to team_avg_cgpa"})
    viol = df["team_min_cgpa"].notna() & df["team_avg_cgpa"].notna() & (df["team_min_cgpa"] > df["team_avg_cgpa"])
    df.loc[viol, "team_min_cgpa"] = np.nan
    add("team_min_cgpa", "rule_fixed", viol.sum())
    rules.append({"rule": "team_min_cgpa must not exceed team_avg_cgpa", "rows_broken": int(viol.sum()),
                  "how_fixed": "team_min_cgpa set to NaN, then imputed as avg - median gap"})

    # 8. imputation
    q["missing_before_imputation"] = {c: int(v) for c, v in df.isna().sum().items()}
    for c in RANGES:
        if c in ("team_avg_cgpa", "team_min_cgpa"):
            continue
        med = df[c].median()
        med = float(np.round(med)) if c in INT_COLS else float(med)
        add(c, "imputed", df[c].isna().sum())
        df[c] = df[c].fillna(med)
    both = df["team_avg_cgpa"].notna() & df["team_min_cgpa"].notna()
    gap = float((df.loc[both, "team_avg_cgpa"] - df.loc[both, "team_min_cgpa"]).median())
    q["median_cgpa_gap"] = round(gap, 4)
    solo_now = df["group_size"] == 1
    a_miss, m_miss = df["team_avg_cgpa"].isna(), df["team_min_cgpa"].isna()
    # avg missing, min present -> avg = min + gap (solo: avg = min)
    r = a_miss & ~m_miss
    df.loc[r, "team_avg_cgpa"] = np.where(solo_now[r], df.loc[r, "team_min_cgpa"],
                                          np.minimum(df.loc[r, "team_min_cgpa"] + gap, 4.0))
    df.loc[a_miss & m_miss, "team_avg_cgpa"] = float(df["team_avg_cgpa"].median())
    add("team_avg_cgpa", "imputed", a_miss.sum())
    df.loc[m_miss, "team_min_cgpa"] = np.clip(df.loc[m_miss, "team_avg_cgpa"] - gap, 0, df.loc[m_miss, "team_avg_cgpa"])
    add("team_min_cgpa", "imputed", m_miss.sum())
    for c in CAT_COLS_ALL:
        add(c, "imputed", df[c].isna().sum())
        df[c] = df[c].fillna(df[c].mode().iloc[0])
    nd = df["proposal_date"].isna().sum()
    if nd:
        df["proposal_date"] = df["proposal_date"].fillna(df["proposal_date"].median())
        add("proposal_date", "imputed", nd)
    # re-enforce
    post1 = df["team_min_cgpa"] > df["team_avg_cgpa"]
    df.loc[post1, "team_min_cgpa"] = df.loc[post1, "team_avg_cgpa"]
    post2 = (df["group_size"] == 1) & (df["team_min_cgpa"] != df["team_avg_cgpa"])
    df.loc[post2, "team_min_cgpa"] = df.loc[post2, "team_avg_cgpa"]
    add("team_min_cgpa", "rule_fixed", post1.sum() + post2.sum())
    rules.append({"rule": "Re-check after imputation: min <= avg and solo min == avg",
                  "rows_broken": int(post1.sum() + post2.sum()), "how_fixed": "team_min_cgpa set to team_avg_cgpa"})

    # 9. dtypes
    for c in INT_COLS:
        df[c] = df[c].round().astype(int)
    for c in ("team_avg_cgpa", "team_min_cgpa", "mid_eval_score", "final_defense_score"):
        df[c] = df[c].round(2)
    df["proposal_date"] = df["proposal_date"].dt.strftime("%Y-%m-%d")

    table = []
    for c, d in counts.items():
        table.append({"column": c, **d, "total": int(sum(d.values()))})
    q.update({
        "cleaned_rows": int(len(df)),
        "rule_violations": rules,
        "corrections_per_column": sorted(table, key=lambda r: -r["total"]),
        "total_poor_outcomes": int(df[TARGET].sum()),
        "poor_outcome_rate_pct": round(100 * df[TARGET].mean(), 2),
    })
    return df, q


def verify_clean(df: pd.DataFrame) -> list[str]:
    for c, (lo, hi) in RANGES.items():
        assert df[c].between(lo, hi).all(), f"{c} out of range"
    assert sorted(df["campus"].unique()) == sorted(CAMPUSES), df["campus"].unique()
    assert set(df["leader_gender"]) == {"Male", "Female"}
    assert not df["group_id"].duplicated().any()
    assert (df["team_min_cgpa"] <= df["team_avg_cgpa"]).all()
    s = df["group_size"] == 1
    assert (df.loc[s, "team_min_cgpa"] == df.loc[s, "team_avg_cgpa"]).all()
    assert df.isna().sum().sum() == 0
    return ["no out-of-range values", "exactly 8 campuses", "2 genders", "no duplicate group_id",
            "min <= avg everywhere", f"min == avg for all {int(s.sum())} solo groups", "no NaN left"]


# =============================================================== features
def add_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    gs = pd.to_numeric(X["group_size"], errors="coerce").clip(lower=1)
    sag = pd.to_numeric(X["supervisor_active_groups"], errors="coerce").clip(lower=1)
    X["commits_per_member"] = pd.to_numeric(X["github_commits_8wk"], errors="coerce") / gs
    X["cgpa_gap"] = pd.to_numeric(X["team_avg_cgpa"], errors="coerce") - pd.to_numeric(X["team_min_cgpa"], errors="coerce")
    X["meetings_per_load"] = pd.to_numeric(X["supervisor_meetings_8wk"], errors="coerce") / sag
    return X


# =============================================================== A2 dashboard
def rate_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby(col, observed=True)[TARGET].agg(["mean", "size"]).reset_index()
    g.columns = [col, "poor_rate_pct", "groups"]
    g["poor_rate_pct"] = (100 * g["poor_rate_pct"]).round(1)
    g[col] = g[col].astype(str)
    return g


def dashboard_tables(df: pd.DataFrame) -> dict:
    d = add_features(df)
    d["CGPA band"] = pd.cut(d["team_avg_cgpa"], [-0.01, 2.5, 3.0, 3.5, 4.01],
                            labels=["1: <2.5", "2: 2.5-3.0", "3: 3.0-3.5", "4: >=3.5"], right=False)
    d["Meetings (8 wks)"] = pd.cut(d["supervisor_meetings_8wk"], [-1, 2, 4, 6, 8, 31],
                                   labels=["1: 0-2", "2: 3-4", "3: 5-6", "4: 7-8", "5: 9+"])
    d["Supervisor load"] = pd.cut(d["supervisor_active_groups"], [0, 4, 7, 10, 16],
                                  labels=["1: 1-4", "2: 5-7", "3: 8-10", "4: 11+"])
    d["Project / sponsor"] = d["project_type"] + " | sponsor " + d["has_industry_sponsor"]
    q = pd.qcut(d["commits_per_member"].rank(method="first"), 4, labels=False)
    edges = d.groupby(q)["commits_per_member"].agg(["min", "max"]).round(1)
    names = {i: f"Q{i + 1}: {edges.loc[i, 'min']}-{edges.loc[i, 'max']}" for i in edges.index}
    d["Commits per member (quartile)"] = q.map(names)
    out = {}
    for key in ["CGPA band", "Meetings (8 wks)", "Supervisor load", "project_type", "Project / sponsor",
                "Commits per member (quartile)"]:
        out[key] = rate_by(d, key)
    return out


def insight(t: pd.DataFrame, col: str, min_groups: int = 20) -> str:
    t = t[t["groups"] >= min_groups]
    hi, lo = t.loc[t["poor_rate_pct"].idxmax()], t.loc[t["poor_rate_pct"].idxmin()]
    return (f"Highest poor-outcome rate: {hi[col]} at {hi['poor_rate_pct']}% ({int(hi['groups'])} groups); "
            f"lowest: {lo[col]} at {lo['poor_rate_pct']}%, a{hi['poor_rate_pct'] - lo['poor_rate_pct']:.1f} pp gap.")


# =============================================================== A3 models
def make_pipeline(estimator) -> Pipeline:
    num = NUM_INPUTS + ENGINEERED
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT_INPUTS),
    ])
    return Pipeline([("fe", FunctionTransformer(add_features)), ("pre", pre), ("clf", estimator)])


def candidate_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
                                                n_jobs=-1, random_state=RANDOM_STATE),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                                               class_weight="balanced", random_state=RANDOM_STATE),
    }


def best_f1_threshold(y, p) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.nanargmax(f1))])


def metrics(y, p, thr) -> dict:
    yhat = (p >= thr).astype(int)
    return {"accuracy": accuracy_score(y, yhat), "precision": precision_score(y, yhat, zero_division=0),
            "recall": recall_score(y, yhat), "f1": f1_score(y, yhat), "roc_auc": roc_auc_score(y, p),
            "confusion_matrix": confusion_matrix(y, yhat).tolist()}


# =============================================================== A4 tiers
def tier_of(p: float, low: float, high: float) -> str:
    return "High" if p >= high else ("Medium" if p >= low else "Low")


def tier_table(y, p, low, high) -> pd.DataFrame:
    t = pd.DataFrame({"y": np.asarray(y), "tier": [tier_of(v, low, high) for v in p]})
    g = t.groupby("tier")["y"].agg(["size", "sum", "mean"]).reindex(["Low", "Medium", "High"])
    g.columns = ["groups", "actual_poor", "observed_poor_rate_pct"]
    g["observed_poor_rate_pct"] = (100 * g["observed_poor_rate_pct"]).round(1)
    return g.reset_index()


def choose_tiers(oof_y, oof_p, test_y, test_p, threshold) -> dict:
    def ok(lo, hi):
        t = tier_table(test_y, test_p, lo, hi)
        r = t["observed_poor_rate_pct"].tolist()
        return (t["groups"].fillna(0) >= 10).all() and r[0] < r[1] < r[2]

    # default: High = model threshold (consistent with the predicted label); Low = OOF median prob
    hi = threshold
    lo = float(min(np.quantile(oof_p, 0.5), threshold * 0.6))
    note = "Default cut-offs (Low < OOF median prob, High >= model decision threshold) gave Low < Medium < High."
    if not ok(lo, hi):
        found = None
        for qh in np.arange(0.95, 0.69, -0.025):
            for ql in np.arange(0.30, 0.71, 0.05):
                l, h = float(np.quantile(oof_p, ql)), float(np.quantile(oof_p, qh))
                if l < h and ok(l, h):
                    found = (l, h, ql, qh)
                    break
            if found:
                break
        if found is None:
            raise RuntimeError("No monotonic tier cut-offs found")
        lo, hi = found[0], found[1]
        note = (f"Default cut-offs were not monotonic on test; searched OOF quantiles and used "
                f"Low<q{found[2]:.2f}, High>=q{found[3]:.2f}.")
    return {"low": round(lo, 4), "high": round(hi, 4), "note": note}


# =============================================================== A5 explainability
def references(train: pd.DataFrame) -> dict:
    ref = {c: float(train[c].median()) for c in NUM_INPUTS}
    for c in INT_COLS:
        if c in ref:
            ref[c] = int(round(ref[c]))
    ref.update({c: str(train[c].mode().iloc[0]) for c in CAT_INPUTS})
    if ref["team_min_cgpa"] > ref["team_avg_cgpa"]:
        ref["team_min_cgpa"] = ref["team_avg_cgpa"]
    return ref


def _ref_value(feature, row, ref):
    """Reference value; keeps CGPA pair consistent (min <= avg, solo min == avg)."""
    return ref[feature]


def contributions(model, X: pd.DataFrame, ref: dict) -> pd.DataFrame:
    """Per-row, per-feature contribution = p_actual - p_with_feature_at_reference (batch)."""
    X = X[RAW_INPUTS].reset_index(drop=True)
    base = model.predict_proba(X)[:, 1]
    out = {}
    for f in RAW_INPUTS:
        Xr = X.copy()
        Xr[f] = ref[f]
        if f == "team_min_cgpa":
            Xr[f] = np.minimum(Xr[f], Xr["team_avg_cgpa"])
        if f == "team_avg_cgpa":
            Xr[f] = np.maximum(Xr[f], Xr["team_min_cgpa"])
        out[f] = base - model.predict_proba(Xr)[:, 1]
    c = pd.DataFrame(out)
    c.insert(0, "_p", base)
    return c


def _fmt(f, v):
    if f in CAT_INPUTS:
        return str(v)
    return f"{v:.2f}" if isinstance(v, float) and not float(v).is_integer() else f"{int(v)}"


def explain_one(model, row: dict, ref: dict, low: float, high: float) -> dict:
    X = pd.DataFrame([row])
    c = contributions(model, X, ref).iloc[0]
    p = float(c["_p"])
    contrib = c.drop("_p").astype(float)
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:3]
    tier = tier_of(p, low, high)
    up = [f for f in top.index if top[f] > 0.005]
    down = [f for f in top.index if top[f] < -0.005]
    parts = [f"This group is rated {tier} risk ({100 * p:.0f}%)."]
    if up:
        parts.append("The biggest reasons are: " + " and ".join(
            f"{LABELS[f]} of {_fmt(f, row[f])} (typical {_fmt(f, ref[f])})" for f in up) + ".")
    else:
        parts.append("None of its inputs pushes the risk above a typical group by much.")
    if down:
        parts.append("Its " + " and ".join(f"{LABELS[f]} ({_fmt(f, row[f])})" for f in down)
                     + (" helps" if len(down) == 1 else " help") + " reduce the risk.")
    raisers = contrib[contrib > 0.005].sort_values(ascending=False)
    action = ACTIONS.get(raisers.index[0], NO_ACTION) if len(raisers) else NO_ACTION
    return {"p": p, "tier": tier, "contrib": contrib, "top": top, "text": " ".join(parts), "action": action,
            "top_driver": raisers.index[0] if len(raisers) else None}


# =============================================================== A5 fairness
def group_stats(d: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for k, g in d.groupby(col):
        pos = g["y"].sum()
        rows.append({col: k, "groups": len(g), "actual_positives": int(pos),
                     "predicted_poor_rate_pct": round(100 * g["pred"].mean(), 1),
                     "actual_poor_rate_pct": round(100 * g["y"].mean(), 1),
                     "recall_pct": round(100 * g.loc[g["y"] == 1, "pred"].mean(), 1) if pos else None})
    return pd.DataFrame(rows)


def fairness(test_df: pd.DataFrame, flag_pp: float = 10.0) -> dict:
    res = {}
    gender = group_stats(test_df, "leader_gender")
    campus = group_stats(test_df, "campus")
    campus_ok = campus[campus["groups"] >= 30].reset_index(drop=True)
    res["gender"] = gender.to_dict("records")
    res["campus"] = campus_ok.to_dict("records")
    res["campus_excluded_lt30"] = campus[campus["groups"] < 30]["campus"].tolist()

    def gap(t, key, metric):
        t = t.dropna(subset=[metric])
        hi, lo = t.loc[t[metric].idxmax()], t.loc[t[metric].idxmin()]
        return {"high": hi[key], "high_value": hi[metric], "low": lo[key], "low_value": lo[metric],
                "gap_pp": round(hi[metric] - lo[metric], 1)}

    res["gender_gap_predicted"] = gap(gender, "leader_gender", "predicted_poor_rate_pct")
    res["gender_gap_recall"] = gap(gender, "leader_gender", "recall_pct")
    res["campus_gap_predicted"] = gap(campus_ok, "campus", "predicted_poor_rate_pct")
    res["campus_gap_recall"] = gap(campus_ok, "campus", "recall_pct")

    all_t = pd.concat([gender.rename(columns={"leader_gender": "g"}), campus_ok.rename(columns={"campus": "g"})])
    min_n, min_pos = int(all_t["groups"].min()), int(all_t["actual_positives"].min())
    low_conf = min_n < 100 or min_pos < 20
    flags = []
    for name, k in [("gender predicted rate", "gender_gap_predicted"), ("gender recall", "gender_gap_recall"),
                    ("campus predicted rate", "campus_gap_predicted"), ("campus recall", "campus_gap_recall")]:
        if res[k]["gap_pp"] >= flag_pp:
            flags.append(f"{name} ({res[k]['high']} vs {res[k]['low']}: {res[k]['gap_pp']} pp)")
    take = (("Disparity flagged (>= 10 pp): " + "; ".join(flags) + ". ") if flags
            else "No gap reaches the 10 pp flag. ")
    take += (f"Confidence is {'LOW' if low_conf else 'reasonable'}: the smallest audited group has {min_n} test groups "
             f"and as few as {min_pos} actual poor outcomes, so recall gaps in particular can move a lot by chance.")
    res.update({"flags": flags, "low_confidence": low_conf, "min_group_n": min_n, "min_group_positives": min_pos,
                "takeaway": take})
    return res


# =============================================================== A6/A7 helpers
LEVERS = {
    "supervisor_meetings_8wk": ("+4 meetings", lambda r: {**r, "supervisor_meetings_8wk": min(30, r["supervisor_meetings_8wk"] + 4)}),
    "github_commits_8wk": ("+50% commits (min +40)", lambda r: {**r, "github_commits_8wk": int(min(1000, max(r["github_commits_8wk"] * 1.5, r["github_commits_8wk"] + 40)))}),
    "supervisor_active_groups": ("-3 supervisor groups", lambda r: {**r, "supervisor_active_groups": max(1, r["supervisor_active_groups"] - 3)}),
    "has_industry_sponsor": ("add industry sponsor", lambda r: {**r, "has_industry_sponsor": "Yes"}),
}


def best_lever(model, row: dict) -> list[dict]:
    p0 = float(model.predict_proba(pd.DataFrame([row])[RAW_INPUTS])[:, 1][0])
    out = []
    for k, (desc, fn) in LEVERS.items():
        p1 = float(model.predict_proba(pd.DataFrame([fn(row)])[RAW_INPUTS])[:, 1][0])
        out.append({"lever": k, "change": desc, "p_before": p0, "p_after": p1, "drop_pp": 100 * (p0 - p1)})
    return sorted(out, key=lambda d: -d["drop_pp"])


# =============================================================== build
def build(verbose: bool = True) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    ART.mkdir(exist_ok=True)
    raw = pd.read_csv(DATA_PATH, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df, dq = clean(raw)
    checks = verify_clean(df)
    log("Cleaning assertions passed:", "; ".join(checks))
    df.to_csv(CLEAN_PATH, index=False)

    X, y = df, df[TARGET].values
    tr_idx, te_idx = train_test_split(np.arange(len(df)), test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    train, test = df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)
    ytr, yte = train[TARGET].values, test[TARGET].values
    skf = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

    results, fitted, oofs = {}, {}, {}
    for name, est in candidate_models().items():
        pipe = make_pipeline(est)
        oof = cross_val_predict(pipe, train[RAW_INPUTS], ytr, cv=skf, method="predict_proba")[:, 1]
        thr = best_f1_threshold(ytr, oof)
        pipe.fit(train[RAW_INPUTS], ytr)
        pte = pipe.predict_proba(test[RAW_INPUTS])[:, 1]
        m = metrics(yte, pte, thr)
        m.update({"threshold": thr, "cv_roc_auc": roc_auc_score(ytr, oof), "cv_f1_at_threshold": f1_score(ytr, oof >= thr)})
        results[name], fitted[name], oofs[name] = m, pipe, oof
        log(f"{name:22s} CV-AUC={m['cv_roc_auc']:.4f} test AUC={m['roc_auc']:.4f} F1={m['f1']:.4f} "
            f"recall={m['recall']:.4f} thr={thr:.4f}")

    final = sorted(results, key=lambda k: (round(results[k]["cv_roc_auc"], 3), results[k]["recall"], results[k]["f1"]))[-1]
    model, thr = fitted[final], results[final]["threshold"]
    pte = model.predict_proba(test[RAW_INPUTS])[:, 1]
    tiers = choose_tiers(ytr, oofs[final], yte, pte, thr)
    ttab = tier_table(yte, pte, tiers["low"], tiers["high"])
    log("Tiers:", tiers, "\n", ttab.to_string(index=False))

    ref = references(train)
    contrib = contributions(model, test, ref)
    drivers = contrib.drop(columns="_p").apply(lambda r: r.idxmax() if r.max() > 0.005 else "none", axis=1)
    scored = test[["group_id", "leader_gender", "campus"] + RAW_INPUTS + [TARGET]].copy()
    scored["probability"] = pte
    scored["predicted_poor"] = (pte >= thr).astype(int)
    scored["tier"] = [tier_of(p, tiers["low"], tiers["high"]) for p in pte]
    scored["top_risk_driver"] = drivers.values
    scored.to_csv(TEST_SCORED_PATH, index=False)

    fair = fairness(scored.rename(columns={TARGET: "y", "predicted_poor": "pred"}))
    defaults = {c: (int(v) if c in INT_COLS else v) for c, v in ref.items()}

    best = results[final]
    justification = (
        f"{final} has the highest 5-fold cross-validated ROC-AUC on the training set ({best['cv_roc_auc']:.3f}), "
        f"so it ranks risky groups best on data it was not fit to; on the held-out test set it reaches recall "
        f"{best['recall']:.2f}, F1 {best['f1']:.2f} and ROC-AUC {best['roc_auc']:.3f}. "
        f"With only {dq['poor_outcome_rate_pct']:.1f}% poor outcomes, a model that predicts 'pass' for everyone is "
        f"~{100 - dq['poor_outcome_rate_pct']:.0f}% accurate yet useless, so accuracy is misleading. For an early-warning "
        f"tool a missed at-risk group (false negative) costs far more than an extra check-in, so recall and F1 at a "
        f"tuned threshold ({thr:.3f}, chosen on out-of-fold predictions) matter most.")

    out = {
        "data_quality": dq, "cleaning_checks": checks,
        "dropped_features": [{"feature": f, "reason": r} for f, r in DROPPED_FEATURES],
        "model_features": RAW_INPUTS + ENGINEERED,
        "split": {"train": int(len(train)), "test": int(len(test)), "test_positives": int(yte.sum())},
        "models": results, "final_model": final, "threshold": thr, "justification": justification,
        "tiers": tiers, "tier_table_test": ttab.to_dict("records"),
        "reference": ref, "defaults": defaults, "fairness": fair,
        "base_rate_test": float(yte.mean()),
    }
    joblib.dump(model, MODEL_PATH)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log(f"Final model: {final} (threshold {thr:.4f}). Artifacts saved to {ART}")
    return out


def artifacts_exist() -> bool:
    return all(p.exists() for p in (MODEL_PATH, RESULTS_PATH, CLEAN_PATH, TEST_SCORED_PATH))
