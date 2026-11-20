"""FYP-Pulse: Final Year Project Early Warning System (Streamlit UI). All logic lives in core.py."""
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import core

st.set_page_config(page_title="FYP-Pulse", page_icon="📊", layout="wide")


def _read_artifacts():
    model = joblib.load(core.MODEL_PATH)
    with open(core.RESULTS_PATH, encoding="utf-8") as f:
        res = json.load(f)
    clean_df = pd.read_csv(core.CLEAN_PATH, dtype={"group_id": str})
    scored = pd.read_csv(core.TEST_SCORED_PATH, dtype={"group_id": str})
    # smoke test: a stale/incompatible model fails here and triggers a rebuild
    model.predict_proba(pd.DataFrame([res["defaults"]])[core.RAW_INPUTS])
    return model, res, clean_df, scored


@st.cache_resource(show_spinner="Loading models (rebuilds once if artifacts are stale)...")
def load_all():
    try:
        if not core.artifacts_exist():
            raise FileNotFoundError("artifacts missing")
        return _read_artifacts()
    except Exception:
        core.build(verbose=False)  # missing or stale (e.g. scikit-learn version mismatch): rebuild once
        return _read_artifacts()


@st.cache_data
def dash_tables(df):
    return core.dashboard_tables(df)


model, R, df, scored = load_all()
dq, thr = R["data_quality"], R["threshold"]
LOW, HIGH = R["tiers"]["low"], R["tiers"]["high"]
REF, DEF = R["reference"], R["defaults"]
COLORS = {"Low": "#2e9e4f", "Medium": "#e0b000", "High": "#d33a3a"}


def badge(tier):
    return (f"<span style='background:{COLORS[tier]};color:{'#000' if tier == 'Medium' else '#fff'};"
            f"padding:4px 14px;border-radius:14px;font-weight:700'>{tier} risk</span>")


st.title("FYP-Pulse: Final Year Project Early Warning System")
st.caption("Predicts which groups are likely to end with a poor outcome using only information known at the end of week 8.")

tabs = st.tabs(["Data Quality", "Dashboard", "Models", "Predict", "Fairness", "Watchlist", "What-if"])

# ------------------------------------------------------------------ Data Quality
with tabs[0]:
    st.header("Data Quality")
    c = st.columns(5)
    c[0].metric("Initial rows", dq["initial_rows"])
    c[1].metric("Cleaned rows", dq["cleaned_rows"])
    c[2].metric("Duplicates removed", sum(dq["duplicates_removed"].values()))
    c[3].metric("Poor outcomes", dq["total_poor_outcomes"])
    c[4].metric("Poor-outcome rate", f"{dq['poor_outcome_rate_pct']}%")
    st.write(f"Duplicates: **{dq['duplicates_removed']['exact_duplicate_rows']}** exact duplicate rows and "
             f"**{dq['duplicates_removed']['duplicate_group_id']}** repeated group_id. "
             f"Non-ISO dates converted: **{dq['dates_non_iso_converted']}** (unparseable: {dq['dates_unparseable']}). "
             f"Rows dropped for missing target: {dq['rows_dropped_missing_target']}.")
    st.subheader("Rule violations")
    st.dataframe(pd.DataFrame(dq["rule_violations"]).rename(
        columns={"rule": "Rule", "rows_broken": "Rows broken", "how_fixed": "How fixed"}), hide_index=True, width='stretch')
    st.subheader("Values corrected per column")
    st.dataframe(pd.DataFrame(dq["corrections_per_column"]), hide_index=True, width='stretch')
    st.subheader("Missing values per column (before imputation)")
    miss = pd.Series(dq["missing_before_imputation"], name="missing").to_frame()
    st.dataframe(miss[miss["missing"] > 0], width='stretch')
    st.caption("Cleaning assertions passed: " + "; ".join(R["cleaning_checks"]) + ".")

# ------------------------------------------------------------------ Dashboard
with tabs[1]:
    st.header("Dashboard: what separates risky groups?")
    T = dash_tables(df)

    def chart(key, title, x, note=None):
        st.subheader(title)
        t = T[key]
        st.bar_chart(t.set_index(key)["poor_rate_pct"], y_label="Poor-outcome rate (%)")
        st.info(core.insight(t, key))
        if note:
            st.caption(note)

    a, b = st.columns(2)
    with a:
        chart("CGPA band", "1. Team factor: poor-outcome rate by team average CGPA", "")
    with b:
        chart("Meetings (8 wks)", "2. Supervisor factor: by supervisor meetings in 8 weeks", "")
    a, b = st.columns(2)
    with a:
        chart("Supervisor load", "2b. Supervisor factor: by supervisor's active groups", "")
    with b:
        chart("project_type", "3. Project context: by project type", "")
    chart("Project / sponsor", "3b. Project context: project type and industry sponsor", "")
    chart("Commits per member (quartile)", "4. Engineered feature: GitHub commits per member (quartiles)", "",
          note="commits_per_member = github_commits_8wk / group_size.")
    st.markdown(
        "**What `commits_per_member` measures:** each member's share of coding activity in the first 8 weeks. "
        "**Why it matters:** a large team can show a high total commit count while each person does very little; "
        "low per-member activity is an early sign of disengagement. Two other engineered features feed the model: "
        "`cgpa_gap` (team average minus lowest member CGPA, the weakest-link spread) and "
        "`meetings_per_load` (supervisor meetings divided by the supervisor's active groups).")

# ------------------------------------------------------------------ Models
with tabs[2]:
    st.header("Models")
    st.subheader("Dropped features and why")
    st.dataframe(pd.DataFrame(R["dropped_features"]), hide_index=True, width='stretch')
    st.write("**Features used:** " + ", ".join(R["model_features"]))
    st.write(f"Stratified 80/20 split: {R['split']['train']} train / {R['split']['test']} test "
             f"({R['split']['test_positives']} poor outcomes in test). Thresholds chosen on 5-fold out-of-fold predictions.")
    rows = []
    for n, m in R["models"].items():
        rows.append({"Model": n + (" (chosen)" if n == R["final_model"] else ""), "Threshold": round(m["threshold"], 4),
                     "Accuracy": round(m["accuracy"], 3), "Precision": round(m["precision"], 3),
                     "Recall": round(m["recall"], 3), "F1": round(m["f1"], 3),
                     "Test ROC-AUC": round(m["roc_auc"], 3), "CV ROC-AUC": round(m["cv_roc_auc"], 3)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
    cols = st.columns(len(R["models"]))
    for col, (n, m) in zip(cols, R["models"].items()):
        with col:
            st.write(f"**{n}**")
            cm = m["confusion_matrix"]
            st.dataframe(pd.DataFrame(cm, index=["Actual: pass", "Actual: poor"], columns=["Pred: pass", "Pred: poor"]))
    st.subheader("Why this model")
    st.success(f"Final model: **{R['final_model']}**. " + R["justification"])

# ------------------------------------------------------------------ Predict
with tabs[3]:
    st.header("Predict: week-8 risk check")
    with st.form("predict"):
        c1, c2, c3 = st.columns(3)
        program = c1.selectbox("Program", core.PROGRAMS, index=core.PROGRAMS.index(DEF["program"]))
        ptype = c2.selectbox("Project type", core.PROJECT_TYPES, index=core.PROJECT_TYPES.index(DEF["project_type"]))
        spons = c3.selectbox("Industry sponsor", ["No", "Yes"], index=["No", "Yes"].index(DEF["has_industry_sponsor"]))
        gs = c1.number_input("Group size", 1, 5, int(DEF["group_size"]))
        avg = c2.number_input("Team average CGPA", 0.0, 4.0, float(DEF["team_avg_cgpa"]), 0.01)
        mn = c3.number_input("Team minimum CGPA", 0.0, 4.0, float(DEF["team_min_cgpa"]), 0.01)
        exp = c1.number_input("Supervisor experience (years)", 0, 45, int(DEF["supervisor_experience_years"]))
        act = c2.number_input("Supervisor active groups", 1, 15, int(DEF["supervisor_active_groups"]))
        rev = c3.number_input("Proposal revisions", 0, 10, int(DEF["proposal_revisions"]))
        wk = c1.number_input("Weeks to proposal approval", 0, 30, int(DEF["weeks_to_proposal_approval"]))
        mt = c2.number_input("Supervisor meetings (8 wks)", 0, 30, int(DEF["supervisor_meetings_8wk"]))
        cm_ = c3.number_input("GitHub commits (8 wks)", 0, 1000, int(DEF["github_commits_8wk"]))
        mid = c1.number_input("Mid-evaluation score", 0.0, 100.0, float(DEF["mid_eval_score"]), 0.5)
        go = st.form_submit_button("Predict risk")
    if go:
        if mn > avg:
            st.error("Team minimum CGPA cannot exceed team average CGPA.")
            st.stop()
        if gs == 1 and abs(mn - avg) > 1e-9:
            st.error("For a group of one, minimum CGPA must equal average CGPA.")
            st.stop()
        row = dict(program=program, project_type=ptype, has_industry_sponsor=spons, group_size=gs,
                   team_avg_cgpa=avg, team_min_cgpa=mn, supervisor_experience_years=exp,
                   supervisor_active_groups=act, proposal_revisions=rev, weeks_to_proposal_approval=wk,
                   supervisor_meetings_8wk=mt, github_commits_8wk=cm_, mid_eval_score=mid)
        ex = core.explain_one(model, row, REF, LOW, HIGH)
        p = ex["p"]
        st.session_state["last_row"] = row
        label = "Poor outcome likely" if p >= thr else "Likely to pass"
        st.subheader(label)
        st.markdown(f"Probability of poor outcome: **{p:.4f}** ({100 * p:.1f}%) &nbsp; {badge(ex['tier'])}",
                    unsafe_allow_html=True)
        st.caption(f"Model decision threshold: {thr:.4f}. Tier cut-offs: Low < {LOW:.4f} <= Medium < {HIGH:.4f} <= High.")
        st.markdown("#### Why?")
        st.write(ex["text"])
        top = ex["top"]
        tdf = pd.DataFrame({"Feature": [core.LABELS[f] for f in top.index],
                            "Entered value": [core._fmt(f, row[f]) for f in top.index],
                            "Effect": ["▲ raises risk" if v > 0 else "▼ lowers risk" for v in top.values],
                            "Size (pp)": [round(100 * v, 1) for v in top.values]})
        st.dataframe(tdf, hide_index=True, width='stretch')
        cdf = pd.DataFrame({"effect_pp": 100 * ex["contrib"]})
        cdf.index = [core.LABELS[f] for f in cdf.index]
        st.bar_chart(cdf.sort_values("effect_pp"), y_label="Effect on risk (pp, + raises)")
        st.markdown("#### Suggested support step")
        st.warning(ex["action"])
    st.divider()
    st.subheader("Risk tiers: cut-offs and observed rates on the test set")
    st.write(f"Cut-offs from training out-of-fold probabilities: **Low** below {LOW:.4f}, **Medium** from {LOW:.4f} "
             f"to below {HIGH:.4f}, **High** at {HIGH:.4f} or above. {R['tiers']['note']}")
    tt = pd.DataFrame(R["tier_table_test"]).rename(columns={"tier": "Tier", "groups": "Groups", "actual_poor": "Actual poor",
                                                            "observed_poor_rate_pct": "Observed poor rate (%)"})
    st.dataframe(tt, hide_index=True, width='stretch')
    st.caption(f"Base rate in test set: {100 * R['base_rate_test']:.1f}%.")

# ------------------------------------------------------------------ Fairness
with tabs[4]:
    st.header("Fairness check (test set, final model)")
    F = R["fairness"]
    st.subheader("By leader gender")
    st.dataframe(pd.DataFrame(F["gender"]), hide_index=True, width='stretch')
    st.subheader("By campus (campuses with at least 30 test groups)")
    st.dataframe(pd.DataFrame(F["campus"]), hide_index=True, width='stretch')
    if F["campus_excluded_lt30"]:
        st.caption("Excluded (fewer than 30 test groups): " + ", ".join(F["campus_excluded_lt30"]))
    for k, name in [("campus_gap_predicted", "Campus gap in predicted rate"), ("campus_gap_recall", "Campus gap in recall"),
                    ("gender_gap_predicted", "Gender gap in predicted rate"), ("gender_gap_recall", "Gender gap in recall")]:
        g = F[k]
        st.write(f"**{name}:** {g['high']} ({g['high_value']}%) vs {g['low']} ({g['low_value']}%) = **{g['gap_pp']} pp**")
    (st.warning if F["flags"] or F["low_confidence"] else st.success)(F["takeaway"])
    st.caption("Gender and campus are NOT model inputs; this is an audit of outcomes across groups.")

# ------------------------------------------------------------------ Watchlist (A6)
with tabs[5]:
    st.header("Watchlist: highest-risk groups (test set)")
    K = st.slider("Number of groups to watch (K)", 10, 100, 30, 5)
    wl = scored.sort_values("probability", ascending=False).head(K).copy()
    wl["top_risk_driver"] = wl["top_risk_driver"].map(lambda f: core.LABELS.get(f, "none"))
    show = wl[["group_id", "probability", "tier", "top_risk_driver"]].reset_index(drop=True)
    show["probability"] = show["probability"].round(4)
    caught = int(wl[core.TARGET].sum())
    total_pos = int(scored[core.TARGET].sum())
    base = scored[core.TARGET].mean()
    c = st.columns(4)
    c[0].metric("Truly poor in top K", f"{caught} of {total_pos}")
    c[1].metric("Precision@K", f"{100 * caught / K:.1f}%")
    c[2].metric("Expected by random pick", f"{K * base:.1f}")
    c[3].metric("Lift vs random", f"{caught / max(K * base, 1e-9):.2f}x")
    st.dataframe(show, hide_index=True, width='stretch')
    st.download_button("Download watchlist (CSV)", show.to_csv(index=False).encode("utf-8"),
                       "fyp_watchlist.csv", "text/csv")

# ------------------------------------------------------------------ What-if (A7)
with tabs[6]:
    st.header("What-if: levers the department can influence")
    base_row = st.session_state.get("last_row") or {**DEF}
    if "last_row" not in st.session_state:
        st.info("Showing a typical group. Run a prediction in the Predict tab to start from your own group.")
    c = st.columns(4)
    m2 = c[0].slider("Supervisor meetings (8 wks)", 0, 30, int(base_row["supervisor_meetings_8wk"]))
    g2 = c[1].slider("GitHub commits (8 wks)", 0, 1000, int(base_row["github_commits_8wk"]))
    a2 = c[2].slider("Supervisor active groups", 1, 15, int(base_row["supervisor_active_groups"]))
    s2 = c[3].selectbox("Industry sponsor", ["No", "Yes"], index=["No", "Yes"].index(base_row["has_industry_sponsor"]), key="wi_sp")
    new = {**base_row, "supervisor_meetings_8wk": m2, "github_commits_8wk": g2,
           "supervisor_active_groups": a2, "has_industry_sponsor": s2}
    e0 = core.explain_one(model, base_row, REF, LOW, HIGH)
    e1 = core.explain_one(model, new, REF, LOW, HIGH)
    b, a = st.columns(2)
    b.markdown(f"**Before:** {100 * e0['p']:.1f}% &nbsp; {badge(e0['tier'])}", unsafe_allow_html=True)
    a.markdown(f"**After:** {100 * e1['p']:.1f}% &nbsp; {badge(e1['tier'])} &nbsp; "
               f"({100 * (e1['p'] - e0['p']):+.1f} pp)", unsafe_allow_html=True)
    st.write("**Recommended action now:** " + e1["action"])
    if st.button("Which lever helps most?"):
        lv = pd.DataFrame(core.best_lever(model, base_row))
        lv["p_before"] = (100 * lv["p_before"]).round(1)
        lv["p_after"] = (100 * lv["p_after"]).round(1)
        lv["drop_pp"] = lv["drop_pp"].round(1)
        st.dataframe(lv.rename(columns={"p_before": "Risk before (%)", "p_after": "Risk after (%)",
                                        "drop_pp": "Drop (pp)"}), hide_index=True, width='stretch')
        top = lv.iloc[0]
        st.success(f"Biggest drop: {top['lever']} ({top['change']}) lowers risk by {top['drop_pp']} pp.")
    st.caption("The model shows association, not a guaranteed effect of the change.")
