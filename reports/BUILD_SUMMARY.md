# FYP-Pulse Build Summary

## What was built
Streamlit early-warning app predicting `poor_outcome` from week-8 information only. Tabs: Data Quality, Dashboard, Models, Predict (with explanation + support step), Fairness, plus bonus Watchlist (A6) and What-if (A7).

**Files** (in `fyp_pulse/`)
- `core.py` all logic (cleaning, features, models, tiers, explainability, fairness, lever analysis); no streamlit import
- `build.py` runs the pipeline once; `app.py` Streamlit UI (auto-builds if artifacts are missing)
- `requirements.txt`, `data/fyp_pulse_train.csv` (original CSV, untouched copy)
- `artifacts/`: `model.joblib`, `results.json`, `fyp_clean.csv`, `test_scored.csv`
- Leftovers from the earlier Phase-1 attempt: `src/cleaning.py`, `data/raw/` (unused by the app)

## A1 data quality
- Initial rows **1840**; cleaned rows **1800**; duplicates removed: **40 exact**, **0** extra by group_id (total 40)
- Dates: 1004 non-ISO converted to YYYY-MM-DD, 0 unparseable (slash dates read day-first; 332 had day > 12 and none had month > 12, so day-first is consistent)
- Rows dropped for missing target: 0
- Rule violations: solo group min != avg: **2** (min set = avg); min > avg: **10** (set NaN, then imputed avg - median gap 0.29); re-check after imputation: 0
- Corrections per column (standardized / out-of-range->NaN / rule-fixed / imputed): proposal_date 1004/0/0/0; campus 259/0/0/0; leader_gender 98/0/0/0; github_commits_8wk 0/5/0/112; supervisor_experience_years 0/4/0/108; supervisor_meetings_8wk 0/4/0/94; mid_eval_score 0/4/0/90; weeks_to_proposal_approval 0/4/0/88; proposal_revisions 0/0/0/80; has_industry_sponsor 0/0/0/73; team_avg_cgpa 0/6/0/65; supervisor_active_groups 0/0/0/69; team_min_cgpa 0/4/12/59; project_type 0/0/0/53; group_size 0/6/0/6
- Total poor outcomes **288**; poor-outcome rate **16.0%**
- All assertions pass (ranges, 8 campuses, 2 genders, unique group_id, min <= avg, min == avg for 78 solo groups, no NaN)
- Note: "imputed" counts include values that were first set to NaN as out-of-range, and the missing-before-imputation table is taken after the range check.

## Features
Used: group_size, team_avg_cgpa, team_min_cgpa, supervisor_experience_years, supervisor_active_groups, proposal_revisions, weeks_to_proposal_approval, supervisor_meetings_8wk, github_commits_8wk, mid_eval_score, program, project_type, has_industry_sponsor + engineered commits_per_member, cgpa_gap, meetings_per_load.

Dropped: final_defense_score and extension_granted (leakage, known after week 8); group_id (identifier); proposal_date (calendar cohort); leader_gender and campus (protected/context, audited in Fairness); poor_outcome (target).

## Models (test set, 360 groups, 58 poor; threshold = max F1 on 5-fold out-of-fold train predictions)
| Model | Thr | Acc | Prec | Recall | F1 | Test AUC | CV AUC |
|---|---|---|---|---|---|---|---|
| **Logistic Regression (chosen)** | 0.687 | 0.825 | 0.465 | 0.569 | 0.512 | 0.802 | 0.827 |
| Random Forest | 0.394 | 0.772 | 0.372 | 0.603 | 0.461 | 0.796 | 0.802 |
| HistGradientBoosting | 0.466 | 0.756 | 0.364 | 0.690 | 0.476 | 0.791 | 0.795 |

Confusion matrices [[TN, FP],[FN, TP]]: LR [[264,38],[25,33]]; RF [[243,59],[23,35]]; HGB [[232,70],[18,40]].
Chosen: Logistic Regression, highest CV ROC-AUC and best test F1/AUC. HGB has higher recall but lower precision/F1. Accuracy is misleading at a 16% base rate (always predicting "pass" gives ~84%).

## Risk tiers (from train out-of-fold probabilities of the final model)
Decision threshold 0.6866. Cut-offs: Low < 0.3158 (OOF median), Medium 0.3158 to < 0.6866, High >= 0.6866 (no adjustment was needed).
Observed test poor rate: **Low 3.3%** (180 groups, 6 poor), **Medium 17.4%** (109, 19), **High 46.5%** (71, 33). Monotonic.

## Fairness (test set)
- Gender: Female n=150 predicted 22.7%, actual 14.0%, recall 61.9%; Male n=210 predicted 17.6%, actual 17.6%, recall 54.1%. Gaps: predicted 5.1 pp, recall 7.8 pp (below the 10 pp flag).
- Campus (all 8 have >= 30 test groups): largest predicted-rate gap **Peshawar 36.7% vs Lahore 14.1% = 22.6 pp**; largest recall gap **Multan 80.0% vs Lahore 40.0% = 40.0 pp**. Both are flagged.
- Takeaway: disparity flagged for campus, but confidence is LOW: smallest group has 30 test groups and campuses have only 5-10 actual positives each, so recall gaps can be chance. Peshawar also has the highest actual rate (26.7%).

## Assumptions and problems
- Windows Application Control initially blocked scikit-learn's compiled files; the user allowed the folder. First build also hit a pandas-3 string-dtype bug that inflated out-of-range counts; fixed and rebuilt (model results unchanged).
- Tiers: Low cut-off = median OOF probability; High = the model's F1 threshold. Median impute for numerics, mode for categoricals; missing dates would use median date (none occurred).
- Explanations: reference-value substitution (training median/mode); the CGPA pair is kept consistent (min <= avg). Contributions are not additive.
- Test-set results come from a single 20% split (58 positives), so metrics carry noticeable uncertainty.
- Tested: build.py run end to end, `streamlit.testing` AppTest of all tabs (load, Predict submit, validation error, What-if lever button) with no exceptions. A real browser was not used.

## Run
```
pip install -r requirements.txt
python build.py        # optional; app.py builds automatically if artifacts/ is missing
streamlit run app.py
```
