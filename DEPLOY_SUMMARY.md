# Deploy Summary

## API / secrets audit
Searched app.py, core.py, build.py, requirements.txt (and the former src/cleaning.py) for `requests, urllib, httpx, http://, https://, openai, anthropic, api_key, API_KEY, os.environ, os.getenv, st.secrets, .env`: **0 matches**.
The app needs no API key, no secrets and makes no network calls; it uses only `data/fyp_pulse_train.csv`. (README.md contains the share.streamlit.io URL as documentation text only.)

## requirements.txt (pinned to installed versions)
```
streamlit==1.67.0
scikit-learn==1.9.1
pandas==3.1.1
numpy==2.5.4
joblib==1.6.0
```

## Changes
- Paths were already relative to `core.py` (`Path(__file__).resolve().parent`).
- `app.py`: `load_all()` is `st.cache_resource`; it loads artifacts, smoke-tests the model with a prediction, and on any failure (missing files, corrupt or version-stale model) runs `core.build()` once and reloads.
- Artifacts (`model.joblib`, `results.json`, `fyp_clean.csv`, `test_scored.csv`) are committed.
- Added `.gitignore`, `README.md`.

## Files removed
`src/` (cleaning.py, __init__.py), `data/raw/`, `data/clean/`, all `__pycache__/`.

## Tests
- `build.py` run from a different working directory (`C:\Users\afnan`): OK, 13.9 s.
- `streamlit run app.py --server.headless true --server.port 8599`: `/_stcore/health` returned `ok`; server stopped.
- Streamlit AppTest, no exceptions or errors: initial load, Predict submit, What-if "Which lever helps most?", Watchlist at K=50 (24 of 58 poor groups captured, precision 48%, 2.98x lift vs random).
- Fallback: with `model.joblib` overwritten by garbage, a fresh app run rebuilt it automatically with no errors.
- Not tested: a real Linux / Streamlit Cloud run.

## Git / GitHub
- Local git repo on branch `main` with one commit.
- **Not pushed**: the GitHub CLI (`gh`) is not installed here, so I could not create the repo. Run these yourself (create an empty repo named `fyp-pulse` on github.com first, without a README):
```
git remote add origin https://github.com/<my-username>/fyp-pulse.git
git push -u origin main
```
- Note: a public repo makes `data/fyp_pulse_train.csv` public. Make the repo private if the data should not be shared (Streamlit Cloud can deploy private repos too).

## Streamlit Community Cloud steps
1. Open https://share.streamlit.io and sign in with GitHub.
2. Click "Create app" -> "Deploy a public app from GitHub".
3. Repository: `<username>/fyp-pulse`, Branch: `main`, Main file path: `app.py`.
4. Advanced settings -> Python version 3.12 (3.11+ required for pandas 3).
5. Click Deploy; wait for the build; the URL will look like https://<name>.streamlit.app.

## Still manual
Create the GitHub repo and push (commands above), then do the Streamlit Cloud steps. If the cloud build cannot install a pinned version, the first failing package in the build log tells you which pin to relax.
