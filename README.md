# FYP-Pulse: Final Year Project Early Warning System

Streamlit app that predicts which final-year-project groups are likely to end with a poor outcome, using only information known at the end of week 8. It includes data-quality reporting, a dashboard, model comparison, a prediction form with explanations, a fairness audit, a watchlist and a what-if tool. It uses only `data/fyp_pulse_train.csv`: no API keys, no secrets, no network calls.

## Run locally
```
pip install -r requirements.txt && streamlit run app.py
```
Model files in `artifacts/` are committed; if they are missing or incompatible with the installed scikit-learn, the app rebuilds them automatically (about 15 seconds). `python build.py` rebuilds them manually.

## Deploy on Streamlit Community Cloud
1. Open https://share.streamlit.io and sign in with GitHub.
2. Create app -> Deploy a public app from GitHub.
3. Repository `<username>/fyp-pulse`, branch `main`, main file `app.py`.
4. Advanced settings -> Python 3.12.
5. Deploy.
