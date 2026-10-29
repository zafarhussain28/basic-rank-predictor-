# JEE Rank & Percentile Predictor

A Streamlit application that uses the repository's trained scikit-learn models to estimate JEE percentile and rank from raw marks. Predictions are estimates, not official examination results.

## Model inputs and artifacts

Both models expect one numeric feature, `raw_marks`, in a one-column input frame. The project dataset contains integer marks from 0 through 300 and has the targets `approx_percentile` and `approx_rank`. The training notebook fits separate Random Forest regressors for those targets; the saved artifacts were serialized with scikit-learn 1.6.1.

- `percentile_model (1).pkl` predicts `approx_percentile`.
- `rank_model (1).pkl` predicts `approx_rank`.

The app loads these repository-relative files and caches them for the Streamlit session. It does not retrain the models. Only load trusted model artifacts: joblib/pickle files can execute code when deserialized.

## Run locally or in Codespaces

Python 3.12 is selected in `.python-version`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`. In Codespaces, forward port 8501 when prompted.

Run the model and input checks with:

```bash
python -m unittest -v
```

## Deploy on Streamlit Community Cloud

1. Push the repository to GitHub using your normal workflow.
2. In Streamlit Community Cloud, create an app from the repository and select `app.py` as the entry point.
3. Select Python 3.12, matching `.python-version`, and deploy. No secrets or external services are required.

The two model artifacts are approximately 6.4 MB each and are already tracked in this repository. The dataset remains in the repository for reproducibility but is not needed at app runtime. No deployment or repository settings are changed by this project setup.

## Limitations

The app uses the supplied 301-row dataset and approximate labels. Its outputs do not represent official results and should not be used as a guarantee for rank, counselling, admission, or eligibility. Predictions outside the training marks range are rejected.
