# Hospital Deterioration Prediction System — Streamlit App

An interactive dashboard for exploring machine learning approaches to early prediction of patient deterioration using admission-time clinical features.

## What's inside

- **Overview:** cohort, target prevalence, research workflow, and high-level finding.
- **Model Comparison:** held-out patient-level results for Logistic Regression, Random Forest, and XGBoost; ROC-AUC, Average Precision, repeated cross-validation summary, and confusion matrices.
- **Threshold Explorer:** compare default, F1-optimal, and recall-focused thresholds and see how false positives and false negatives change.
- **Patient Predictor:** an optional demonstration form that becomes active only after training on a compatible CSV in the app session.
- **Data & Methodology:** cohort/target checks, selected/excluded variables, descriptive feature comparisons, limitations, and a CSV uploader for training.

## Important note about the input data

The analysis uses a merged hourly panel, selects one `hour_from_admission == 0` record per patient, and applies a stratified 80:20 patient-level split. The current evaluation contains 10,000 patients, 309 positive labels (3.09%), 8,000 training patients, and 2,000 test patients.

## Run locally (Windows)

1. Install Python 3.10 or newer.
2. Open Command Prompt or PowerShell in this folder.
3. Create a virtual environment (recommended):

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

4. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Start the website:

   ```powershell
   streamlit run app.py
   ```

6. Streamlit will print a local URL (usually `http://localhost:8501`). Open it in your browser.

## Enable live model demonstrations

1. Open **Data & Methodology** inside the app.
2. Upload the compatible merged hourly panel CSV. The app checks for the required columns and confirms that the target agrees with `deterioration_event == 1` and `deterioration_hour` between 0 and 12.
3. Click **Train models**.
4. After training completes, open **Patient Predictor** and use hypothetical/de-identified values.

The CSV must contain the following required columns:

- Identifiers/time/label checks: `patient_id`, `hour_from_admission`, `deterioration_event`, `deterioration_hour`, `deterioration_within_12h_from_admission`
- Predictors: `heart_rate`, `respiratory_rate`, `spo2_pct`, `temperature_c`, `systolic_bp`, `diastolic_bp`, `mobility_score`, `wbc_count`, `lactate`, `creatinine`, `crp_level`, `hemoglobin`, `age`, `gender`, `comorbidity_index`, `admission_type`

A header-only schema file can be downloaded in the Data & Methodology page. It is not a dataset and contains no patient records.

## Model training in the app

The app uses this modelling pipeline: median imputation for numeric features, most-frequent imputation and one-hot encoding for categorical features, scaling for Logistic Regression only, balanced classes, and a patient-level 80:20 split with `random_state=2026`. Thresholds are selected from five-fold out-of-fold probabilities on training data. The test set is kept out of threshold selection.

Training from a CSV fits three pipelines and may take noticeable CPU time. Models live in the current Streamlit session; restarting the app clears this session state.

## Safety and research limitations

- This is an academic prototype, **not** a medical device, diagnosis tool, or clinical decision-support system validated for care.
- Do not enter identifiable or actual patient information.
- The available dataset does not contain feature-specific clinical measurement/result timestamps, so availability at the exact decision time needs independent verification.
- The outcome is rare, positive-class estimates are uncertain, and the final study did not establish a definitive best model.
- External or prospective validation has not been performed.

## Project files

- `app.py` — Streamlit app source.
- `requirements.txt` — Python dependencies.
- `.streamlit/config.toml` — app theme and local server defaults.
- `run_windows.bat` — Windows launch helper.
