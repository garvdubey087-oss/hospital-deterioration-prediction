from __future__ import annotations

"""Hospital Deterioration Prediction System."""

from io import BytesIO, StringIO
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


# -----------------------------------------------------------------------------
# Dataset and evaluation settings
# -----------------------------------------------------------------------------
APP_TITLE = "Hospital Deterioration Prediction System"
TARGET = "deterioration_within_12h_from_admission"
FEATURES = [
    "heart_rate",
    "respiratory_rate",
    "spo2_pct",
    "temperature_c",
    "systolic_bp",
    "diastolic_bp",
    "mobility_score",
    "wbc_count",
    "lactate",
    "creatinine",
    "crp_level",
    "hemoglobin",
    "age",
    "gender",
    "comorbidity_index",
    "admission_type",
]
NUMERIC_FEATURES = [feature for feature in FEATURES if feature not in {"gender", "admission_type"}]
CATEGORICAL_FEATURES = ["gender", "admission_type"]
REQUIRED_COLUMNS = FEATURES + [TARGET, "patient_id", "hour_from_admission", "deterioration_event", "deterioration_hour"]
MODEL_NAMES = ["Logistic Regression", "Random Forest", "XGBoost"]

# Test set contains 2,000 previously unseen patients.
STATIC_SUMMARY = {
    "hourly_rows": 417_866,
    "hourly_columns": 27,
    "patients": 10_000,
    "positive_patients": 309,
    "negative_patients": 9_691,
    "positive_pct": 3.09,
    "train_patients": 8_000,
    "test_patients": 2_000,
    "test_positive": 62,
    "test_negative": 1_938,
}

STATIC_F1_RESULTS = pd.DataFrame(
    [
        {
            "Model": "Logistic Regression",
            "Threshold": 0.7990,
            "Accuracy": 0.9260,
            "Balanced Accuracy": 0.5949,
            "Precision": 0.1293,
            "Recall": 0.2419,
            "F1-score": 0.1685,
            "ROC-AUC": 0.7696,
            "Average Precision": 0.1303,
            "True negatives": 1837,
            "False positives": 101,
            "False negatives": 47,
            "True positives": 15,
        },
        {
            "Model": "Random Forest",
            "Threshold": 0.1229,
            "Accuracy": 0.9170,
            "Balanced Accuracy": 0.5825,
            "Precision": 0.1061,
            "Recall": 0.2258,
            "F1-score": 0.1443,
            "ROC-AUC": 0.7447,
            "Average Precision": 0.1205,
            "True negatives": 1820,
            "False positives": 118,
            "False negatives": 48,
            "True positives": 14,
        },
        {
            "Model": "XGBoost",
            "Threshold": 0.6760,
            "Accuracy": 0.8925,
            "Balanced Accuracy": 0.5776,
            "Precision": 0.0820,
            "Recall": 0.2419,
            "F1-score": 0.1224,
            "ROC-AUC": 0.7413,
            "Average Precision": 0.0958,
            "True negatives": 1770,
            "False positives": 168,
            "False negatives": 47,
            "True positives": 15,
        },
    ]
)

STATIC_THRESHOLDS = {
    "Logistic Regression": {"Default 0.50": 0.50, "F1-optimal": 0.7990, "Recall-focused (80%)": 0.3961},
    "Random Forest": {"Default 0.50": 0.50, "F1-optimal": 0.1229, "Recall-focused (80%)": 0.0330},
    "XGBoost": {"Default 0.50": 0.50, "F1-optimal": 0.6760, "Recall-focused (80%)": 0.2355},
}

STATIC_STRATEGY_METRICS = {
    "Logistic Regression": {
        "Default 0.50": {"Accuracy": 0.6925, "Balanced accuracy": 0.7008, "Precision": 0.0686, "Recall": 0.7097, "F1-score": 0.1252, "TN": 1341, "FP": 597, "FN": 18, "TP": 44},
        "F1-optimal": {"Accuracy": 0.9260, "Balanced accuracy": 0.5949, "Precision": 0.1293, "Recall": 0.2419, "F1-score": 0.1685, "TN": 1837, "FP": 101, "FN": 47, "TP": 15},
        "Recall-focused (80%)": {"Accuracy": 0.5775, "Balanced accuracy": 0.6961, "Precision": 0.0576, "Recall": 0.8226, "F1-score": 0.1077, "TN": 1104, "FP": 834, "FN": 11, "TP": 51},
    },
    "Random Forest": {
        "Default 0.50": {"Accuracy": 0.9690, "Balanced accuracy": 0.5000, "Precision": 0.0000, "Recall": 0.0000, "F1-score": 0.0000, "TN": 1938, "FP": 0, "FN": 62, "TP": 0},
        "F1-optimal": {"Accuracy": 0.9170, "Balanced accuracy": 0.5825, "Precision": 0.1061, "Recall": 0.2258, "F1-score": 0.1443, "TN": 1820, "FP": 118, "FN": 48, "TP": 14},
        "Recall-focused (80%)": {"Accuracy": 0.5650, "Balanced accuracy": 0.6741, "Precision": 0.0541, "Recall": 0.7903, "F1-score": 0.1012, "TN": 1081, "FP": 857, "FN": 13, "TP": 49},
    },
    "XGBoost": {
        "Default 0.50": {"Accuracy": 0.7560, "Balanced accuracy": 0.6789, "Precision": 0.0740, "Recall": 0.5968, "F1-score": 0.1317, "TN": 1475, "FP": 463, "FN": 25, "TP": 37},
        "F1-optimal": {"Accuracy": 0.8925, "Balanced accuracy": 0.5776, "Precision": 0.0820, "Recall": 0.2419, "F1-score": 0.1224, "TN": 1770, "FP": 168, "FN": 47, "TP": 15},
        "Recall-focused (80%)": {"Accuracy": 0.5065, "Balanced accuracy": 0.6751, "Precision": 0.0514, "Recall": 0.8548, "F1-score": 0.0970, "TN": 960, "FP": 978, "FN": 9, "TP": 53},
    },
}

REPEATED_CV = pd.DataFrame(
    [
        {"Model": "Logistic Regression", "ROC-AUC mean": 0.7644, "ROC-AUC SD": 0.0370, "Average Precision mean": 0.1050, "Average Precision SD": 0.0306},
        {"Model": "Random Forest", "ROC-AUC mean": 0.7188, "ROC-AUC SD": 0.0277, "Average Precision mean": 0.0807, "Average Precision SD": 0.0147},
        {"Model": "XGBoost", "ROC-AUC mean": 0.7221, "ROC-AUC SD": 0.0366, "Average Precision mean": 0.0812, "Average Precision SD": 0.0201},
    ]
)

BOOTSTRAP_DIFFS = pd.DataFrame(
    [
        {"Metric": "ROC-AUC", "Model A": "Logistic Regression", "Model B": "Random Forest", "Difference A − B": 0.0250, "95% CI lower": -0.0159, "95% CI upper": 0.0679, "Includes zero": True},
        {"Metric": "Average Precision", "Model A": "Logistic Regression", "Model B": "Random Forest", "Difference A − B": 0.0098, "95% CI lower": -0.0811, "95% CI upper": 0.0906, "Includes zero": True},
        {"Metric": "ROC-AUC", "Model A": "Logistic Regression", "Model B": "XGBoost", "Difference A − B": 0.0283, "95% CI lower": -0.0055, "95% CI upper": 0.0660, "Includes zero": True},
        {"Metric": "Average Precision", "Model A": "Logistic Regression", "Model B": "XGBoost", "Difference A − B": 0.0345, "95% CI lower": -0.0388, "95% CI upper": 0.1004, "Includes zero": True},
        {"Metric": "ROC-AUC", "Model A": "Random Forest", "Model B": "XGBoost", "Difference A − B": 0.0034, "95% CI lower": -0.0327, "95% CI upper": 0.0401, "Includes zero": True},
        {"Metric": "Average Precision", "Model A": "Random Forest", "Model B": "XGBoost", "Difference A − B": 0.0247, "95% CI lower": -0.0129, "95% CI upper": 0.0719, "Includes zero": True},
    ]
)

MEAN_FEATURES = pd.DataFrame(
    [
        {"Feature": "Heart rate", "No deterioration": 84.458, "Deterioration within 12h": 89.860, "Unit": "beats/min"},
        {"Feature": "Respiratory rate", "No deterioration": 18.536, "Deterioration within 12h": 20.706, "Unit": "breaths/min"},
        {"Feature": "SpO₂", "No deterioration": 95.319, "Deterioration within 12h": 94.217, "Unit": "%"},
        {"Feature": "Systolic BP", "No deterioration": 116.654, "Deterioration within 12h": 110.689, "Unit": "mmHg"},
        {"Feature": "Lactate", "No deterioration": 1.516, "Deterioration within 12h": 1.705, "Unit": "mmol/L (as labelled in source data)"},
        {"Feature": "CRP", "No deterioration": 24.075, "Deterioration within 12h": 28.881, "Unit": "dataset units"},
    ]
)


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
      :root {
        --ink:#e7eef8; --muted:#a2b1c5; --teal:#4fd1c5; --teal-soft:#143a3b;
        --line:#29384d; --paper:#0a0f1b; --panel:#111a2b; --panel-2:#172338; --navy:#07101d;
      }
      html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color:var(--ink); }
      .stApp { background:radial-gradient(circle at 80% 0%,#14243a 0%,#0a0f1b 48%); color:var(--ink); }
      [data-testid="stMain"] { background:transparent; }
      [data-testid="stHeader"] { background:rgba(10,15,27,.92); }
      [data-testid="stToolbar"] { right:1rem; }
      .block-container { padding-top:1.45rem; padding-bottom:3rem; max-width:1440px; }
      [data-testid="stSidebar"] { background:linear-gradient(180deg,#070d18 0%,#101c2e 100%); border-right:1px solid #25354a; }
      [data-testid="stSidebar"] * { color:#e7eef8; }
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#aebed1; }
      [data-testid="stSidebar"] [data-testid="stRadio"] label { color:#e7eef8 !important; }
      h1,h2,h3,h4,h5,h6 { color:var(--ink); }
      p, li, label, .stMarkdown, [data-testid="stCaptionContainer"] { color:var(--ink); }
      .hero { background:linear-gradient(118deg,#101b2d 0%,#12344b 60%,#075e5b 100%); padding:2rem 2.1rem; border:1px solid rgba(103,232,220,.18); border-radius:22px; color:#fff; margin-bottom:1.2rem; box-shadow:0 16px 42px rgba(0,0,0,.24); }
      .hero .eyebrow { font-size:.74rem; letter-spacing:.15em; text-transform:uppercase; font-weight:700; color:#78e2d5; margin-bottom:.65rem; }
      .hero h1 { font-family:'Manrope',sans-serif; font-size:clamp(1.8rem,3.2vw,2.75rem); font-weight:800; line-height:1.12; color:#fff; margin:0 0 .8rem 0; }
      .hero p { max-width:850px; font-size:1rem; line-height:1.65; color:#d4e2f0; margin:0; }
      .pill-row { display:flex; flex-wrap:wrap; gap:.45rem; margin-top:1.2rem; }
      .pill { display:inline-block; border:1px solid rgba(145,240,226,.26); background:rgba(255,255,255,.055); color:#d9fffa; border-radius:999px; padding:.35rem .7rem; font-size:.76rem; font-weight:600; }
      .section-label { color:var(--teal); font-size:.74rem; text-transform:uppercase; font-weight:800; letter-spacing:.12em; margin:.3rem 0 .4rem 0; }
      .section-title { font-family:'Manrope',sans-serif; font-weight:800; font-size:1.45rem; color:var(--ink); margin:0 0 .7rem 0; }
      .muted { color:var(--muted); line-height:1.65; }
      .info-card { background:linear-gradient(145deg,#121e31,#101827); border:1px solid var(--line); border-radius:16px; padding:1.15rem 1.2rem; height:100%; box-shadow:0 8px 22px rgba(0,0,0,.12); }
      .info-card h3 { font-family:'Manrope',sans-serif; font-size:1.02rem; font-weight:800; margin:.15rem 0 .45rem 0; color:#eef4fb; }
      .info-card p { color:#aab8ca; font-size:.91rem; line-height:1.58; margin:0; }
      .small-kicker { color:var(--teal); font-size:.72rem; text-transform:uppercase; letter-spacing:.1em; font-weight:800; }
      .callout { border-radius:14px; padding:1rem 1.1rem; background:#102f31; border:1px solid #205a59; color:#c9f5ec; line-height:1.6; }
      .warnbox { border-radius:14px; padding:1rem 1.1rem; background:#302719; border:1px solid #69512a; color:#f6dba2; line-height:1.6; }
      .dangerbox { border-radius:14px; padding:1rem 1.1rem; background:#351d28; border:1px solid #733343; color:#ffcbd2; line-height:1.6; }
      .workflow-number { width:30px; height:30px; display:inline-flex; justify-content:center; align-items:center; border-radius:50%; background:#123b3b; color:#74e3d6; font-weight:800; margin-bottom:.6rem; border:1px solid #26635f; }
      div[data-testid="stMetric"] { background:linear-gradient(145deg,#121e31,#0f1726); padding:1rem 1.05rem; border:1px solid var(--line); border-radius:16px; box-shadow:0 8px 22px rgba(0,0,0,.12); }
      div[data-testid="stMetricLabel"] { color:#a7b6c9; font-weight:600; }
      div[data-testid="stMetricValue"] { color:#f0f6fc; font-family:'Manrope',sans-serif; font-weight:800; }
      div[data-testid="stMetricDelta"] { color:#72dfc9; }
      .stButton > button, .stDownloadButton > button, [data-testid="stLinkButton"] a { border-radius:10px; font-weight:700; border:1px solid #277e79; background:#122737; color:#dffdfa; }
      .stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stLinkButton"] a:hover { border-color:#53d5c9; color:#fff; background:#173844; }
      .stButton > button[kind="primary"] { background:#0e8f86; color:white; border-color:#36bfb2; }
      .stButton > button[kind="primary"]:hover { background:#0ba79a; color:white; }
      input, textarea { color:#e7eef8 !important; background:#111b2c !important; border-color:#35445b !important; }
      [data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-baseweb="textarea"] { background:#111b2c !important; border-color:#35445b !important; color:#e7eef8 !important; }
      [data-baseweb="select"] span, [data-baseweb="select"] input, [data-baseweb="input"] input { color:#e7eef8 !important; }
      [data-baseweb="popover"] [role="listbox"], [data-baseweb="menu"] { background:#172338 !important; border:1px solid #34445b !important; }
      [role="option"] { background:#172338 !important; color:#e7eef8 !important; }
      [role="option"]:hover { background:#234052 !important; }
      [data-testid="stFileUploader"] section { background:#111b2c; border:1px dashed #3b536b; border-radius:12px; }
      [data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span { color:#aebed1; }
      [data-testid="stExpander"] { border:1px solid var(--line); border-radius:12px; background:#101a2a; }
      [data-testid="stExpander"] summary { color:#e7eef8; }
      [data-testid="stDataFrame"], [data-testid="stTable"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; background:#101a2a; }
      [data-testid="stDataFrame"] * { color:#e7eef8; }
      [data-testid="stTabs"] button { color:#aebed1; }
      [data-testid="stTabs"] button[aria-selected="true"] { color:#62dbcf; }
      code { color:#a7f3e9; background:#152438; border-radius:5px; }
      pre { background:#0d1625 !important; border:1px solid #29384d !important; }
      hr { border-color:var(--line); }
      a { color:#70dcd0; }
      footer { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


def hero(title: str, subtitle: str, eyebrow: str = "CLINICAL DATA SCIENCE · RESEARCH PROTOTYPE") -> None:
    st.markdown(
        f"""
        <div class="hero">
          <div class="eyebrow">{eyebrow}</div>
          <h1>{title}</h1>
          <p>{subtitle}</p>
          <div class="pill-row">
            <span class="pill">12-hour prediction window</span>
            <span class="pill">Patient-level evaluation</span>
            <span class="pill">3 classifiers compared</span>
            <span class="pill">Not for clinical use</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_heading(label: str, title: str, description: str | None = None) -> None:
    st.markdown(f'<div class="section-label">{label}</div><div class="section-title">{title}</div>', unsafe_allow_html=True)
    if description:
        st.markdown(f'<div class="muted">{description}</div>', unsafe_allow_html=True)


def card(kicker: str, title: str, text: str) -> None:
    st.markdown(
        f'<div class="info-card"><div class="small-kicker">{kicker}</div><h3>{title}</h3><p>{text}</p></div>',
        unsafe_allow_html=True,
    )


def fmt_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def static_confusion_matrix(model_name: str, strategy: str) -> np.ndarray:
    metric = STATIC_STRATEGY_METRICS[model_name][strategy]
    return np.array([[metric["TN"], metric["FP"]], [metric["FN"], metric["TP"]]])


def draw_confusion_matrix(cm: np.ndarray, model_name: str, strategy: str) -> go.Figure:
    labels = ["No deterioration", "Deterioration"]
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Predicted: no", "Predicted: yes"],
            y=["Actual: no", "Actual: yes"],
            colorscale=[[0, "#172638"], [0.5, "#176f68"], [1, "#10b981"]],
            showscale=False,
            text=cm,
            texttemplate="%{text:,}",
            textfont={"size": 18, "color": "#f1f7fb"},
            hovertemplate="%{y}<br>%{x}<br>Count: %{z:,}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{model_name} · {strategy}",
        margin=dict(l=10, r=10, t=45, b=10),
        height=300,
        paper_bgcolor="#0a0f1b",
        plot_bgcolor="#111a2b",
        font=dict(color="#e7eef8", family="DM Sans"),
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed"),
    )
    return fig


# -----------------------------------------------------------------------------
# Model-building helpers used only if a compatible CSV is uploaded in the app
# -----------------------------------------------------------------------------
def validate_panel(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError("The CSV is missing required columns: " + ", ".join(missing))

    clean = df.copy()
    expected_target = (
        clean["deterioration_event"].eq(1)
        & pd.to_numeric(clean["deterioration_hour"], errors="coerce").between(0, 12)
    ).astype(int)
    actual_target = pd.to_numeric(clean[TARGET], errors="coerce")
    mismatches = int((expected_target != actual_target).sum())
    if mismatches:
        raise ValueError(
            f"Target validation failed: {mismatches:,} rows disagree with the rule "
            "deterioration_event = 1 and deterioration_hour in [0, 12]. Check your target definition."
        )

    admission = clean.loc[pd.to_numeric(clean["hour_from_admission"], errors="coerce").eq(0)].copy()
    if admission.empty:
        raise ValueError("No hour_from_admission == 0 records were found, so an admission-time model cannot be built.")
    if admission["patient_id"].duplicated().any():
        duplicate_count = int(admission["patient_id"].duplicated().sum())
        raise ValueError(f"Found {duplicate_count:,} duplicate patient identifiers among hour-0 rows.")
    if admission[TARGET].isna().any():
        raise ValueError("The target contains missing values in the hour-0 cohort.")
    admission[TARGET] = admission[TARGET].astype(int)
    if admission[TARGET].nunique() != 2:
        raise ValueError("The hour-0 cohort must contain both target classes.")
    counts = admission[TARGET].value_counts()
    if int(counts.min()) < 5:
        raise ValueError("At least five patients in each target class are needed for 5-fold threshold selection.")
    return admission, missing


def make_preprocessor(numeric_features: list[str], categorical_features: list[str], scale_numeric: bool = False) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def make_model(
    model_name: str,
    numeric_features: list[str],
    categorical_features: list[str],
    scale_pos_weight: float,
) -> Pipeline:
    if model_name == "Logistic Regression":
        classifier = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
        scale_numeric = True
    elif model_name == "Random Forest":
        classifier = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        scale_numeric = False
    elif model_name == "XGBoost":
        classifier = XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        scale_numeric = False
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(numeric_features, categorical_features, scale_numeric)),
            ("classifier", classifier),
        ]
    )


def choose_thresholds(y_train: pd.Series, oof_probabilities: np.ndarray) -> dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y_train, oof_probabilities)
    if len(thresholds) == 0:
        return {"F1-optimal": 0.5, "Recall-focused (80%)": float("nan")}
    f1_values = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    best_f1_idx = int(np.argmax(f1_values))
    f1_threshold = float(thresholds[best_f1_idx])
    eligible = np.flatnonzero(recall[:-1] >= 0.80)
    if len(eligible):
        best_recall_idx = int(eligible[np.argmax(precision[eligible])])
        recall_threshold = float(thresholds[best_recall_idx])
    else:
        recall_threshold = float("nan")
    return {"F1-optimal": f1_threshold, "Recall-focused (80%)": recall_threshold}


def train_uploaded_dataset(df: pd.DataFrame) -> dict[str, Any]:
    admission_df, _ = validate_panel(df)
    X = admission_df[FEATURES].copy()
    y = admission_df[TARGET].astype(int).copy()
    patient_ids = admission_df["patient_id"].copy()

    X_train, X_test, y_train, y_test, pid_train, pid_test = train_test_split(
        X, y, patient_ids, test_size=0.20, random_state=2026, stratify=y
    )
    if set(pid_train.tolist()).intersection(set(pid_test.tolist())):
        raise ValueError("Patient overlap was detected after splitting; modelling stopped.")

    numeric_features = X_train.select_dtypes(include="number").columns.tolist()
    categorical_features = X_train.select_dtypes(exclude="number").columns.tolist()
    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    models = {
        name: make_model(name, numeric_features, categorical_features, scale_pos_weight)
        for name in MODEL_NAMES
    }

    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    thresholds_by_model: dict[str, dict[str, float]] = {}
    oof_by_model: dict[str, np.ndarray] = {}
    for name, model in models.items():
        oof = cross_val_predict(
            clone(model), X_train, y_train, cv=inner_cv, method="predict_proba", n_jobs=1
        )[:, 1]
        oof_by_model[name] = oof
        thresholds_by_model[name] = choose_thresholds(y_train, oof)

    trained_models: dict[str, Pipeline] = {}
    test_probabilities: dict[str, np.ndarray] = {}
    result_rows: list[dict[str, Any]] = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        probabilities = model.predict_proba(X_test)[:, 1]
        test_probabilities[name] = probabilities
        strategy_thresholds = {"Default 0.50": 0.50, **thresholds_by_model[name]}
        roc_auc = roc_auc_score(y_test, probabilities)
        avg_precision = average_precision_score(y_test, probabilities)
        for strategy, threshold in strategy_thresholds.items():
            if pd.isna(threshold):
                continue
            predictions = (probabilities >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
            result_rows.append(
                {
                    "Model": name,
                    "Threshold Strategy": strategy,
                    "Threshold": float(threshold),
                    "Accuracy": accuracy_score(y_test, predictions),
                    "Balanced accuracy": balanced_accuracy_score(y_test, predictions),
                    "Precision": precision_score(y_test, predictions, zero_division=0),
                    "Recall": recall_score(y_test, predictions, zero_division=0),
                    "F1-score": f1_score(y_test, predictions, zero_division=0),
                    "ROC-AUC": roc_auc,
                    "Average Precision": avg_precision,
                    "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
                }
            )

    eval_results = pd.DataFrame(result_rows)
    return {
        "models": trained_models,
        "thresholds": thresholds_by_model,
        "results": eval_results,
        "test_probabilities": test_probabilities,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "features": FEATURES,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "patients": len(admission_df),
        "positives": int(y.sum()),
        "negatives": int((y == 0).sum()),
        "train_patients": len(X_train),
        "test_patients": len(X_test),
        "test_positives": int(y_test.sum()),
        "test_negatives": int((y_test == 0).sum()),
        "prevalence": float(y.mean()),
        "gender_options": sorted(X["gender"].dropna().astype(str).unique().tolist()),
        "admission_type_options": sorted(X["admission_type"].dropna().astype(str).unique().tolist()),
        "numeric_defaults": X.median(numeric_only=True).to_dict(),
        "training_scale_pos_weight": scale_pos_weight,
    }


def model_lab_metrics(bundle: dict[str, Any], strategy: str = "F1-optimal") -> pd.DataFrame:
    chosen = bundle["results"].loc[bundle["results"]["Threshold Strategy"] == strategy].copy()
    return chosen.sort_values("ROC-AUC", ascending=False).reset_index(drop=True)


def make_schema_csv() -> bytes:
    return pd.DataFrame(columns=REQUIRED_COLUMNS).to_csv(index=False).encode("utf-8")


# -----------------------------------------------------------------------------
# Sidebar navigation
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("# 🩺 Deterioration Lab")
    st.markdown("A research dashboard for 12-hour clinical deterioration prediction.")
    st.divider()
    page = st.radio(
        "NAVIGATION",
        ["Overview", "Model Comparison", "Threshold Explorer", "Patient Predictor", "Data & Methodology"],
        label_visibility="visible",
    )
    st.divider()
    st.markdown("**Project**")
    st.caption("Project by Garv Dubey")
    st.caption("Clinical deterioration research")
    st.markdown(
        "<div style='margin-top:1rem;padding:.75rem;border:1px solid rgba(208,235,239,.18);border-radius:12px;background:rgba(255,255,255,.045);font-size:.78rem;line-height:1.5;color:#d7e6ec;'>Not a diagnosis. No clinical decisions should be made from this app.</div>",
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Pages
# -----------------------------------------------------------------------------
if page == "Overview":
    hero(
        "Can admission-time data help flag deterioration?",
        "An interactive view of a research study predicting whether a patient experiences clinical deterioration within 12 hours of admission. The final analysis compares Logistic Regression, Random Forest and XGBoost on previously unseen patients.",
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Patients analysed", "10,000", "one hour-0 record per patient")
    k2.metric("Positive target", "309", "3.09% of patients")
    k3.metric("Candidate features", "16", "vitals, labs and demographics")
    k4.metric("Held-out test set", "2,000", "62 positive patients")

    st.write("")
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        section_heading("THE MAIN FINDING", "No clear model winner yet", "The models show modest positive-class detection under an admission-time, patient-level design. At the F1-optimal operating points, Logistic Regression has the strongest point estimates, but uncertainty intervals overlap.")
        st.markdown(
            "<div class='callout'><b>What the results really say:</b> the data contain some ranking signal, but precision is low because deterioration is uncommon. A threshold that detects more positive patients also produces many false alerts. The study does not establish clinical readiness or a statistically conclusive best classifier.</div>",
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown("**Current point estimates at F1-optimal thresholds**")
        chart_df = STATIC_F1_RESULTS.melt(
            id_vars="Model", value_vars=["ROC-AUC", "Average Precision"], var_name="Metric", value_name="Score"
        )
        fig = px.bar(
            chart_df,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            color_discrete_sequence=["#0d9488", "#7b8fa8"],
            text=chart_df["Score"].map(lambda x: f"{x:.3f}"),
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(height=360, margin=dict(l=5, r=5, t=20, b=5), paper_bgcolor="#0a0f1b", plot_bgcolor="#111a2b", legend_title_text="", yaxis_title="Score", xaxis_title="", font=dict(family="DM Sans", color="#e7eef8"))
        st.plotly_chart(fig, width="stretch")

    with right:
        section_heading("THE COHORT", "Severe class imbalance", "Only 309 of 10,000 patients have the positive target in the final admission-time cohort.")
        donut = go.Figure(
            data=[go.Pie(
                labels=["No deterioration within 12h", "Deterioration within 12h"],
                values=[9691, 309],
                hole=.72,
                marker=dict(colors=["#dce6ee", "#0d9488"], line=dict(color="#0a0f1b", width=3)),
                textinfo="none",
                hovertemplate="%{label}<br>%{value:,} patients (%{percent})<extra></extra>",
            )]
        )
        donut.update_layout(
            height=275,
            showlegend=True,
            legend=dict(orientation="h", y=-.06, x=0, font=dict(size=11)),
            margin=dict(l=5, r=5, t=5, b=35),
            paper_bgcolor="#0a0f1b",
            annotations=[dict(text="3.09%<br><span style='font-size:11px'>positive</span>", x=.5, y=.5, font=dict(size=22, color="#e7eef8", family="Manrope"), showarrow=False)],
        )
        st.plotly_chart(donut, width="stretch")
        st.markdown("<div class='warnbox'><b>Why accuracy can mislead:</b> a model can look accurate by mostly predicting the majority class. Recall, precision, F1-score and Average Precision provide a fuller picture here.</div>", unsafe_allow_html=True)

    st.write("")
    section_heading("THE WORKFLOW", "From raw hourly panel to patient-level evaluation")
    steps = st.columns(5, gap="small")
    step_items = [
        ("01", "Validate", "Check target consistency and inspect rows, columns, duplicates and missing values."),
        ("02", "Isolate hour 0", "Retain one admission-time record per patient, not hundreds of repeated hourly rows."),
        ("03", "Build pipelines", "Impute, encode categories and scale numeric fields only when appropriate."),
        ("04", "Tune thresholds", "Use training out-of-fold probabilities for F1-oriented and recall-oriented cutoffs."),
        ("05", "Test & quantify", "Evaluate on held-out patients; repeated CV and paired bootstrap assess uncertainty."),
    ]
    for col, (number, title, text) in zip(steps, step_items):
        with col:
            st.markdown(f"<div class='info-card'><div class='workflow-number'>{number}</div><h3>{title}</h3><p>{text}</p></div>", unsafe_allow_html=True)

    st.write("")
    st.markdown(
        "<div class='callout'><b>Evaluation design:</b> the primary analysis uses one hour-0 record per patient, with a patient-level train-test split. This keeps each patient's records within a single partition.</div>",
        unsafe_allow_html=True,
    )

elif page == "Model Comparison":
    hero(
        "Three models. One held-out patient set.",
        "Explore held-out test results, confusion counts, threshold-independent metrics, and cross-validation variability.",
        eyebrow="MODEL EVALUATION",
    )
    section_heading("HELD-OUT TEST", "Comparison at F1-optimal thresholds", "Thresholds were derived from five-fold out-of-fold predictions within the training cohort. The test set contains 2,000 patients, including 62 positive cases.")
    display_df = STATIC_F1_RESULTS[["Model", "Threshold", "Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "Average Precision"]].copy()
    for col in ["Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "Average Precision"]:
        display_df[col] = display_df[col].map(lambda value: f"{value:.4f}")
    display_df["Threshold"] = display_df["Threshold"].map(lambda value: f"{value:.4f}")
    st.dataframe(display_df, width="stretch", hide_index=True)

    st.write("")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        section_heading("RANKING QUALITY", "ROC-AUC and Average Precision")
        metric_choice = st.selectbox("Metric", ["ROC-AUC", "Average Precision"], key="comparison_metric")
        metric_plot = STATIC_F1_RESULTS[["Model", metric_choice]].sort_values(metric_choice, ascending=True)
        fig = px.bar(metric_plot, x=metric_choice, y="Model", orientation="h", text=metric_plot[metric_choice].map(lambda v: f"{v:.4f}"), color=metric_choice, color_continuous_scale=["#d5eee9", "#0d9488"])
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(height=320, margin=dict(l=5, r=45, t=15, b=5), coloraxis_showscale=False, paper_bgcolor="#0a0f1b", plot_bgcolor="#111a2b", xaxis_title=metric_choice, yaxis_title="", font=dict(family="DM Sans", color="#e7eef8"))
        st.plotly_chart(fig, width="stretch")
    with c2:
        section_heading("OPERATING POINT", "Confusion matrix")
        cm_model = st.selectbox("Model for confusion matrix", MODEL_NAMES, key="cm_model")
        st.plotly_chart(draw_confusion_matrix(static_confusion_matrix(cm_model, "F1-optimal"), cm_model, "F1-optimal"), width="stretch")

    section_heading("ROBUSTNESS", "Repeated stratified cross-validation")
    st.markdown("The notebook repeats five-fold stratified validation twice on training patients, yielding 10 validation folds per model. These results assess variability; they do not replace the held-out test result.")
    cv_long = REPEATED_CV.melt(
        id_vars="Model",
        value_vars=["ROC-AUC mean", "Average Precision mean"],
        var_name="Metric summary",
        value_name="Mean",
    )
    fig = px.bar(cv_long, x="Model", y="Mean", color="Metric summary", barmode="group", color_discrete_sequence=["#0d9488", "#8a9caf"], text=cv_long["Mean"].map(lambda v: f"{v:.3f}"))
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(height=320, margin=dict(l=5, r=5, t=15, b=5), paper_bgcolor="#0a0f1b", plot_bgcolor="#111a2b", legend_title_text="", yaxis_title="Fold-average score", xaxis_title="", font=dict(family="DM Sans", color="#e7eef8"))
    st.plotly_chart(fig, width="stretch")
    cv_show = REPEATED_CV.copy()
    for col in cv_show.columns[1:]:
        cv_show[col] = cv_show[col].map(lambda x: f"{x:.4f}")
    st.dataframe(cv_show, width="stretch", hide_index=True)

    st.markdown(
        "<div class='callout'><b>Interpretation:</b> Logistic Regression has the highest point estimates for ROC-AUC (0.7696), Average Precision (0.1303) and F1-score (0.1685) at its selected F1 threshold. This is not proof that it is truly superior: the paired bootstrap confidence intervals for differences between models include zero.</div>",
        unsafe_allow_html=True,
    )

elif page == "Threshold Explorer":
    hero(
        "A cutoff changes what the model flags.",
        "Explore three decision strategies. The same underlying scores can create different numbers of missed cases and false alerts depending on the chosen threshold.",
        eyebrow="PRECISION · RECALL · THRESHOLDS",
    )
    model_choice = st.selectbox("Choose model", MODEL_NAMES, key="threshold_model")
    strategy_choice = st.radio("Decision strategy", ["Default 0.50", "F1-optimal", "Recall-focused (80%)"], horizontal=True)
    selected = STATIC_STRATEGY_METRICS[model_choice][strategy_choice]
    threshold = STATIC_THRESHOLDS[model_choice][strategy_choice]
    if np.isnan(threshold):
        st.warning("The recall-focused threshold was not available for this model in the current saved experiment.")
    else:
        st.markdown(
            f"<div class='callout'><b>{model_choice} · {strategy_choice}</b><br>Threshold = <b>{threshold:.4f}</b>. A record is flagged by this prototype when its model score is at or above the selected cutoff. This is a research decision rule, not a clinical action threshold.</div>",
            unsafe_allow_html=True,
        )
    st.write("")
    a, b, c, d = st.columns(4)
    a.metric("Precision", fmt_pct(selected["Precision"]))
    b.metric("Recall / sensitivity", fmt_pct(selected["Recall"]))
    c.metric("F1-score", f"{selected['F1-score']:.4f}")
    d.metric("Accuracy", fmt_pct(selected["Accuracy"]))

    c_left, c_right = st.columns([1, 1], gap="large")
    with c_left:
        st.plotly_chart(draw_confusion_matrix(static_confusion_matrix(model_choice, strategy_choice), model_choice, strategy_choice), width="stretch")
    with c_right:
        section_heading("WHAT IT MEANS", "Count the trade-off")
        st.markdown(
            f"""
            <div class="info-card">
              <div class="small-kicker">At this cutoff</div>
              <h3>{selected['TP']} of 62 positive test patients detected</h3>
              <p><b>{selected['FN']}</b> positive patients were missed (false negatives).<br><br>
              <b>{selected['FP']}</b> non-deterioration patients were flagged (false positives).<br><br>
              <b>{selected['TP'] + selected['FP']}</b> patients were flagged in total, and <b>{selected['TN']}</b> negatives were correctly not flagged.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        if strategy_choice == "Recall-focused (80%)":
            st.markdown("<div class='warnbox'><b>Recall-focused strategy:</b> it detects more of the 62 positive test patients, but precision is only about 5–6%. That means many flags are false positives in this cohort.</div>", unsafe_allow_html=True)
        elif strategy_choice == "Default 0.50" and model_choice == "Random Forest":
            st.markdown("<div class='dangerbox'><b>A useful warning:</b> at 0.50, Random Forest predicts every test patient as negative. Its accuracy is 96.90%, simply reflecting the majority class, while recall is 0%.</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='callout'><b>Threshold selection matters.</b> A threshold that improves one metric can reduce another. Select thresholds using training/validation data, not by tuning directly on the held-out test set.</div>", unsafe_allow_html=True)

    st.write("")
    section_heading("ALL STRATEGIES", "Side-by-side test-set results")
    rows = []
    for model in MODEL_NAMES:
        for strat, values in STATIC_STRATEGY_METRICS[model].items():
            rows.append({"Model": model, "Strategy": strat, "Threshold": STATIC_THRESHOLDS[model][strat], **values})
    all_strategies = pd.DataFrame(rows)
    for col in ["Threshold", "Accuracy", "Balanced accuracy", "Precision", "Recall", "F1-score"]:
        all_strategies[col] = all_strategies[col].map(lambda v: f"{v:.4f}")
    st.dataframe(all_strategies, width="stretch", hide_index=True)

elif page == "Patient Predictor":
    hero(
        "Explore one hypothetical patient record.",
        "Upload the hourly panel dataset and train the pipelines in this session to enable a live model demonstration. A compatible dataset is needed before live predictions can be generated.",
        eyebrow="INTERACTIVE MODEL LAB",
    )
    st.markdown(
        "<div class='dangerbox'><b>Research-only safeguard:</b> do not enter identifiable or real patient information. This prototype is not validated for patient care, does not provide a calibrated clinical risk estimate, and must not be used to make diagnoses or treatment decisions.</div>",
        unsafe_allow_html=True,
    )
    bundle = st.session_state.get("training_bundle")
    if bundle is None:
        st.info("No model has been trained in this session yet. Open **Data & Methodology**, upload a compatible CSV, and click **Train models**. Live prediction is enabled after retraining the models on the uploaded data.")
        st.markdown("### What the form will use")
        st.markdown("The final analysis uses 16 admission-time predictors: six vital signs, mobility score, five laboratory measurements, age, gender, comorbidity index and admission type. Identifier, outcome-timing and unverified timestamp-dependent fields are excluded.")
    else:
        st.markdown(
            f"<div class='callout'><b>Session model ready.</b> Trained on {bundle['train_patients']:,} patients and evaluated on {bundle['test_patients']:,} held-out patients from your uploaded dataset. Positive-class prevalence in the hour-0 cohort: {bundle['prevalence']:.2%}.</div>",
            unsafe_allow_html=True,
        )
        with st.form("patient_prediction_form"):
            st.markdown("### Admission-time features")
            left, right = st.columns(2, gap="large")
            defaults = bundle["numeric_defaults"]
            values: dict[str, Any] = {}
            fields_left = ["heart_rate", "respiratory_rate", "spo2_pct", "temperature_c", "systolic_bp", "diastolic_bp", "mobility_score", "wbc_count"]
            fields_right = ["lactate", "creatinine", "crp_level", "hemoglobin", "age", "comorbidity_index", "gender", "admission_type"]
            labels = {
                "heart_rate": "Heart rate (beats/min)",
                "respiratory_rate": "Respiratory rate (breaths/min)",
                "spo2_pct": "Oxygen saturation SpO₂ (%)",
                "temperature_c": "Temperature (°C)",
                "systolic_bp": "Systolic blood pressure (mmHg)",
                "diastolic_bp": "Diastolic blood pressure (mmHg)",
                "mobility_score": "Mobility score",
                "wbc_count": "WBC count",
                "lactate": "Lactate",
                "creatinine": "Creatinine",
                "crp_level": "CRP level",
                "hemoglobin": "Hemoglobin",
                "age": "Age (years)",
                "comorbidity_index": "Comorbidity index",
            }
            with left:
                for feature in fields_left:
                    default = float(defaults.get(feature, 0.0))
                    if feature in {"mobility_score"}:
                        values[feature] = st.number_input(labels[feature], value=int(round(default)), step=1, key=f"pred_{feature}")
                    else:
                        values[feature] = st.number_input(labels[feature], value=float(default), format="%.3f", key=f"pred_{feature}")
            with right:
                for feature in fields_right:
                    if feature in {"gender", "admission_type"}:
                        opts = bundle["gender_options"] if feature == "gender" else bundle["admission_type_options"]
                        if not opts:
                            opts = ["unknown"]
                        values[feature] = st.selectbox(labels.get(feature, feature.replace("_", " ").title()), opts, key=f"pred_{feature}")
                    elif feature in {"age", "comorbidity_index"}:
                        default = int(round(float(defaults.get(feature, 0))))
                        values[feature] = st.number_input(labels[feature], value=default, step=1, key=f"pred_{feature}")
                    else:
                        default = float(defaults.get(feature, 0.0))
                        values[feature] = st.number_input(labels[feature], value=default, format="%.3f", key=f"pred_{feature}")
            model_choice = st.selectbox("Model", MODEL_NAMES, key="live_model_choice")
            strategy_choice = st.selectbox("Threshold strategy", ["F1-optimal", "Recall-focused (80%)", "Default 0.50"], key="live_strategy_choice")
            submitted = st.form_submit_button("Run research demonstration", type="primary", width="stretch")

        if submitted:
            model_input = pd.DataFrame([{feature: values[feature] for feature in FEATURES}], columns=FEATURES)
            model = bundle["models"][model_choice]
            score = float(model.predict_proba(model_input)[:, 1][0])
            if strategy_choice == "Default 0.50":
                cutoff = 0.50
            else:
                cutoff = bundle["thresholds"][model_choice][strategy_choice]
            if pd.isna(cutoff):
                st.warning("A recall-focused training threshold was not identified for this fitted model. Try the F1-optimal or default threshold.")
            else:
                flagged = score >= float(cutoff)
                st.write("")
                r1, r2, r3 = st.columns(3)
                r1.metric("Model score (uncalibrated)", f"{score:.4f}")
                r2.metric("Selected cutoff", f"{float(cutoff):.4f}")
                r3.metric("Prototype output", "Flagged" if flagged else "Not flagged")
                if flagged:
                    st.markdown("<div class='warnbox'><b>Prototype output: flagged.</b> This means the model score crossed the selected experimental threshold. It is not a diagnosis or a validated estimate of a patient's true probability of deterioration.</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='callout'><b>Prototype output: below threshold.</b> The score did not cross this experimental cutoff. This does not establish that a patient is safe or will not deteriorate.</div>", unsafe_allow_html=True)
                row = bundle["results"]
                test_row = row[(row["Model"] == model_choice) & (row["Threshold Strategy"] == strategy_choice)]
                if not test_row.empty:
                    metric = test_row.iloc[0]
                    st.caption(f"On the held-out cohort for this uploaded dataset, this model/strategy had precision {metric['Precision']:.1%}, recall {metric['Recall']:.1%}, and {int(metric['FP'])} false positives. Those are dataset-level evaluation metrics, not guarantees for this hypothetical record.")

elif page == "Data & Methodology":
    hero(
        "Trace the data. Understand the limits.",
        "The research uses the merged hourly panel for quality checks and target validation, then isolates one hour-0 record per patient for the primary model comparison.",
        eyebrow="DATA AUDIT · REPRODUCIBILITY",
    )
    section_heading("DATASET OVERVIEW", "Admission-time cohort")
    a, b, c, d = st.columns(4)
    a.metric("Hourly panel", "417,866 rows", "27 columns")
    b.metric("Unique patients", "10,000", "10,000 hour-0 records")
    c.metric("Duplicate rows", "0", "as reported in the notebook")
    d.metric("Missing values", "0", "as reported in the notebook")

    st.write("")
    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        section_heading("FEATURE SET", "16 admission-time predictors")
        feature_groups = {
            "Vital signs": ["heart_rate", "respiratory_rate", "spo2_pct", "temperature_c", "systolic_bp", "diastolic_bp"],
            "Clinical / functional": ["mobility_score"],
            "Laboratory values": ["wbc_count", "lactate", "creatinine", "crp_level", "hemoglobin"],
            "Demographic / baseline": ["age", "gender", "comorbidity_index"],
            "Admission metadata": ["admission_type"],
        }
        for group, items in feature_groups.items():
            with st.expander(f"{group} · {len(items)} feature(s)", expanded=(group == "Vital signs")):
                st.write(", ".join(items))
        st.markdown("**Excluded from primary predictors**")
        st.markdown("`patient_id`, `hour_from_admission`, `los_hours`, `deterioration_event`, `deterioration_hour`, the target label, and variables flagged for additional provenance/timing verification (`oxygen_device`, `oxygen_flow`, `nurse_alert`, `sepsis_risk_score`, `baseline_risk_score`).")
    with right:
        section_heading("TARGET VALIDATION", "The label matches event timing")
        audit = pd.DataFrame(
            [
                {"Timing check": "Unique patients", "Count": 10_000},
                {"Timing check": "Patients with a recorded event", "Count": 1_938},
                {"Timing check": "Events recorded at hour 0", "Count": 0},
                {"Timing check": "Events during hours 1–12", "Count": 309},
                {"Timing check": "Events after hour 12", "Count": 1_629},
                {"Timing check": "Target mismatches in notebook", "Count": 0},
            ]
        )
        st.dataframe(audit, width="stretch", hide_index=True)
        st.markdown("<div class='callout'><b>Target definition:</b> `deterioration_event = 1` and `deterioration_hour` between 0 and 12. No events were recorded at hour 0, so the recorded target also matched a strict hours 1–12 definition in the notebook.</div>", unsafe_allow_html=True)

    st.write("")
    section_heading("SOURCE", "Dataset references")
    st.markdown("The report identifies the Hospital Deterioration Dataset by T. Masryo as the source. Use the dataset documentation to confirm versions, column definitions, and timing assumptions.")
    ref1, ref2 = st.columns(2, gap="medium")
    with ref1:
        st.link_button("Open dataset on Hugging Face", "https://huggingface.co/datasets/tarekmasryo/hospital-deterioration-dataset", width="stretch")
    with ref2:
        st.link_button("Open data dictionary", "https://huggingface.co/datasets/tarekmasryo/hospital-deterioration-dataset/blob/main/docs/data_dictionary.md", width="stretch")

    st.write("")
    section_heading("DESCRIPTIVE ANALYSIS", "Admission-time group means")
    st.markdown("These are descriptive group averages. Differences are not causal effects and are not, by themselves, proof that a feature predicts deterioration for an individual.")
    chosen_features = st.multiselect(
        "Select features to compare",
        MEAN_FEATURES["Feature"].tolist(),
        default=["Heart rate", "Respiratory rate", "SpO₂", "Systolic BP"],
    )
    means_long = MEAN_FEATURES[MEAN_FEATURES["Feature"].isin(chosen_features)].melt(
        id_vars=["Feature", "Unit"], value_vars=["No deterioration", "Deterioration within 12h"], var_name="Group", value_name="Mean"
    )
    if not means_long.empty:
        fig = px.bar(means_long, x="Feature", y="Mean", color="Group", barmode="group", color_discrete_sequence=["#9aabba", "#0d9488"], hover_data=["Unit", "Mean"])
        fig.update_layout(height=350, margin=dict(l=5, r=5, t=15, b=5), paper_bgcolor="#0a0f1b", plot_bgcolor="#111a2b", legend_title_text="", xaxis_title="", yaxis_title="Group mean", font=dict(family="DM Sans", color="#e7eef8"))
        st.plotly_chart(fig, width="stretch")

    st.write("")
    section_heading("REPRODUCIBILITY LAB", "Train models using your CSV")
    st.markdown("Upload the merged hourly panel used by the notebook. The app checks the target against event timing, selects hour-0 rows, checks for one row per patient, applies a stratified 80:20 patient-level split (random state 2026), selects thresholds using training out-of-fold probabilities, and fits the three pipelines.")
    st.download_button(
        "Download required-column template (header only)",
        data=make_schema_csv(),
        file_name="hospital_deterioration_required_schema.csv",
        mime="text/csv",
        help="This is a blank column template, not a dataset. It contains no patient records.",
    )
    uploaded = st.file_uploader("Upload merged hourly panel CSV", type=["csv"], key="training_dataset_uploader")
    if uploaded is not None:
        st.caption(f"Selected file: {uploaded.name} · {uploaded.size / (1024 * 1024):.2f} MB")
        if st.button("Train models", type="primary", width="stretch"):
            try:
                with st.spinner("Validating the dataset and fitting three pipelines. Cross-validated threshold selection may take a while on CPU..."):
                    uploaded_df = pd.read_csv(BytesIO(uploaded.getvalue()))
                    new_bundle = train_uploaded_dataset(uploaded_df)
                    new_bundle["source_filename"] = uploaded.name
                    st.session_state["training_bundle"] = new_bundle
                st.success("Training completed. Open Patient Predictor to explore a hypothetical input using these session models.")
                st.rerun()
            except Exception as exc:
                st.error(f"Training was stopped: {exc}")
    elif st.session_state.get("training_bundle") is not None:
        st.success(f"A trained bundle is already available in this session from {st.session_state['training_bundle'].get('source_filename', 'uploaded data')}.")
    else:
        st.info("No uploaded dataset is included in the starter package. The dashboard remains usable; live model training requires your CSV.")

    bundle = st.session_state.get("training_bundle")
    if bundle is not None:
        st.write("")
        section_heading("SESSION TRAINING RESULTS", "Your uploaded dataset's performance")
        st.caption(f"Source: {bundle.get('source_filename', 'uploaded CSV')} · Train/test split: {bundle['train_patients']:,}/{bundle['test_patients']:,} patient records · positives in test: {bundle['test_positives']:,}")
        strategy = st.selectbox("Threshold strategy for uploaded-data results", ["F1-optimal", "Recall-focused (80%)", "Default 0.50"], key="uploaded_result_strategy")
        dyn = model_lab_metrics(bundle, strategy)
        showcols = ["Model", "Threshold", "Accuracy", "Balanced accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "Average Precision", "FP", "FN", "TP", "TN"]
        dyn_display = dyn[showcols].copy()
        for col in ["Threshold", "Accuracy", "Balanced accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "Average Precision"]:
            dyn_display[col] = dyn_display[col].map(lambda value: f"{value:.4f}")
        st.dataframe(dyn_display, width="stretch", hide_index=True)
        st.download_button(
            "Download uploaded-data model comparison",
            data=dyn.to_csv(index=False).encode("utf-8"),
            file_name="uploaded_data_model_comparison.csv",
            mime="text/csv",
        )

    st.write("")
    section_heading("IMPORTANT LIMITATIONS", "What this study cannot establish")
    limitations = st.columns(2, gap="medium")
    with limitations[0]:
        card("DATA TIMING", "Feature availability is unverified", "The available CSV does not include feature-specific measurement or lab-result timestamps. A value on the hour-0 panel does not prove that clinicians would have had it available at the intended prediction instant.")
        st.write("")
        card("TARGET PREVALENCE", "Rare positive class", "Only 309 of 10,000 patients are positive. Threshold-dependent precision and recall estimates are uncertain, especially with just 62 positive patients in the held-out test set.")
    with limitations[1]:
        card("GENERALISATION", "No external validation", "External validation on another hospital or prospective deployment has not been performed in this study. Performance may change in different settings and patient populations.")
        st.write("")
        card("MODEL COMPARISON", "No definitive winner", "All paired bootstrap confidence intervals for ROC-AUC and Average Precision differences include zero. Current point-estimate differences do not prove one algorithm is superior.")

    with st.expander("Research methods in one view"):
        st.markdown(
            """
            - **Unit of analysis:** one hour-0 record per patient; 10,000 patients.
            - **Target:** `deterioration_within_12h_from_admission`.
            - **Split:** stratified 80:20 patient-level split; 8,000 training and 2,000 test patients; random state 2026.
            - **Models:** Logistic Regression (`class_weight='balanced'`), Random Forest (300 trees, `balanced_subsample`, `min_samples_leaf=2`), and XGBoost (200 estimators, depth 3, learning rate 0.05, `scale_pos_weight` for class imbalance).
            - **Preprocessing:** median numeric imputation; most-frequent categorical imputation + one-hot encoding; scaling only for Logistic Regression. Preprocessing is contained within each pipeline and fitted on training folds.
            - **Threshold selection:** five-fold stratified training out-of-fold probabilities; F1-optimal and recall-focused (at least 80% training recall) cutoffs.
            - **Uncertainty:** repeated stratified cross-validation (5 folds × 2 repeats) and 2,000 paired stratified bootstrap resamples of the held-out test set.
            """
        )

st.write("")
st.markdown(
    "<div style='padding-top:1rem;border-top:1px solid #29384d;color:#8fa1b7;font-size:.78rem;display:flex;justify-content:space-between;gap:1rem;flex-wrap:wrap;'><span>Hospital Deterioration Prediction System</span><span>Garv Dubey</span></div>",
    unsafe_allow_html=True,
)
