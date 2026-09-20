import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

from xgboost import XGBClassifier


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Hospital Deterioration Prediction",
    page_icon="🏥",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.title {
    font-size: 38px;
    font-weight: 700;
    color: #12355b;
}

.subtitle {
    font-size: 18px;
    color: #555;
    margin-bottom: 25px;
}

.card {
    padding: 20px;
    border-radius: 15px;
    background-color: white;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.08);
    text-align: center;
}

.metric-title {
    font-size: 15px;
    color: #666;
}

.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #12355b;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="title">🏥 Hospital Deterioration Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning based prediction of patient deterioration within 12 hours'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "merged hospital deterioration hourly panel dataset.csv"
    )

    return df


df = load_data()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Dashboard")

page = st.sidebar.radio(
    "Select Section",
    [
        "🏠 Overview",
        "📊 Dataset",
        "🤖 Model Prediction",
        "📈 Model Comparison",
        "🔍 Feature Importance",
        "📉 Confusion Matrix",
        "📈 ROC Curves"
    ]
)


# =========================================================
# MODEL TRAINING
# =========================================================

@st.cache_resource
def train_models(df):

    data = df.copy()

    # Target
    target = "deterioration_within_12h_from_admission"

    # Remove unnecessary variables
    drop_columns = [
        "patient_id",
        "los_hours",
        "deterioration_event",
        "deterioration_hour"
    ]

    data = data.drop(
        columns=drop_columns,
        errors="ignore"
    )

    # Separate X and y
    X = data.drop(columns=[target])
    y = data[target]

    # Convert categorical variables
    categorical_columns = [
        "gender",
        "oxygen_device",
        "admission_type"
    ]

    X = pd.get_dummies(
        X,
        columns=categorical_columns,
        drop_first=True
    )

    # Fill missing values
    X = X.fillna(X.median(numeric_only=True))

    # Scaling
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    X_scaled = pd.DataFrame(
        X_scaled,
        columns=X.columns
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # =====================================================
    # LOGISTIC REGRESSION
    # =====================================================

    log_reg = LogisticRegression(
        random_state=42,
        max_iter=1000
    )

    log_reg.fit(X_train, y_train)

    log_pred = log_reg.predict(X_test)
    log_prob = log_reg.predict_proba(X_test)[:, 1]

    # =====================================================
    # RANDOM FOREST
    # =====================================================

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)[:, 1]

    # =====================================================
    # XGBOOST
    # =====================================================

    xgb = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )

    xgb.fit(X_train, y_train)

    xgb_pred = xgb.predict(X_test)
    xgb_prob = xgb.predict_proba(X_test)[:, 1]

    # =====================================================
    # METRICS
    # =====================================================

    models = {
        "Logistic Regression": {
            "model": log_reg,
            "pred": log_pred,
            "prob": log_prob
        },

        "Random Forest": {
            "model": rf,
            "pred": rf_pred,
            "prob": rf_prob
        },

        "XGBoost": {
            "model": xgb,
            "pred": xgb_pred,
            "prob": xgb_prob
        }
    }

    results = []

    for name, info in models.items():

        pred = info["pred"]
        prob = info["prob"]

        results.append({

            "Model": name,

            "Accuracy": accuracy_score(
                y_test,
                pred
            ),

            "Precision": precision_score(
                y_test,
                pred,
                zero_division=0
            ),

            "Recall": recall_score(
                y_test,
                pred,
                zero_division=0
            ),

            "F1-Score": f1_score(
                y_test,
                pred,
                zero_division=0
            ),

            "ROC-AUC": roc_auc_score(
                y_test,
                prob
            )
        })

    results_df = pd.DataFrame(results)

    return (
        models,
        results_df,
        X_test,
        y_test,
        X.columns,
        scaler
    )


# =========================================================
# TRAIN
# =========================================================

with st.spinner("Training Machine Learning models..."):

    (
        models,
        results_df,
        X_test,
        y_test,
        feature_names,
        scaler
    ) = train_models(df)


# =========================================================
# OVERVIEW
# =========================================================

if page == "🏠 Overview":

    st.header("🏥 Hospital Deterioration Prediction")

    st.write(
        """
        This system uses Machine Learning to predict whether a patient
        is likely to experience deterioration within the next 12 hours.

        Three classification models are used:

        - Logistic Regression
        - Random Forest
        - XGBoost
        """
    )

    st.divider()

    # Dataset metrics

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Records",
            f"{len(df):,}"
        )

    with col2:

        st.metric(
            "Total Features",
            len(df.columns)
        )

    with col3:

        st.metric(
            "Patients",
            df["patient_id"].nunique()
        )

    with col4:

        st.metric(
            "Target",
            "12-Hour Risk"
        )

    st.divider()

    st.subheader("🎯 Prediction Target")

    st.info(
        """
        The target variable is:

        `deterioration_within_12h_from_admission`

        **0 = No deterioration**

        **1 = Deterioration within 12 hours**
        """
    )

    st.subheader("📊 Model Performance")

    display_results = results_df.copy()

    for column in display_results.columns[1:]:

        display_results[column] = (
            display_results[column] * 100
        ).round(2).astype(str) + "%"

    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# DATASET
# =========================================================

elif page == "📊 Dataset":

    st.header("📊 Dataset Exploration")

    st.write(
        "Hospital deterioration hourly panel dataset"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Rows",
            f"{df.shape[0]:,}"
        )

    with col2:

        st.metric(
            "Columns",
            df.shape[1]
        )

    st.subheader("First 20 Records")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.subheader("Dataset Information")

    info_df = pd.DataFrame({

        "Column": df.columns,

        "Data Type": [
            str(dtype)
            for dtype in df.dtypes
        ],

        "Missing Values": [
            df[col].isnull().sum()
            for col in df.columns
        ],

        "Unique Values": [
            df[col].nunique()
            for col in df.columns
        ]
    })

    st.dataframe(
        info_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("📈 Target Distribution")

    target_counts = df[
        "deterioration_within_12h_from_admission"
    ].value_counts()

    fig, ax = plt.subplots()

    ax.bar(
        ["No Deterioration", "Deterioration"],
        [
            target_counts.get(0, 0),
            target_counts.get(1, 0)
        ]
    )

    ax.set_ylabel("Number of Records")
    ax.set_title("Target Variable Distribution")

    st.pyplot(fig)


# =========================================================
# MODEL PREDICTION
# =========================================================

elif page == "🤖 Model Prediction":

    st.header("🤖 Patient Deterioration Prediction")

    st.write(
        """
        Enter patient information below and select a Machine Learning
        model to generate a deterioration risk prediction.
        """
    )

    selected_model = st.selectbox(
        "Select Model",
        list(models.keys())
    )

    st.divider()

    # Basic patient information

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input(
            "Age",
            min_value=0,
            max_value=120,
            value=50
        )

        heart_rate = st.number_input(
            "Heart Rate",
            min_value=20.0,
            max_value=250.0,
            value=75.0
        )

        respiratory_rate = st.number_input(
            "Respiratory Rate",
            min_value=5.0,
            max_value=60.0,
            value=18.0
        )

        spo2 = st.number_input(
            "SpO2 (%)",
            min_value=50.0,
            max_value=100.0,
            value=97.0
        )

    with col2:

        temperature = st.number_input(
            "Temperature (°C)",
            min_value=30.0,
            max_value=45.0,
            value=37.0
        )

        systolic_bp = st.number_input(
            "Systolic BP",
            min_value=50.0,
            max_value=250.0,
            value=120.0
        )

        diastolic_bp = st.number_input(
            "Diastolic BP",
            min_value=30.0,
            max_value=150.0,
            value=80.0
        )

        oxygen_flow = st.number_input(
            "Oxygen Flow",
            min_value=0.0,
            max_value=50.0,
            value=0.0
        )

    with col3:

        mobility_score = st.number_input(
            "Mobility Score",
            min_value=0,
            max_value=10,
            value=2
        )

        nurse_alert = st.selectbox(
            "Nurse Alert",
            [0, 1]
        )

        wbc = st.number_input(
            "WBC Count",
            min_value=0.0,
            max_value=100.0,
            value=7.0
        )

        lactate = st.number_input(
            "Lactate",
            min_value=0.0,
            max_value=20.0,
            value=1.2
        )

    col4, col5, col6 = st.columns(3)

    with col4:

        creatinine = st.number_input(
            "Creatinine",
            min_value=0.0,
            max_value=20.0,
            value=1.0
        )

        crp = st.number_input(
            "CRP Level",
            min_value=0.0,
            max_value=500.0,
            value=10.0
        )

    with col5:

        hemoglobin = st.number_input(
            "Hemoglobin",
            min_value=0.0,
            max_value=30.0,
            value=13.0
        )

        sepsis_risk = st.number_input(
            "Sepsis Risk Score",
            min_value=0.0,
            max_value=1.0,
            value=0.2
        )

    with col6:

        comorbidity = st.number_input(
            "Comorbidity Index",
            min_value=0,
            max_value=20,
            value=2
        )

        baseline_risk = st.number_input(
            "Baseline Risk Score",
            min_value=0.0,
            max_value=1.0,
            value=0.2
        )

    gender = st.selectbox(
        "Gender",
        df["gender"].dropna().unique()
    )

    oxygen_device = st.selectbox(
        "Oxygen Device",
        df["oxygen_device"].dropna().unique()
    )

    admission_type = st.selectbox(
        "Admission Type",
        df["admission_type"].dropna().unique()
    )

    # =====================================================
    # CREATE INPUT
    # =====================================================

    if st.button(
        "🔮 Predict Deterioration Risk",
        type="primary"
    ):

        input_data = pd.DataFrame({

            "hour_from_admission": [
                0
            ],

            "heart_rate": [
                heart_rate
            ],

            "respiratory_rate": [
                respiratory_rate
            ],

            "spo2_pct": [
                spo2
            ],

            "temperature_c": [
                temperature
            ],

            "systolic_bp": [
                systolic_bp
            ],

            "diastolic_bp": [
                diastolic_bp
            ],

            "oxygen_device": [
                oxygen_device
            ],

            "oxygen_flow": [
                oxygen_flow
            ],

            "mobility_score": [
                mobility_score
            ],

            "nurse_alert": [
                nurse_alert
            ],

            "wbc_count": [
                wbc
            ],

            "lactate": [
                lactate
            ],

            "creatinine": [
                creatinine
            ],

            "crp_level": [
                crp
            ],

            "hemoglobin": [
                hemoglobin
            ],

            "sepsis_risk_score": [
                sepsis_risk
            ],

            "age": [
                age
            ],

            "gender": [
                gender
            ],

            "comorbidity_index": [
                comorbidity
            ],

            "admission_type": [
                admission_type
            ],

            "baseline_risk_score": [
                baseline_risk
            ]
        })

        # One-hot encoding
        input_data = pd.get_dummies(
            input_data,
            columns=[
                "gender",
                "oxygen_device",
                "admission_type"
            ],
            drop_first=True
        )

        # Match training columns
        input_data = input_data.reindex(
            columns=feature_names,
            fill_value=0
        )

        # Scale
        input_scaled = scaler.transform(
            input_data
        )

        model = models[selected_model]["model"]

        probability = model.predict_proba(
            input_scaled
        )[0][1]

        prediction = model.predict(
            input_scaled
        )[0]

        st.divider()

        if prediction == 1:

            st.error(
                f"⚠️ Predicted Deterioration Risk: "
                f"{probability * 100:.2f}%"
            )

            st.warning(
                "The model predicts a deterioration event."
            )

        else:

            st.success(
                f"✅ Predicted Deterioration Risk: "
                f"{probability * 100:.2f}%"
            )

            st.info(
                "The model does not predict deterioration."
            )


# =========================================================
# MODEL COMPARISON
# =========================================================

elif page == "📈 Model Comparison":

    st.header("📈 Machine Learning Model Comparison")

    st.write(
        """
        Comparison of Logistic Regression, Random Forest,
        and XGBoost using Accuracy, Precision, Recall,
        F1-Score and ROC-AUC.
        """
    )

    display_df = results_df.copy()

    st.dataframe(
        display_df.style.format({
            "Accuracy": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1-Score": "{:.4f}",
            "ROC-AUC": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Performance Comparison")

    chart_df = results_df.set_index(
        "Model"
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    chart_df.plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylabel("Score")
    ax.set_xlabel("Models")
    ax.set_ylim(0, 1.05)

    plt.xticks(rotation=0)

    st.pyplot(fig)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

elif page == "🔍 Feature Importance":

    st.header("🔍 Feature Importance")

    selected_model = st.selectbox(
        "Select Tree-Based Model",
        [
            "Random Forest",
            "XGBoost"
        ]
    )

    model = models[
        selected_model
    ]["model"]

    importance = model.feature_importances_

    feature_importance = pd.DataFrame({

        "Feature": feature_names,

        "Importance": importance

    }).sort_values(
        "Importance",
        ascending=False
    )

    top10 = feature_importance.head(10)

    st.subheader("Top 10 Important Features")

    st.dataframe(
        top10,
        use_container_width=True,
        hide_index=True
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.barh(
        top10["Feature"],
        top10["Importance"]
    )

    ax.set_xlabel(
        "Importance Score"
    )

    ax.set_title(
        f"Top 10 Features - {selected_model}"
    )

    ax.invert_yaxis()

    st.pyplot(fig)


# =========================================================
# CONFUSION MATRIX
# =========================================================

elif page == "📉 Confusion Matrix":

    st.header("📉 Confusion Matrix")

    selected_model = st.selectbox(
        "Select Model",
        list(models.keys())
    )

    pred = models[
        selected_model
    ]["pred"]

    cm = confusion_matrix(
        y_test,
        pred
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "No Deterioration",
            "Deterioration"
        ],
        yticklabels=[
            "No Deterioration",
            "Deterioration"
        ],
        ax=ax
    )

    ax.set_xlabel(
        "Predicted Label"
    )

    ax.set_ylabel(
        "Actual Label"
    )

    ax.set_title(
        f"Confusion Matrix - {selected_model}"
    )

    st.pyplot(fig)


# =========================================================
# ROC CURVES
# =========================================================

elif page == "📈 ROC Curves":

    st.header("📈 ROC Curve Comparison")

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    for name, info in models.items():

        probability = info["prob"]

        fpr, tpr, _ = roc_curve(
            y_test,
            probability
        )

        auc_score = roc_auc_score(
            y_test,
            probability
        )

        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{name} (AUC = {auc_score:.4f})"
        )

    ax.plot(
        [0, 1],
        [0, 1],
        "--"
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "ROC Curve Comparison"
    )

    ax.legend()

    ax.grid(True)

    st.pyplot(fig)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Hospital Deterioration Prediction System | "
    "Machine Learning Project"
)