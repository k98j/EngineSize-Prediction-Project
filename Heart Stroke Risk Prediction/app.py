from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "knn_heart_model.pkl"
SCALER_PATH = BASE_DIR / "heart_scaler.pkl"
COLUMNS_PATH = BASE_DIR / "heart_columns.pkl"


st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="heart",
    layout="centered",
)


@st.cache_resource
def load_artifacts():
    missing = [
        path.name
        for path in (MODEL_PATH, SCALER_PATH, COLUMNS_PATH)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required model file(s): " + ", ".join(missing)
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    expected_columns = joblib.load(COLUMNS_PATH)
    return model, scaler, expected_columns


def build_model_input(values, expected_columns):
    input_df = pd.DataFrame(0.0, index=[0], columns=expected_columns)

    numeric_values = {
        "Age": values["age"],
        "RestingBP": values["resting_bp"],
        "Cholesterol": values["cholesterol"],
        "FastingBS": values["fasting_bs"],
        "MaxHR": values["max_hr"],
        "Oldpeak": values["oldpeak"],
    }
    for col, value in numeric_values.items():
        if col in input_df.columns:
            input_df.loc[0, col] = value

    categorical_values = {
        f"Sex_{values['sex']}": 1,
        f"ChestPainType_{values['chest_pain']}": 1,
        f"RestingECG_{values['resting_ecg']}": 1,
        f"ExerciseAngina_{values['exercise_angina']}": 1,
        f"ST_Slope_{values['st_slope']}": 1,
    }
    for col, value in categorical_values.items():
        if col in input_df.columns:
            input_df.loc[0, col] = value

    return input_df


try:
    model, scaler, expected_columns = load_artifacts()
except Exception as exc:
    st.error(f"Could not load model files: {exc}")
    st.stop()


st.title("Heart Disease Risk Predictor")
st.caption("KNN model demo trained on heart disease data")
st.markdown("Enter patient details below and click **Predict risk**.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 100, 40)
        sex = st.selectbox("Sex", ["M", "F"])
        chest_pain = st.selectbox("Chest pain type", ["ATA", "NAP", "TA", "ASY"])
        resting_bp = st.number_input(
            "Resting blood pressure (mm Hg)",
            min_value=80,
            max_value=220,
            value=120,
        )
        cholesterol = st.number_input(
            "Cholesterol (mg/dL)",
            min_value=0,
            max_value=650,
            value=200,
        )
        fasting_bs = st.selectbox("Fasting blood sugar > 120 mg/dL", [0, 1])

    with col2:
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        max_hr = st.slider("Max heart rate", 60, 220, 150)
        exercise_angina = st.selectbox("Exercise-induced angina", ["Y", "N"])
        oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 6.5, 1.0, 0.1)
        st_slope = st.selectbox("ST slope", ["Up", "Flat", "Down"])

    submitted = st.form_submit_button("Predict risk", type="primary")


if submitted:
    values = {
        "age": age,
        "sex": sex,
        "chest_pain": chest_pain,
        "resting_bp": resting_bp,
        "cholesterol": cholesterol,
        "fasting_bs": fasting_bs,
        "resting_ecg": resting_ecg,
        "max_hr": max_hr,
        "exercise_angina": exercise_angina,
        "oldpeak": oldpeak,
        "st_slope": st_slope,
    }

    input_df = build_model_input(values, expected_columns)
    scaled_input = scaler.transform(input_df)
    prediction = int(model.predict(scaled_input)[0])

    probabilities = (
        model.predict_proba(scaled_input)[0]
        if hasattr(model, "predict_proba")
        else None
    )

    st.subheader("Prediction result")
    if prediction == 1:
        st.error("High risk of heart disease")
    else:
        st.success("Low risk of heart disease")

    if probabilities is not None:
        risk_probability = float(probabilities[1])
        st.metric("Estimated high-risk probability", f"{risk_probability:.1%}")

    with st.expander("Show model input columns"):
        st.dataframe(input_df, use_container_width=True)


st.info(
    "This is an educational ML demo, not a medical diagnosis. "
    "Please consult a qualified doctor for health decisions."
)
