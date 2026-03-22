import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Bulldozer Price Prediction", layout="wide")
st.title("🚜 Bulldozer Sale Price Prediction")

MODEL_PATH = "model/model.pkl"
FEATURE_PATH = "model/features.pkl"

# ---------------------- LOAD DATA ----------------------
@st.cache_data
def load_data():
    df = pd.read_csv(
        "data/bluebook-for-bulldozers/TrainAndValid.csv",
        low_memory=False,
        parse_dates=["saledate"]
    )
    return df

# ---------------------- PREPROCESS ----------------------
def preprocess_data(df):
    df = df.copy()

    df["saleYear"] = df.saledate.dt.year
    df["saleMonth"] = df.saledate.dt.month
    df["saleDay"] = df.saledate.dt.day
    df["saleDayOfWeek"] = df.saledate.dt.dayofweek
    df["saleDayOfYear"] = df.saledate.dt.dayofyear
    df.drop("saledate", axis=1, inplace=True)

    for label, content in df.items():
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                df[label + "_is_missing"] = pd.isnull(content)
                df[label] = content.fillna(content.median())
        else:
            df[label + "_is_missing"] = pd.isnull(content)
            df[label] = pd.Categorical(content).codes + 1

    return df

# ---------------------- TRAIN MODEL ----------------------
def train_model(df):
    df = df.sort_values(by="saledate")
    df = preprocess_data(df)

    df_train = df[df.saleYear != 2012]
    df_val = df[df.saleYear == 2012]

    X_train = df_train.drop("SalePrice", axis=1)
    y_train = df_train["SalePrice"]
    X_val = df_val.drop("SalePrice", axis=1)
    y_val = df_val["SalePrice"]

    model = RandomForestRegressor(
        n_estimators=40,
        min_samples_leaf=1,
        min_samples_split=14,
        max_features=0.5,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train, y_train)

    # ✅ Save model & features
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(X_train.columns, FEATURE_PATH)

    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    r2 = r2_score(y_val, preds)

    return model, mae, r2, X_train.columns

# ---------------------- LOAD MODEL ----------------------
def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURE_PATH):
        model = joblib.load(MODEL_PATH)
        cols = joblib.load(FEATURE_PATH)
        return model, cols
    return None, None

# ---------------------- UI ----------------------

st.sidebar.header("⚙️ Controls")

if st.sidebar.button("Train Model"):
    df = load_data()
    st.write("## Raw Data Sample")
    st.dataframe(df.head())

    with st.spinner("Training model..."):
        model, mae, r2, cols = train_model(df)

    st.success("✅ Model trained & saved!")
    st.write(f"Validation MAE: {mae:.2f}")
    st.write(f"Validation R²: {r2:.3f}")

    st.session_state["model"] = model
    st.session_state["cols"] = cols

if st.sidebar.button("Load Saved Model"):
    model, cols = load_model()

    if model is not None:
        st.session_state["model"] = model
        st.session_state["cols"] = cols
        st.success("✅ Model loaded successfully!")
    else:
        st.error("❌ No saved model found. Train first.")

# ---------------------- PREDICTION ----------------------

if "model" in st.session_state:
    st.write("## 📊 Make Predictions on Test Data")

    if st.button("Predict on Test.csv"):
        df_test = pd.read_csv(
            "data/bluebook-for-bulldozers/Test.csv",
            low_memory=False,
            parse_dates=["saledate"]
        )

        df_test = preprocess_data(df_test)

        # Align columns with training
        df_test = df_test.reindex(columns=st.session_state["cols"], fill_value=0)

        preds = st.session_state["model"].predict(df_test)

        st.write("### Sample Predictions")
        st.dataframe(pd.DataFrame({"Prediction": preds}).head())

        # Download predictions
        out = pd.DataFrame({"SalePrice": preds})
        csv = out.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Predictions",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )

else:
    st.info("👉 Train or Load a model from the sidebar to start.")