import streamlit as st
import pandas as pd
import plotly.express as px
from tensorflow.keras.models import load_model

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Concatenate, SimpleRNN, Dense, Flatten, Lambda, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from IPython.display import clear_output
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


import nbimporter
from load_model_exec import run_load_model,load_csv_file


csv2 = 'Data/BE_All_Feeders__BE2B1__with_weather.csv'
csv1 = 'Data/TY_All_Feeders__TY2B10__with_weather.csv'
model = load_model("h6_model.keras")


st.set_page_config(page_title="CSV Dashboard + Forecasting", layout="wide")
st.title("📊 CSV Dashboard Tool with Forecasting")

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader(f"Data Preview for {uploaded_file.name}")
    st.dataframe(df, use_container_width=True)

    st.subheader("Graph Options")
    columns = df.columns.tolist()

    x_col = st.selectbox("X-axis", options=columns)
    y_col = st.selectbox("Y-axis", options=columns)
    chart_type = st.selectbox("Chart Type", ["Line", "Bar", "Scatter"])

    if st.button("Generate Chart"):
        if chart_type == "Line":
            fig = px.line(df, x=x_col, y=y_col)
        elif chart_type == "Bar":
            fig = px.bar(df, x=x_col, y=y_col)
        else:
            fig = px.scatter(df, x=x_col, y=y_col)
        st.plotly_chart(fig, use_container_width=True)

# === Forecasting Section ===
if st.button("Run Model Forecast"):
    if uploaded_file is not None:
        import tempfile
        extracted = uploaded_file.name.removesuffix(".csv")
        uploaded_file.seek(0)  # Reset the file pointer
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        
    else:
        st.warning("⚠️ Please upload a CSV file before running the forecast.")
        st.stop()

    try:
        model = load_model("h6_model.keras")
        df_plot, extracted1 = run_load_model(tmp_file_path, model)
        st.success("✅ Forecasting and plotting completed successfully.")
        
        st.write("Forecasted Data Preview")
        st.dataframe(df_plot.head())

        st.subheader("Graph Options")
        x_col = st.selectbox("X-axis", options=[df_plot.columns[0]])
        y_col1 = st.selectbox("Y-axis 1", options=df_plot.columns, index=1)
        y_col2 = st.selectbox("Y-axis 2", options=df_plot.columns, index=2)

        fig2 = px.line(df_plot, x=x_col, y=[y_col1, y_col2],
                       title=(f"<span style='font-size:20px'>{extracted}</span><br>"
                               f"{y_col1} and {y_col2} over {x_col}"),
                       markers=True)
        st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error("❌ An error occurred during model loading or forecasting.")
        st.exception(e)
