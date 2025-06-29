import streamlit as st
import pandas as pd
import plotly.express as px
from tensorflow.keras.models import load_model

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import nbimporter
from load_model_exec import run_load_model

import tempfile
from pycallgraph2 import PyCallGraph
from pycallgraph2.output import GraphvizOutput

# Page setup
st.set_page_config(page_title="CSV Dashboard + Forecasting", layout="wide")
#st.title(f"<center>BLPC Dashboard</center  >")
st.markdown("<h1 style='text-align: center;'>BLPC Dashboard</h1>", unsafe_allow_html=True)
# Upload CSV file
uploaded_file = st.file_uploader("Upload Data File",type="csv")

if uploaded_file is not None:
    # Display raw CSV info
    st.markdown("### Data Preview")
    df = pd.read_csv(uploaded_file)
    st.markdown(f"<p style='text-align: center;'>Raw Data</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center;'>{uploaded_file}</h1>", unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True)

    # Charting Section
    with st.expander("Analytics", expanded=True):
        st.markdown("#### Chart")
        columns = df.columns.tolist()

        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("X-axis", options=columns)
        with col2:
            y_col = st.selectbox("Y-axis", options=columns)

        chart_type = st.radio("Chart Type", ["Line", "Bar", "Scatter"], horizontal=True)

        if chart_type == "Line":
            fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
        elif chart_type == "Bar":
            fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
        else:
            fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")

        st.plotly_chart(fig, use_container_width=True)

    # Run model automatically
    st.markdown("### Forecast")
    try:
        # Save uploaded file to temp path
        extracted = uploaded_file.name.removesuffix(".csv")
        uploaded_file.seek(0)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        st.markdown(f"### {extracted}") 
        # Run forecasting
        model = load_model("h6_model.keras")
        df_plot, extracted_name, mae,rmse,mape = run_load_model(tmp_file_path, model)

        st.success(f"✅ Forecast for **{extracted}** completed successfully.")
        #st.dataframe(df_plot.head(), use_container_width=True)

        # Forecast graph options
        st.markdown("## Forecast Chart Options")
        col3, col4, col5 = st.columns(3)
        with col3:
            x_col = st.selectbox("X-axis", options=df_plot.columns, index=0, key="x2")
        with col4:
            y_col1 = st.selectbox("Y-axis 1", options=df_plot.columns, index=1, key="y1")
        with col5:
            #y_col2 = st.selectbox("Y-axis 2", options=df_plot.columns, index=2, key="y2")
            y_col2 = "Predicted_Future"

        fig2 = px.line(df_plot, x=x_col, y=[y_col1, y_col2],
                       title=(f"<span style='font-size:20px'>{extracted}</span><br>"
                               f"{y_col1} and {y_col2} over {x_col}"),
                       markers=True)
        fig2.update_layout(title_x=0.4,xaxis_title="Time",yaxis_title="Power (kW)")
        st.plotly_chart(fig2, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{mae:.2f}")
        col2.metric("RMSE", f"{rmse:.2f}")
        col3.metric("MAPE", f"{mape:.2f}%")

        if 'pv_kw' in df.columns:
            top_load = df['pv_kw'].max()
            avg_load = df['pv_kw'].mean()
            bottom_load = df['pv_kw'].min()
        else:
            st.warning(" 'pv_kw' column not found for load calculation.")
        
        col_top, col_avg, col_bot = st.columns(3)
        col_top.metric("Top Load", f"{top_load:,.2f} kW")
        col_avg.metric("Average Load", f"{avg_load:,.2f} kW")
        col_bot.metric("Bottom Load", f"{bottom_load:,.2f} kW")
        


    
    
    except Exception as e:
        st.error("❌ An error occurred while processing the forecast.")
        st.exception(e)
