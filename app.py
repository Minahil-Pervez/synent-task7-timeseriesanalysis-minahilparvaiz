import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -------------------------
# Page Congiguration
# -------------------------

st.set_page_config(
    page_title="AI Stock Forecast Dashboard",
    page_icon="📈",
    layout="wide"

)

st.title("📈 AI Stock Forecast Dashboard")
st.write("Upload stock data and analyze trends with forecasting.")

# ----------------------------
# Uplod CSV
# ----------------------------

uploaded_file = st.file_uploader(
    "Upload Stock CSV",
    type=["csv"]
)

if uploaded_file is not None:

    # ----------------------
    # Load Data
    # ---------------------

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    required_columns =[
        "Date",
        "Open",
        "High",
        "Low",
        "Close"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns

    ]

    if missing_columns:

        st.error(
            f"Missing columns: {missing_columns}"
        )

        st.stop()

    # -----------------------
    # Preprocessing
    # -----------------------

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    # ------------------------
    # Dataset info
    # ------------------------

    st.subheader("Dataset Information")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("rows", df.shape[0])
    with col2:
        st.metric("rows", df.shape[1])

    # -----------------------
    # Candlestic Chart
    # -----------------------

    st.subheader("Candlestick Chart")

    candle_fig = go.Figure(
        data=[
            go.Candlestick( 
                x=df["Date"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"]
            )
        ]
    )
    
    candle_fig.update_layout(
        height=600
    )

    st.plotly_chart(
        candle_fig,
        use_container_width=True
    )
    

    # -------------------
    # Closing Trend Price
    #--------------------

    st.subheader("Closing Price Trend")

    trend_fig = go.Figure()

    trend_fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Close"],
            mode="lines",
            name="Close Price"        
        )
    )   

    st.plotly_chart(
        trend_fig,
        use_container_width=True
    ) 

    # --------------------
    # Moving Average
    # --------------------

    st.subheader("Moving Average Analysis")

    window = st.slider(
        "Select Moving Average Window",
        min_value=5,
        max_value=100,
        value=20
    )

    df["Moving_Average"] = (
        df["Close"]
        .rolling(window=window)
        .mean()
    )

    ma_fig = go.Figure()

    ma_fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Close"],
            mode="lines",
            name="Close Price"
        )
    )

    ma_fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Moving_Average"],
            mode="lines",
            name=f"{window} Day MA"
        )
    )

    st.plotly_chart(
        ma_fig,
        use_container_width=True 
    )

    # ------------------------
    # Prophet forecasting
    # ------------------------

    st.subheader("AI Forecasting")

    forecast_days = st.slider(
        "Days to Forecast",
        min_value=7,
        max_value=365,
        value=30
    )

    prophet_df = df[
        ["Date", "Close"]
    ].rename(
        columns={
             "Date": "ds",
           "Close": "y"
        }
    )

    model = Prophet()

    model.fit(prophet_df)

    future = model.make_future_dataframe(
        periods=forecast_days
    )

    forecast = model.predict(future)

    # ---------------------------
    # Forecast Plot
    # ---------------------------

    st.subheader("Forecast Result")
    forecast_plot = model.plot(
        forecast
    )

    st.pyplot(
        forecast_plot
    )

    # -----------------------
    # Forecasting Components
    # -----------------------

    st.subheader("Trend Components")

    components_plot = model.plot_components(
        forecast
    )

    st.pyplot(
        components_plot
    )

    # ---------------------
    # Accuracy Metrics
    # ---------------------

    st.subheader("Model Accuracy")

    actual_values = prophet_df["y"]

    predicted_values = forecast["yhat"][
        :len(actual_values)
    ]

    mae = mean_absolute_error(
        actual_values,
        predicted_values
    )
    
    rmse = np.sqrt(
        mean_squared_error(
            actual_values,
            predicted_values
        )
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric( 
            "MAE",
            round(mae, 2)
        )

    with col2:
        st.metric(
            "RMSE",
            round(rmse, 2)
        )

    # ---------------------------
    # Forecast Table
    # ---------------------------

    st.subheader("Forecast Data")

    forecast_table = forecast[
        [
            "ds",
            "yhat",
            "yhat_lower",
            "yhat_upper"
        ]
    ].tail(forecast_days)

    st.dataframe(
        forecast_table
    )
else:

    st.info(
        "Upload a CSV file to begin."
    )
    