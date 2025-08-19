import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
from datetime import date, timedelta
import os

# --- App Configuration ---
st.set_page_config(page_title="Stock Trend Analyzer", page_icon="📈")

# --- Model and Scaler Loading ---
@st.cache_resource
def load_prediction_model(ticker):
    """Loads the trained GRU model for a specific ticker."""
    model_path = f'models/{ticker}_model.h5'
    if os.path.exists(model_path):
        return load_model(model_path)
    return None

@st.cache_resource
def load_scaler_object(ticker):
    """Loads the saved scaler object for a specific ticker."""
    scaler_path = f'scalers/{ticker}_scaler.joblib'
    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)
    return None

# --- UI Elements ---
st.title("🤖 AI Stock Trend Analyzer")
st.write("This app uses specialist GRU models to predict the next day's closing price.")

# --- Auto-detect available models ---
models_dir = 'models'
if os.path.exists(models_dir):
    AVAILABLE_STOCKS = [f.replace('_model.h5', '') for f in os.listdir(models_dir) if f.endswith('_model.h5')]
    if not AVAILABLE_STOCKS:
        st.error("No trained models found in the 'models' folder.")
        AVAILABLE_STOCKS = []
else:
    st.error("The 'models' folder was not found. Please run the training script first.")
    AVAILABLE_STOCKS = []

# Create a dropdown menu with the auto-detected stocks
if AVAILABLE_STOCKS:
    ticker = st.selectbox("Select a Stock (you can also type to search):", AVAILABLE_STOCKS)

    # --- Prediction Logic ---
    if st.button("Predict Next Day's Price"):
        if not ticker:
            st.warning("Please select a stock.")
        else:
            model = load_prediction_model(ticker)
            scaler = load_scaler_object(ticker)
            
            if model is None or scaler is None:
                st.error(f"Model/Scaler for {ticker} could not be loaded. Ensure the files exist.")
            else:
                try:
                    # Fetch live data for the selected stock
                    end_date = date.today()
                    start_date = end_date - timedelta(days=3*365) # Fetch 3 years of data for the chart
                    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                    
                    if len(data) < 60:
                        st.error(f"Not enough live data for '{ticker}' to make a prediction.")
                    else:
                        st.write(f"### Recent Prices for {ticker}")
                        st.dataframe(data.tail())

                        last_60_days = data['Close'].tail(60).values.reshape(-1, 1)
                        scaled_last_60_days = scaler.transform(last_60_days)
                        X_pred = np.reshape(scaled_last_60_days, (1, 60, 1))

                        with st.spinner('Predicting...'):
                            predicted_price_scaled = model.predict(X_pred)
                        
                        predicted_price = scaler.inverse_transform(predicted_price_scaled)[0][0]

                        st.success(f"Predicted Next Day's Closing Price for {ticker}: **₹{predicted_price:.2f}**")
                        
                        # THIS IS THE CORRECTED LINE:
                        chart_data = data['Close'].reset_index()
                        chart_data.columns = ['Date', 'Close Price']
                        st.line_chart(chart_data.set_index('Date'))

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")