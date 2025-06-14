import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

st.set_page_config(page_title="📈 Prophet Forecast App", layout="wide")
st.title("📈 Sales Forecast using Prophet (Custom Date Range)")

# File upload
uploaded_file = st.file_uploader("📤 Upload a CSV with 'Date' and 'Sales' columns", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # Basic validation
    if 'Date' not in data.columns or 'Sales' not in data.columns:
        st.error("❌ CSV must contain 'Date' and 'Sales' columns.")
    else:
        # Convert and prepare
        data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
        df = data.rename(columns={'Date': 'ds', 'Sales': 'y'}).sort_values("ds")

        st.success("✅ File uploaded successfully!")
        st.markdown(f"🗓️ **Available data range:** `{df['ds'].min().date()}` to `{df['ds'].max().date()}`")

        # Prediction range input
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("🔽 Prediction Start Date", value=df['ds'].max().date() + pd.Timedelta(days=1), min_value=df['ds'].max().date() + pd.Timedelta(days=1))
        with col2:
            end_date = st.date_input("🔼 Prediction End Date", value=df['ds'].max().date() + pd.Timedelta(days=90), min_value=start_date)

        if st.button("🚀 Run Forecast"):
            # Prophet modeling
            model = Prophet()
            model.fit(df)

            # Generate future dates
            full_future_dates = pd.date_range(start=df['ds'].min(), end=end_date)
            future_df = pd.DataFrame({'ds': full_future_dates})
            forecast = model.predict(future_df)

            # Filter forecast for selected range
            forecast_filtered = forecast[(forecast['ds'] >= pd.to_datetime(start_date)) & (forecast['ds'] <= pd.to_datetime(end_date))]

            # Plot
            st.subheader(f"📊 Forecast from `{start_date}` to `{end_date}`")
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(df['ds'], df['y'], label="Historical Sales", color="blue")
            ax.plot(forecast_filtered['ds'], forecast_filtered['yhat'], label="Predicted Sales", color="green")
            ax.fill_between(forecast_filtered['ds'], forecast_filtered['yhat_lower'], forecast_filtered['yhat_upper'], color='lightgreen', alpha=0.4, label="Confidence Interval")
            ax.set_title("📈 Sales Forecast")
            ax.set_xlabel("Date")
            ax.set_ylabel("Sales")
            ax.legend()
            st.pyplot(fig)

            # Rename for display
            display_df = forecast_filtered[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={
                'ds': 'Date',
                'yhat': 'Predicted Sales',
                'yhat_lower': 'Lower Estimate',
                'yhat_upper': 'Upper Estimate'
            })

            st.subheader("📋 Forecast Table")
            st.write(display_df.reset_index(drop=True))

            # Accuracy on historical data
            try:
                historical_pred = forecast.set_index('ds').loc[df['ds']]['yhat']
                rmse = np.sqrt(mean_squared_error(df['y'], historical_pred))
                mae = mean_absolute_error(df['y'], historical_pred)

                st.markdown(f"✅ **RMSE (historical):** `{rmse:.2f}`")
                st.markdown(f"✅ **MAE (historical):** `{mae:.2f}`")
            except:
                st.warning("⚠️ Could not compute accuracy metrics.")
