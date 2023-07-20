import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
import plotly.graph_objs as go

START = "2019-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

def main():
    st.title('Stock Forecast App') 

    stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
    selected_stock = st.selectbox('Select dataset for prediction', stocks)

    n_years = st.slider('Years of prediction:', 1, 4)
    period = n_years * 365

    @st.cache_data()
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data

    data_load_state = st.text('Loading data...')
    data = load_data(selected_stock)
    data_load_state.text('Loading data... done!')

    # Display raw data
    st.subheader('Raw data')
    st.write(data.tail())

    # Plot raw data
    plot_raw_data(data)

    # Preparing data for Prophet
    df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})

    # Predicting the stock market price using the Prophet library
    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    # Display forecast data
    st.subheader('Forecast data')
    st.write(forecast.tail())

    # Display forecast plot
    st.write(f'Forecast plot for {n_years} years')
    plot_forecast_data(forecast)

    # Display forecast components
    st.write("Forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)

def plot_raw_data(data):
    # Plotting raw data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.update_layout(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

def plot_forecast_data(forecast):
    # Plotting forecast data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='none', name='Forecast Lower Bound'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='none', name='Forecast Upper Bound'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], line=dict(color='orange'), mode='lines', name='Forecast'))
    fig.update_layout(title_text='Stock Price Forecast', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
