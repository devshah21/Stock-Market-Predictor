import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2019-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

def main():
    st.title('Stock Forecast App') 

    stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
    selected_stock = st.selectbox('Select dataset for prediction', stocks)

    n_years = st.slider('Years of prediction:', 1, 4)
    period = n_years * 365

    @st.cache
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
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    # Display forecast components
    st.write("Forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)

def plot_raw_data(data):
    # Plotting raw data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
