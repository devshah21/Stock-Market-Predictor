import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
import plotly.graph_objs as go

START = "2019-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Custom CSS style for the slider
CUSTOM_CSS = """
<style>
/* Custom Streamlit slider style */
.st-eb {
    color: #555;
    font-size: 16px;
}
.st-f6 {
    color: #555;
    font-size: 14px;
}
/* Custom Streamlit slider track color */
.st-c0 {
    background-color: #ddd;
}
/* Custom Streamlit slider thumb color */
.st-c4 {
    background-color: #007BFF;
}
</style>
"""

def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


def plot_raw_data(data):
    # Plotting raw data
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], mode='lines', name="Stock Open", line=dict(color='royalblue')))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name="Stock Close", line=dict(color='darkorange')))

    fig.update_layout(
        title_text='Stock Price History',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=True,
        showlegend=True,
        legend=dict(x=0, y=1)
    )

    st.plotly_chart(fig)


def plot_forecast_data(forecast):
    # Plotting forecast data
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color='royalblue')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='none', name='Lower Bound', fillcolor='rgba(135, 206, 250, 0.2)'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='none', name='Upper Bound', fillcolor='rgba(135, 206, 250, 0.2)'))

    fig.update_layout(
        title_text='Stock Price Forecast',
        xaxis_title='Date',
        yaxis_title='Price',
        showlegend=True,
        legend=dict(x=0, y=1),
        xaxis_rangeslider_visible=False,
        xaxis=dict(showgrid=True),
        yaxis=dict(showgrid=True),
    )

    st.plotly_chart(fig)


def main():
    # Apply custom CSS style to the slider
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.title('Stock Forecast App')

    stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
    selected_stock = st.selectbox('Select dataset for prediction', stocks)

    n_years = st.slider('Years of prediction:', 1, 4)
    period = n_years * 365

    # Load data
    with st.spinner('Loading data...'):
        data = load_data(selected_stock)
    st.success('Loading data... done!')

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
    st.subheader(f'Forecast plot for {n_years} years')
    plot_forecast_data(forecast)

    # Display forecast components
    st.subheader("Forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)


if __name__ == "__main__":
    main()
