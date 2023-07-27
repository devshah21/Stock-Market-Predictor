import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras.layers import Bidirectional, Dropout, Activation, Dense, LSTM
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.models import Sequential
from datetime import date

START = "2019-01-01"
TODAY = date.today().strftime("%Y-%m-%d")


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


def prepare_data(df_train):
    # Find the column with numeric values (assumed to be the 'Close' column)
    close_col_name = None
    for col in df_train.columns:
        if np.issubdtype(df_train[col].dtype, np.number):
            close_col_name = col
            break

    if close_col_name is None:
        raise ValueError("No numeric column found in the data for 'Close' price.")

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_train[close_col_name].values.reshape(-1, 1))

    # Create sequences for LSTM
    sequence_length = 60
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i + sequence_length])
        y.append(scaled_data[i + sequence_length])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler

seq_len = 60
dropout = 0.2
window_size = seq_len - 1

def create_lstm_model(input_shape):
    model = tf.keras.Sequential()

    model.add(
        LSTM(window_size, return_sequences=True, input_shape=input_shape)
    )

    model.add(Dropout(rate=dropout))
    # Bidirectional allows for training of sequence data forwards and backwards
    model.add(
        Bidirectional(LSTM((window_size * 2), return_sequences=True))
    )

    model.add(Dropout(rate=dropout))
    model.add(
        Bidirectional(LSTM(window_size, return_sequences=False))
    )

    model.add(Dense(units=1))
    # Linear activation function: activation is proportional to the input
    model.add(Activation('linear'))

    return model



def main():
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

    # Preparing data for LSTM
    df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
    df_train.set_index('ds', inplace=True)

    X, y, scaler = prepare_data(df_train)

    # Create LSTM model
    input_shape = (X.shape[1], 1)
    model = create_lstm_model(input_shape)

    # Compile and fit the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
    model.fit(X, y, epochs=50, batch_size=32)

    # Forecast using the fitted model
    last_sequence = X[-1]
    forecast = []
    for _ in range(period):
        next_pred = model.predict(last_sequence.reshape(1, last_sequence.shape[0], 1))
        forecast.append(next_pred[0][0])
        last_sequence = np.append(last_sequence[1:], next_pred[0][0])

    # Inverse transform the forecasted data
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

    # Generate forecast index
    forecast_index = pd.date_range(df_train.index[-1], periods=period + 1)

    # Remove the last date from the index to make it right-closed
    forecast_index = forecast_index[:-1]

    # Create a dataframe with forecast data
    forecast_df = pd.DataFrame({'yhat': forecast.flatten()}, index=forecast_index)

    # Display forecast data
    st.subheader('Forecast data')
    st.write(forecast_df.tail())

    # Display forecast plot
    st.subheader(f'Forecast plot for {n_years} years')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_train.index, y=df_train['y'], mode='lines', name="Stock Price", line=dict(color='royalblue')))
    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['yhat'], mode='lines', name="Forecast", line=dict(color='darkorange')))
    fig.update_layout(title_text='Stock Price Forecast', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)


if __name__ == "__main__":
    main()
