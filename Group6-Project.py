import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from streamlit.dataframe_util import Data
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import seaborn as sns

# MongoDB client connection
uri = "mongodb+srv://Emmaculate:1234Test@stockpricecluster.jzxjcmk.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(uri, server_api=ServerApi('1'))

try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")

def fetch_mongo_data(database_name, collection_name, symbol):
    db = client[database_name]
    collection = db[collection_name]
    data = pd.DataFrame(list(collection.find({'symbol': symbol})))
    
    if not data.empty:
        if 'timestamp' in data.columns:
            data.set_index('timestamp', inplace=True)
        elif 'reportDate' in data.columns:
            data.set_index('reportDate', inplace=True)
        else:
            print("Neither 'timestamp' nor 'reportDate' columns found in data.")
            return pd.DataFrame()
        data.index = pd.to_datetime(data.index)
        return data
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no data is found

st.title('Stock Data Dashboard')

popular_symbols = ["AAPL", "GOOGL", "A", "AA", "AACG", "AADI", "AAL", "V", "NVDA", "JPM", "AAM", "AAMC"]
symbol = st.sidebar.selectbox("Select a stock symbol:", popular_symbols, index=2)

data_type = st.selectbox('Select Data Type', ['Intraday', 'Earnings', 'Earnings Calendar', 'Daily'])
portfolio_size = st.number_input('Enter the size of your portfolio:', min_value=0.0)

if st.button('Fetch Data'):
    collection_name = data_type.lower().replace(" ", "_") + "_data"
    
    data = fetch_mongo_data('stock_database', collection_name, symbol)
    
    if not data.empty:
        st.write(f"Displaying {data_type} data for {symbol}")
        st.write(data.head())
        st.write("Columns available:", data.columns)
        
        if 'open' in data.columns:
            plt.figure(figsize=(10, 5))
            plt.plot(data.index, data['open'], label='Opening Price')
            plt.title(f'{symbol} {data_type} Opening Prices')
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            st.pyplot(plt)
        else:
            st.write(f"'open' column not found in the data.")
    else:
        st.write(f"No data found for {symbol}.")

def data_stock(symbol):
    data = yf.download(symbol)
    data = data[["Open", "High", "Low", "Close", "Volume"]]
    return data

def calculate_price_difference(data):
    latest_price = data.iloc[-1]["Close"]
    previous_year_price = data.iloc[-252]["Close"] if len(data) > 252 else data.iloc[0]["Close"]
    price_difference = latest_price - previous_year_price
    percentage_difference = (price_difference / previous_year_price) * 100
    return price_difference, percentage_difference


if symbol:
    stock_data = data_stock(symbol)

    if not stock_data.empty:
        price_difference, percentage_difference = calculate_price_difference(stock_data)
        latest_close_price = stock_data.iloc[-1]["Close"]
        max_52_week_high = stock_data["High"].tail(252).max()
        min_52_week_low = stock_data["Low"].tail(252).min()

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Close Price", f"${latest_close_price:.2f}")
        with col2:
            st.metric("Price Difference (YoY)", f"${price_difference:.2f}", f"{percentage_difference:+.2f}%", delta_color='normal')
        with col3:
            st.metric("52-Week High", f"${max_52_week_high:.2f}")
        with col4:
            st.metric("52-Week Low", f"${min_52_week_low:.2f}")

st.subheader("Stock Candlestick Chart")

stock_chart = go.Figure(data=[go.Candlestick(x=stock_data.index,
                                             open=stock_data['Open'],
                                             high=stock_data['High'],
                                             low=stock_data['Low'],
                                             close=stock_data['Close'])])
stock_chart.update_layout(title=f"{symbol} Stock Candlestick Chart", xaxis_rangeslider_visible=False)
st.plotly_chart(stock_chart, use_container_width=True)

def predict_stock_price(data):
    data['Date'] = data.index
    data['Date'] = data['Date'].map(pd.Timestamp.toordinal)  # Convert dates to ordinal
    X = data[['Date']]  # Features
    y = data['Close']  # Target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    
    future_date = pd.Timestamp.now().toordinal() + 30  # Predicting for 30 days into the future
    future_price = model.predict([[future_date]])[0]
    
    return future_price, mse

def calculate_volatility(data):
    data['Returns'] = data['Close'].pct_change()
    volatility = data['Returns'].std()  # Standard deviation of returns
    
    if volatility > 0.02:
        return "Highly Volatile"
    elif volatility > 0.01:
        return "Medium Risk"
    else:
        return "Stable"
    
def evaluate_valuation(current_pe, industry_pe_standard):
    if current_pe < industry_pe_standard * 0.9:
        return "Under-valued"
    elif current_pe > industry_pe_standard * 1.1:
        return "Over-valued"
    else:
        return "Fairly Valued"

col1, col2, col3 = st.columns(3, gap="medium")  
with(col1):
    if st.button('Predict Price'):
        future_price, mse = predict_stock_price(stock_data)
        st.metric('Predicted Price', f"${future_price:.2f}", delta=None, delta_color="inverse", label_visibility="visible")
        st.metric('Mean Squared Error', f"{mse:.2f}", delta=None, delta_color="inverse", label_visibility="visible")
with(col2):
    if st.button('Evaluate Volatility'):
        volatility = calculate_volatility(stock_data)
        st.write(f"Volatility: {volatility}")
with(col3):
    if st.button('Evaluate Valuation'):
        current_pe = st.number_input('Enter Current P/E Ratio:')
        industry_pe_standard = st.number_input('Enter Industry Standard P/E Ratio:')
        valuation = evaluate_valuation(current_pe, industry_pe_standard)
        st.write(f"Valuation: {valuation}")

def plot_volume_over_time(data, symbol):
        fig, ax1 = plt.subplots(figsize=(10, 5))

        ax1.set_xlabel('Date')
        ax1.set_ylabel('Closing Price', color='tab:blue')
        ax1.plot(data.index, data['Close'], color='tab:blue', label='Close Price')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Volume', color='tab:orange')
        ax2.bar(data.index, data['Volume'], color='tab:orange', alpha=0.5)
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        plt.title(f'{symbol} Closing Price and Volume Over Time')
        fig.tight_layout()
        st.pyplot(fig)
if st.button('Show Volume Over Time'):
        plot_volume_over_time(stock_data, symbol)

def correlation_marix(symbols):
    data = pd.DataFrame
    for symbol in symbols:
        data = yf.download(symbol, period='1mo')['Close']
        data = data_stock(symbol)    
    
    corr = data.corr()
    plt.figure(figsize=(10, 5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Stock Correlation')
    st.pyplot(plt)
symbols = st.multiselect('select Multiple Stock symbols', ['AAPL', 'GOOGL', 'A', 'AA', 'AACG', 'AADI', 'AAL', 'V', 'NVDA', 'JPM', 'AAM', 'AAMC'])
if st.button('Show Stock Correlation Matrix'):
    correlation_marix(symbols)
