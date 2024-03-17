# Import libraries

import pandas as pd
import numpy as np
from datetime import timedelta
from datetime import datetime
import yfinance as yf
from scipy.signal import argrelextrema
import time
from pandas import ExcelWriter
import pandas_ta as ta
from dateutil.relativedelta import relativedelta
import datetime as dt
import warnings
warnings.filterwarnings('ignore')
import streamlit as st


# Add a horizontal rule and then display the centered title

st.markdown("<h1 style='text-align: center;'>Daily Stock Data Analysis</h1>", unsafe_allow_html=True)
st.markdown("---")

# import nifty 500 stocks

tickers = pd.read_csv("stock_info/symbol.csv")

# Get List of NSE 500
tickers = tickers['Ticker'].values

#symbols = st.sidebar.selectbox('Enter a Stock Symbol', tickers)

st.sidebar.info("Created and designed by Shiv")

# define interval
begin_time = datetime.today()-timedelta(days=465)
end_time = datetime.today()

# download the historical data
@st.cache_data
def get(tickers, startdate, enddate):
    @st.cache_resource
    def data(tickers):
        return (yf.download(tickers, start=startdate, end=enddate))
    datas = map(data, tickers)
    return(pd.concat(datas, keys=tickers, names=['Ticker', 'Date']))


all_data = get(tickers, begin_time, end_time)

# get the only closing price
df = np.round(all_data[['Adj Close','Volume','Open','High','Low']].reset_index(),2) 
df.rename(columns={'Adj Close':'Close'}, inplace=True)

# Group by ticker
grouped_data = df.groupby('Ticker')

# Define a function to apply technical analysis
@st.cache_data
def apply_ta(group_df):
    group_df.ta.sma(length=20, append=True)
    group_df.ta.sma(length=50, append=True)  # Add 50-day SMA
    group_df.ta.sma(length=200, append=True)  # Add 200-day SMA
    group_df['VOL_SMA_20'] = group_df['Volume'].rolling(window=20, min_periods=1).mean()
    group_df['previous_peaks'] = group_df['High'].rolling(window=55,min_periods=1).max()
    group_df['Prev_Close'] = group_df['Close'].shift(1)
    group_df['Ch%'] = (group_df['Close']-group_df['Prev_Close'])/group_df['Prev_Close']*100
    group_df['Close_55'] = group_df['Close'].shift(55)
    group_df['Return_55'] = np.round(group_df['Close']/group_df['Close_55'],2)
    group_df['Vol_20d_Cross'] = np.where(group_df['VOL_SMA_20']<group_df['Volume'],'Yes','No')
    group_df.ta.rsi(length=14, append=True)
    group_df['RSI_High'] = group_df['RSI_14'].rolling(window=20,min_periods=1).max()
    group_df['RSI_Low'] = group_df['RSI_14'].rolling(window=20,min_periods=1).min()

      
    group_df.ta.atr(length=14, append=True)
    group_df.ta.adx(length=14, append=True)
    group_df.ta.cci(length=20, append=True)
    group_df.ta.mfi(length=14, append=True)
    
    # Add Supertrend
    #group_df.ta.supertrend(append=True)
    group_df['supertrend'] = group_df.ta.supertrend(period=10, multiplier=3, append=True)['SUPERT_7_3.0']

    # Add MACD
    #group_df.ta.macd(append=True)
    group_df['MACD'] = ta.macd(group_df['Close'])['MACD_12_26_9']
    group_df['MACD_Signal'] = ta.macd(group_df['Close'])['MACDs_12_26_9']
    group_df['MACD_Histogram'] = ta.macd(group_df['Close'])['MACDh_12_26_9']
    group_df['macd_cross'] = np.where(group_df['MACD']>group_df['MACD_Signal'],'True','False')
      
    
    # Add Bollinger Bands
    group_df.ta.bbands(length=20, append=True)
   
    # Calculate Pivot Points, Resistance, and Support levels manually
    group_df['pivot'] = (group_df['High'] + group_df['Low'] + group_df['Close']) / 3
    group_df['r1'] = 2 * group_df['pivot'] - group_df['Low']
    group_df['r2'] = group_df['pivot'] + group_df['High'] - group_df['Low']
    group_df['r3'] = group_df['High'] + 2 * (group_df['pivot'] - group_df['Low'])
    group_df['s1'] = 2 * group_df['pivot'] - group_df['High']
    group_df['s2'] = group_df['pivot'] - group_df['High'] + group_df['Low']
    group_df['s3'] = group_df['Low'] - 2 * (group_df['High'] - group_df['pivot'])
    
    # Calculate Fibonacci retracement levels manually using group_df
    group_df['fibonacci_38.2'] = group_df['Close'] - 0.382 * (group_df['High'] - group_df['Low'])
    group_df['fibonacci_50.0'] = group_df['Close']
    group_df['fibonacci_61.8'] = group_df['Close'] + 0.618 * (group_df['High'] - group_df['Low'])

    return group_df

# Apply technical analysis using apply function
df_with_ta = grouped_data.apply(apply_ta)

# calculate RS factor
symbol = '^NSEI'

nse_df = yf.download(symbol, begin_time, end_time)
nse_df['symbol'] = symbol
nse_df = nse_df.reset_index()
nse_df['PrevClose'] = np.round(nse_df.groupby('symbol')['Close'].shift(1).reset_index(0,drop=True),2)
nse_df['Close_55'] = np.round(nse_df.groupby('symbol')['Close'].shift(55).reset_index(0,drop=True),2)
nse_df['nifty_return'] = nse_df['Close']/nse_df['Close_55']
nse_df = nse_df.drop_duplicates(subset=['symbol'], keep='last' )
today_close = np.round(nse_df['Close']).values
change = np.round(nse_df['Close'] - nse_df['PrevClose']).values
deno = nse_df['nifty_return'].values
df_with_ta['return'] = np.round(df_with_ta['Close']/df_with_ta['Close_55'],2)
df_with_ta['RS'] = np.round(df_with_ta['return']/deno -1,2)

# keep last rows only

data = df_with_ta.drop_duplicates(subset=['Ticker'], keep='last' )

# drop unnecessory columns

# rename columns
data = data.rename(columns = {'previous_peaks':'High_55','RSI_14':'RSI','ATRr_14':'ATR','ADX_14':'ADX',
                                          'CCI_20_0.015':'CCI','MFI_14':'MFI','BBL_20_2.0':'BB_Low','BBU_20_2.0':'BB_UP',
                                            'BBP_20_2.0':'BB_B%','supertrend':'Supertrend'})

drop_col = ['DMP_14', 'DMN_14','SUPERT_7_3.0', 'SUPERTd_7_3.0', 'SUPERTl_7_3.0', 'SUPERTs_7_3.0','BBM_20_2.0']

data = data.drop(columns = drop_col, axis =1)

data['Date'] = pd.to_datetime(data['Date']).dt.date


final_data = data[(data['RS']>0.1) & (data['Close']>data['SMA_20']) & (data['SMA_20']>data['SMA_50']) &  (data['SMA_50']>data['SMA_200']) &
                  (data['Close']>data['Supertrend']) & (data['macd_cross']=='True')  & (data['MACD_Histogram']>0) & (data['ADX']>20)]


col1, col2 = st.columns(2)

with col1:
    st.markdown('<div style="text-align: center"><span style="font-size: 14px; color: Red;">Overall Bullish Stocks</span></div>', unsafe_allow_html=True)
    st.dataframe(final_data, hide_index=True)

with col2:
    st.markdown('<div style="text-align: center"><span style="font-size: 14px; color: Red;">Oversold Stocks</span></div>', unsafe_allow_html=True)
    buy_low = data[(data['RSI']<35) & (data['Close'] * 1.03 < data['BB_Low']) & (data['BB_B%']<0) & (data['Close'] > data['SMA_200'])][['Date','Ticker','Close','Ch%','High_55','BB_B%','RSI','RSI_High','CCI','MFI','r1', 'r2', 'r3', 's1', 's2', 's3', 'fibonacci_38.2','fibonacci_50.0', 'fibonacci_61.8']]
    st.dataframe(buy_low, hide_index=True)


st.markdown('----------------------------------------------------------------------------------------------------------')

col3, col4 = st.columns(2)

with col3:
     
    st.markdown('<div style="text-align: center"><span style="font-size: 14px; color: Red;">Overall Volume High Stocks</span></div>', unsafe_allow_html=True)
    high_vol = data[(data['Vol_20d_Cross']=='Yes') & (data['Ch%']>1)][['Date','Ticker','Close','Ch%','High_55','BB_B%','RSI','RSI_High','CCI','MFI','r1', 'r2', 'r3', 's1', 's2', 's3', 'fibonacci_38.2','fibonacci_50.0', 'fibonacci_61.8']]
    st.dataframe(high_vol, hide_index=True)

with col4:
    
    st.markdown('<div style="text-align: center"><span style="font-size: 14px; color: Red;">macd_neg_cross</span></div>', unsafe_allow_html=True)
    macd_neg_cross = data[(data['macd_cross']=='True') & (data['MACD_Histogram']>0) & (data['MACD']<0) & (data['MACD_Signal']<0) & (data['MACD'] > data['MACD_Signal']) ][['Date','Ticker','Close','Ch%','High_55','BB_B%','RSI','RSI_High','CCI','MFI','r1', 'r2', 'r3', 's1', 's2', 's3', 'fibonacci_38.2','fibonacci_50.0', 'fibonacci_61.8']]
    st.dataframe(macd_neg_cross, hide_index=True)

st.markdown('----------------------------------------------------------------------------------------------------------')

col5, col6 = st.columns(2)

with col5:
    
    st.markdown('<div style="text-align: center"><span style="font-size: 14px; color: Red;">High Vol Near SMA50</span></div>', unsafe_allow_html=True)
    vol_sma50 = data[(data['Ch%']>1) & (data['Close'] > 1.01 * data['SMA_50']) & (data['Vol_20d_Cross']=='Yes')][['Date','Ticker','Close','Ch%','High_55','BB_B%','RSI','RSI_High','CCI','MFI','r1', 'r2', 'r3', 's1', 's2', 's3', 'fibonacci_38.2','fibonacci_50.0', 'fibonacci_61.8']]
    st.dataframe(vol_sma50, hide_index=True)

with col6:

    st.markdown('<div style="text-align: center"><span style="font-size: 14px; color: Red;">Volume Breakout With Bullish Pattern</span></div>', unsafe_allow_html=True)
    vol_break = final_data[(final_data['Vol_20d_Cross']=='Yes') & (final_data['Ch%']>1) & (final_data['RSI']>55) & (final_data['BB_B%'] <0.9) & (final_data['CCI']>80)][['Date','Ticker','Close','Ch%','High_55','BB_B%','RSI','RSI_High','CCI','MFI','r1', 'r2', 'r3', 's1', 's2', 's3', 'fibonacci_38.2','fibonacci_50.0', 'fibonacci_61.8']]
    st.dataframe(vol_break, hide_index=True)


if 'data' not in st.session_state:
    # Save the data to session state
    st.session_state.data = data

