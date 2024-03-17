import streamlit as st
import pandas as pd
import yfinance as yf
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
import pandas_ta as ta
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from functools import reduce
import matplotlib.pyplot as plt
import plotly.graph_objects as go



st.title('Stock Price Predictions App')
#st.sidebar.info('Welcome to the Stock Price Prediction App. Choose your options below')
#st.sidebar.info("Created and designed by Shiv")

# Retrieve the data from session state
comb_data = st.session_state.data


def main():
    option = st.sidebar.selectbox('Make a choice', ['Visualize','Recent Data', 'Predict'])
    if option == 'Visualize':
        tech_indicators()
    elif option == 'Recent Data':
        dataframe()
    else:
        predict()


@st.cache_data
def download_data(op, start_date, end_date):
    df = yf.download(op, start=start_date, end=end_date, progress=False)
    return df

# Function to identify and visualize RSI divergence
def visualize_rsi_divergence(data, rsi_period=14):

    # Calculate RSI
    data['RSI'] = ta.rsi(data['Close'], length=rsi_period)

    # Identify Bullish Divergence
    bullish_divergence = (data['Close'] < data['Close'].shift(1)) & (data['RSI'] > data['RSI'].shift(1))
    print(bullish_divergence)
    
    # Identify Bearish Divergence
    bearish_divergence = (data['Close'] > data['Close'].shift(1)) & (data['RSI'] < data['RSI'].shift(1))

    # Plot the RSI and Divergence
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Close Price', color=color)
    ax1.plot(data.index, data['Close'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('RSI', color=color)
    ax2.plot(data.index, data['RSI'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Highlight Bullish Divergence
    ax1.plot(data.index[bullish_divergence], data['Close'][bullish_divergence], '^', markersize=10, color='g', label='Bullish Divergence')

    # Highlight Bearish Divergence
    ax1.plot(data.index[bearish_divergence], data['Close'][bearish_divergence], 'v', markersize=10, color='r', label='Bearish Divergence')

    plt.title('RSI Divergence')
    plt.legend()
    st.pyplot(fig)

df = pd.read_csv("symbol.csv")
tickers = df['Ticker'].values
stocks = st.sidebar.selectbox('Enter a Stock Symbol', tickers)
stocks = stocks.upper()
today = datetime.date.today() + datetime.timedelta(days=1)
duration = st.sidebar.number_input('Enter the duration', value=252)
before = today - datetime.timedelta(days=duration)
start_date = st.sidebar.date_input('Start Date', value=before)
end_date = st.sidebar.date_input('End date', today)
if st.sidebar.button('Send'):
    if start_date < end_date:
        st.sidebar.success('Start date: `%s`\n\nEnd date: `%s`' %(start_date, end_date))
        download_data(stocks, start_date, end_date)
    else:
        st.sidebar.error('Error: End date must fall after start date')


data = download_data(stocks, start_date, end_date)
scaler = StandardScaler()

def tech_indicators():
    st.header('Technical Indicators')
    option = st.radio('Choose a Technical Indicator to Visualize', ['Close', 'BB', 'MACD', 'RSI','RSI_Divergence','Volume','Overall'], horizontal=True)

    # Bollinger bands
    bb_indicator = BollingerBands(data.Close)
    bb = data
    bb['bb_h'] = bb_indicator.bollinger_hband()
    bb['bb_l'] = bb_indicator.bollinger_lband()
    bb['bb_%'] = bb_indicator.bollinger_pband() 
    # Creating a new dataframe
    bb = bb[['Close', 'bb_h', 'bb_l','bb_%']]

    # MACD
    data['macd'] = MACD(data.Close).macd()

    data['macd_signal'] = MACD(data.Close).macd_signal()

    data['macd_hist'] = MACD(data.Close).macd_diff()
    # RSI
    rsi = RSIIndicator(data.Close, window=14).rsi()

    data['RSI'] = ta.rsi(data['Close'], length=14)
    
    # SMA_20
    sma_20 = SMAIndicator(data.Close, window=20).sma_indicator()
    data['sma_20'] = sma_20

    # SMA_50
    sma_50 = SMAIndicator(data.Close, window=50).sma_indicator()
    data['sma_50'] = sma_50

    #SMA_200
    sma_200 = SMAIndicator(data.Close, window=200).sma_indicator()
    data['sma_200'] = sma_200

   
    if option == 'Close':

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Close', line=dict(color='blue')))
        fig1.add_trace(go.Scatter(x=data.index, y=data['sma_20'], name='SMA_20', line=dict(color='green')))
        fig1.add_trace(go.Scatter(x=data.index, y=data['sma_50'], name='SMA_50', line=dict(color='red')))
        fig1.add_trace(go.Scatter(x=data.index, y=data['sma_200'], name='SMA_200', line=dict(color='purple')))
        #fig1.update_layout(title='Price and Moving Averages')
        fig1.update_layout(title='Price and Moving Averages', width=800, height=500)

        # Create a figure for the second plot
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='red')))
        fig2.update_layout(title='Relative Strength Index (RSI)')

        # Show the plots using Streamlit
        st.plotly_chart(fig1)
        st.plotly_chart(fig2)   
        
        
    elif option == 'Volume':
        data['Prev_Close'] = data['Close'].shift(1)

        data['Ch%'] = (data['Close']-data['Prev_Close'])/data['Prev_Close']*100

        # Define color based on 'Close' values
        colors = ['blue' if change>0 else 'red' for change in data['Ch%']]

        # Create a figure for the plot
        fig = go.Figure()

        # Add trace for 'Volume' as a bar chart with color based on 'Close'
        fig.add_trace(go.Bar(x=data.index, y=data['Volume'], marker_color=colors))

        # Update layout
        fig.update_layout(title='Volume with Color Coded Close', xaxis_title='Date', yaxis_title='Volume')

        # Show the plot using Streamlit
        st.plotly_chart(fig)
          

    elif option == 'BB':
        st.write(bb.tail(1))
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=bb.index, y=bb['Close'], name='macd', line=dict(color='blue')))
        fig1.add_trace(go.Scatter(x=bb.index, y=bb['bb_h'], name='bb_h', line=dict(color='green')))
        fig1.add_trace(go.Scatter(x=bb.index, y=bb['bb_l'], name='bb_l', line=dict(color='red')))
        fig1.update_layout(title='Bollinger Bands', width=800, height=500)
        # Create a figure for the second plot
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=bb.index, y=bb['bb_%'], name='bb_%', line=dict(color='blue')))
        fig2.add_trace(go.Scatter(x=bb.index, y=[0] * len(bb), mode='lines', name='Zero Line', line=dict(color='red', dash='dash')))
        fig2.update_layout(title='Bollinger Bands Percentage with Zero Line', width=800, height=500)
        st.plotly_chart(fig1)
        st.plotly_chart(fig2) 

        #st.line_chart(bb)
    elif option == 'MACD':
        #st.write(macd.tail(1))
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=data.index, y=data['macd'], name='macd', line=dict(color='blue')))
        fig1.add_trace(go.Scatter(x=data.index, y=data['macd_signal'], name='macd_signal', line=dict(color='red')))
        fig1.update_layout(title='MACD Chart', width=800, height=500)
        # Create a figure for the second plot
        fig2 = go.Figure()
        # Define color based on 'Close' values
        colors = ['green' if change>0 else 'red' for change in data['macd_hist']]

        fig2.add_trace(go.Bar(x=data.index, y=data['macd_hist'], name='macd_hist', marker_color = colors))
        #fig.add_trace(go.Bar(x=data.index, y=data['Volume'], marker_color=colors))
        fig2.update_layout(title='macd_hist')

        # Show the plots using Streamlit
        st.plotly_chart(fig1)
        st.plotly_chart(fig2) 
      

        #st.line_chart(macd)
    elif option == 'RSI':
        st.write('Relative Strength Indicator')
        st.write(rsi.tail(1))
        st.line_chart(rsi)
    
    elif option == 'RSI_Divergence':
        dataset = download_data(stocks, start_date, end_date)
        visualize_rsi_divergence(dataset)
        
    else:       
        st.write('Overall Summary')
 
      
        # CREATE BOX DATA
    
        cum_data = comb_data[comb_data['Ticker']==stocks]
        st.write(stocks)
        
        rsi = round(cum_data['RSI'].iloc[-1],2)
        rsi_high = round(cum_data['RSI_High'].iloc[-1],2)
        rsi_low = round(cum_data['RSI_Low'].iloc[-1],2)
        close = round(cum_data['Close'].iloc[-1],2)
        ch = round(cum_data['Ch%'].iloc[-1],2)
        SMA_20 = round(cum_data['SMA_20'].iloc[-1],2)
        SMA_50 = round(cum_data['SMA_50'].iloc[-1],2)
        SMA_200 = round(cum_data['SMA_200'].iloc[-1],2)
        bb = round(cum_data['BB_B%'].iloc[-1],2)
        r1 = round(cum_data['r1'].iloc[-1],2)
        r2 = round(cum_data['r2'].iloc[-1],2)
        r3 = round(cum_data['r3'].iloc[-1],2)
        s1 = round(cum_data['s1'].iloc[-1],2)
        s2 = round(cum_data['s2'].iloc[-1],2)
        s3 = round(cum_data['s3'].iloc[-1],2)
        fib1 = round(cum_data['fibonacci_38.2'].iloc[-1],2)
        fib2 = round(cum_data['fibonacci_50.0'].iloc[-1],2)
        fib3 = round(cum_data['fibonacci_61.8'].iloc[-1],2)
        supertrend = round(cum_data['Supertrend'].iloc[-1],2)
        cci = round(cum_data['CCI'].iloc[-1],2)
        vol = cum_data['Vol_20d_Cross'].iloc[-1]
        macd = cum_data['macd_cross'].iloc[-1]
        macd_hist = round(cum_data['MACD_Histogram'].iloc[-1],2)
        adx = round(cum_data['ADX'].iloc[-1],2)

        # Data or variables for the boxes
        box_data = [
            {'title': 'Close', 'content': close},
            {'title': 'Ch%', 'content': ch},
            {'title': 'SMA_20', 'content': SMA_20},
            {'title': 'SMA_50', 'content': SMA_50},
            {'title': 'SMA_200', 'content': SMA_200},
            {'title': 'RSI', 'content': rsi},
            {'title': 'RSI_High', 'content': rsi_high},
            {'title': 'RSI_Low', 'content': rsi_low},
            {'title': 'ADX', 'content': adx},
            {'title': 'Supertrend', 'content': supertrend},
            {'title': 'BB_B%', 'content': bb},
            {'title': 'CCI', 'content': cci},
            {'title': 'Vol_20d_Cross', 'content': vol},
            {'title': 'macd_cross', 'content': macd},
            {'title': 'MACD_Hist', 'content': macd_hist},
            {'title': 'R1', 'content': r1},
            {'title': 'R2', 'content': r2},
            {'title': 'R3', 'content': r3},
            {'title': 'S1', 'content': s1},
            {'title': 'S2', 'content': s2},
            {'title': 'S3', 'content': s3},
            {'title': 'Fib1', 'content': fib1},
            {'title': 'Fib2', 'content': fib2},
            {'title': 'Fib3', 'content': fib3}
            
        ]

        # Define the layout
        num_cols = 8

        # Create the boxes
        for i in range(0, len(box_data), num_cols):
            cols = st.columns(num_cols)
            for j, col in enumerate(cols):
                if i + j < len(box_data):
                    with col:
                        st.markdown(
                            f"""
                            <div style="width: 100px; height: 100px; border: 1px solid black; padding: 10px;">
                                <h3 style="font-size: 14px; color: black;">{box_data[i+j]['title']}</h3>
                                <p style="font-size: 12px; color: blue;">{box_data[i+j]['content']}</p>
                                
                            </div>
                            """,
                            unsafe_allow_html=True
                        )


def dataframe():
    st.header('Recent Data')
    st.dataframe(data.tail(10))



def predict():
    model = st.radio('Choose a model', ['LinearRegression', 'RandomForestRegressor', 'ExtraTreesRegressor', 'KNeighborsRegressor', 'XGBoostRegressor'])
    num = st.number_input('How many days forecast?', value=5)
    num = int(num)
    if st.button('Predict'):
        if model == 'LinearRegression':
            engine = LinearRegression()
            model_engine(engine, num)
        elif model == 'RandomForestRegressor':
            engine = RandomForestRegressor()
            model_engine(engine, num)
        elif model == 'ExtraTreesRegressor':
            engine = ExtraTreesRegressor()
            model_engine(engine, num)
        elif model == 'KNeighborsRegressor':
            engine = KNeighborsRegressor()
            model_engine(engine, num)
        else:
            engine = XGBRegressor()
            model_engine(engine, num)


def model_engine(model, num):
    # getting only the closing price
    df = data[['Close']]
    # shifting the closing price based on number of days forecast
    df['preds'] = data.Close.shift(-num)
    # scaling the data
    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)
    # storing the last num_days data
    x_forecast = x[-num:]
    # selecting the required values for training
    x = x[:-num]
    # getting the preds column
    y = df.preds.values
    # selecting the required values for training
    y = y[:-num]

    #spliting the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=7)
    # training the model
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    st.text(f'r2_score: {r2_score(y_test, preds)} \
            \nMAE: {mean_absolute_error(y_test, preds)}')
    # predicting stock price based on the number of days
    forecast_pred = model.predict(x_forecast)
    day = 1
    for i in forecast_pred:
        st.text(f'Day {day}: {i}')
        day += 1


if __name__ == '__main__':
    main()