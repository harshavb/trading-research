import pandas as pd
import yfinance as yf
import numpy as np
from darts import TimeSeries
from darts.metrics import mape
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam


def get_historical_data(symbol, start_date, end_date):
    data = yf.download(symbol, start=start_date, end=end_date)
    return data

def preprocess_data(data):
    data = data[['Close']]
    data.index = pd.to_datetime(data.index)
    return data

def scale_data(train_data, test_data):
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    return train_scaled, test_scaled, scaler

def prepare_sequences(data, window_size):
    x, y = [], []

    for i in range(window_size, len(data)):
        x.append(data[i - window_size:i])
        y.append(data[i])

    x, y = np.array(x), np.array(y)

    # Add the following print statements
    print("data length:", len(data))
    print("window_size:", window_size)
    print("x length:", len(x))
    print("y length:", len(y))

    return x, y


def trading_algorithm(data):
    window_size = 60
    train_data_ratio = 0.6  # Use 80% of the data for training
    test_data_ratio = 1 - train_data_ratio  # Use the remaining 20% for testing

    train_data = data[:int(len(data) * train_data_ratio)]
    test_data = data[int(len(data) * train_data_ratio):]

    # The rest of the function remains the same
    train_scaled, test_scaled, scaler = scale_data(train_data, test_data)

    x_train, y_train = prepare_sequences(train_scaled, window_size)
    x_test, y_test = prepare_sequences(test_scaled, window_size)

    model = Sequential()
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(window_size, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    model.fit(x_train, y_train, batch_size=32, epochs=200, verbose=0)

    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)  # Add this line to reshape x_test
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    buy_signals = []
    sell_signals = []
    for i in range(1, len(predictions)):
        if predictions[i] > predictions[i - 1]:
            buy_signals.append(test_data.index[i + window_size])
        else:
            sell_signals.append(test_data.index[i + window_size])

    return buy_signals, sell_signals

def calculate_profit(buy_signals, sell_signals, data, initial_capital):
    capital = initial_capital
    shares = 0

    transactions = list(zip(buy_signals, sell_signals))
    for buy_date, sell_date in transactions:
        buy_price = data.loc[buy_date]['Close']
        sell_price = data.loc[sell_date]['Close']

        shares_to_buy = capital // buy_price
        shares += shares_to_buy
        capital -= shares_to_buy * buy_price

        capital += shares * sell_price
        shares = 0

    profit = capital - initial_capital
    percentage_return = (profit / initial_capital) * 100

    return percentage_return

def main():
    symbol = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2022-1-31'

    data = get_historical_data(symbol, start_date, end_date)
    preprocessed_data = preprocess_data(data)
    buy_signals, sell_signals = trading_algorithm(preprocessed_data)

    initial_capital = 10000  # Set an initial capital amount in USD
    percentage_return = calculate_profit(buy_signals, sell_signals, preprocessed_data, initial_capital)
    print("Percentage return: {:.2f}%".format(percentage_return))



    
    # Visualize the buy and sell signals on a plot
    plt.figure(figsize=(16, 8))
    plt.plot(preprocessed_data['Close'], label='Close Price', alpha=0.35)
    plt.scatter(buy_signals, preprocessed_data.loc[buy_signals].Close, label='Buy Signal', marker='^', color='green')
    plt.scatter(sell_signals, preprocessed_data.loc[sell_signals].Close, label='Sell Signal', marker='v', color='red')
    plt.title('Stock Price with Buy and Sell Signals')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend(loc='upper left')
    plt.show()

if __name__ == "__main__":
    main()
