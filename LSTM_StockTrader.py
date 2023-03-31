# Jaime Cortez, Capstone for Spring 2023
# Adapted from https://medium.com/analytics-vidhya/keras-tensorflow-to-predict-market-movements-and-backtest-using-backtrader-d51b0b3e9070

### How it works ###
# Gather Data from any ticker (I chose SPY) then select the date range you would like to train on
# Select the test size percentage (default = .5)
# Create a model. For this example, it is sequential (LSTM)
# Optional: Add/Remove indicators, change % of cash to use each trade, change commission
# Run program

### Other notes ###
# Ctrl-alt-S for python interpreter stuff
# use "noinspection" to disable warnings in certain places
# talib (technical analysis library) requires some advanced setup, use Google to guide you.

# TODO Figure out what should be modified each loop. Also figure out how to do it.
# ideas so far:
# LSTM epochs, number and type of layers, number of cells in each layer, batch size.
# Data timeframe, resolution, indicators
# Stock/ETF selection

### Library edits ###
# Zipline warning: commented out the warning part of the function
# plot error(drawdown): need to look up which line to change in which library

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0-3: info, warning, and error messages get printed (0 = all, 3 = none)
import contextlib
import warnings
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
# noinspection PyUnresolvedReferences
import yfinance as yf
import datetime
import pyfolio as pf
from pyfolio.timeseries import perf_stats
import backtrader as bt
from backtrader.feeds import PandasData
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.layers import Dense  # , Dropout
from keras.models import Sequential
from talib import RSI, BBANDS, MACD

# ignore warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

##### START: USER-EDITABLE THINGS #####
# You can also add indicators in the technical indicators section further down.
plotGraphs = False          # if true, plot all the graphs
modelVerboseness = 0        # verbose = 0 means no progress bars (for epochs), 1 = show bars, 2 = show epoch number only.
printModelSummary = False    # if true, displays a model summary consisting of layers, cells, type, etc.
printReturnsVolatilityTrades = False   # if true, display returns and volatility values, as well as number of trades
printRunOutput = False       # if true, displays the buys and sells of the model during runtime.
printPortfolioStartEnd = False  # if true, prints the starting and ending values and dates of the portfolio

# ticker and the start and end dates for testing, as well as other values
ticker = 'SPY'  # The ticker we want to use
start = datetime.datetime(2013, 1, 1)   # year, month, day
end = datetime.datetime(2023, 1, 1)
test_size_percentage = .6  # The bot will be trained on the first x% of the data, and tested on the next (1-x)% of the data

# Define the model to use for our machine learning trading model
def create_model():
    my_seed = 100
    np.random.seed(my_seed)
    tf.random.set_seed(my_seed)

    # Modify the model's layers here
    nmodel = Sequential()

    nmodel.add(Dense(10, activation='relu', input_dim=len(cols)))
    nmodel.add(Dense(1, activation='sigmoid'))

    nmodel.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])  # 'rms prop',
    return nmodel
##### END: USER-EDITABLE THINGS #####

# class to define the columns we will provide
class SignalData(PandasData):
    """
    Define pandas DataFrame structure
    """
    OHLCV = ['open', 'high', 'low', 'close', 'volume']
    cols = OHLCV + ['predicted']

    # create lines
    lines = tuple(cols)

    # define parameters
    params = {c: -1 for c in cols}
    params.update({'datetime': None})
    params = tuple(params.items())

# define backtesting strategy class
class MLStrategy(bt.Strategy):
    params = dict(
    )

    def __init__(self):
        # keep track of open, close prices and predicted value in the series
        self.data_predicted = self.datas[0].predicted
        self.data_open = self.datas[0].open
        self.data_close = self.datas[0].close

        # keep track of pending orders/buy price/buy commission
        self.order = None
        self.price = None
        self.comm = None

    # logging function
    def log(self, txt):
        dt = self.datas[0].datetime.date(0).isoformat()
        print(f'{dt}, {txt}')

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # order already submitted/accepted - no action required
            return

        # report executed order
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f'BUY EXECUTED --- Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f},Commission: {order.executed.comm:.2f}'
                )
                self.price = order.executed.price
                self.comm = order.executed.comm
            else:
                self.log(
                    f'SELL EXECUTED --- Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f},Commission: {order.executed.comm:.2f}'
                )

        # report failed order
        elif order.status in [order.Canceled, order.Margin,
                              order.Rejected]:
            # self.log('Order Failed')
            match order.status:
                case order.Canceled:
                    self.log('Order canceled')
                case order.Margin:
                    self.log('Order invalid due to Margin')
                case order.Rejected:
                    self.log('Order rejected')

        # set no pending order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(f'OPERATION RESULT --- Gross: {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f}')

    # We have set cheat_on_open = True.This means that we calculated the signals on day t's close price,
    # but calculated the number of shares we wanted to buy based on day t+1's open price.
    def next_open(self):
        if not self.position:
            if self.data_predicted > 0:
                # calculate the max number of shares ('all-in')
                size = int(self.broker.getcash() * 0.98 / self.datas[0].open)
                # buy order
                #     self.log(f'BUY CREATED --- Size: {size}, Cash: {self.broker.getcash():.2f}, Open: {self.data_open[0]}, Close: {self.data_close[0]}')
                self.buy(size=size)
        else:
            if self.data_predicted < 0:
                # sell order
                #      self.log(f'SELL CREATED --- Size: {self.position.size}')
                self.sell(size=self.position.size)


##### DATA GATHERING #####
# read in csv file with all data including open, high, low, etc.
stock_all = pd.read_csv("stock_data/" + ticker + ".csv", index_col=0)
stock_all.index = pd.to_datetime(stock_all.index)   # convert index to DateTimeIndex

# isolate adjusted close data (adjusted close accounts for dividends and splits)
stock_ac = stock_all['Adj Close']
stock_ac = stock_ac.squeeze()

### This code below shows our data is in the right/wrong format (converting from csv changes metadata like index type)
# download data from yahoo finance
#stock_yf = yf.download(ticker, progress=False, actions=True, start=start, end=end)
#stock_yf.to_csv("stock_data/SPY.csv")       # this saves yf data

# print and compare information for inconsistencies. Information should match perfectly for program to work
#stock_ac.info()
#stockyf.info()

# converting series -> dataframe to add additional columns
stock = pd.DataFrame(stock_ac)
stock.rename(columns={'Adj Close': 'close'}, inplace=True)

# calculate daily log returns and market direction
stock['returns'] = np.log(stock / stock.shift(1))
stock.dropna(inplace=True)
stock['direction'] = np.sign(stock['returns']).astype(int)

# define the number of lags
lags = [1, 2, 3, 4, 5]
# compute lagged log returns
cols = []
for lag in lags:
    col = f'rtn_lag{lag}'
    stock[col] = stock['returns'].shift(lag)
    cols.append(col)

##### TECHNICAL INDICATORS #####
# Compute and append RSI (Relative Strength Index)
stock['rsi'] = RSI(stock.close)
cols.append('rsi')

# Compute and append Bollinger Bands
high, mid, low = BBANDS(stock.close, timeperiod=20)
stock = stock.join(pd.DataFrame({'bb_high': high, 'bb_low': low}, index=stock.index))
cols.append('bb_high')
cols.append('bb_low')

# Compute and append Moving Average Convergence/Divergence
stock['macd'] = MACD(stock.close)[0]
cols.append('macd')

##### TRAINING #####
# split the dataset in training and test datasets
train, test = train_test_split(stock.dropna(), test_size=test_size_percentage, shuffle=False)

# sort the data on date index
train = train.copy().sort_index()
test = test.copy().sort_index()

# normalize the training dataset
mu, std = train.mean(), train.std()
train_ = (train - mu) / mu.std()
# create the model
model = create_model()
# map market direction of (1,-1) to (1,0)
train['direction_'] = np.where(train['direction'] > 0, 1, 0)

# fit the model for training dataset
r = model.fit(train_[cols], train['direction_'], batch_size=16, epochs=3, verbose=modelVerboseness)
if printModelSummary:
    model.summary()

##### TESTING #####
# normalize the test dataset
# noinspection PyRedeclaration
mu, std = test.mean(), test.std()
test_ = (test - mu) / std
# map market direction of (1,-1) to (1,0)
test['direction_'] = np.where(test['direction'] > 0, 1, 0)
# evaluate the model with test dataset
model.evaluate(test_[cols], test['direction_'], verbose=modelVerboseness)

# predict the direction and map it (1,0)
pred = np.where(model.predict(test_[cols], verbose=modelVerboseness) > 0.5, 1, 0)
pred[:10].flatten()

# based on prediction calculate the position for strategy
test['position_strategy'] = np.where(pred > 0, 1, -1)
# calculate daily returns for the strategy
test['strategy_return'] = test['position_strategy'] * test['returns']

# calculate total return and std. deviation of each strategy
if printReturnsVolatilityTrades:
    print('\nTotal Returns:')
    print(test[['returns', 'strategy_return']].sum().apply(np.exp))
    print('\nAnnual Volatility:')
    print(test[['returns', 'strategy_return']].std() * 252 ** 0.5)

    # number of trades over time for the strategy
    print('Number of trades = ', (test['position_strategy'].diff() != 0).sum())

# backtesting start and end dates
if printPortfolioStartEnd:
    start_port = test.index[0]
    end_port = test.index[-1]
    print(start_port)
    print(end_port)

# fetch the daily pricing data
prices = stock_all

##### BACKTRADER #####
# rename the columns as needed for Backtrader
prices.drop(['Close', 'Dividends', 'Stock Splits'], inplace=True, axis=1)
prices.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Adj Close': 'close', 'Volume': 'volume',
                       }, inplace=True)

# add the predicted column to prices dataframe. This will be used as signal for buy or sell
predictions = test.strategy_return
predictions = pd.DataFrame(predictions)
predictions.rename(columns={'strategy_return': 'predicted'}, inplace=True)
prices = predictions.join(prices, how='right').dropna()
prices[['predicted']].sum().apply(np.exp)

# instantiate SignalData class
# noinspection PyArgumentList
data = SignalData(dataname=prices)
# instantiate Cerebro, add strategy, data, initial cash, commission and pyfolio for performance analysis
# noinspection PyArgumentList
cerebro = bt.Cerebro(cheat_on_open=True)
cerebro.addstrategy(MLStrategy)
cerebro.adddata(data, name=ticker)
cerebro.broker.setcash(100000.0)
cerebro.broker.setcommission(commission=0.001)
cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')

# run the backtest
if printPortfolioStartEnd:
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

if not printRunOutput:
    with contextlib.redirect_stdout(None):
        backtest_result = cerebro.run()
else:
    backtest_result = cerebro.run()

if printPortfolioStartEnd:
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

# Extract inputs for pyfolio
strat = backtest_result[0]
pyfoliozer = strat.analyzers.getbyname('pyfolio')
returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
returns.name = 'Strategy'

# get benchmark returns
benchmark_rets = stock['returns']
benchmark_rets.index = benchmark_rets.index.tz_localize('UTC')
benchmark_rets = benchmark_rets.filter(returns.index)
benchmark_rets.name = f'{ticker}'

# get/print performance statistics for strategy # TODO save sharpe ratio into a file
myStatsSeries = perf_stats(returns=returns)
mySharpe = myStatsSeries[3]
print('Sharpe Ratio = {}'.format(mySharpe))
# print('test' + mySharpe)
# with open('output/some_output.txt', 'w') as sys.stdout:
#    print(perf_stats(returns=returns).)

# visualize everything (ticker stock chart, cumulative returns, and a combined set of graphs (drawdown, rolling sharpe, etc.)
if plotGraphs:
    fig, ax = plt.subplots(2, 1, sharex='all', figsize=(8, 4))  # True is the same as 'all'
    ax[0].plot(stock.close, label=f'{ticker} Adj Close')
    ax[0].set(title=f'{ticker} Closing Price', ylabel='Price')
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(stock['returns'], label='Daily Returns')
    ax[1].set(title=f'{ticker} Daily Returns', ylabel='Returns')
    ax[1].grid(True)
    plt.legend()
    plt.tight_layout()
    plt.draw()

    # plot cumulative returns
    # noinspection PyRedeclaration
    fig, ax = plt.subplots(1, 1, sharex='all', figsize=(8, 4))
    ax.plot(test.returns.cumsum().apply(np.exp), linestyle='dashed', label=f'{ticker} Buy and Hold')
    ax.plot(test.strategy_return.cumsum().apply(np.exp), linestyle='dotted', label='Strategy')
    ax.set(title=f'{ticker} Buy and Hold vs. Strategy', ylabel='Cumulative Returns')
    ax.grid(True)
    ax.legend()
    plt.draw()

    # plot performance for strategy vs benchmark
    # noinspection PyRedeclaration
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 7), constrained_layout=True)
    axes = ax.flatten()
    pf.plot_drawdown_periods(returns=returns, ax=axes[0])
    axes[0].grid(True)
    pf.plot_rolling_returns(returns=returns,
                            factor_returns=benchmark_rets,
                            ax=axes[1], title=f'Strategy vs {ticker}')
    axes[1].grid(True)
    pf.plot_drawdown_underwater(returns=returns, ax=axes[2])
    axes[2].grid(True)
    pf.plot_rolling_sharpe(returns=returns, ax=axes[3])
    axes[3].grid(True)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show(block=True)    # we use show here to block the program from ending
# If we loop, everything up to the end of the file should be looped.
