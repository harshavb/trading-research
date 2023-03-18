# Adapted from https://medium.com/analytics-vidhya/keras-tensorflow-to-predict-market-movements-and-backtest-using-backtrader-d51b0b3e9070
### How it works ###
# Gather Data from any ticker (I chose SPY) then select the date range you would like to train on
# Select the test size percentage (default = .5)
# create a model (ctrl+f "create_model"). For this example, it is sequential (LSTM). Add/remove layers, change # of cells, etc.
# Optional: Add/Remove indicators, change % of cash to use each trade
# Run program
###
# Ctrl-alt-S for python interpreter stuff

### KERAS AND TENSORFLOW
# make the necessary imports
import numpy as np
#import pytalib.indicators.base
from matplotlib import pyplot as plt
import pandas as pd
#import seaborn as sns
import yfinance as yf
#import warnings
import datetime
import pyfolio as pf
import backtrader as bt
from backtrader.feeds import PandasData
import warnings
from sklearn.model_selection import train_test_split
import tensorflow as tf
#from keras import layers
from keras.layers import Dense  # , Dropout
from keras.models import Sequential
from talib import RSI, BBANDS, MACD

# ignore warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

### DATA GATHERING
# ticker and the start and end dates for testing, as well as other values
ticker = 'SPY'  # The ticker we want to use
start = datetime.datetime(2016, 1, 1)
end = datetime.datetime(2023, 1, 1)
test_size_percentage = .8   # The bot will be trained on the first x of the data, and tested on the next (1-x) of the data

# download ticker stock price from yahoo finance
stock = yf.download(ticker, progress=True, actions=True, start=start, end=end)['Adj Close']
stock = pd.DataFrame(stock)
stock.rename(columns={'Adj Close': 'close'}, inplace=True)
stock.head(2)

# calculate daily log returns and market direction
stock['returns'] = np.log(stock / stock.shift(1))
stock.dropna(inplace=True)
stock['direction'] = np.sign(stock['returns']).astype(int)
stock.head(3)

# visualize the closing price and daily returns
fig, ax = plt.subplots(2, 1, sharex='all', figsize=(8, 4))     # True is the same as 'all'
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
# plt.savefig('images/chart1', dpi=300)

# define the number of lags
lags = [1, 2, 3, 4, 5]
# compute lagged log returns
cols = []
for lag in lags:
    col = f'rtn_lag{lag}'
    stock[col] = stock['returns'].shift(lag)
    cols.append(col)
stock.head(2)

# RSI - Relative Strength Index
stock['rsi'] = RSI(stock.close)

# append to feature columns list
cols.append('rsi')
stock.tail(2)

# Compute Bollinger Bands
high, mid, low = BBANDS(stock.close, timeperiod=20)
stock = stock.join(pd.DataFrame({'bb_high': high, 'bb_low': low}, index=stock.index))

# append to feature columns list
cols.append('bb_high')
cols.append('bb_low')
# Compute Moving Average Convergence/Divergence
stock['macd'] = MACD(stock.close)[0]

# append to feature columns list
cols.append('macd')
# # let's look at the head and tail of our dataframe
# stock.head().append(stock.tail())

# len(cols)
# split the dataset in training and test datasets
train, test = train_test_split(stock.dropna(), test_size=test_size_percentage, shuffle=False)

# sort the data on date index
train = train.copy().sort_index()
test = test.copy().sort_index()


# # view train dataset
# train.tail()
# # view test dataset
# test.tail()

# define a function to create the deep neural network model
def create_model():
    my_seed = 100
    np.random.seed(my_seed)
    tf.random.set_seed(my_seed)

    # Modify the model's layers here
    nmodel = Sequential()

    nmodel.add(Dense(100, activation='relu', input_dim=len(cols)))
    nmodel.add(Dense(1, activation='sigmoid'))

    nmodel.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])    # 'rms prop',
    return nmodel


# normalized the training dataset
mu, std = train.mean(), train.std()
train_ = (train - mu) / mu.std()
# create the model
model = create_model()
# map market direction of (1,-1) to (1,0)
train['direction_'] = np.where(train['direction'] > 0, 1, 0)
# cols
# train_.head()
# %%time
# fit the model for training dataset
r = model.fit(train_[cols], train['direction_'], batch_size=16, epochs=3, verbose=True)
model.summary()

# normalized the test dataset
mu, std = test.mean(), test.std()
test_ = (test - mu) / std
# map market direction of (1,-1) to (1,0)
test['direction_'] = np.where(test['direction'] > 0, 1, 0)
# evaluate the model with test dataset
model.evaluate(test_[cols], test['direction_'])

# predict the direction and map it (1,0)
pred = np.where(model.predict(test_[cols]) > 0.5, 1, 0)
pred[:10].flatten()

# based on prediction calculate the position for strategy
test['position_strategy'] = np.where(pred > 0, 1, -1)
# calculate daily returns for the strategy
test['strategy_return'] = test['position_strategy'] * test['returns']
# test.head()
# calculate total return and std. deviation of each strategy
print('\nTotal Returns:')
print(test[['returns', 'strategy_return']].sum().apply(np.exp))
print('\nAnnual Volatility:')
print(test[['returns', 'strategy_return']].std() * 252 ** 0.5)

# number of trades over time for the strategy
print('Number of trades = ', (test['position_strategy'].diff() != 0).sum())

# plot cumulative returns
fig, ax = plt.subplots(1, 1, sharex='all', figsize=(8, 4))
ax.plot(test.returns.cumsum().apply(np.exp), linestyle='dashed', label=f'{ticker} Buy and Hold')
ax.plot(test.strategy_return.cumsum().apply(np.exp), linestyle='dotted', label='Strategy')
ax.set(title=f'{ticker} Buy and Hold vs. Strategy', ylabel='Cumulative Returns')
ax.grid(True)
ax.legend()

plt.draw()
# plt.savefig('images/chart2')

# backtesting start and end dates
start = test.index[0]
end = test.index[-1]
print(start)
print(end)

# fetch the daily pricing data from yahoo finance
prices = yf.download(ticker, progress=True, actions=True, start=start, end=end)
prices.head(2)

# rename the columns as needed for Backtrader
prices.drop(['Close', 'Dividends', 'Stock Splits'], inplace=True, axis=1)
prices.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Adj Close': 'close', 'Volume': 'volume',
                       }, inplace=True)
prices.head(3)

# add the predicted column to prices dataframe. This will be used as signal for buy or sell
predictions = test.strategy_return
predictions = pd.DataFrame(predictions)
predictions.rename(columns={'strategy_return': 'predicted'}, inplace=True)
prices = predictions.join(prices, how='right').dropna()
prices.head(2)

prices[['predicted']].sum().apply(np.exp)

OHLCV = ['open', 'high', 'low', 'close', 'volume']


# class to define the columns we will provide
class SignalData(PandasData):
    """
    Define pandas DataFrame structure
    """
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
            #self.log('Order Failed')
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


# instantiate SignalData class
data = SignalData(dataname=prices)
# instantiate Cerebro, add strategy, data, initial cash, commission and pyfolio for performance analysis
cerebro = bt.Cerebro(stdstats=False, cheat_on_open=True)
cerebro.addstrategy(MLStrategy)
cerebro.adddata(data, name=ticker)
cerebro.broker.setcash(100000.0)
cerebro.broker.setcommission(commission=0.001)
cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
# run the backtest
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
backtest_result = cerebro.run()
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

# Extract inputs for pyfolio
strat = backtest_result[0]
pyfoliozer = strat.analyzers.getbyname('pyfolio')
returns, positions, transactions, gross_lev = pyfoliozer.get_pf_items()
returns.name = 'Strategy'
returns.head(2)

# get benchmark returns
benchmark_rets = stock['returns']
benchmark_rets.index = benchmark_rets.index.tz_localize('UTC')
benchmark_rets = benchmark_rets.filter(returns.index)
benchmark_rets.name = f'{ticker}'
benchmark_rets.head(2)

# get performance statistics for strategy
pf.show_perf_stats(returns)

# plot performance for strategy vs benchmark
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
# fig.suptitle('XXX', fontsize=16, y=0.990)

plt.grid(True)
plt.legend()
plt.tight_layout()

plt.draw()
plt.show(block=True)
# plt.savefig('images/chart3', dpi=300)
