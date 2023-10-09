# Trading Research: Risk Reward Analysis in Trading Strategies
Capstone Project for CSE 485

## Overview
The BacktestTransformer and ForecastTransformer are two models which can make stock price predictions. They can be fed stock data along with any other relevant data, as long as it is prepared beforehand. In the given folders, there are two types of data. The first type deal with technical analysis, and the second type combines the history of many stocks (parallel stock). The models are robust, so many other types of data can be fed, but we mainly focus on stocks for now.

### BacktestTransformer
- Univariate/Multivariate Model which can be fed data to make single step predictions for a target variable.
- The predictions and ground truth are plotted together so you can see how the model's predictions compare with real data.

### ForecastTransformer
- Univariate/Multivariate Model which can be fed data to predict multiple steps into the future.
- The predictions go past the ground truth data, so they can not be evaluated against real data.
