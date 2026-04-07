# Mumbai AQI Fluctuation — Autoregressive Stochastic Model

A data science project modeling Mumbai's daily Air Quality Index (AQI) 
as a stochastic time series using ARIMA and Markov Chain analysis.

## Features
- STL decomposition of AQI time series (2018–2024)
- Stationarity testing via ADF and KPSS tests
- ACF/PACF analysis for model order identification
- ARIMA(p,d,q) forecasting with 95% confidence intervals
- Markov Chain transition matrix and stationary distribution

## Dataset
2,431 daily AQI observations for Mumbai (May 2018 – Dec 2024)  
Source: CPCB via GitHub (cp099/India-Air-Quality-Dataset) + Kaggle

## Live App
[View on Streamlit](https://mumbaiaqimodel-6vghqbhzfmumiqgx2kfdo8.streamlit.app/)

## Tech Stack
Python · Streamlit · Plotly · Statsmodels · Pandas · SciPy
