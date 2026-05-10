# Time Series Analysis with statsmodels in Python

The statsmodels library combines traditional methods with modern Python capabilities for business forecasting and analysis.

### Time Series Analysis with `statsmodels` in Python 

#### The statsmodels library combines traditional methods with modern Python capabilities for business forecasting and analysis.
**`statsmodels`** provides a lot of tools for statistical modeling, including time series. I've written about other libraries so I thought I should include something about the most popular library out there. If you are using ARIMA or SARIMA it is the logical place to start.

`statsmodels` covers univariate and multivariate time series modeling. It includes lots of statistical tests to assess model assumptions and performance. It has "MS Excel" like outputs of key metrics. And it works well with `pandas` and `numpy`.

Let's check it out. We'll analyze a simulated time series dataset to demonstrate some of the key features of `statsmodels.`

Import required libraries:


#### Generate or Load Time Series Data
Simulate a time series with trend and seasonality:



#### Time Series Decomposition
Use `seasonal_decompose` to split the series into trend, seasonal, and residual components.



#### Step 4: Check for Stationarity 

Use the Augmented Dickey-Fuller (ADF) test to assess stationarity.



#### Autocorrelation and Partial Autocorrelation
Visualize the ACF and PACF to determine lag dependencies.



#### Fit an ARIMA Model
Fit an ARIMA model to the data for forecasting.



#### Forecast Future Values 

Use the fitted model to forecast holdout values.



Our model predicts much slower growth than the series actually has.

#### Holt-Winters Exponential Smoothing


Holt-Winters also doesn't predict the changes very accurately.

So I decided to test this with Tensorflow (Keras) using an LSTM model.



This model did a much better job than the ARIMA and Holt-Winters models. It still under-estimated each value but the trend is clearly closer than the other models.

This doesn't really bother me. ARIMA has limits but I still like statsmodels as a library --- the better prediction from Tensorflow is a function of a different model, not a better API.

#### Real world example: ERCOT energy demand data
Let's look at actual data. This is hourly load demand data for power in ERCOT (basically Texas) from Jan 7--11 2025. This data is available from their website. I'm not going to repeat the code but I'll share the graphs.


The data has clear patterns.


Using ARIMA, we can forecast demand. The cone of uncertainty is large. This could be reduced by adding in more data.


ARIMA actually did a good job here.


Let's look at Holt-Winters.


Not bad. Let's look at Keras.


The LSTM still does a better job predicting the data.

### Key Takeaways
`statsmodels` is the benchmark I use to measure all other time series libraries. It isn't perfect but it is good and if I could only use one library for the rest of time, I would choose `statsmodels`.

Code for this project is available on [GitHub](https://github.com/kylejones200/time_series/blob/main/Statsmodels.ipynb).
