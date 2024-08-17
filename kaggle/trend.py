#!/Users/yz/github/myenv/bin/python
from pathlib import Path
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

simplefilter("ignore")  # ignore warnings to clean up output cells

# Set Matplotlib defaults
plt.rc("figure", autolayout=True, figsize=(11, 5))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)

# Load book Traffic dataset
book = pd.read_csv(
    "book_sales.csv",
    index_col='Date',
    parse_dates=['Date'],
).drop('Paperback', axis=1)
print(book.head())

moving_average = book.rolling(
    window=12,       # 12-day window
    center=True,      # puts the average at the center of the window
    min_periods=6,  # choose about half the window size
).mean()              # compute the mean (could also do median, std, min, max, ...)

ax = book.plot(style=".", color="0.5")
moving_average.plot(
    ax=ax, linewidth=3, title="book sales Moving Average", legend=False,
)
# moving average is for trend observation

from statsmodels.tsa.deterministic import DeterministicProcess

dp = DeterministicProcess(
    index=book.index,  # dates from the training data
    constant=False,       # dummy feature for the bias (y_intercept)
    order=2,             # the time dummy (trend)
    drop=True,           # drop terms if necessary to avoid collinearity
)
# `in_sample` creates features for the dates given in the `index` argument
X = dp.in_sample()

print(book.index)
print(X)

from sklearn.linear_model import LinearRegression

y = book["Hardcover"]  # the target

# The intercept is the same as the `const` feature from
# DeterministicProcess. LinearRegression behaves badly with duplicated
# features, so we need to be sure to exclude it here.
model = LinearRegression()
model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)

ax = book.plot(style=".", color="0.5", title="book sales - Linear Trend")
_ = y_pred.plot(ax=ax, linewidth=3, label="Trend")

# fit polinomial model is for trend prediction

plt.show()