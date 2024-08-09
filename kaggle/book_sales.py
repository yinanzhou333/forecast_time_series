#!/Users/yz/github/myenv/bin/python
import pandas as pd
df = pd.read_csv(
    "book_sales.csv",
    index_col='Date',
    parse_dates=['Date'],
).drop('Paperback', axis=1)
print(df.head())

import numpy as np
df['Time'] = np.arange(len(df.index))
print(df.head())

import matplotlib.pyplot as plt
import seaborn as sns
plt.rc(
    "figure",
    autolayout=True,
    figsize=(11, 4),
    titlesize=18,
    titleweight='bold',
)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)

fig, ax = plt.subplots()
ax.plot('Time', 'Hardcover', data=df, color='0.75')
ax = sns.regplot(x='Time', y='Hardcover', data=df, ci=None, scatter_kws=dict(color='0.25'))
ax.set_title('Time Plot of Hardcover Sales');
#plt.show()

df['Lag_1'] = df['Hardcover'].shift(1)
#df = df.reindex(columns=['Hardcover', 'Lag_1'])
print(df.head())

fig, ax = plt.subplots()
ax = sns.regplot(x='Lag_1', y='Hardcover', data=df, ci=None, scatter_kws=dict(color='0.25'))
ax.set_aspect('equal')
ax.set_title('Lag Plot of Hardcover Sales');
#plt.show()

#target = weight_1 * feature_1 + weight_2 * feature_2 + bias
#feature_1 is the time dummy 0,1,2,...
#feature_2 is the lag_1

# Define features (X) and target (y)
X = df[['Time', 'Lag_1']][1:]
y = df['Hardcover'][1:]
# Initialize the linear regression model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
# Fit the model
model.fit(X, y)
# Retrieve the coefficients and bias
weight_1, weight_2 = model.coef_
bias = model.intercept_
print(f"Weight 1: {weight_1}")
print(f"Weight 2: {weight_2}")
print(f"Bias: {bias}")

y_pred = model.predict(X)
plt.figure()
plt.scatter(X['Time'],y,color='blue',label='true data')
plt.scatter(X['Time'],y_pred,color='red',label='prediction')
plt.xlabel('Time dummy')
plt.ylabel('Hardcover')
plt.legend()
plt.show()

