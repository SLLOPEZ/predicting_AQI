#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#import dataset
df = pd.read_csv('global_air_pollution_dataset.csv')

#exploring data
df.shape

df.info()

df.head()

df.tail()

df.columns

df.isna().sum()

df.nunique()

df.duplicated().sum()

df.describe()

from pandas.core.dtypes.api import is_numeric_dtype

def exploratory_vis(df, columns):
  for col in columns:
    if pd.api.types.is_numeric_dtype(df[col]):
      sns.histplot(df[col])
      plt.title(f'Histogram of {col}')
      plt.xlabel(col)
      plt.ylabel('Count')
      plt.show()

plt.style.use('seaborn-v0_8')

exploratory_vis(df, df.columns)

df['Country'].value_counts()

df['City'].value_counts()

df['AQI Category'].value_counts()

df['CO AQI Category'].value_counts()

df['Ozone AQI Category'].value_counts()

df['NO2 AQI Category'].value_counts()

df['PM2.5 AQI Category'].value_counts()

df.corr()

sns.heatmap(df.corr(),annot = True, cmap = 'coolwarm')

#data preprocessing

#drop the missing value rows
df.dropna(inplace=True)

cat_cols = ['AQI Category','CO AQI Category','Ozone AQI Category','NO2 AQI Category','PM2.5 AQI Category']
num_cols = ['AQI Value', 'CO AQI Value', 'Ozone AQI Value','NO2 AQI Value', 'PM2.5 AQI Value']

df = pd.get_dummies(df, columns = cat_cols, drop_first = True)

df.columns

from sklearn.utils import shuffle

df = shuffle(df)

#splitting into X and y

X = df.drop(['AQI Value','Country', 'City'], axis = 1)
y = df['AQI Value']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2)

#run models

#linear regression model
from sklearn.linear_model import LinearRegression

lg = LinearRegression()

lg.fit(X_train, y_train)

from sklearn import metrics
from sklearn.metrics import r2_score

y_pred = lg.predict(X_test)

mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print('MAE: ', mae,'\n', 'MSE: ', mse, '\n', 'RMSE: ', rmse, '\n', 'R2 score: ', r2)

#decision tree

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()

dt.fit(X_train, y_train)

ydt_pred = dt.predict(X_test)

mae_dt = metrics.mean_absolute_error(y_test, ydt_pred)
mse_dt = metrics.mean_squared_error(y_test, ydt_pred)
rmse_dt = np.sqrt(mse)
r2_dt = r2_score(y_test, ydt_pred)

print('MAE: ', mae_dt,'\n', 'MSE: ', mse_dt, '\n', 'RMSE: ', rmse_dt, '\n', 'R2 score: ', r2_dt)

# random forest model
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 100, random_state = 42 )

rf.fit(X_train, y_train)

yrf_pred = rf.predict(X_test)

mae_rf = metrics.mean_absolute_error(y_test, yrf_pred)
mse_rf = metrics.mean_squared_error(y_test, yrf_pred)
rmse_rf = np.sqrt(mse)
r2_rf = r2_score(y_test, yrf_pred)

print('MAE: ', mae_rf,'\n', 'MSE: ', mse_rf, '\n', 'RMSE: ', rmse_rf, '\n', 'R2 score: ', r2_rf)

