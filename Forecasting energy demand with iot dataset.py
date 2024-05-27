#!/usr/bin/env python
# coding: utf-8

# # IMPORTING NECESSARY LIBRARIES

# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import xgboost as xgb
from xgboost import XGBRegressor
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, max_error
from math import sqrt
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# # LOADING THE DATASET

# In[2]:


data = pd.read_csv('iot_load_data.csv')


# # VIEWING THE DATASET

# ## To display the contents of the Dataset by accessing the pandas Dataframe named data.

# In[3]:


data


# ## data.describe()  summarizes our data in `data`. It provides statistics like mean, standard deviation for numerical columns, helping us to understand data spread.

# In[4]:


data.describe()


# ## data.info() in Pandas gives us a quick peek at our data's structure. It shows details like number of rows, columns, data types, memory usage and missing values.

# In[5]:


data.info()


# # DATA PREPROCESSING

# ## To determine the preprocessing and data cleaning steps required for the dataset in `iot_load_data.csv`, We first inspected the data to identify any potential issues such as missing values, incorrect data types, outliers, or inconsistencies.

# ### Step 1: Check for missing values in each column and removing them if any

# In[6]:


data_missing_values = data.isnull().sum()

print('\Missing Values in Each Column:')
print(data_missing_values)


# In[7]:


data_missing_values = data_missing_values[data_missing_values > 0]

if not data_missing_values.empty:
    for column in data_missing_values.index:
        if data[column].dtype == 'object':
            data[column] = data[column].fillna(data[column].mode()[0])
        else:
            data[column] = data[column].fillna(data[column].median())
    print('Missing values have been handled.')
else:
    print('No missing values to handle.')


# 
# ### Step 2: The 'Timestamp' column is recognized as an object type, which suggests it is stored as a string. This has been converted to a datetime type for our forecasting model  and checked we whether it has been updated.

# In[8]:


data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data['Day'] = data['Timestamp'].dt.day
data['Month'] = data['Timestamp'].dt.month
data['Year'] = data['Timestamp'].dt.year

# Confirm the conversion
data_types = data.dtypes
print('Updated Data Types:')
print(data_types)


# ### Step 3: Checking for outliers in numerical columns to ensure data quality and we applied Capping outliers,it is a technique used in data analysis to address extreme values that fall outside the typical range of the data. It involves setting thresholds and replacing outlier values with those thresholds.

# In[9]:


# Check for outliers using the IQR method
def detect_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

# Apply outlier detection to a few numerical columns
outliers_electricity = detect_outliers(data, 'Electricity_Load')
outliers_temperature = detect_outliers(data, 'Temperature')
outliers_humidity = detect_outliers(data, 'Humidity')

print('Outliers in Electricity Load:')
print(outliers_electricity)
print('\nOutliers in Temperature:')
print(outliers_temperature)
print('\nOutliers in Humidity:')
print(outliers_humidity)


# In[10]:


# Handling outliers by capping them to the upper and lower bounds
def cap_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data[column] = data[column].clip(lower=lower_bound, upper=upper_bound)
    return data

data = cap_outliers(data, 'Electricity_Load')
data = cap_outliers(data, 'Temperature')
data = cap_outliers(data, 'Humidity')

print('Outliers have been capped for Electricity Load, Temperature, and Humidity.')


# ### Step 4: Standardizing categorical data by ensuring consistent casing and removing any leading/trailing spaces

# In[11]:


data['Time_of_Day'] = data['Time_of_Day'].str.strip().str.lower()
data['External_Factors'] = data['External_Factors'].str.strip().str.lower()

unique_time_of_day = data['Time_of_Day'].unique()
unique_external_factors = data['External_Factors'].unique()

print('Unique Time of Day categories after standardization:')
print(unique_time_of_day)
print('\nUnique External Factors categories after standardization:')
print(unique_external_factors)


# In[12]:


le = LabelEncoder()

data['Time_of_Day_encoded'] = le.fit_transform(data['Time_of_Day'])
data['External_Factors_encoded'] = le.fit_transform(data['External_Factors'])


# In[13]:


data


# # Performing exploratory data analysis (EDA)

# ## Histograms of Numerical Columns: This visualization shows the distribution of values in the numerical columns

# In[14]:


numerical_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]

fig, axes = plt.subplots(nrows=int((len(numerical_cols) - 1) / 3) + 1, ncols=3, figsize=(15, 30))  

for i, col in enumerate(numerical_cols):
    ax = axes.flat[i]  
    data[col].hist(ax=ax)
    ax.set_title(col)
    ax.grid(True)

for ax in axes.flat[len(numerical_cols):]:
    ax.axis('off')

# Show the plot
plt.tight_layout() 
plt.show()


# ## Boxplots of Numerical Columns of key variables: These boxplots provide a view of the central tendency and spread of the numerical data, along with any remaining outliers after capping.

# In[15]:


plt.figure(facecolor='white')

fig, axes = plt.subplots(5, 2, figsize=(15, 15))

sns.boxplot(ax=axes[0, 0], x=data['Electricity_Load'])
axes[0, 0].set_title('Electricity Load Distribution')

sns.boxplot(ax=axes[0, 1], x=data['Temperature'])
axes[0, 1].set_title('Temperature Distribution')

sns.boxplot(ax=axes[1, 0], x=data['Humidity'])
axes[1, 0].set_title('Humidity Distribution')

sns.boxplot(ax=axes[1, 1], x=data['Day_Ahead_Demand'])
axes[1, 1].set_title('Day Ahead Demand Distribution')

sns.boxplot(ax=axes[2, 0], x=data['Real_Time_LMP'])
axes[2, 0].set_title('Real Time LMP Distribution')

sns.boxplot(ax=axes[2, 1], x=data['System_Load'])
axes[2, 1].set_title('System Load Distribution')

sns.boxplot(ax=axes[3, 0], x=data['Previous_Load'])
axes[3, 0].set_title('Previous Load Distribution')

sns.boxplot(ax=axes[3, 1], x=data['Transportation_Data'])
axes[3, 1].set_title('Transportation Data Distribution')

sns.boxplot(ax=axes[4, 0], x=data['Operational_Metrics'])
axes[4, 0].set_title('Operational Metrics Distribution')

sns.boxplot(ax=axes[4, 1], x=data['IoT_Sensor_Data'])
axes[4, 1].set_title('IoT Sensor Data Distribution')

plt.tight_layout()
plt.show()


# ## Visualizing the distribution of key variables to understand their spread and central tendency better using Histograms.

# In[16]:


plt.figure(facecolor='white')

fig, axes = plt.subplots(5, 2, figsize=(15, 15))

sns.histplot(ax=axes[0, 0], x=data['Electricity_Load'], kde=True)
axes[0, 0].set_title('Electricity Load Distribution')

sns.histplot(ax=axes[0, 1], x=data['Temperature'], kde=True)
axes[0, 1].set_title('Temperature Distribution')

sns.histplot(ax=axes[1, 0], x=data['Humidity'], kde=True)
axes[1, 0].set_title('Humidity Distribution')

sns.histplot(ax=axes[1, 1], x=data['Day_Ahead_Demand'], kde=True)
axes[1, 1].set_title('Day Ahead Demand Distribution')

sns.histplot(ax=axes[2, 0], x=data['Real_Time_LMP'], kde=True)
axes[2, 0].set_title('Real Time LMP Distribution')

sns.histplot(ax=axes[2, 1], x=data['System_Load'], kde=True)
axes[2, 1].set_title('System Load Distribution')

sns.histplot(ax=axes[3, 0], x=data['Previous_Load'], kde=True)
axes[3, 0].set_title('Previous Load Distribution')

sns.histplot(ax=axes[3, 1], x=data['Transportation_Data'], kde=True)
axes[3, 1].set_title('Transportation Data Distribution')

sns.histplot(ax=axes[4, 0], x=data['Operational_Metrics'], kde=True)
axes[4, 0].set_title('Operational Metrics Distribution')

sns.histplot(ax=axes[4, 1], x=data['IoT_Sensor_Data'], kde=True)
axes[4, 1].set_title('IoT Sensor Data Distribution')

plt.tight_layout()
plt.show()


# ## Count plots for categorical columns

# In[17]:


plt.figure(facecolor='white')

categorical_columns = ['Day_of_Week', 'Time_of_Day', 'External_Factors']
plt.figure(figsize=(15, 10), facecolor='white')
for i, column in enumerate(categorical_columns):
    plt.subplot(2, 2, i+1)
    sns.countplot(x=column, data=data, palette='Set3')
    plt.title('Count Plot of ' + column)
    plt.xticks(rotation=45)
    
plt.tight_layout()
plt.show()


# ## Analyzing the correlations between these variables to understand their relationships

# In[18]:


numeric_data = data.select_dtypes(include=[np.number])

plt.figure(figsize=(20, 15))
corr_matrix = numeric_data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


# ## Scatter plot: To analyse the relationship between Electricity Load with other columns in the dataset

# In[19]:


numerical_cols = [col for col in data.columns if col != 'Electricity_Load' and pd.api.types.is_numeric_dtype(data[col])]

fig, axes = plt.subplots(nrows=int((len(numerical_cols) - 1) / 3) + 1, ncols=3, figsize=(15, 60))  
for i, col in enumerate(numerical_cols):
    ax = axes.flat[i]
    ax.scatter(data[col], data['Electricity_Load'])  
    ax.set_xlabel(col)
    ax.set_ylabel('Electricity Load')
    ax.set_title('Electricity Load vs. {}'.format(col))
    ax.grid(True)

for ax in axes.flat[len(numerical_cols):]:
    ax.axis('off')

plt.tight_layout() 
plt.show()


# ## Performing time series analysis on the 'Electricity_Load' variable: Plotting the time series data and Decomposing the time series to observe trend, seasonality, and residuals

# In[20]:


load_data = data.set_index('Timestamp')

plt.figure(figsize=(10,5))
plt.plot(load_data['Electricity_Load'], label='Electricity Load')
plt.title('Electricity Load Over Time')
plt.xlabel('Time')
plt.ylabel('Electricity Load')
plt.legend()
plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose

result = seasonal_decompose(load_data['Electricity_Load'], model='additive', period=365)
result.plot()
plt.show()


# ## Identifying any seasonal patterns in the electricity load time series:  The monthly average electricity load over time, highlighting any long-term seasonal trends is analysed and the distribution of electricity load for each month, revealing seasonal patterns is analysed.

# In[21]:


monthly_load = load_data['Electricity_Load'].resample('ME').mean()

plt.figure(figsize=(15, 6))
plt.plot(monthly_load, label='Monthly Average Electricity Load')
plt.title('Monthly Average Electricity Load Over Time')
plt.xlabel('Time')
plt.ylabel('Electricity Load')
plt.legend()
plt.show()

plt.figure(figsize=(15, 6))
sns.boxplot(x=load_data.index.month, y=load_data['Electricity_Load'])
plt.title('Seasonal Patterns in Electricity Load')
plt.xlabel('Month')
plt.ylabel('Electricity Load')
plt.show()


# ## Grouping the data by month and calculating the mean electricity load for each month and Sorting the months by mean electricity load in descending order to identify peak load months

# In[22]:


monthly_load_mean = load_data['Electricity_Load'].groupby(load_data.index.month).mean()

peak_load_months = monthly_load_mean.sort_values(ascending=False)

print(peak_load_months)


# The peak load months based on the seasonal patterns are June, May, and November, with June having the highest average electricity load.

# ## Analyzing Autocorrelation in Electricity Load Data: To investigate the presence and patterns of dependence within the electricity load time series.

# In[23]:


plt.figure(facecolor='white')

plot_acf(load_data['Electricity_Load'], lags=50)
plt.title('Autocorrelation Function (ACF)')
plt.show()

plt.figure(facecolor='white')
plot_pacf(load_data['Electricity_Load'], lags=50)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()


# ## Trend analysis: Original electricity load time series and the moving average (with a 30-day window) is analysed.

# In[24]:


plt.figure(facecolor='white')

load_data['Moving_Average'] = load_data['Electricity_Load'].rolling(window=30).mean()
plt.plot(load_data['Electricity_Load'], label='Original')
plt.plot(load_data['Moving_Average'], label='Moving Average', color='red')
plt.xlabel('Date')
plt.ylabel('Electricity Load')
plt.title('Trend Analysis of Electricity Load')
plt.legend()
plt.show()


# # Developing Forecast models

# ## 1.Forecasting Energy demand using LSTM

# In[25]:


values = data['Electricity_Load'].values.reshape(-1,1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_values = scaler.fit_transform(values)
train_size = int(len(scaled_values) * 0.8)
train, test = scaled_values[0:train_size], scaled_values[train_size:len(scaled_values)]

def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 1
X_train, y_train = create_dataset(train, look_back)
X_test, y_test = create_dataset(test, look_back)

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Building the LSTM model
model = Sequential([LSTM(50, activation='relu', input_shape=(1, look_back)), Dense(1)])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, batch_size=72, verbose=2)

# Make predictions
y_pred = model.predict(X_test)
y_test = scaler.inverse_transform([y_test])
y_pred = scaler.inverse_transform(y_pred)

# Model Evaluation
rmse = sqrt(mean_squared_error(y_test[0], y_pred[:,0]))
mae = mean_absolute_error(y_test[0], y_pred[:,0])
mape = np.mean(np.abs((y_test[0] - y_pred[:,0]) / y_test[0])) * 100
r2 = r2_score(y_test[0], y_pred[:,0])
evs = explained_variance_score(y_test[0], y_pred[:,0])
max_err = max_error(y_test[0], y_pred[:,0])

# Plotting the forecast against the actual values
plt.figure(figsize=(10, 6))
plt.plot(y_test[0], label='Actual')
plt.plot(y_pred[:,0], label='Predicted')
plt.title('Forecasting Energy Demand using LSTM model')
plt.legend()
plt.show()


# ## 2.Forecasting Energy Demand using VAR model

# In[26]:


# Select relevant columns for VAR model
var_data = data[['Electricity_Load', 'Temperature', 'Humidity', 'Day_of_Week', 'Holiday_Indicator', 'Previous_Load', 'Transportation_Data', 'Operational_Metrics', 'IoT_Sensor_Data', 'Day_Ahead_Demand', 'Real_Time_LMP', 'Regulation_Capacity', 'Day_Ahead_LMP', 'Day_Ahead_EC', 'Day_Ahead_CC', 'Day_Ahead_MLC', 'Real_Time_EC', 'Real_Time_CC', 'Real_Time_MLC', 'System_Load', 'Day', 'Month', 'Year']
]

# Split the data into train and test sets
train, test = train_test_split(var_data, test_size=0.2, shuffle=False)

# Fit the VAR model
model = VAR(train)
model_fitted = model.fit()

# Forecast
forecast_input = train.values[-model_fitted.k_ar:]
forecast = model_fitted.forecast(y=forecast_input, steps=len(test))
forecast_data = pd.DataFrame(forecast, index=test.index, columns=train.columns)

# MODEL EVALUATION
var_rmse = np.sqrt(mean_squared_error(test['Electricity_Load'], forecast_data['Electricity_Load']))
var_mae = mean_absolute_error(test['Electricity_Load'], forecast_data['Electricity_Load'])
var_mape = np.mean(np.abs((test['Electricity_Load'] - forecast_data['Electricity_Load']) / test['Electricity_Load'])) * 100
var_r2 = r2_score(test['Electricity_Load'], forecast_data['Electricity_Load'])
var_evs = explained_variance_score(test['Electricity_Load'], forecast_data['Electricity_Load'])
var_max_err = max_error(test['Electricity_Load'], forecast_data['Electricity_Load'])

# Plot the forecast
plt.figure(figsize=(10, 5))
plt.title('Forecasting Energy Demand using VAR model')
plt.plot(train.index, train['Electricity_Load'], label='Train')
plt.plot(test.index, test['Electricity_Load'], label='Test')
plt.plot(forecast_data.index, forecast_data['Electricity_Load'], label='Forecast')
plt.legend()
plt.xlabel('Year')  # Change to 'Year' if you only want to show years
plt.ylabel('Electricity Load')
plt.show()


# ## 3.Forecasting Energy Demand using SARIMA model

# In[27]:


electricity_load = data['Electricity_Load']
train_size = int(len(electricity_load) * 0.8)
train, test = electricity_load[:train_size], electricity_load[train_size:]

# Fit the SARIMA model
model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model_fit = model.fit(disp=False)

# Forecast
forecast = model_fit.forecast(steps=len(test))

# Evaluate the model
sarima_mae = mean_absolute_error(test, forecast)
sarima_mape = np.mean(np.abs((test - forecast) / test)) * 100
sarima_r2 = r2_score(test, forecast)
sarima_evs = explained_variance_score(test, forecast)
sarima_max_err = max_error(test, forecast)
mse = mean_squared_error(test, forecast)
sarima_rmse = np.sqrt(mse)

# Plot the results
plt.figure(facecolor='white')
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.title('Forecasting Energy Demand using SARIMA model')
plt.plot(test.index, forecast, label='Forecast')
plt.legend()
plt.show()


# ## 4.Forecasting Energy Demand using SVR model

# In[28]:


features = ['Temperature', 'Humidity', 'Day_of_Week', 'Time_of_Day', 'Holiday_Indicator', 'Previous_Load', 'Transportation_Data', 'Operational_Metrics', 'IoT_Sensor_Data', 'External_Factors', 'Day_Ahead_Demand', 'Real_Time_LMP', 'Regulation_Capacity', 'Day_Ahead_LMP', 'Day_Ahead_EC', 'Day_Ahead_CC', 'Day_Ahead_MLC', 'Real_Time_EC', 'Real_Time_CC', 'Real_Time_MLC']
target = 'Electricity_Load'

X = data[features]
y = data[target]

X = pd.get_dummies(X, columns=['Time_of_Day', 'External_Factors'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the SVR model
svr_model = SVR(kernel='rbf')
svr_model.fit(X_train, y_train)

# Make predictions
y_pred = svr_model.predict(X_test)

plt.figure(facecolor='white')
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.xlabel('Sample Index')
plt.ylabel('Electricity Load')
plt.title('Forecasting Energy demand using SVR model')
plt.legend()
plt.show()

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
svr_rmse = np.sqrt(mse)
svr_mae = mean_absolute_error(y_test, y_pred)
svr_mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
svr_r2 = r2_score(y_test, y_pred)
svr_evs = explained_variance_score(y_test, y_pred)
svr_max_err = max_error(y_test, y_pred)


# ## 5.Forecasting Energy demand using ARIMA Model

# In[29]:


def calculate_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

# Splitting the data into training and testing sets
train_size = int(len(load_data) * 0.8)
train, test = load_data['Electricity_Load'][:train_size], load_data['Electricity_Load'][train_size:]

# Fitting the ARIMA model
model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()

# Forecasting
forecast = model_fit.forecast(steps=len(test))
arima_rmse = calculate_rmse(test, forecast)
arima_mae = mean_absolute_error(test, forecast)
arima_mape = np.mean(np.abs((test - forecast) / test)) * 100
arima_r2 = r2_score(test, forecast)
arima_evs = explained_variance_score(test, forecast)
arima_max_err = max_error(test, forecast)

# Plotting the results
plt.figure(facecolor='white')
plt.plot(train, label='Training Data')
plt.plot(test.index, test, label='Actual Load')
plt.plot(test.index, forecast, label='Forecasted Load', color='red')
plt.xlabel('Date')
plt.ylabel('Electricity Load')
plt.title('Forecasting Energy demand using ARIMA Model')
plt.legend()
plt.show()


# ## 6.Forecasting Energy demand using Holt Winters Model

# In[30]:


def calculate_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

# Holt-Winters model
hw_model = ExponentialSmoothing(train, seasonal='add', seasonal_periods=365)
hw_fit = hw_model.fit()
hw_forecast = hw_fit.forecast(steps=len(test))
hw_rmse = calculate_rmse(test, hw_forecast)
hw_mae = mean_absolute_error(test, hw_forecast)
hw_mape = np.mean(np.abs((test - hw_forecast) / test)) * 100
hw_r2 = r2_score(test, hw_forecast)
hw_evs = explained_variance_score(test, hw_forecast)
hw_max_err = max_error(test, hw_forecast)

# Plotting the results
plt.figure(facecolor='white')
plt.plot(train, label='Training Data')
plt.plot(test.index, test, label='Actual Load')
plt.plot(test.index, hw_forecast, label='Holt-Winters Forecast', color='green')
plt.xlabel('Date')
plt.ylabel('Electricity Load')
plt.title('Forecasting Energy demand using Holt Winters Model')
plt.legend()
plt.show()


# ## 7.Forecasting Energy demand using XGBoost Model

# In[31]:


pip install xgboost


# In[32]:


xgb_data = data[['Electricity_Load', 'Temperature', 'Humidity', 'Day_of_Week', 'Holiday_Indicator', 'Previous_Load', 'Transportation_Data', 'Operational_Metrics', 'IoT_Sensor_Data', 'Day_Ahead_Demand', 'Real_Time_LMP', 'Regulation_Capacity', 'Day_Ahead_LMP', 'Day_Ahead_EC', 'Day_Ahead_CC', 'Day_Ahead_MLC', 'Real_Time_EC', 'Real_Time_CC', 'Real_Time_MLC', 'System_Load', 'Day', 'Month', 'Year']
]

# Split the data into train and test sets
X = xgb_data.drop(columns=['Electricity_Load'])
y = xgb_data['Electricity_Load']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Fit the XGBoost model
xgb_model = XGBRegressor(objective='reg:squarederror')
xgb_model.fit(X_train, y_train)

# Predict
xgb_forecast = xgb_model.predict(X_test)

# Calculate RMSE for XGBoost model
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_forecast))
xgb_mae = mean_absolute_error(y_test, xgb_forecast)
xgb_mape = np.mean(np.abs((y_test - xgb_forecast) / y_test)) * 100
xgb_r2 = r2_score(y_test, xgb_forecast)
xgb_evs = explained_variance_score(y_test, xgb_forecast)
xgb_max_err = max_error(y_test, xgb_forecast)


# Plot the forecast
plt.figure(figsize=(10, 5), facecolor='white')
plt.plot(y_train.index, y_train, label='Train')
plt.plot(y_test.index, y_test, label='Test')
plt.plot(y_test.index, xgb_forecast, label='Forecast')
plt.legend()
plt.title('Forecasting Energy demand using XGBoost Model')
plt.xlabel('Date')
plt.ylabel('Electricity Load')
plt.show()


# In[33]:


pip install tabulate


# In[34]:


# Create a list of lists to store the table data
table_data = [["Algorithm", "RMSE", "MAE", "MAPE", "R-squared", "Explained Variance", "Max Error"]]

# Append the results for each algorithm to the table_data list
table_data.append(["LSTM", rmse, mae, f"{mape:.2f}%", f"{r2:.2f}", f"{evs:.2f}", max_err])
table_data.append(["VAR", var_rmse, var_mae, f"{var_mape:.2f}%", f"{var_r2:.2f}", f"{var_evs:.2f}", var_max_err])
table_data.append(["SARIMA", rmse, sarima_mae, f"{sarima_mape:.2f}%", f"{sarima_r2:.2f}", f"{sarima_evs:.2f}", sarima_max_err])
table_data.append(["SVR", rmse, svr_mae, f"{svr_mape:.2f}%", f"{svr_r2:.2f}", f"{svr_evs:.2f}", svr_max_err])
table_data.append(["ARIMA", arima_rmse, arima_mae, f"{arima_mape:.2f}%", f"{arima_r2:.2f}", f"{arima_evs:.2f}", arima_max_err])
table_data.append(["Holt-Winters", hw_rmse, hw_mae, f"{hw_mape:.2f}%", f"{hw_r2:.2f}", f"{hw_evs:.2f}", hw_max_err])
table_data.append(["XGBoost", xgb_rmse, xgb_mae, f"{xgb_mape:.2f}%", f"{xgb_r2:.2f}", f"{xgb_evs:.2f}", xgb_max_err])

# Print the table
print(tabulate(table_data, headers="firstrow", tablefmt="grid"))


# In[35]:


data


# In[36]:


# Define features and target
features = ['Day','Month','Year','Time_of_Day_encoded','Electricity_Load', 'Temperature', 'Humidity', 'Holiday_Indicator', 'Previous_Load', 'Transportation_Data', 'Operational_Metrics','System_Load','External_Factors_encoded'] 
target='Electricity_Load'
            
X = data[features]
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the XGBoost model
xgb_model = xgb.XGBRegressor(objective='reg:squarederror')

# Hyperparameter tuning using Grid Search
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters from Grid Search
best_params = grid_search.best_params_
print('Best parameters found: ', best_params)

# Train the model with the best parameters
best_xgb_model = xgb.XGBRegressor(**best_params)
best_xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = best_xgb_model.predict(X_test)

# Combine the actual and predicted values for plotting
train_dates = data['Timestamp'][:len(y_train)]
test_dates = data['Timestamp'][len(y_train):len(y_train) + len(y_test)]

# Plot the results
plt.figure(figsize=(14, 7))
plt.plot(train_dates, y_train, label='Training Data', color='blue')
plt.plot(test_dates, y_test, label='Testing Data', color='black')
plt.plot(test_dates, y_pred, label='Forecasting', color='red')
plt.xlabel('Date')
plt.ylabel('Electricity Load')
plt.title('Training, Testing, and Forecasting Electricity Load')
plt.legend()
plt.show()

# Evaluate the model
xgb_new_rmse = mean_squared_error(y_test, y_pred, squared=False)
xgb_new_mae = mean_absolute_error(y_test, y_pred)
xgb_new_mape = (abs((y_test - y_pred) / y_test).mean()) * 100
xgb_new_r2 = r2_score(y_test, y_pred)
xgb_new_evs = explained_variance_score(y_test, y_pred)
xgb_new_max_err = max_error(y_test, y_pred)

print('XGBoost Root Mean Square Error:', xgb_new_rmse)
print('XGBoost Mean Absolute Error:', xgb_new_mae)
print('XGBoost Mean Absolute Percentage Error:', xgb_new_mape)
print('XGBoost R-squared:', xgb_new_r2)
print('XGBoost Explained Variance Score:', xgb_new_evs)
print('XGBoost Max Error:', xgb_new_max_err)


# In[37]:


algorithms = ["LSTM", "VAR", "SARIMA", "SVR", "ARIMA", "Holt-Winters", "XGBoost"]
performance_metrics = {
    "RMSE": [141.491, 141.781, 141.491, 141.491, 147.042, 151.81, 1.442],
    "MAE": [121.732, 121.992, 122.422, 126.445, 125.012, 129.025, 1.053],
    "MAPE": [17.33, 17.31, 17.50, 17.40, 18.63, 18.19, 14.29],
    "Max Error": [249.047, 251.987, 260.437, 256.111, 287.414, 387.528, 10.5],
    "R-squared": [-0.0, -0.0, -0.01, -0.02, -0.08, -0.15, 0.99],
    "Explained Variance": [-0.0, -0.0, -0.01, -0.0, 0.0, -0.15, 0.99]
}

metrics_to_visualize = ["RMSE", "MAE", "MAPE", "Max Error", "R-squared", "Explained Variance"]

num_rows = 2
num_cols = int(len(metrics_to_visualize) / num_rows) + (len(metrics_to_visualize) % num_rows > 0)

fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, 8))  # Adjust figure size as needed

width = 0.35
for i, metric in enumerate(metrics_to_visualize):
    row = i // num_cols
    col = i % num_cols
    x = [j - width/len(metrics_to_visualize) + col * width for j in range(len(algorithms))]
    axes[row, col].bar(x, performance_metrics[metric], width, label=metric, color='C' + str(i))
    axes[row, col].set_xlabel('Algorithm')
    axes[row, col].set_ylabel('Metric Value')
    axes[row, col].set_title(metric)
    axes[row, col].set_xticks([j + width/2 for j in range(len(algorithms))], algorithms, rotation=45, ha='right') 

fig.suptitle('Performance Metric Comparison Across Models', fontsize=12) 
plt.tight_layout()
plt.show()


# In[38]:


from joblib import dump 

dump(best_xgb_model, "Energy_demand_forecast_model.pkl" )


# In[ ]:




