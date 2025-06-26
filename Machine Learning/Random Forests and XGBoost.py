#!/usr/bin/env python
# coding: utf-8

# # Comparing Random Forest and XGBoost
# 

# In this lab, I'll create and measure the relative performances of Random Forest and XGBoost regression models for predicting house prices using the California Housing Dataset.

# In[1]:


get_ipython().system('pip install numpy==2.2.0')
get_ipython().system('pip install scikit-learn==1.6.0')
get_ipython().system('pip install matplotlib==3.9.3')
get_ipython().system('pip install xgboost==2.1.3')


# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import time


#  Get the dataset below. Then, right click anywhere on the page, and save as.

# Then, upload the file to the working directory.

# https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv

# In[8]:


import os
# Print current working directory to load the csv file
print(os.getcwd())


# In[11]:


# Load the California Housing dataset
data = pd.read_csv("/home/af2ea8f5-1f9a-4091-9c13-db2b59ea1801/Skill Network/Machine Learning/my_version/housing.csv") 
data.head()


# In[15]:


X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[16]:


# Observations and features in the dataset
N_observations, N_features = X.shape
print('Number of Observations: ' + str(N_observations))
print('Number of Features: ' + str(N_features))


# ### Initialize models
# 

# In[17]:


# Initialize models
n_estimators=100
rf = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
xgb = XGBRegressor(n_estimators=n_estimators, random_state=42)


# In[25]:


# One-hot encode ocean_proximity
X_encoded = pd.get_dummies(X, columns=['ocean_proximity'])

# Now split again with the encoded data
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)


# In[27]:


# Sanitize feature names to remove special characters
X_train.columns = X_train.columns.str.replace(r"[<>[\]]", "_", regex=True)
X_test.columns = X_test.columns.str.replace(r"[<>[\]]", "_", regex=True)


# In[28]:


# Fit models
# Measure training time for Random Forest
start_time_rf = time.time()
rf.fit(X_train, y_train)
end_time_rf = time.time()
rf_train_time = end_time_rf - start_time_rf

# Measure training time for XGBoost
start_time_xgb = time.time()
xgb.fit(X_train, y_train)
end_time_xgb = time.time()
xgb_train_time = end_time_xgb - start_time_xgb


# ### Use the fitted models to make predictions on the test set.

# In[29]:


# Measure prediction time for Random Forest
start_time_rf = time.time()
y_pred_rf = rf.predict(X_test)
end_time_rf = time.time()
rf_pred_time = end_time_rf - start_time_rf

# Measure prediciton time for XGBoost
start_time_xgb = time.time()
y_pred_xgb = xgb.predict(X_test)
end_time_xgb = time.time()
xgb_pred_time = end_time_xgb - start_time_xgb


# ### Calulate the MSE and R^2 values for both models

# In[30]:


mse_rf = mean_squared_error(y_test, y_pred_rf)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_rf = r2_score(y_test, y_pred_rf)
r2_xgb = r2_score(y_test, y_pred_xgb)


# In[31]:


# Print the MSE and R^2 values for both models
print(f'Random Forest:  MSE = {mse_rf:.4f}, R^2 = {r2_rf:.4f}')
print(f'      XGBoost:  MSE = {mse_xgb:.4f}, R^2 = {r2_xgb:.4f}')


# MSE and R^2 values of XGBoost model is better than Random Forest, but the differences aren't overwhelming.

# ###  Print the timings for each model

# In[32]:


print(f'Random Forest:  Training Time = {rf_train_time:.3f} seconds, Testing time = {rf_pred_time:.3f} seconds')
print(f'      XGBoost:  Training Time = {xgb_train_time:.3f} seconds, Testing time = {xgb_pred_time:.3f} seconds')


# What is very impressive is the difference in computation time between XGBoost and Random Forest for both training and testing.

# ### Calculate the standard deviation of the test data

# In[33]:


# Standard deviation of y_test
std_y = np.std(y_test)


# In[34]:


start_time_rf = time.time()
y_pred_rf = rf.predict(X_test)
end_time_rf = time.time()
rf_pred_time = end_time_rf - start_time_rf

# Measure prediciton time for XGBoost
start_time_xgb = time.time()
y_pred_xgb = xgb.predict(X_test)
end_time_xgb = time.time()
xgb_pred_time = end_time_xgb - start_time_xgb

mse_rf = mean_squared_error(y_test, y_pred_rf)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_rf = r2_score(y_test, y_pred_rf)
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f'Random Forest:  MSE = {mse_rf:.4f}, R^2 = {r2_rf:.4f}')
print(f'      XGBoost:  MSE = {mse_xgb:.4f}, R^2 = {r2_xgb:.4f}')
print(f'Random Forest:  Training Time = {rf_train_time:.3f} seconds, Testing time = {rf_pred_time:.3f} seconds')
print(f'      XGBoost:  Training Time = {xgb_train_time:.3f} seconds, Testing time = {xgb_pred_time:.3f} seconds')
std_y = np.std(y_test)


# ### Visualize the results

# In[36]:


plt.figure(figsize=(14, 6))

# Random Forest plot
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_rf, alpha=0.5, color="blue",ec='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2,label="perfect model")
plt.plot([y_test.min(), y_test.max()], [y_test.min() + std_y, y_test.max() + std_y], 'r--', lw=1, label="+/-1 Std Dev")
plt.plot([y_test.min(), y_test.max()], [y_test.min() - std_y, y_test.max() - std_y], 'r--', lw=1, )
plt.ylim(y_test.min(), y_test.max())
plt.title("Random Forest Predictions vs Actual")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()


# XGBoost plot
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_xgb, alpha=0.5, color="orange",ec='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2,label="perfect model")
plt.plot([y_test.min(), y_test.max()], [y_test.min() + std_y, y_test.max() + std_y], 'r--', lw=1, label="+/-1 Std Dev")
plt.plot([y_test.min(), y_test.max()], [y_test.min() - std_y, y_test.max() - std_y], 'r--', lw=1, )
plt.ylim(y_test.min(), y_test.max())
plt.title("XGBoost Predictions vs Actual")
plt.xlabel("Actual Values")
plt.legend()
plt.tight_layout()
plt.show()

