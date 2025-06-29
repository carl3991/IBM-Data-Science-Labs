#!/usr/bin/env python
# coding: utf-8

# # Evaluating Random Forest Performance
# 

# ## Introduction
# In this lab, I will:
# - Use the California Housing data set included in scikit-learn to predict the median house price based on various attributes.
# - Create a random forest regression model and evaluate its performance.
# - Investigate the feature importances for the model.
# 

# In[3]:


get_ipython().system('pip install numpy==2.2.0')
get_ipython().system('pip install pandas==2.2.3')
get_ipython().system('pip install scikit-learn==1.6.0')
get_ipython().system('pip install matplotlib==3.9.3')
get_ipython().system('pip install scipy==1.14.1')


# ## Importing the required libraries

# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import skew


# ### Load the California Housing data set
# 

#  Get the dataset below. Then, right click anywhere on the page, and save as. Then, upload the file to the working directory.

# https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv

# In[9]:


import os
# Print current working directory to load the csv file
print(os.getcwd())


# In[10]:


# Print data
data = pd.read_csv("/home/af2ea8f5-1f9a-4091-9c13-db2b59ea1801/Skill Network/Machine Learning/housing.csv")
data.head()                   
                


# In[13]:


# Dropping ocean_proximity for numerical analysis
X = data.drop(columns=['median_house_value', 'ocean_proximity'])
y = data['median_house_value']


# ### Print the description of the California Housing data set
# 

# In[12]:


print(data.info)


# ### Split the data into training and testing sets
# 

# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## Explore the training data
# 

# In[15]:


# Create a copy of X_train as a DataFrame
eda = pd.DataFrame(data=X_train, columns=X.columns)

# Add the target variable
eda['median_house_value'] = y_train

# Summary statistics
eda.describe()


# Considering the 25th to the 75th percentile range, most of the median house prices fall within 119,800 and 265,000 dollars.

# In[16]:


# Plot the distribution
plt.hist(1e5*y_train, bins=30, color='lightgreen', edgecolor='black')
plt.title(f'Median House Value Distribution\nSkewness: {skew(y_train):.2f}')
plt.xlabel('Median House Value')
plt.ylabel('Frequency')


# Evidently, the distribution is skewed and there are quite a few clipped values at around $500,000.

# ### Model fitting and prediction
# Let's fit a random forest regression model to the data and use it to make median house price predicions.

# In[17]:


# Initialize and fit the Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

# Predict on test set
y_pred_test = rf_regressor.predict(X_test)


# ### Estimate out-of-sample MAE, MSE, RMSE, and R²
# 

# In[18]:


mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse = root_mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")


# The mean absolute error is $32,044.4 
# 
# On average, predicted median house prices are off by $32k.
# 
# Mean squared error is less intuitive to interpret, but is usually what is being minimized by the model fit.
# On the other hand, taking the square root of MSE yields a dollar value, here RMSE = $49,840.0963
# 
# An R-squared score of 0.81 is not considered very high. It means the model explains about %81 of the variance in median house prices, although this interpretation can be misleading for complex data with nonlinear relationships, skewed values, and outliers. R-squared can still be useful for comparing models though.

# ### Plot Actual vs Predicted values

# In[22]:


plt.scatter(y_test, y_pred_test, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Random Forest Regression - Actual vs Predicted')
plt.show()


# ### Plot the histogram of the residual errors (dollars)

# In[24]:


# Calculate the residual errors
residuals = 1e5*(y_test - y_pred_test)

# Plot the histogram of the residuals
plt.hist(residuals, bins =30, color='lightblue', edgecolor='black')
plt.title(f'Median House Value Prediction Residuals')
plt.xlabel('Median House Value Prediction Error ($)')
plt.ylabel('Frequency')
print('Average Error = ' + str(int(np.mean(residuals))))
print('Standard deviation of error = ' + str(int(np.std(residuals))))


# The residuals are normally distributed.

# ### Plot the model residual errors by median house value.

# In[26]:


# Create a DataFrame to make sorting easy
residuals_df = pd.DataFrame({
    'Actual': 1e5*y_test,
    'Residuals': residuals
})

# Sort the DataFrame by the actual target values
residuals_df = residuals_df.sort_values(by='Actual')

# Plot the residuals
plt.scatter(residuals_df['Actual'], residuals_df['Residuals'], marker='o', alpha=0.4,ec='k')
plt.title('Median House Value Prediciton Residuals Ordered by Actual Median Prices')
plt.xlabel('Actual Values (Sorted)')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()


# The average error as a function of median house price is actually increasing from negative to positive values. In other words, lower median prices tend to be overpredicted while higher median prices tend to be underpredicted.

#  ### Display the feature importances as a bar chart.

# In[28]:


# Feature importances
importances = rf_regressor.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

# Plot feature importances
plt.bar(range(X.shape[1]), importances[indices],  align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importances in Random Forest Regression")
plt.show()


# It makes sense that median incomes and house prices would be correlated, so it's not surprising that median income is the most important feature. Since location is implied by two separate variables, latitude and longitude that share equal importances, we might speculate that location is really the second most important feature. This is because replacing latitude and longitude with a categorical location at an appropriate level of granularity (suburb, city, etc.) would likely have a combined lat/lng importance, which might outweigh average occupancy.

# ### Final thoughts:
# 
# Compared to linear regression, random forest regression is quite robust against outliers and skewed distributions. This is because random forest regression doesn't make any assumptions about the data distribution, where linear regression performs best with normally distributed data.
# Standardizing the data isn't necessary like it is for distance-based algortihms like KNN or SVMs.

# In[ ]:




