#!/usr/bin/env python
# coding: utf-8

# #  Multi-class Classification
# 

# ### Import Necessary Libraries
# 

# In[1]:


get_ipython().system('pip install numpy==2.2.0')
get_ipython().system('pip install pandas==2.2.3')
get_ipython().system('pip install scikit-learn==1.6.0')
get_ipython().system('pip install matplotlib==3.9.3')
get_ipython().system('pip install seaborn==0.13.2')


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')


# ## About the dataset
# The data set being used for this lab is the "Obesity Risk Prediction" data set publically available on <a href="https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition">UCI Library</a> under the <a href="https://creativecommons.org/licenses/by/4.0/legalcode">CCA 4.0</a> license. The data set has 17 attributes in total along with 2,111 samples. 
# 
# The attributes of the dataset are descibed below.
# 

# <style type="text/css">
# .tg  {border-collapse:collapse;border-spacing:0;}
# .tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
#   overflow:hidden;padding:10px 5px;word-break:normal;}
# .tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
#   font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
# .tg .tg-7zrl{text-align:left;vertical-align:bottom}
# </style>
# <table class="tg"><thead>
#   <tr>
#     <th class="tg-7zrl">Variable Name</th>
#     <th class="tg-7zrl">Type</th>
#     <th class="tg-7zrl">Description</th>
#   </tr></thead>
# <tbody>
#   <tr>
#     <td class="tg-7zrl">Gender</td>
#     <td class="tg-7zrl">Categorical</td>
#     <td class="tg-7zrl"></td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">Age</td>
#     <td class="tg-7zrl">Continuous</td>
#     <td class="tg-7zrl"></td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">Height</td>
#     <td class="tg-7zrl">Continuous</td>
#     <td class="tg-7zrl"></td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">Weight</td>
#     <td class="tg-7zrl">Continuous</td>
#     <td class="tg-7zrl"></td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">family_history_with_overweight</td>
#     <td class="tg-7zrl">Binary</td>
#     <td class="tg-7zrl">Has a family member suffered or suffers from overweight?</td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">FAVC</td>
#     <td class="tg-7zrl">Binary</td>
#     <td class="tg-7zrl">Do you eat high caloric food frequently?</td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">FCVC</td>
#     <td class="tg-7zrl">Integer</td>
#     <td class="tg-7zrl">Do you usually eat vegetables in your meals?</td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">NCP</td>
#     <td class="tg-7zrl">Continuous</td>
#     <td class="tg-7zrl">How many main meals do you have daily?</td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">CAEC</td>
#     <td class="tg-7zrl">Categorical</td>
#     <td class="tg-7zrl">Do you eat any food between meals?</td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">SMOKE</td>
#     <td class="tg-7zrl">Binary</td>
#     <td class="tg-7zrl">Do you smoke?</td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">CH2O</td>
#     <td class="tg-7zrl">Continuous</td>
#     <td class="tg-7zrl">How much water do you drink daily?</td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">SCC</td>
#     <td class="tg-7zrl">Binary</td>
#     <td class="tg-7zrl">Do you monitor the calories you eat daily?</td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">FAF</td>
#     <td class="tg-7zrl">Continuous</td>
#     <td class="tg-7zrl">How often do you have physical activity?</td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">TUE</td>
#     <td class="tg-7zrl">Integer</td>
#     <td class="tg-7zrl">How much time do you use technological devices such as cell phone, videogames, television, computer and others?</td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">CALC</td>
#     <td class="tg-7zrl">Categorical</td>
#     <td class="tg-7zrl">How often do you drink alcohol?</td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">MTRANS</td>
#     <td class="tg-7zrl">Categorical</td>
#     <td class="tg-7zrl">Which transportation do you usually use?</td>
#   </tr>
#   <tr>
#     <td class="tg-7zrl">NObeyesdad</td>
#     <td class="tg-7zrl">Categorical</td>
#     <td class="tg-7zrl">Obesity level</td>
#   </tr>
# </tbody></table>
# 

# In[6]:


import os
# Print current working directory
print(os.getcwd())


# In[9]:


# Load dataset
data = pd.read_csv("/home/af2ea8f5-1f9a-4091-9c13-db2b59ea1801/Skill Network/Machine Learning/ObesityDataSet.csv")
data.head()


# ## Preprocessing the data
# 

# ### Feature scaling
# Scale the numerical features to standardize their ranges for better model performance.
# 

# In[10]:


# Standardizing continuous numerical features
continuous_columns = data.select_dtypes(include=['float64']).columns.tolist()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[continuous_columns])

# Converting to a DataFrame
scaled_df = pd.DataFrame(scaled_features, columns=scaler.get_feature_names_out(continuous_columns))

# Combining with the original dataset
scaled_data = pd.concat([data.drop(columns=continuous_columns), scaled_df], axis=1)


# Standardization of data is important to better define the decision boundaries between classes by making sure that the feature variations are in similar scales. The data is now ready to be used for training and testing.

# ### One-hot encoding
# Convert categorical variables into numerical format using one-hot encoding.
# 

# In[11]:


# Identifying categorical columns
categorical_columns = scaled_data.select_dtypes(include=['object']).columns.tolist()
categorical_columns.remove('NObeyesdad')  # Exclude target column

# Applying one-hot encoding
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_features = encoder.fit_transform(scaled_data[categorical_columns])

# Converting to a DataFrame
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))

# Combining with the original dataset
prepped_data = pd.concat([scaled_data.drop(columns=categorical_columns), encoded_df], axis=1)


# ### Encode the target variable
# 

# In[12]:


# Encoding the target variable
prepped_data['NObeyesdad'] = prepped_data['NObeyesdad'].astype('category').cat.codes
prepped_data.head()


# ### Separate the input and target data

# In[13]:


# Preparing final dataset
X = prepped_data.drop('NObeyesdad', axis=1)
y = prepped_data['NObeyesdad']


# ## Model training and evaluation 
# 

# In[14]:


# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ### Logistic Regression with One-vs-All
# In the One-vs-All approach:
# 
# * The algorithm trains a single binary classifier for each class.
# * Each classifier learns to distinguish a single class from all the others combined.
# * If there are k classes, k classifiers are trained.
# * During prediction, the algorithm evaluates all classifiers on each input, and selects the class with the highest confidence score as the predicted class.
# 
# #### Advantages:
# * Simpler and more efficient in terms of the number of classifiers (k)
# * Easier to implement for algorithms that naturally provide confidence scores (e.g., logistic regression, SVM).
# 
# #### Disadvantages:
# * Classifiers may struggle with class imbalance since each binary classifier must distinguish between one class and the rest.
# * Requires the classifier to perform well even with highly imbalanced datasets, as the "all" group typically contains more samples than the "one" class.
# 

# In[15]:


# Training logistic regression model using One-vs-All (default)
model_ova = LogisticRegression(multi_class='ovr', max_iter=1000)
model_ova.fit(X_train, y_train)


# In[16]:


# Predictions
y_pred_ova = model_ova.predict(X_test)

# Evaluation metrics for OvA
print("One-vs-All (OvA) Strategy")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ova),2)}%")


# ### Logistic Regression with OvO
# 
# In the One-vs-One approach:
# * The algorithm trains a binary classifier for every pair of classes in the dataset.
# * If there are k classes, this results in $k(k-1)/2$ classifiers.
# * Each classifier is trained to distinguish between two specific classes, ignoring the rest.
# * During prediction, all classifiers are used, and a "voting" mechanism decides the final class by selecting the class that wins the majority of pairwise comparisons.
# 
# #### Advantages:
# * Suitable for algorithms that are computationally expensive to train on many samples because each binary classifier deals with a smaller dataset (only samples from two classes).
# * Can be more accurate in some cases since classifiers focus on distinguishing between two specific classes at a time.
# 
# #### Disadvantages:
# * Computationally expensive for datasets with a large number of classes due to the large number of classifiers required.
# * May lead to ambiguous predictions if voting results in a tie.
# 

# In[17]:


# Training logistic regression model using One-vs-One
model_ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000))
model_ovo.fit(X_train, y_train)


# In[18]:


# Predictions
y_pred_ovo = model_ovo.predict(X_test)

# Evaluation metrics for OvO
print("One-vs-One (OvO) Strategy")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ovo),2)}%")


# #### Let's Experiment with different test sizes in the train_test_split method (e.g., 0.1, 0.3) and observe the impact on model performance.

# In[19]:


for test_size in [0.1, 0.3]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    model_ova.fit(X_train, y_train)
    y_pred = model_ova.predict(X_test)
    print(f"Test Size: {test_size}")
    print("Accuracy:", accuracy_score(y_test, y_pred))


# #### Plot a bar chart of feature importance comparing the coefficients from the One vs All and the One vs One logistic regression model. 

# In[23]:


# Feature importance ova
feature_importance = np.mean(np.abs(model_ova.coef_), axis=0)
plt.barh(X.columns, feature_importance)
plt.title("Feature Importance One Vs ALL")
plt.xlabel("Importance")
plt.show()


# In[26]:


# Extract and average coefficients across all OvO binary models
coef_list = [np.abs(est.coef_).flatten() for est in model_ovo.estimators_]
feature_importance2 = np.mean(coef_list, axis=0)

# Plot
plt.barh(X.columns, feature_importance2)
plt.title("Feature Importance One Vs One")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()


# ### Writing a function (obesity_risk_pipeline) to automate this entire pipeline:
# - Loading and preprocessing the data
# - Training the model
# - Evaluating the model
#   
# The function should accept the file path and test set size as the input arguments.

# In[27]:


def obesity_risk_pipeline(data_path, test_size=0.2):
    # Load data
    data = pd.read_csv(data_path)

    # Standardizing continuous numerical features
    continuous_columns = data.select_dtypes(include=['float64']).columns.tolist()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[continuous_columns])
    
    # Converting to a DataFrame
    scaled_df = pd.DataFrame(scaled_features, columns=scaler.get_feature_names_out(continuous_columns))
    
    # Combining with the original dataset
    scaled_data = pd.concat([data.drop(columns=continuous_columns), scaled_df], axis=1)

    # Identifying categorical columns
    categorical_columns = scaled_data.select_dtypes(include=['object']).columns.tolist()
    categorical_columns.remove('NObeyesdad')  
    
    # Applying one-hot encoding
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_features = encoder.fit_transform(scaled_data[categorical_columns])
    
    # Converting to a DataFrame
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))
    
    # Combining with the original dataset
    prepped_data = pd.concat([scaled_data.drop(columns=categorical_columns), encoded_df], axis=1)
    
    # Encoding the target variable
    prepped_data['NObeyesdad'] = prepped_data['NObeyesdad'].astype('category').cat.codes

    # Preparing final dataset
    X = prepped_data.drop('NObeyesdad', axis=1)
    y = prepped_data['NObeyesdad']
   
    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    # Training and evaluation
    model = LogisticRegression(multi_class='multinomial', max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))


# In[34]:


# Trying the function
obesity_risk_pipeline('ObesityDataSet.csv', test_size=0.2)

