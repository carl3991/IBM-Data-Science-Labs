#!/usr/bin/env python
# coding: utf-8

# # Evaluating Classification Models
# 

# In this lab, I will:
# - Use the breast cancer data set included in scikit-learn to predict whether a tumor is benign or malignant
# - Create two classification models and evaluate them. 
# - Add some Gaussian random noise to the features to simulate measurement errors.

# In[1]:


get_ipython().system('pip install numpy==2.2.0')
get_ipython().system('pip install pandas==2.2.3')
get_ipython().system('pip install scikit-learn==1.6.0')
get_ipython().system('pip install matplotlib==3.9.3')
get_ipython().system('pip install seaborn==0.13.2')


# ## Import the required libraries
# 

# In[2]:


import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# ### Load the Breast Cancer data set
# 

# In[3]:


data = load_breast_cancer()
X, y = data.data, data.target
labels = data.target_names
feature_names = data.feature_names


# ### Print the description of the Breast Cancer data set
# 

# In[4]:


print(data.DESCR)


# In[5]:


# Print target names for making the predictions
print(data.target_names)


# ### Standardize the data
# 

# In[7]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ### Add some noise
# It's to simulate random measurement error.

# In[8]:


# Add Gaussian noise to the data set
np.random.seed(42)  # For reproducibility
noise_factor = 0.5 # Adjust this to control the amount of noise
X_noisy = X_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X.shape)

# Load the original and noisy data sets into a DataFrame for comparison and visualization
df = pd.DataFrame(X_scaled, columns=feature_names)
df_noisy = pd.DataFrame(X_noisy, columns=feature_names)


# In[9]:


# Display the first few rows of the standardized original and noisy data sets for comparison
print("Original Data (First 5 rows):")
df.head()


# In[10]:


print("\nNoisy Data (First 5 rows):")
df_noisy.head()


# ### Visualizing the noise content. 
# You can get a good idea of how much noise there is in the features by comparing values in the previous tables. You can also visualize the differences in several ways.
# Let's begin by plotting the histograms of one of the features with and without noise for comparison.
# #### Histograms

# In[11]:


plt.figure(figsize=(12, 6))

# Original Feature Distribution (Noise-Free)
plt.subplot(1, 2, 1)
plt.hist(df[feature_names[5]], bins=20, alpha=0.7, color='blue', label='Original')
plt.title('Original Feature Distribution')
plt.xlabel(feature_names[5])
plt.ylabel('Frequency')

# Noisy Feature Distribution
plt.subplot(1, 2, 2)
plt.hist(df_noisy[feature_names[5]], bins=20, alpha=0.7, color='red', label='Noisy') 
plt.title('Noisy Feature Distribution')
plt.xlabel(feature_names[5])  
plt.ylabel('Frequency')

plt.tight_layout()  # Ensures proper spacing between subplots
plt.show()


# The noise-free histogram is skewed to the left and appears to a log-normal distribution, while the noisy histogram is less skewed, tending toward a normal distribution.
# 

# #### Plots
# You can also plot the two features together to get a sense of their differences.
# 

# In[13]:


plt.figure(figsize=(12, 6))
plt.plot(df[feature_names[5]], label='Original',lw=3)
plt.plot(df_noisy[feature_names[5]], '--',label='Noisy',)
plt.title('Scaled Feature Comparison with and without Noise')
plt.xlabel(feature_names[5])
plt.legend()
plt.tight_layout()
plt.show()


# #### Scatterplot
# Finally, you can compare the two features ussing a scatterplot. This gives you an excellent idea of how well the two features are correlated.
# 

# In[14]:


plt.figure(figsize=(12, 6))
plt.scatter(df[feature_names[5]], df_noisy[feature_names[5]],lw=5)
plt.title('Scaled feature comparison with and without noise')
plt.xlabel('Original Feature')
plt.ylabel('Noisy Feature')
plt.tight_layout()
plt.show()


# ### Split the data, and fit the KNN and SVM models to the noisy training data
# 

# In[15]:


# Split the data set into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_noisy, y, test_size=0.3, random_state=42)

# Initialize the models
knn = KNeighborsClassifier(n_neighbors=5)
svm = SVC(kernel='linear', C=1, random_state=42)

# Fit the models to the training data
knn.fit(X_train, y_train)
svm.fit(X_train, y_train)


# ### Evaluate the models
# 

# In[16]:


y_pred_knn = knn.predict(X_test)
y_pred_svm = svm.predict(X_test)


# In[17]:


# Print results
print(f"KNN Testing Accuracy: {accuracy_score(y_test, y_pred_knn):.3f}")
print(f"SVM Testing Accuracy: {accuracy_score(y_test, y_pred_svm):.3f}")

print("\nKNN Testing Data Classification Report:")
print(classification_report(y_test, y_pred_knn))

print("\nSVM Testing Data Classification Report:")
print(classification_report(y_test, y_pred_svm))


# #### Plot the confusion matrices
# 

# In[18]:


conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(conf_matrix_knn, annot=True, cmap='Blues', fmt='d', ax=axes[0],
            xticklabels=labels, yticklabels=labels)

axes[0].set_title('KNN Testing Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(conf_matrix_svm, annot=True, cmap='Blues', fmt='d', ax=axes[1],
            xticklabels=labels, yticklabels=labels)
axes[1].set_title('SVM Testing Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()


# By convention, a positive test for malignancy means a diagnosis of a mass being malignant. Thus, a benign prediction is a negative prediction. The worse-case scenario then is a false negative prediction, where the test incorrectly predicts that the mass is benign.
# For the KNN model, the number of false negatives is 7, while for the SVM model the count is 2. We can say that the SVM model has a higher prediction sensitivity than the KNN model does.

# SVM outperformed KNN in terms of precision, recall, and F1-score for both for the individual classes and their overall averages. This indicates that SVM is a stronger classifier. Although KNN performed quite well with an accuracy of 94%, SVM has better ability to correctly classify both malignant and beinign cases, with fewer errors. 

# ### Obtain the prediction results using the training data.

# In[19]:


y_pred_train_knn = knn.predict(X_train)
y_pred_train_svm = svm.predict(X_train)


# In[20]:


# Evaluate the models on the training data
print(f"KNN Training Accuracy: {accuracy_score(y_train, y_pred_train_knn):.3f}")
print(f"SVM Training Accuracy: {accuracy_score(y_train, y_pred_train_svm):.3f}")

print("\nKNN Training Classification Report:")
print(classification_report(y_train, y_pred_train_knn))

print("\nSVM Training Classification Report:")
print(classification_report(y_train, y_pred_train_svm))


# ### Plot the confusion matrices for the training data

# In[21]:


# Plot the confusion matrices
conf_matrix_knn = confusion_matrix(y_train, y_pred_train_knn)
conf_matrix_svm = confusion_matrix(y_train, y_pred_train_svm)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(conf_matrix_knn, annot=True, cmap='Blues', fmt='d', ax=axes[0],
            xticklabels=labels, yticklabels=labels)

axes[0].set_title('KNN Training Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(conf_matrix_svm, annot=True, cmap='Blues', fmt='d', ax=axes[1],
            xticklabels=labels, yticklabels=labels)
axes[1].set_title('SVM Training Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()


# 
# | Model | Phase |  Accuracy |
# | ------------  | -------- | --------- |
# | KNN  | Train  | 95.5% |
# | KNN  | Test   | 93.6% |
# | SVM  | Train  | 97.2% |
# | SVM  | Test   | 97.1% |
# 
# For the SVM model, the training and testing accuracies are essentially the same at about 97%. This is ideal - the SVM model is likely not overfit.
# For the KNN model, however, the training accuracy is about 2% higher that the test accuracy, indicating there might be some overfitting.
# 
# In summary, the SVM model is both more convincing and has a higher accuracy than the KNN model. 

# In[ ]:




