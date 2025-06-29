#!/usr/bin/env python
# coding: utf-8

# # K-means Clustering
# 

# In[1]:


get_ipython().system('pip install numpy')
get_ipython().system('pip install pandas')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install scikit-learn')
get_ipython().system('pip install plotly')


# In[2]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets import make_blobs 
from sklearn.preprocessing import StandardScaler
import plotly.express as px

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# <h2 id="random_generated_dataset">K-Means on a synthetic data set</h2>
# 

# First, we need to set a random seed. Use <b>numpy's random.seed()</b> function, where the seed will be set to <b>0</b>.
# 

# In[3]:


np.random.seed(0)


# Next, we will be making random clusters of points by using the `make_blobs` class. The `make_blobs` class can take in many inputs, but we will be using these specific ones. <br> <br>
# <b> <u> Input </u> </b>
# <ul>
#     <li> <b>n_samples</b>: The total number of points equally divided among clusters. </li>
#     <ul> <li> Value will be: 5000 </li> </ul>
#     <li> <b> centres </b>: The number of centres to generate, or the fixed centre locations. </li>
#     <ul> <li> Value will be: [[4, 4], [-2, -1], [2, -3],[1,1]] </li> </ul>
#     <li> <b>cluster_std</b>: The standard deviation of the clusters. </li>
#     <ul> <li> Value will be: 0.9 </li> </ul>
# </ul>
# <br>
# <b> <u> Output </u> </b>
# <ul>
#     <li> <b>X</b>: Array of shape [n_samples, n_features]. (Feature Matrix)</li>
#     <ul> <li> The generated samples. </li> </ul> 
#     <li> <b>y</b>: Array of shape [n_samples]. (Response Vector)</li>
#     <ul> <li> The integer labels for cluster membership of each sample. </li> </ul>
# </ul>
# 

# In[4]:


X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)


# In[5]:


# Display scatter plot
plt.scatter(X[:, 0], X[:, 1], marker='.',alpha=0.3,ec='k',s=80)


#  #### Let's set up our k-means Clustering.

# The KMeans class has many parameters that can be used, but we will be using these three:
# 
# - `init`: Initialization method of the centroids.
#   - Value will be: `k-means++` 
#   - `k-means++`: Selects initial cluster centres for k-means clustering in a smart way to speed up convergence.
# - `n_clusters`: The number of clusters to form as well as the number of centroids to generate. 
#   -  Value will be: 4 (since we have 4 centres) 
# - `n_init`: Number of times the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia. 
#    - Value will be: 12  

# In[6]:


k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)


# In[7]:


# Fit kmeans model
k_means.fit(X)


# In[8]:


# Get the label for each point in the model using the k_means.labels_ attribute
k_means_labels = k_means.labels_
k_means_labels


# In[9]:


# Get the coordinates of the cluster centers using k_means.cluster_centers_
k_means_cluster_centers = k_means.cluster_centers_
k_means_cluster_centers


# <h2 id="creating_visual_plot">Creating the Visual Plot</h2>
# 

# In[13]:


# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(6, 4))

# Colors uses a color map, which will produce an array of colors based on
# the number of labels there are. We use set(k_means_labels) to get the unique labels.

colors = plt.cm.tab10(np.linspace(0, 1, len(set(k_means_labels))))

# Create a plot
ax = fig.add_subplot(1, 1, 1)

# For loop that plots the data points and centroids.
# k will range from 0-3, which will match the possible clusters that each
# data point is in.
for k, col in zip(range(len([[4, 4], [-2, -1], [2, -3], [1, 1]])), colors):

    # Create a list of all data points, where the data points that are 
    # in the cluster are labeled as true, else they are labeled as false.
    
    my_members = (k_means_labels == k)

    # Define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]

    # Plots the datapoints with color col.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.',ms=10)

    # Plots the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

# Title of the plot
ax.set_title('KMeans')

# Remove x-axis ticks
ax.set_xticks(())

# Remove y-axis ticks
ax.set_yticks(())
plt.show()


# ## Trying with  a number of cluster  k=3.

# In[14]:


k_means3 = KMeans(init="k-means++", n_clusters=3, n_init=12)
k_means3.fit(X)
fig = plt.figure(figsize=(6, 4))
colors = plt.cm.tab10(np.linspace(0, 1, len(set(k_means3.labels_))))
ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(len(k_means3.cluster_centers_)), colors):
    my_members = (k_means3.labels_ == k)
    cluster_center = k_means3.cluster_centers_[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.',ms=10)
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)
plt.show()


# ### Trying with a number of cluster k=5

# In[16]:


k_means5 = KMeans(init="k-means++", n_clusters=5, n_init=12)
k_means5.fit(X)
fig = plt.figure(figsize=(6, 4))
colors = plt.cm.tab10(np.linspace(0, 1, len(set(k_means5.labels_))))
ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(len(k_means5.cluster_centers_)), colors):
    my_members = (k_means5.labels_ == k)
    cluster_center = k_means5.cluster_centers_[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.',ms=10)
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)
plt.show()


# For k=3, the value of within-cluster sum of squares will be higher that that for k=4, since the points from different natural clusters are being grouped together, leading to underfitting of the k-means model. For k=5, the value of will be lesser than that for k=4, since the points are distributed into mode clusters than needed, leading to over-fitting of the k-means model.

# <h1 id="customer_segmentation_K_means">Customer Segmentation with k-means</h1>
# 

# ### Load data from CSV file  

# https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/Cust_Segmentation.csv

# In[18]:


import os
# Print current working directory to load the csv file after downloading it on device
print(os.getcwd())


# In[19]:


cust_df = pd.read_csv("/home/af2ea8f5-1f9a-4091-9c13-db2b59ea1801/Skill Network/Machine Learning/Cust_Segmentation.csv")

cust_df.head()


# As you can see, `Address` in this dataset is a categorical variable. The k-means algorithm doesn't work directly with categorical variables because the Euclidean distance function isn't meaningful for them. You could one-hot encode the categorical feature, but for illustration purposes let's run k-means clustering without it.
# 

# In[20]:


cust_df = cust_df.drop('Address', axis=1)


# In[21]:


# Drop NaNs from the dataframe
cust_df = cust_df.dropna()
cust_df.info()


# After dropping NaNs we still have 700 rows out of the original 849. Let's proceed with this smaller data set.
# 

# #### Normalizing over the standard deviation
# 

# In[22]:


X = cust_df.values[:,1:] 
Clus_dataSet = StandardScaler().fit_transform(X)


# <h2 id="modeling">Modeling</h2>
# 

# Let's apply k-means to the data set. 
# 

# #### Cluster the data with k=3.

# In[23]:


clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_


# <h2 id="insights">Insights</h2>
# 
# We assign the k-means cluster labels to each row in the dataframe.
# 

# In[24]:


cust_df["Clus_km"] = labels


# We can easily check the centroid values by averaging the features in each cluster. These values indicate the central point of the cluster from the vantage point of the field in question.
# 

# In[26]:


cust_df.groupby('Clus_km').mean


# Now, let's look at the distribution of customers based on their education, age and income.
# 

# In[27]:


area = np.pi * ( X[:, 1])**2  
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(float), cmap='tab10', ec='k',alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)
plt.show()


# We can also see this distribution in 3 dimensions for better understanding. Here, the education parameter will represent the third axis instead of the marker size.

# In[28]:


# Create interactive 3D scatter plot
fig = px.scatter_3d(X, x=1, y=0, z=3, opacity=0.7, color=labels.astype(float))

fig.update_traces(marker=dict(size=5, line=dict(width=.25)), showlegend=False)
fig.update_layout(coloraxis_showscale=False, width=1000, height=800, scene=dict(
        xaxis=dict(title='Education'),
        yaxis=dict(title='Age'),
        zaxis=dict(title='Income')
    ))  # Remove color bar, resize plot

fig.show()


# In[ ]:




