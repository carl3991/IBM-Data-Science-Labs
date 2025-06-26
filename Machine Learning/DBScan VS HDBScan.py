#!/usr/bin/env python
# coding: utf-8

# # Comparing DBSCAN and HDBSCAN Clustering 
# 

# In[1]:


get_ipython().system('pip install numpy==2.2.0')
get_ipython().system('pip install pandas==2.2.3')
get_ipython().system('pip install scikit-learn==1.6.0')
get_ipython().system('pip install matplotlib==3.9.3')
get_ipython().system('pip install hdbscan==0.8.40')
get_ipython().system('pip install geopandas==1.0.1')
get_ipython().system('pip install contextily==1.6.2')
get_ipython().system('pip install shapely==2.0.6')


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import hdbscan
from sklearn.preprocessing import StandardScaler

# geographical tools
import geopandas as gpd  # pandas dataframe-like geodataframes for geographical data
import contextily as ctx  # used for obtianing a basemap of Canada
from shapely.geometry import Point

import warnings
warnings.filterwarnings('ignore')


# ## Download the Canada map for reference
# 

# https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/YcUk-ytgrPkmvZAh5bf7zA/Canada.zip
# 
# 

# In[5]:


import os
print(os.listdir('.'))


# In[8]:


print(os.getcwd())


# In[10]:


import rasterio
tif_path = 'Canada.tif'

with rasterio.open(tif_path) as src:
    print("CRS:", src.crs)


# ## Include a plotting function
# 

# In[11]:


# Write a function that plots clustered locations and overlays them on a basemap.

def plot_clustered_locations(df,  title='Museums Clustered by Proximity'):
    """
    Plots clustered locations and overlays on a basemap.
    
    Parameters:
    - df: DataFrame containing 'Latitude', 'Longitude', and 'Cluster' columns
    - title: str, title of the plot
    """
    
    # Load the coordinates intto a GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']), crs="EPSG:4326")
    
    # Reproject to Web Mercator to align with basemap 
    gdf = gdf.to_crs(epsg=3857)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Separate non-noise, or clustered points from noise, or unclustered points
    non_noise = gdf[gdf['Cluster'] != -1]
    noise = gdf[gdf['Cluster'] == -1]
    
    # Plot noise points 
    noise.plot(ax=ax, color='k', markersize=30, ec='r', alpha=1, label='Noise')
    
    # Plot clustered points, colured by 'Cluster' number
    non_noise.plot(ax=ax, column='Cluster', cmap='tab10', markersize=30, ec='k', legend=False, alpha=0.6)
    
    # Add basemap of  Canada
    ctx.add_basemap(ax, source='./Canada.tif', zoom=4)
    
    # Format plot
    plt.title(title, )
    plt.xlabel('Longitude', )
    plt.ylabel('Latitude', )
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    
    # Show the plot
    plt.show()


# ## Explore the data and extract what you need from it
# 

# #### Start by loading the data set into a Pandas DataFrame and displaying the first few rows.
# 

# https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/r-maSj5Yegvw2sJraT15FA/ODCAF-v1-0.csv
# 

# In[13]:


df = pd.read_csv("/home/af2ea8f5-1f9a-4091-9c13-db2b59ea1801/Skill Network/Machine Learning/ODCAF_v1.0.csv", encoding="ISO-8859-1")


# In[14]:


df.head()


# In[15]:


df.isnull().sum()


# In[16]:


# Display the facility types and their counts.
df.ODCAF_Facility_Type.value_counts()


# In[17]:


# Filter the data to only include museums.
df = df[df.ODCAF_Facility_Type == 'museum']
df.ODCAF_Facility_Type.value_counts()


# In[18]:


# Select only the Latitude and Longitude features as inputs
df = df[['Latitude', 'Longitude']]
df.info()


# In[19]:


# Remove observations with no coordinates 
df = df[df.Latitude!='..']

# Convert to  coordinates float type
df[['Latitude','Longitude']] = df[['Latitude','Longitude']].astype('float')


# Build a DBSCAN model

# In[20]:


# In this case we know how to scale the coordinates. Using standardization would be an error becaues we aren't using the full range of the lat/lng coordinates.
# Since latitude has a range of +/- 90 degrees and longitude ranges from 0 to 360 degrees, the correct scaling is to double the longitude coordinates (or half the Latitudes)
coords_scaled = df.copy()
coords_scaled["Latitude"] = 2*coords_scaled["Latitude"]


# ### Apply DBSCAN with Euclidean distance to the scaled coordinates
# 

# In[21]:


# minimum number of samples needed to form a neighbourhood
min_samples=3 

# neighbourhood search radius
eps=1.0 

# distance measure 
metric='euclidean' 
dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(coords_scaled)


# ### Add cluster labels to the DataFrame
# 

# In[22]:


# Assign the cluster labels
df['Cluster'] = dbscan.fit_predict(coords_scaled) 

# Display the size of each cluster
df['Cluster'].value_counts()


# ### Plot the museums on a basemap of Canada, colored by cluster label.
# 

# In[23]:


plot_clustered_locations(df, title='Museums Clustered by Proximity')


# Here, the clusters are not uniformly dense. For example, the points are quite densely packed in a few regions but are relatively sparse in between. DBSCAN agglomerates neighboring clusters together when they are close enough.

# Let's see how a hierarchical density-based clustering algorithm such as like `HDBSCAN` performs.

# ## Build an HDBSCAN clustering model
# 

# #### Initialize an HDBSCAN model
# 

# In[24]:


min_samples=None
min_cluster_size=3
hdb = hdbscan.HDBSCAN(min_samples=min_samples, min_cluster_size=min_cluster_size, metric='euclidean') 


# ### Assign the cluster labels to your unscaled coordinate dataframe and display the counts of each cluster label.

# In[25]:


# Assign labels
df['Cluster'] = hdb.fit_predict(coords_scaled)  # Another way to assign the labels

# Display the size of each cluster
df['Cluster'].value_counts()


# Unlike the case for DBSCAN, clusters quite uniformly sized, although there is a quite lot of noise identified.

# ### Plot the hierarchically clustered museums on a basemap of Canada, colored by cluster label.

# In[26]:


plot_clustered_locations(df, title='Museums Hierarchically Clustered by Proximity')


# ### Remarks

# 1. `HDBSCAN` produces more compact, uniformly sized clusters, whereas `DBSCAN`tends to group dense neighboring regions into a few large clusters. You might notice that `DBSCAN` merged several adjacent museums into superclusters, while `HDBSCAN` split them up more deliberately.
# 
# 2. Initially it may look like HDBSCAN marked more points as noise (label -1), but because it uses a more nuanced approach to density, it's actually filtering out points that DBSCAN might've forced into clusters—possibly incorrectly. 
# 
# 3. HDBSCAN handles areas of varying point density. For instance, downtown Toronto or Vancouver might naturally have tighter, denser museum clusters, while rural areas are sparser. **HDBSCAN adapts**, whereas DBSCAN uses a fixed neighborhood radius, which isn’t always optimal across the full dataset.
