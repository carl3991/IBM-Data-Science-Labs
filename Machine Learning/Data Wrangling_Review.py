#!/usr/bin/env python
# coding: utf-8

# # Data Wrangling
# 

# ## Import libraries

# In[ ]:


#install specific version of libraries used in lab
#! mamba install pandas==1.3.3
#! mamba install numpy=1.21.2


# In[2]:


import pandas as pd
import matplotlib.pylab as plt


# Link to the automibile csv file...

# https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv

# In[3]:


import os
print(os.getcwd())


# In[4]:


auto_df = pd.read_csv('/home/af2ea8f5-1f9a-4091-9c13-db2b59ea1801/Skill Network/Machine Learning/auto.csv')
auto_df.head()


# No headers spotted. Let's create a list of headers for the dataset.

# In[5]:


headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]


# In[6]:


auto_df = pd.read_csv('/home/af2ea8f5-1f9a-4091-9c13-db2b59ea1801/Skill Network/Machine Learning/auto.csv', names = headers)
auto_df.head()


# Voila!!!

# # Identify and handle missing values
# 

# In[8]:


import numpy as np

# replace "?" to NaN
auto_df.replace("?", np.nan, inplace = True)
auto_df.head(5)


# In[9]:


missing_data = auto_df.isnull()
missing_data.head(5)


# <h4>Count missing values in each column</h4>
# <p>
# Using a for loop in Python, you can quickly figure out the number of missing values in each column. As mentioned above, "True" represents a missing value and "False" means the value is present in the data set.  In the body of the for loop the method ".value_counts()" counts the number of "True" values. 
# </p>
# 

# In[10]:


for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")    


# Based on the summary above, each column has 205 rows of data and seven of the columns containing missing data:
# <ol>
#     <li>"normalized-losses": 41 missing data</li>
#     <li>"num-of-doors": 2 missing data</li>
#     <li>"bore": 4 missing data</li>
#     <li>"stroke" : 4 missing data</li>
#     <li>"horsepower": 2 missing data</li>
#     <li>"peak-rpm": 2 missing data</li>
#     <li>"price": 4 missing data</li>
# </ol>
# 

# ### Alternative to find missing values.

# In[11]:


missing_values = auto_df.isnull().sum()
print('All Missing Values:\n',missing_values)


# ### Deal with missing data
# <b>How should you deal with missing data?</b>
# 
# <ol>
#     <li>Drop data<br>
#         a. Drop the whole row<br>
#         b. Drop the whole column
#     </li>
#     <li>Replace data<br>
#         a. Replace it by mean<br>
#         b. Replace it by frequency<br>
#         c. Replace it based on other functions
#     </li>
# </ol>
# 

# You should only drop whole columns if most entries in the column are empty. In the data set, none of the columns are empty enough to drop entirely.
# You have some freedom in choosing which method to replace data; however, some methods may seem more reasonable than others. Apply each method to different columns:
# 
# <b>Replace by mean:</b>
# <ul>
#     <li>"normalized-losses": 41 missing data, replace them with mean</li>
#     <li>"stroke": 4 missing data, replace them with mean</li>
#     <li>"bore": 4 missing data, replace them with mean</li>
#     <li>"horsepower": 2 missing data, replace them with mean</li>
#     <li>"peak-rpm": 2 missing data, replace them with mean</li>
# </ul>
# 
# <b>Replace by frequency:</b>
# <ul>
#     <li>"num-of-doors": 2 missing data, replace them with "four". 
#         <ul>
#             <li>Reason: 84% sedans are four doors. Since four doors is most frequent, it is most likely to occur</li>
#         </ul>
#     </li>
# </ul>
# 
# <b>Drop the whole row:</b>
# <ul>
#     <li>"price": 4 missing data, simply delete the whole row
#         <ul>
#             <li>Reason: You want to predict price. You cannot use any data entry without price data for prediction; therefore any row now without price data is not useful to you.</li>
#         </ul>
#     </li>
# </ul>
# 

# <h4>Calculate the mean value for the "normalized-losses" column </h4>
# 

# In[12]:


avg_norm_loss = auto_df["normalized-losses"].astype("float").mean(axis=0)
print("Average of normalized-losses:", avg_norm_loss)


# <h4>Replace "NaN" with mean value in "normalized-losses" column</h4>
# 

# In[19]:


auto_df['normalized_losses']  = auto_df["normalized-losses"].replace(np.nan, avg_norm_loss)


# <h4>Calculate the mean value for the "bore" column</h4>
# 

# In[16]:


avg_bore = auto_df['bore'].astype('float').mean(axis=0)
print("Average of bore:", avg_bore)


# <h4>Replace "NaN" with the mean value in the "bore" column</h4>
# 

# In[20]:


auto_df['bore'] = auto_df["bore"].replace(np.nan, avg_bore)


# In[21]:


#Calculate the mean vaule for "stroke" column
avg_stroke = auto_df["stroke"].astype("float").mean(axis = 0)
print("Average of stroke:", avg_stroke)

# replace NaN by mean value in "stroke" column
auto_df['stroke'] = auto_df["stroke"].replace(np.nan, avg_stroke)


# In[22]:


# Calculate the mean value for the "horsepower" column
avg_horsepower = auto_df['horsepower'].astype('float').mean(axis=0)
print("Average horsepower:", avg_horsepower)

# Replace "NaN" with the mean value in the "horsepower" column
auto_df['horsepower'] = auto_df['horsepower'].replace(np.nan, avg_horsepower)


# In[25]:


# Calculate the mean value for the "peak-rpm" column

avg_peakrpm=auto_df['peak-rpm'].astype('float').mean(axis=0)
print("Average peak rpm:", avg_peakrpm)

# Replace "NaN" with the mean value in the "peak-rm" column
auto_df['peak-rpm'] = auto_df['peak-rpm'].replace(np.nan, avg_peakrpm)


# To see which values are present in a particular column, we can use the ".value_counts()" method:

# In[28]:


auto_df['num-of-doors'].value_counts()


# You can see that four doors is the most common type. We can also use the ".idxmax()" method to calculate the most common type automatically:

# In[29]:


auto_df['num-of-doors'].value_counts().idxmax()


# The replacement procedure is very similar to what you have seen previously:

# In[32]:


#replace the missing 'num-of-doors' values by the most frequent 
auto_df['num-of-doors'] = auto_df["num-of-doors"].replace(np.nan, "four")


# Finally, drop all rows that do not have price data:

# In[34]:


# simply drop whole row with NaN in "price" column
auto_df.dropna(subset=["price"], axis=0, inplace=True)

# reset index, because we droped two rows
auto_df.reset_index(drop=True, inplace=True)


# In[35]:


auto_df.head()


# <h4>Let's list the data types for each column</h4>
# 

# In[37]:


auto_df.dtypes


# As you can see above, some columns are not of the correct data type. Numerical variables should have type 'float' or 'int', and variables with strings such as categories should have type 'object'. For example, the numerical values 'bore' and 'stroke' describe the engines, so you should expect them to be of the type 'float' or 'int'; however, they are shown as type 'object'.

# <h4>Convert data types to proper format</h4>
# 

# In[39]:


auto_df[["bore", "stroke"]] = auto_df[["bore", "stroke"]].astype("float")
auto_df[["normalized-losses"]] = auto_df[["normalized-losses"]].astype("int")
auto_df[["price"]] = auto_df[["price"]].astype("float")
auto_df[["peak-rpm"]] = auto_df[["peak-rpm"]].astype("float")


# In[40]:


auto_df.dtypes


# Finally obtained the cleansed data set with no missing values and with all data in its proper format.

# ## Data Standardization
# <p>
# We usually collect data from different agencies in different formats.
# (Data standardization is also a term for a particular type of data normalization where you subtract the mean and divide by the standard deviation.)
# </p>
#     
# <b>What is standardization?</b>
# <p>Standardization is the process of transforming data into a common format, allowing the researcher to make the meaningful comparison.
# </p>
# 
# <b>Example</b>
# <p>Transform mpg to L/100km:</p>
# <p>In the data set, the fuel consumption columns "city-mpg" and "highway-mpg" are represented by mpg (miles per gallon) unit.</p>
# <p>Let's apply <b>data transformation</b> to transform mpg into L/100km.</p>
# 

# In[54]:


# Convert mpg to L/100km by mathematical operation (235 divided by mpg)
auto_df['city-L/100km'] = 235/auto_df["city-mpg"]

# check your transformed data 
print(auto_df['city-L/100km'])


# ### Transform mpg to L/100km in the column of "highway-mpg" and change the name of column to "highway-L/100km

# In[55]:


# transform mpg to L/100km by mathematical operation (235 divided by mpg)
auto_df["highway-mpg"] = 235/auto_df["highway-mpg"]

# rename column name from "highway-mpg" to "highway-L/100km"
auto_df.rename(columns={'"highway-mpg"':'highway-L/100km'}, inplace=True)

# check your transformed data 
auto_df.head()


# ## Data Normalization
# 
# <b>Why normalization?</b>
# <p>Normalization is the process of transforming values of several variables into a similar range. Typical normalizations include 
# <ol>
#     <li>scaling the variable so the variable average is 0</li>
#     <li>scaling the variable so the variance is 1</li> 
#     <li>scaling the variable so the variable values range from 0 to 1</li>
# </ol>
# </p>
# 
# <b>Example</b>
# <p>To demonstrate normalization, say you want to scale the columns "length", "width" and "height".</p>
# <p><b>Target:</b> normalize those variables so their value ranges from 0 to 1</p>
# <p><b>Approach:</b> replace the original value by (original value)/(maximum value)</p>
# 

# In[56]:


# replace (original value) by (original value)/(maximum value)
auto_df['length'] = auto_df['length']/auto_df['length'].max()
auto_df['width'] = auto_df['width']/auto_df['width'].max()


# In[57]:


# Normalize 'height column'
auto_df['height'] = auto_df['height']/auto_df['height'].max() 

# show the scaled columns
auto_df[["length","width","height"]].head()


# ## Binning
# <b>Why binning?</b>
# <p>
#     Binning is a process of transforming continuous numerical variables into discrete categorical 'bins' for grouped analysis.
# </p>
# 
# <b>Example: </b>
# <p> "horsepower" column is a real valued variable ranging from 48 to 288 and it has 59 unique values. What if you only care about the price difference between cars with high horsepower, medium horsepower, and little horsepower (3 types)? We can rearrange them into three ‘bins' to simplify analysis.</p>

# In[58]:


auto_df["horsepower"]=auto_df["horsepower"].astype(int, copy=True)


# In[64]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot
plt.pyplot.hist(auto_df["horsepower"], color='lightgreen')

plt.pyplot.xlabel("Horsepower")
plt.pyplot.ylabel("Count")
plt.pyplot.title("Horsepower Bins")


# <p>Find 3 bins of equal size bandwidth by using Numpy's <code>linspace(start_value, end_value, numbers_generated</code> function.</p>
# <p>Since you want to include the minimum value of horsepower, set start_value = min(df["horsepower"]).</p>
# <p>Since you want to include the maximum value of horsepower, set end_value = max(df["horsepower"]).</p>
# <p>Since you are building 3 bins of equal length, you need 4 dividers, so numbers_generated = 4.</p>
# 

# Build a bin array with a minimum value to a maximum value by using the bandwidth calculated above. The values will determine when one bin ends and another begins.

# In[65]:


bins = np.linspace(min(auto_df["horsepower"]), max(auto_df["horsepower"]), 4)
bins


# In[66]:


# Set group names
group_names = ['Low', 'Medium', 'High']


# Apply the function "cut" to determine what each value of auto_df['horsepower'] belongs to.

# In[67]:


auto_df['horsepower-binned'] = pd.cut(auto_df['horsepower'], bins, labels=group_names, include_lowest=True )
auto_df[['horsepower','horsepower-binned']].head(20)


# See the number of vehicles in each bin:

# In[68]:


auto_df["horsepower-binned"].value_counts()


# In[70]:


# Plot the distribution of each bin
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot
pyplot.bar(group_names, auto_df["horsepower-binned"].value_counts(),color='darkblue')
plt.pyplot.xlabel("Horsepower")
plt.pyplot.ylabel("Count")
plt.pyplot.title("Horsepower Bins")


# Successfully narrowed down the intervals from 59 to 3!

# <h3>Bins Visualization</h3>

# In[72]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot

# draw historgram of attribute "horsepower" with bins = 3
plt.pyplot.hist(auto_df["horsepower"], bins = 3)
plt.pyplot.xlabel("Horsepower")
plt.pyplot.ylabel("Count")
plt.pyplot.title("Horsepower Bins")


# This shows the binning result for the attribute "horsepower".

# ## Indicator Variable
# <b>What is an indicator variable?</b>
# <p>
#     An indicator variable (or dummy variable) is a numerical variable used to label categories. They are called 'dummies' because the numbers themselves don't have inherent meaning. 
# </p>
# 
# <b>Why use indicator variables?</b>
# <p>
#     You use indicator variables so you can use categorical variables for regression analysis.
# </p>
# <b>Example</b>
# <p>
#     The column "fuel-type" has two unique values: "gas" or "diesel". Regression doesn't understand words, only numbers. To use this attribute in regression analysis, you can convert "fuel-type" to indicator variables.
# </p>
# 
# <p>
#     Use the Panda method 'get_dummies' to assign numerical values to different categories of fuel type. 
# </p>
# 

# In[77]:


auto_df.columns


# In[80]:


# Get the indicator variables 
dummy_variable_1 = pd.get_dummies(auto_df["fuel-type"])
dummy_variable_1.head()


# In[81]:


# Change the column names
dummy_variable_1.rename(columns={'gas':'fuel-type-gas', 'diesel':'fuel-type-diesel'}, inplace=True)
dummy_variable_1.head()


# In the data frame, column 'fuel-type' now has values for 'gas' and 'diesel' as 0s and 1s.

# In[82]:


# merge data frame "auto_df" and "dummy_variable_1" 
auto_df = pd.concat([auto_df, dummy_variable_1], axis=1)

# drop original column "fuel-type" from "df"
auto_df.drop("fuel-type", axis = 1, inplace=True)


# In[83]:


auto_df.head(3)


# The last two columns are now the indicator variable representation of the fuel-type variable. They're all 0s and 1s now.

# In[84]:


# create an indicator variable for the column "aspiration"
dummy_variable_2 = pd.get_dummies(auto_df['aspiration'])

# change column names for clarity
dummy_variable_2.rename(columns={'std':'aspiration-std', 'turbo': 'aspiration-turbo'}, inplace=True)

# show first 5 instances of data frame "dummy_variable_1"
dummy_variable_2.head()


# In[85]:


# merge the new dataframe to the original datafram
auto_dfdf = pd.concat([auto_df, dummy_variable_2], axis=1)

# drop original column "aspiration" from "df"
auto_df.drop('aspiration', axis = 1, inplace=True)


# In[86]:


auto_df.to_csv('clean_auto_df.csv')

