#!/usr/bin/env python
# coding: utf-8

# <div style="border-width:5px; border-style:double; border-radius:10px; border-top-color:black; border-bottom-color:darkblue; padding:15px; box-shadow:3px 3px 10px rgba(0,0,0,0.3);background:linear-gradient(to right, darkblue, orange); border-right-color:darkblue; border-left-color:darkblue"> 
# <h1 style="text-align:center; font-weight:bold; font-size: 45px; color:white"> Titanic Classifier</h1>
# </div>

# ---

# ## Project Description
# 
# This project will use cross validation and a hyperparameter grid search to optimize machine learning pipeline. I will use the Titanic Survival Dataset to build a classification model to predict whether a passenger survived the sinking of the Titanic, based on attributes of each passenger in the data set.
# 
# I'll start with building a Random Forest Classifier, then modify the pipeline to replace it with a Logistic Regression estimator instead. 

# ### Install the required libraries
# 

# In[1]:


get_ipython().system('pip install numpy')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install pandas')
get_ipython().system('pip install scikit-learn')
get_ipython().system('pip install seaborn')


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# ### Description of the Titanic Dataset

# | Variable   |	Definition   |
#  |:------|:--------------------------------|
#  |survived | survived? 0 = No, 1 = yes  |
#  |pclass | Ticket class (int)  |
#  |sex	 |sex |
#  |age	 | age in years  |
#  |sibsp  |	# of siblings / spouses aboard the Titanic |
#  |parch  |	# of parents / children aboard the Titanic |
#  |fare   |	Passenger fare   || Variable   |	Definition   |
#  |:------|:--------------------------------|
#  |survived | survived? 0 = No, 1 = yes  |
#  |pclass | Ticket class (int)  |
#  |sex	 |sex |
#  |age	 | age in years  |
#  |sibsp  |	# of siblings / spouses aboard the Titanic |
#  |parch  |	# of parents / children aboard the Titanic |
#  |fare   |	Passenger fare   |
#  |embarked | Port of Embarkation |
#  |class  |Ticket class (obj)   |
#  |who    | man, woman, or child  |
#  |adult_male | True/False |
#  |alive  | yes/no  |
#  |alone  | yes/no  |
#  |embarked | Port of Embarkation |
#  |class  |Ticket class (obj)   |
#  |who    | man, woman, or child  |
#  |adult_male | True/False |
#  |alive  | yes/no  |
#  |alone  | yes/no  |
# 

# ## Loading the Titanic Dataset with Seaborn
# 

# In[4]:


titanic = sns.load_dataset('titanic')
titanic.head()


# Let's check for missing data. Handling missing data well can boost the classifier’s performance.

# In[6]:


# Checking for missing data
titanic.isnull().sum()


# These will be handled using the `SimpleImputer` library later.

# ## Selecting Relevant Features and the Target
# 

# #### Features to drop
# `deck` has a lot of missing values so we'll drop it. `age` has quite a few missing values as well. It appears that `embarked` and `embark_town` are not relevant, so we'll drop them as well. It's unclear what `alive` refers to, so we'll ignore it.
# #### Target
# `survived` will be our target class variable.
# 

# In[9]:


features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'class', 'who', 'adult_male', 'alone']
target = 'survived'

X = titanic[features]
y = titanic[target]


# ### How balanced are the classes?
# 

# In[10]:


y.value_counts()


# In[13]:


# Calculating the ratio of survived passengers
tot = 549 + 342
ratio = 342/ tot
print('Ratio of Survived Passengers:\n', round(ratio,4))


# About 38% of the passengers in the data set survived.  
# Because of this slight imbalance, we should `stratify the data` when performing train/test split and for cross-validation.

# ---

# # Model Preprocessing

# In[15]:


# Splitting the dataset and stratifying y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[21]:


# Defining preprocessing transformers for numerical and categorical features
numerical_features = X_train.select_dtypes(include=['number']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()


# In[17]:


# Defining separate preprocessing pipelines for both feature types
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


# In[18]:


# Combining the transformers into a single column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])


# In[19]:


# Creating a model pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])


# In[20]:


# Defining a parameter grid
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}


# In[22]:


# Grid search cross-validation method
cv = StratifiedKFold(n_splits=5, shuffle=True)


# In[23]:


# Fitting the model
model = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring='accuracy', verbose=2)
model.fit(X_train, y_train)


# In[24]:


# Model prediction
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


# ### Confusion Matrix

# In[26]:


# Generating the confusion matrix 
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
plt.title('Titanic Classification Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()


# ---

# ## Feature Importances

# In[27]:


model.best_estimator_['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)


# In[28]:


feature_importances = model.best_estimator_['classifier'].feature_importances_

# Combine the numerical and one-hot encoded categorical feature names
feature_names = numerical_features + list(model.best_estimator_['preprocessor']
                                        .named_transformers_['cat']
                                        .named_steps['onehot']
                                        .get_feature_names_out(categorical_features))


# In[32]:


importance_df = pd.DataFrame({'Feature': feature_names,
                              'Importance': feature_importances
                             }).sort_values(by='Importance', ascending=False)

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='lightgreen')
plt.gca().invert_yaxis() 
plt.title('Most Important Features in Predicting Passenger Survival')
plt.xlabel('Importance Score')
plt.yticks(rotation=45)
plt.show()

# Print test score 
test_score = model.score(X_test, y_test)
print(f"\nTest set accuracy: {test_score:.2%}")


# The test set accuracy is somewhat satisfactory. However,regarding the feature importances, it's crucially important to realize that there is most likely plenty of dependence amongst these variables, and a more detailed modeling approach including correlation analysis is required to draw proper conclusions. For example, no doubt there is significant information shared by the variables `age`, `sex_male`, and `who_man`.

# ---

# # Logistic Regression

# In[33]:


# Replacing RandomForestClassifier with LogisticRegression
pipeline.set_params(classifier=LogisticRegression(random_state=42))

# Updating the model's estimator to use the new pipeline
model.estimator = pipeline

# New grid with Logistic Regression parameters
param_grid = {
    # 'classifier__n_estimators': [50, 100],
    # 'classifier__max_depth': [None, 10, 20],
    # 'classifier__min_samples_split': [2, 5],
    'classifier__solver' : ['liblinear'],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__class_weight' : [None, 'balanced']
}

model.param_grid = param_grid

# Fitting the updated pipeline with Logistic Regression
model.fit(X_train, y_train)

# new predictions
y_pred = model.predict(X_test)


# In[35]:


# Displaying the clasification report for the new model
print(classification_report(y_test, y_pred))


# All of the scores are slightly better for logistic regression than for random forest classification, although the differences are not significant. 

# In[36]:


# Display the confusion matrix for the new model 
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
plt.title('Titanic Classification Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()


# The results show a slight improvement, with one more true positive and one more true negative.

# ### Extracting the logistic regression feature coefficients and plotting their magnitude in a bar chart.
# 

# In[37]:


coefficients = model.best_estimator_.named_steps['classifier'].coef_[0]

# Combine numerical and categorical feature names
numerical_feature_names = numerical_features
categorical_feature_names = (model.best_estimator_.named_steps['preprocessor']
                                     .named_transformers_['cat']
                                     .named_steps['onehot']
                                     .get_feature_names_out(categorical_features)
                            )
feature_names = numerical_feature_names + list(categorical_feature_names)


# In[39]:


# Create a DataFrame for the coefficients
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
}).sort_values(by='Coefficient', ascending=False, key=abs)  # Sort by absolute values

# Plotting
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Coefficient'].abs(), color='skyblue')
plt.gca().invert_yaxis()
plt.title('Feature Coefficient Magnitudes for Logistic Regression Model')
plt.xlabel('Coefficient Magnitude')
plt.show()

# Print test score
test_score = model.best_estimator_.score(X_test, y_test)
print(f"\nTest set accuracy: {test_score:.2%}")


# Although the performances of the two models are virtually identical, the features that are important to the two models are very different. This suggests there must be more work to do to better grasp the actual feature importances. As mentioned above, it's crucially important to realize that there is most likely plenty of dependence amongst these variables, and a more detailed modeling approach including correlation analysis is required to draw proper conclusions. For example, there is significant information implied between the variables `who_man`, `who_woman`, and `who_child`, because if a person is neither a man nor a woman, then they muct be a child.
