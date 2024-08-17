#!/usr/bin/env python
# coding: utf-8

# # Process
# 1. Import Packages
# 2. Import Data - read
# 3. Check Data - info, describe, data type
# 5. Partition Data into - Get Num, Cat Col names
# 6. Partition into (X, y)
# 7. Preprocessing - Identify, Treat (both Num, Cat - missing or outlier or scaling/encoding) 
# 8. Build Model - Hyper tune
# 9. Predict
# 10. Performance
# 11. Export Predictions, Feature Importance, Performance
# 
# ### Additional
# - Feature Selection
# - Feature Reduction
# - Feature Engineering

# # Import Package

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


# # Import Data

# In[2]:


df = pd.read_csv('/kaggle/input/bank-marketing-uci/bank.csv', sep = ';')


# # Check Data

# In[3]:


df.info()


# In[4]:


df.describe()


# # Define numerical and categorical columns

# In[5]:


num_cols = df.select_dtypes(include=['int64', 'float64']).columns
cat_cols = df.select_dtypes(include=['object']).columns
cat_cols = cat_cols.drop('y')


# In[6]:


num_cols


# In[7]:


cat_cols


# # Convert the binary output variable to numeric

# In[8]:


df['y'] = df['y'].map({'yes': 1, 'no': 0})


# # Preprocessing

# In[9]:


num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])


# In[10]:


cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])


# In[11]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)])


# # Combine preprocessing with classifier

# In[12]:


model = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier())])


# # Define target variable and features

# In[13]:


X = df.drop('y', axis=1)
y = df['y']


# # Splitting the dataset into training set and test set

# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# # Hyperparameter tuning

# In[15]:


param_grid = { 
    'classifier__n_estimators': [200, 500],
    'classifier__max_features': ['sqrt', 'log2'],
    'classifier__max_depth' : [4,5,6,7,8],
    'classifier__criterion' :['gini', 'entropy']
}


# In[16]:


CV_model = GridSearchCV(model, param_grid, cv= 5)
CV_model.fit(X_train, y_train)


# In[17]:


print(CV_model.best_params_)


# # Predict

# In[18]:


y_pred = CV_model.predict(X_test)


# # Performance

# In[19]:


print(classification_report(y_test, y_pred))


# # Exporting Predictions, Performance

# In[20]:


np.savetxt("predictions.csv", y_pred, delimiter=",")
with open("performance.txt", 'w') as f:
    f.write(classification_report(y_test, y_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




