#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[1]:


import os
import pandas as  pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from collections import Counter
import warnings; warnings.filterwarnings("ignore")
import missingno as msno
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

USE_CATEGORICAL = True
USE_CROSS_VALIDATION = True
USE_ITERATIVE_IMPUTER = True
USE_OVERSAMPLING = True
USE_QDA = False


# In[2]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Look at the data 🔍

# ### Train dataset

# In[3]:


train_df = pd.read_csv("/kaggle/input/tabular-playground-series-aug-2022/train.csv")
print("train_df shape:", train_df.shape)
print(f"Amount of failures {train_df['failure'].sum()} of {train_df.shape[0]}")
train_df.head()


# ### Test dataset

# In[4]:


test_df = pd.read_csv("/kaggle/input/tabular-playground-series-aug-2022/test.csv")
print("test_df shape:", test_df.shape)
test_df.head()


# ### Show NaN values

# In[5]:


msno.matrix(train_df,labels=[train_df.columns],figsize=(30,16),fontsize=12)


# Great correlation representation by [desalegngeb](https://www.kaggle.com/code/desalegngeb/tps08-logisticregression-and-some-fe):

# In[6]:


corr = train_df.drop('id', axis=1).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.color_palette("rainbow", as_cmap=True)
f, ax = plt.subplots(figsize=(12, 12), facecolor='#EAECEE')
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1., center=0, annot=False,
           square=True, linewidths=.5, cbar_kws={"shrink": 0.75})


# In[7]:


train_df['attribute_2*3'] = train_df['attribute_2'] * train_df['attribute_3']
test_df['attribute_2*3'] = test_df['attribute_2'] * test_df['attribute_3']


# # Firstly select only the numeric features 📊

# In[8]:


print("Different output classes:", *train_df["failure"].unique())
descriptive_features = []
for i in range(18):
    descriptive_features.append("measurement_"+str(i))
descriptive_features.append("loading")
descriptive_features.append("attribute_2")
descriptive_features.append("attribute_3")
print("descriptive_features:", *descriptive_features)


# # Add categorical features 📖

# #### Categorical data is firstly not used, because it resulted in the end in the worse score. We'll put it in the drawer for now🤷‍♂️

# In[9]:


if USE_CATEGORICAL:
    categorical_descriptive_features = ["product_code", "attribute_0", "attribute_1"]
    descriptive_features.extend(categorical_descriptive_features)
    descriptive_features


# # Create train & test data 🗂️

# In[10]:


train_data = train_df[descriptive_features]
train_labels = train_df["failure"]
test_data = test_df[descriptive_features]


# In[11]:


if USE_CATEGORICAL:
    train_data = pd.get_dummies(train_data, columns = categorical_descriptive_features)
    test_data = pd.get_dummies(test_data, columns = categorical_descriptive_features)


# In[12]:


print(train_data.shape)
print(test_data.shape)


# #### When CATEGORICAL_DATA is used shapes are different, because of the one hot encoding of the categorical features🤔

# In[13]:


print("Columns in train_data", train_data.columns)
print("Columns in test_data", test_data.columns)


# #### <span style="color:red">Only relevant when using categorical data:</span>
# #### *product_code* is not matching as well as *attribute_1_material_8* (**train_df**) with *attribute_1_material_7* (**test_df**)
# #### That's why we just drop them 😂

# In[14]:


if USE_CATEGORICAL:
    train_columns_to_drop = ["product_code_A", "product_code_B", "product_code_C", "product_code_D", "product_code_E", "attribute_1_material_8"]
    train_data = train_data.drop(columns = train_columns_to_drop)
    test_columns_to_drop = ["product_code_F", "product_code_G", "product_code_H", "product_code_I", "attribute_1_material_7"]
    test_data = test_data.drop(columns = test_columns_to_drop)
    print("All columns are equal: ", (train_data.columns == test_data.columns).all())


# # Handle the NaN data 🧹

# In[15]:


if USE_ITERATIVE_IMPUTER:
    multi_imp = IterativeImputer(max_iter = 9, random_state = 42, verbose = 0, skip_complete = True, n_nearest_features = 10, tol = 0.001)
    multi_imp.fit(train_data)
    transformed_values_train = multi_imp.transform(train_data)
    transformed_values_test = multi_imp.transform(test_data)
else: #SimpleImputer
    train_imputer = SimpleImputer(missing_values=np.NaN, strategy = "mean")
    transformed_values_train = train_imputer.fit_transform(train_data)
    print("Train NaN values: ", np.isnan(transformed_values_train).sum())
    
    test_imputer = SimpleImputer(missing_values=np.NaN, strategy = "mean")
    transformed_values_test = test_imputer.fit_transform(test_data)
    print("Test NaN values: ",np.isnan(transformed_values_test).sum())


# # Normalizing the data ⚖️

# In[16]:


scaler = preprocessing.StandardScaler()
scaled_train_data = scaler.fit_transform(transformed_values_train)
scaled_test_data = scaler.fit_transform(transformed_values_test)


# # Dealing with inbalanced dataset

# In[17]:


print(Counter(train_labels))
plt.subplots(figsize=(8, 8))
plt.pie(train_labels.value_counts(), startangle=90, wedgeprops={'width':0.3})
plt.text(0, 0, f"{train_labels.value_counts()[0] / train_labels.count() * 100:.2f}%", ha='center', va='center', fontweight='bold', fontsize=42)
plt.legend(train_labels.value_counts().index, ncol=2, loc='lower center', fontsize=16)
plt.show()


# In[18]:


if USE_OVERSAMPLING:
    oversample = SMOTE()
    scaled_train_data, train_labels = oversample.fit_resample(scaled_train_data, train_labels)
    scaler = preprocessing.StandardScaler()
    scaled_train_data = scaler.fit_transform(scaled_train_data)


# #### View on the data after oversampling

# In[19]:


if USE_OVERSAMPLING:
    print(Counter(train_labels))
    plt.subplots(figsize=(8, 8))
    plt.pie(train_labels.value_counts(), startangle=90, wedgeprops={'width':0.3})
    plt.text(0, 0, f"{train_labels.value_counts()[0] / train_labels.count() * 100:.2f}%", ha='center', va='center', fontweight='bold', fontsize=42)
    plt.legend(train_labels.value_counts().index, ncol=2, loc='lower center', fontsize=16)
    plt.show()


# # Train simple LogisticRegression classifier using GridSearch for cross validation🦾

# In[20]:


train_X, val_X, train_Y, val_Y = train_test_split(scaled_train_data, train_labels, test_size=0.05, random_state=42, stratify = train_labels)


# ### Logistic train

# In[21]:


if USE_CROSS_VALIDATION:
    param_grid = {'C': [0.0001, 0.001, 0.01, 0.05 ,0.1, 0.5, 1],
                 'penalty':['l2','l1'],
                 'solver': ['lbfgs', 'sag', 'newton-cg', 'liblinear' ]}  
    classifier = GridSearchCV(LogisticRegression(max_iter = 200), param_grid, cv = 5, verbose = 3) 
    classifier.fit(train_X, train_Y)
    print(classifier.best_params_)
else:
    classifier = LogisticRegression(max_iter = 200)
    classifier.fit(train_X, train_Y)


# ### QDA train

# In[22]:


if USE_QDA:
    if USE_CROSS_VALIDATION:
        param_grid = {'reg_param': np.arange(0,1,0.1)}  
        QDA = GridSearchCV(QuadraticDiscriminantAnalysis(), param_grid, cv = 5, verbose = 3) 
        QDA.fit(train_X, train_Y)
        print(QDA.best_params_)
    else:
        QDA = QuadraticDiscriminantAnalysis()
        QDA.fit(train_X, train_Y)


# # Predicting and analysing🧑‍🏫

# ### Logistic

# In[23]:


predicted = classifier.predict(val_X)
print(classification_report(val_Y, predicted))


# ### QDA

# In[24]:


if USE_QDA:
    predicted_QDA = QDA.predict(val_X)
    print(classification_report(val_Y, predicted_QDA))


# ### Predict result

# In[25]:


pred_y = classifier.predict_proba(scaled_test_data)[:,1]


# In[26]:


if USE_QDA:
    pred_y_QDA = QDA.predict_proba(scaled_test_data)[:,1]


# In[27]:


submission = pd.read_csv("/kaggle/input/tabular-playground-series-aug-2022/sample_submission.csv")
if USE_QDA:
    submission.failure = (pred_y + pred_y_QDA) / 2
else:
    submission.failure = pred_y
submission


# # Submitting 🚀

# In[28]:


submission.to_csv("submission.csv", index=False)

