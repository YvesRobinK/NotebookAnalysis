#!/usr/bin/env python
# coding: utf-8

# # 🌐 Advanced Feature Expansion - 🌿  LightGBM
# 
# ## Introduction 🌟
# Welcome to this Jupyter notebook developed for the Open Problems – Single-Cell Perturbations to Predict how small molecules change gene expression in different cell types.
# ### Inspiration and Credits 🙌
# This notebook is inspired by the work of CW, available at [this Kaggle project](https://www.kaggle.com/code/chesterx/0-584-feature-augmentation-lightgbm?scriptVersionId=149238646). I extend my gratitude to CW for sharing their insights and code.
# 
# 🌟 Explore my profile and other public projects, and don't forget to share your feedback! 
# 👉 [Visit my Profile](https://www.kaggle.com/zulqarnainali) 👈
# 
# 🙏 Thank you for taking the time to review my work, and please give it a thumbs-up if you found it valuable! 👍
# 
# ## Purpose 🎯
# The primary purpose of this notebook is to:
# - Load and preprocess the competition data 📁
# - Engineer relevant features for model training 🏋️‍♂️
# - Train predictive models to make target variable predictions 🧠
# - Submit predictions to the competition environment 📤
# 
# ## Notebook Structure 📚
# This notebook is structured as follows:
# 1. **Data Preparation**: In this section, we load and preprocess the competition data.
# 2. **Feature Engineering**: We generate and select relevant features for model training.
# 3. **Model Training**: We train machine learning models on the prepared data.
# 4. **Prediction and Submission**: We make predictions on the test data and submit them for evaluation.
# 
# 
# ## How to Use 🛠️
# To use this notebook effectively, please follow these steps:
# 1. Ensure you have the competition data and environment set up.
# 2. Execute each cell sequentially to perform data preparation, feature engineering, model training, and prediction submission.
# 3. Customize and adapt the code as needed to improve model performance or experiment with different approaches.
# 
# **Note**: Make sure to replace any placeholder paths or configurations with your specific information.
# 
# ## Acknowledgments 🙏
# We acknowledge Open Problems in Single-Cell Analysis organizers for providing the dataset and the competition platform.
# 
# Let's get started! Feel free to reach out if you have any questions or need assistance along the way.
# 👉 [Visit my Profile](https://www.kaggle.com/zulqarnainali) 👈
# 
# 
# 

# ## Importing Required Libraries 📁

# In[1]:


# Suppressing Warnings 🚫
import warnings  # 🤐 suppress warnings
warnings.filterwarnings('ignore')

# Importing Required Libraries 📁
import os
import gc
import glob
import random
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from scipy import stats
from pathlib import Path
from itertools import groupby

# Importing Plotting Libraries 📊
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px

# etting Up Matplotlib for Inline Plotting 📈
get_ipython().run_line_magic('matplotlib', 'inline')

# Listing Files in Input Directory 📂
get_ipython().system('ls ../input/*')


# ## Reading Training Data 📊

# In[2]:


# Reading Training Data 
de_train = pd.read_parquet('../input/open-problems-single-cell-perturbations/de_train.parquet')

# ℹ️ Displaying the Shape of the Training Data ℹ️
de_train.shape


# ## Reading ID Mapping Data 📊

# In[3]:


# Reading ID Mapping Data 
id_map = pd.read_csv('../input/open-problems-single-cell-perturbations/id_map.csv')

# ℹ️ Displaying the Shape of the ID Mapping Data ℹ️
id_map.shape


# ## Reading Sample Submission Data 📊

# In[4]:


# Reading Sample Submission Data 
sample_submission = pd.read_csv('../input/open-problems-single-cell-perturbations/sample_submission.csv', index_col='id')

# ℹ️ Displaying the Shape of the Sample Submission Data ℹ️
sample_submission.shape


# ## Creating Feature Lists 📊

# In[5]:


# Creating Feature Lists 
xlist = ['cell_type', 'sm_name']
_ylist = ['cell_type', 'sm_name', 'sm_lincs_id', 'SMILES', 'control']

# 📝 Cell Title: Selecting Target Features 📝
y = de_train.drop(columns=_ylist)

# ℹ️ Displaying the Shape of the Target Features ℹ️
y.shape


# ## One-Hot Encoding Categorical Features 📊

# In[6]:


# One-Hot Encoding Categorical Features 
train = pd.get_dummies(de_train[xlist], columns=xlist)

# ℹ️ Displaying the Shape of the Encoded Training Data ℹ️
train.shape


# ## One-Hot Encoding Test Data 📊

# In[7]:


# One-Hot Encoding Test Data 
test = pd.get_dummies(id_map[xlist], columns=xlist)

# ℹ️ Displaying the Shape of the Encoded Test Data ℹ️
test.shape


# ## Finding Uncommon Features 📊

# In[8]:


# Finding Uncommon Features 
uncommon = [f for f in train if f not in test]

# ℹ️ Counting the Number of Uncommon Features ℹ️
len(uncommon)


# ## Removing Uncommon Features from Training Data 📊

# In[9]:


# Removing Uncommon Features from Training Data
X = train.drop(columns=uncommon)

# ℹ️ Displaying the Number of Features in Training and Test Data ℹ️
X.shape[1], test.shape[1]


# ## Checking Feature Equality 📊

# In[10]:


# Checking Feature Equality 
list(X.columns) == list(test.columns)


# ## Mean Root Root Mean Square Error Function 📝

# In[11]:


# Define a function to calculate Mean Root Root Mean Square Error for Pandas DataFrames
def mrrmse_pd(y_pred: pd.DataFrame, y_true: pd.DataFrame):
    # Calculate the squared error, take the mean along rows, apply square root, and then take the overall mean
    return ((y_pred - y_true)**2).mean(axis=1).apply(np.sqrt).mean()


# ## Mean Root Root Mean Square Error Function (NumPy) 📝

# In[12]:


# Define a function to calculate Mean Root Root Mean Square Error using NumPy
def mrrmse_np(y_pred, y_true):
    # Calculate the squared error, take the mean along rows, apply square root, and then take the overall mean
    return np.sqrt(np.square(y_true - y_pred).mean(axis=1)).mean()


# ## Selecting Columns for 'de_cell_type' 📊

# In[13]:


# Selecting Columns for 'de_cell_type'
de_cell_type = de_train.iloc[:, [0] + list(range(5, de_train.shape[1]))]

# Selecting Columns for 'de_sm_name' 📊
de_sm_name = de_train.iloc[:, [1] + list(range(5, de_train.shape[1]))]

# ℹ️ Displaying the Shapes of the Resulting DataFrames ℹ️
de_cell_type.shape, de_sm_name.shape


# ## 📓 Calculating Mean for 'de_cell_type' and 'de_sm_name' 

# In[14]:


# Calculating Mean for 'de_cell_type' and 'de_sm_name' 
mean_cell_type = de_cell_type.groupby('cell_type').mean().reset_index()
mean_sm_name = de_sm_name.groupby('sm_name').mean().reset_index()

# Displaying Mean for 'de_cell_type' 📊
display(mean_cell_type)

# Displaying Mean for 'de_sm_name' 📊
display(mean_sm_name)


# ## Extracting Rows from 'mean_cell_type' for 'de_cell_type' 📊

# In[15]:


# Extracting Rows from 'mean_cell_type' for 'de_cell_type' 
rows = []
for name in de_cell_type['cell_type']:
    mean_rows = mean_cell_type[mean_cell_type['cell_type'] == name].copy()
    rows.append(mean_rows)




# ## Creating 'tr_cell_type' DataFrame 📊

# In[16]:


# Creating 'tr_cell_type' DataFrame 
tr_cell_type = pd.concat(rows)
tr_cell_type = tr_cell_type.reset_index(drop=True)

# ℹ️ Displaying 'tr_cell_type' DataFrame ℹ️
tr_cell_type


# ## Extracting Rows from 'mean_sm_name' for 'de_sm_name' 📊

# In[17]:


# Extracting Rows from 'mean_sm_name' for 'de_sm_name' 
rows = []
for name in de_sm_name['sm_name']:
    mean_rows = mean_sm_name[mean_sm_name['sm_name'] == name].copy()
    rows.append(mean_rows)


# ## Creating 'tr_sm_name' DataFrame 📊

# In[18]:


# Creating 'tr_sm_name' DataFrame 
tr_sm_name = pd.concat(rows)
tr_sm_name = tr_sm_name.reset_index(drop=True)

# ℹ️ Displaying 'tr_sm_name' DataFrame ℹ️
tr_sm_name


# ## Extracting Rows from 'mean_cell_type' for 'id_map' 📊

# In[19]:


# Extracting Rows from 'mean_cell_type' for 'id_map' 
rows = []
for name in id_map['cell_type']:
    mean_rows = mean_cell_type[mean_cell_type['cell_type'] == name].copy()
    rows.append(mean_rows)


# ## Creating 'te_cell_type' DataFrame 📊

# In[20]:


# Creating 'te_cell_type' DataFrame 
te_cell_type = pd.concat(rows)
te_cell_type = te_cell_type.reset_index(drop=True)

# ℹ️ Displaying 'te_cell_type' DataFrame ℹ️
te_cell_type


# ## Extracting Rows from 'mean_sm_name' for 'id_map' 📊

# In[21]:


# Extracting Rows from 'mean_sm_name' for 'id_map' 
rows = []
for name in id_map['sm_name']:
    mean_rows = mean_sm_name[mean_sm_name['sm_name'] == name].copy()
    rows.append(mean_rows)

# Creating 'te_sm_name' DataFrame 
te_sm_name = pd.concat(rows)
te_sm_name = te_sm_name.reset_index(drop=True)

# ℹ️ Displaying 'te_sm_name' DataFrame ℹ️
te_sm_name


# ## Extracting the First Column of 'y' DataFrame 📊

# In[22]:


# Extracting the First Column of 'y' DataFrame 
y0 = y.iloc[:, 0].copy()

# ℹ️ Displaying 'y0' Series ℹ️
y0


# ## Joining Columns from 'tr_cell_type' and 'tr_sm_name' to 'X' 📊

# In[23]:


# Joining Columns from 'tr_cell_type' and 'tr_sm_name' to 'X' 
X0 = X.join(tr_cell_type.iloc[:, 0+1]).copy()
X0 = X0.join(tr_sm_name.iloc[:, 0+1], lsuffix='_cell_type', rsuffix='_sm_name')

# ℹ️ Displaying 'X0' DataFrame ℹ️
X0


# ## Joining Columns from 'te_cell_type' and 'te_sm_name' to 'test' 

# In[24]:


# Joining Columns from 'te_cell_type' and 'te_sm_name' to 'test' 
test0 = test.join(te_cell_type.iloc[:, 0+1]).copy()
test0 = test0.join(te_sm_name.iloc[:, 0+1], lsuffix='_cell_type', rsuffix='_sm_name')

# ℹ️ Displaying 'test0' DataFrame ℹ️
test0


# ## Calculating and Displaying Correlation Matrix

# In[25]:


# Calculating and Displaying Correlation Matrix for Selected Columns 
X0_corr = X0.copy()
X0_corr['y0'] = y0

# Calculate the correlation matrix for selected columns
corr = X0_corr.iloc[:, 131:].corr(numeric_only=True).round(3)

# Display the correlation matrix with a background gradient
corr.style.background_gradient(cmap='Pastel1')


# ## Correlation Heatmap for Training Set 

# In[26]:


# Creating and Displaying a Correlation Heatmap for Training Set 
cor_matrix = X0_corr.iloc[:, 131:].corr()

# Create a heatmap using seaborn
fig = plt.figure(figsize=(6, 6))
cmap = sns.diverging_palette(240, 10, s=75, l=50, sep=1, n=6, center='light', as_cmap=False)
sns.heatmap(cor_matrix, center=0, annot=True, cmap=cmap, linewidths=5)

# Set the title and display the heatmap
plt.suptitle('Train Set (Heatmap)', y=0.92, fontsize=16, c='darkred')
plt.show()


# ## Importing Machine Learning Libraries and Splitting Data 📊

# In[27]:


# Importing Machine Learning Libraries and Splitting Data 
import lightgbm as lgb
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X0, y0, test_size=0.20, random_state=421)


# In[28]:


# Creating and Training a LightGBM Regressor Model 
model = lgb.LGBMRegressor()

# Fit the model with the training data
model.fit(X_train, y_train)


# In[29]:


# Making Predictions with the Trained Model 
predictions = model.predict(X_test)


# In[30]:


# Visualizing Distribution of Residuals for Column N 
N = 0

# Set the style and create a plot
plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(8, 4), facecolor='lightyellow')

# Set the title and background color
plt.title(f'Column: #{N}', fontsize=12)
plt.gca().set_facecolor('lightgray')

# Create a distribution plot for residuals
sns.distplot(y_test.values - predictions, bins=100, color='red')

# Add a legend
plt.legend(['y_true', 'y_pred'], loc=1)
plt.show()


# ## LightGBM - Final mode

# In[31]:


# Creating and Configuring a LightGBM Regressor Model 📊
model = lgb.LGBMRegressor()

# 📊 Cell Title: Creating and Configuring a K-Nearest Neighbors Regressor Model 📊
# model = KNeighborsRegressor(n_neighbors=13)

# 📊 Cell Title: Creating and Configuring a Linear Support Vector Regressor Model 📊
# model = LinearSVR(max_iter=2000, epsilon=0.1)


# In[32]:


# Making Predictions for Multiple Columns 📊
pred = []

# Loop over each column in the target 'y'
for i in range(y.shape[1]):
    
    # Copy the i-th column of 'y'
    yi = y.iloc[:, i].copy()
    
    # Prepare the training data for the i-th column
    Xi = X.join(tr_cell_type.iloc[:, i+1]).copy()
    Xi = Xi.join(tr_sm_name.iloc[:, i+1], lsuffix='_cell_type', rsuffix='_sm_name')
    
    # Prepare the test data for the i-th column
    testi = test.join(te_cell_type.iloc[:, i+1]).copy()
    testi = testi.join(te_sm_name.iloc[:, i+1], lsuffix='_cell_type', rsuffix='_sm_name')
    
    # Fit the model and make predictions for the i-th column
    model.fit(Xi, yi)
    pred.append(model.predict(testi))
    
# Get the number of prediction arrays in 'pred'
len(pred)


# In[33]:


# Creating a DataFrame for Predictions 📊
prediction = pd.DataFrame(pred).T

# Set the column names and index name
prediction.columns = de_train.columns[5:]
prediction.index.name = 'id'

# Display the 'prediction' DataFrame
prediction


# In[34]:


# Saving Predictions to a CSV File 📓

# Save the 'prediction' DataFrame to a CSV file
prediction.to_csv('prediction.csv')

# Use the 'ls' command to list the files in the current directory
get_ipython().system('ls')


# ## Importing Data and Combining Submissions 📓

# In[35]:


# Import the first dataset and display its shape
import1 = pd.read_csv('../input/op2-603/op2_603.csv', index_col='id')
import1.shape

# Import the second dataset and display its shape
import2 = pd.read_csv('../input/op2-720/op2_720.csv', index_col='id')
import2.shape

# Import the third dataset and display its shape
import3 = pd.read_csv('../input/op2-604/submission_df.csv', index_col='id')
import3.shape

# Import the fourth dataset and display its shape
import4 = pd.read_csv('../input/op2-607/OP2_607.csv', index_col='id')
import4.shape

# Create a list of column names to be used for combining submissions
col = list(de_train.columns[5:])

# Copy the sample_submission DataFrame
submission = sample_submission.copy()

# Combine the submissions using weighted averages
submission[col] = (import1[col] * 0.22) + (import2[col] * 0.17) + (import3[col] * 0.22) + (import4[col] * 0.22) + (prediction[col] * 0.17)

# Display the shape of the combined submission DataFrame
submission.shape


# In[36]:


# Saving Submission Data to a CSV File 📓

# Save the 'submission' DataFrame to a CSV file
submission.to_csv('submission.csv')

# Use the 'ls' command to list the files in the current directory
get_ipython().system('ls')


# ## Explore More! 👀
# Thank you for exploring this notebook! If you found this notebook insightful or if it helped you in any way, I invite you to explore more of my work on my profile.
# 
# 👉 [Visit my Profile](https://www.kaggle.com/zulqarnainali) 👈
# 
# ## Feedback and Gratitude 🙏
# We value your feedback! Your insights and suggestions are essential for our continuous improvement. If you have any comments, questions, or ideas to share, please don't hesitate to reach out.
# 
# 📬 Contact me via email: [zulqar445ali@gmail.com](mailto:zulqar445ali@gmail.com)
# 
# I would like to express our heartfelt gratitude for your time and engagement. Your support motivates us to create more valuable content.
# 
# Happy coding and best of luck in your data science endeavors! 🚀
# 
