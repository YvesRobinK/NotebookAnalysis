#!/usr/bin/env python
# coding: utf-8

# # 1. Introduction
# 
# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 1px; color:#04bcc9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #04bcc9;">1.1 Background</p>
# 
# Welcome to this notebook on **Water Quality Prediction** as part of the ongoing ML Olympiad series. This is a **regression** problem where the goal is to predict a continuous variable that represents water quality. We are given a mixture of **continuous and categorical features** to work with however they are **masked**.
# 
# The **competition metric** is **Root Mean Squared Log Error (RMSLE)**.

# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 1px; color:#04bcc9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #04bcc9;">1.2 Libraries</p>

# In[1]:


# Core
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style='darkgrid', font_scale=1.4)
import itertools
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
import time
import umap

# Sklearn
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold, PredefinedSplit
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score
from sklearn.metrics import roc_auc_score, plot_confusion_matrix, plot_roc_curve, roc_curve, mean_squared_error, mean_squared_log_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.utils import resample

# Models
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB


# # 2. Data
# 
# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 1px; color:#04bcc9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #04bcc9;">2.1 Load data</p>

# In[2]:


# Load data
train = pd.read_csv('/kaggle/input/ml-olympiad-waterqualityprediction/train.csv', index_col = 'id')
test = pd.read_csv('/kaggle/input/ml-olympiad-waterqualityprediction/test.csv', index_col='id')
sub = pd.read_csv('/kaggle/input/ml-olympiad-waterqualityprediction/sample_submission.csv')

# Print shape and preview
print('Train shape:', train.shape)
print('Test shape:', test.shape)
train.head()


# **Feature descriptions:**
# 
# > 1. The 6 `category` columns are various categorical features for a data point such as country of data collection, the site from which the data is collected, media of sample, etc.
# > 2. The 9 `feature` columns are the various demographic features that affect the pollution of water in a particular region such as population density, GDP, droughts in a region, literacy rate of students in a region, etc.
# > 3. The 10 `composition` columns are the compositions of various elements like paper, plastic wastes, cardboard, etc. in water.
# > 4. The `unit` value is the unit of measurement in which the result value is measured.
# > 5. The `result` value is a floating number that expresses the quality of water based on the various factors provided in the dataset.

# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 1px; color:#04bcc9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #04bcc9;">2.2 Missing values</p>
# 
# There are a **small number of missing values**.

# In[3]:


# Number of missing values per column
na_df = pd.DataFrame([train.isna().sum(),test.isna().sum()]).T
na_df.columns = ["Train","Test"]
na_df


# Let's check the **distribution of missing values** to see if there is a pattern.

# In[4]:


print("Number of rows with at least 1 missing values (train):", (train.isna().sum(axis=1)>0).sum())
print("Number of rows with at least 1 missing values (test):", (test.isna().sum(axis=1)>0).sum())


# Of the features that have missing values, whenever one of these features has a missing value all of them do. That is, **they occur at the same time**.

# In[5]:


# Heatmap of missing values
plt.figure(figsize=(12,5))
sns.heatmap(train.reset_index(drop=True)[train.reset_index(drop=True).isna().sum(axis=1)!=0].isna().T, cmap='Blues')
plt.title('Heatmap of missing values')
plt.xlabel("Index")
plt.show()


# The location of these missing values appears to **not be entirely random** as there are small clusters. 

# In[6]:


plt.figure(figsize=(20,4))
plt.scatter(x=np.arange(len(train)),y=(train.isna().sum(axis=1)!=0))
plt.xlabel("Index")
plt.ylabel("IsMissing")
plt.title("Location of samples with missing values")
plt.show()


# We can check to see if the **mean and standard deviation of the target** between samples with and without missing values is different. 

# In[7]:


print(f"Mean result of samples with missing values {train[train.isna().sum(axis=1)!=0]['result'].mean():.3f}")
print(f"Standard deviation result of samples with missing values {train[train.isna().sum(axis=1)!=0]['result'].std():.3f}")
print(f"\nMean result of samples without missing values {train[train.isna().sum(axis=1)==0]['result'].mean():.3f}")
print(f"Standard deviation result of samples without missing values {train[train.isna().sum(axis=1)==0]['result'].std():.3f}")


# To check if this is difference is significant, we can run a quick **one-way ANOVA test**.

# In[8]:


# One-way ANOVA
from scipy.stats import f_oneway
f_oneway(train[train.isna().sum(axis=1)!=0]['result'],train[train.isna().sum(axis=1)==0]['result'])


# Since the p-value is above 0.05, we can **accept the null hypothesis** that these two groups are from the **same distribution**.
# 
# This tells us we can **safely fill** in the missing values using e.g. the **median** for each feature from the **rest** of the training data. 

# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 1px; color:#04bcc9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #04bcc9;">2.3 Data types</p>
# 
# The feature names describe the **data types** quite well. 

# In[9]:


train.dtypes


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 1px; color:#04bcc9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #04bcc9;">2.4 Unique values</p>
# 
# There are some **categorial columns** with **very high cardinalities** (number of unique values) and on the other hand, all the **continuous columns** have a **much lower cardinality** than what is normal.

# In[10]:


# Number of unique values per column
nu_df = pd.DataFrame([train.nunique(),test.nunique()]).T
nu_df.columns = ["Train","Test"]
nu_df


# It makes sense to me to **convert** `categoryA`, `categoryC` and `categoryE` into **continuous features** because of how many unique values they have.

# In[11]:


train[["categoryA","categoryC","categoryE"]].reset_index(drop=True).head(3)


# To convert these to numerical features, we can **split on the underscore** and just take the **number at the end**.

# In[12]:


for i in ["A","C","E"]:
    train[f"category{i}"] = train[f"category{i}"].apply(lambda x: float(x.split("_")[-1]))
    test[f"category{i}"] = test[f"category{i}"].apply(lambda x: float(x.split("_")[-1]))


# # 3. EDA
# 
# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 1px; color:#04bcc9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #04bcc9;">3.1 Distributions</p>
# 
# **Target:**

# In[13]:


plt.figure(figsize=(12,4))
sns.histplot(data=train, x="result")
plt.title("Target distribution")
plt.show()


# Note: The target values **lie between 0 and 1** so if we wanted to we could treat this as a **classification problem** and predict the probability of the result. 

# In[14]:


print("Target maximum:", train["result"].max())
print("Target minimum:", train["result"].min())


# **Continuous features:**

# In[15]:


# Continuous columns
float_cols = [col for (col, d) in zip(train.iloc[:,:-1].columns,train.iloc[:,:-1].dtypes) if d == "float64"]

# Figure with subplots
fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(25,20))

for i, ax in enumerate(axes.flat):
    try:
        sns.scatterplot(ax=ax, data=train, x=float_cols[i], y="result")
        ax.set_title(f'{float_cols[i]} distribution')
    except:
        ax.set_visible(False)

# Improves appearance
fig.tight_layout()
plt.legend()
plt.show()


# **Categorical columns:**

# In[16]:


# Categorical columns
cat_cols = [f"category{i}" for i in ["A","B","C","D","E","F"]]+["unit"]

# Figure with subplots
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(20,15))

for i, ax in enumerate(axes.flat):
    try:
        if i%2==1:
            sns.violinplot(ax=ax, data=train, x=cat_cols[i], y="result")
            ax.set_title(f'{cat_cols[i]} distribution')
        else:
            sns.scatterplot(ax=ax, data=train, x=cat_cols[i], y="result")
            ax.set_title(f'{cat_cols[i]} distribution')
    except:
        ax.set_visible(False)

# Improves appearance
fig.tight_layout()
plt.legend()
plt.show()


# Observe how the three categorical columns on the right are **heavily imbalanced** and each have very **different joint distributions** between categories. 

# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 1px; color:#04bcc9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #04bcc9;">3.2 Correlations</p>

# In[17]:


# Heatmap of correlations
plt.figure(figsize=(25,20))
corr = train.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = 1
sns.heatmap(corr, mask=mask,linewidths=.5, annot=True, cmap="Blues")
plt.title('Heatmap of correlations')
plt.show()


# Although most features are **weakly correlated** with the **target**, many features are **highly correlated** to other **features**. We might want to consider **feature selection algorithms** to remove high correlation between input features.

# In[18]:


# Pairplot
g = sns.PairGrid(train[[f"feature{i}" for i in ["A","B","C","D","E","F","G","H","I"]]], height=2, aspect=1.2, corner=True)
g.map_lower(sns.regplot, color='green')
g.map_diag(sns.histplot)
plt.suptitle('Feature pairplot', y=1.02, fontsize=42)
plt.show()


# In[19]:


# Pairplot
g = sns.PairGrid(train[[f"composition{i}" for i in ["A","B","C","D","E","F","G","H","I","J"]]], height=2, aspect=1.2, corner=True)
g.map_lower(sns.regplot, color='purple')
g.map_diag(sns.histplot)
plt.suptitle('Composition pairplot', y=1.02, fontsize=42)
plt.show()


# # 4. Pre-processing
# 
# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 1px; color:#04bcc9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #04bcc9;">4.1 Features and labels</p>
# 
# Split the **target** from the **features**.

# In[20]:


# Labels
y = train['result']

# Features
X = train.drop('result', axis=1)
X_test = test


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 1px; color:#04bcc9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #04bcc9;">4.2 Fill missing values</p>
# 
# Fill missing values with **median** of each column. Median is much more **robust to outliers** than mean.

# In[21]:


# Location of missing values
X_na = X[X.isna().sum(axis=1)>0].index
X_test_na = X_test[X_test.isna().sum(axis=1)>0].index

# Fill missing values with median
X = X.fillna(X.median())
X_test = X_test.fillna(X_test.median())


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 1px; color:#04bcc9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #04bcc9;">4.3 Feature engineering</p>
# 
# Create **new features** here.

# In[22]:


# Location of missing values
X["IsMissing"] = X.index.isin(X_na).astype(int)
X_test["IsMissing"] = X_test.index.isin(X_test_na).astype(int)


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 1px; color:#04bcc9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #04bcc9;">4.4 Encoding</p>
# 
# **One-hot encode** the remaining categorical columns.

# In[23]:


# One-hot encode categorical columns
X = pd.get_dummies(X, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 1px; color:#04bcc9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #04bcc9;">4.5 Scaling</p>
# 
# Scaling transforms every column to have **mean 0 and standard deviation 1**. This makes it easier to models to compare different features.

# In[24]:


# Scale data
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 1px; color:#04bcc9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #04bcc9;">4.6 Train-valid split</p>
# 
# Let's create a **validation set** to evaluate our models on.

# In[25]:


# Split data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Print feature matrix sizes
print("X_train shape:", X_train.shape)
print("X_val shape:", X_valid.shape)


# # 5. Modelling
# 
# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 1px; color:#04bcc9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #04bcc9;">5.1 Define models</p>
# 
# We will train several models and tune their parameters using **grid/random search**. Later we will **evaluate** these models and **ensemble** them to produce the best predictions.

# In[26]:


# Regressors
regressors = {
    "LinearRegression" : LinearRegression(),
    "Lasso" : Lasso(random_state=0),
    "Ridge" : Ridge(random_state=0),
    "KNN" : KNeighborsRegressor(),
    "SVC" : SVR(),
    "RandomForest" : RandomForestRegressor(random_state=0),
    "ExtraTrees" : ExtraTreesRegressor(random_state=0),
    "XGBoost" : XGBRegressor(random_state=0, use_label_encoder=False, eval_metric='rmsle'),
    "LGBM" : LGBMRegressor(random_state=0),
    "CatBoost" : CatBoostRegressor(random_state=0, verbose=False)
}


# In[27]:


# Grids for grid search
LR_grid = {'alpha' : [0.25, 0.5, 0.75, 1, 1.25, 1.5]}

KNN_grid = {'n_neighbors': [3, 5, 7, 9],
            'p': [1, 2]}

SVC_grid = {'C': [0.25, 0.5, 0.75, 1, 1.25, 1.5],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']}

RF_grid = {'n_estimators': [50, 100, 150, 200, 250, 300],
        'max_depth': [4, 6, 8, 10, 12]}

boosted_grid = {'n_estimators': [50, 100, 150, 200],
        'max_depth': [4, 8, 12],
        'learning_rate': [0.05, 0.1, 0.15]}


# In[28]:


# Dictionary of all grids
grid = {
    "LinearRegression" : {},
    "Lasso" : LR_grid,
    "Ridge" : LR_grid,
    "KNN" : KNN_grid,
    "SVC" : SVC_grid,
    "RandomForest" : RF_grid,
    "ExtraTrees" : RF_grid,
    "XGBoost" : boosted_grid,
    "LGBM" : boosted_grid,
    "CatBoost" : boosted_grid,
}


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 1px; color:#04bcc9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #04bcc9;">5.2 Train models</p>
# 
# **Train models** one at a time with **hyper-parameter tuning**. Save **validation predictions** for find best ensemble weights later.

# In[29]:


# Train models
for key, regressor in regressors.items():
    # Start timer
    start = time.time()
        
    # Tune hyperparameters
    reg = RandomizedSearchCV(estimator=regressor, param_distributions=grid[key], n_iter=20, scoring='neg_mean_squared_log_error', n_jobs=-1, cv=5)

    # Train using PredefinedSplit
    reg.fit(X_train, y_train)

    # Validation set predictions
    val_preds = reg.predict(X_valid)
    val_preds[val_preds < 0] = 0
    score = mean_squared_log_error(y_valid, val_preds, squared=False)

    # Test set predictions
    test_preds = reg.predict(X_test)
    test_preds[test_preds < 0] = 0
    
    # Stop timer
    stop = time.time()
    
    # Print score and time
    print('Model:', key)
    print('Validation RMSLE:', score)
    print('Training time (mins):', np.round((stop - start)/60,2))
    print('')
    
    # Save valid preds
    pd.DataFrame({"result":val_preds}).to_csv(f"{key}_val_preds.csv", index=False)
    
    # Save test preds
    ss = sub.copy()
    ss["result"] = test_preds
    ss.to_csv(f"{key}_test_preds.csv", index=False)


# # 6. Ensembling
# 
# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 1px; color:#04bcc9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #04bcc9;">6.1 Prepare predictions</p>
# 
# **Join** validation and test predictions into two **dataframes**.

# In[30]:


# Join valid preds
valid_df = pd.DataFrame(index=np.arange(len(y_valid)))
for i in regressors.keys():
    df = pd.read_csv(f"/kaggle/working/{i}_val_preds.csv")
    df.rename(columns={"result": i}, inplace=True)
    valid_df = pd.concat([valid_df,df], axis=1)
    
# Join test preds
test_preds = pd.DataFrame(index=np.arange(len(X_test)))
for i in regressors.keys():
    df = pd.read_csv(f"/kaggle/working/{i}_test_preds.csv")
    df.rename(columns={"result": i}, inplace=True)
    test_preds = pd.concat([test_preds,df], axis=1)
    
valid_df.head(3)


# **Evaluate** validation predictions and **sort** dataframes by performance.

# In[31]:


# Evaluate valid preds
scores = {}
for col in valid_df.columns:
    scores[col] = mean_squared_log_error(y_valid, valid_df[col], squared=False)

# Sort scores
scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}

# Sort oof_df and test_preds
valid_df = valid_df[list(scores.keys())]
test_preds = test_preds[list(scores.keys())]

scores


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 1px; color:#04bcc9; font-size:100%; text-align:left;padding: 0px; border-bottom: 3px solid #04bcc9;">6.2 Hill Climbing</p>
# 
# For more details on **hill climbing**, [see here](https://www.kaggle.com/code/samuelcortinhas/ps-s3e3-hill-climbing-like-a-gm).

# In[32]:


# Initialise
STOP = False
current_best_ensemble = valid_df.iloc[:,0]
current_best_test_preds = test_preds.iloc[:,0]
MODELS = valid_df.iloc[:,1:]
weight_range = np.arange(-0.5,0.51,0.01)
history = [mean_squared_log_error(y_valid, current_best_ensemble, squared=False)]
i=0

# Hill climbing
while not STOP:
    i+=1
    potential_new_best_cv_score = mean_squared_log_error(y_valid, current_best_ensemble, squared=False)
    k_best, wgt_best = None, None
    for k in MODELS:
        for wgt in weight_range:
            potential_ensemble = (1-wgt) * current_best_ensemble + wgt * MODELS[k]
            potential_ensemble[potential_ensemble < 0] = 0
            cv_score = mean_squared_log_error(y_valid, potential_ensemble, squared=False)
            if cv_score < potential_new_best_cv_score:
                potential_new_best_cv_score = cv_score
                k_best, wgt_best = k, wgt
    
    if k_best is not None:
        current_best_ensemble = (1-wgt_best) * current_best_ensemble + wgt_best * MODELS[k_best]
        current_best_ensemble[current_best_ensemble < 0] = 0
        current_best_test_preds = (1-wgt_best) * current_best_test_preds + wgt_best * test_preds[k_best]
        current_best_test_preds[current_best_test_preds < 0] = 0
        MODELS.drop(k_best, axis=1, inplace=True)
        if MODELS.shape[1]==0:
            STOP = True
        print(f'Iteration: {i}, Model added: {k_best}, Best weight: {wgt_best:.2f}, Best RMSE: {potential_new_best_cv_score:.5f}')
        history.append(potential_new_best_cv_score)
    else:
        STOP = True


# In[33]:


plt.figure(figsize=(10,4))
plt.plot(np.arange(len(history))+1, history, marker="x")
plt.title("RMSLE vs. Number of Models with Hill Climbing")
plt.xlabel("Number of models")
plt.ylabel("AUC")
plt.show()


# **Distribution of predictions:**

# In[34]:


plt.figure(figsize=(10,4))
sns.histplot(current_best_test_preds)
plt.title("Distribution of final predictions")
plt.show()


# **Clip predictions:**

# In[35]:


# Just in case
current_best_test_preds[current_best_test_preds < 0] = 0
current_best_test_preds[current_best_test_preds > 1] = 1


# # 7. Submission
# 
# **Save predictions** to csv files for the competition.

# In[36]:


# Submit predictions
submission = sub.copy()
submission["result"] = current_best_test_preds
submission.to_csv("submission.csv", index=False)

