#!/usr/bin/env python
# coding: utf-8

# # Introduction

# The objective is to classify 10 different bacteria species given genomic sequencing data. This data has been compressed so that for example ATATGGCCTT becomes A2T4G2C2. From this lossy data we need recover the genome fingerprint to find the corresponding bacteria. 
# 
# **Acknowledgements:**
# * [TPS - Feb 2022](https://www.kaggle.com/sfktrkl/tps-feb-2022) by [Safak Tukeli](https://www.kaggle.com/sfktrkl).
# * [TPSFEB22-01 EDA which makes sense](https://www.kaggle.com/ambrosm/tpsfeb22-01-eda-which-makes-sense) by [AmbrosM](https://www.kaggle.com/ambrosm). 
# * [Analysis of Identification Method for Bacterial Species and Antibiotic Resistance Genes Using Optical Data From DNA Oligomers](https://www.frontiersin.org/articles/10.3389/fmicb.2020.00257/full) by R. Wood et al. 
# * [Semi-supervised Learning & ExtraTrees](https://www.kaggle.com/vpallares/semi-supervised-learning-extratrees/notebook?scriptVersionId=87570753) by [Vicente Pallares](https://www.kaggle.com/vpallares).
# * [TPS - Feb 2022 - Only 4 Features - A, T, C, G](https://www.kaggle.com/roberterffmeyer/tps-feb-2022-only-4-features-a-t-c-g/notebook) by [Robert Erffmeyer](https://www.kaggle.com/roberterffmeyer).

# # Libraries

# In[1]:


# Core
import numpy as np
import pandas as pd
import seaborn as sns
sns.set(style='darkgrid', font_scale=1.4)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from itertools import combinations
import math
from math import factorial
import statistics
import scipy.stats
from scipy.stats import pearsonr
import time
from datetime import datetime
import matplotlib.dates as mdates
import dateutil.easter as easter

# Sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, plot_roc_curve, roc_curve
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.decomposition import PCA

# Models
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

# Tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks


# # Data

# **Load data**

# In[2]:


# Save to df
train_data=pd.read_csv('../input/tabular-playground-series-feb-2022/train.csv', index_col='row_id')
test_data=pd.read_csv('../input/tabular-playground-series-feb-2022/test.csv', index_col='row_id')

# Shape and preview
print('Training data df shape:',train_data.shape)
print('Test data df shape:',test_data.shape)
train_data.head()


# **Missing values**

# In[3]:


print('Number of missing values in training set:',train_data.isna().sum().sum())
print('')
print('Number of missing values in test set:',test_data.isna().sum().sum())


# **Duplicates**

# In[4]:


print(f'Duplicates in training set: {train_data.duplicated().sum()}')
print('')
print(f'Duplicates in test set: {test_data.duplicated().sum()}')


# **Describe**

# In[5]:


train_data.describe()


# *Initial thoughts:*
# * This is a big dataset. It would be helpful to reduce the size for storage and compute reasons. 
# * Each column represents one of 286 combinations for compressed genome sequence. They represent a one-hot encoding. It would be ideal to transform these into 4 columns where each one counts how many of A-G-C-T units appear. 
# * There is a small range between min and max values and some of the data is negative which doesn't make much biological sense. It would be good clean this data and transform it. 
# * Perhaps biological knowledge would be helpful here. I imagine some combinations are more likely than others (some maybe even impossible).
# * There are many duplicates, so we could save on computational cost by removing these.

# # EDA

# **Target distribution**

# In[6]:


# Figure
plt.figure(figsize=(15,5))

# Countplot
sns.countplot(data=train_data, x='target')

# Aesthetics
plt.xticks(rotation=40, ha='right')
plt.title('Target distribution')


# The target is (highly) balanced, which is great.

# **PCA plot**

# In[7]:


# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(train_data.drop('target', axis=1))

# Convert to data frame
principal_df = pd.DataFrame(data = X_pca, columns = ['PC1', 'PC2'])
principal_df = pd.concat([principal_df,pd.Series(LabelEncoder().fit_transform(train_data['target']))], axis=1)

# Figure size
plt.figure(figsize=(8,6))

# Scatterplot
plt.scatter(principal_df.iloc[:,0], principal_df.iloc[:,1], c=principal_df.iloc[:,2], s=5, cmap='tab10')

# Aesthetics
plt.title('PCA plot in 2D', fontsize=15)
plt.xlabel('PC1', fontsize=15)
plt.ylabel('PC2', fontsize=15)


# Quite a lot of overlap. However, AmbrosM managed to better separate the data by reverse engineering the pipeline from the original [paper](https://www.frontiersin.org/articles/10.3389/fmicb.2020.00257/full). We'll see this in a bit.

# # Feature Engineering

# **Reverse engineering**

# The feature values were originally integers. The integers were divided by 1000000 and a constant was subtracted. We can reverse this process to get the integers back.

# In[8]:


def bias_of(s):
    w = int(s[1:s.index('T')])
    x = int(s[s.index('T')+1:s.index('G')])
    y = int(s[s.index('G')+1:s.index('C')])
    z = int(s[s.index('C')+1:])
    return factorial(10) / (factorial(w) * factorial(x) * factorial(y) * factorial(z) * 4**10)

elements = [e for e in train_data.columns if e != 'row_id' and e != 'target']

train_data[elements] = pd.DataFrame({col: ((train_data[col] + bias_of(col)) * 1000000).round().astype(int)
                        for col in elements})
test_data[elements] = pd.DataFrame({col: ((test_data[col] + bias_of(col)) * 1000000).round().astype(int)
                       for col in elements})

train_data.head()


# **GCD**

# From AmbrosM
# > For every sample, the researchers did one of four things:
# > 
# > 1. They put 1000000 decamers into the machine and saved the machine's output.
# > 2. They put 100000 decamers into the machine and multiplied the machine's output by 10.
# > 3. They put 1000 decamers into the machine and multiplied the machine's output by 1000.
# > 4. They put 100 decamers into the machine and multiplied the machine's output by 10000.
# > 
# > With this procedure, the row sums are always 1000000 and we get gcd values of 1, 10, 1000 and 10000.

# In[9]:


# Create gcd feature
train_data['gcd'] = np.gcd.reduce(train_data[elements], axis=1)
test_data['gcd'] = np.gcd.reduce(test_data[elements], axis=1)

# Print unique gcd values and their distributions
np.unique(train_data['gcd'], return_counts=True), np.unique(test_data['gcd'], return_counts=True)


# **PCA**

# The higher the gcd, the more the noise is amplified by the scalling factor. So we can expect the variance of the data to increase. We can visualise this by doing pca plots for different gcd values. 

# In[10]:


plt.figure(figsize=(12,12))

# Train PCA
pca=PCA(n_components=2)
pca.fit(train_data.loc[train_data['gcd'] == 1, elements])

for i, scale in enumerate(np.sort(train_data['gcd'].unique())):
    # Transform data onto PCA space
    X_pca=pca.transform(train_data.loc[train_data['gcd'] == scale, elements])
    
    # Convert to data frame
    principal_df = pd.DataFrame(data = X_pca, columns = ['PC1', 'PC2'])
    principal_df = pd.concat([principal_df,pd.Series(LabelEncoder().fit_transform(train_data[train_data['gcd'] == scale]['target']))], axis=1)
    
    # Plot pca
    ax=plt.subplot(2, 2, i+1)
    plt.scatter(principal_df.iloc[:,0], principal_df.iloc[:,1], axes=ax, c=principal_df.iloc[:,2], s=2, cmap='tab10')
    plt.title(f"gcd={scale}")


# **Duplicates on a gcd basis**

# In[11]:


def plot_duplicates_per_gcd(df, title):
    plt.figure(figsize=(14, 3))
    plt.tight_layout()
    for i, gcd in enumerate(np.unique(df.gcd)):
        plt.subplot(1, 4, i+1)
        duplicates = df[df.gcd == gcd][elements].duplicated().sum()
        non_duplicates = len(df[df.gcd == gcd]) - duplicates
        plt.pie([non_duplicates, duplicates],
                labels=['not duplicate', 'duplicate'],
                colors=['lightgray', 'b'],
                startangle=90)
        plt.title(f'GCD = {gcd}')
    plt.subplots_adjust(wspace=0.8)
    plt.suptitle(title)
    plt.show()
        
plot_duplicates_per_gcd(train_data, title='Duplicates in Training')
plot_duplicates_per_gcd(test_data, title='Duplicates in Test')


# The number of duplicates for gcd 1 or 10 is very small, whereas for gcd 1000 or 10000 the proportion of duplicates is over 50%! Perhaps we could use pseudolabels to improve the accuracy on predictions for gcd 1000 and 10000.

# **Drop duplicates**

# In[12]:


# Drop duplicates to save on computing time
train_data=train_data.drop_duplicates(keep='first')


# **Distribution on a gcd basis**

# Let's check the target distribution is also balanced within each gcd subset after duplicates have been removed. This would make it feasible to train 4 separate classifiers for each gcd value.

# In[13]:


# Figure
plt.figure(figsize=(12,8))

for index, i in enumerate([1, 10, 1000, 10000]):
    ax=plt.subplot(2,2,index+1)
    sns.countplot(data=train_data[train_data['gcd']==1], x='target', axes=ax)
    ax.set(xticklabels=[])
    ax.set(xlabel=None)
    ax.set(ylabel=None)
    plt.title(f'gcd={i}')


# The target is still highly balanced within each gcd group so it is possible to create a model for each subset.

# # Measurement errors

# **Train test drift**

# The test set has a slightly different distribution to the train set. We need to understand this to improve the accuracy of our models.

# In[14]:


# From https://www.kaggle.com/ambrosm/tpsfeb22-01-eda-which-makes-sense#Comparing-train-and-test
scale = 1

# Compute the PCA
pca = PCA(n_components=2)
pca.fit(train_data[elements][train_data['gcd'] == scale])

# Transform the data so that the components can be analyzed
Xt_tr = pca.transform(train_data[elements][train_data['gcd'] == scale])
Xt_te = pca.transform(test_data[elements][test_data['gcd'] == scale])

# Plot a scattergram, projected to two PCA components, of training and test data
plt.figure(figsize=(6,6))
plt.scatter(Xt_tr[:,0], Xt_tr[:,1], c='b', s=1, label='Train')
plt.scatter(Xt_te[:,0], Xt_te[:,1], c='r', s=1, label='Test')
plt.title("The test data deviate from the training data")
plt.legend()
plt.show()


# The researchers added noise to the test set to model mutations and measurement errors when reading the decamers. I read the paper and I believe they modelled this noise as follows:
# 1. Mutations and measurement errors are combined into a single global error rate p (e.g. p=0.01).
# 2. p gives the probablity that any decamer has been incorrectly read by the machine. 
# 3. So for every decamer, we can flip a coin with heads probablity p. If it comes up tails we say the decamer has been correctly read by the machine and leave it alone. If it comes up heads, we say the decamer is erroneous and sample a new one from the bias distribution. 
# 4. It gets a bit trickier when gcd>1 as the data has been scaled. For example, assume gcd=10 and the decamer count for that feature is 50; we flip 50/10=5 coins and see how many come up heads. Say 3 coins come up tails, then we say 3*10=30 of the 50 decamers were correctly read by the machine and the other 20 were not. We then sample from the bias distribution twice and scale the answer by the gcd (i.e. 10). This ensures that the rows still have the same gcd and still sum to 1000000.

# **Bias distribution**

# In[15]:


# Plot bias/noise distribution
plt.figure(figsize=(12,5))
plt.plot([bias_of(e) for e in elements])
plt.title('Bias distribution (i.e. noise distribution)')


# **Model for measurement errors**

# In[16]:


def add_noise(train_data, elements, p):
    '''Add noise to training data,
    train_data : training set,
    elements : features A0T0G0C10 to A10T0G0C0,
    m : error rate.
    '''
    # Start time
    start=time.time()
    
    # Bias distribution
    grid_bias=[bias_of(e) for e in elements]
    grid_index=np.arange(len(elements))

    train_with_noise=train_data.copy()
    
    for i in train_with_noise.index:
        incorrect_count=0
        for j in elements:
            # Simulate decamers incorrectly labelled because of noise
            incorrect=np.random.binomial((train_with_noise.loc[i,j]/train_with_noise.loc[i,'gcd']).astype(int),p)

            if incorrect>0:
                # Subtract correctly labelled samples (in batches of gcd)
                train_with_noise.loc[i,j]-=incorrect*train_with_noise.loc[i,'gcd']

                # Total incorrectly labelled samples in row
                incorrect_count+=incorrect*train_with_noise.loc[i,'gcd']

        # Choose new samples (noise) according to bias distribution
        noise_index=np.random.choice(grid_index, size=(incorrect_count/train_with_noise.loc[i,'gcd']).astype(int), p=grid_bias)

        # Add the noise to the training set
        unique_index, counts = np.unique(noise_index, return_counts=True)
        train_with_noise.loc[i,[elements[u] for u in unique_index]]+=counts*train_with_noise.loc[i,'gcd']
        
        if (i%1000)==0:
            print(f'Iteration:{i}', f'time: {np.round((time.time()-start)/60,2)} mins')

    return train_with_noise


# In[17]:


# Create noisy train data
#train_with_noise=add_noise(train_data, elements, 1e-2)

# I ran the above function and saved it to this dataset
train_with_noise=pd.read_csv('../input/tps-feb-22-train-set-with-measurement-errors/train_with_noise.csv')
train_with_noise.head()


# # Modelling

# In[18]:


target_encoder = LabelEncoder()
train_with_noise['target']=target_encoder.fit_transform(train_with_noise['target'])
def run_model(train_data, test_data, scale):
    # Labels and features
    X=train_data[train_data['gcd']==scale].copy()
    y=X['target']
    X=X.drop('target', axis=1)
    
    # Test subset
    gcd_test=test_data[test_data['gcd']==scale]
    y_preds_index=gcd_test.index
    
    # Initialise outputs
    scores = []
    y_probs = []
    
    # Cross-validation
    folds = StratifiedKFold(n_splits=N_SPLITS, random_state=0, shuffle=True)
    for fold, (train_id, test_id) in enumerate(folds.split(X, y)):
        X_train = X.iloc[train_id]
        y_train = y.iloc[train_id]
        X_valid = X.iloc[test_id]
        y_valid = y.iloc[test_id]
        
        # Model
        model = ExtraTreesClassifier(
            n_estimators=ESTIMATORS,
            random_state=0,
            n_jobs=-1
        )
        
        # Train and predict
        model.fit(X_train, y_train)
        valid_pred = model.predict(X_valid)
        valid_score = accuracy_score(y_valid, valid_pred)
        
        print('gcd:', scale,', fold:', fold + 1,', accuracy:', valid_score)
        scores.append(valid_score)
        
        # Predict only on corresponding gcd subset in test_data
        y_probs.append(model.predict_proba(gcd_test))
        
    # Mean of probabilities
    y_proba=np.array(y_probs).sum(axis=0)/N_SPLITS
    
    # Mean accuracy
    print(f'Mean accuracy score for gcd={scale}:', np.array(scores).mean())
    
    return y_proba, y_preds_index


# **gcd = 1, 10**

# In[19]:


N_SPLITS = 5
ESTIMATORS = 1000


# In[20]:


y_proba_concat = np.array([]).reshape(0,10)
y_preds_index_concat = np.array([])
for scale in [1, 10]:
    # Run model
    y_proba, y_preds_index = run_model(train_with_noise, test_data, scale)
    
    # Store predictions and corresponding indices
    y_proba_concat=np.concatenate((y_proba_concat, y_proba),axis=0)
    y_preds_index_concat=np.concatenate((y_preds_index_concat, y_preds_index),axis=0)
    
# Recover class names
y_preds=np.argmax(y_proba_concat, axis=1)
y_preds=y_preds.astype(int)
y_preds=target_encoder.inverse_transform(y_preds)

# Save predictions to df
preds_df=pd.DataFrame({'row_id': y_preds_index_concat.astype(int), 'target': y_preds})


# **gcd = 1000, 10000**

# In[21]:


N_SPLITS = 10
ESTIMATORS = 3000


# In[22]:


y_proba_concat2 = np.array([]).reshape(0,10)
y_preds_index_concat2 = np.array([])
for scale in [1000, 10000]:
    # Run model
    y_proba, y_preds_index = run_model(train_with_noise, test_data, scale)
    
    # Store predictions and corresponding indices
    y_proba_concat2=np.concatenate((y_proba_concat2, y_proba),axis=0)
    y_preds_index_concat2=np.concatenate((y_preds_index_concat2, y_preds_index),axis=0)
    
# Recover class names
y_preds2=np.argmax(y_proba_concat2, axis=1)
y_preds2=y_preds2.astype(int)
y_preds2=target_encoder.inverse_transform(y_preds2)

# Save predictions to df
preds_df=preds_df.append(pd.DataFrame({'row_id': y_preds_index_concat2.astype(int), 'target': y_preds2}))
preds_df=preds_df.sort_values('row_id')


# **Check distribution of predictions**

# In[23]:


# Compare distribution of predictions to training set
train_share=pd.DataFrame({'share':100*train_data['target'].value_counts()/len(train_data)})
preds_share=pd.DataFrame({'pred_share':100*preds_df['target'].value_counts()/len(preds_df)})
df_share=pd.concat([train_share,preds_share], axis=1)
df_share


# # Submission

# In[24]:


# Save to csv
preds_df.to_csv('submission.csv', index=False)

# Check format
preds_df.head()

