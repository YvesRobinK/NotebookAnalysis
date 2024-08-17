#!/usr/bin/env python
# coding: utf-8

# ### This notebook has the following Notebook as reference on detailed analysis and reasonable way of handling the missing values. Apart from that there are few other good notebooks from where this notebook got some value addition pieces of ideas! I wish to thank my fellow kagglers who compel me to learn and grow!
# 
# ### [Kaggle Notebook] [Jane TF Keras LSTM](https://www.kaggle.com/rajkumarl/jane-tf-keras-lstm) (to fill missing values)
# 

# # 1. CREATE ENVIRONMENT

# In[1]:


# Import necessary libraries
import numpy as np
import pandas as pd
import datatable
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn import tree
import graphviz
import shap
import gc
import warnings

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
SEED = 2222
np.random.seed(SEED)


# # 2. LOAD DATA AND OPTIMIZE MEMORY

# In[2]:


train_path = '../input/jane-street-market-prediction/train.csv'

# use datatable to load big data file
train_file = datatable.fread(train_path).to_pandas()
train_file.info()


# In[3]:


# It is found from info() that there are only two datatypes - float64 and int32
# Reduce memeory usage by adopting suitable datatypes
for c in train_file.columns:
    min_val, max_val = train_file[c].min(), train_file[c].max()
    if train_file[c].dtype == 'float64':
        if min_val>np.finfo(np.float16).min and max_val<np.finfo(np.float16).max:
            train_file[c] = train_file[c].astype(np.float16)
        elif min_val>np.finfo(np.float32).min and max_val<np.finfo(np.float32).max:
            train_file[c] = train_file[c].astype(np.float32)
    elif train_file[c].dtype == 'int32':
        if min_val>np.iinfo(np.int8).min and max_val<np.iinfo(np.int8).max:
            train_file[c] = train_file[c].astype(np.int8)
        elif min_val>np.iinfo(np.int16).min and max_val<np.iinfo(np.int16).max:
            train_file[c] = train_file[c].astype(np.int16)
train_file.info()


# ### That's a great reduction in memory usage (around 74% reduction)! It will help us go further efficiently!

# # 3. HANDLING MISSING VALUES

# In[4]:


print('There are %s NAN values in the train data'%train_file.isnull().sum().sum())


# In[5]:


features = [f'feature_{i}' for i in range(130)]
val_range = train_file[features].max()-train_file[features].min()
filler = pd.Series(train_file[features].min()-0.01*val_range, index=features)
# This filler value will be used as a constant replacement of missing values 


# A function to fill all missing values with negative outliers as discussed in the referred notebook
# https://www.kaggle.com/rajkumarl/jane-tf-keras-lstm
def fill_missing(df):
    df[features] = df[features].fillna(filler)
    return df  

train = fill_missing(train_file)

train.info()


# In[6]:


filler.plot(figsize=(20,5),kind='bar',rot=90, color='green')
plt.show()


# In[7]:


print("Now we have %d missing values in our data" %train.isnull().sum().sum())


# In[8]:


# Let's define our target
# Take mean of resp, resp_1, resp_2, resp_3, resp_4
# Create target named 'action' which takes value of 1 if the above mean is higher than 0.5, else 0 
y = train[[c for c in train.columns if 'resp' in c]]
y = (y>0)*1
train['action'] = (y.mean(axis=1)>0.5).astype(np.int8)


# In[9]:


# save memory by deleting useless variables
del val_range
del y
gc.collect()


# # 4. FEATURE ENGINEERING

# ### We cannot handle the whole data during exploration. Rather let's use a small representative part. 
# ### [This notebook](https://www.kaggle.com/rajkumarl/jane-tf-keras-lstm) suggests that Day_242 is one among the days which have most probable number of opportunities per day. 

# In[10]:


day_242 = train.query('date==242')


# ### Generate Features using Natural Logarithm and Square Root. Each feature has negative values, hence shift all values linearly to make them all non_zero positive, because logarithm and square root cannot be applied on negative numbers (yields NAN)

# In[11]:


# Generate Features using Linear shifting, Natural Logarithm and Square Root
for f in [f'feature_{i}' for i in range(1,130)]: # linear shifting to value above 1.0
    day_242['pos_'+str(f)] = (day_242[f]+abs(train[f].min())+1).astype(np.float16)
for f in [f'feature_{i}' for i in range(1,130)]: # Natural log of all the values
    day_242['log_'+str(f)] = np.log(day_242['pos_'+str(f)]).astype(np.float16)
for f in [f'feature_{i}' for i in range(1,130)]: # Square root of all the values
    day_242['sqrt_'+str(f)] = np.sqrt(day_242['pos_'+str(f)]).astype(np.float16)
day_242.info()


# In[12]:


# Linearly shifted values are used for log and sqrt transformations
# However they are useless since we have our original values which are 100% correlated
# Let's drop them from our data
day_242.drop([f'pos_feature_{i}' for i in range(1,130)], inplace=True, axis=1)
day_242.info()


# In[13]:


# Find Correlation among features and remove highly correlated features
corr = day_242.corr(method='pearson').abs().unstack().sort_values(kind='quicksort', ascending=False).reset_index()
corr.rename(columns={'level_0':'feature_A', 'level_1':'feature_B', 0:'Corr_Coeff'}, inplace=True)
corr = corr[corr['Corr_Coeff']<=0.8]
corr.dropna(inplace=True)


# In[14]:


# Let's have a look at correlation table
corr.head()


# In[15]:


# Which features correlate more with the target?
corr[corr['feature_A']=='action'].head(10)


# In[16]:


# Visualize correlation coefficients of features with respect to our target
corr[corr['feature_A']=='action'].iloc[5:40].plot(x='feature_B', y='Corr_Coeff', rot=90, figsize=(20,5), kind='bar', colormap='viridis')
plt.show()


# In[17]:


gen_features = [f for f in day_242.columns if 'feature' in f]
len(gen_features)


# ### Develop a Random Forest Classifer to analyze features further

# In[18]:


train_X, train_y = day_242[gen_features], day_242['action']
model = RandomForestClassifier(max_features='auto', random_state=SEED).fit(train_X, train_y)


# In[19]:


# How does the classification take place in regard of the first tree in Random Forest?
graph = tree.export_graphviz(model.estimators_[0], 
                             out_file=None, 
                             feature_names=gen_features, 
                             rounded=True, 
                             filled=True, 
                             precision=2)
graphviz.Source(graph)


# In[20]:


# How does the classification take place in regard of the second tree in Random Forest?
graph = tree.export_graphviz(model.estimators_[1], 
                             out_file=None, 
                             feature_names=gen_features, 
                             rounded=True, 
                             filled=True, 
                             precision=2)
graphviz.Source(graph)


# # 5. ANALYZING INTERACTION OF FEATURES

# In[21]:


# SHAP values are popular in finding hidden patterns among features
# define an shap value explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(train_X)
# Initialize Javascript Visualization
print('Initializing JavaScript visualization')
shap.initjs() 


# ### Visualize Features and their interaction effects with SHAP values

# In[22]:


for i in range(1,20):
    shap.dependence_plot(f'feature_{i}', shap_values[1], train_X)


# ### Insights: features 1 to 19
# #### Features 3, 4, 5, 6, 7, 17 and 19 have good effects on predicting target
# #### Features and Corresponding interacting features can be (3, 6), (5, log 42), (6, log 42), (9, log 20), (11, log 99), (12, 66), (15, 26), (19, 26)

# In[23]:


for i in range(20,40):
    shap.dependence_plot(f'feature_{i}', shap_values[1], train_X)


# ### Insights: features 20 to 39
# #### Features 21, 25, 26, 27, 28, 37, 38 and 39 have good effects on predicting target
# #### Features and Interacting features can be (21, log 42), (22, log 37), (28, log 39), (37, 45), (39, 95)

# In[24]:


for i in range(40,60):
    shap.dependence_plot(f'feature_{i}', shap_values[1], train_X)


# ### Insights: features 40 to 59
# #### Features 40, 44, 45, 53, 54, 55, 57, and 58 have good effects on predicting target
# #### Features do not seem to have remarkable interactions among them

# In[25]:


for i in range(60,80):
    shap.dependence_plot(f'feature_{i}', shap_values[1], train_X)


# ### Insights: features 60 to 79
# #### Features 60, 61, 62, 63, 64, 65, 66,67, 68, 69, and 71 have good effects on predicting target
# #### Features and Interacting features can be (65, log 91), (66, 0), (74, log 103)

# In[26]:


for i in range(80,100):
    shap.dependence_plot(f'feature_{i}', shap_values[1], train_X)


# ### Insights: features 80 to 99
# #### Features 80, 81, 82, 84, 86, 87, 89, 90, 94, 98, and 99 have good effects on predicting target
# #### Features and Interacting features can be (81, log 66), (88, sqrt 29), (92, sqrt 95), (94, 65), (96, log 67), (98, log 20), (99, log 126)

# In[27]:


for i in range(100,120):
    shap.dependence_plot(f'feature_{i}', shap_values[1], train_X)


# ### Insights: features 100 to 119
# #### Features 101, 103, 104, 107, 110, 113, 114, 115, 116, 118, and 119 have good effects on predicting target
# #### Features and Interacting features can be (101, 4), (109, log 7), (111, log 87), (112, log 97), (114, 40), (118, log 112)

# In[28]:


for i in range(120,130):
    shap.dependence_plot(f'feature_{i}', shap_values[1], train_X)


# ### Insights: features 120 to 129
# #### Features 121, 123, 124, 125, 127, and 129 have good effects on predicting target
# #### Features and Interacting features can be (122, 35)

# ### We are about to generate new and powerful features by properly interacting above found, correlated features

# In[29]:


# Have a copy of dataframe to experiment
sam = day_242.copy()


# In[30]:


print(sam.feature_0.value_counts())
# feature_0 is a categorical variable
# Make it OneHot
sam['feature_0']=((sam['feature_0']+1)/2).astype(np.int8)
print("\nAfter One Hot...\n", sam.feature_0.value_counts())


# ## Transforming possible features to higher order forms

# In[31]:


# From the Shap Dependence plots above, the following features seem to have cubic relationship with target
cubic = [37, 39, 67, 68, 89, 98, 99, 118, 119, 121, 124, 125, 127]
for i in cubic:
    f = f'feature_{i}'
    threes = np.full((len(sam[f])), 3)
    sam['cub_'+f] =np.power(sam[f], threes) 


# In[32]:


# From the Shap Dependence plots above, the following features seem to have quadratic relationship with target
quad = [6, 37, 39, 40, 53, 60, 61, 62, 63, 64, 67, 68, 89, 98, 99, 101, 113, 116, 118, 119, 121, 123, 124, 125, 127]
for i in quad:
    f = f'feature_{i}'
    sam['quad_'+f] =np.square(sam[f]) 


# In[33]:


# Test the correlation of newly generated features with respect to target 
new = sam.corr(method='pearson').abs().unstack().sort_values(kind='quicksort', ascending=False).reset_index()
new.rename(columns={0:'coeff'}, inplace=True)
new = new[new['coeff']<0.8]
new.dropna(inplace=True)


# In[34]:


# Test cubic features
new[new['level_0'].str.contains('action')][new['level_1'].str.contains('cub')].head()


# In[35]:


# Test a random feature
new[new['level_0'].str.contains('action')][new['level_1'].str.contains('67')].head(6)


# ### Great Success. `feature_67` (randomly selected) has negligible correlation with target. But its cubic and quadratic orders exhibit excellent correlation! 

# ## Create features by manipulating two different features by closely observing patterns in SHAP plots

# In[36]:


# features that can be added together or subtracted
add_pairs = [(3,6), (15,26), (19,26), (30,37), (34,33), (35,39),(94,65), (101,4)]
for i,j in add_pairs:
    sam[f'add_{i}_{j}'] = sam[f'feature_{i}']+sam[f'feature_{j}']
    sam[f'sub_{i}_{j}'] = sam[f'feature_{i}']-sam[f'feature_{j}']

add_log_pairs = [(9,20), (22,37), (28,39), (29,25), (65,91), (74,103),(99,126), (109,7), (111,87), (112,97), (118,112)]
for i,j in add_log_pairs:
    sam[f'add_{i}_log{j}'] = sam[f'feature_{i}']+sam[f'log_feature_{j}']
    sam[f'sub_{i}_log{j}'] = sam[f'feature_{i}']-sam[f'log_feature_{j}']


# In[37]:


# features that can be multiplied together
mul_pairs = [(5,42), (12,66), (37,45), (39,95), (122,35)]
for i,j in mul_pairs:
    sam[f'mul_{i}_{j}'] = sam[f'feature_{i}']*sam[f'feature_{j}']

mul_log_pairs = [(5,42), (6,42), (11,99), (21,42), (81,66), (98,20), (122,35)]
for i,j in mul_log_pairs:
    sam[f'mul_{i}_log{j}'] = sam[f'feature_{i}']*sam[f'log_feature_{j}']


# In[38]:


# Test the correlation of newly generated features with respect to target 
n = sam.corr(method='pearson').abs().unstack().sort_values(kind='quicksort', ascending=False).reset_index()
n.rename(columns={0:'coeff'}, inplace=True)
n = n[n['coeff']<0.9]
n.dropna(inplace=True)


# In[39]:


# Test latest features by plotting their correlation
corr_action = n[n['level_0']=='action'].iloc[5:40, :].reset_index()
ax = corr_action.plot(y='coeff', kind='bar', xticks=np.arange(35), rot=90, figsize=(16,5))
ax.set_xticklabels(corr_action['level_1'])
plt.show()


# ### It can be visualized that a lot many generated features show great correlation with target than their original counterparts
# 
# ### For instance, feature `sub_3_6` is found to be more correlated to target than individual features `feature_3` or `feature_6`

# In[40]:


print('Feature Generation completed')


# # 6. FEATURE SELECTION

# ### We generated a lot of features. Since our dataset is in Gigabytes, it is mandatory that we have to cut useless features down. At the same time, we have to use a little larger portion of dataset to make feature selection.
# ### Let's choose data from day_201 to day_300 for feature selection. At first, all the feature generation steps must be applied to those data.

# In[41]:


def feature_transforms(df):
    # Generate Features using Linear shifting, Natural Logarithm and Square Root
    for f in [f'feature_{i}' for i in range(1,130)]: 
        # linear shifting to value above 1.0
        df['pos_'+str(f)] = (df[f]+abs(train[f].min())+1).astype(np.float16)
    for f in [f'feature_{i}' for i in range(1,130)]: 
        # Natural log of all the values
        df['log_'+str(f)] = np.log(df['pos_'+str(f)]).astype(np.float16)
    for f in [f'feature_{i}' for i in range(1,130)]: 
        # Square root of all the values
        df['sqrt_'+str(f)] = np.sqrt(df['pos_'+str(f)]).astype(np.float16)
    
    # Linearly shifted values are used for log and sqrt transformations
    # However they are useless since we have our original values which are 100% correlated
    # Let's drop them from our data
    df.drop([f'pos_feature_{i}' for i in range(1,130)], inplace=True, axis=1)
    
    # From the Shap Dependence plots, the following features seem to have cubic relationship with target
    cubic = [37, 39, 67, 68, 89, 98, 99, 118, 119, 121, 124, 125, 127]
    for i in cubic:
        f = f'feature_{i}'
        threes = np.full((len(df[f])), 3)
        df['cub_'+f] =np.power(df[f], threes) 
        
    # From the Shap Dependence plots, the following features seem to have quadratic relationship with target
    quad = [6, 37, 39, 40, 53, 60, 61, 62, 63, 64, 67, 68, 89, 98, 99, 101, 113, 116, 118, 119, 121, 123, 124, 125, 127]
    for i in quad:
        f = f'feature_{i}'
        df['quad_'+f] =np.square(df[f]) 
    
    return df


# In[42]:


def manipulate_pairs(df):
    # features that can be added together or subtracted
    add_pairs = [(3,6), (15,26), (19,26), (30,37), (34,33), (35,39),(94,65), (101,4)]
    for i,j in add_pairs:
        df[f'add_{i}_{j}'] = df[f'feature_{i}']+df[f'feature_{j}']
        df[f'sub_{i}_{j}'] = df[f'feature_{i}']-df[f'feature_{j}']

    add_log_pairs = [(9,20), (22,37), (28,39), (29,25), (65,91), (74,103),(99,126), (109,7), (111,87), (112,97), (118,112)]
    for i,j in add_log_pairs:
        df[f'add_{i}_log{j}'] = df[f'feature_{i}']+df[f'log_feature_{j}']
        df[f'sub_{i}_log{j}'] = df[f'feature_{i}']-df[f'log_feature_{j}']
    # features that can be multiplied together
    mul_pairs = [(5,42), (12,66), (37,45), (39,95), (122,35)]
    for i,j in mul_pairs:
        df[f'mul_{i}_{j}'] = df[f'feature_{i}']*df[f'feature_{j}']

    mul_log_pairs = [(5,42), (6,42), (11,99), (21,42), (81,66), (98,20), (122,35)]
    for i,j in mul_log_pairs:
        df[f'mul_{i}_log{j}'] = df[f'feature_{i}']*df[f'log_feature_{j}']
    return df


# In[43]:


# Obtain data from day 201 to 300
df = train.query('date>200')
df = df.query('date<=300')


# In[44]:


# Perform feature generation
df = feature_transforms(df)
df = manipulate_pairs(df)


# In[45]:


# Number of features we have now
latest_features = df.columns.drop(['action','resp','resp_1','resp_2','resp_3','resp_4'])
len(latest_features)


# In[46]:


# Feature selection
# Select 100 features (100 is random, can be varied)
selector = SelectKBest(f_classif, k=100)
temp = selector.fit_transform(df[latest_features], df['action'])
df_new = pd.DataFrame(selector.inverse_transform(temp), index=df.index, columns=latest_features)
selected_features = df_new.columns[df_new.var() != 0]
print(selected_features)


# ### Conclusion Note: These `selected_features` can be applied to any ML/DL model to check for performance. If these features outperform or underperform original features with your model, kindly comment here! I develop a model with these features on my own. It will be published soon. 

# ### Thank you for your time!
