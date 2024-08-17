#!/usr/bin/env python
# coding: utf-8

# # Part 1: Best Practices for Picking a Machine Learning Model
# 
# The number of shiny models out there can be overwhelming, which means a lot of times people fallback on a few they trust the most, and use them on all new problems. This can lead to sub-optimal results.
# 
# Today we're going to learn how to quickly and efficiently narrow down the space of available models to find those that are most likely to perform best on your problem type. We'll also see how we can keep track of our models' performances using Weights and Biases and compare them.
# 
# 
# ## What We'll Cover
# - Model selection in competitive data science vs real world
# - A Royal Rumble of Models
# - Comparing Models
# 
# Let's get started!
# 
# ## If you like this kernel, please give it an upvote. Thank you! :)
# 
# Unlike Lord of the Rings, in machine learning there is no one ring (model) to rule them all. Different classes of models are good at modeling the underlying patterns of different types of datasets. For instance, decision trees work well in cases where your data has a complex shape:
# 
# ![](https://paper-attachments.dropbox.com/s_4F3706A8D77436E0E5FD3135841A28E08D5140BFE82FFC2240B0ABB0C9741645_1568921166295_Screenshot+2019-09-19+12.25.23.png)
# 
# Whereas linear models work best where the dataset is linearly separable:
# 
# ![](https://paper-attachments.dropbox.com/s_4F3706A8D77436E0E5FD3135841A28E08D5140BFE82FFC2240B0ABB0C9741645_1568921166358_Screenshot+2019-09-19+12.25.35.png)
# 
# 
# Before we begin, let’s dive a little deeper into the disparity between model selection in the real world vs for competitive data science.
# 
# 
# # Model selection in competitive data science vs real world
# 
# As William Vorhies said in his blog post “The Kaggle competitions are like formula racing for data science. Winners edge out competitors at the fourth decimal place and like Formula 1 race cars, not many of us would mistake them for daily drivers. The amount of time devoted and the sometimes extreme techniques wouldn’t be appropriate in a data science production environment.”
# Kaggle models are indeed like racing cars, they're not built for everyday use. Real world production models are more like a Lexus - reliable but not flashy.
# Kaggle competitions and the real world optimize for very different things, with some key differences being:
# 
# ## Problem Definition
# 
# The real world allows you to define your problem and choose the metric that encapsulates the success of your model. This allows you to optimize for a more complex utility function than just a singular metric, where Kaggle competitions come with a single pre-defined metric and don't let you define the problem efficiently.
# 
# 
# ## Metrics
# 
# In the real world we care about inference and training speeds, resource and deployment constraints and other performance metrics, whereas in Kaggle competitions the only thing we care about is the one evaluation metric. Imagine we have a model with 0.98 accuracy that is very resource and time intensive, and another with 0.95 accuracy that is much faster and less compute intensive. In the real world, for a lot of domains we might prefer the 0.95 accuracy model because maybe we care more about the time to inference. In Kaggle competitions, it doesn't matter how long it takes to train the model or how many GPUs it requires, higher accuracy is always better.
# 
# 
# ## Interpretability
# 
# Similarly in the real world, we prefer simpler models that are easier to explain to stakeholders, whereas in Kaggle we pay no heed to model complexity. Model interpretability is important because it allows to take concrete actions to solve the underlying problem. For example, in the real world looking at our model and being able to see a correlation between a feature (e.g. potholes on a street), and the problem (e.g. likelihood of car accident on the street), is more helpful than increasing the prediction accuracy by 0.005%.
# 
# 
# ## Data Quality
# 
# Finally in Kaggle competitions, our dataset is collected and wrangled for us. Anyone who's done data science knows that is almost never the case in real life. But being able to collect and structure our data also gives us more control over the data science process.
# 
# 
# ## Incentives
# 
# All this incentivizes a massive amount of time spent tuning our hyperparameters to extract the last drops of performance from our model, and at times convoluted feature engineer methodologies. While Kaggle competitions are an excellent way to learn data science and feature engineering, they don't address real world concerns like model explainability, problem definition, or deployment constraints.
# 
# 
# # A Royal Rumble of Models
# 
# It’s time to start selecting models!
# 
# When picking our initial set of models to test, we want to be mindful of a few things:
# 
# ## Pick a diverse set of initial models
# 
# Different classes of models are good at modeling different kinds of underlying patterns in data. So a good first step is to quickly test out a few different classes of models to know which ones capture the underlying structure of your dataset most efficiently! Within the realm of our problem type (regression, classification, clustering) we want to try a mixture of tree based, instance based, and kernel based models. Pick a model from each class to test out. We'll talk more about the different model types in the 'models to try' section below.
# 
# 
# ## Try a few different parameters for each model
# 
# While we don't want to spend too much time finding the optimal set of hyper-parameters, we do want to try a few different combinations of hyper-parameters to allow each model class to have the chance to perform well.
# 
# 
# ## Pick the strongest contenders
# 
# We can use the best performing models from this stage to give us intuition around which class of models we want to further dive into. Your Weights and Biases dashboard will guide you to the class of models that performed best for your problem.
# 
# 
# ## Dive deeper into models in the best performing model classes.
# 
# Next we select more models belonging to the best performing classes of models we shortlisted above! For example if linear regression seemed to work best, it might be a good idea to try lasso or ridge regression as well.
# 
# 
# ## Explore the hyper-parameter space in more detail.
# 
# At this stage, I'd encourage you to spend some time tuning the hyper-parameters for your candidate models. (The next post in this series will dive deeper into the intuition around selecting the best hyper-parameters for your models.) At the end of this stage you should have the best performing versions of all your strongest models.
# 
# 
# ## Making the final selection - Kaggle
# 
# - **Pick final submissions from diverse models**
# Ideally we want to select the best models from more than one class of models. This is because if you make your selections from just one class of models and it happens to be the wrong one, all your submissions will perform poorly. Kaggle competitions usually allow you to pick more than one entry for your final submission. I'd recommend choosing predictions made by your strongest models from different classes to build some redundancy into your submissions.
# 
# - **The leaderboard is not your friend, your cross-validation scores are**
# The most important thing to remember is that the public leaderboard is not your friend. Picking you models solely based on your public leaderboard scores will lead to overfitting the training dataset. And when the private leaderboard is revealed after the competition ends, sometimes you might see your rank dropping a lot. You can avoid this little pitfall by using cross-validation when training your models. Then pick the models with the best cross-validation scores, instead of the best leaderboard scores. By doing this you counter overfitting by measuring your model's performance against multiple validation sets instead of just the one subset of test data used by the public leaderboard.
# 
# 
# ## Making the final selection - Real world
# 
# - **Resource constraints**
# Different models hog different types of resources and knowing whether you’re deploying the models on a IoT/mobile device with a small hard drive and processor or a in cloud can be crucial in picking the right model.
# 
# - **Training time vs Prediction time vs Accuracy**
# Knowing what metric(s) you’re optimizing for is also crucial for picking the right model. For instance self driving cars need blazing fast prediction times, whereas fraud detection systems need to quickly update their models to stay up to date with the latest phishing attacks. For other cases like medical diagnosis, we care about the accuracy (or area under the ROC curve) much more than the training times.
# 
# - **Complexity vs Explainability Tradeoff**
# More complex models can use orders of magnitude more features to train and make predictions, require more compute but if trained correctly can capture really interesting patterns in the dataset. This also makes them convoluted and harder to explain though. Knowing how important it is to easily to explain the model to stakeholders vs capturing some really interesting trends while giving up explainability is key to picking a model.
# 
# - **Scalability**
# Knowing how fast and how big your model needs to scale can help you narrow down your choices appropriately.
# 
# - **Size of training data**
# For really large datasets or those with many features, neural networks or boosted trees might be an excellent choice. Whereas smaller datasets might be better served by logistic regression, Naive Bayes, or KNNs.
# 
# - **Number of parameters**
# Models with a lot of parameters give you lots of flexibility to extract really great performance. However there maybe cases where you don’t have the time required to, for instance, train a neural network's parameters from scratch. A model that works well out of the box would be the way to go in this case!
# 
# 
# # Comparing Models
# 
# Weights and Biases lets you track and compare the performance of you models with one line of code.
# Once you have selected the models you’d like to try, train them and simply add **wandb.log({'score': cv_score})** to log your model state. Once you’re done training, you can compare your model performances in one easy dashboard!

# ## Now that we have some context, let's get started!
# 
# I encourage you to fork this kernel and play with the code!

# In[1]:


# Please turn the 'Internet' toggle On in the Settings panel to your left, in order to make changes to this kernel.
# Essentials
import numpy as np
import pandas as pd
import datetime
import random

# Plots
import seaborn as sns
import matplotlib.pyplot as plt

# Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.svm import SVR
from mlxtend.regressor import StackingCVRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

# Stats
from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

# Misc
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

pd.set_option('display.max_columns', None)

# Ignore useless warnings
import warnings
warnings.filterwarnings(action="ignore")
pd.options.display.max_seq_items = 8000
pd.options.display.max_rows = 8000

import os
print(os.listdir("../input/kernel-files"))

# Set random state for numpy
np.random.seed(42)


# In[2]:


# Read in the dataset as a dataframe
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')


# In[3]:


train.head()


# # Feature Engineering

# In[4]:


# Remove the Ids from train and test, as they are unique for each row and hence not useful for the model
train_ID = train['Id']
test_ID = test['Id']
train.drop(['Id'], axis=1, inplace=True)
test.drop(['Id'], axis=1, inplace=True)
train.shape, test.shape


# In[5]:


# log(1+x) transform
train["SalePrice"] = np.log1p(train["SalePrice"])


# In[6]:


# Remove outliers
train.drop(train[(train['OverallQual']<5) & (train['SalePrice']>200000)].index, inplace=True)
train.drop(train[(train['GrLivArea']>4500) & (train['SalePrice']<300000)].index, inplace=True)
train.reset_index(drop=True, inplace=True)


# In[7]:


# Split features and labels
train_labels = train['SalePrice'].reset_index(drop=True)
train_features = train.drop(['SalePrice'], axis=1)
test_features = test

# Combine train and test features in order to apply the feature transformation pipeline to the entire dataset
all_features = pd.concat([train_features, test_features]).reset_index(drop=True)
all_features.shape


# In[8]:


# Fill missing values
# determine the threshold for missing values
def percent_missing(df):
    data = pd.DataFrame(df)
    df_cols = list(pd.DataFrame(data))
    dict_x = {}
    for i in range(0, len(df_cols)):
        dict_x.update({df_cols[i]: round(data[df_cols[i]].isnull().mean()*100,2)})
    
    return dict_x

missing = percent_missing(all_features)
df_miss = sorted(missing.items(), key=lambda x: x[1], reverse=True)
print('Percent of missing data')
df_miss[0:10]


# In[9]:


# Some of the non-numeric predictors are stored as numbers; convert them into strings 
all_features['MSSubClass'] = all_features['MSSubClass'].apply(str)
all_features['YrSold'] = all_features['YrSold'].astype(str)
all_features['MoSold'] = all_features['MoSold'].astype(str)


# In[10]:


def handle_missing(features):
    # the data description states that NA refers to typical ('Typ') values
    features['Functional'] = features['Functional'].fillna('Typ')
    # Replace the missing values in each of the columns below with their mode
    features['Electrical'] = features['Electrical'].fillna("SBrkr")
    features['KitchenQual'] = features['KitchenQual'].fillna("TA")
    features['Exterior1st'] = features['Exterior1st'].fillna(features['Exterior1st'].mode()[0])
    features['Exterior2nd'] = features['Exterior2nd'].fillna(features['Exterior2nd'].mode()[0])
    features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])
    features['MSZoning'] = features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
    
    # the data description stats that NA refers to "No Pool"
    features["PoolQC"] = features["PoolQC"].fillna("None")
    # Replacing the missing values with 0, since no garage = no cars in garage
    for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
        features[col] = features[col].fillna(0)
    # Replacing the missing values with None
    for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
        features[col] = features[col].fillna('None')
    # NaN values for these categorical basement features, means there's no basement
    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        features[col] = features[col].fillna('None')
        
    # Group the by neighborhoods, and fill in missing value by the median LotFrontage of the neighborhood
    features['LotFrontage'] = features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

    # We have no particular intuition around how to fill in the rest of the categorical features
    # So we replace their missing values with None
    objects = []
    for i in features.columns:
        if features[i].dtype == object:
            objects.append(i)
    features.update(features[objects].fillna('None'))
        
    # And we do the same thing for numerical features, but this time with 0s
    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    numeric = []
    for i in features.columns:
        if features[i].dtype in numeric_dtypes:
            numeric.append(i)
    features.update(features[numeric].fillna(0))    
    return features

all_features = handle_missing(all_features)


# In[11]:


# Let's make sure we handled all the missing values
missing = percent_missing(all_features)
df_miss = sorted(missing.items(), key=lambda x: x[1], reverse=True)
print('Percent of missing data')
df_miss[0:10]


# In[12]:


# Fix skewed features
# Fetch all numeric features
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric = []
for i in all_features.columns:
    if all_features[i].dtype in numeric_dtypes:
        numeric.append(i)


# In[13]:


# Find skewed numerical features
skew_features = all_features[numeric].apply(lambda x: skew(x)).sort_values(ascending=False)

high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index

print("There are {} numerical features with Skew > 0.5 :".format(high_skew.shape[0]))
skewness = pd.DataFrame({'Skew' :high_skew})
skew_features.head(10)


# In[14]:


# Normalize skewed features
for i in skew_index:
    all_features[i] = boxcox1p(all_features[i], boxcox_normmax(all_features[i] + 1))


# ## Create interesting features

# ML models have trouble recognizing more complex patterns (and we're staying away from neural nets for this competition), so let's help our models out by creating a few features based on our intuition about the dataset, e.g. total area of floors, bathrooms and porch area of each house.

# In[15]:


all_features['BsmtFinType1_Unf'] = 1*(all_features['BsmtFinType1'] == 'Unf')
all_features['HasWoodDeck'] = (all_features['WoodDeckSF'] == 0) * 1
all_features['HasOpenPorch'] = (all_features['OpenPorchSF'] == 0) * 1
all_features['HasEnclosedPorch'] = (all_features['EnclosedPorch'] == 0) * 1
all_features['Has3SsnPorch'] = (all_features['3SsnPorch'] == 0) * 1
all_features['HasScreenPorch'] = (all_features['ScreenPorch'] == 0) * 1
all_features['YearsSinceRemodel'] = all_features['YrSold'].astype(int) - all_features['YearRemodAdd'].astype(int)
all_features['Total_Home_Quality'] = all_features['OverallQual'] + all_features['OverallCond']
all_features = all_features.drop(['Utilities', 'Street', 'PoolQC',], axis=1)
all_features['TotalSF'] = all_features['TotalBsmtSF'] + all_features['1stFlrSF'] + all_features['2ndFlrSF']
all_features['YrBltAndRemod'] = all_features['YearBuilt'] + all_features['YearRemodAdd']

all_features['Total_sqr_footage'] = (all_features['BsmtFinSF1'] + all_features['BsmtFinSF2'] +
                                 all_features['1stFlrSF'] + all_features['2ndFlrSF'])
all_features['Total_Bathrooms'] = (all_features['FullBath'] + (0.5 * all_features['HalfBath']) +
                               all_features['BsmtFullBath'] + (0.5 * all_features['BsmtHalfBath']))
all_features['Total_porch_sf'] = (all_features['OpenPorchSF'] + all_features['3SsnPorch'] +
                              all_features['EnclosedPorch'] + all_features['ScreenPorch'] +
                              all_features['WoodDeckSF'])
all_features['TotalBsmtSF'] = all_features['TotalBsmtSF'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
all_features['2ndFlrSF'] = all_features['2ndFlrSF'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)
all_features['GarageArea'] = all_features['GarageArea'].apply(lambda x: np.exp(6) if x <= 0.0 else x)
all_features['GarageCars'] = all_features['GarageCars'].apply(lambda x: 0 if x <= 0.0 else x)
all_features['LotFrontage'] = all_features['LotFrontage'].apply(lambda x: np.exp(4.2) if x <= 0.0 else x)
all_features['MasVnrArea'] = all_features['MasVnrArea'].apply(lambda x: np.exp(4) if x <= 0.0 else x)
all_features['BsmtFinSF1'] = all_features['BsmtFinSF1'].apply(lambda x: np.exp(6.5) if x <= 0.0 else x)

all_features['haspool'] = all_features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
all_features['has2ndfloor'] = all_features['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
all_features['hasgarage'] = all_features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
all_features['hasbsmt'] = all_features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
all_features['hasfireplace'] = all_features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)


# ## Feature transformations
# Let's create more features by calculating the log and square transformations of our numerical features. We do this manually, because ML models won't be able to reliably tell if log(feature) or feature^2 is a predictor of the SalePrice.

# In[16]:


def logs(res, ls):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pd.Series(np.log(1.01+res[l])).values)   
        res.columns.values[m] = l + '_log'
        m += 1
    return res

log_features = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',
                 'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea',
                 'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',
                 'TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF',
                 'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','YearRemodAdd','TotalSF']

all_features = logs(all_features, log_features)


# In[17]:


def squares(res, ls):
    m = res.shape[1]
    for l in ls:
        res = res.assign(newcol=pd.Series(res[l]*res[l]).values)   
        res.columns.values[m] = l + '_sq'
        m += 1
    return res 

squared_features = ['YearRemodAdd', 'LotFrontage_log', 
              'TotalBsmtSF_log', '1stFlrSF_log', '2ndFlrSF_log', 'GrLivArea_log',
              'GarageCars_log', 'GarageArea_log']
all_features = squares(all_features, squared_features)


# ## Encode categorical features

# In[18]:


all_features = pd.get_dummies(all_features).reset_index(drop=True)
all_features.shape


# In[19]:


all_features.head()


# In[20]:


all_features.shape


# In[21]:


# Remove any duplicated column names
all_features = all_features.loc[:,~all_features.columns.duplicated()]


# In[22]:


X = all_features.iloc[:len(train_labels), :]
X_test = all_features.iloc[len(train_labels):, :]
X.shape, train_labels.shape, X_test.shape

X_train, X_valid, y_train, y_valid = train_test_split(X, train_labels, test_size=0.15, random_state=42)


# # Picking the right model

# In[23]:


# Please turn the 'Internet' toggle On in the Settings panel to your left, in order to make changes to this kernel.
get_ipython().system('pip install wandb -q')


# In[24]:


# WandB
import wandb
import keras
from wandb.keras import WandbCallback
from sklearn.model_selection import cross_val_score
# Import models (add your models here)
from sklearn import svm
from sklearn.linear_model import Ridge, RidgeCV
from xgboost import XGBRegressor


# ## Model 1 - SVR

# In[25]:


# Initialize wandb run
wandb.init(anonymous='allow', project="pick-a-model")


# In[26]:


get_ipython().run_cell_magic('wandb', '', "# Initialize and fit model (add your classifier here)\nsvr = svm.SVR(C= 20, epsilon= 0.008, gamma=0.0003)\nsvr.fit(X_train, y_train)\n\n# Get CV scores\nscores = cross_val_score(svr, X_train, y_train, cv=5)\n\n# Log scores\nfor score in scores:\n    wandb.log({'cross_val_score': score})\n")


# ## Model 2 - XGBoost

# In[27]:


# Initialize wandb run
wandb.init(anonymous='allow', project="pick-a-model")


# In[28]:


get_ipython().run_cell_magic('wandb', '', "# Initialize and fit model (add your classifier here)\nxgb = XGBRegressor(learning_rate=0.01,\n                       n_estimators=6000,\n                       max_depth=4,\n                       min_child_weight=0,\n                       gamma=0.6,\n                       subsample=0.7,\n                       colsample_bytree=0.7,\n                       objective='reg:linear',\n                       nthread=-1,\n                       scale_pos_weight=1,\n                       seed=27,\n                       reg_alpha=0.00006,\n                       random_state=42)\nxgb.fit(X_train, y_train)\n\n# Get CV scores\nscores = cross_val_score(xgb, X_train, y_train, cv=3)\n\n# Log scores\nfor score in scores:\n    wandb.log({'cross_val_score': score})\n")


# ## Model 3 - Ridge Regression

# In[29]:


# Initialize wandb run
wandb.init(anonymous='allow', project="pick-a-model")


# In[30]:


get_ipython().run_cell_magic('wandb', '', "# Initialize and fit model (add your classifier here)\nridge = Ridge(alpha=1e-3)\nridge.fit(X_train, y_train)\n\n# Get CV scores\nscores = cross_val_score(ridge, X_train, y_train, cv=5)\n\n# Log scores\nfor score in scores:\n    wandb.log({'cross_val_score': score})\n")


# ## Model 4 - Neural Network

# In[31]:


wandb.init(anonymous='allow', project="picking-a-model", name="neural_network")


# In[32]:


get_ipython().run_cell_magic('wandb', '', "# Model\nmodel = Sequential()\nmodel.add(Dense(50, input_dim=378, kernel_initializer='normal', activation='relu'))\nmodel.add(Dense(20, kernel_initializer='normal', activation='relu'))\nmodel.add(Dense(1, kernel_initializer='normal'))\n\n# Compile model\nmodel.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adadelta())\n\nmodel.fit(X_train, y_train, epochs=20, batch_size=10, verbose=0,\n        callbacks=[WandbCallback(validation_data=(X_valid, y_valid))])\n")


# ## Identify the best performing model
# 
# If you go to the runs page generated by Weights & Biases above, you can find how your model performed. If you click on the name of the project, you can compare all your models' metrics together and pick the best one.
# 
# That’s it now you have all the tools you need to pick the right models for your problem!

# # Part II - A Whirlwind Tour of Machine Learning Models
# In Part I, we talked about the part art, part science of picking the perfect machine learning model.
# 
# In Part II, we dive deeper into the different machine learning models you can train and when you should use them!
# 
# In general tree-based models perform best in Kaggle competitions. The other models make great candidates for ensembling. For computer vision challenges, CNNs outperform everything. For natural language processing, LSTMs or GRUs are your best bet!
# 
# With that said, below is a non-exhaustive laundry list of models to try, along with some context for each model.
# 
# ## Regression
# ### Regression → Linear Regression → Vanilla Linear Regression
# 
# Advantages
# 
# - Captures linear relationships in the dataset well
# - Works well if you have a few well defined variables and need a simple predictive model
# - Fast training speed and prediction speeds
# - Does well on small datasets
# - Interpretable results, easy to explain
# - Easy to update the model when new data comes in
# - No parameter tuning required (the regularized linear models below need to tune the regularization parameter)
# - Doesn't need feature scaling (the regularized linear models below need feature scaling)
# - If dataset has redundant features, linear regression can be unstable
# 
# Disadvantages
# 
# - Doesn't work well for non-linear data
# - Low(er) prediction accuracy
# - Can overfit (see regularized models below to counteract this)
# - Doesn't separate signal from noise well – cull irrelevant features before use
# - Doesn't learn feature interactions in the dataset
#         
# ### Regression → Linear Regression → Lasso, Ridge, Elastic-Net Regression
# 
# Advantages
# 
# - These models are linear regression with regularization
# - Help counteract overfitting
# - These models are much better at generalizing because they are simpler
# - They work well when we only care about a few features
# 
# Disadvantages
# 
# - Need feature scaling
# - Need to tune the regularization parameter
# 
# 
# ### Regression → Regression Trees → Decision Tree
# 
# Advantages
# 
# - Fast training speed and prediction speeds
# - Captures non-linear relationships in the dataset well
# - Learns feature interactions in the dataset
# - Great when your dataset has outliers
# - Great for finding the most important features in the dataset
# - Doesn't need feature scaling
# - Decently interpretable results, easy to explain
# 
# Disadvantages
# 
# - Low(er) prediction accuracy
# - Requires some parameter tuning
# - Doesn't do well on small datasets
# - Doesn't separate signal from noise well
# - Not easy to update the model when new data comes in
# - Used very rarely in practice, use ensembled trees instead
# - Can overfit (see ensembled models below)
# 
# 
# ### Regression → Regression Trees → Ensembles
# 
# Advantages
# 
# - Collates predictions from multiple trees
# - High prediction accuracy - does really well in practice
# - Preferred algorithm in Kaggle competitions
# - Great when your dataset has outliers
# - Captures non-linear relationships in the dataset well
# - Great for finding the most important features in the dataset
# - Separates signal vs noise
# - Doesn't need feature scaling
# - Perform really well on high-dimensional data
# 
# Disadvantages
# 
# - Slower training speed
# - Fast prediction speed
# - Not easy to interpret or explain
# - Not easy to update the model when new data comes in
# - Requires some parameter tuning - Harder to tune
# - Doesn't do well on small datasets
# 
# 
# 
# ### Regression → Deep Learning
# 
# Advantages
# 
# - High prediction accuracy - does really well in practice
# - Captures very complex underlying patterns in the data
# - Does really well with both big datasets and those with high-dimensional data
# - Easy to update the model when new data comes in
# - The network's hidden layers reduce the need for feature engineering remarkably
# - Is state of the art for computer vision, machine translation, sentiment analysis and speech recognition tasks
# 
# Disadvantages
# 
# - Very long training speed
# - Need a huge amount of computing power
# - Need feature scaling
# - Not easy to explain or interpret results
# - Need lots of training data because it learns a vast number of parameters
# - Outperformed by Boosting algorithms for non-image, non-text, non-speech tasks
# - Very flexible, come with lots of different architecture building blocks, thus require expertise to design the architecture
# 
# 
# 
# ### Regression → K Nearest Neighbors (Distance Based)
# 
# Advantages
# 
# - Fast training speed
# - Doesn't need much parameter tuning
# - Interpretable results, easy to explain
# - Works well for small datasets (<100k training set)
# 
# Disadvantages
# 
# - Low(er) prediction accuracy
# - Doesn't do well on small datasets
# - Need to pick a suitable distance function
# - Needs feature scaling to work well
# - Prediction speed grows with size of dataset
# - Doesn't separate signal from noise well – cull irrelevant features before use
# - Is memory intensive because it saves every observation
# - Also means they don't work well with high-dimensional data
# 2. Classification - Predict a class or class probabilities
# 
# ## Classification
# ### Classification → Logistic Regression
# 
# Advantages
# 
# - Classifies linearly separable data well
# - Fast training speed and prediction speeds
# - Does well on small datasets
# - Decently interpretable results, easy to explain
# - Easy to update the model when new data comes in
# - Can avoid overfitting when regularized
# - Can do both 2 class and multiclass classification
# - No parameter tuning required (except when regularized, we need to tune the regularization parameter)
# - Doesn't need feature scaling (except when regularized)
# - If dataset has redundant features, linear regression can be unstable
# 
# Disadvantages
# 
# - Doesn't work well for non-linearly separable data
# - Low(er) prediction accuracy
# - Can overfit (see regularized models below)
# - Doesn't separate signal from noise well – cull irrelevant features before use
# - Doesn't learn feature interactions in the dataset
# 
# 
# 
# ### Classification → Support Vector Machines (Distance based)
# 
# Advantages
# 
# - High prediction accuracy
# - Doesn't overfit, even on high-dimensional datasets, so its great for when you have lots of features
# - Works well for small datasets (<100k training set)
# - Work well for text classification problems
# 
# Disadvantages
# 
# - Not easy to update the model when new data comes in
# - Is very memory intensive
# - Doesn't work well on large datasets
# - Not easy to update the model when new data comes in
# - Requires you choose the right kernel in order to work
# - The linear kernel models linear data and works fast
# - The non-linear kernels can model non-linear boundaries and can be slow
# - Use Boosting instead!
# 
# 
# 
# ### Classification → Naive Bayes (Probability based)
# 
# Advantages
# 
# - Performs really well on text classification problems
# - Fast training speed and prediction speeds
# - Does well on small datasets
# - Separates signal from noise well
# - Performs well in practice
# - Simple, easy to implement
# - Works well for small datasets (<100k training set)
# - The naive assumption about the independence of features and their potential distribution lets it avoid overfitting
# - Also if this condition of independence holds, Naive Bayes can work on smaller datasets and can have faster training speed
# - Doesn't need feature scaling
# - Not memory intensive
# - Decently interpretable results, easy to explain
# - Scales well with the size of the dataset
# 
# Disadvantages
# 
# - Low(er) prediction accuracy
# 
# 
# 
# ### Classification → K Nearest Neighbors (Distance Based)
# 
# Advantages
# 
# - Fast training speed
# - Doesn't need much parameter tuning
# - Interpretable results, easy to explain
# - Works well for small datasets (<100k training set)
# 
# Disadvantages
# 
# - Low(er) prediction accuracy
# - Doesn't do well on small datasets
# - Need to pick a suitable distance function
# - Needs feature scaling to work well
# - Prediction speed grows with size of dataset
# - Doesn't separate signal from noise well – cull irrelevant features before use
# - Is memory intensive because it saves every observation
# - Also means they don't work well with high-dimensional data
# 
# 
# 
# ### Classification → Classification Tree → Decision Tree
# 
# Advantages
# 
# - Fast training speed and prediction speeds
# - Captures non-linear relationships in the dataset well
# - Learns feature interactions in the dataset
# - Great when your dataset has outliers
# - Great for finding the most important features in the dataset
# - Can do both 2 class and multiclass classification
# - Doesn't need feature scaling
# - Decently interpretable results, easy to explain
# 
# Disadvantages
# 
# - Low(er) prediction accuracy
# - Requires some parameter tuning
# - Doesn't do well on small datasets
# - Doesn't separate signal from noise well
# - Used very rarely in practice, use ensembled trees instead
# - Not easy to update the model when new data comes in
# - Can overfit (see ensembled models below)
# 
# 
# ### Classification → Classification Tree → Ensembles
# 
# Advantages
# 
# - Collates predictions from multiple trees
# - High prediction accuracy - does really well in practice
# - Preferred algorithm in Kaggle competitions
# - Captures non-linear relationships in the dataset well
# - Great when your dataset has outliers
# - Great for finding the most important features in the dataset
# - Separates signal vs noise
# - Doesn't need feature scaling
# - Perform really well on high-dimensional data
# 
# Disadvantages
# 
# - Slower training speed
# - Fast prediction speed
# - Not easy to interpret or explain
# - Not easy to update the model when new data comes in
# - Requires some parameter tuning - Harder to tune
# - Doesn't do well on small datasets
# 
# 
# 
# ### Classification → Deep Learning
# 
# Advantages
# 
# - High prediction accuracy - does really well in practice
# - Captures very complex underlying patterns in the data
# - Does really well with both big datasets and those with high-dimensional data
# - Easy to update the model when new data comes in
# - The network's hidden layers reduce the need for feature engineering remarkably
# - Is state of the art for computer vision, machine translation, sentiment analysis and speech recognition tasks
# 
# Disadvantages
# 
# - Very long training speed
# - Not easy to explain or interpret results
# - Need a huge amount of computing power
# - Need feature scaling
# - Need lots of training data because it learns a vast number of parameters
# - Outperformed by Boosting algorithms for non-image, non-text, non-speech tasks
# - Very flexible, come with lots of different architecture building blocks, thus require expertise to design the architecture
# 3. Clustering - Organize the data into groups to maximize similarity
# 
# ## Clustering 
# ### Clustering → DBSCAN
# 
# Advantages
# 
# - Scalable to large datasets
# - Detects noise well
# - Don't need to know the number of clusters in advance
# - Doesn't make an assumption that the shape of the cluster is globular
# 
# Disadvantages
# 
# - Doesn't always work if your entire dataset is densely packed
# - Need to tune the density parameters – epsilon and min_samples to the right values to get good results
# 
# 
# ### Clustering → KMeans
# Advantages
# 
# - Great for revealing the structure of the underlying dataset
# - Simple, easy to interpret
# - Works well if you know the number of clusters in advance
# Disadvantages
# 
# - Doesn't always work if your clusters aren't globular and similar in size
# - Needs to know the number of clusters in advance - Need to tune the choice of k clusters to get good results
# - Memory intensive
# - Doesn't scale to large datasets
# 
# ## Misc - Models not included in this post
# - Dimensionality Reduction Algorithms
# - Clustering algorithms - Gaussian Mixture Model and Hierarchical clustering
# - Computer Vision – Convolutional Neural Networks, Image classification, Object Detection, Image segmentation
# - Natural Language Processing – RNNs (LSTM or GRUs)
# - Reinforcement Learning
# 
# 
# ## Ensembling Your Models
# 
# Ensembling models is a really powerful technique that helps reduce overfitting, and make more robust predictions by combining outputs from different models. It is especially an essential tool for winning Kaggle competitions.
# When picking models to ensemble together, we want to pick them from different model classes to ensure they have different strengths and weaknesses and thus capture different patterns in the dataset. This greater diversity leads to lower bias. We also want to make sure their performance is comparable in order to ensure stability of predictions generated.
# We can see here that the blending these models actually resulted in much lower loss than any single model was able to produce alone. Part of the reason is that while all these models are pretty good at making predictions, they get different predictions right and by combining them together, we're able to combine all their different strengths into a super strong model.
# 
# ![](https://paper-attachments.dropbox.com/s_4F3706A8D77436E0E5FD3135841A28E08D5140BFE82FFC2240B0ABB0C9741645_1568838886109_cv_scores.png)
# 
#     # Blend models in order to make the final predictions more robust to overfitting
#     def blended_predictions(X):
#         return ((0.1 * ridge_model_full_data.predict(X)) + \\
#                 (0.2 * svr_model_full_data.predict(X)) + \\
#                 (0.1 * gbr_model_full_data.predict(X)) + \\
#                 (0.1 * xgb_model_full_data.predict(X)) + \\
#                 (0.1 * lgb_model_full_data.predict(X)) + \\
#                 (0.05 * rf_model_full_data.predict(X)) + \\
#                 (0.35 * stack_gen_model.predict(np.array(X))))
# 
# There are 4 types of ensembling (including blending):
# 
# - **Bagging:** Train many base models with different randomly chosen subsets of data, with replacement. Let the base models vote on final predictions. Used in RandomForests.
# - **Boosting:** Iteratively train models and update the importance of getting each training example right after each iteration. Used in GradientBoosting.
# - **Blending:** Train many different types of base models and make predictions on a holdout set. Train a new model out of their predictions, to make predictions on the test set. (Stacking with a holdout set).
# - **Stacking:** Train many different types of base models and make predictions on k-folds of the dataset. Train a new model out of their predictions, to make predictions on the test set.
# 
# 
# ---
# 
# ## If you like this kernel, please give it an upvote. Thank you! :)

# In[ ]:




