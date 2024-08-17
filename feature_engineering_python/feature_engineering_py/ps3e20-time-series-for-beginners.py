#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This is the second Playground competition this year where we have time-series problem. The previous competition had a problem where every country in the test dataset had no almost no variation. Hopefully, there won't be such problem this time.
# 
# In this competition, we have to forecast emission in multiple locations in Rwanda in 2022. We are using historical data from January 1st, 2019 until December 31st, 2021 to train our models.
# 
# **Note**: All features in this competition, as it turns out, are complete noises. I recommend you to skip to the addendum or go read [@kdmitrie](https://www.kaggle.com/code/kdmitrie/pgs320-the-shortest-solution-lb-22-97), [@danbraswell](https://www.kaggle.com/code/danbraswell/no-ml-public-lb-23-02231), or [@patrick0302](https://www.kaggle.com/code/patrick0302/align-2019-2020-emission-values-with-2021)'s notebooks if you really care about getting the best result in the competition.

# # Loading Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import folium
import re
import missingno
import geopy

from category_encoders import OneHotEncoder, MEstimateEncoder, GLMMEncoder, OrdinalEncoder, CatBoostEncoder
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, KFold, TimeSeriesSplit
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor, StackingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import HuberRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, roc_auc_score, roc_curve
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

sns.set_theme(style = 'white', palette = 'colorblind')
pal = sns.color_palette('colorblind')

pd.set_option('display.max_rows', 100)


# In[2]:


train = pd.read_csv(r'../input/playground-series-s3e20/train.csv')
test_1 = pd.read_csv(r'../input/playground-series-s3e20/test.csv')

train.drop('ID_LAT_LON_YEAR_WEEK', axis = 1, inplace = True)
test = test_1.drop('ID_LAT_LON_YEAR_WEEK', axis = 1)


# # Descriptive Statistics
# 
# Let's know our data descriptively first before diving in! We start by taking a peek at the top 10 row of each dataset, and then see the statistics of each features.

# In[3]:


train.head(10)


# We have 75 columns in the training dataset. This means that there are actually 74 features that we can use, which is a lot!. Let's summarize all of them into a table below.

# In[4]:


desc = pd.DataFrame(index = list(train))
desc['count'] = train.count()
desc['nunique'] = train.nunique()
desc['%unique'] = desc['nunique'] / len(train) * 100
desc['null'] = train.isnull().sum()
desc['type'] = train.dtypes
desc = pd.concat([desc, train.describe().T.drop('count', axis = 1)], axis = 1)
desc


# There are a lot of features with missing value in the dataset, some even only have less than 1% non-missing values. Imputation will be one of the major challenge in this competition. Let's put those feature names into a list.

# In[5]:


uv_aerosol_layer = [col for col in train.columns if 'UvAerosolLayerHeight' in col]


# In[6]:


test.head(10)


# Test dataset seems to be similar to train dataset here. Let's try to see the statistics.

# In[7]:


desc = pd.DataFrame(index = list(test))
desc['count'] = test.count()
desc['nunique'] = test.nunique()
desc['%unique'] = desc['nunique'] / len(test) * 100
desc['null'] = test.isnull().sum()
desc['type'] = test.dtypes
desc = pd.concat([desc, test.describe().T.drop('count', axis = 1)], axis = 1)
desc


# There is no categorical feature at all in both datasets. The null values proportion also seems to be the same. Another interesting thing is that there are less weeks in test dataset than train dataset. This means that we can just ignore some periods from the train data.

# # Missing Value

# In[8]:


plt.figure(figsize = (10, 5), dpi = 300)

missingno.matrix(train)
plt.title('Missing Value Matrices', fontsize = 40, weight = 'bold')
plt.show()


# # Emission Distribution
# 
# Let's see the distribution of the emission first.

# In[9]:


plt.figure(figsize = (20, 10), dpi = 300)

sns.kdeplot(data = train, x = 'emission', fill = True)
    
plt.title('Emission Distribution', fontsize = 24, fontweight = 'bold')
plt.show()


# It looks like our target is extremely skewed positively, but there is no need to do anything to it.

# # Emission Trend
# 
# Now let's start visualizing our emission trend. We can try spotting any trends visible in the visualization! We begin by visualizing emission over week per year.

# In[10]:


plt.figure(figsize = (20, 10), dpi = 300)

sns.lineplot(data = train, x = 'week_no', y = 'emission', hue = 'year', errorbar = None, palette = 'colorblind')
    
plt.title('Emission Over Week Per Year', fontsize = 24, fontweight = 'bold')
plt.show()


# There is consistent jump around 15th week and the 42nd week. You can also see the slump in 2020 because of CoVID-19, even if that does not stop the usual massive spike that happens around April from happening. Now, let's try dividing the emission by location. We can also try to do log-transformation for more understandable visualization (don't actually do this for training and inference though).
# 
# Thanks to @kawaiicoderuwu for the code.

# In[11]:


#thanks to @kawaiicoderuwu
#https://www.kaggle.com/competitions/playground-series-s3e20/discussion/428320

year_week = pd.to_datetime((train.year * 100 + train.week_no).astype('str') + '0', format = '%Y%W%w')

plt.figure(figsize = (20, 10), dpi = 300)

sns.lineplot(x = year_week, y = np.log1p(train.emission), hue = (train.longitude.astype('str') + train.latitude.astype('str')), errorbar = None)

plt.legend().remove()
plt.title('Log Emission Over Time Per Location', fontsize = 24, fontweight = 'bold')
plt.show()


# We can see that some locations have extremely low emission that it might as well be zero. Two locations, however, have extremely high emission that they somehow separate themselves from the rest.

# # Correlation
# 
# In order to be able to describe relationship between features, we can try creating correlation matrix as follows.

# In[12]:


def heatmap(dataset, label = None):
    columns = list(dataset)
    corr = dataset.corr()
    plt.figure(figsize = (14, 10), dpi = 300)
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask = mask, cmap = 'viridis', linewidths = .1)
    plt.yticks(range(len(columns)), columns, fontsize = 5)
    plt.xticks(range(len(columns)), columns, fontsize = 5)
    plt.title(f'{label} Dataset Correlation Matrix\n', fontsize = 25, weight = 'bold')
    plt.show()


# In[13]:


heatmap(train, 'Train')


# There are a lot of things going on in the visualization it seems. We can try to use hierarchial clustering to simplify the information. However, in order to do that, we need to drop `UvAerosolLayerHeight` feature group because there are too many null values in it.

# In[14]:


def distance(data, label = ''):
    #thanks to @sergiosaharovsky for the fix
    corr = data.corr(method = 'spearman')
    dist_linkage = linkage(squareform(1 - abs(corr)), 'complete')
    
    plt.figure(figsize = (10, 8), dpi = 300)
    dendro = dendrogram(dist_linkage, labels=data.columns, leaf_rotation=90, )
    plt.title(f'Feature Distance in {label} Dataset', weight = 'bold', size = 22)
    plt.show()


# In[15]:


distance(train.drop(uv_aerosol_layer, axis = 1), 'Train')


# Now we can easily see which features are the most similar and which aren't. Interestingly, `emission` doesn't seem to have any features that are similar to it. The closest one is `longitude`, which doesn't seem to be that similar in the first place.

# # Geographical Visualization
# 
# We have latitude and longitude of a coordinate as a feature in our dataset, so why don't we try to visualize it? (Thanks to @inversion and @kenpachi99 for the code)

# In[16]:


#thanks to @inversion and @kenpachi99
#https://www.kaggle.com/code/inversion/getting-started-eda

train_coords = train.drop_duplicates(subset = ['latitude', 'longitude'])
geometry = gpd.points_from_xy(train_coords.longitude, train_coords.latitude)
geo_df = gpd.GeoDataFrame(
    train_coords[["latitude", "longitude"]], geometry=geometry
)

# Create a canvas to plot your map on
all_data_map = folium.Map(prefer_canvas=True)

# Create a geometry list from the GeoDataFrame
geo_df_list = [[point.xy[1][0], point.xy[0][0]] for point in geo_df.geometry]

# Iterate through list and add a marker for each location
for coordinates in geo_df_list:

    # Place the markers 
    all_data_map.add_child(
        folium.CircleMarker(
            location=coordinates,
            radius = 1,
            weight = 4,
            zoom =10,
            color =  "red"),
        )
all_data_map.fit_bounds(all_data_map.get_bounds())
all_data_map


# Interestingly, the coordinate does not only come from Rwanda, but also from region surrounding it, such as Democratic Republic of Congo, Tanzania, Burundi, and Uganda.

# # Country and State Extraction
# 
# We can try extracting the address from coordinate with GeoPy libraries and Nominatim API inside it.

# In[17]:


def location_finder(): 
    #put all unique coordinates into a DataFrame
    location = (train.latitude.astype('str') + ', ' + train.longitude.astype('str')).drop_duplicates().reset_index()
    
    #define geolocator as Nominatim API
    geolocator = Nominatim(user_agent = 'Iqbal Syah Akbar', timeout = 10)
    
    #define rate limiter to prevent too much requests
    rgeocode = RateLimiter(geolocator.reverse, min_delay_seconds = 1)
    return location[0].apply(rgeocode)

extraction_result = location_finder()


# The resulting files is actually in JSON. We need to preprocess the data again so we can load it into DataFrame.

# In[18]:


get_country = lambda x: x.raw['address']['country'] 
get_state = lambda x: x.raw['address']['state']
get_full_address = lambda x: x.raw['display_name']

location = train[['latitude', 'longitude']].drop_duplicates().reset_index().drop('index', axis = 1)
location['country'] = extraction_result.apply(get_country) 
location['state'] = extraction_result.apply(get_state)
location['full_address'] = extraction_result.apply(get_full_address)

location


# # Country Proportion
# 
# Now that we have done extracting the address, we can see how many coordinates are located within which country.

# In[19]:


fig, ax = plt.subplots(1, 2, figsize = (16, 5))
ax = ax.flatten()

ax[0].pie(
    location['country'].value_counts(), 
    shadow = True, 
    explode = [.1 for i in range(0, 5)], 
    autopct = '%1.f%%',
    textprops = {'size' : 14, 'color' : 'white'}
)

sns.countplot(data = location, y = 'country', ax = ax[1], order = location['country'].value_counts().index)
ax[1].yaxis.label.set_size(20)
plt.yticks(fontsize = 12)
ax[1].set_xlabel('Count', fontsize = 20)
ax[1].set_ylabel(None)
plt.xticks(fontsize = 12)

fig.suptitle('Country Proportion', fontsize = 25, fontweight = 'bold')
plt.tight_layout()


# As expected, Rwanda has the most coordinates as it becomes the topic of this competition. What surprises us, however, is that its proportion is only 32%.

# # Preparation
# 
# We will split train dataset into predictors dataset (`X`) and target dataset (`y`). LeaveOneGroupOut will be used for cross-validation strategy to split the dataset based on `year`.

# In[20]:


X = train.copy()
y = X.pop('emission')

seed = 42
k = LeaveOneGroupOut()

np.random.seed(seed)


# # Feature Engineering
# 
# For feature engineering, we will define two classes for our model pipeline. One is to drop features that aren't important, and another one is to extract date features from `year` and `week`.

# In[21]:


def feature_dropper(x):
    return x[['latitude', 'longitude', 'week_no', 'is_covid']]

FeatureDropper = FunctionTransformer(feature_dropper)


# In[22]:


def date_processor(x):
    x_copy = x.copy()
    year_week = pd.to_datetime((x_copy.year * 100 + x_copy.week_no).astype('str') + '0', format = '%Y%W%w')
    x_copy['month'] = year_week.dt.month
    x_copy['is_covid'] = (x_copy.year == 2020) & (x_copy.week_no > 8)
    return x_copy

DateProcessor = FunctionTransformer(date_processor)


# # Cross-Validation Function
# 
# This cross-validation function is built in a way so we can detect overfitting easily and to easily get the out-of-fold prediction.

# In[23]:


def cross_val_score(model, cv = k, label = ''):
    
    X = train.copy()
    y = X.pop('emission')
    
    #initiate prediction arrays and score lists
    val_predictions = np.zeros((len(train)))
    #train_predictions = np.zeros((len(train)))
    train_scores, val_scores = [], []
    
    #training model, predicting prognosis probability, and evaluating log loss
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, X.year)):
        
        #define train set
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        
        #define validation set
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]
        
        #train model
        model.fit(X_train, y_train)
        
        #make predictions
        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val)
                  
        val_predictions[val_idx] += val_preds
        
        #evaluate model for a fold
        train_score = mean_squared_error(y_train, train_preds, squared = False)
        val_score = mean_squared_error(y_val, val_preds, squared = False)
        
        #append model score for a fold to list
        train_scores.append(train_score)
        val_scores.append(val_score)
    
    print(f'Val Score: {np.mean(val_scores):.5f} ± {np.std(val_scores):.5f} | Train Score: {np.mean(train_scores):.5f} ± {np.std(train_scores):.5f} | {label}')
    
    return val_scores, val_predictions


# # Model
# 
# We will train linear regression, random forest, and gradient boosting models without any parameter tuning to see their CV score, and then select which one is the best.

# In[24]:


score_list, oof_list = pd.DataFrame(), pd.DataFrame()

models = [
    #('ridge', Ridge(random_state = seed)),
    ('rf', RandomForestRegressor(random_state = seed)),
    ('et', ExtraTreesRegressor(random_state = seed)),
    ('xgb', XGBRegressor(random_state = seed)),
    ('lgb', LGBMRegressor(random_state = seed)),
    ('cb', CatBoostRegressor(random_state = seed, verbose = 0)),
    #('gb', GradientBoostingRegressor(random_state = seed)),
    #('hgb', HistGradientBoostingRegressor(random_state = seed))
]


# In[25]:


for (label, model) in models:
     score_list[label], oof_list[label] = cross_val_score(
         make_pipeline(DateProcessor, FeatureDropper, model),
         label = label,
     )


# In[26]:


plt.figure(figsize = (8, 4), dpi = 300)
sns.barplot(data = score_list.reindex((score_list).mean().sort_values().index, axis = 1), palette = 'viridis', orient = 'h')
plt.title('Score Comparison', weight = 'bold', size = 20)
plt.show()


# Random Forest has the best CV score out of all models, followed by Extra Trees. On the other hand, LightGBM is the worst model, followed by CatBoost.

# In[27]:


plt.figure(figsize = (20, 10), dpi = 300)

sns.lineplot(x = year_week, y = train.emission, errorbar = None)
sns.lineplot(x = year_week, y = oof_list.rf, errorbar = None)
    
plt.title('Actual vs OOF Prediction', fontsize = 24, fontweight = 'bold')
plt.show()


# # Ensemble
# 
# Now that we have evaluated each models and get the OOF prediction, we can try using Voting Ensemble. First, we will determine the weight by fitting Ridge on the OOF prediction and the actual emission and use the coefficient, and then we can put it inside the Voting Ensemble.

# In[28]:


weights = Ridge(random_state = seed).fit(oof_list, y).coef_

pd.DataFrame(weights, index = oof_list.columns, columns = ['weight per model'])


# In[29]:


voter = make_pipeline(
    DateProcessor,
    FeatureDropper,
    VotingRegressor(models, weights = weights)
)

_ = cross_val_score(voter)


# # Retraining
# 
# We will retrain our model on the entire dataset this time to get the optimal result.

# In[30]:


voter.fit(X, y)
prediction = voter.predict(test) * 1.05


# # Submission
# 
# Finally, we can create our submission file by creating a DataFrame with ID and target feature and then write it into csv.

# In[31]:


test_1.drop(list(test_1.drop('ID_LAT_LON_YEAR_WEEK', axis = 1)), axis = 1, inplace = True)

test_1['emission'] = prediction
test_1.to_csv('submission.csv', index = False)


# In[32]:


plt.figure(figsize = (20, 10), dpi = 300)

sns.lineplot(x = test.week_no, y = test_1.emission, errorbar = None)
sns.lineplot(x = train.week_no, y = train.emission, hue = train.year, errorbar = None, palette = [pal[1], pal[2], pal[3]])

plt.legend([2022, 2019, 2020, 2021])
    
plt.title('Predicted Emission in 2022 vs Previous Years Emission', fontsize = 24, fontweight = 'bold')
plt.show()


# Thank you for reading!

# # ADDENDUM: Using Previous Years Emission As Submission
# 
# As people have discussed, all features in the dataset are nothing more than noises. Our only choice is to use previous years emission as prediction for 2022 and further. For simplicity, we will use @kdmitrie's code for this. Simply put, we will use the maximum emission for each week and each location across multiple years.
# 
# If you want to upvote this notebook, please don't forget to also upvote [@kdmitrie's notebook](https://www.kaggle.com/code/kdmitrie/pgs320-the-shortest-solution-lb-22-97) too.

# In[33]:


test['emission'] = prediction
train = pd.concat([train, test])

test_1.emission = np.max([train[(train.year == y) & (train.week_no < 49)].emission for y in range(2019, 2022)], axis=0)
test_1.to_csv('hacky_submission.csv', index = False)

