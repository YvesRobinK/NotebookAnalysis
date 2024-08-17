#!/usr/bin/env python
# coding: utf-8

# # <p style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#032352; font-size:105%; text-align:left;padding:2.5px; border-bottom: 2px solid #090A0A; background:#EBF6F9" > Package imports
# </p>

# In[1]:


# General imports:-

# Data processing:-
import numpy as np;
import pandas as pd;
import re;
from scipy.stats import mode, iqr, anderson, shapiro, normaltest;
pd.options.display.max_rows = 50;
pd.set_option('display.float_format', '{:,.4f}'.format);

# Geospatial analysis:-
from geopy.distance import geodesic;
from haversine import haversine;

# Visualization:-
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt;
import seaborn as sns;
from matplotlib.colors import LinearSegmentedColormap;

# Others:-
from warnings import filterwarnings; filterwarnings('ignore');
from tqdm.notebook import tqdm;
from termcolor import colored;
from gc import collect;
from IPython.display import clear_output;


# In[2]:


# Model imports:-

from sklearn import datasets;
from sklearn.metrics import mean_squared_error;
from sklearn.feature_selection import mutual_info_regression;
from sklearn.pipeline import Pipeline;
from sklearn.base import BaseEstimator, TransformerMixin;
from sklearn.preprocessing import FunctionTransformer;
from sklearn_pandas import DataFrameMapper, gen_features;
from sklearn.model_selection import KFold;

from lightgbm import LGBMRegressor;
from xgboost import XGBRegressor;
from catboost import CatBoostRegressor;


# # <p style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#032352; font-size:105%; text-align:left;padding:2.5px; border-bottom: 2px solid #090A0A; background:#EBF6F9" > Executive Summary and initial thoughts
# </p>

# <div style="color: #050505;
#            display:fill;
#            border-radius:12px;
#            background-color: #F8F9FA;
#            font-size:110%;
#            font-family: Calibri;
#            letter-spacing: 0.5px;
#            border: 1px solid #000205;
#            ">
# 
# * This competition is a part of the Playground Series for 2023 January edition. It is labelled as Playground Series Episode1-Season3. The assignment involves predicting the median house price for the approchable dataset, using the 8 features provided. The challenge extends from Jan3- Jan10, 2023.     
# * The competition metric is RMSE. Smaller the value, better is the assignment result.   
# * We will also try and elicit EDA based details and possibly elicit inferences from the data apriori.   
# * This is a regression challenge. We plan to use simple ML models as a baseline approach. Tree based approaches like LightGBM, XgBoost and Catboost will be used herewith with parameter tuning and cross-validation.  
# </p>
# </div>

# # <p style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#032352; font-size:105%; text-align:left;padding:2.5px; border-bottom: 2px solid #090A0A; background:#EBF6F9" > Data preprocessing and analysis
# </p>

# <div style="color: #050505;
#            display:fill;
#            border-radius:5px;
#            background-color: #F8F9FA;
#            font-size:110%;
#            font-family: Calibri;
#            letter-spacing: 0px;
#            border: 1.5px solid #000205;
#            ">
# 
# We aim to import the relevant datasets and pre-process them with the below factors- 
# 1. Null and outlier check
# 2. Comparison with alternative datasets - California house prices
# 3. Data transforms and feature creation
# 4. Any centering and scaling strategies as deemed suitable
# </p>
# </div>

# In[3]:


# Assigning global variables to be used throughout:- 
target = 'MedHouseVal';
key_cities  = ['SJC', 'SFO', '51Q', 'SEE', 'MYF','SAN','SDM','SDM','LAX'];
dist_measure = 'haversine';
target_log_xform_req = 'N';
n_folds = 10;

grid_specs = {'visible': True, 'which': 'both', 'linestyle': '--', 
              'color': 'lightgrey', 'linewidth': 0.50};
title_specs = {'fontsize': 12, 'fontweight': 'bold', 'color': 'tab:blue'};

def PrintColor(text:str, color:str = 'blue', attrs:list = ['bold', 'dark']):
    "Prints color outputs using termcolor-colored using a text F-string";
    print(colored(text, color = color, attrs = attrs));


# In[4]:


get_ipython().run_cell_magic('time', '', '\n# Importing the data:-\npath = "/kaggle/input/playground-series-s3e1/";\nxytrain = pd.read_csv(path + \'train.csv\', encoding = \'utf8\', index_col= \'id\');\nxtest = pd.read_csv(path + \'test.csv\', encoding = \'utf8\', index_col= \'id\');\nsub_fl = pd.read_csv(path + \'sample_submission.csv\', encoding = \'utf8\');\n\n# Displaying the datasets:-\nPrintColor(f"\\nTrain data\\n");\ndisplay(xytrain.head(5));\n\nPrintColor(f"\\nTrain data\\n");\ndisplay(xytrain.head(5));\n\nPrintColor(f"\\nTrain data\\n");\ndisplay(xtest.head(5));\n\nPrintColor(f"\\nAlternative House Price dataset- Scikit-learn\\n");\nalt_xytrain= datasets.fetch_california_housing(as_frame=True, return_X_y=False)[\'frame\'];\ndisplay(alt_xytrain.head(5));\n\nPrintColor(f"\\nExtracting features from the competition data\\n");\nfeatures = xtest.columns;\ndisplay(features);\n\nPrintColor(f"\\nConverting the target into existing value * 100_000\\n");\nxytrain[target] = xytrain[target] * 100_000;\nalt_xytrain[target] = alt_xytrain[target] * 100_000;\n\nprint();\n')


# In[5]:


# Preprocessing the city locations in california for further use:-
city_loc = pd.read_csv("/kaggle/input/california-city-locations/CalCityLocation.csv");

city_loc['Location'] = city_loc[['Latitude', 'Longitude']].apply(tuple, axis=1);

PrintColor(f"\nCity locations within California\n");
display(city_loc);

# Obtaining key cities for geo-spatial analysis:-
loc_dict = \
city_loc.loc[city_loc.CityCode.isin(key_cities), ['CityCode', 'Location']].\
set_index('CityCode').to_dict()['Location'];

PrintColor(f"\nSelected city locations within California- dictionary\n");
display(loc_dict);


# ## <p style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#032352; font-size:100%; text-align: left; padding:2.5px; border-bottom: 1px solid #090A0A; background:#EBF6F9" > A. Preprocessing and data augmentation
# </p>
# 
# **We extract basic information and descriptive statistics elements herewith on an augmented data from the competition data and the original dataset.**

# In[6]:


PrintColor("\nExtracting attribute information from the alternative dataset\n");
print(datasets.fetch_california_housing()['DESCR']);


# In[7]:


PrintColor(f"\nExtracting unique records per feature from the competition and alternative data\n");

PrintColor(f"\nTrain data unique records\n");
display(xytrain.nunique());

PrintColor(f"\nTest data unique records\n");
display(xtest.nunique());

PrintColor(f"\nSklearn data unique records\n");
display(alt_xytrain.nunique());


# In[8]:


PrintColor(f"\nData information- train and test competition and alternative data", color = 'red');

PrintColor(f"\nTrain data information\n");
display(xytrain.info());

PrintColor(f"\nTest data information\n");
display(xtest.info());

PrintColor(f"\nScikit-learn data information\n");
display(alt_xytrain.info());


# In[9]:


PrintColor(f"\nData description- train and test competition and alternative data\n");

ntiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99];

_tr = \
xytrain.drop([target], axis=1).\
describe(percentiles= ntiles).transpose().drop(['count'], axis=1)
_tr.insert(0, 'source', 'Train');
_tr = _tr.reset_index().set_index(['index', 'source']);

_test = \
xtest.drop([target], axis=1, errors= 'ignore').\
describe(percentiles= ntiles).transpose().drop(['count'], axis=1)
_test.insert(0, 'source', 'Test');
_test = _test.reset_index().set_index(['index', 'source']);

_alt = \
alt_xytrain.drop([target], axis=1, errors= 'ignore').\
describe(percentiles= ntiles).transpose().drop(['count'], axis=1)
_alt.insert(0, 'source', 'Sklearn');
_alt = _alt.reset_index().set_index(['index', 'source']);

display(pd.concat([_tr, _test, _alt], axis=0).sort_index(ascending = [True, False]).\
style.format(formatter = '{:,.2f}'));

del ntiles, _tr, _test, _alt;


# In[10]:


get_ipython().run_cell_magic('time', '', '# Performing data augmentation:-\nPrintColor(f"\\nPerforming data augmentation between train and sklearn data\\n");\n\nPrintColor(f"Pre augmentation train data size = {xytrain.shape}", color = \'red\');\nxytrain = pd.concat((xytrain, alt_xytrain), axis = 0, ignore_index = True).drop_duplicates();\nPrintColor(f"Post augmentation train data size = {xytrain.shape}\\n", color = \'red\');\n\nPrintColor(f"\\nPost augmentation train data\\n");\ndisplay(xytrain.head(5));\ndisplay(xytrain.tail(5));\n')


# ## <p style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#032352; font-size:100%; text-align: left; padding:2.5px; border-bottom: 1px solid #090A0A; background:#EBF6F9" > B. Visualization- individual distributions
# </p>

# In[11]:


get_ipython().run_cell_magic('time', '', '\ndef MakeFtreKDEPlot():\n    """\n    This function develops feature KDE plots for the train-test features\n    """;\n    \n    fig, ax = plt.subplots(len(features), 2, figsize = (20, 50));\n\n    for i, Ftre in tqdm(enumerate(features)):\n        a = ax[i, 0];\n        xytrain[Ftre].plot.kde(ax = a, color = "tab:blue");\n        a.grid(**grid_specs);\n        a.set_title(f"\\n{Ftre}- Train\\n", **title_specs);\n        a.set(xlabel = \'\', ylabel = \'\');\n        \n        b = ax[i, 1];\n        xtest[Ftre].plot.kde(ax = b, color = "tab:blue");\n        b.grid(**grid_specs);\n        b.set_title(f"\\n{Ftre}- Test\\n", **title_specs);\n        b.set(xlabel = \'\', ylabel = \'\');  \n        \n        del a,b;\n    \n    plt.tight_layout();\n    plt.show();\n    print();\n')


# In[12]:


get_ipython().run_cell_magic('time', '', 'PrintColor(f"\\nFeature KDE plots\\n");\nMakeFtreKDEPlot();\nprint();\ncollect();\n')


# ## <p style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#032352; font-size:100%; text-align: left; padding:2.5px; border-bottom: 1px solid #090A0A; background:#EBF6F9" > C. Normality tests for features
# </p>

# In[13]:


# Creating output dataframe:-
normaltest_results = pd.DataFrame(data= None, index= range(len(features[0:-2])*2), 
                                  columns = ['Source', 'Feature', 'NormalTest', 'Shapiro', 'Anderson'], dtype = np.float32);

ad_cutoff = anderson(xytrain['MedInc'])[1][2];

for i, col in enumerate(features[0:-2]):
    normaltest_results.loc[i] = \
    ['Train', col, normaltest(xytrain[col])[1],shapiro(xytrain[col])[1], anderson(xytrain[col])[0]]; 
    
    normaltest_results.loc[len(features[0:-2]) + i] = \
    ['Test', col, normaltest(xtest[col])[1],shapiro(xtest[col])[1], anderson(xtest[col])[0]];                                  
    
normaltest_results['Is_Normal'] = \
np.where((normaltest_results['NormalTest'] > 0.05) | (normaltest_results['Shapiro'] > 0.05) | (normaltest_results['Anderson'] <= ad_cutoff), 
         "Y", "N");

PrintColor(f"\nNormality test results for train-test data\n");
display(normaltest_results.style.format(precision = 2).\
        applymap(lambda x: 'background-color: #ffebe6;color: red; font-weight:bold' if x == "N" 
                 else 'background-color: #f0f0f5; color: #000000'));


# ## <p style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#032352; font-size:100%; text-align: left; padding:2.5px; border-bottom: 1px solid #090A0A; background:#EBF6F9" > D. Visualization- feature target analysis
# </p>

# In[14]:


get_ipython().run_cell_magic('time', '', 'fig, ax = plt.subplots(4,2, figsize = (20, 20), sharey= True);\n\nPrintColor(f"\\nFeature-target interaction plots\\n");\n\nfor i, col in tqdm(enumerate(features)):\n    a = ax[i//2, i%2];\n    sns.scatterplot(x = xytrain[col], y = xytrain[target], markers=True, ax = a, color = \'#298EDF\');\n    a.grid(**grid_specs);\n    a.set_title(f"\\n{col}\\n", **title_specs);\n    a.set(xlabel = \'\', ylabel = \'\');\n    del a;\n\nplt.tight_layout();\nplt.show();\nprint(\'\\n\');\n')


# ## <p style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#032352; font-size:100%; text-align: left; padding:2.5px; border-bottom: 1px solid #090A0A; background:#EBF6F9" > E. Outlier detection
# </p>

# In[15]:


get_ipython().run_cell_magic('time', '', '\n# Entailing outlier related values from the training data:-\n_q1 = np.percentile(xytrain[features], q = 25, axis=0);\n_q3 = np.percentile(xytrain[features], q = 75, axis=0);\n_iqr = _q3 - _q1;\n_otl_ub = _q3 + 1.5 * _iqr;\n_otl_lb = _q1 - 1.5 * _iqr;\n\n# Plotting the heatmap for outlier location in the train and test set:-\nfig, ax = plt.subplots(2,1, figsize = (14, 40));\nsns.heatmap(xytrain[(xytrain[features] >= _otl_ub) | (xytrain[features] <= _otl_lb)].drop([\'Latitude\', \'Longitude\', target], axis=1), \n            cbar = None, cmap = \'rainbow\', ax= ax[0]);\nax[0].set_title(f"\\nOutlier location in the train data\\n", **title_specs);\n\nsns.heatmap(xtest[(xtest[features] >= _otl_ub) | (xtest[features] <= _otl_lb)].drop([\'Latitude\', \'Longitude\'], axis=1), \n            cbar = None, cmap = \'rainbow\', ax= ax[1]);\nax[1].set_title(f"\\nOutlier location in the test data based on training data bounds\\n", **title_specs);\nplt.show();\n\n# Counting the outliers in the training and test sets:-\n_ = np.sign(xytrain[(xytrain[features] >= _otl_ub) | (xytrain[features] <= _otl_lb)].fillna(0.0));\nPrintColor(f"\\nNumber of outliers per training data column features\\n");\ndisplay({col: np.int16(_[col].sum()) for col in features if col not in ([\'Latitude\', \'Longitude\'])});\n\n_ = np.sign(xtest[(xtest[features] >= _otl_ub) | (xtest[features] <= _otl_lb)].fillna(0.0));\nPrintColor(f"\\nNumber of outliers per test data column features based on training data bounds\\n");\ndisplay({col: np.int16(_[col].sum()) for col in features if col not in ([\'Latitude\', \'Longitude\'])});\n\n\ndel _, _q1, _q3, _iqr, _otl_lb, _otl_ub;\ncollect();\n\nPrintColor(f"\\nWe will ignore the latitude and longitude data columns as they do not align to this analysis\\n",\n          color = \'red\');\n')


# ## <p style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#032352; font-size:100%; text-align: left; padding:2.5px; border-bottom: 1px solid #090A0A; background:#EBF6F9" > F. Geospatial feature analysis
# </p>
# 
# **We shall analyse the latitude and longitude features in this section and try and derive some insights regarding the location of the house**

# In[16]:


fig, ax = plt.subplots(1,1,figsize = (12,8));
sns.scatterplot(data = xytrain, x = 'Longitude', y= 'Latitude', hue = np.log1p(xytrain[target]),
           palette = 'viridis', markers = True, sizes = 0.75, ax = ax);
ax.grid(**grid_specs);
ax.set_title(f"\nMedian house prices by location defined by geospatial coordinates- training data\n", **title_specs);
ax.set(xlabel = '\nLongitude\n', ylabel = "\nLatitude\n");
plt.tight_layout();
plt.show();
collect();


# # <p style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#032352; font-size:100%; text-align: left; padding:2.5px; border-bottom: 1px solid #090A0A; background:#EBF6F9" > G. Target analysis
# </p>
# 
# **We will try and derive some inferences from the target in this small section**

# In[17]:


fig, ax = plt.subplots(1,2, figsize= (16,6), sharey= True);
xytrain[target].plot.hist(bins = 20, ax= ax[0], color = 'tab:blue');
ax[0].grid(**grid_specs);
ax[0].set_title(f"\nTarget column\n", **title_specs);

np.log1p(xytrain[target]).plot.hist(bins = 20, ax= ax[1], color = 'tab:blue');
ax[1].grid(**grid_specs);
ax[1].set_title(f"\nTarget column- log transformed\n", **title_specs);

plt.tight_layout();
plt.show();


# In[18]:


# Analysing potential target capping:-
PrintColor(f"\nStudying the target distribution\n");
display(xytrain[[target]].describe().style.format(formatter = '{:,.2f}'));

# Making the KDE plot for the target:-
PrintColor(f"\nTarget KDE plot\n");
fig, ax = plt.subplots(1,1, figsize = (10,6));
xytrain[target].plot.kde(ax = ax);
ax.grid(**grid_specs);
ax.set_title(f"Target KDE plot\n", **title_specs);
plt.show();
del fig, ax;

PrintColor(f"\nStudying the target capping at upper extremity\n");
display(xytrain.loc[xytrain[target] >= 500000].groupby(xytrain[target]).size());

PrintColor(f"\nStudying the target capping at lower extremity\n");
display(xytrain.loc[xytrain[target] <= 160000].groupby(xytrain[target]).size());


# # <p style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#032352; font-size:100%; text-align: left; padding:2.5px; border-bottom: 1px solid #090A0A; background:#EBF6F9" > Key notes from preliminary EDA
# </p>
# 
# <div style="color: #050505;
#            display:fill;
#            border-radius:12px;
#            background-color: #F8F9FA;
#            font-size:110%;
#            font-family: Calibri;
#            letter-spacing: 0.5px;
#            border: 1px solid #000205;
#            ">
# 
# * The features and target are all non-normal. Log transformation is necessary to normalize the data
# * All features are numerical and the table has no nulls
# * Target is capped on the upper end at 500,010 value, 2792 instances with the capping are unearthed, we may have to calibrate the model to factor this unusual capping
# * Lower end flooring is limited and can be ignored for now
# * Features posit outliers, log transformation may be needed along with robust scalar to treat them for now
# * Location based data is interesting, we may have to analyse the distance to the nearest city and then perhaps develop some pseudo features from these
# * We will calculate the distance from Los Angeles, San Diego, San Jose, San Francisco for the houses
# * From the scatter plots, we are clear that the house prices are influened by median income, rooms and bedrooms for sure. This is expected from domain elements also. 
# </p>
# </div>

# # <p style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#032352; font-size:100%; text-align: left; padding:2.5px; border-bottom: 1px solid #090A0A; background:#EBF6F9" > Data pipeline development
# </p>
# 
# **We will build a pipeline with auxiliary features, transform the existing features and develop the distances as specified above to complete this step**
# 
# Latitude and Longitude rotational features and associated segment is taken from the link- https://www.kaggle.com/code/alexandershumilin/playground-series-s3-e1-lightautoml

# In[19]:


class DataTransformer(BaseEstimator, TransformerMixin):
    """
    This class creates the below-
    1. log transformation of the existing features
    2. calculates secondary features using existing features
    """;
    
    def __init__(self, features):
        self.features = features[0:-2];
    
    def fit(self, X, y= None, **fit_params):
        return self;
    
    def transform(self, X, y= None, **transform_params):
        df = X.copy();  
        df['Location'] = df[['Latitude', 'Longitude']].apply(tuple, axis=1);
       
        df = pd.concat((df, np.log1p(df[self.features]).add_prefix('LN_')), axis=1);             
        df['OccupancyPerRoom'] = np.log1p(df['AveOccup'] / df['AveRooms']);
        df['OccupancyPerBedRoom'] = np.log1p(df['AveOccup'] / df['AveBedrms']);
        df['AvgRoomtoBedRoom'] = np.log1p(df['AveRooms'] / df['AveBedrms']);
        df['OtherRooms'] = df['AveRooms'] - df['AveBedrms'];
        df['MedIncPop'] = np.log1p(df['MedInc'] / df['Population']);
        df['RoomMedInc'] = np.log1p(df['AveRooms'] / df['MedInc']);
        df['BedroomMedInc'] = np.log1p(df['AveBedrms'] / df['MedInc']);
        df['AvgOccupMedInc'] = np.log1p(df['AveOccup'] / df['MedInc']);
        
        df['Rot_15_x'] = (np.cos(np.radians(15)) * df['Longitude']) + (np.sin(np.radians(15)) * df['Latitude']);  
        df['Rot_15_y'] = (np.cos(np.radians(15)) * df['Latitude']) + (np.sin(np.radians(15)) * df['Longitude']);   
        df['Rot_30_x'] = (np.cos(np.radians(30)) * df['Longitude']) + (np.sin(np.radians(30)) * df['Latitude']);  
        df['Rot_30_y'] = (np.cos(np.radians(30)) * df['Latitude']) + (np.sin(np.radians(30)) * df['Longitude']); 
        df['Rot_45_x'] = (np.cos(np.radians(45)) * df['Longitude']) + (np.sin(np.radians(45)) * df['Latitude']);
        df['Rot_45_y'] = (np.cos(np.radians(45)) * df['Latitude']) + (np.sin(np.radians(45)) * df['Longitude']);      
               
        self.op_col = df.columns;
        return df;
    
    def get_feature_names_in(self, X, y= None):
        return X.columns;
    
    def get_feature_names_out(self, X, y= None):
        return self.op_col;    


# In[20]:


def CalcDist(location: tuple, dist_measure:str):
    """
    This function takes a tuple of latitude and longitude and compare to a dictonary of locations where
    key = location name and value = (lat, long)
    """;
    
    global loc_dict;
    
    distance = [];
    for city in loc_dict.keys():
        if dist_measure == 'geodesic':
            distance.append(geodesic(location, loc_dict[city]).kilometers);
        elif dist_measure == 'haversine':
            distance.append(haversine(location, loc_dict[city], 'km'));

    return distance;

class DistCalc(BaseEstimator, TransformerMixin):
    """
    This class calculates the distance from the house to the 4 key cities- 
    Los Angeles, San Diego, San Jose, San Francisco and the minimum distance to assess remoteness of property
    """;
    
    def __init__(self, dist_measure): 
        self.dist_measure = dist_measure;
 
    def fit(self, X, y= None, **fit_params):
        return self;   
      
    def transform(self, X, y= None, **transform_params):
        df = X.copy();    
        _ = \
        df.apply(lambda x: CalcDist(x.Location, self.dist_measure), axis=1).apply(pd.Series);
        _.columns = ['Dist_' + col for col in list(loc_dict.keys())];  
        # Calculating average distance:-
        _['Mean_Dist'] = np.mean(_, axis=1);
        df = pd.concat([df, _], axis=1);  
        del _;
        
        self.op_col = df.columns;
        return df;
    
    def get_feature_names_in(self, X, y= None):
        return X.columns;
    
    def get_feature_names_out(self, X, y= None):
        return self.op_col;       


# In[21]:


get_ipython().run_cell_magic('time', '', '\n# Implementing the pipeline on the train-test data:-\nif target_log_xform_req == \'Y\': \n    ytrain = np.log1p(xytrain[target]/ 100_000);\nelif target_log_xform_req == \'N\':   \n    ytrain = xytrain[target]/ 100_000;\n    \nxtrain = xytrain.drop(target, axis=1);\n\npipe = Pipeline(steps = [(\'DataXform\',DataTransformer(features = features)),\n                         (\'CalcDist\', DistCalc(dist_measure = dist_measure))],\n               verbose= True);\n\nPrintColor(f"\\nImplementing the data pipeline:-\\n", color = \'red\');\nXtrain = pipe.fit_transform(xtrain, ytrain);\nPrintColor(f"\\nTrain data after pipeline (Xtrain) has a shape of {Xtrain.shape}\\n");\nXtest = pipe.transform(xtest);\nPrintColor(f"\\nTest data after pipeline (Xtest) has a shape of {Xtest.shape}\\n");\n\nposteda_features = pipe.get_feature_names_out();\nPrintColor(f"\\nFeatures post EDA pipeline\\n");\ndisplay(posteda_features);\n\ncollect();\nprint(\'\\n\');\n')


# # <p style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#032352; font-size:100%; text-align: left; padding:2.5px; border-bottom: 1px solid #090A0A; background:#EBF6F9" > Feature analysis after EDA
# </p>
# 
# **We will develop feature interactions with correlation and mutual information using the new features and auxiliary features. We will also develop basic plots to elicit the efficacy of new features.**

# In[22]:


get_ipython().run_cell_magic('time', '', 'fig, ax = plt.subplots(np.int8(np.ceil(len(posteda_features)/2)),2, figsize = (20, 60), sharey= True);\n\nPrintColor(f"\\nFeature-target interaction plots post pipeline\\n");\n\nfor i, col in tqdm(enumerate(posteda_features)):\n    if col != \'Location\':\n        a = ax[i//2, i%2];\n        sns.scatterplot(x = Xtrain[col], y = ytrain, markers=True, ax = a, color = \'#298EDF\');\n        a.grid(**grid_specs);\n        a.set_title(f"\\n{col}\\n", **title_specs);\n        a.set(xlabel = \'\', ylabel = \'\');\n        del a;\n\nplt.tight_layout();\nplt.show();\nprint(\'\\n\');\n')


# In[23]:


# Creating new features after EDA:-
new_features = [col for col in posteda_features if re.findall(r"\Dist_",col) == []];

PrintColor(f"\nNew features after EDA\n");
display(new_features);


# In[24]:


# Correlation plot:-
fig, ax = plt.subplots(1,1, figsize= (24,20));
_corr = Xtrain[new_features].corr();
sns.heatmap(data= _corr, cbar = None, cmap= 'Blues', annot = True, fmt='.2%',
            annot_kws= {'fontweight': 'bold', 'fontsize': 12},
            linewidths= 1.25, linecolor= 'white', mask= np.triu(np.ones_like(_corr)),
            ax = ax);
ax.set_title(f"\nCorrelation plot after EDA with extended features\n", **title_specs);

plt.tight_layout();
plt.show();


# In[25]:


# Mutual Information:-
_num_features = Xtrain[[col for col in new_features if col != 'Location']].columns;
_minfo = mutual_info_regression(Xtrain[_num_features],ytrain, random_state = 7);

fig, ax = plt.subplots(1,1, figsize= (14,6));
_minfo = pd.DataFrame(data = _minfo, index = _num_features,
                      columns = ['MutualInfo']);
_minfo.plot.bar(ax = ax, color = 'tab:blue');
ax.grid(**grid_specs);
ax.set_title(f"\nMutual Information Regression after EDA\n", **title_specs);

plt.tight_layout();
plt.yticks(np.arange(0, 0.55, 0.05));
plt.show();
collect();


# In[26]:


get_ipython().run_cell_magic('time', '', '# Feature selection based on mutual info and correlation:-\nftre_sel = \\\npd.concat([_minfo,pd.concat([Xtrain[_num_features], ytrain], axis=1).corr()\\\n           [[target]].drop([target], axis=0)],\n          axis=1);\nftre_sel.columns = [\'MutualInfo\', \'Correlation\'];\nftre_sel[\'CombinedMeasure\'] = (abs(ftre_sel[\'Correlation\']) + ftre_sel[\'MutualInfo\'])/2;\n\nPrintColor(f"\\nFeature selection with correlation and mutual information\\n");\ndisplay(ftre_sel.sort_values(\'CombinedMeasure\', ascending = False));\n\ncollect();\n')


# In[27]:


# Shortlisting features based on correlation and mutual information:-
sel_features = \
['LN_MedInc','AvgOccupMedInc','RoomMedInc', 'BedroomMedInc', 'Latitude', 'OccupancyPerRoom',
 'AvgRoomtoBedRoom', 'Longitude', 'Mean_Dist', 'LN_AveRooms', 'OtherRooms', 'LN_AveOccup',
 'MedIncPop', 'OccupancyPerBedRoom', 'HouseAge', 'LN_AveBedrms', 'LN_Population',
 'Rot_15_x', 'Rot_15_y', 'Rot_30_x', 'Rot_30_y', 'Rot_45_x', 'Rot_45_y'
];

PrintColor(f"\nNew features after EDA and dropping features\n");
display(sel_features);


# # <p style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#032352; font-size:100%; text-align: left; padding:2.5px; border-bottom: 1px solid #090A0A; background:#EBF6F9" > Model development
# </p>

# **We will do the below in the section-**
# * We will use tree ensemble models herewith and elicit CV based predictions. 
# * For a baseline model, we will use the default parameters
# 
# LGBM model parameters are borrowed from the link- https://www.kaggle.com/code/soupmonster/simple-lightgbm-baseline

# In[28]:


# Model instances:-
LGBM = LGBMRegressor(learning_rate = 0.02, lambda_l1 = 1.945, num_leaves = 87,
                     feature_fraction = 0.79, bagging_fraction = 0.93, bagging_freq = 4,
                     min_data_in_leaf = 103,max_depth = 17,num_iterations = 100000,
                     metric ='rmse');
XGB = XGBRegressor(n_estimators = 100000, random_state = 42, 
                   objective = 'reg:squarederror', learning_rate = 0.02);
CB = CatBoostRegressor(n_estimators = 100000, random_state = 42, loss_function = 'RMSE');

# Creating a model dictionary to be used subsequently:-
model_master = {'LGBM': LGBM, 'XgBoost': XGB, 'CatBoost': CB};

# CV instance:-
cv = KFold(n_splits = n_folds, shuffle= True, random_state= 42);

# CV output structure:-
CVscores = pd.DataFrame(data= None, index = range(n_folds), columns = model_master.keys(), 
                        dtype = np.float32);

# Predictions output structure:-
test_preds = pd.DataFrame(data = np.zeros((len(Xtest), len(model_master.keys()))),
                          index= Xtest.index, columns = model_master.keys(), 
                          dtype = np.float32);

PrintColor(f"\nDisplaying model output structure\n");
display(CVscores);
print();
display(test_preds.head(5));


# In[29]:


# Model implementation:-
for i, (train_idx, dev_idx) in tqdm(enumerate(cv.split(Xtrain, ytrain))):
    PrintColor(f"\nCurrent fold = {i}");
    
    xtr, ytr = Xtrain[sel_features].loc[Xtrain.index.isin(train_idx)], ytrain.loc[ytrain.index.isin(train_idx)];
    xdev, ydev = Xtrain[sel_features].loc[Xtrain.index.isin(dev_idx)], ytrain.loc[ytrain.index.isin(dev_idx)];
    
    for label, model in model_master.items():
        PrintColor(f"\nCurrent method = {label}\n", color = 'red');
        model.fit(xtr, ytr, eval_set = [(xdev, ydev)], early_stopping_rounds= 750, 
                  verbose= 1500 if label == 'LGBM' else 3500);
        
        if target_log_xform_req == 'Y': 
            pred = np.expm1(model.predict(Xtest[sel_features]));
        elif target_log_xform_req == 'N':
            pred = model.predict(Xtest[sel_features]);

        test_preds[label] = test_preds[label] + (pred/ n_folds);
        dev_pred = model.predict(xdev[sel_features]);
        CVscores.loc[i, label] = mean_squared_error(ydev, dev_pred, squared = False); 
    
    del dev_pred, pred, xtr, ytr, xdev, ydev;  
    collect();
    print(f"-------- end of fold {i} --------");


# In[30]:


PrintColor(f"\nCV Score Summary:-\n");
display(CVscores.style.format('{:.2%}'));

PrintColor(f"\nMean CV Score Summary:-\n");
display(CVscores.mean(axis=0));


# In[31]:


# Preparing the submission file:-
sub_fl[target] = (test_preds['CatBoost']* 0.40 + test_preds['LGBM'] * 0.40 + test_preds['XgBoost'] * 0.20).values;

# Analyzing the post-model predictions with the training data:-
ntiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99];
_descr = pd.concat([sub_fl[target].describe(percentiles= ntiles),
                    ytrain.describe(percentiles= ntiles)], axis=1);
_descr.columns = ['Test', 'Train'];

PrintColor(f"\nData descriptions- test and train after model predictions\n");
display(_descr);

collect();
del _descr, ntiles;


# In[32]:


# Preparing the submission file
sub_fl.to_csv('submission.csv', index= False);
display(sub_fl.head(10));

# Saving the test-set predictions and CV scores:-
test_preds.to_csv("Test_Preds.csv");
CVscores.to_csv("CVScores.csv");

collect();
PrintColor(f"-------------------------- End of code --------------------------");

