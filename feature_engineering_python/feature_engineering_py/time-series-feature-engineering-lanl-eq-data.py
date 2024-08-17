#!/usr/bin/env python
# coding: utf-8

# Often we get the time series data but due to lack of other features, we cannot convert the time series problem into a regression (or classification) problem. Recently, I came across ["tsfresh" package](https://tsfresh.readthedocs.io/en/latest/text/introduction.html) which helps in extracting features from time series data such as: mean, max, min, median, 0.4 quantile, 0.7 quantile, linear trend attribute intercept etc. Once we extract these features, the problem converts to a machine learning problem instead of a time series one, and we can apply ML models instead of applying time series models.
# 
# It is possible to calculate these statistical figures for given data by writing codes manually as well, but tsfresh automates the process. Also, tsfresh can extract more meaningful (features which have more impact on time series data) parameters as compared to which we can think of by writing manual code.  

# In[1]:


import os
os.listdir("../input/LANL-Earthquake-Prediction")


# In[2]:


import pandas as pd
import numpy as np
train_data = pd.read_csv(os.path.join("../input/LANL-Earthquake-Prediction",'train.csv'), dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})


# I am using the earthquake data here. It is a time series data having two fields.
# * **acoustic_data** = the time when signal was generated at the epicenter 
# * **time_to_failure** = the time taken in seconds when quake is felt on the surface of earth

# In[3]:


train_data.head(10)


# Apparently, all 10 rows are showing same "time_to_failure" above. Is it really so? Let's try to see it with increased precision.

# In[4]:


pd.options.display.precision = 12
train_data.head(10)


# Now we can spot the difference!

# In[5]:


train_data.shape


# In[6]:


train_data.info()


# The train data set is huge here. Let us work with a fraction of data set. Our main purpose is to extract features here. We are not going to solve the entire ML modelling in this notebook.

# In[7]:


import gc
gc.collect()


# In[8]:


# 1st sample = 0.0025% of total data
train_acoustic_data_sample_1 = train_data['acoustic_data'].values[::40000]
train_time_to_failure_sample_1 = train_data['time_to_failure'].values[::40000]


# In[9]:


import matplotlib.pyplot as plt
def plot_data(train_ad_sample_df, train_ttf_sample_df):
    fig, ax = plt.subplots(2,1, figsize=(13, 10))
    ax[0].set_title("Acoustic Data: {:.4f} % sampled data".format(float(train_ad_sample_df.shape[0]/train_data.shape[0])*100))
    ax[0].plot(train_ad_sample_df, color='red')
    ax[0].set_ylabel('acoustic data', color='red')
    ax[0].set_xlabel('index', color='red')
    ax[1].set_title("Time to Failure: {:.4f} % sampled data".format(float(train_ad_sample_df.shape[0]/train_data.shape[0])*100))
    ax[1].plot(train_ttf_sample_df, color='green')
    ax[1].set_ylabel('time to failure', color='green')
    ax[1].set_xlabel('index', color='green')


# In[10]:


plot_data(train_acoustic_data_sample_1, train_time_to_failure_sample_1)
del train_acoustic_data_sample_1
del train_time_to_failure_sample_1


# **Observation for 1st Sample Data Set :** We have taken 0.0025% sample data here, where each data point is situated at 40000 gap. The 'acoustic data' is not varying much whereas 'time to failure' is varying aggressively when plotted against the index.

# In[11]:


# 2nd sample = 0.25% of total data
train_acoustic_data_sample_2 = train_data['acoustic_data'].values[::400]
train_time_to_failure_sample_2 = train_data['time_to_failure'].values[::400]


# In[12]:


plot_data(train_acoustic_data_sample_2, train_time_to_failure_sample_2)
del train_acoustic_data_sample_2
del train_time_to_failure_sample_2


# **Observation for 2nd Sample Data Set :** We have taken 0.25% sample data here, where each data point is situated at 400 gap. The 'acoustic data' is now varying comparatively high w.r.t. index. 'Time to failure' is varying on a similar aggressive level as before when plotted against the index.

# In[13]:


# 3rd sample = 5% of total data
train_acoustic_data_sample_3 = train_data['acoustic_data'].values[::20]
train_time_to_failure_sample_3 = train_data['time_to_failure'].values[::20]


# In[14]:


plot_data(train_acoustic_data_sample_3, train_time_to_failure_sample_3)
del train_acoustic_data_sample_3
del train_time_to_failure_sample_3


# **Observation for 3rd Sample Data Set :** We have taken 5% sample data here, where each data point is situated at 20 gap. Now, both 'acoustic data' and 'time to failure' are varying aggressively w.r.t. index.
# 
# **Overall Observation:**
# As we are increasing our sample size, the variation range of both 'acoustic data' and 'time to failure' are increasing.

# In[15]:


gc.collect()


# In[16]:


import tsfresh
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.feature_extraction.settings import from_columns


# In[17]:


# Sample training data
train_data = train_data[:6000000]
train_data = train_data.reset_index()


# We will segregate 6000000 records into 600 segments each having 10000 rows. Each segment is then allotted an id. So our data will be having ids from 1 till 600. The purpose of creating "id" and "index" columns in the data set is: "tsfresh" requires the data to follow a [particular format](https://tsfresh.readthedocs.io/en/latest/text/data_formats.html).

# In[18]:


rows = 10000
idlist = []
for n in range(1,601):  #600 segments
    idlist = idlist + [n for i in range(rows)]


# In[19]:


gc.collect()


# In[20]:


train_data['id'] = idlist
train_data.head()


# In[21]:


train_data.tail()


# In[22]:


gc.collect()


# Well, we are now keeping only 'acoustic data' as our independent variable 'x' and separating out the 'time to failure' as dependent variable 'y'.

# In[23]:


y = train_data['time_to_failure']
x = train_data.drop(columns = 'time_to_failure')


# In[24]:


y


# In[25]:


del train_data


# From 'y', we will now separate out 600 data points (equally distant at 10000 gap) for using them as a "target" later for feature extraction purpose.

# In[26]:


target = y[9999::10000]


# In[27]:


target


# In[28]:


target.index = range(1,601)


# In[29]:


target


# In[30]:


gc.collect()


# In[31]:


x = x.rename(columns = {'index':'time'})


# Now, we will extract full features using 'x' as independent variable data set.

# In[32]:


extracted_features = extract_features(x, column_id="id", column_sort="time", default_fc_parameters=EfficientFCParameters())


# Let us have a look at what is the entire list of features we have extracted.

# In[33]:


extracted_features.head(10)


# Well, we have extracted 773 features i.e. too many to deal with. We need to have only the features having the highest impact on "target" (using **Regression Model** here to judge the impact). For that purpose, we will use a threshold called **"FDR level"** which is the theoretical expected percentage of irrelevant features among all created features. 

# In[34]:


gc.collect()


# Before performing smaller feature set generation, we need to impute the big feature set first.

# In[35]:


impute(extracted_features)


# In[36]:


#The fdr level is the threshold of feature importance. Ii is set as very low to get smaller number of features.
small_feat_set = select_features(extracted_features, target, fdr_level = 0.005, ml_task = 'regression')


# In[37]:


target.shape


# In[38]:


target = target.values.reshape(600,1)


# In[39]:


small_feat_set.shape


# In[40]:


small_feat_set.info()


# Well, now we can see that from our 773 features, really important features are only 27 in number!

# In[41]:


gc.collect()


# Let us check if there is any pair of features having high multicollinearity.

# In[42]:


# Correlation Heatmap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
corr = small_feat_set.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(20, 20))
cmap = sns.diverging_palette(1, 200, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=None, center=0,square=True, annot=False, linewidths=.5, cbar_kws={"shrink": 0.8})


# We need to drop the highly correlated features to avoid perfect multicollinearity. For that, we need to spot the highly correlated feature pairs.

# In[43]:


get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.prototype._should_scroll = function() {\n    return False;\n}\n')


# In[44]:


# Spot the categorical feature pairs with high correlation
threshold = 0.9999
high_corrs = (corr[abs(corr) > threshold][corr != 1.0]).unstack().dropna().to_dict()
unique_high_corrs = pd.DataFrame(list(set([(tuple(sorted(key)), high_corrs[key]) for key in high_corrs])), columns=['feature_pair', 'correlation_coefficient'])
unique_high_corrs = unique_high_corrs.loc[abs(unique_high_corrs['correlation_coefficient']).argsort()[::-1]]
pd.options.display.max_colwidth = 200
unique_high_corrs


# In[45]:


gc.collect()


# In[46]:


small_feat_set = small_feat_set.drop(['acoustic_data__count_above_mean', 'acoustic_data__mean', 
                                      'acoustic_data__linear_trend__attr_"intercept"', 
                                      'acoustic_data__agg_linear_trend__attr_"intercept"__chunk_len_10__f_agg_"mean"',
'acoustic_data__agg_linear_trend__attr_"intercept"__chunk_len_5__f_agg_"mean"'],axis=1)


# In[47]:


small_feat_set.shape


# In[48]:


gc.collect()


# Finally, we have only 23 important features!

# In[49]:


import warnings
warnings.filterwarnings("ignore")


# In[50]:


gc.collect()


# Alright. We will now determine the feature importance of our extracted features using various ML models like Random Forest Regressor and Extra Trees Regressor respectively.

# In[51]:


import eli5
from eli5.sklearn import PermutationImportance
from sklearn.ensemble import RandomForestRegressor as rf

perm = PermutationImportance(rf(n_estimators=100, random_state=12345).fit(small_feat_set,target),random_state=56789).fit(small_feat_set,target)
eli5.show_weights(perm, feature_names = small_feat_set.columns.tolist())


# In[52]:


from sklearn.ensemble import ExtraTreesRegressor as et

perm = PermutationImportance(et(max_features='auto').fit(small_feat_set,target),random_state=12345).fit(small_feat_set,target)
eli5.show_weights(perm, feature_names = small_feat_set.columns.tolist())


# From both the models, we have found that "acoustic_data__c3__lag_1", "acoustic_data__c3__lag_2", "acoustic_data__c3__lag_3" have high influence on the target variable. 

# 

# In[ ]:




