#!/usr/bin/env python
# coding: utf-8

# # TPS-Jan22 | CatBoost using PyCaret
# 
# # 📝 Agenda
# >1. [📚 Loading libraries and files](#Loading)
# >2. [🔍 Exploratory Data Analysis with PyCaret](#EDA)
# >3. [⚙️ Feature Engineering](#FeatureEngineering)
# >4. [🏋️ Model Training & Inference](#TrainingInference)

# # What is PyCaret?

# ![PyCaret logo](https://raw.githubusercontent.com/pycaret/pycaret/master/docs/images/logo.png)

# [PyCaret](https://pycaret.org/) is an open source Python machine learning library inspired by the caret R package.
# 
# The goal of the caret package is to automate the major steps for evaluating and comparing machine learning algorithms for classification and regression. The main benefit of the library is that a lot can be achieved with very few lines of code and little manual configuration. The PyCaret library brings these capabilities to Python.
# 
# 📌 According to the [PyCaret official website](https://pycaret.org/guide/):
# > PyCaret is an open-source, **low-code** machine learning library in Python that aims to reduce the cycle time from hypothesis to insights. It is well suited for **seasoned data scientist**s who want to increase the productivity of their ML experiments by using PyCaret in their workflows or for **citizen data scientists** and those **new to data science** with little or no background in coding. PyCaret allows you to go from preparing your data to deploying your model within seconds using your choice of notebook environment.
# 
# The PyCaret library automates many steps of a machine learning project, such as:
# * Defining the data transforms to perform `setup()`
# * Evaluating and comparing standard models `compare_models()`
# * Tuning model hyperparameters `tune_model()`
# 
# As well as many more features not limited to creating ensembles, saving models, and deploying models.

# ___
# # <a name="Loading">📚 Loading libraries and files</a>

# In[1]:


get_ipython().run_cell_magic('capture', '', '!pip install pycaret[full]\n\nimport os\nimport warnings\n\nimport numpy as np  # linear algebra\nimport pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n\nimport seaborn as sns\nimport matplotlib.pyplot as plt\n\nimport math\nfrom pathlib import Path\n\nimport dateutil.easter as easter\nfrom pycaret.regression import *\n\n# Mute warnings\nwarnings.filterwarnings("ignore")\n')


# In[2]:


get_ipython().system('tree ../input/')


# In[3]:


data_dir = Path('../input/tabular-playground-series-jan-2022')
holiday_dir = Path('../input/public-and-unofficial-holidays-nor-fin-swe-201519')
gdp_dir = Path('../input/gdp-20152019-finland-norway-and-sweden')

train = pd.read_csv(
    data_dir / 'train.csv',
    dtype={
        'country': 'category',
        'store': 'category',
        'product': 'category',
        'num_sold': 'float32',
    },
    parse_dates=['date'],
    infer_datetime_format=True,
    index_col='row_id'
)

test = pd.read_csv(
    data_dir / "test.csv",
    dtype={
        'country': 'category',
        'store': 'category',
        'product': 'category',
    },
    parse_dates=['date'],
    infer_datetime_format=True,
    index_col='row_id'
)

target_col = train.columns.difference(test.columns)[0]

holiday_data = pd.read_csv(holiday_dir / 'holidays.csv')

gdp = pd.read_csv(
    gdp_dir / 'GDP_data_2015_to_2019_Finland_Norway_Sweden.csv', index_col='year')


# ___
# # <a name="EDA">🔍 Exploratory Data Analysis with PyCaret</a>

# In[4]:


eda = setup(data=train, target=target_col, session_id=123 , profile=True, silent=True)


# ___
# # <a name="FeatureEngineering">⚙️ Feature Engineering</a>

# 📌 This part has been updated and largely inspired by these notebooks:
# > * [TPSJAN22-03 Linear Model](https://www.kaggle.com/ambrosm/tpsjan22-03-linear-model) & [TPSJAN22-06 LightGBM Quickstart](https://www.kaggle.com/ambrosm/tpsjan22-06-lightgbm-quickstart) by [AmbrosM](https://www.kaggle.com/ambrosm)<br />
# > * [TPS Jan 22 - EDA + modelling](https://www.kaggle.com/samuelcortinhas/tps-jan-22-eda-modelling) by [Samuel Cortinhas](https://www.kaggle.com/samuelcortinhas)
# > * [TPS Jan 2022 CatBoost with PyCaret](https://www.kaggle.com/bernhardklinger/tps-jan-2022-catboost-with-pycaret) by [Bernhard Klinger](https://www.kaggle.com/bernhardklinger)

# In[5]:


# Categorical features
categorical_cols = train.select_dtypes('category').columns.tolist()


# It has been shown that the GDP helps to improve the our mode. Thus, let's transform our target considering the **GDP deflator**, which is a measure of inflation.

# In[6]:


K_FOLDS = 3
GDP_EXPONENT = 1.2120618918594863 
# c.f https://www.kaggle.com/ambrosm/tpsjan22-03-linear-model

gdp.columns = gdp.columns.str[4:]
gdp = gdp.apply(lambda x: x**GDP_EXPONENT)
scaler = gdp.iloc[K_FOLDS+1] / gdp
gdp_map = scaler.stack().to_dict()

train[target_col] = pd.Series(
    list(zip(train.date.dt.year,train.country))
).map(gdp_map) * train[target_col]

train[target_col] = np.log1p(train.num_sold)


# We are dealing with time-series data, therefore it is relevant to consider the impact of holidays, which naturally play a large role in business activities.

# In[7]:


def holiday_features(holiday_df, df):
    
    fin_holiday = holiday_df.loc[holiday_df.country == 'Finland']
    swe_holiday = holiday_df.loc[holiday_df.country == 'Sweden']
    nor_holiday = holiday_df.loc[holiday_df.country == 'Norway']
    
    df['fin holiday'] = df.date.isin(fin_holiday.date).astype(int)
    df['swe holiday'] = df.date.isin(swe_holiday.date).astype(int)
    df['nor holiday'] = df.date.isin(nor_holiday.date).astype(int)
    
    df['holiday'] = np.zeros(df.shape[0]).astype(int)
    
    df.loc[df.country == 'Finland', 'holiday'] = df.loc[df.country == 'Finland', 'fin holiday']
    df.loc[df.country == 'Sweden', 'holiday'] = df.loc[df.country == 'Sweden', 'swe holiday']
    df.loc[df.country == 'Norway', 'holiday'] = df.loc[df.country == 'Norway', 'nor holiday']
    
    df.drop(['fin holiday', 'swe holiday', 'nor holiday'], axis=1, inplace=True)
    
    # Easter
    easter_date = df.date.apply(lambda date: pd.Timestamp(easter.easter(date.year)))
    df['days_from_easter'] = (df.date - easter_date).dt.days.clip(-5, 65)
    
    # Last Sunday of May (Mother's Day)
    sun_may_date = df.date.dt.year.map({
        2015: pd.Timestamp(('2015-5-31')),
        2016: pd.Timestamp(('2016-5-29')),
        2017: pd.Timestamp(('2017-5-28')),
        2018: pd.Timestamp(('2018-5-27')),
        2019: pd.Timestamp(('2019-5-26'))
    })
    #new_df['days_from_sun_may'] = (df.date - sun_may_date).dt.days.clip(-1, 9)
    
    # Last Wednesday of June
    wed_june_date = df.date.dt.year.map({
        2015: pd.Timestamp(('2015-06-24')),
        2016: pd.Timestamp(('2016-06-29')),
        2017: pd.Timestamp(('2017-06-28')),
        2018: pd.Timestamp(('2018-06-27')),
        2019: pd.Timestamp(('2019-06-26'))
    })
    df['days_from_wed_jun'] = (df.date - wed_june_date).dt.days.clip(-5, 5)
    
    # First Sunday of November (second Sunday is Father's Day)
    sun_nov_date = df.date.dt.year.map({
        2015: pd.Timestamp(('2015-11-1')),
        2016: pd.Timestamp(('2016-11-6')),
        2017: pd.Timestamp(('2017-11-5')),
        2018: pd.Timestamp(('2018-11-4')),
        2019: pd.Timestamp(('2019-11-3'))
    })
    df['days_from_sun_nov'] = (df.date - sun_nov_date).dt.days.clip(-1, 9)
    
    return df

train = holiday_features(holiday_data, train)
test  = holiday_features(holiday_data, test)


# Next, the cardinality of each categorical feature is quite low, and that we do not want to impose an ordinal order, **one-hot encoding** may be a good way to encode our categorical features.

# In[8]:


train = pd.get_dummies(train, columns=categorical_cols)
test  = pd.get_dummies(test, columns=categorical_cols)


# Since we have a <code>date</code>-typed feature here, and models are rarely able to use dates and times as they are, we would benefit from encoding it as categorical variables as this can often yield useful information about temporal patterns.
# 
# Furthermore, time-series data (such as product sales) often have distributions that differs from week days to week-ends for example, it is likely that using the day of the week as a new feature is a relevant option we have.

# In[9]:


def new_date_features(df):
    df['year'] = df.date.dt.year 
    df['quarter'] = df.date.dt.quarter
    df['month'] = df.date.dt.month  
    df['week'] = df.date.dt.week 
    df['day'] = df.date.dt.day  
    df['weekday'] = df.date.dt.weekday
#     df['day_of_week'] = df.date.dt.dayofweek  
    df['day_of_year'] = df.date.dt.dayofyear  
#     df['week_of_year'] = df.date.dt.weekofyear
    df['day_of_month'] = df.date.dt.days_in_month  
    df['is_weekend'] = np.where((df['weekday'] == 5) | (df['weekday'] == 6), 1, 0)
    df['is_friday'] = np.where((df['weekday'] == 4), 1, 0)
    
    df.drop('date', axis=1, inplace=True)
    
    return df
    
train = new_date_features(train)
test  = new_date_features(test)


# Finally, here are our datasets.

# In[10]:


display(train, test)


# ___
# # <a name="TrainingInference">🏋️ Model Training & Inference</a>

# Submissions are evaluated on SMAPE between forecasts and actual values.

# ![SMAPE formula](https://media.geeksforgeeks.org/wp-content/uploads/20211120224204/smapeformula.png)

# In[11]:


def smape(actual, predicted):
    numerator = np.abs(predicted - actual)
    denominator = (np.abs(actual) + np.abs(predicted)) / 2
    
    return np.mean(numerator / denominator)*100


# ### Initializing setup

# In[12]:


NB_MODELS = 3

models = []

for i in range (NB_MODELS):
    print ('Fit Model', i)
    reg = setup(
        data = train,
        target = target_col,
        data_split_shuffle = False, 
        create_clusters = False,
        fold_strategy = 'groupkfold',
        fold_groups = 'year',
        use_gpu = True,
        silent = True,
        fold = K_FOLDS,
        n_jobs = -1,
    )
    
    add_metric('SMAPE', 'SMAPE', smape, greater_is_better=False)
    
    models.append(create_model('catboost'))


# ### Interpret the model

# Analyzing feature importance:

# In[13]:


plot_model(models[0], 'feature')


# Interpretation of the model, based on the [SHapley Additive exPlanations (SHAP)](https://shap.readthedocs.io/en/latest/):
# 
# > **SHAP** is an approach that aims at **explaining the output of any Machine Learning model.** This tool has the particularity of connecting game theory with local explanations by unifying several old methods such as LIME, DeepLIFT and Shapley value (in a cooperative game, Shapley value gives (in a cooperative game, Shapley value gives a fair distribution of payoffs to the players).

# In[14]:


interpret_model(models[0])


# ### Blending the models

# 📌 According to the [PyCaret documentation](https://pycaret.org/blend-models/):
# > Blending models is a method of ensembling which uses consensus among estimators to generate final predictions. The idea behind blending is to combine different machine learning algorithms and use a majority vote or the average predicted probabilities in case of classification to predict the final outcome.

# In[15]:


blend = blend_models(models)


# ### Finalization & Inference

# In[16]:


final_blend = finalize_model(blend)


# Analyzing the prediction error:

# In[17]:


plot_model(final_blend, 'error')


# **Tip:** Since the SMAPE evaluation metric is asymmetric. In this case, underestimated values are much more penalized than overestimated values. Then, feel free to round your predictions **up** to the nearest value, or use any rounding technique that may be relevant.<br />
# <br />
# 📌 You will find more by having a glance to these awesome notebooks: 
# > * [SMAPE Weirdness](https://www.kaggle.com/cpmpml/smape-weirdness) by [CPMP](https://www.kaggle.com/cpmpml)
# > * [TPS Jan 2022: A simple average model (no ML)](https://www.kaggle.com/carlmcbrideellis/tps-jan-2022-a-simple-average-model-no-ml) by [Carl McBride Ellis](https://www.kaggle.com/carlmcbrideellis).
# > * [🌪 Ensembling and rounding techniques comparison](https://www.kaggle.com/fergusfindley/ensembling-and-rounding-techniques-comparison) by [Fergus Findley](https://www.kaggle.com/fergusfindley)

# In[18]:


# Fit-Based Weights Geo-Rounded
# from https://www.kaggle.com/fergusfindley/ensembling-and-rounding-techniques-comparison
def geometric_round(arr):
    result_array = arr
    result_array = np.where(result_array < np.sqrt(np.floor(arr)*np.ceil(arr)), np.floor(arr), result_array)
    result_array = np.where(result_array >= np.sqrt(np.floor(arr)*np.ceil(arr)), np.ceil(arr), result_array)

    return result_array


# In[19]:


y_pred = np.expm1(
    predict_model(final_blend, data=test)['Label']
)

y_pred = geometric_round(np.array(y_pred).transpose()).astype(int)
y_pred


# ### Submission

# In[20]:


submission = pd.read_csv('../input/tabular-playground-series-jan-2022/sample_submission.csv')
submission[target_col] = y_pred

submission.to_csv('submission.csv', index=False)

submission

