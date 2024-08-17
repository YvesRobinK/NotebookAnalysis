#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Hey, thanks for viewing my Kernel!
# 
# If you like my work, please, leave an upvote: it will be really appreciated and it will motivate me in offering more content to the Kaggle community ! 😊
# 
# ![Directions](https://upload.wikimedia.org/wikipedia/commons/1/1a/Brosen_windrose.svg)
# [source](https://en.wikipedia.org/wiki/Cardinal_direction)

# In[1]:


import pandas as pd
import numpy as np
import warnings

warnings.simplefilter("ignore")
train = pd.read_csv('../input/tabular-playground-series-mar-2022/train.csv')
test = pd.read_csv('../input/tabular-playground-series-mar-2022/test.csv')
sub = pd.read_csv('../input/tabular-playground-series-mar-2022/sample_submission.csv')

display(train.head())
display(sub.head())


# In[2]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int8','int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
 
    return df


# In[3]:


train = reduce_mem_usage(train)
test = reduce_mem_usage(test)


# In[4]:


print(train.shape)
print(test.shape)


# In[5]:


display(train.isna().sum())
display(test.isna().sum())


# In[6]:


display(train.duplicated().sum())
display(test.duplicated().sum())


# In[7]:


display(train.nunique())
display(test.nunique())


# # Distributions

# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(16, 8))
mean, std = train['congestion'].mean(), train['congestion'].std()
normal_dist = np.random.normal(mean, std, len(train))
sns.kdeplot(data=train, x='congestion', ax=ax, label='congestion')
sns.kdeplot(x=normal_dist, ax=ax, label='normal distribution')
ax.set_title('KDE Plot')
plt.legend();


# In[9]:


from scipy import stats

congestion_cdf = stats.norm.cdf(train['congestion'])
norm_cdf = stats.norm.cdf(normal_dist)
fig, ax = plt.subplots(figsize=(16, 8))
sns.lineplot(x=train['congestion'], y=congestion_cdf, ax=ax, label='congestion', alpha=0.5, linewidth=3)
sns.lineplot(x=normal_dist, y=norm_cdf, ax=ax, label='normal distribution', alpha=0.5, linewidth=3)
ax.set_title('Cumulative Distribution Function')
plt.legend();


# In[10]:


train_group = train.groupby('time', as_index=False).agg({'congestion': 'mean'})
test['congestion'] = train['congestion'].mean()
test_group = test.groupby('time', as_index=False).agg({'congestion': 'mean'})
fig, ax = plt.subplots(figsize=(16, 8))
sns.lineplot(data=train_group, x='time', y='congestion', ax=ax, label='train');
sns.lineplot(data=test_group, x='time', y='congestion', ax=ax, label='test');


# In[11]:


fig, ax = plt.subplots(figsize=(16, 8))
sns.boxplot(data=train, x='direction', y='congestion', ax=ax);


# In[12]:


fig, ax = plt.subplots(figsize=(16, 8))
sns.boxplot(data=train, x='x', y='congestion', hue='y', ax=ax);


# In[13]:


def count_same_way(df, ax=None, colormapbool=False, time=''):
    direction_dict = {
        'EB': [1, 0],
        'NB': [0, 1],
        'SB': [0, -1],
        'WB': [-1, 0],
        'NE': [1, 1],
        'SW': [-1, -1],
        'NW': [-1, 1],
        'SE': [1, -1]
    }
    
    if True:
        import matplotlib.colors as mcolors
        import matplotlib.cm as cm
        
        if ax == None:
            fig, ax = plt.subplots(figsize=(12, 16))
        sns.scatterplot(data=df, x='x', y='y', ax=ax)
        colormap = cm.Reds
        normalize = mcolors.Normalize(vmin=0, vmax=72)
        
        for x in df['x'].unique():
            for y in df['y'].unique():
                temp_df = df.loc[(df['x'] == x) & (df['y'] == y), ['direction', 'congestion']]
                temp_df = temp_df.groupby('direction', as_index=False).agg({'congestion':'mean'})
                for direction in temp_df['direction']:
                    xx, yy = direction_dict[direction]
                    x1 = x + xx / 4
                    y1 = y + yy / 4
                    mean_congestion = temp_df.loc[temp_df['direction'] == direction, 'congestion'].values[0]
                    linewidth = mean_congestion / 10
                    ax.plot([x, x1], [y, y1], linewidth=linewidth, color=colormap(normalize(mean_congestion)))
        ax.set_title('Roadway-Congestion Relationship\n' +'time: '+time)
        if colormapbool:
            scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
            scalarmappaple.set_array(np.arange(0,72))
            plt.colorbar(scalarmappaple, ax=ax)


# In[14]:


times = train['time'].unique()
rows = 3
cols = 4
fig, axes = plt.subplots(rows, cols, figsize=(24, 24))
index = 0
for row in range(rows):
    for col in range(cols):
        colormapbool = False
        if col == cols - 1:
            colormapbool = True
        count_same_way(train.loc[train['time'] == times[index], :], ax=axes[row][col], 
                       time=times[index], colormapbool=colormapbool)
        index += 1


# In[15]:


fig, ax = plt.subplots(figsize=(16, 12))
count_same_way(train, ax=ax, time='all time', colormapbool=True)


# # Feature Engineering

# In[16]:


def create_same_way_dict(df):
    direction_dict = {
        'EB': [1, 0],
        'NB': [0, 1],
        'SB': [0, -1],
        'WB': [-1, 0],
        'NE': [1, 1],
        'SW': [-1, -1],
        'NW': [-1, 1],
        'SE': [1, -1]
    }
    df['location'] = df['x'].astype(str) + df['y'].astype(str)
    
    for key in direction_dict.keys():
        direction_same_dict = {}
        direction_same_list = []
        dir_x, dir_y = direction_dict[key]
        for y in df['y'].unique():
            for x in df['x'].unique():
                new_x, new_y = x + dir_x, y + dir_y
                neighbors = []
                while str(new_x) + str(new_y) in df['location'].unique():
                    neighbors.append(str(new_x) + str(new_y))
                    new_x, new_y = new_x + dir_x, new_y + dir_y
                direction_same_list.append({str(x) + str(y):neighbors})
        
        direction_same_dict[key] = direction_same_list
        print(direction_same_dict)
        break

#create_same_way_dict(train)


# In[17]:


import holidays

def create_time_features(df, time_col):
    df[time_col] = pd.to_datetime(df[time_col])
    df['week']= df[time_col].dt.week
    #df['year'] = 'Y' + df[time_col].dt.year.astype(str)
    df['quarter'] = 'Q' + df[time_col].dt.quarter.astype(str)
    df['day'] = df[time_col].dt.day
    df['dayofyear'] = df[time_col].dt.dayofyear
    df.loc[(df[time_col].dt.is_leap_year) & (df.dayofyear >= 60),'dayofyear'] -= 1
    df['weekend'] = df[time_col].dt.weekday >=5
    df['weekday'] = 'WD' + df[time_col].dt.weekday.astype(str)
    df['month']= 'M' + df[time_col].dt.month.astype(str)
    df['hour']= 'h' + df[time_col].dt.hour.astype(str)
    df['minute']= 'm' + df[time_col].dt.minute.astype(str)
    
    holidays_list = holidays.US(years=df[time_col].dt.year.values)
    
    df['holiday'] = 0
    df.loc[df[time_col].isin(list(holidays_list.keys())), 'holiday'] = 1
    
    return df


# In[18]:


train = create_time_features(train, 'time')
test = create_time_features(test, 'time')
train.head()


# In[19]:


def plot_periodogram(ts, detrend='linear', ax=None, title=''):
    from scipy.signal import periodogram
    fs = pd.Timedelta("1Y") / pd.Timedelta("1D")
    # fs = train['time'].max() - train['time'].min()
    # fs.total_seconds() / (60 * 60* 24)
    fs = 182.48611111111111
    freqencies, spectrum = periodogram(
        ts,
        fs=fs,
        detrend=detrend,
        window="boxcar",
        scaling='spectrum',
    )
    if ax is None:
        _, ax = plt.subplots()
    ax.step(freqencies, spectrum, color="purple")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels(
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)",
            "Biweekly (26)",
            "Weekly (52)",
            "Semiweekly (104)",
        ],
        rotation=90,
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram - " + title)
    return ax


# In[20]:


train_group = train.groupby('time', as_index=False).agg({'congestion': 'mean'})
fig, ax = plt.subplots(figsize=(16, 8))
plot_periodogram(train_group['congestion'], ax=ax, title='Congestion');


# In[21]:


def create_deterministic_features(df, col, furier_order=1):
    from statsmodels.tsa.deterministic import CalendarFourier, DeterministicProcess

    fourier = CalendarFourier(freq="A", order=furier_order)

    dp = DeterministicProcess(
        index=df[col],
        constant=False,               # dummy feature for bias (y-intercept)
        order=1,                     # trend (order 1 means linear)
        #seasonal=True,               # weekly seasonality (indicators)
        additional_terms=[fourier],  # annual seasonality (fourier)
        drop=True                   # drop terms to avoid collinearity
    )
    
    new_df = dp.in_sample()
    new_df.index = df.index
    df_concated = pd.concat([df, new_df], axis=1)

    return df_concated


# In[22]:


train_test = pd.concat([train, test], 0)
train_test.reset_index(inplace=True)
train_test_featured = create_deterministic_features(train_test, 'time', furier_order=3)
train_featured = train_test_featured.loc[:len(train),:]
test_featured = train_test_featured.loc[len(train):,:]

train_featured.head()


# In[23]:


def create__time_sincos_wave(df, time_col=''):
    import math
    
    # Minute
    df['minute_sin'] = np.sin(df[time_col].dt.minute / 59 * 2 * math.pi)
    df['minute_cos'] = np.cos(df[time_col].dt.minute / 59 * 2 * math.pi)
    
    # Hour
    df['hour_sin'] = np.sin(df[time_col].dt.hour / 23 * 2 * math.pi)
    df['hour_cos'] = np.cos(df[time_col].dt.hour / 23 * 2 * math.pi)
    
    # Day of Week
    df['dayofweek_sin'] = np.sin(df[time_col].dt.dayofweek / 6 * 2 * math.pi)
    df['dayofweek_cos'] = np.cos(df[time_col].dt.dayofweek / 6 * 2 * math.pi)
    
    # Day of Month
    df['dayofmonth_sin'] = np.sin(df[time_col].dt.day / 31 * 2 * math.pi)
    df['dayofmonth_cos'] = np.cos(df[time_col].dt.day / 31 * 2 * math.pi)
    
    # Day of Year
    df['dayofyear_sin'] = np.sin(df[time_col].dt.dayofyear / 365 * 2 * math.pi)
    df['dayofyear_cos'] = np.cos(df[time_col].dt.dayofyear / 365 * 2 * math.pi)
    
    # Week of Month
    df['weekofmonth_sin'] = np.sin(df[time_col].apply(lambda x: (x.day-1) // 7) / 3 * 2 * math.pi)
    df['weekofmonth_cos'] = np.cos(df[time_col].apply(lambda x: (x.day-1) // 7) / 3 * 2 * math.pi)
    
    # Week of Year
    df['weekofyear_sin'] = np.sin(df[time_col].dt.dayofyear / 52 * 2 * math.pi)
    df['weekofyear_cos'] = np.cos(df[time_col].dt.dayofyear / 52 * 2 * math.pi)
    
    # Month of Quarter
    df['monthofquarter_sin'] = np.sin(df[time_col].apply(lambda x: (x.month-1) // 4) / 2 * 2 * math.pi)
    df['monthofquarter_cos'] = np.cos(df[time_col].apply(lambda x: (x.month-1) // 4) / 2 * 2 * math.pi)
    
    # Month of Year
    df['monthofyear_sin'] = np.sin(df[time_col].dt.month / 11 * 2 * math.pi)
    df['monthofyear_cos'] = np.cos(df[time_col].dt.month / 11 * 2 * math.pi)
    
    # Quarter of Year
    df['quarterofyear_sin'] = np.sin(df[time_col].dt.quarter / 3 * 2 * math.pi)
    df['quarterofyear_cos'] = np.cos(df[time_col].dt.quarter / 3 * 2 * math.pi)
    
    #return df


# In[24]:


create__time_sincos_wave(train_featured, time_col='time')
create__time_sincos_wave(test_featured, time_col='time')


# In[25]:


train_featured.dropna(inplace=True)
train_featured.to_pickle('train_featured.pkl')
test_featured.to_pickle('test_featured.pkl')


# In[26]:


train_featured_group = train_featured.groupby('time', as_index=False).agg({
    'congestion':'median', 'minute_sin':'median', 'minute_cos':'median', 'hour_sin':'median', 'hour_cos':'median', 'dayofweek_sin':'median', 'dayofweek_cos':'median',
    'dayofmonth_sin':'median', 'dayofmonth_cos':'median', 'dayofyear_sin':'median', 'dayofyear_cos':'median', 'weekofmonth_sin':'median', 'weekofmonth_cos':'median',
    'weekofyear_sin':'median', 'weekofyear_cos':'median', 'monthofquarter_sin':'median', 'monthofquarter_cos':'median', 'monthofyear_sin':'median', 'monthofyear_cos':'median',
    'quarterofyear_sin':'median', 'quarterofyear_cos':'median'
})
train_featured_group['congestion_norm'] = (train_featured_group['congestion'] - train_featured_group['congestion'].min()) / (
    train_featured_group['congestion'].max() - train_featured_group['congestion'].min())
train_featured_group['congestion_norm'] = (train_featured_group['congestion_norm'] - 0.5) * 2


# In[27]:


fig, axes = plt.subplots(3, 1, figsize=(16, 16))
sns.lineplot(data=train_featured_group, x='time', y='hour_sin', alpha=0.5, label='sin', ax=axes[0])
sns.lineplot(data=train_featured_group, x='time', y='hour_cos', alpha=0.5, label='cos', ax=axes[0])
sns.lineplot(data=train_featured_group, x='time', y='congestion_norm', alpha=0.5, label='congestion', ax=axes[0])
axes[0].set_title('Hours')
axes[0].legend();
sns.lineplot(data=train_featured_group, x='time', y='dayofweek_sin', alpha=0.5, label='sin', ax=axes[1])
sns.lineplot(data=train_featured_group, x='time', y='dayofweek_cos', alpha=0.5, label='cos', ax=axes[1])
sns.lineplot(data=train_featured_group, x='time', y='congestion_norm', alpha=0.5, label='congestion', ax=axes[1])
axes[1].set_title('Day of Week')
axes[1].legend();
sns.lineplot(data=train_featured_group, x='time', y='dayofmonth_sin', alpha=0.5, label='sin', ax=axes[2])
sns.lineplot(data=train_featured_group, x='time', y='dayofmonth_cos', alpha=0.5, label='cos', ax=axes[2])
sns.lineplot(data=train_featured_group, x='time', y='congestion_norm', alpha=0.5, label='congestion', ax=axes[2])
axes[2].set_title('Day of Month')
axes[2].legend();


# # Pearson Correlation

# In[28]:


pearson_corr = pd.DataFrame(train_featured.corrwith(train_featured['congestion'], method='pearson'), 
                            columns=['congestion'])

def p_value_warning_background(cell_value):
    highlight = 'background-color: lightcoral;'
    default = ''
    if cell_value > 0.03 or cell_value < -0.03:
            return highlight
    return default

pearson_corr.style.applymap(p_value_warning_background)


# # Modeling

# In[29]:


from catboost import CatBoostRegressor

train_featured['x'] = train_featured['x'].astype(str)
train_featured['y'] = train_featured['y'].astype(str)
numeric_cols = train_featured.select_dtypes(include=np.number).columns.tolist()
object_cols = list(set(train_featured.columns) - set(numeric_cols))
numeric_cols.remove("congestion")
object_cols.remove("time")
ignore_cols = ['index', 'row_id']

train_featured.drop(ignore_cols, 1, inplace=True)
test_featured.drop(ignore_cols, 1, inplace=True)

cat_base = CatBoostRegressor(
    #ignored_features=ignore_cols,
    cat_features=object_cols,
    eval_metric='MAE'
)


# In[30]:


X_train = train_featured.drop(['congestion', 'time'], 1)
y_train = train_featured['congestion']
cat_base.fit(X_train, y_train, silent=True)


# In[31]:


X_test = test_featured.drop(['congestion', 'time'], 1)
preds = pd.DataFrame(cat_base.predict(X_test), columns=['preds'])
preds = preds.round()
preds.head()


# In[32]:


sub['congestion'] = preds['preds']
sub.to_csv('baseline_preds.csv', index=False)


# # Feature Importance

# In[33]:


def plot_feature_importance(importance,names,model_type):
    
    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    
    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)
    
    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    
    #Define size of bar plot
    plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title(model_type + 'FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')


# In[34]:


#plot the catboost result
plot_feature_importance(cat_base.get_feature_importance(), X_train.columns, 'CATBOOST')


# In[35]:


import shap

explainer = shap.TreeExplainer(cat_base)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)


# In[36]:


shap.initjs()
shap.force_plot(explainer.expected_value, shap_values, X_test)

