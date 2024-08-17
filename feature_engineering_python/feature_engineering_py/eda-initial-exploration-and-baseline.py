#!/usr/bin/env python
# coding: utf-8

# <h2>Microsoft Malware Prediction</h2>
# 
# <h2>1. Introduction</h2>
# 
# The goal of this competition is to predict a Windows machine’s probability of getting infected by malware, based on different properties of that machine. This is a binary classification problem and submissions are evaluated on the <b>area under the ROC curve</b>. For a great explanation about this metric, check [this link](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc).
# 
# This exploration notebook will cover all relevant data and show how to setup a simple model to predict the target at the end.
# 
# ![](https://storage.googleapis.com/kaggle-competitions/kaggle/10683/logos/thumb76_76.png?t=2018-09-19-16-55-15)

# In[ ]:


import gc
import numpy as np
import pandas as pd
# Seaborn and matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
# Plotly library
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
# Sklearn and lightgbm
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgbm
# Set some configurations
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
pd.set_option('max_columns', 100)
pd.set_option('max_rows', 100)
init_notebook_mode(connected=True)

# Set data types to save memory - from: https://www.kaggle.com/theoviel/load-the-totality-of-the-data
dtypes = {
    'MachineIdentifier':                                    'category',
    'ProductName':                                          'category',
    'EngineVersion':                                        'category',
    'AppVersion':                                           'category',
    'AvSigVersion':                                         'category',
    'IsBeta':                                               'int8',
    'RtpStateBitfield':                                     'float16',
    'IsSxsPassiveMode':                                     'int8',
    'DefaultBrowsersIdentifier':                            'float16',
    'AVProductStatesIdentifier':                            'float32',
    'AVProductsInstalled':                                  'float16',
    'AVProductsEnabled':                                    'float16',
    'HasTpm':                                               'int8',
    'CountryIdentifier':                                    'int16',
    'CityIdentifier':                                       'float32',
    'OrganizationIdentifier':                               'float16',
    'GeoNameIdentifier':                                    'float16',
    'LocaleEnglishNameIdentifier':                          'int8',
    'Platform':                                             'category',
    'Processor':                                            'category',
    'OsVer':                                                'category',
    'OsBuild':                                              'int16',
    'OsSuite':                                              'int16',
    'OsPlatformSubRelease':                                 'category',
    'OsBuildLab':                                           'category',
    'SkuEdition':                                           'category',
    'IsProtected':                                          'float16',
    'AutoSampleOptIn':                                      'int8',
    'PuaMode':                                              'category',
    'SMode':                                                'float16',
    'IeVerIdentifier':                                      'float16',
    'SmartScreen':                                          'category',
    'Firewall':                                             'float16',
    'UacLuaenable':                                         'float32',
    'Census_MDC2FormFactor':                                'category',
    'Census_DeviceFamily':                                  'category',
    'Census_OEMNameIdentifier':                             'float16',
    'Census_OEMModelIdentifier':                            'float32',
    'Census_ProcessorCoreCount':                            'float16',
    'Census_ProcessorManufacturerIdentifier':               'float16',
    'Census_ProcessorModelIdentifier':                      'float16',
    'Census_ProcessorClass':                                'category',
    'Census_PrimaryDiskTotalCapacity':                      'float32',
    'Census_PrimaryDiskTypeName':                           'category',
    'Census_SystemVolumeTotalCapacity':                     'float32',
    'Census_HasOpticalDiskDrive':                           'int8',
    'Census_TotalPhysicalRAM':                              'float32',
    'Census_ChassisTypeName':                               'category',
    'Census_InternalPrimaryDiagonalDisplaySizeInInches':    'float16',
    'Census_InternalPrimaryDisplayResolutionHorizontal':    'float16',
    'Census_InternalPrimaryDisplayResolutionVertical':      'float16',
    'Census_PowerPlatformRoleName':                         'category',
    'Census_InternalBatteryType':                           'category',
    'Census_InternalBatteryNumberOfCharges':                'float32',
    'Census_OSVersion':                                     'category',
    'Census_OSArchitecture':                                'category',
    'Census_OSBranch':                                      'category',
    'Census_OSBuildNumber':                                 'int16',
    'Census_OSBuildRevision':                               'int32',
    'Census_OSEdition':                                     'category',
    'Census_OSSkuName':                                     'category',
    'Census_OSInstallTypeName':                             'category',
    'Census_OSInstallLanguageIdentifier':                   'float16',
    'Census_OSUILocaleIdentifier':                          'int16',
    'Census_OSWUAutoUpdateOptionsName':                     'category',
    'Census_IsPortableOperatingSystem':                     'int8',
    'Census_GenuineStateName':                              'category',
    'Census_ActivationChannel':                             'category',
    'Census_IsFlightingInternal':                           'float16',
    'Census_IsFlightsDisabled':                             'float16',
    'Census_FlightRing':                                    'category',
    'Census_ThresholdOptIn':                                'float16',
    'Census_FirmwareManufacturerIdentifier':                'float16',
    'Census_FirmwareVersionIdentifier':                     'float32',
    'Census_IsSecureBootEnabled':                           'int8',
    'Census_IsWIMBootEnabled':                              'float16',
    'Census_IsVirtualDevice':                               'float16',
    'Census_IsTouchEnabled':                                'int8',
    'Census_IsPenCapable':                                  'int8',
    'Census_IsAlwaysOnAlwaysConnectedCapable':              'float16',
    'Wdft_IsGamer':                                         'float16',
    'Wdft_RegionIdentifier':                                'float16',
    'HasDetections':                                        'int8'
}


# The hidden cell above define the data type for each column. Now let's have a look at the data:

# In[ ]:


train = pd.read_csv('../input/train.csv', dtype=dtypes)
test = pd.read_csv('../input/test.csv', dtype=dtypes)
print("train shape", train.shape, "test shape", test.shape)
train.head(4)


# We have almost nine millions rows for training and more than 80 features. Each row is a machine that should have a unique identifier, let's see if that's true:

# In[ ]:


machine_id = pd.concat([train.MachineIdentifier, test.MachineIdentifier])
print("Max machines for each id:", machine_id.value_counts().max())
del machine_id


# <h3>Target variable</h3>
# 
# The data was sampled in a special way so that we have 50% of each label instead of a (probably) imbalanced dataset.

# In[ ]:


target_count = train.HasDetections.value_counts()
pie_trace = go.Pie(labels=target_count.index, values=target_count.values)
layout = dict(title= "HasDetections distribution", height=400, width=800)
fig = dict(data=[pie_trace], layout=layout)
iplot(fig)


# <h2>2. Cardinality and missing values</h2>
# 
# Most features are categorical in this dataset, so in this section we'll have a look at the number of categories that each features has in the train and test group. 
# 
# First, we need to list all numerical features. Some of these features can also be considered categorical, i.e. number of cores has discrete values 1, 2, 4, 6, 8...

# In[ ]:


numeric_features_list = [
    'AVProductsInstalled',
    'AVProductsEnabled',
    'Census_ProcessorCoreCount',
    'Census_PrimaryDiskTotalCapacity',
    'Census_SystemVolumeTotalCapacity',
    'Census_TotalPhysicalRAM',
    'Census_InternalPrimaryDiagonalDisplaySizeInInches',
    'Census_InternalPrimaryDisplayResolutionHorizontal',
    'Census_InternalPrimaryDisplayResolutionVertical',
    'Census_InternalBatteryNumberOfCharges'
]


# I'm also removing binary features from the next plot. As we can see, there are unseen categories in the test set and some features have thousands of categories, which means that One-Hot is not a good encoding option. It will be difficult to deal with these columns, especially because we only have an id for most items and not the real name for analysis (i.e. we don't know city name, only the identifier).
# 
# <b>Note</b>: I'm considering NaN (missing value) as a category

# In[ ]:


categorical_features_list = [f for f in train.columns if f not in numeric_features_list and 
                             train[f].dtype != 'int8' and f != 'MachineIdentifier' and f != 'HasDetections']

cardinality = pd.DataFrame({
    'feature': categorical_features_list,
    'nunique_train': [train[f].nunique(dropna=False) for f in categorical_features_list],
    'nunique_test': [test[f].nunique(dropna=False) for f in categorical_features_list],  
})
cardinality.sort_values(by='nunique_train', inplace=True)

trace0 = go.Bar(y=cardinality.feature, x=cardinality.nunique_train,
                orientation='h', marker=dict(color='rgba(222,45,38,0.9)'), name='train')
trace1 = go.Bar(y=cardinality.feature, x=cardinality.nunique_test,
                orientation='h', marker=dict(color='rgb(204,204,204)'), name='test')

layout = go.Layout(
    title='Categorical cardinality', height=1600, width=800,
    xaxis=dict(
        title='Number of categories',
        titlefont=dict(size=16, color='rgb(107, 107, 107)'),
        domain=[0.3, 1]
    ),
    barmode='group',
    bargap=0.1,
    bargroupgap=0.1
)
fig = go.Figure(data=[trace0, trace1], layout=layout)
iplot(fig)


# <h3>Missing values</h3>
# 
# There are some columns with almost all values missing. Another interesting thing is the difference between train and test for the SMode column:

# In[ ]:


missing_train = train.isna().sum() / len(train)
missing_train = missing_train[missing_train > 0]
missing_train.sort_values(ascending=True, inplace=True)

missing_test = test.isna().sum() / len(test)
missing_test = missing_test[missing_test > 0]
missing_test.sort_values(ascending=True, inplace=True)

trace1 = go.Bar(y=missing_train.index, x=missing_train.values, orientation='h',
                marker=dict(color='rgb(49,130,189)'), name='train')
trace2 = go.Bar(y=missing_train.index, x=missing_test[missing_train.index],
                orientation='h', marker=dict(color='rgb(204,204,204)'), name='test')

layout = go.Layout(
    title='Missing values', height=1600, width=800,
    xaxis=dict(
        title='Percentage of missing values',
        titlefont=dict(size=16, color='rgb(107, 107, 107)'),
        domain=[0.25, 1]
    ),
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1
)

fig = go.Figure(data=[trace1, trace2], layout=layout)
iplot(fig)


# <h2>3. Defender and Antivirus</h2>
# 
# <h3>Product Name</h3>
# 
# The first feature in this section is the Microsoft Defender product. Almost 99% of computers are using win8defender and 1% Microsoft Security Essentials. The same happens for the test set.

# In[ ]:


tmp_df = pd.DataFrame()
tmp_df['train_set'] = train.ProductName.value_counts() / len(train) * 100
tmp_df['test_set'] = test.ProductName.value_counts() / len(test) * 100
tmp_df


# <h3>Engine and App version</h3>
# 
# In the next plots the left barplot is the train dataset and the right is the test dataset. The left plot (train data) is divided in positive and negative target. We can see that the most common versions are slightly different for train and test datasets.

# In[ ]:


def target_test_barplot(col, title, max_items=None, percentage=False):
    """Custom barplot with train dataset split by target and test set."""
    test_count = test[col].value_counts().iloc[:max_items]
    neg_count = train[train.HasDetections == 0][col].value_counts()
    pos_count = train[train.HasDetections == 1][col].value_counts()
    # Order the first plot by the number of examples in train
    train_count = train[col].value_counts().iloc[:max_items]
    if percentage:
        test_count /= len(test)
        neg_count /= len(train)
        pos_count /= len(train)
    trace0 = go.Bar(
        x=train_count.index.astype('str'),
        y=neg_count[train_count.index],
        name='Target 0',
        marker=dict(color='rgba(204,204,204, 0.8)')
    )
    trace1 = go.Bar(
        x=train_count.index.astype('str'),
        y=pos_count[train_count.index],
        name='Target 1',
        marker=dict(color='rgba(222,45,38,0.9)')
    )
    trace2 = go.Bar(
        x=test_count.index.astype('str'),
        y=test_count.values,
        name='Test set',
        marker=dict(color='rgb(49,130,189)')
        #marker=dict(color='rgba(204,204,204, 0.9)'),
    )
    fig = tools.make_subplots(rows=1, cols=2, print_grid=False)
    fig.append_trace(trace0, 1, 1)
    fig.append_trace(trace1, 1, 1)
    fig.append_trace(trace2, 1, 2)

    fig['layout'].update(height=400, width=800, title=title, barmode='stack')
    fig['layout']['xaxis1'].update(type='category')
    fig['layout']['xaxis2'].update(type='category')
    iplot(fig)
    del neg_count, pos_count, test_count; gc.collect()

target_test_barplot('EngineVersion', 'Engine Version - top 6', 6, True)
target_test_barplot('AppVersion', 'App Version - top 8', 8, True)


# <h3>IsBeta, IsSxsPassiveMode</h3>
# 
# * isBeta: binary feature; the name is self explanatory - only 67 machines have a beta defender.
# * IsSxsPassiveMode: Also binary; doesnt have any description yet.
# 
# Values are in %

# In[ ]:


print(train.IsBeta.value_counts(dropna=False)/len(train)*100)
print(train.IsSxsPassiveMode.value_counts(dropna=False)/len(train)*100)


# The next feature, RtpStateBitfield, also doesnt have a description:

# In[ ]:


target_test_barplot('RtpStateBitfield', 'RtpStateBitfield')


# <h3>Antivirus installed and Enabled</h3>
# 
# These two numerical columns doesn't have a description yet, but I think it's the number of installed and enabled AV softwares.
# 
# <b>About the chart</b>: The red line is the percentage of machines with malware for that number of antivirus. The gray line is the percentage of machines in that category compared to the total train size.
# 
# In the left chart we can see that most machines have one antivirus, but the probability of being infected is smaller as this number increases. In the right plot, 97% of machines have only one product enabled.

# In[ ]:


col1, col2 = 'AVProductsInstalled', 'AVProductsEnabled'
f1 = train.groupby(col1)['HasDetections'].apply(lambda x: sum(x) / len(x))
f2 = train.groupby(col2)['HasDetections'].apply(lambda x: sum(x) / len(x))
f1 = f1[1:6]

len_items1 = train.groupby(col1)['HasDetections'].size() / len(train)
len_items2 = train.groupby(col2)['HasDetections'].size() / len(train)

trace0 = go.Scatter(
    x=f1.index.astype('str'),
    y=f1.values,
    name="Target 1",
    marker=dict(color='rgba(222,45,38,0.8)')
)
trace1 = go.Scatter(
    x=f2.index.astype('str'),
    y=f2.values,
    name="Target 1",
    marker=dict(color='rgba(222,45,38,0.8)')
)
trace2 = go.Scatter(
    x=f1.index.astype('str'),
    y=len_items1[f1.index],
    name="Percentage from total",
    marker=dict(color='rgba(204,204,204, 0.9)')
)
trace3 = go.Scatter(
    x=f2.index.astype('str'),
    y=len_items2[f2.index],
    name="Percentage from total",
    marker=dict(color='rgba(204,204,204, 0.9)')
)
fig = tools.make_subplots(rows=1, cols=2, print_grid=False)
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)
fig.append_trace(trace2, 1, 1)
fig.append_trace(trace3, 1, 2)

fig['layout'].update(height=440, width=800, title="Antivirus installed and enabled",
                     showlegend=False)
fig['layout']['xaxis1'].update(type='category', title=col1)
fig['layout']['xaxis2'].update(type='category', title=col2)
iplot(fig)
del f1, f2; gc.collect();


# <h3>AvSigVersion</h3>
# 
# This is the most important feature in the current kernels using GBM and is described as the current defender state information.
# 
# <b>About the chart</b>: The next plot has the number of occurrences in y-axis and the percentage of positive labels in x-axis. You can check the category name with the hoover in each point.

# In[ ]:


def scatter_train(col, title, color=None):
    """Plot the count of each category in y-axis and the positve label in x-axis."""
    count = train[col].value_counts(ascending=True)
    positive = train[train.HasDetections == 1][col].value_counts()
    trace = go.Scattergl(x=positive[count.index]/count.values,
                         y=count.values, mode='markers',
                         text=count.index.values.astype('str'),
                         marker=dict(color=color))

    layout = go.Layout(
        title=title, height=500, width=800,
        xaxis=dict(title='Percentage of positive label'),
        yaxis=dict(title='Number of samples in train'),
    )
    fig = go.Figure(data=[trace], layout=layout)
    iplot(fig)

scatter_train('AvSigVersion', 'Defender state (AvSigVersion)')


# <h3>AV Product State</h3>
# 
# ID for the specific configuration of a user's antivirus software. The configuration number 53447 has almost 6M machines with 55% detection rate, while the others 28.969 configurations have less than half million machines and very different detection ratios. In the next plot I'll remove configuration 53447 for better visualization:

# In[ ]:


col = 'AVProductStatesIdentifier'
count = train[col].value_counts(ascending=True)
count.drop(labels=[53447], inplace=True)
positive = train[train.HasDetections == 1][col].value_counts()
trace = go.Scattergl(x=positive[count.index]/count.values,
                     y=count.values, mode='markers',
                     text=count.index.values.astype('str'),
                     marker=dict(color='#DA6C38'))

layout = go.Layout(
    title="Antivirus product state", height=600, width=800,
    xaxis=dict(title='Percentage of positive label'),
    yaxis=dict(title='Number of samples in train'),
)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)


# <h3>Trusted Platform Module</h3>
# 
# > Trusted Platform Module (TPM, also known as ISO/IEC 11889) is an international standard for a secure cryptoprocessor, a dedicated microcontroller designed to secure hardware through integrated cryptographic keys.
# 
# In this case I think this is actually the chip that provides this module for cryptographic keys and security.

# In[ ]:


train.HasTpm.value_counts(dropna=False) / len(train) * 100


# <h3>Firewall</h3>

# In[ ]:


train.Firewall.value_counts(dropna=False) / len(train) * 100


# <h2>4. Geographic location and Organization</h2>

# <h3>City and Country</h3>
# 
# There are 222 countries and 107.366 cities in the train dataset:

# In[ ]:


scatter_train('CountryIdentifier', 'Countries in train set', color=None)


# In[ ]:


scatter_train('CityIdentifier', 'Cities in train set', color='#DA6C38')


# <h2>5. Platform and OS</h2>
# 
# All machines are running windows with most being windows 10; nothing relevant here.

# In[ ]:


target_test_barplot('Platform', 'OS Platform')


# <h3>Processor</h3>
# 
# It seems that x86 architecture is less susceptible to malwares or maybe the users are more carefull:

# In[ ]:


target_test_barplot('Processor', 'processor')


# <h3>OS Build</h3>
# 
# Build of the current operating system.

# In[ ]:


target_test_barplot('OsBuild', 'OS Build', 8, percentage=True)


# <h3>Smart Screen</h3>
# 
# > Windows SmartScreen is a cloud-based anti-phishing and anti-malware component included in several Microsoft products, including Windows 8 and later, Internet Explorer, Microsoft Edge and Outlook.com. It is designed to help protect users against attacks that utilize social engineering and drive-by downloads to infect a system by scanning URLs accessed by a user against a blacklist of websites containing known threats. With the Windows 10 Creators Update, Microsoft placed the SmartScreen settings into the Windows Defender Security Center. (Wikipedia)

# In[ ]:


target_test_barplot('SmartScreen', 'Windows Smart Screen', percentage=True)


# <h2>6. Census</h2>
# 
# Statistics about the hardware

# <h3>Machine Type</h3>

# In[ ]:


target_test_barplot('Census_ChassisTypeName', 'Chassis Type - top 5', 5, percentage=True)


# <h3>CPU Cores and Memory</h3>
# 
# These two features can also be treated as categoricals.
# 
# Most machines (around 95%)  are using 2, 4 or 8 cores: 

# In[ ]:


target_test_barplot('Census_ChassisTypeName', 'Number of Cores - top 5', 5, percentage=True)


# In[ ]:


def distplot(col, label):
    plt.figure(figsize=(8,4))
    ax = sns.kdeplot(train[col].dropna(), label=label)
    ax = sns.kdeplot(train[train.HasDetections == 1][col].dropna(),
                     label=label + '; target=1')

distplot('Census_TotalPhysicalRAM', 'Physical RAM')


# There are only 85,000 machines with more than 16 GB of RAM memory. I removed these items for better visualization:

# In[ ]:


plt.figure(figsize=(8,4))
limit = 18000
ax = sns.kdeplot(train[train.Census_TotalPhysicalRAM < limit].Census_TotalPhysicalRAM.dropna(),
                 label='Physical RAM')
ax = sns.kdeplot(train[(train.HasDetections == 1) & 
                       (train.Census_TotalPhysicalRAM < limit)].Census_TotalPhysicalRAM.dropna(),
                 label='Physical RAM; target=1')


# <h3>Disk capacity</h3>
# 
# Primary disk capacity in MB.

# In[ ]:


distplot('Census_PrimaryDiskTotalCapacity', 'Disk capacity')


# <h3>System volume capacity</h3>
# 
# Capacity (in MB) for the volume where the system is installed.

# In[ ]:


distplot('Census_SystemVolumeTotalCapacity', 'System volume capacity')


# <h2>7. Baseline for predictions</h2>
# 
# Let's build a simple [Gradient boosting](http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/) pipeline and see how it does.
# 
# Since kernel memory is limited, I'll be using a sample with 5M rows of train data. I'm also deleting the test frame and reading it later.

# In[ ]:


# Sample rows from train data
train = train.sample(n=5000000, random_state=2018)

target = train['HasDetections']  # Save the target
del train['HasDetections'], test; gc.collect()

# Drop some features that are not usefull
drop_feats = ['Census_ProcessorClass', 'IsBeta', 'HasTpm', 'Platform',
              'AutoSampleOptIn', 'ProductName', 'MachineIdentifier', 'PuaMode']
train.drop(drop_feats, axis=1, inplace=True)
print("train shape", train.shape)


# <h3>Label encode</h3>
# 
# Simplified label encode for categorical features

# In[ ]:


def label_encoder(df, categorical_cols, indexer=None):
    if not indexer:
        indexer = {}
        for col in categorical_cols:
            df[col], indexer[col] = pd.factorize(df[col])
        return df, indexer
    else:
        for col in categorical_columns:
            df[col] = indexer[col].get_indexer(df[col])
        return df
    
categorical_columns = [c for c in train.columns if c not in
                       numeric_features_list and train[c].dtype != 'int8']
train, indexer = label_encoder(train, categorical_columns)


# <h3>Some basic Feature Engineering</h3>
# 
# Just trying a few ratios with numerical features:

# In[ ]:


def feature_engineering(df):
    # Memory per CPU core
    df['ram_per_core'] = df.Census_TotalPhysicalRAM / df.Census_ProcessorCoreCount
    # Memory to disk space ratio
    df['ram_to_disk_capacity'] = df.Census_TotalPhysicalRAM / df.Census_PrimaryDiskTotalCapacity
    # Space on system partition to total space on disk
    df['system_volume_to_disk_capacity'] = df.Census_SystemVolumeTotalCapacity / df.Census_PrimaryDiskTotalCapacity
    return df

train = feature_engineering(train)
train.head(4)


# Reduce the memory usage:

# In[ ]:


def reduce_memory(df):
    """Reduce memory usage of a dataframe by setting data types. """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Initial df memory usage is {:.2f} MB for {} columns'
          .format(start_mem, len(df.columns)))

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:
            cmin = df[col].min()
            cmax = df[col].max()
            if str(col_type)[:3] == 'int':
                # Can use unsigned int here too
                if cmin > np.iinfo(np.int8).min and cmax < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif cmin > np.iinfo(np.int16).min and cmax < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif cmin > np.iinfo(np.int32).min and cmax < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif cmin > np.iinfo(np.int64).min and cmax < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if cmin > np.finfo(np.float16).min and cmax < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif cmin > np.finfo(np.float32).min and cmax < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    memory_reduction = 100 * (start_mem - end_mem) / start_mem
    print('Final memory usage is: {:.2f} MB - decreased by {:.1f}%'.format(end_mem, memory_reduction))
    return df

train = reduce_memory(train)


# <h3>Cross-validation</h3>
# 
# I'm using the .cv function from lightgbm api to perform cross-validation. It's much shorter (in code) than using Sklearn KFold and has the advantage of returning the standard deviation between folds on each round. I'm also using early stopping rounds to avoid overfitting.
# 
# <b>Categorical features parameter</b>: LightGBM has a partition method to deal with label encoded features, however when the cardinality is too high it's better to just leave the feature as integers (or use more advanced encoding like target encode). Refer to this [page](https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html) for more details about this topic.

# In[ ]:


params = {
    'boosting': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'random_state': 2018,
    'verbosity': -1,
    'learning_rate': 0.04,
    'num_leaves': 68,
    'max_depth': 7,
    'colsample_bytree': 0.85,
    'subsample': 0.85,
    'min_data_in_leaf': 80,
    'reg_alpha': 0.1
}

feat_names = list(train.columns)
# Select categorical features with reasonable cardinality; testing with all
categorical_features = [f for f in categorical_columns if train[f].nunique() < 999999]
# Create lightgbm dataset
train_dataset = lgbm.Dataset(train, label=target, feature_name=feat_names,
                             categorical_feature=categorical_features,
                             free_raw_data=False)
del train, target; gc.collect()

# Perform cross-validation
cv_auc = lgbm.cv(params, train_dataset, num_boost_round=6000, early_stopping_rounds=20)
cv_best_round = len(cv_auc['auc-mean'])
print("CV score:", cv_auc['auc-mean'][-1],
      "std between folds:", cv_auc['auc-stdv'][-1],
      "num rounds", cv_best_round)


# <h3>Train model</h3>
# 
# The .cv function doesn't return a booster class (model), so we have to train the model again, but now we have the right number of rounds from cross-validation. 

# In[ ]:


model = lgbm.train(params, train_dataset, num_boost_round=cv_best_round)
del train_dataset; gc.collect()


# <h3>Load test and make predictions</h3>

# In[ ]:


test = pd.read_csv('../input/test.csv', dtype=dtypes)
test_id = test.MachineIdentifier  # Save machine ids for submission
test.drop(drop_feats, axis=1, inplace=True)
test = label_encoder(test, categorical_columns, indexer)
test = feature_engineering(test)
test = reduce_memory(test)


# I was having problems with memory when making predictions, so I'm using batches of 100,000 machines:

# In[ ]:


predictions = np.zeros(len(test))
chunk_size = 100000
for i in range(0, len(test), chunk_size):
    predictions[i:i + chunk_size] = model.predict(test.iloc[i:i + chunk_size])
del test; gc.collect()


# <h3>Submission file</h3>

# In[ ]:


sub_df = pd.DataFrame({"MachineIdentifier": test_id.values})
sub_df["HasDetections"] = predictions
sub_df.to_csv("submit.csv", index=False)
sub_df[:5]


# <h3>Feature importance</h3>

# In[ ]:


feat_importance = pd.DataFrame()
feat_importance["feature"] = feat_names
feat_importance["gain"] = model.feature_importance(importance_type='gain')
feat_importance.sort_values(by='gain', ascending=True, inplace=True)

trace = go.Bar(y=feat_importance.feature, x=feat_importance.gain,
               orientation='h', marker=dict(color='rgb(49,130,189)'))

layout = go.Layout(
    title='Feature importance', height=1600, width=800,
    showlegend=False,
    xaxis=dict(
        title='Importance by gain',
        titlefont=dict(size=14, color='rgb(107, 107, 107)'),
        domain=[0.25, 1]
    ),
)

fig = go.Figure(data=[trace], layout=layout)
iplot(fig)


# <h2>Work in progress...</h2>
