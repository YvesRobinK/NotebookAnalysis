#!/usr/bin/env python
# coding: utf-8

# ### <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#006600; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #003300">🧠 Notebook At a Glance</p>

# ![image.png](attachment:3ebb806e-016a-4324-9ef5-81882c62fb4a.png)

# #### <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#006600; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #003300">✅ Data preparation overview</p>

# ![image.png](attachment:60d401fd-2f8b-447d-8126-8657b094184f.png)

# <div style="border-radius:10px; border:#DEB887 solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#DEB887'>💡 Notes:</font></h3>
# 
# * tdcsfog dataset are collected from the lab. Therefore, data quality is good comparing with defog dataset.
# * defog dataset are collected from subjects' home, therefore there are two additioanl columns (valid & test) to check the quality of data.
# * On evaluation precess, it only uses validly annotated data. Therefore, it's reasonable to remove invalid data from defog dataset.
# * However, as the document mentioned, we can use it for developing semi or unsupervised model (with notype dataset)
# * As the size of dataset is quite huge, we should convert data type to reduce memory usage.
# * Need to build two different model. one for tdcsfog and one for defog. Because, via EDA, you could find that both datasets' distribution is different.
# * This is just an initial plan for modeling. It could be changed as we dig more into the dataset.

# In[1]:


# import library
import os
import random
import cv2
import pandas as pd
import numpy as np


# In[2]:


# Reduce Memory Usage
# reference : https://www.kaggle.com/code/arjanso/reducing-dataframe-memory-size-by-65 @ARJANGROEN

def reduce_memory_usage(df):
    
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype.name
        if ((col_type != 'datetime64[ns]') & (col_type != 'category')):
            if (col_type != 'object'):
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
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        pass
            else:
                df[col] = df[col].astype('category')
    mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage became: ",mem_usg," MB")
    
    return df


# In[3]:


#reference: https://www.kaggle.com/code/ghrangel/read-data-and-merge

DATA_ROOT_DEFOG = '/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/train/defog/'
defog = pd.DataFrame()
for root, dirs, files in os.walk(DATA_ROOT_DEFOG):
    for name in files:       
        f = os.path.join(root, name)
        df_list= pd.read_csv(f)
        words = name.split('.')[0]
        df_list['file']= name.split('.')[0]
        defog = pd.concat([defog, df_list], axis=0)

defog
       


# In[4]:


defog = reduce_memory_usage(defog)


# > #### 💬 we reduced memory usage from 954MB to 335MB

# In[5]:


defog = defog[(defog['Task']==1)&(defog['Valid']==1)]


# > #### 💬 As I mentioned above, We are going to use valid data only.

# In[6]:


print('the shape of defog dataset is {}'.format(defog.shape))


# > #### 💬 Now it's time to combine it with metadata.

# In[7]:


defog_metadata = pd.read_csv("/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/defog_metadata.csv")
defog_metadata


# In[8]:


defog_m= defog_metadata.merge(defog, how = 'inner', left_on = 'Id', right_on = 'file')
defog_m.drop(['file','Valid','Task'], axis = 1, inplace = True)
defog_m


# In[9]:


# summary table function
def summary(df):
    print(f'data shape: {df.shape}')
    summ = pd.DataFrame(df.dtypes, columns=['data type'])
    summ['#missing'] = df.isnull().sum().values * 100
    summ['%missing'] = df.isnull().sum().values / len(df)
    summ['#unique'] = df.nunique().values
    desc = pd.DataFrame(df.describe(include='all').transpose())
    summ['min'] = desc['min'].values
    summ['max'] = desc['max'].values
    summ['first value'] = df.loc[0].values
    summ['second value'] = df.loc[1].values
    summ['third value'] = df.loc[2].values
    
    return summ


# > #### 💬 Let's look at the summary table for defog dataset (data from subjects' home)

# In[10]:


summary(defog_m)


# In[11]:


# garbage collection for memory
import gc
gc.collect()


# > #### 💬 prepare tdcsfog dataset for modeling (data collected from the lab🥼)

# In[12]:


DATA_ROOT_TDCSFOG = '/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/train/tdcsfog/'
tdcsfog = pd.DataFrame()
for root, dirs, files in os.walk(DATA_ROOT_TDCSFOG):
    for name in files:       
        f = os.path.join(root, name)
        df_list= pd.read_csv(f)
        words = name.split('.')[0]
        df_list['file']= name.split('.')[0]
        tdcsfog = pd.concat([tdcsfog, df_list], axis=0)
tdcsfog


# In[13]:


tdcsfog = reduce_memory_usage(tdcsfog)


# > #### 💬 we reduced memory usage from 484MB to 154MB

# In[14]:


tdcsfog_metadata = pd.read_csv("/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/tdcsfog_metadata.csv")
tdcsfog_metadata


# In[15]:


tdcsfog_m= tdcsfog_metadata.merge(tdcsfog, how = 'inner', left_on = 'Id', right_on = 'file')
tdcsfog_m.drop(['file'], axis = 1, inplace = True)
tdcsfog_m


# In[16]:


# garbage collection for memory
import gc
gc.collect()


# #### <p style="font-family:JetBrains Mono; font-weight:bold; letter-spacing: 2px; color:#006600; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #003300">✅ Feature engineering and modeling</p>

# <div style="border-radius:10px; border:#DEB887 solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#DEB887'>💡 Notes:</font></h3>
# 
# * For illustrative purpose, we will develop very simple multi-classification model with LGBM.
# 
# * This is time series data, therefore we should creat time-related varaible so that it could reflect the change along with the time.
#     
# * On this notebook, I wiil skip time series feature engineering process for now.
#     
# * In the ground truth, only one event class has a non-zero value for each Id, but there is no restriction on the values of predicted scores. -> multi-class task!

# ![image.png](attachment:ad9ae941-adb6-4adf-80db-f3ffbea0df60.png)

# In[17]:


from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GridSearchCV
import warnings
warnings.filterwarnings('ignore')


# In[18]:


conditions = [
    (defog_m['StartHesitation'] == 1),
    (defog_m['Turn'] == 1),
    (defog_m['Walking'] == 1)]
choices = ['StartHesitation', 'Turn', 'Walking']
defog_m['event'] = np.select(conditions, choices, default='Normal')


# In[19]:


defog_m['event'].value_counts().to_frame().style.background_gradient()


# > #### 💬 Turn is the most frequently occured event while StartHesitation rarely occurs...

# In[20]:


train_df = defog_m[['AccV','AccML','AccAP','event']]


# > #### 💬 As I mentioned on the notes, I will skip feature engineering process and just use three sensor data as inputs of the model. This model does not consider time-related effect.

# In[21]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

train_df['target'] = le.fit_transform(train_df['event'])


# In[22]:


X = train_df.drop(['event','target'], axis=1)
y = train_df['target']


# > #### 💬 train simple LGBM model without hyper-parameter tuning. the size of dataset is quite huge, it might take a lot of time for traing a decent model.

# In[23]:


import lightgbm as lgb


# split dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1004)

#Converting the dataset in proper LGB format
d_train=lgb.Dataset(X_train, label=y_train)
#setting up the parameters
params={}
params['learning_rate']=0.03
params['boosting_type']='gbdt' #GradientBoostingDecisionTree
params['objective']='multiclass' #Multi-class target feature
params['metric']='multi_logloss' #metric for multi-class
params['max_depth']=7
params['num_class']=4 #no.of unique values in the target class not inclusive of the end value
params['verbose']=-1
#training the model
clf=lgb.train(params,d_train,1000)  #training the model on 1,000 epocs
#prediction on the test dataset
y_pred_1=clf.predict(X_test)


# > #### 💬 Let's look at what it predicts

# In[24]:


y_pred_1[:1]


# > #### 💬 it shows the probability of beloing each class (event) ; you can take the highest probability by using numpy argmax function as below, and check average precision.

# According to the Evaluation notice, it says "Submissions are evaluated by the Mean Average Precision of predictions for each event class. We compute the average precision on predicted confidence scores separately for each of the three event classes (see the Data Description for more details) and take the average of these three scores to get the overall score."
# ![image.png](attachment:0b18ab7f-dfb2-423c-bf9a-d4be68dc93ff.png)

# In[25]:


# 'macro' option is to calculate metrics for each label, and find their unweighted mean. 
# This does not take label imbalance into account.
from sklearn.metrics import precision_score
precision_score(y_test, np.argmax(y_pred_1, axis=-1), average='macro')


# 

# > #### 💬 Creat inference table (test dataset) and make prediction

# In[26]:


test_defog_path = '/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/test/defog/02ab235146.csv'
test_defog = pd.read_csv(test_defog_path)
name = os.path.basename(test_defog_path)
id_value = name.split('.')[0]
test_defog['Id_value'] = id_value
test_defog['Id'] = test_defog['Id_value'].astype(str) + '_' + test_defog['Time'].astype(str)
test_defog = test_defog[['Id','AccV','AccML','AccAP']]
test_defog.set_index('Id',inplace=True)


# In[27]:


# predict event probability
test_defog_pred=clf.predict(test_defog)
test_defog['event'] = np.argmax(test_defog_pred, axis=-1)


# In[28]:


# expand event column it to three columns
test_defog['StartHesitation'] = np.where(test_defog['event']==1, 1, 0)
test_defog['Turn'] = np.where(test_defog['event']==2, 1, 0)
test_defog['Walking'] = np.where(test_defog['event']==3, 1, 0)


# In[29]:


test_defog.head(10)


# > #### 💬 apply the same process for tdcsfog dataset, but I am not going to train another model for tdcsfog. Instead, I will just use the same model trained from defog dataset. I recommend you to develop two different model because the data distribution is quite different.

# In[30]:


test_tdcsfog_path = '/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/test/tdcsfog/003f117e14.csv'
test_tdcsfog = pd.read_csv(test_tdcsfog_path)
name = os.path.basename(test_tdcsfog_path)
id_value = name.split('.')[0]
test_tdcsfog['Id_value'] = id_value
test_tdcsfog['Id'] = test_tdcsfog['Id_value'].astype(str) + '_' + test_tdcsfog['Time'].astype(str)
test_tdcsfog = test_tdcsfog[['Id','AccV','AccML','AccAP']]
test_tdcsfog.set_index('Id',inplace=True)


# In[31]:


test_tdcsfog_pred=clf.predict(test_tdcsfog)
test_tdcsfog['event'] = np.argmax(test_tdcsfog_pred, axis=-1)


# In[32]:


test_tdcsfog['StartHesitation'] = np.where(test_tdcsfog['event']==1, 1, 0)
test_tdcsfog['Turn'] = np.where(test_tdcsfog['event']==2, 1, 0)
test_tdcsfog['Walking'] = np.where(test_tdcsfog['event']==3, 1, 0)
test_tdcsfog.reset_index('Id', inplace=True)


# In[33]:


test_tdcsfog.head(10)


# In[34]:


submit = pd.concat([test_tdcsfog,test_defog])
submit = submit[['Id', 'StartHesitation', 'Turn','Walking']]


# In[35]:


submit.head(10)


# > #### 💬 Let's compare it with sample submission data.

# In[36]:


sample = pd.read_csv('/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/sample_submission.csv')


# In[37]:


sample.head(10)


# in progress....

# <div style="border-radius:10px; border:#DEB887 solid; padding: 15px; background-color: #FFFAF0; font-size:100%; text-align:left">
# 
# <h3 align="left"><font color='#DEB887'>💡 Notes:</font></h3>
# 
# * Again, this notebook is created just for illustrate a simple example.
#     
# * Hope it helps, and keep it up for competition!!
# 
