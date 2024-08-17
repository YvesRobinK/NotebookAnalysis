#!/usr/bin/env python
# coding: utf-8

# 
# <div style="background-color:rgba(0, 120, 255, 0.6);border-radius:5px;display:fill">
#     <h1><center>Tabular Playground Series - April 2022</center></h1>
# </div>
# We are given 12 sensor data and based on the sensor data we need to do binary classification. Time series classification is very well known concept in  industry applications. For example, in a typical stress test patient is asked to run on the treadmill with various sensors attached to him as shown below. This sensor information is used in Medical diagonasis. Various efforts are made by several other kagglers to solve this issue. 
# 
# <img src="https://interpretersontherun.files.wordpress.com/2018/04/homer-simpson-running-gif-downsized.gif?w=500&h=351&crop=1&zoom=2" width="750" align="center">>
# 
# 
# 
# <div class="alert alert-block alert-info"> 📌 
#     <b>In this public notebook, I will showcase how some ofthe tricks which might be helpful for this and future competitions </div>
# 
# 
# 
# 
# 
# ## <span style="color:crimson;"> DONT FORGET TO UPVOTE IF YOU FIND IT USEFUL.......!!!!!! </span>

# ## <font color="#blue">The structure of notebook.</font>
# <a id="1"></a><h2></h2>
# ### <a href='#1'>1. Introduction </a><br>
# ### <a href='#2'>2. Importing Libraries and preprocessing</a><br>
# ### <a href='#3'>3. Trick-1: Converting tabular data to image data</a><br>
# ### <a href='#4'>4. Trick-2: Train and tune your model with one line code, Grand Master way</a><br>
# ### <a href='#5'>5. Trick-3: QC your model with one line code</a><br>
# ### <a href='#6'>6. Trick-4: Speedup your model with Intex</a><br>
# ### <a href='#7'>7. Trick-5: Faster dataframe processing using moodle</a><br>
# ### <a href='#8'>8. Trick-6: Fast aggregation of large data using Python datatable</a><br>
# ### <a href='#9'>9. Trick-7: Get quick intuition with Lazypredict</a><br>
# ### <a href='#10'>10. Trick-8: Automatic EDA with Dataprep </a><br>
# ### <a href='#11'>11. References</a><br>

# <a id="2"></a><h2></h2>
# <div style="background-color:rgba(255, 69, 0, 0.5);border-radius:5px;display:fill">
#     <h1><center>Importing Libraries and preprocessing</center></h1>
# </div>

# In[1]:


get_ipython().system('pip install -Uqq fastbook')
import fastbook
fastbook.setup_book()
from fastbook import *
from fastai.vision.all import *


# In[2]:


## Basic packages
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import re
sns.set_style('whitegrid')
from sklearn.preprocessing import MinMaxScaler
plt.rc('image', cmap='Greys')
import warnings
warnings.filterwarnings('ignore')
pd.set_option("display.precision", 8)


# In[3]:


# read sensor information with more than 600 features
# ref: https://www.kaggle.com/code/lucasmorin/feature-engineering-aggregation-functions
train = pd.read_parquet('../input/fedata-tpsapril/train.parquet')
train.head(2)


# In[4]:


# lets investigate the train data
train.reset_index(inplace=True)
train.describe()


# In[5]:


# is there any missing data
train.isnull().sum()


# In[6]:


# merging the train data with the target information
labels = pd.read_csv("../input/tabular-playground-series-apr-2022/train_labels.csv")
train_df_lable = pd.merge(train, labels, on="sequence")
train_df_lable.tail(2)


# In[7]:


# other kagglers suggested, groupkfold based on the subject is good idea. Lets include that
#including subject column to the dataframe
train_df = pd.read_csv('../input/tabular-playground-series-apr-2022/train.csv')
train_df_subject=train_df[['sequence','subject']].copy()
train_df_subject.drop_duplicates(inplace=True)
train_df_lable=pd.merge(train_df_lable, train_df_subject, on="sequence")
train_df_lable.tail(2)


# In[8]:


#loading test data
# ref: https://www.kaggle.com/code/lucasmorin/feature-engineering-aggregation-functions
test = pd.read_parquet('../input/fedata-tpsapril/test.parquet')
test.reset_index(inplace=True)
test.head(2)


# In[9]:


#adding subject column to the test dataframe
test_df = pd.read_csv('../input/tabular-playground-series-apr-2022/test.csv')
test_df_subject=test_df[['sequence','subject']].copy()
test_df_subject.drop_duplicates(inplace=True)
test_df2=pd.merge(test, test_df_subject, on="sequence")
# test_with_lable=pd.merge(test_with_lable, test_df_subject, on="sequence")
test_df2.shape


# <a id="3"></a><h2></h2>
# <div style="background-color:rgba(255, 69, 0, 0.5);border-radius:5px;display:fill">
#     <h1><center>Converting tabular data to image data</center></h1>
# </div>
# <br>
# @ambrosm was able to minimize the number of feature to less than fifty and able to reach 94% accuracy. Some other kagglers used around 600 features to reach 97.7% accuracy. I also want to take advange of well established image classfication techniques. Hence, I was motivated to find a way to handle handle large number of features and convert them to images.
# The stragegy here is to convert the each row into different image hence typical image classfication techniques can be used. There are total 692 columns in train dataset. We will drop one column so we can convert each row into an image of size 23x30 with sequence as ID of that image.
# 

# In[10]:


train_df_lable.set_index("sequence", inplace = True)
train_df_lable.shape


# In[11]:


# lets drop one column to make it 690 column+ target column
train_df2=train_df_lable.drop(columns=['sensor_00_mean'])
# fill in the Nan values
train_df2=train_df2.interpolate()
train_df2.isnull().sum()


# In[12]:


# The 690 columns are converted into image of size 23 x 30
img_rows = 23
img_cols = 30
train_df_nolable=train_df2.drop(columns=['state'])


# ### There are total 692 columns. Similar to training we will drop one column so we can convert each row into an image of size 23x30 with sequence as ID of that image.

# In[13]:


test_with_nolable=test_df2.drop(columns=['sensor_00_mean'])
test_with_nolable.set_index("sequence", inplace = True)
test_with_nolable.head(2)


# In[14]:


test_with_nolable.shape, train_df_nolable.shape


# ### Some preprocessing before converting to image

# In[15]:


#scaling the train data
cols = train_df_nolable.columns 
train_scaled=pd.DataFrame(columns=cols)

for i in range(len(cols)):
    df = train_df_nolable[cols[i]]
    df_scaled = (df-df.min())/(df.max()-df.min()+1E-8)*(256**3)
    train_scaled[cols[i]] = df_scaled


# In[16]:


#scaling the test data also
cols = test_with_nolable.columns 
test_scaled=pd.DataFrame(columns=cols)

for i in range(len(cols)):
    df = test_with_nolable[cols[i]]
    df_scaled = (df-df.min())/(df.max()-df.min()+1E-8)
    test_scaled[cols[i]] = df_scaled


# 
# The RGB color model is an additive color model in which the red, green, and blue primary colors of light are added together in various ways to reproduce a broad array of colors. The name of the model comes from the initials of the three additive primary colors, red, green, and blue.
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/8/89/RGB_colors.gif" width="750" align="center">>

# In[17]:


# prepare a function to visualize the images
def plot_sensor(images, labels, indexes):
    num_row = 5
    num_col = 5
    fig, axes = plt.subplots(num_row, num_col, constrained_layout=True,  sharex=True, sharey=True, figsize=(3*num_col,2*num_row))
    for i in range(len(images)):
        ax = axes[i//num_col, i%num_col]
        image = images[i].reshape(img_rows, img_cols, 1)
        ax.imshow(image, cmap='Spectral')
        ax.set_title(f'{labels[i]}\n{indexes[i]}')
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.5)
    plt.show()


# In[18]:


target = train_df2.state
num_classes = target.nunique()
FEATURES = [col for col in train_df2.columns if col not in ['state']]


# In[19]:


# Plotting first 25 sequences
images = train_df2[:25]  
labels = images.state.values
indexes = images.index.values
images = images[FEATURES].values.reshape(images.shape[0], img_rows, img_cols, 1).astype('float32')
plot_sensor(images, labels, indexes)


# In[20]:


# Converting the 690 columns to 23x30 columns and visualize one of the sample 
train_df2['state'] = train_df2['state'].astype('bool') 
arr = tensor((train_scaled.iloc[1,0:690]).astype(np.float))
Long1 = torch.reshape(arr,(23,30))


# ### identifying the blue colors in RGB color scheme

# In[21]:


Blue_clr = (Long1/65536).int()
def bluepic(img):
  df = pd.DataFrame(img)
  return df.style.set_properties(**{'font-size':'6pt'}).background_gradient('Blues')
bluepic(Blue_clr)


# In[22]:


Green_clr = ((Long1 - Blue_clr * 65536)/256).int()
def greenpic(img):
  df = pd.DataFrame(img)
  return df.style.set_properties(**{'font-size':'6pt'}).background_gradient('Greens')
greenpic(Green_clr)


# In[23]:


#R = LONG - B * 65536 - G * 256
Red_clr = (Long1 - Blue_clr *65536 - Green_clr * 256).int()
def redpic(img):
  df = pd.DataFrame(img)
  return df.style.set_properties(**{'font-size':'6pt'}).background_gradient('Reds')
redpic(Red_clr)


# ### Lets visualize the whole RGB image

# In[24]:


def bigpic(img):
  df = pd.DataFrame(img)
  return df.style.set_properties(**{'font-size':'6pt'}).background_gradient('gist_rainbow')
bigpic(Long1.int())


# ### bit of post processing

# In[25]:


pic = torch.stack((Red_clr.flatten(),Green_clr.flatten(),Blue_clr.flatten()))
pic.shape


# In[26]:


pic = torch.reshape(pic,(3,23,30))
pic = torch.permute(pic, (1, 2, 0))
img = Image.fromarray(pic.numpy(), 'RGB')
show_image(img)


# ### Finally we managed to convert a row into an image with RGB color format. Lets visualize some more samples

# In[27]:


target_values = train_df2['state'].unique()
samples_df = pd.DataFrame(columns=train_df2.columns)

for i in range(len(target_values)):
    df_filter = train_df2[train_df2['state']==target_values[i]][0:10]
    samples_df = samples_df.append(df_filter)    
samples_df.shape


# In[28]:


def create_pics(train_scaled):
    
  pic_rgb=[]
  for i in range(len(train_scaled)):
    #cast into float tensor
    arr = tensor((train_scaled.iloc[i,0:690]).astype(np.float))

    #reshape into a picture 23x30
    Long1 = torch.reshape(arr,(23,30))

    #compute RGB channells
    Blue1 = (Long1/65536).int()
    Green1 = ((Long1 - Blue1 * 65536)/256).int()
    Red1 = (Long1 - Blue1 *65536 - Green1 * 256).int()

    #create yeansor image
    pic = torch.stack((Red1.flatten(),Green1.flatten(),Blue1.flatten()))
    pic = torch.reshape(pic,(3,23,30))
    pic = torch.permute(pic, (1, 2, 0))

    pic_rgb.append(pic)

  pic_rgb_t = torch.stack(pic_rgb)

  return pic_rgb_t


# In[29]:


sample_pics = create_pics(samples_df)
fig, ax = plt.subplots(2,2, figsize=(20, 20))

for c in range(2):
    for r in range(2):
        image = sample_pics[c+r]
        ax[r, c].imshow(image)
        ax[r, c].set_title(target_values[c])
        ax[r, c].axis("off")


# ### A new datset can be creted which can act as an input to the any image classfication algorithums. Uncomment this section if you want to save. I have commented to save the memory

# In[30]:


# new_pics_train = create_pics(train_scaled)
# #new_pics_train.shape
# train_y_labels = train_df2['state']
# new_pics_test = create_pics(test_scaled)
# torch.save(new_pics_train,'./train_Images')
# torch.save(new_pics_test,'./test_Images')


# In[31]:


from IPython import get_ipython
get_ipython().magic('reset -sf') 


# ### <a href='#1'>Back to top </a><br>

# <a id="4"></a><h2></h2>
# <div style="background-color:rgba(255, 69, 0, 0.5);border-radius:5px;display:fill">
#     <h1><center>Trick-2: Train and tune your model with one line code</center></h1>
# </div>
# This library is developed by <b> Grand Master Absheik Thakur </b>, which he used in several competitions. See the reference list for full details <b>

# In[32]:


# preprocessing
get_ipython().system('pip install autoxgb')


# In[33]:


get_ipython().system('autoxgb train  --train_filename ../input/tabular-playground-series-apr-2022/train.csv  --test_filename ../input/tabular-playground-series-apr-2022/test.csv  --idx Id  --task regression  --targets state  --time_limit 3600  --output petf  --use_gpu')


# <b> Points to note while using this trick: <br>
# 1) make sure all the feature engineering is completed for train and test dataset and store them in respective csv files. For illustration purpose I have ignored this. If you are planning to use this, please include all the features<br>
# 2) The predictions will be saved in the working directory as test_predictions.csv. <br>

# ### <a href='#1'>Back to top </a><br>

# <a id="5"></a><h2></h2>
# <div style="background-color:rgba(255, 69, 0, 0.5);border-radius:5px;display:fill">
#     <h1><center>Trick-3: QC your model with one line code </center></h1>
# </div>

# Here we will utilize DeepChecks to validate your machine learning models and data, such as verifying your data’s integrity, inspecting its distributions, validating data splits, evaluating the model. Interestingly it will take only seven lines of code to do that.<br>
# <br>
# <br>
# <img src="https://docs.deepchecks.com/stable/_images/pipeline_when_to_validate.svg" width="750" align="center">>
# <br>
# Here I will use one of the public notebook and QC the model as an example. All the credits for the model goes to the original author. Consider upvoting them.
# https://www.kaggle.com/code/cv13j0/tps-apr-2022-xgboost-model

# In[34]:


import torch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import re
sns.set_style('whitegrid')
from sklearn.preprocessing import MinMaxScaler
plt.rc('image', cmap='Greys')
import warnings

# Notebook Configuration...

# Amount of data we want to load into the Model...
DATA_ROWS = None
# Dataframe, the amount of rows and cols to visualize...
NROWS = 100
NCOLS = 15
# Main data location path...
BASE_PATH = '...'
pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_columns', NCOLS) 
pd.set_option('display.max_rows', NROWS)
# Load the CSVs into a pandas dataframe for future data manipulation.
trn_data = pd.read_csv('/kaggle/input/tabular-playground-series-apr-2022/train.csv')
trn_label_data = pd.read_csv('/kaggle/input/tabular-playground-series-apr-2022/train_labels.csv')
tst_data = pd.read_csv('/kaggle/input/tabular-playground-series-apr-2022/test.csv')

sub = pd.read_csv('/kaggle/input/tabular-playground-series-apr-2022/sample_submission.csv')

trn_summary = trn_data[['sequence', 'subject', 'step']].groupby(['sequence', 'subject']).count().reset_index()
trn_summary[trn_summary['subject'] == 66].shape
summary_by_subject = trn_summary[['sequence', 'subject']].groupby(['subject']).count().reset_index()
trn_unique_subjects = set(list(trn_data['subject'].unique()))
tst_unique_subjects = set(list(tst_data['subject'].unique()))
overlap_subjets = trn_unique_subjects.intersection(tst_unique_subjects)
print('Repeated Subjects in Test Dataset:', len(overlap_subjets))
from scipy.stats import kurtosis
def kurtosis_func(series):
    '''
    Describe something...
    '''
    return kurtosis(series)

def q01(series):
    return np.quantile(series, 0.01)

def q05(series):
    return np.quantile(series, 0.05)

def q95(series):
    return np.quantile(series, 0.95)

def q99(series):
    return np.quantile(series, 0.99)

def aggregated_features(df, aggregation_cols = ['sequence'], prefix = ''):
    agg_strategy = {'sensor_00': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median', 'skew', kurtosis_func, q01, q05, q95, q99],
                    'sensor_01': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median', 'skew', kurtosis_func, q01, q05, q95, q99],
                    'sensor_02': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median', 'skew', kurtosis_func, q01, q05, q95, q99],
                    'sensor_03': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median', 'skew', kurtosis_func, q01, q05, q95, q99],
                    'sensor_04': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median', 'skew', kurtosis_func, q01, q05, q95, q99],
                    'sensor_05': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median', 'skew', kurtosis_func, q01, q05, q95, q99],
                    'sensor_06': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median', 'skew', kurtosis_func, q01, q05, q95, q99],
                    'sensor_07': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median', 'skew', kurtosis_func, q01, q05, q95, q99],
                    'sensor_08': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median', 'skew', kurtosis_func, q01, q05, q95, q99],
                    'sensor_09': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median', 'skew', kurtosis_func, q01, q05, q95, q99],
                    'sensor_10': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median', 'skew', kurtosis_func, q01, q05, q95, q99],
                    'sensor_11': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median', 'skew', kurtosis_func, q01, q05, q95, q99],
                    'sensor_12': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median', 'skew', kurtosis_func, q01, q05, q95, q99],
                   }
    group = df.groupby(aggregation_cols).aggregate(agg_strategy)
    group.columns = ['_'.join(col).strip() for col in group.columns]
    group.columns = [str(prefix) + str(col) for col in group.columns]
    group.reset_index(inplace = True)
    
    temp = (df.groupby(aggregation_cols).size().reset_index(name = str(prefix) + 'size'))
    group = pd.merge(temp, group, how = 'left', on = aggregation_cols,)
    return group
trn_merge_data = aggregated_features(trn_data, aggregation_cols = ['sequence', 'subject'])
tst_merge_data = aggregated_features(tst_data, aggregation_cols = ['sequence', 'subject'])
trn_subjects_merge_data = aggregated_features(trn_data, aggregation_cols = ['subject'], prefix = 'subject_')
tst_subjects_merge_data = aggregated_features(tst_data, aggregation_cols = ['subject'], prefix = 'subject_')
trn_data['sensor_00_lag_01'] = trn_data['sensor_00'].shift(1)
trn_data['sensor_00_lag_10'] = trn_data['sensor_00'].shift(10)
trn_merge_data = trn_merge_data.merge(trn_label_data, how = 'left', on = 'sequence')
trn_merge_data = trn_merge_data.merge(trn_subjects_merge_data, how = 'left', on = 'subject')
tst_merge_data = tst_merge_data.merge(tst_subjects_merge_data, how = 'left', on = 'subject')
ignore = ['sequence', 'state', 'subject']
features = [feat for feat in trn_merge_data.columns if feat not in ignore]
target_feature = 'state'
from sklearn.model_selection import train_test_split
test_size_pct = 0.10
X_train, X_valid, y_train, y_valid = train_test_split(trn_merge_data[features], trn_merge_data[target_feature], test_size = test_size_pct, random_state = 42)


# In[35]:


from xgboost  import XGBClassifier
params = {'n_estimators': 40,  # I changed it from 4096 to 40 
          'max_depth': 7,
          'learning_rate': 0.15,
          'subsample': 0.95,
          'colsample_bytree': 0.60,
          'reg_lambda': 1.50,
          'reg_alpha': 6.10,
          'gamma': 1.40,
          'random_state': 69,
          'objective': 'binary:logistic',
         }
xgb = XGBClassifier(**params)
xgb.fit(X_train, y_train, eval_set = [(X_valid, y_valid)], eval_metric = ['auc'], early_stopping_rounds = 128, verbose = 50)
from sklearn.metrics import roc_auc_score
preds = xgb.predict_proba(X_valid)[:, 1]
score = roc_auc_score(y_valid, preds)
from sklearn.metrics import roc_auc_score
preds = xgb.predict_proba(tst_merge_data[features])[:, 1]
# end of public notebook


# In[36]:


# Preprocessing to prepare train and test datasets
get_ipython().system('pip install deepchecks --user')
from deepchecks.tabular.suites import full_suite
from deepchecks.tabular.datasets.classification import iris
from deepchecks.tabular import Dataset

features.append("sequence")
train=trn_merge_data[features]
t_lbls=trn_merge_data[['sequence','state']]
train_with_lable=pd.merge(train, t_lbls, on="sequence")
label_col='state'
train_dataset = Dataset(train_with_lable, label=label_col, cat_features=[])
test_with_lable = tst_merge_data[features]
sub['state'] = preds
test_with_lable=pd.merge(test_with_lable, sub, on="sequence")
test_dataset = Dataset(test_with_lable, label=label_col, cat_features=[])


# ## Now the QC the model with two lines

# In[37]:


suite = full_suite()
suite.run(train_dataset=train_dataset, test_dataset=test_dataset, model=xgb)


# In[38]:


from IPython import get_ipython
get_ipython().magic('reset -sf') 


# ### <a href='#1'>Back to top </a><br>

# <a id="6"></a><h2></h2>
# <div style="background-color:rgba(255, 69, 0, 0.5);border-radius:5px;display:fill">
#     <h1><center>Trick-4: Speedup your model with scikit-learn-intelex </center></h1>
# </div>

# This is an nice trick to spped-up your model without changing your Sklearn based model. All we need to do is add two lines codes to import the library and patch it. I would like to thank **Devlikamov Vlad** for introducing this library. See an example <br>
# 
# <img src="https://miro.medium.com/max/1400/1*C-PtCPom21g8zDIiDZsnNQ.png" width="750" align="center">>
# 
# <br> 
# <b> See the references for full informaiton on this. Dont forget to include the following two lines in your code. <br>

# In[ ]:


from sklearnex import patch_sklearn
patch_sklearn()


# ### <a href='#1'>Back to top </a><br>

# <a id="7"></a><h2></h2>
# <div style="background-color:rgba(255, 69, 0, 0.5);border-radius:5px;display:fill">
#     <h1><center>Trick-5: Faster dataframe processing using modin </center></h1>
# </div>
# <br>
# I would like to thank Grand Master @cpmpml for introducing this package to me. Modin accelerates Pandas queries by 4x on an 8-core machine, only requiring users to change a single line of code in their notebooks. The system has been designed for existing Pandas users who would like their programs to run faster and scale better without significant code changes. <br>
# 
# <img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F2750279%2Ff7ff51190490b2f662f160da6ac8b499%2F0.png?generation=1568037893769516&alt=media" width="750" align="center">>
# 
# <br>
# <b>
# Just include the following three lines to imporve your speed.

# In[ ]:


get_ipython().system('pip install modin')
import modin.pandas as pd


# ### <a href='#1'>Back to top </a><br>

# <a id="8"></a><h2></h2>
# <div style="background-color:rgba(255, 69, 0, 0.5);border-radius:5px;display:fill">
#     <h1><center>Trick-6: Fast aggregation of large data with Python datatable </center></h1>
# </div>
# <br>
# As dataset sizes are getting bigger, people are paying more attention to out-of-memory, multi-threaded data preprocessing tools to escape the performance limitations of Pandas. I would like to thank Grand Master @sudalairajkumar for introducing this package to me. This package is developed by H2O.AI team. This package might not show great improvement in this competitation but its perform extremly well on large datasets. Its the best package for handling any large tabular datasets. It is can be used for fast aggregation of large datasets, low latency add/update/remove of columns, quicker ordered joins, and a fast file reader.
# <br>
# <img src="https://miro.medium.com/max/446/0*w7dsjAY9CKNY7owL.png" width="325" align="center">>
# 
# 
# Grand Master @sudalairajkumar has done extensive analysis for this package. It can be found below link <br>
# https://www.kaggle.com/code/sudalairajkumar/getting-started-with-python-datatable
# <br>
# This package is not native to Kaggle and can be accessed using following code
# 

# In[ ]:


get_ipython().system('pip install https://s3.amazonaws.com/h2o-release/datatable/stable/datatable-0.8.0/datatable-0.8.0-cp36-cp36m-linux_x86_64.whl')


# ### <a href='#1'>Back to top </a><br>

# <a id="9"></a><h2></h2>
# <div style="background-color:rgba(255, 69, 0, 0.5);border-radius:5px;display:fill">
#     <h1><center>Trick-7: Get quick intuition with Lazypredict </center></h1>
# </div>
# <br>
# I would like to thank @BEXGBoost for introducing this library to me. Lazypredict can be used to train almost all Sklearn models plus XGBoost and LightGBM in a single line of code. It only has two estimators - one for regression and one for classification. Fitting either one on a dataset with a given target will evaluate more than 30 base models and generate a report with their rankings on several popular metrics.
# <br>
# <br>
# <img src="https://media-exp1.licdn.com/dms/image/C5612AQHZTuPnWQT6KQ/article-cover_image-shrink_600_2000/0/1520128748970?e=1654128000&v=beta&t=AIT6ChKW_1vVK4ueFhQn8q5T3Um6huaN0wZD6ouZDDI" width="325" align="center">>
# 
# 
# <br>
# See an example below. There are some other libraries such as Pycaret, Teapot which can do the similar tasks. Its worth checking them out.

# In[ ]:


# Load data and split
X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit LazyRegressor
reg = LazyRegressor(ignore_warnings=True, random_state=1121218, verbose=False)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)  # pass all sets


# ### <a href='#1'>Back to top </a><br>

# <a id="10"></a><h2></h2>
# <div style="background-color:rgba(255, 69, 0, 0.5);border-radius:5px;display:fill">
#     <h1><center>Trick-8: Automatic EDA with Dataprep  </center></h1>
# </div>
# <br>
# Dataprep is one of the Automatic EDA library to get indication of the dataframe. Its one of the easy to use libary to get deep insight of the data.
# <br>
# <br>
# <img src="https://editor.analyticsvidhya.com/uploads/8465968747470733a2f2f6769746875622e636f6d2f7366752d64622f64617461707265702f7261772f646576656c6f702f6173736574732f6c6f676f2e706e67.png" width="325" align="center">>
# 
# 
# Once the dataframe is loaded with just one line code, we can get indepth analysis of the data. Its worth checking out.

# In[ ]:


get_ipython().system('pip install dataprep')
# load the data
plot(df)


# ## <span style="color:crimson;"> DONT FORGET TO UPVOTE IF YOU FIND IT USEFUL.......!!!!!! </span>

# <a id="11"></a><h2></h2>
# <div style="background-color:rgba(255, 69, 0, 0.5);border-radius:5px;display:fill">
#     <h1><center>References</center></h1>
# </div>

# 1. https://www.kaggle.com/code/remekkinas/bacteria-image-conv2d-cv-grad-cam <br>
# 2. https://www.kaggle.com/code/austinpowers/turn-it-into-a-toon-tps-feb-22 <br>
# 3. https://www.kaggle.com/code/lucasmorin/feature-engineering-aggregation-functions <br>
# 4. https://www.kaggle.com/code/abhishek/autoxgb-petfinder <br>
# 5. https://www.kaggle.com/code/abhishek/autoxgb-nov-2021-tps <br>
# 6. https://www.kaggle.com/code/lordozvlad/let-s-speed-up-your-kernels-using-sklearnex/notebook <br>
# 7. https://www.kaggle.com/discussions/product-feedback/108155 <br>
# 8. https://medium.com/intel-analytics-software/save-time-and-money-with-intel-extension-for-scikit-learn-33627425ae4
# 9. https://www.kaggle.com/code/bextuychiev/7-coolest-packages-top-kagglers-are-using/notebook#3.-Lazypredict
# 10. https://www.analyticsvidhya.com/blog/2021/05/dataprep-library-perform-eda-faster/
