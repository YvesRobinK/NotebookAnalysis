#!/usr/bin/env python
# coding: utf-8

# # HI!!!!
# 
# <img src="https://media.giphy.com/media/3ornk57KwDXf81rjWM/giphy.gif" width=50%>
# 
# # Lets Get started

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



# # Reducing Memory of Data

# In[2]:


def reduce_memory_usage(df):
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != 'object':
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
    
    return df


# In[3]:


df=pd.read_csv("../input/tabular-playground-series-dec-2021/train.csv")
reduce_memory_usage(df)
test=pd.read_csv("../input/tabular-playground-series-dec-2021/test.csv")
reduce_memory_usage(test)


# In[4]:


df.head()


# In[5]:


df.describe().T


# # Checking for NULL

# In[6]:


df.isnull().sum()


# In[7]:


df.info()


# In[8]:


for col in df.columns:
    print(f"The total unique values in {col} are {len(df[col].unique())}")


# As **Soil_Type7** and **Soil_Type15** are  having only 1 type of data need to be removed from the data frame

# <img src="https://media.giphy.com/media/nIUkbJV97FzDicnDtQ/giphy.gif">

# # Data Description
# - Elevation - Elevation in meters
# - Aspect - Aspect in degrees azimuth
# - Slope - Slope in degrees
# - Horizontal_Distance_To_Hydrology - Horz Dist to nearest surface water features
# - Vertical_Distance_To_Hydrology - Vert Dist to nearest surface water features
# - Horizontal_Distance_To_Roadways - Horz Dist to nearest roadway
# - Hillshade_9am (0 to 255 index) - Hillshade index at 9am, summer solstice
# - Hillshade_Noon (0 to 255 index) - Hillshade index at noon, summer solstice
# - Hillshade_3pm (0 to 255 index) - Hillshade index at 3pm, summer solstice
# - Horizontal_Distance_To_Fire_Points - Horz Dist to nearest wildfire ignition points
# - Wilderness_Area (4 binary columns, 0 = absence or 1 = presence) - Wilderness area designation
# - Soil_Type (40 binary columns, 0 = absence or 1 = presence) - Soil Type designation
# - Cover_Type (7 types, integers 1 to 7) - Forest Cover Type designation

# In[9]:


import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('bmh')


# In[10]:


plt.figure(figsize=(15,10))
sns.countplot(df.Cover_Type)
plt.plot()


# In[11]:


df['Cover_Type'].value_counts(ascending=False)


# # Exploratory Data Analysis

# In[12]:


try:
    fig, axes=plt.subplots(2,5,figsize=(30,15))
    j=0
    i=0
    for k in range(1,11):
        if j==5:
            i+=1
            j=0
        sns.kdeplot(df.loc[:,df.columns[k]],ax=axes[i,j])
        plt.gca().set_title(f"{df.columns[k]}")
        plt.tight_layout()
        j+=1
except:
    print("Got all the columns")


# # Getting Outliers
# 
# <img src="https://media.giphy.com/media/Jszr4owyso6Xo9wjWT/giphy.gif">

# In[13]:


def outlier_function(df, col_name):
    first_quartile = np.percentile(np.array(df[col_name].tolist()), 25)
    third_quartile = np.percentile(np.array(df[col_name].tolist()), 75)
    IQR = third_quartile - first_quartile
    
    upper_limit = third_quartile+(3*IQR)
    lower_limit = first_quartile-(3*IQR)
    outlier_count = 0
    
    for value in df[col_name].tolist():
        if (value < lower_limit) | (value > upper_limit):
            outlier_count += 1
    return lower_limit, upper_limit, outlier_count


# In[14]:


for col in  df.columns[:10]:
    out=outlier_function(df,col)
    if out[2]>0:
        print(f"There are {out[2]} outliers in {col}")


# In[15]:


try:
    fig_out, axes_out=plt.subplots(2,5,figsize=(30,15))
    j=0
    i=0
    for k in range(1,11):
        if j==5:
            i+=1
            j=0
        sns.boxplot(y=df.columns[k],x=df.columns[-1],data=df,ax=axes_out[i,j])
        plt.gca().set_title(f"{df.columns[k]}")
        j+=1
except:
    print("Got all the columns")


# # Getting the Correlation HeatMap
# 
# 

# In[16]:


plt.figure(figsize=(15,10))
sns.heatmap(df.corr())
plt.tight_layout()
plt.plot()


# In[17]:


df.drop(["Soil_Type7","Soil_Type15"],axis=1,inplace=True)


# In[18]:


df=pd.concat([df,
              df[df["Cover_Type"]==5],
              df[df["Cover_Type"]==5],
              df[df["Cover_Type"]==5],
              df[df["Cover_Type"]==5],
              df[df["Cover_Type"]==5],
              df[df["Cover_Type"]==5],
              df[df["Cover_Type"]==4],
              df[df["Cover_Type"]==4],
              df[df["Cover_Type"]==4],
              df[df["Cover_Type"]==4],
              df[df["Cover_Type"]==4],
              df[df["Cover_Type"]==4]],ignore_index=True)


# In[19]:


test.drop(["Soil_Type7","Soil_Type15"],axis=1,inplace=True)


# In[20]:


feature_col=df.columns[1:-1]


# In[21]:


df.head()


# In[22]:


test.head()


# In[23]:


df.to_csv("train.csv",index=False)
test.to_csv("test.csv",index=False)


# <img src="https://media.giphy.com/media/3o6ZtkShGlfpCJVCXm/giphy.gif">

# # Installing AutoXGBoost
# 
# ## This uses XGBoost as well as Optuna together and chooses the best for us, so we only have to do the hard work of Feature Engineering and the rest modeleling can be handled by the Library

# In[24]:


get_ipython().system(' pip install scikit-learn --upgrade --force-reinstall')


# In[25]:


get_ipython().system(' pip install --no-deps autoxgb')


# In[26]:


from autoxgb import AutoXGB


# In[27]:


train_filename="./train.csv"
output="submission_4"
test_filename="./test.csv"
idx="Id"
targets=["Cover_Type"]
use_gpu=True
num_folds=5
seed=42
num_trials=5
time_limit=600


# In[28]:


axgb=AutoXGB(
    train_filename=train_filename,
    output=output,
    test_filename=test_filename,
    idx=idx,
    targets=targets,
    use_gpu=use_gpu,
    num_folds=num_folds,
    num_trials=num_trials,
    time_limit=time_limit
)


# In[29]:


axgb.train()


# In[30]:


import os
import numpy as np


# In[31]:


import pandas as pd
test_pred=pd.read_csv(f"./submission_4/test_predictions.csv")


# In[32]:


test_pred.info()


# In[33]:


test_pred.head()


# ## So Here we have the probilites of each prediction need to convert it to submission Format 

# In[34]:


test_pred.shape


# In[35]:


i=0
li=[]
try: 
    while True:
        li.append(np.argmax(test_pred.loc[i,test_pred.columns[1:]])+1)
        
        i+=1
except:
    if len(li)==test_pred.shape[0]:
        print("Got all prediction 😁")
    else:
        print("Error Occured💀")


# In[36]:


submit=pd.concat([pd.DataFrame(test_pred["Id"]),pd.DataFrame(li)],axis=1)


# In[37]:


submit.columns=["Id","Cover_Type"]


# In[38]:


submit.head()


# In[39]:


submit.to_csv("submission.csv",index=False)


# # Please Upvote if you like The kernel!! 
# 
# <img src="https://media.giphy.com/media/SfYTJuxdAbsVW/giphy.gif" width=70%>

# In[ ]:




