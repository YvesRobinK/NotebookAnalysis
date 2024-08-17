#!/usr/bin/env python
# coding: utf-8

# # Objective of this notebook
# 
# * I aim from this notebook to provide simple guide on how to extract and engineer features from the dataset
# * Prepared from a Beginner to beginners

# ## Insights
# * [Youtube Prediction Feature Eng](https://www.kaggle.com/satoshiss/youtube-prediction-feature-eng#Popular-Tag?)
# * [PogModel LightGBM + FE + Explanation](https://www.kaggle.com/michau96/pogmodel-lightgbm-fe-explanation)
# 
# 

# ## Import the libraries and Read the data

# In[1]:


# Import liberaries
import pandas as pd
pd.set_option('display.max_column' , 500)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#Read the data 
train  = pd.read_parquet('/kaggle/input/kaggle-pog-series-s01e01/train.parquet')
test = pd.read_parquet('/kaggle/input/kaggle-pog-series-s01e01/test.parquet')
sample_submission = pd.read_csv('/kaggle/input/kaggle-pog-series-s01e01/sample_submission.csv')

#Concat the two dataframes
df= pd.concat([train, test]).reset_index(drop = True)

print ('\nColumns in train set not found in the test set :' ,set(train.columns)-set(test.columns) )
print ('_'*50)
print ("\n")

#Drop Un-needed columns
df.drop(['video_id' , 'channelId' , 'view_count' ,  'dislikes', 'likes',
         'comment_count' , 'thumbnail_link' , 'ratings_disabled' ,
         'comments_disabled' , 'id' , 'isTest']  , 
        axis = 1 , inplace =True)

df.head(2)


# # Data Cleaning
# 
# * Remove outliers
# * Normalize Transformation
# * Fill the Nan's

# In[2]:


#fill nan values in the description column  with str 
df.description.fillna('not provided', inplace = True)



#remove the outliers 
df.duration_seconds = df.duration_seconds.apply(lambda x : np.nan if x> 70000 else x)
#Normalize transformation
df.duration_seconds = np.log1p(df.duration_seconds)



#fill the Nan values using pandas groupby method - will try channel title
maper = df.groupby('channelTitle')['duration_seconds'].mean().to_dict()
new_column = df['channelTitle'].map(maper)
df.duration_seconds = df.duration_seconds.fillna(new_column)
#Some channel titel has duration_seconds with nan values
#will try to fill it using the categoryId
maper = df.groupby('categoryId')['duration_seconds'].mean().to_dict()
new_column = df['categoryId'].map(maper)
df.duration_seconds = df.duration_seconds.fillna(new_column)


# In[3]:


df.info()


# # Feature Engineering 

# ##  (FE-1) Calculate the Time delta duration from publish date to trending date

# In[4]:


#convert trending date to datetime column
df.trending_date = pd.to_datetime(df.trending_date)
#Remove the time zone from the published date
df.publishedAt = df.publishedAt.dt.tz_localize(None)

# create a column with timedelta as total hours, as a float type
df['Delta_in_hours'] = (df.trending_date - df.publishedAt) / pd.Timedelta(hours=1)

#Check the distribution of the new feature
fig , ax = plt.subplots()
df['Delta_in_hours'].hist(ax = ax)
ax.set_title('Distribution of Delta Duarion in Hours from published date to trending date');


# * we can found some negative values which is not logic
# * Also we need to normalize the distribution of the new column

# In[5]:


#Remove the negative values
df['Delta_in_hours'] = df['Delta_in_hours'].apply(lambda x: np.nan if x<=0 else x )

#fill the Nan values with groupby method ( will use the channeltitle first then the categoryId )
#channel title
maper = df.groupby("channelTitle")["Delta_in_hours"].mean().to_dict()
series = df['channelTitle'].map(maper)
df['Delta_in_hours'] = df['Delta_in_hours'].fillna(series)
#categoryId
maper = df.groupby("categoryId")["Delta_in_hours"].mean().to_dict()
series = df['categoryId'].map(maper)
df['Delta_in_hours'] = df['Delta_in_hours'].fillna(series)

#Now let's normalize the distribution
df['log_Delta_in_hours'] = np.log1p(df['Delta_in_hours'])
#Check the distribution one more time
fig , ax = plt.subplots()
df['log_Delta_in_hours'].hist(ax = ax)
ax.set_title('Distribution of log Delta Duarion in Hours ');


# In[6]:


"""

## (2) So what was the rate of views per time  ( view count / detla duration )
#Calculate the average views frequency


df['Views_Per_hours'] = df.view_count / df.Delta_in_hours
df['Views_Per_minutes'] = df.view_count /  df.Delta_in_minutes
df.head(2)

#handle the inf values

#let's handel the Inf values 
df.loc[~np.isfinite(df['Views_Per_hours']), 'Views_Per_hour']= np.nan
df.loc[~np.isfinite(df['Views_Per_minutes']), 'Views_Per_minutes']= np.nan

#fill the nan values with the mode
df['Views_Per_hours'].fillna(df['Views_Per_hours'].mode()[0] , inplace = True)
df['Views_Per_minutes'].fillna(df['Views_Per_minutes'].mode()[0] , inplace = True)


fig , ax = plt.subplots(2,1)
df['Views_Per_hours'].hist(ax = ax[0])
df['Views_Per_minutes'].hist(ax = ax[1])
ax[0].set_title('Distribution of Views_Per_hour')
ax[1].set_title('Distribution of Views_Per_minutes')
plt.tight_layout()


#Normalize the distribution using power transfromation
from sklearn.preprocessing import PowerTransformer
PT = PowerTransformer(method='yeo-johnson', standardize=True, copy=True)

df['PT_Views_Per_hours'] = PT.fit_transform(df['Views_Per_hours'].values.reshape(-1, 1))
df['PT_Views_Per_minutes'] = PT.fit_transform(df['Views_Per_minutes'].values.reshape(-1, 1))

fig , ax = plt.subplots(2,1)
df['PT_Views_Per_hours'].hist(ax = ax[0])
df['PT_Views_Per_hours'].hist(ax = ax[1])
ax[0].set_title('Distribution of PT_Views_Per_hours')
ax[1].set_title('Distribution of PT_Views_Per_hours')
plt.tight_layout()

"""
print ('skip')


# ## (FE-2) Let's extract the quarter , year , month and day from the date time columns ( publishedAt and trending date)

# In[7]:


# let's do that for published date
df['published_year'] = df.publishedAt.dt.year
df['published_month'] = df.publishedAt.dt.month
df['published_day'] = df.publishedAt.dt.dayofweek
df['published_Quarter'] =df.publishedAt.dt.quarter
#and now for trending date
df['trending_date_year'] = df.trending_date.dt.year
df['trending_date_month'] = df.trending_date.dt.month
df['trending_date_day'] = df.trending_date.dt.dayofweek
df['trending_date_Quarter'] = df.trending_date.dt.quarter
df.head(2)


# In[8]:


df.groupby('published_day')['target'].mean()


# we can see the higher target was at day 4 and the lowest was at day 6 <br>
# so we need to encode those number to be in ranked in linear mannar from high to low 

# In[9]:


def encode_the_numbers (column):
    """
    function to encode the pandas column depend on thier average target from low to high
    """
    helper_df = df.groupby(column)['target'].mean().sort_values(ascending = False).reset_index().reset_index()
    maper = helper_df.groupby(column)["index"].mean().to_dict()
    df[column] = df[column].map(maper)


# In[10]:


columns_to_encode = ['published_year' , 'published_month' , 'published_day' ,'published_Quarter' , 
                    'trending_date_year' ,'trending_date_month' , 'trending_date_day' , 'trending_date_Quarter']
#encode the columns
for column in columns_to_encode:
    encode_the_numbers (column)


# In[11]:


#let's check it out
fig , ax = plt.subplots (3,1)
df.groupby('published_day')['target'].mean().plot(kind = 'barh' , ax = ax[0])
df.groupby('trending_date_day')['target'].mean().plot(kind = 'barh' , ax = ax[1])
df.groupby('published_Quarter')['target'].mean().plot(kind = 'barh' , ax = ax[2])
plt.tight_layout()


# ## (FE-3)  Encode the categoryId to be in linear shape
# 
# using the same previous function re-sort the categoryID based on thier average target 

# In[12]:


encode_the_numbers ('categoryId')
fig , ax = plt.subplots ()
df.groupby('categoryId')['target'].mean().plot(kind = 'barh' , ax = ax)


# ## (FE-4) What can we get from the title

# In[13]:


df['number_of_words_in_title'] = df.title.str.count(' ').add(1)
df['number_of_letters_in_title'] = df.title.str.len()
df['title_uppercases'] = df['title'].str.findall(r'[A-Z]').str.len()/df['number_of_letters_in_title']
df['title_lowercases'] = df['title'].str.findall(r'[a-z]').str.len()/df['number_of_letters_in_title']
#thanks to 
#https://www.kaggle.com/michau96/pogmodel-lightgbm-fe-explanation
#He add also df[['title_sentiment_polarity', 'title_sentiment_subjectivity']] but i am not famillier with this liberiry


# ## (FE-5) Break every thing in the descriotion column

# In[14]:


df['number_of_links_in_discribtion'] = df.description.str.count('https://')
df['number_of_hashtags_in_discribtion'] = df.description.str.count('#')
df['number_of_exlimination_in_discribtion'] = df.description.str.count('!')
df['number_of_words_in_discribtion'] = df.description.str.count(' ').add(1)
df['number_of_letters_in_discribtion'] = df.description.str.len()
df['description_uppercases'] = df['title'].str.findall(r'[A-Z]').str.len()/df['number_of_letters_in_discribtion']
df['description_lowercases'] = df['title'].str.findall(r'[a-z]').str.len()/df['number_of_letters_in_discribtion']



# we need to normalize the destribution of the recently generated columns

# In[15]:


from sklearn.preprocessing import PowerTransformer
PT = PowerTransformer()
columns_to_normalize =['number_of_links_in_discribtion' , 'number_of_hashtags_in_discribtion' ,
                      'number_of_exlimination_in_discribtion' , 'number_of_words_in_discribtion' , 
                      'number_of_letters_in_discribtion' ,'description_uppercases' , 
                      'description_lowercases']
#normalize the columns
for column in columns_to_normalize:
    df[column] = PT.fit_transform(df[column].values.reshape(-1, 1))


# ## (FE-6) encode the has_thumbnail column

# In[16]:


#encode
from sklearn.preprocessing import LabelEncoder 
encoder = LabelEncoder()
df['has_thumbnail'] = encoder.fit_transform(df['has_thumbnail'])


# ## (FE-7) What we can do for the channel title
# 
# Strategy:
# not all the channels in the train set exist in the test set
# 
# * we will find the intersection channells
# * sort the intersection channells according to the average target
# * encode the sorted channels as acontinues variabels
# * map every thing to the df

# In[17]:


#find the intersection channels
train_channelTitle_set = set(train.channelTitle.unique().tolist())
test_channelTitle_set = set(test.channelTitle.unique().tolist())
intersection_channel = list(test_channelTitle_set.intersection(train_channelTitle_set))

#loop for every channel
#slice the dataframe of the channel and calculate the average of the target
average_target = []

for channel in intersection_channel:
    train_slicer = train [train.channelTitle == channel]
    average_target.append(train_slicer.target.mean())

    
#create a new dataframe and sort based on the average target
#reindex the dataframe and use the new index as the channel label encoder 
channel_Data_frame = pd.DataFrame({'channel_title':intersection_channel , 
                                  "average_target": average_target , })

channel_Data_frame = channel_Data_frame.sort_values(by = 'average_target' , ascending = False).reset_index(drop = True).reset_index().rename(columns ={'index':'label'})
#Create a dic with the sorted channels and thier index number 
map1 = channel_Data_frame.groupby("channel_title")["label"].mean().to_dict()
#map the new column
df['channel_encodding'] = df['channelTitle'].map(map1)
#fill the remaning channels with -99
df['channel_encodding'].fillna(-99 , inplace = True)


# ## (FE-8) Count the number of tages per video 

# In[18]:


df['number_of_tags'] = df.tags.str.count("|").add(1)


# ## (FE-9)  create a dummy columns for the most frequent tags in the test dataset (1  =  tag exsit , 0  = tag not exist)

# In[19]:


#Find the count of each tag in the test set

from collections import Counter
#construct the counter
tags = Counter()

#loop on each tag in each row and add it to the counter
for row in range(test.shape[0]):
    for tag in test.tags[row].split('|'):
        tags[tag]+=1
        
#create dataframe from the dict and sort it according to the frequency
tags_df = pd.DataFrame.from_dict(tags, orient='index').reset_index()
tags_df = tags_df.rename(columns = {'index': 'Tag' ,0: 'Frequency'})
tags_df = tags_df.sort_values(by = 'Frequency' , ascending = False)
tags_df.head(15)


# In[20]:


#Create a new dummy columns for the top 20 tags found in the test data set
new_dummy_columns = tags_df.head(20).Tag.tolist()

for column in new_dummy_columns:
    #check if the column name exsist in the tag column
    df[column] = df.tags.str.count(column)
    #encode binary 1: exsist , 0:not found
    df[column] = df[column].apply(lambda x :1 if x != 0 else x)


# ## (FE-10) Normalize the Target

# In[21]:


from sklearn.preprocessing import PowerTransformer
PT_target = PowerTransformer()
df['target'] = PT_target.fit_transform(df.target.values.reshape(-1, 1))


# In[22]:


df.target.hist()


# ## (FE-11) Weight the Tags combination with average traget
# startegy:
# * go for the test set and find all tags in test.tags.split
# * for each tag we need to know the crosponding average target
# * retern back to the df and calculate the average target for each tag combination

# In[23]:


#from the perviouse cell we have the tags in the test dataset 
# that will take very long to go throw all the tags
# I will slice the tags for that accours at least 25 times in the test set
tags_in_test = tags_df[tags_df.Frequency>24].Tag.tolist()


import statistics
def find_the_mean_target_of_tag (word):
    """
    a function to get the average target for a specific tag
    input : word (str)
    output : average target (float)
    """
    target_tag = []
    
    for tag , target in zip (train.tags , train.target):
        if word in tag.split('|'):
            target_tag.append(target)
    try:
        #return the average target
        return statistics.mean(target_tag)
    except:
        #if the test tag was not found in the train data set
        return 0


#now lets apply the function on all tags in dataset
tags_targets = {}
for tag in tags_in_test:
    tags_targets[tag] = find_the_mean_target_of_tag(tag)
    


# In[24]:


#Now we will use the tag_target dict to weight every tags combination in the concanated df

#create a new list to add it to the data frame later
wieghted_tags_combination = []
#loop for each row in the dataframe
for tag in df.tags:
    sum_T = 0
    count = 0
    #loop for each word in tags combination
    for word in tag.split('|'):
        if word in tags_targets.keys():
            sum_T += tags_targets[word]
            count +=1
    try:
        average = sum_T/count
    except:
        #in case the word not in the tag
        average = 0
        
    wieghted_tags_combination.append(average)

#add the new column to the df    
df['tags_combination target'] = wieghted_tags_combination


# # Now let's find the corrlation between the numerical columns and our target

# In[25]:


corr_df = df.select_dtypes('number').drop('target', axis=1).corrwith(df.target).sort_values().reset_index().rename(columns = {'index':'feature' ,0:'correlation'})

fig , ax = plt.subplots(figsize  = (5,20))
ax.barh(y =corr_df.feature , width = corr_df.correlation )
ax.set_title('correlation between featuer and target'.title() ,
            fontsize = 16 , fontfamily = 'serif' , fontweight = 'bold')
plt.show();


# # Feature selection

# In[26]:


columns_with_low_correlation = corr_df[(corr_df.correlation >-0.03) & (corr_df.correlation<0.03)].feature.tolist()

df.drop(columns_with_low_correlation  , axis = 1 , inplace = True)


# # Build the Model

# In[27]:


numerical_df = df.select_dtypes('number')
train  = numerical_df[numerical_df.target.notnull()]
test = numerical_df[numerical_df.target.isnull()].drop('target' , axis = 1).values


# In[28]:


X = train.drop('target' , axis = 1).values
y = train.target.values


# In[29]:


from sklearn.preprocessing import StandardScaler
SC = StandardScaler ()
X = SC.fit_transform(X)
test = SC.transform(test)


# In[30]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)


# In[31]:


from sklearn.linear_model import Ridge
Ridge=Ridge()

from sklearn.neighbors import KNeighborsRegressor
knn=KNeighborsRegressor()

from sklearn.linear_model import BayesianRidge
Bayesian = BayesianRidge()

from sklearn.tree import DecisionTreeRegressor
dec_tree = DecisionTreeRegressor()

from sklearn.svm import SVR
SVR=SVR()

from sklearn.ensemble import RandomForestRegressor
RandomForestRegressor = RandomForestRegressor()


#evaluation
from sklearn.metrics import mean_absolute_error
"""


regs = [Ridge ,knn , Bayesian ,dec_tree ,SVR ,RandomForestRegressor]
names = ['Ridge' ,'knn' , 'Bayesian' ,'dec_tree' ,'SVR' ,'RandomForestRegressor']
"""

regs = [RandomForestRegressor]
names = ['RandomForestRegressor']

for reg , name in zip (regs , names):
    
    reg.fit(X_train, y_train)
    predict_train = reg.predict(X_train)
    predict_test = reg.predict(X_test)
        
        
        
    


# In[32]:


predict_train = PT_target.inverse_transform(predict_train.reshape(-1, 1))
predict_test = PT_target.inverse_transform(predict_test.reshape(-1, 1))
y_train = PT_target.inverse_transform(y_train.reshape(-1, 1))
y_test = PT_target.inverse_transform(y_test.reshape(-1, 1))
        
    
print (name)
print ('train MAE : ' , mean_absolute_error(y_train , predict_train))
print ('test MAE: ' , mean_absolute_error(y_test , predict_test))
print ('_________________________')


# ## Submission

# In[33]:


#predict
submision_prediction = RandomForestRegressor.predict(test)
#inverse_transfrom
submision_prediction = PT_target.inverse_transform(submision_prediction.reshape(-1, 1))
#get the id from the test set
test_set = pd.read_parquet('/kaggle/input/kaggle-pog-series-s01e01/test.parquet')

#create a submission df
sub = pd.DataFrame()

sub['id'] = test_set.id.values
sub['target'] = submision_prediction

sub.to_csv('sub_v8.csv' , index=  False)
sub.head()

