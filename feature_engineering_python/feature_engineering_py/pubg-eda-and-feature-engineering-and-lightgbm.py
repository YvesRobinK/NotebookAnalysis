#!/usr/bin/env python
# coding: utf-8

# # Import Library

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
import lightgbm as lgb

warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.display.float_format = '{:,.3f}'.format
sns.set()


# <h1><b> TRAIN DATA</b></h1>

# # Load Dataset

# In[2]:


pubg_data = pd.read_csv("/kaggle/input/pubg-finish-placement-prediction/train_V2.csv")
train = pubg_data


# In[3]:


get_ipython().run_cell_magic('time', '', 'train.info()\n\n# Memory usages in Bytes\nprint("Reduced Memory size: ",train.memory_usage(index=True).sum()/(1024*1024), "MB")\n')


# In[4]:


# Memory usages in Bytes
print(train.memory_usage(index=True).sum()/(1024*1024), "MB")


# # Reducing Dataset Memory

# ### Here we are reducing the memory of the dataset by downcasting the datatypes of the column as small as possible so that there will be less time consumption for applying computational operations on it

# In[5]:


for column_name in train:
    if train[column_name].dtype=='float64':
        train[column_name] = pd.to_numeric(train[column_name], downcast= 'float')
    if train[column_name].dtype=='int64':
        train[column_name] = pd.to_numeric(train[column_name],downcast='integer')


# In[6]:


get_ipython().run_cell_magic('time', '', 'train.info()\n')


# ### Here we can see memory is reduced by significant amount

# In[7]:


# Memory usages in Bytes
print("Reduced Memory size: ",train.memory_usage(index=True).sum()/(1024*1024), "MB")

print("Data Description:")
train.describe().drop('count').T


# ### Number of NULL value in Data

# In[8]:


train.isna().sum()


# #### Here we can see there is only one null value for winPlacePerc feature, means it is an illegal match, hence we can drop that row as 1 row in compare to the size of the dataset won't affect the output!!

# In[9]:


train.dropna(inplace=True)


# #### Dropping the Id column as it will be of no use

# In[10]:


train.drop(['Id'], axis=1, inplace=True)


# In[11]:


labelencoder = LabelEncoder()


# In[12]:


train['matchType'].value_counts()


# In[13]:


## So from the above there are many matchtypes with combination of fpp, tpp , solo, duo ,squad,etc.
## So we are generalizing them into only solo, duo and squad.
## After that applying LabelEncoding to matchType column

train['matchType'] = train['matchType'].apply(lambda x: 'solo' if ('solo' in x) else 'duo' if ('duo' in x) or ('crash' in x) else 'squad')

train['matchType'] = labelencoder.fit_transform(train['matchType'])
train['matchType'].value_counts()


# In[14]:


train1 = train.copy()
# train2 = train.copy()


# ## EDA

# In[15]:


train1.columns


# ### Univariate Analysis

# ### Plotting graph for some of the discrete columns

# In[16]:


# Discrete Columns
dis_cols_1 = ['assists', 'boosts', 'heals', 'DBNOs', 'headshotKills', 'kills']
dis_cols_2 = ['killStreaks', 'revives', 'roadKills', 'teamKills', 'weaponsAcquired','vehicleDestroys']

def discreteGraph(dis_cols):
    fig = plt.figure(figsize = (20, 15))

    index = 1
    for col in dis_cols:
        plt.subplot(3, 2, index)
        sns.countplot(x=col, data=train1)
        index += 1
    plt.tight_layout()
    plt.show()


# In[17]:


discreteGraph(dis_cols_1)


# ### Here from the graph we can see all the above feature have their most of the value lying in the zero region and their higher values graph are somewhat sparse on the basis of count.

# ### Player Types

# In[18]:


players = train1['matchType'].value_counts()
print("Squad Players  : ",players[2])
print("Duo Players  : ",players[0])
print("Solo Players  : ",players[1])
print("Total Players : ",players[0]+players[1]+players[2])


# ### So here we have the players as above distributed and the data consist of total 3112875 players

# ### Kills Analysis

# In[19]:


print("Kills")
print("99% of the players have kills less than or equal to", train1['kills'].quantile(0.99))
print("Whereas maximum kill is ", train1['kills'].max())
print("And the minimum kill is", train1['kills'].min())


# ### Continuous columns

# In[20]:


cont_dist = ['killPoints', 'longestKill', 'maxPlace', 'rankPoints', 'rideDistance', 
             'swimDistance', 'walkDistance', 'winPoints', 'winPlacePerc']


# In[21]:


## Correlation matrix
plt.subplots(figsize=(25, 20))
sns.heatmap(train1.corr(), annot=True)


# - We thought of reducing the dataset size by grouping the rows if they are a part of team, means a row will be a team now or an individual if its solo.
# - So we have groupby the dataset by groupId and applied the aggregrate function to all the features with mean, max, min according to the feature.
# 
# - Like 
#     - Kills : we have taken sum of the kills of team , walkDistance : max,
#     - if the columns values are same for all the team players like
#     - winPlacePerc, matchType, teamKills, etc we can take mean or max as it will be same for all the players.
# 
#     - So basically the feature which desribe any teamwork we will take sum of it ( e.g kils, assists)
#     - If its a scaling feature we are taking mean of it
#     - If the feature describes quality of player in a team we'll take max of it hence his/her team is affected positively

# In[22]:


train2 = train.copy()
train2 = train2.groupby(by=['groupId']).agg({'matchId':'max', 'assists':'sum', 'boosts':'sum','damageDealt':'sum', 'DBNOs':'sum', 
                                            'headshotKills':'sum','heals':'sum', 'killPlace':'mean', 'killPoints':'max', 'kills':'sum',
                                            'killStreaks':'max','longestKill':'mean','matchDuration':'max', 'maxPlace':'mean','numGroups':'mean',
                                            'rankPoints' : 'max', 'matchType':'mean','revives':'sum','rideDistance':'max', 'roadKills':'sum',
                                            'swimDistance':'max','teamKills':'sum', 'vehicleDestroys':'sum', 'walkDistance':'max',
                                            'weaponsAcquired':'sum','winPoints':'max', 'winPlacePerc':'max'})



# In[23]:


print("Memory Before :",(train1.memory_usage(index=True).sum()/(1024 * 1024)).round(2), " MB ")
print("Memory After : ", (train2.memory_usage(index=True).sum()/(1024 * 1024)).round(2), " MB ")


# - Here we have significantly reduced the dataset memory, but is it legit ? lets see some plots and figure out

# In[24]:


fig, axes = plt.subplots(5, 2, figsize=(20, 25))
ind = 0
for i in range(5):
    for j in range(2):
        sns.distplot(train2, ax=axes[i, j], x = train2[cont_dist[ind]], kde = True)
        sns.distplot(train1, ax=axes[i, j], x = train1[cont_dist[ind]], kde = True)
        plt.xlabel(cont_dist[ind])
        ind += 1
        if(ind == len(cont_dist)):
            break
plt.tight_layout()
plt.show()


# - We have plotted density graph for the original dataset and the reduced dataset looks like the both distribution looks similar.
# - Also lets check the correlation of all the features with winPlacePerc before and after

# ### Correlation Matrix

# In[25]:


cor1 = pd.DataFrame()
cor1["Original Dataset"] = train1.corr()['winPlacePerc']
cor1["Reduced Dataset"] = train2.corr()['winPlacePerc']

print(cor1)


# - Hence from the above result there is not much of difference of correlation of feature wih winPlacePerc too
# - So going forward the reduced dataset can be a candidate for the training purpose

# In[26]:


temp1 = train2[train2['winPlacePerc']>0.6].sample(100000)
plt.figure(figsize=(15, 10))
plt.scatter(temp1['walkDistance'], temp1['boosts'], s=(temp1['kills']+5)*100, c=temp1['winPlacePerc'], cmap='Greens', edgecolor='black', linewidth=1, alpha=0.8)
cbar = plt.colorbar()
cbar.set_label("Win Place Perc")
plt.xlabel("Walk Distance")
plt.ylabel("Boosts")


# ### From the above graph we can observe that as boosts consumption increases players chance to win the match increases, also logically a player which has high chance of winning tends to be in fight and needs boost also we can see walkDistance also matters in winnning as it will be high for the player/team who has high chances of winning, because to be in the game players have to be in safe zone for that they need to travel.

# In[27]:


temp1 = train2[train2['winPlacePerc']>0.6].sample(5000)
plt.figure(figsize=(15, 10))
plt.scatter(temp1['heals'], temp1['boosts'], s=temp1['damageDealt'], c=temp1['winPlacePerc'], cmap='Greens', edgecolor='black', linewidth=1, alpha=0.8)
cbar = plt.colorbar()
cbar.set_label("Win Place Perc")
plt.xlabel("Heals")
plt.ylabel("Boosts")


# In[28]:


temp2 = train2[train2['heals'] < train2['heals'].quantile(0.99)]
temp2 = temp2[temp2['boosts'] < temp2['boosts'].quantile(0.99)]

f,ax1 = plt.subplots(figsize =(10,5))
sns.pointplot(x='heals',y='winPlacePerc',data=temp2,color='green')
sns.pointplot(x='boosts',y='winPlacePerc',data=temp2,color='violet')
plt.text(0,0.8,'Heals',color='green',fontsize = 10)
plt.text(0,0.9,'Boosts',color='violet',fontsize = 10)
plt.ylabel('Win Percentage', color='black')
plt.xlabel('Boost and Heals', color='black')
plt.grid()
plt.legend()


# ### From the above graph we can see Boosts and Heals shows positive relation with winPlacePerc, Boosts shows more than Heal. Maybe we can do some stuff with both of these feature later

# In[29]:


temp = train2[train2['kills'] <= train2['kills'].quantile(.9)]
sns.pointplot(x='kills',y='winPlacePerc',data=temp, hue='matchType')

# duo   - 0
# solo  - 1
# squad - 2


# ### From the above graph we can say that as the number of kills increases chances of winning increases but it does not matter much as we go from match type from solo to squad, because in squad we have to play more strategically and focus is not much on kills in squad

# ### While Analyzing the dataset we found some irregularities in the data, so handling those anomalies now

# In[30]:


train4 = train2.copy()


# ### 1) Have done kills but have not travel any distance

# In[31]:


plt.figure(figsize=(12, 5))
sns.countplot(x='kills', data=train4[(train4['walkDistance'] + train4['rideDistance'] + train4['swimDistance']==0) & (train4['kills'] > 0)])


# ### So the above graph is of the players who travel zero distance yet they have killed enemies seems suspicious, hence removing those rows!!

# In[32]:


train4.drop(train4[(train4['walkDistance'] + train4['rideDistance'] + train4['swimDistance']==0) & (train4['kills'] > 0)].index, axis=0, inplace = True)


# ### 2)

# In[33]:


plt.figure(figsize=(12, 5))
sns.countplot(x='kills', data=train4[(train4['longestKill']==0) & (train4['kills'] > 0)])


# ### So here we can see the longest kill is zero yet there are some non-zero kills, hence dropping those rows too!

# In[34]:


index_drop = train4[(train4['longestKill']==0) & (train4['kills'] > 0)].index
train4.drop(index_drop,axis= 0,inplace= True)


# ### 3)

# In[35]:


plt.figure(figsize=(12, 5))
sns.countplot(x='teamKills', data=train4[(train4['weaponsAcquired']==0) & (train4['teamKills']>0) & (train4['rideDistance']==0)])


# ### In pubg, a player can kill his/her team-mate only if he has grenade(weapon) or he/she has drove a vehicle over his/her team-mate. But from the above condition graph there are some players who have killed teamplayer yet they have not acquire any weapon or drove a car/vehicle!!

# In[36]:


index_drop = train4[(train4['weaponsAcquired']==0) & (train4['teamKills']>0) & (train4['rideDistance']==0)].index
train4.drop(index_drop, axis=0, inplace = True)
print(len(index_drop), " rows dropped!!")


# ### 4)

# In[37]:


plt.figure(figsize=(12, 5))
sns.countplot(x='roadKills', data=train4[(train4['roadKills']>0) & (train4['rideDistance']==0)])


# ### Killing players  from the car but have not ride the car=> illegal data

# In[38]:


index_drop = train4[(train4['roadKills']>0) & (train4['rideDistance']==0)].index
print(index_drop.shape)
train4.drop(index_drop, axis=0, inplace = True)


# ### 5) Have not walked but have consumed heals and boost, its not possible

# In[39]:


index_drop = train4[((train4['heals']>0) | (train4['boosts']>0)) & (train4['walkDistance']==0)].index
print(index_drop.shape)
train4.drop(index_drop, axis=0, inplace = True)


# ### Similarly we have observed some more anamolies like as below:

# ### 6) Its not possible to acquire weapon if a player has not walked a distamce

# In[40]:


index_drop = train4[(train4['weaponsAcquired']>0) & (train4['walkDistance']==0)].index
print(index_drop.shape)
train4.drop(index_drop, axis=0, inplace = True) 


# ### 7) If matchType is solo then there cannot be any assists value, because to assist we need teammate which we don't have, here as the number are somewhat high, so instead of dropping the rows, we imputed that feature with 0.

# In[41]:


index_replace = train4[(train4['matchType']==1) & (train4['assists']>0)].index
print(index_replace.shape)
train4.loc[index_replace,'assists'] = 0


# ### 8) A player cannot assist a teammate if the walkDistance is 0

# In[42]:


index_drop = train4[(train4['assists']>0) & (train4['walkDistance']==0)].index
print(index_drop.shape)
train4.drop(index_drop, axis=0, inplace = True)


# ### 9) A player cannot dealt damage if he has not walked a single meter

# In[43]:


index_drop = train4[(train4['damageDealt']>0) & (train4['walkDistance']==0)].index
print(index_drop.shape)
train4.drop(index_drop, axis=0, inplace = True)


# ### Correlation Matrix

# In[44]:


plt.subplots(figsize=(25, 20))
sns.heatmap(train4.corr(), annot=True)


# In[45]:


def getCorrelatedFeatures(corrdata, threshold):
    train_features = []
    train_value = []
    for i,index in enumerate(corrdata.index):
        if abs(corrdata[index])>threshold:
            train_features.append(index)
            train_value.append(corrdata[index])
            
    df = pd.DataFrame(data = train_value, index = train_features,columns = ['CorrValue'] )
    return df,train_features


# In[46]:


threshold = 0.4
corr_value,train_features = getCorrelatedFeatures(train4.corr()['winPlacePerc'],threshold)
print(corr_value)


# In[47]:


train4 = train4.reset_index()


# In[48]:


# train4.to_csv("Train-GroupByGroupId-RemovedRows.csv", index=False)


# In[49]:


for column_name in train4:
    if train4[column_name].dtype=='float64':
        train4[column_name] = pd.to_numeric(train4[column_name], downcast= 'float')
    if train4[column_name].dtype=='int64':
        train4[column_name] = pd.to_numeric(train4[column_name],downcast='integer')


# ### Feature Engineering

# In[50]:


train5 = train4.copy()


# In[51]:


def feature_engineering(train5):
    train5.insert(train5.shape[1]-1, 'killsPerMeter', train5['kills']/train5['walkDistance'])
    train5['killsPerMeter'].fillna(0, inplace=True)
    train5['killsPerMeter'].replace(np.inf, 0, inplace=True)

    train5.insert(train5.shape[1]-1, 'healsPerMeter', train5['heals']/train5['walkDistance'])
    train5['healsPerMeter'].fillna(0, inplace=True)
    train5['healsPerMeter'].replace(np.inf, 0, inplace=True)

    train5.insert(train5.shape[1]-1, 'totalHeals', train5['heals']+train5['boosts'])

    train5.insert(train5.shape[1]-1, 'totalHealsPerMeter', train5['totalHeals']/train5['walkDistance'])
    train5['totalHealsPerMeter'].fillna(0, inplace=True)
    train5['totalHealsPerMeter'].replace(np.inf, 0, inplace=True)

    train5.insert(train5.shape[1]-1, 'totalDistance', train5['walkDistance']+train5['rideDistance']+train5['swimDistance'])
    train5['totalDistance'].fillna(0, inplace=True)
    train5['totalDistance'].replace(np.inf, 0, inplace=True)

    train5.insert(train5.shape[1]-1, 'headshotRate', train5['headshotKills']/train5['kills'])
    train5['headshotRate'].fillna(0, inplace=True)
    train5['headshotRate'].replace(np.inf, 0, inplace=True)

    train5.insert(train5.shape[1]-1, 'assistsAndRevives', train5['assists']+train5['revives'])

    train5.insert(train5.shape[1]-1, 'itemsAcquired', train5['heals']+train5['boosts']+train5['weaponsAcquired'])
    
    return train5


# In[52]:


train5 = feature_engineering(train5)


# In[53]:


train5.corr()['winPlacePerc']


# In[54]:


# train5.to_csv("Train-GroupByGroupId-RemovedRows-FeatureEng.csv", index=False)


# In[55]:


# train5 = pd.read_csv("../input/Train-GroupByGroupId-RemovedRows-FeatureEng/Train-GroupByGroupId-RemovedRows-FeatureEng.csv")


# <h1> <b>Reformatting test data  </b></h1>

# In[56]:


test = pd.read_csv("/kaggle/input/pubg-finish-placement-prediction/test_V2.csv")
for column_name in test:
    if test[column_name].dtype=='float64':
        test[column_name] = pd.to_numeric(test[column_name], downcast= 'float')
    if test[column_name].dtype=='int64':
        test[column_name] = pd.to_numeric(test[column_name],downcast='integer')


# In[57]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
test['matchType'] = test['matchType'].apply(lambda x: 'solo' if ('solo' in x) else 'duo' if ('duo' in x) or ('crash' in x) else 'squad')
test['matchType'] = labelencoder.fit_transform(test['matchType'])
test['matchType'].value_counts()


# In[58]:


mapping = dict(zip(labelencoder.classes_, range(len(labelencoder.classes_))))
mapping


# In[59]:


test1 = test.groupby(by=['groupId']).agg({'matchId':'max', 'assists':'sum', 'boosts':'sum','damageDealt':'sum', 'DBNOs':'sum', 
                                            'headshotKills':'sum','heals':'sum', 'killPlace':'mean', 'killPoints':'max', 'kills':'sum',
                                            'killStreaks':'max','longestKill':'mean','matchDuration':'max', 'maxPlace':'mean','numGroups':'mean',
                                            'rankPoints' : 'max', 'matchType':'mean','revives':'sum','rideDistance':'max', 'roadKills':'sum',
                                            'swimDistance':'max','teamKills':'sum', 'vehicleDestroys':'sum', 'walkDistance':'max',
                                            'weaponsAcquired':'sum','winPoints':'max'})


# In[60]:


test1 = test1.reset_index()


# In[61]:


# test1.to_csv("Test-GroupByGroupId.csv", index=False)
# test1 = pd.read_csv("../input/Test-GroupByGroupId/Test-GroupByGroupId.csv")


# In[62]:


test2 = test1.copy()

def fea_eng_test(test2):
    test2.insert(test2.shape[1], 'killsPerMeter', test2['kills']/test2['walkDistance'])
    test2['killsPerMeter'].fillna(0, inplace=True)
    test2['killsPerMeter'].replace(np.inf, 0, inplace=True)

    test2.insert(test2.shape[1], 'healsPerMeter', test2['heals']/test2['walkDistance'])
    test2['healsPerMeter'].fillna(0, inplace=True)
    test2['healsPerMeter'].replace(np.inf, 0, inplace=True)

    test2.insert(test2.shape[1], 'totalHeals', test2['heals']+test2['boosts'])

    test2.insert(test2.shape[1], 'totalHealsPerMeter', test2['totalHeals']/test2['walkDistance'])
    test2['totalHealsPerMeter'].fillna(0, inplace=True)
    test2['totalHealsPerMeter'].replace(np.inf, 0, inplace=True)

    test2.insert(test2.shape[1], 'totalDistance', test2['walkDistance']+test2['rideDistance']+test2['swimDistance'])
    test2['totalDistance'].fillna(0, inplace=True)
    test2['totalDistance'].replace(np.inf, 0, inplace=True)

    test2.insert(test2.shape[1], 'headshotRate', test2['headshotKills']/test2['kills'])
    test2['headshotRate'].fillna(0, inplace=True)
    test2['headshotRate'].replace(np.inf, 0, inplace=True)

    test2.insert(test2.shape[1], 'assistsAndRevives', test2['assists']+test2['revives'])

    test2.insert(test2.shape[1], 'itemsAcquired', test2['heals']+test2['boosts']+test2['weaponsAcquired'])
    
    return test2


# In[63]:


test2 = fea_eng_test(test2)


# In[64]:


# test2.to_csv("Test-GroupByGroupId-FeatureEng.csv", index=False)

# test2  = pd.read_csv("../input/Test-GroupByGroupId-FeatureEng/Test-GroupByGroupId-FeatureEng.csv")


# # LightGBM Model

# In[65]:


X = train5.drop(['groupId', 'matchId', 'winPlacePerc'], axis=1)
y = train5['winPlacePerc']


# In[66]:


def model_train(model,Xt_train,Xt_test,yt_train, yt_test):
    model.fit(Xt_train,yt_train)
    score= model.score(Xt_train,yt_train)
    y_pred = model.predict(Xt_test)
    mse = mean_squared_error(yt_test, y_pred)
    print("MSE: {0:.6f}".format(mse))
    print("Training Score:{0:.6f}".format(score)) 


# In[67]:


lgbm_for_reg= LGBMRegressor(colsample_bytree=0.8, learning_rate=0.03, max_depth=30,
              min_split_gain=0.00015, n_estimators=250, num_leaves=2200,reg_alpha=0.1, reg_lambda=0.001, subsample=0.8,
              subsample_for_bin=45000, n_jobs =-1, max_bin =700, num_iterations=5100, min_data_in_bin = 12)


# ## LightGBM Train Model

# In[68]:


lgbm_for_reg.fit(X,y,verbose=1700, eval_set=[(X, y)],early_stopping_rounds=10)


# In[69]:


import pickle
pickle.dump(lgbm_for_reg,open("lgbm_for_reg.pkl",'wb'))


# ## Importanat Parameter

# In[70]:


fig, ax = plt.subplots(figsize=(12,18))
lgb.plot_importance(lgbm_for_reg, max_num_features=50,ax=ax)


# ## Pridiction of Test dataset

# In[71]:


test2 = test2.drop(['groupId', 'matchId'], axis=1)


# In[72]:


ypred = lgbm_for_reg.predict(test2)


# ## Reformatting of test data

# In[73]:


def submitFormat(test, ypred):
    res = test[['Id', 'groupId']]
    res = res.groupby(by=['groupId']).agg(list)
    res['winPlacePerc'] = ypred.tolist()
    res = res.explode('Id')
    res.reset_index(inplace = True)
    res = res[['Id','winPlacePerc']]
    res['winPlacePerc'].round(decimals = 3)
    return res


# In[74]:


res = submitFormat(test, ypred)


# In[75]:


res["winPlacePerc"] = np.where(res["winPlacePerc"] <0, 0, res["winPlacePerc"])
res["winPlacePerc"] = np.where(res["winPlacePerc"] >1, 1, res["winPlacePerc"])


# In[76]:


res


# ## Generate submission File

# In[77]:


res.to_csv("submission.csv", index=False)


# In[ ]:




