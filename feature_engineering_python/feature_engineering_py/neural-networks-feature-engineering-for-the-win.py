#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import datetime
from kaggle.competitions import nflrush
import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import keras
import tensorflow as tf

sns.set_style('darkgrid')
mpl.rcParams['figure.figsize'] = [15,10]


# In[2]:


env = nflrush.make_env()


# In[3]:


train = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object'})


# # Overall analysis

# In[4]:


train.head()


# - Let's see how PlayId is distribuited

# In[5]:


train['PlayId'].value_counts()


# As expected, we have 22 of each playid since we have 22 players.
# 
# Let's look at our target variable(Yards).

# In[6]:


train['Yards'].describe()


# In[7]:


ax = sns.distplot(train['Yards'])
plt.vlines(train['Yards'].mean(), plt.ylim()[0], plt.ylim()[1], color='r', linestyles='--');
plt.text(train['Yards'].mean()-8, plt.ylim()[1]-0.005, "Mean yards travaled", size=15, color='r')
plt.xlabel("")
plt.title("Yards travaled distribution", size=20);


# ## Feature Engineering

# In[8]:


#from https://www.kaggle.com/prashantkikani/nfl-starter-lgb-feature-engg
train['DefendersInTheBox_vs_Distance'] = train['DefendersInTheBox'] / train['Distance']


# # Categorical features

# In[9]:


cat_features = []
for col in train.columns:
    if train[col].dtype =='object':
        cat_features.append((col, len(train[col].unique())))


# Let's preprocess some of those features.

# ## Possession Team

# In[10]:


train[(train['PossessionTeam']!=train['HomeTeamAbbr']) & (train['PossessionTeam']!=train['VisitorTeamAbbr'])][['PossessionTeam', 'HomeTeamAbbr', 'VisitorTeamAbbr']]


# We have some problem with the enconding of the teams such as BLT and BAL or ARZ and ARI.
# 
# Let's try to fix them manually.

# In[11]:


sorted(train['HomeTeamAbbr'].unique()) == sorted(train['VisitorTeamAbbr'].unique())


# In[12]:


diff_abbr = []
for x,y  in zip(sorted(train['HomeTeamAbbr'].unique()), sorted(train['PossessionTeam'].unique())):
    if x!=y:
        print(x + " " + y)


# Apparently these are the only three problems, let's fix it.

# In[13]:


map_abbr = {'ARI': 'ARZ', 'BAL': 'BLT', 'CLE': 'CLV', 'HOU': 'HST'}
for abb in train['PossessionTeam'].unique():
    map_abbr[abb] = abb


# In[14]:


train['PossessionTeam'] = train['PossessionTeam'].map(map_abbr)
train['HomeTeamAbbr'] = train['HomeTeamAbbr'].map(map_abbr)
train['VisitorTeamAbbr'] = train['VisitorTeamAbbr'].map(map_abbr)


# In[15]:


train['HomePossesion'] = train['PossessionTeam'] == train['HomeTeamAbbr']


# In[16]:


train['Field_eq_Possession'] = train['FieldPosition'] == train['PossessionTeam']


# ## Offense formation

# In[17]:


off_form = train['OffenseFormation'].unique()
train['OffenseFormation'].value_counts()


# Since I don't have any knowledge about formations, I am just goig to one-hot encode this feature

# In[18]:


train = pd.concat([train.drop(['OffenseFormation'], axis=1), pd.get_dummies(train['OffenseFormation'], prefix='Formation')], axis=1)
dummy_col = train.columns


# ## Game Clock

# Game clock is supposed to be a numerical feature.

# In[19]:


train['GameClock'].value_counts()


# Since we already have the quarter feature, we can just divide the Game Clock by 15 minutes so we can get the normalized time left in the quarter.

# In[20]:


def strtoseconds(txt):
    txt = txt.split(':')
    ans = int(txt[0])*60 + int(txt[1]) + int(txt[2])/60
    return ans


# In[21]:


train['GameClock'] = train['GameClock'].apply(strtoseconds)


# In[22]:


sns.distplot(train['GameClock'])


# ## Player height

# In[23]:


train['PlayerHeight']


# We know that 1ft=12in, thus:

# In[24]:


train['PlayerHeight'] = train['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))


# ## Time handoff and snap and Player BirthDate

# In[25]:


train['TimeHandoff']


# In[26]:


train['TimeHandoff'] = train['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
train['TimeSnap'] = train['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))


# In[27]:


train['TimeDelta'] = train.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)


# In[28]:


train['PlayerBirthDate'] = train['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))


# Let's use the time handoff to calculate the players age

# In[29]:


seconds_in_year = 60*60*24*365.25
train['PlayerAge'] = train.apply(lambda row: (row['TimeHandoff']-row['PlayerBirthDate']).total_seconds()/seconds_in_year, axis=1)


# In[30]:


train = train.drop(['TimeHandoff', 'TimeSnap', 'PlayerBirthDate'], axis=1)


# ## Wind Speed and Direction

# In[31]:


train['WindSpeed'].value_counts()


# We can see there are some values that are not standardized(e.g. 12mph), we are going to remove mph from all our values.

# In[32]:


train['WindSpeed'] = train['WindSpeed'].apply(lambda x: x.lower().replace('mph', '').strip() if not pd.isna(x) else x)


# In[33]:


train['WindSpeed'].value_counts()


# In[34]:


#let's replace the ones that has x-y by (x+y)/2
# and also the ones with x gusts up to y
train['WindSpeed'] = train['WindSpeed'].apply(lambda x: (int(x.split('-')[0])+int(x.split('-')[1]))/2 if not pd.isna(x) and '-' in x else x)
train['WindSpeed'] = train['WindSpeed'].apply(lambda x: (int(x.split()[0])+int(x.split()[-1]))/2 if not pd.isna(x) and type(x)!=float and 'gusts up to' in x else x)


# In[35]:


#let's replace the ones that has x-y by (x+y)/2
# and also the ones with x gusts up to y
#train['WindSpeed'] = train['WindSpeed'].apply(lambda x: (int(x.split('-')[0])+int(x.split('-')[1]))/2 if not pd.isna(x) and '-' in x else x)
#train['WindSpeed'] = train['WindSpeed'].apply(lambda x: (int(x.split()[0])+int(x.split()[-1]))/2 if not pd.isna(x) and type(x)!=float and 'gusts up to' in x else x)
def str_to_float(txt):
    try:
        return float(txt)
    except:
        return -1


# In[36]:


train['WindSpeed'] = train['WindSpeed'].apply(str_to_float)


# In[37]:


train['WindDirection'].value_counts()


# The wind direction won't affect our model much because we are analyzing running plays so we are just going to drop it.

# In[38]:


train.drop('WindDirection', axis=1, inplace=True)


# ## PlayDirection

# In[39]:


train['PlayDirection'].value_counts()


# In[40]:


train['PlayDirection'] = train['PlayDirection'].apply(lambda x: x is 'right')


# ## Team

# In[41]:


train['Team'] = train['Team'].apply(lambda x: x.strip()=='home')


# ## Game Weather

# In[42]:


train['GameWeather'].unique()


# We are going to apply the following preprocessing:
#  
# - Lower case
# - N/A Indoor, N/A (Indoors) and Indoor => indoor Let's try to cluster those together.
# - coudy and clouidy => cloudy
# - party => partly
# - sunny and clear => clear and sunny
# - skies and mostly => ""

# In[43]:


train['GameWeather'] = train['GameWeather'].str.lower()
indoor = "indoor"
train['GameWeather'] = train['GameWeather'].apply(lambda x: indoor if not pd.isna(x) and indoor in x else x)
train['GameWeather'] = train['GameWeather'].apply(lambda x: x.replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly') if not pd.isna(x) else x)
train['GameWeather'] = train['GameWeather'].apply(lambda x: x.replace('clear and sunny', 'sunny and clear') if not pd.isna(x) else x)
train['GameWeather'] = train['GameWeather'].apply(lambda x: x.replace('skies', '').replace("mostly", "").strip() if not pd.isna(x) else x)


# In[44]:


train['GameWeather'].unique()


# Let's now look at the most common words we have in the weather description

# In[45]:


from collections import Counter
weather_count = Counter()
for weather in train['GameWeather']:
    if pd.isna(weather):
        continue
    for word in weather.split():
        weather_count[word]+=1
        
weather_count.most_common()[:15]


# To encode our weather we are going to do the following map:
#  
# - climate controlled or indoor => 3, sunny or sun => 2, clear => 1, cloudy => -1, rain => -2, snow => -3, others => 0
# - partly => multiply by 0.5
# 
# I don't have any expercience with american football so I don't know if playing in a climate controlled or indoor stadium is good or not, if someone has a good idea on how to encode this it would be nice to leave it in the comments :)

# In[46]:


def map_weather(txt):
    ans = 1
    if pd.isna(txt):
        return 0
    if 'partly' in txt:
        ans*=0.5
    if 'climate controlled' in txt or 'indoor' in txt:
        return ans*3
    if 'sunny' in txt or 'sun' in txt:
        return ans*2
    if 'clear' in txt:
        return ans
    if 'cloudy' in txt:
        return -ans
    if 'rain' in txt or 'rainy' in txt:
        return -2*ans
    if 'snow' in txt:
        return -3*ans
    return 0


# In[47]:


train['GameWeather'] = train['GameWeather'].apply(map_weather)


# ## NflId NflIdRusher

# In[48]:


train['IsRusher'] = train['NflId'] == train['NflIdRusher']


# In[49]:


train.drop(['NflId', 'NflIdRusher'], axis=1, inplace=True)


# In[50]:


train['Y'].describe()


# In[51]:


#train['TimeDelta'] = train.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)
#train['GameWeather'] = train['GameWeather'].apply(lambda x: indoor if not pd.isna(x) and indoor in x else x)
#train['Delta_X_Y'] = train.apply(lambda row: (row['X'] + row['Y']), axis=1)
#from math import sqrt
#train['Delta_X_Y'] = train['Delta_X_Y'].apply(lambda x: sqrt(x) if x!=0 else 0)


# # Baseline model

# Let's drop the categorical features and run a simple random forest in our model

# In[52]:


train = train.sort_values(by=['PlayId', 'Team', 'IsRusher']).reset_index()


# In[53]:


train.drop(['GameId', 'PlayId', 'index', 'IsRusher', 'Team'], axis=1, inplace=True)


# In[54]:


cat_features = []
for col in train.columns:
    if train[col].dtype =='object':
        cat_features.append(col)
        
train = train.drop(cat_features, axis=1)


# We are now going to make one big row for each play where the rusher is the last one

# In[55]:


train.fillna(-999, inplace=True)


# In[56]:


players_col = []
for col in train.columns:
    if train[col][:22].std()!=0:
        players_col.append(col)


# In[57]:


X_train = np.array(train[players_col]).reshape(-1, 11*22)


# In[58]:


play_col = train.drop(players_col+['Yards'], axis=1).columns
X_play_col = np.zeros(shape=(X_train.shape[0], len(play_col)))
for i, col in enumerate(play_col):
    X_play_col[:, i] = train[col][::22]


# In[59]:


X_train = np.concatenate([X_train, X_play_col], axis=1)
y_train = np.zeros(shape=(X_train.shape[0], 199))
for i,yard in enumerate(train['Yards'][::22]):
    y_train[i, yard+99:] = np.ones(shape=(1, 100-yard))


# In[60]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)


# In[61]:


model = keras.models.Sequential([
    keras.layers.Dense(units=300, activation='relu', input_shape=[X_train.shape[1]]),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.15),
    keras.layers.Dense(units=260, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.15),
    keras.layers.Dense(units=199, activation='sigmoid')
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=200, batch_size=32)


# In[62]:


def make_pred(df, sample, env, model):
    df['DefendersInTheBox_vs_Distance'] = df['DefendersInTheBox'] / df['Distance']
    df['OffenseFormation'] = df['OffenseFormation'].apply(lambda x: x if x in off_form else np.nan)
    df = pd.concat([df.drop(['OffenseFormation'], axis=1), pd.get_dummies(df['OffenseFormation'], prefix='Formation')], axis=1)
    missing_cols = set( dummy_col ) - set( test.columns )-set('Yards')
    for c in missing_cols:
        df[c] = 0
    df = df[dummy_col]
    df.drop(['Yards'], axis=1, inplace=True)
    df['PossessionTeam'] = df['PossessionTeam'].map(map_abbr)
    df['HomeTeamAbbr'] = df['HomeTeamAbbr'].map(map_abbr)
    df['VisitorTeamAbbr'] = df['VisitorTeamAbbr'].map(map_abbr)
    df['HomePossesion'] = df['PossessionTeam'] == df['HomeTeamAbbr']
    df['Field_eq_Possession'] = df['FieldPosition'] == df['PossessionTeam']
    df['GameClock'] = df['GameClock'].apply(strtoseconds)
    df['PlayerHeight'] = df['PlayerHeight'].apply(lambda x: 12*int(x.split('-')[0])+int(x.split('-')[1]))
    df['TimeHandoff'] = df['TimeHandoff'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
    df['TimeSnap'] = df['TimeSnap'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ"))
    df['TimeDelta'] = df.apply(lambda row: (row['TimeHandoff'] - row['TimeSnap']).total_seconds(), axis=1)
    df['PlayerBirthDate'] = df['PlayerBirthDate'].apply(lambda x: datetime.datetime.strptime(x, "%m/%d/%Y"))
    seconds_in_year = 60*60*24*365.25
    df['PlayerAge'] = df.apply(lambda row: (row['TimeHandoff']-row['PlayerBirthDate']).total_seconds()/seconds_in_year, axis=1)
    df['WindSpeed'] = df['WindSpeed'].apply(lambda x: x.lower().replace('mph', '').strip() if not pd.isna(x) else x)
    df['WindSpeed'] = df['WindSpeed'].apply(lambda x: (int(x.split('-')[0])+int(x.split('-')[1]))/2 if not pd.isna(x) and '-' in x else x)
    df['WindSpeed'] = df['WindSpeed'].apply(lambda x: (int(x.split()[0])+int(x.split()[-1]))/2 if not pd.isna(x) and type(x)!=float and 'gusts up to' in x else x)
    df['WindSpeed'] = df['WindSpeed'].apply(str_to_float)
    df['PlayDirection'] = train['PlayDirection'].apply(lambda x: x is 'right')
    df['Team'] = df['Team'].apply(lambda x: x.strip()=='home')
    indoor = "indoor"
    df['GameWeather'] = df['GameWeather'].apply(lambda x: indoor if not pd.isna(x) and indoor in x else x)
    df['GameWeather'] = df['GameWeather'].apply(lambda x: x.lower().replace('coudy', 'cloudy').replace('clouidy', 'cloudy').replace('party', 'partly').replace('clear and sunny', 'sunny and clear').replace('skies', '').replace("mostly", "").strip() if not pd.isna(x) else x)
    df['GameWeather'] = df['GameWeather'].apply(map_weather)
    df['IsRusher'] = df['NflId'] == df['NflIdRusher']
    
    df = df.sort_values(by=['PlayId', 'Team', 'IsRusher']).reset_index()
    df = df.drop(['TimeHandoff', 'TimeSnap', 'PlayerBirthDate', 'WindDirection', 'NflId', 'NflIdRusher', 'GameId', 'PlayId', 'index', 'IsRusher', 'Team'], axis=1)
    cat_features = []
    for col in df.columns:
        if df[col].dtype =='object':
            cat_features.append(col)

    df = df.drop(cat_features, axis=1)
    df.fillna(-999, inplace=True)
    X = np.array(df[players_col]).reshape(-1, 11*22)
    play_col = df.drop(players_col, axis=1).columns
    X_play_col = np.zeros(shape=(X.shape[0], len(play_col)))
    for i, col in enumerate(play_col):
        X_play_col[:, i] = df[col][::22]
    X = np.concatenate([X, X_play_col], axis=1)
    X = scaler.transform(X)
    y_pred = model.predict(X)
    for pred in y_pred:
        prev = 0
        for i in range(len(pred)):
            if pred[i]<prev:
                pred[i]=prev
            prev=pred[i]
    y_pred[:, -1] = np.ones(shape=(y_pred.shape[0], 1))
    y_pred[:, 0] = np.zeros(shape=(y_pred.shape[0], 1))
    env.predict(pd.DataFrame(data=y_pred,columns=sample.columns))
    return y_pred


# In[63]:


for test, sample in tqdm.tqdm(env.iter_test()):
    make_pred(test, sample, env, model)


# In[64]:


env.write_submission_file()


# # End
# 
# If you reached this far please comment and upvote this kernel, feel free to make improvements on the kernel and please share if you found anything useful!

# In[ ]:




