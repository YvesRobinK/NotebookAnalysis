#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
    
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## **Importing Data**

# In[2]:


train = pd.read_csv('/kaggle/input/playground-series-s3e13/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s3e13/test.csv')
sample = pd.read_csv('/kaggle/input/playground-series-s3e13/sample_submission.csv')


# In[3]:


train.head()


# In[4]:


train.describe().T


# In[5]:


train.isna().mean()


# In[6]:


train.columns


# ## **Vizualization**

# In[7]:


corr_matrix = train.corr()

plt.figure(figsize = (12,8))
sns.heatmap(corr_matrix, cmap="YlGnBu")
plt.title("Correlation Matrix")
plt.show()


# In[8]:


for col in train.columns[2:-1]:
    ax = sns.countplot(x=col, hue="prognosis", data=train)
    ax.set_title(col)
    ax.legend(title="Prognosis", bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()


# ## **Feature Engineering**

# In[9]:


train = train.drop(['bullseye_rash', 'speech_problem'], axis=1)
test = test.drop(['bullseye_rash', 'speech_problem'], axis=1)


# In[10]:


features = ['sudden_fever', 'headache', 'mouth_bleed', 'nose_bleed',
       'muscle_pain', 'joint_pain', 'vomiting', 'rash', 'diarrhea',
       'hypotension', 'pleural_effusion', 'ascites', 'gastro_bleeding',
       'swelling', 'nausea', 'chills', 'myalgia', 'digestion_trouble',
       'fatigue', 'skin_lesions', 'stomach_pain', 'orbital_pain', 'neck_pain',
       'weakness', 'back_pain', 'weight_loss', 'gum_bleed', 'jaundice', 'coma',
       'diziness', 'inflammation', 'red_eyes', 'loss_of_appetite',
       'urination_loss', 'slow_heart_rate', 'abdominal_pain',
       'light_sensitivity', 'yellow_skin', 'yellow_eyes', 'facial_distortion',
       'microcephaly', 'rigor', 'bitter_tongue', 'convulsion', 'anemia',
       'cocacola_urine', 'hypoglycemia', 'prostraction', 'hyperpyrexia',
       'stiff_neck', 'irritability', 'confusion', 'tremor', 'paralysis',
       'lymph_swells', 'breathing_restriction', 'toe_inflammation',
       'finger_inflammation', 'lips_irritation', 'itchiness', 'ulcers',
       'toenail_loss']


# In[11]:


def count_of_symp(df, features):
    for i in df.index:
        df['ones'][i] = len(df[features].loc[:, df[features].loc[i, :] > 0].columns)
        df['zeros'][i] = len(df[features].loc[:, df[features].loc[i, :] == 0].columns)
    return df


# In[12]:


train[['ones', 'zeros']] = 0
test[['ones', 'zeros']] = 0

train = count_of_symp(train, features)
test = count_of_symp(test, features)


# In[13]:


train.head()


# In[14]:


disease_symptoms = {
    'west_nile_fever': ['fever', 'headache', 'body_aches', 'joint_pain', 'vomiting'],
    'japanese_encephalitis': ['fever', 'headache', 'nausea', 'vomiting', 'fatigue'],
    'tungiasis': ['itchiness', 'pain', 'swelling', 'ulcers', 'toenail_loss'],
    'rift_valley_fever': ['fever', 'headache', 'muscle_pain', 'joint_pain', 'fatigue'],
    'chikungunya': ['fever', 'joint_pain', 'muscle_pain', 'headache', 'nausea'],
    'dengue': ['fever', 'headache', 'joint_pain', 'muscle_pain', 'fatigue'],
    'yellow_fever': ['fever', 'headache', 'jaundice', 'muscle_pain', 'nausea'],
    'zika': ['fever', 'rash', 'joint_pain', 'conjunctivitis', 'muscle_pain'],
    'plague': ['fever', 'chills', 'weakness', 'abdominal_pain', 'swollen_lymph_nodes'],
    'lyme_disease': ['fever', 'fatigue', 'headache', 'muscle_pain', 'joint_pain'],
    'malaria': ['fever', 'chills', 'headache', 'nausea', 'vomiting']
}


# In[15]:


l_illneses = ['west_nile_fever', 'japanese_encephalitis', 'tungiasis',
             'rift_valley_fever', 'chikungunya', 'dengue', 'yellow_fever', 
             'zika', 'plague', 'lyme_disease', 'malaria']
l_prognoses = ['West_Nile_fever' ,'Japanese_encephalitis',
               'Tungiasis' ,'Rift_Valley_fever','Chikungunya' ,'Dengue',
               'Yellow_Fever', 'Zika' ,'Plague', 'Lyme_disease' , 'Malaria']


# In[16]:


train['prognosis'].value_counts()


# In[17]:


from sklearn import preprocessing

le = preprocessing.LabelEncoder()


# In[18]:


train['prognosis'] = le.fit_transform(train['prognosis'])


# In[19]:


train['prognosis'].value_counts()


# In[20]:


from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap.umap_ as umap

def decomposition(df, features):
    X = df[features].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    umap_model = umap.UMAP(n_neighbors=2, min_dist=0.4932426973138444, n_components=10)
    umap_result = umap_model.fit_transform(X)
    df[[f'umap{i}' for i in range(len(umap_result[0]))]] = umap_result
    

    tsne_model = TSNE(perplexity=2, learning_rate=0.1, n_components=9, method='exact')
    tsne_result = tsne_model.fit_transform(X)
    df[[f'tsne{i}' for i in range(len(tsne_result[0]))]] = tsne_result
    
    return df


# In[21]:


X = train[features].values
y = train['prognosis'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

umap_model = umap.UMAP(n_neighbors=2, min_dist=0.4932426973138444, n_components=10)
umap_result = umap_model.fit_transform(X)

tsne_model = TSNE(perplexity=2, learning_rate=0.1, n_components=9, method='exact')
tsne_result = tsne_model.fit_transform(X)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].scatter(umap_result[:, 0], umap_result[:, 1], c=y)
axs[0].set_title('UMAP')

axs[1].scatter(tsne_result[:, 0], tsne_result[:, 1], c=y)
axs[1].set_title('t-SNE')

plt.show()


# In[22]:


train = decomposition(train, features)
test = decomposition(test, features)


# In[23]:


train = train.set_index('id')
test = test.set_index('id')


# In[24]:


from collections import defaultdict

def imp(data, features):
    d = defaultdict(dict)
    for i in data.index:
        a = data[features].loc[:, data[features].loc[i, :] > 0].columns
        for ill in l_illneses:
            l = []
            for j in list(a):
                if j in disease_symptoms[ill]:
                    l.append(j)
            d[i][ill] = []
            if len(a) > 0:
                d[i][ill].append(len(l))
            l = []
    
    df = pd.DataFrame.from_dict(d, orient='index')
    df = df.applymap(lambda x: x[0])
    df = pd.concat([data, df], axis=1)
    
    return df
        


# In[25]:


df_train = imp(train, features)
df_test = imp(test, features)


# In[26]:


class Sub:

    def __init__(self, y_pred):
        self.sorted_prediction_ids = np.argsort(-y_pred, axis=1)
        self.top_3_prediction_ids = self.sorted_prediction_ids[:,:3]
        self.original_shape = self.top_3_prediction_ids.shape

    def mapk_score(self, y_test):
        
        def apk(actual, predicted, k=10):
            if len(predicted)>k:
                predicted = predicted[:k]

            score = 0.0
            num_hits = 0.0

            for i,p in enumerate(predicted):
                if p in actual and p not in predicted[:i]:
                    num_hits += 1.0
                    score += num_hits / (i+1.0)

            if not actual:
                return 0.0

            return score / min(len(actual), k)

        def mapk(actual, predicted, k=10):
            return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])
        
        top_3_predictions = le.inverse_transform(self.top_3_prediction_ids.reshape(-1, 1))
        top_3_predictions = top_3_predictions.reshape(self.original_shape)

        return mapk(y_test.values.reshape(-1, 1), self.top_3_prediction_ids, k=3)

    def sub(self):
        top_3_predictions = le.inverse_transform(self.top_3_prediction_ids.reshape(-1, 1))
        top_3_predictions = top_3_predictions.reshape(self.original_shape)
        
        res = []
        for i in top_3_predictions:
            res.append(' '.join(str(j) for j in i))
            
        return res


# ## **Model Building**

# In[27]:


df_train.columns


# In[28]:


features = [
    'sudden_fever', 'headache', 'mouth_bleed', 'nose_bleed', 'muscle_pain',
       'joint_pain', 'vomiting', 'rash', 'diarrhea', 'hypotension',
       'pleural_effusion', 'ascites', 'gastro_bleeding', 'swelling', 'nausea',
       'chills', 'myalgia', 'digestion_trouble', 'fatigue', 'skin_lesions',
       'stomach_pain', 'orbital_pain', 'neck_pain', 'weakness', 'back_pain',
       'weight_loss', 'gum_bleed', 'jaundice', 'coma', 'diziness',
       'inflammation', 'red_eyes', 'loss_of_appetite', 'urination_loss',
       'slow_heart_rate', 'abdominal_pain', 'light_sensitivity', 'yellow_skin',
       'yellow_eyes', 'facial_distortion', 'microcephaly', 'rigor',
       'bitter_tongue', 'convulsion', 'anemia', 'cocacola_urine',
       'hypoglycemia', 'prostraction', 'hyperpyrexia', 'stiff_neck',
       'irritability', 'confusion', 'tremor', 'paralysis', 'lymph_swells',
       'breathing_restriction', 'toe_inflammation', 'finger_inflammation',
       'lips_irritation', 'itchiness', 'ulcers', 'toenail_loss',
       'ones', 'zeros', 'umap0', 'umap1', 'umap2', 'umap3', 'umap4', 'umap5',
       'umap6', 'umap7', 'umap8', 'umap9', 'tsne0', 'tsne1', 'tsne2', 'tsne3',
       'tsne4', 'tsne5', 'tsne6', 'tsne7', 'tsne8', 'west_nile_fever',
       'japanese_encephalitis', 'tungiasis', 'rift_valley_fever',
       'chikungunya', 'dengue', 'yellow_fever', 'zika', 'plague',
       'lyme_disease', 'malaria']


target = 'prognosis'


# ### **XGBoost and CatBoost**

# In[29]:


get_ipython().system('pip install -q catboost')


# In[30]:


import xgboost as xgb
import catboost as cb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

X = df_train[features]
y = df_train[target]


xgb_params = {
            'n_estimators': 1000,
            'learning_rate': 0.01,
            'max_depth': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.1,
            'n_jobs': -1,
            'eval_metric': 'mlogloss',
            'objective': 'multi:softprob',
            'tree_method': 'hist',
            'verbosity': 0,
            'random_state': 42,
        }

model_xgb = xgb.XGBClassifier(**xgb_params)


cb_params = {
            'n_estimators': 1000,
            'learning_rate': 0.01,
            'depth': 3,
            'subsample': 0.8,
            'colsample_bylevel': 0.1,
            'thread_count': -1,
            'eval_metric': 'MultiClass',
            'objective': 'MultiClass',
            'bootstrap_type': 'Bernoulli',
            'leaf_estimation_method': 'Newton',
            'random_seed': 42,
            'verbose': 0
}

model_cat = cb.CatBoostClassifier(**cb_params)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []
kk = 0
for train_idx, test_idx in kf.split(X):
    if kk == 0:
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        model_xgb.fit(X_train, y_train)
        model_cat.fit(X_train, y_train)

        y_pred_xgb = model_xgb.predict_proba(X_test)
        y_pred_cat = model_cat.predict_proba(X_test)

        train_sub1 = Sub(y_pred_xgb).mapk_score(y_test)
        train_sub2 = Sub(y_pred_cat).mapk_score(y_test)

        scores.append((train_sub1, train_sub2))
        break
    kk+=1
    

print('Cross-validation scores:', scores)
print('Average accuracy:', np.mean(scores))


# In[31]:


fig, ax = plt.subplots(figsize=(14, 15))
xgb.plot_importance(model_xgb, ax=ax)
plt.show()


# In[32]:


feat_importances = model_cat.get_feature_importance()

plt.figure(figsize=(15,6))
plt.bar(range(len(features)), feat_importances, tick_label=features)
plt.xticks(rotation=90)
plt.title('Feature Importances')
plt.show()


# ### **Voting**

# In[33]:


from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(estimators=[
    ('xgb', model_xgb),
    ('cb', model_cat)
], voting='soft', n_jobs=-1, verbose=True)

voting_clf.fit(X_train, y_train)

y_pred = voting_clf.predict_proba(X_test)

# Calculate the accuracy score for the predictions
accuracy = accuracy_score(y_test, voting_clf.predict(X_test))
print('Accuracy:', accuracy)


# In[34]:


train_sub = Sub(y_pred)
train_sub.mapk_score(y_test)
# train_sub.sub()


# ## **Submission**

# In[35]:


test_sub = Sub(model_xgb.predict_proba(df_test[features]))
sample['prognosis'] = test_sub.sub()


# In[36]:


sample


# In[37]:


sample.to_csv('submission.csv', index=False)


# In[ ]:




