#!/usr/bin/env python
# coding: utf-8

# <div style="background-color:#B0C3D2; color:#19180F; font-size:15px; font-family:Verdana; padding:10px; border: 2px solid #19180F; border-radius:10px"> 
# 📌 The competition is closely related to my research paper published on <a href="https://link.springer.com/chapter/10.1007/978-981-13-0923-6_48"> Springer </a>
#         </div>

# <div style="background-color:#B0C3A1; color:#19180F; font-size:15px; font-family:Verdana; padding:10px; border: 2px solid #19180F; border-radius:10px"> 
#     💡<b>What's New</b><br>
#      Enhanced feature engineering via Word2Vec model </div>

# <div style="background-color:#F0E3D2; color:#19180F; font-size:15px; font-family:Verdana; padding:10px; border: 2px solid #19180F; border-radius:10px"> 
# 📌
#     <b> Importing modules.
#     </div>

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import optuna
from tqdm import tqdm
import gensim
from gensim.models import Word2Vec


# <div style="background-color:#F0E3D2; color:#19180F; font-size:15px; font-family:Verdana; padding:10px; border: 2px solid #19180F; border-radius:10px"> 
# 📌<b> Loading dataset with reduced memory usage
#     </div>

# In[2]:


train_logs = pd.read_csv('/kaggle/input/linking-writing-processes-to-writing-quality/train_logs.csv', low_memory=True)#, nrows=10)
test_logs = pd.read_csv('/kaggle/input/linking-writing-processes-to-writing-quality/test_logs.csv', low_memory=True)#, nrows=10)
train_scores = pd.read_csv('/kaggle/input/linking-writing-processes-to-writing-quality/train_scores.csv', low_memory=True)#, nrows=10)


# <div style="background-color:#F0E3D2; color:#19180F; font-size:15px; font-family:Verdana; padding:10px; border: 2px solid #19180F; border-radius:10px"> 
# 📌<b>Merging datasets
#     </div>

# In[3]:


train_data = pd.merge(train_logs, train_scores, on='id')


# <div style="background-color:#F0E3D2; color:#19180F; font-size:15px; font-family:Verdana; padding:10px; border: 2px solid #19180F; border-radius:10px"> 
# 📌<b> Defining features and target variable
#     </div>

# In[4]:


features = train_data.drop(['id', 'score'], axis=1)
target = train_data['score']


# <div style="background-color:#F0E3D2; color:#19180F; font-size:15px; font-family:Verdana; padding:10px; border: 2px solid #19180F; border-radius:10px"> 
# 📌<b> Adding a feature engineering step for fetching text embeddings
#     </div>

# <div style="background-color:#F0E3D2; color:#19180F; font-size:8px; font-family:Verdana; padding:10px; border: 2px solid #19180F; border-radius:10px"> 
# 📌<b> The tradeoff between vector size of word2vec and RAM of the compute exists ! Choosing one that balances it is recommended
#     </div>

# In[5]:


word2vec_model = Word2Vec(sentences=train_logs['text_change'].apply(str.split), vector_size=2, window=1, min_count=1, workers=4)
def preprocess_data(data):
    # Using pd.factorize for categorical features as a low-memory alternative instead of one-hot encoder
    cat_cols = ['activity', 'down_event', 'up_event', 'text_change']

    for col in tqdm(cat_cols):
        data[col], _ = pd.factorize(data[col])

    # Handling missing values
    data.fillna(0, inplace=True)

    return data


# In[6]:


def preprocess_text(data, embeddings_model):
    tqdm.pandas()  
    data['text_change'] = data['text_change'].astype(str)
    
    embedding_cols = [f'embedding_{i}' for i in range(embeddings_model.vector_size)]
    #creating seperate columns for embeddings
    data[embedding_cols] = data['text_change'].progress_apply(lambda x: pd.Series(np.mean([embeddings_model.wv[word] for word in x.split() if word in embeddings_model.wv] or [np.zeros(embeddings_model.vector_size)], axis=0)))
    
    return data,embedding_cols


# <div style="background-color:#F0E3D2; color:#19180F; font-size:15px; font-family:Verdana; padding:10px; border: 2px solid #19180F; border-radius:10px"> 
# 📌<b> Preprocessing features
#     </div>

# In[7]:


# Preprocessing data
train_data = preprocess_data(train_data)
test_logs = preprocess_data(test_logs)

# Preprocessing text features
train_data, embedding_cols = preprocess_text(train_data, word2vec_model)
test_logs,embedding_cols  = preprocess_text(test_logs, word2vec_model)


# <div style="background-color:#F0E3D2; color:#19180F; font-size:15px; font-family:Verdana; padding:10px; border: 2px solid #19180F; border-radius:10px"> 
# 📌<b> Combining text features with numeric features
#     </div>

# In[8]:


features = pd.concat([features, train_data[embedding_cols]], axis=1)
test_features = test_logs.drop('id', axis=1),


# In[9]:


features = preprocess_data(features)


# <div style="background-color:#F0E3D2; color:#19180F; font-size:15px; font-family:Verdana; padding:10px; border: 2px solid #19180F; border-radius:10px"> 
# 📌<b> Splitting data into train and val sets
#     </div>

# In[10]:


X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)


# <div style="background-color:#F0E3D2; color:#19180F; font-size:15px; font-family:Verdana; padding:10px; border: 2px solid #19180F; border-radius:10px"> 
# 📌<b> Objective function for optuna based hyper param optimization
#     </div>

# <div style="background-color:#F0E3D2; color:#19180F; font-size:8px; font-family:Verdana; padding:10px; border: 2px solid #19180F; border-radius:10px"> 
# 📌<b> The tree_method and device set as gpu is commented since weekly quota is over.
#     </div>

# In[11]:


def objective_lgbm(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'text_feature_importance': trial.suggest_float('text_feature_importance', 0.1, 0.9),
    }

    model = LGBMRegressor(**params, random_state=42,n_jobs=-1)#, device="gpu")
    rmse = kfold_cv(X_train, y_train, model)
    return rmse

def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'text_feature_importance': trial.suggest_float('text_feature_importance', 0.1, 0.9),
    }

    model = XGBRegressor(**params, random_state=42,n_jobs=4)#, tree_method='gpu_hist')
    rmse = kfold_cv(X_train, y_train, model)
    return rmse


# <div style="background-color:#F0E3D2; color:#19180F; font-size:15px; font-family:Verdana; padding:10px; border: 2px solid #19180F; border-radius:10px"> 
# 📌<b> Function for performing K fold CV
#     </div>

# In[12]:


def kfold_cv(X, y, model, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmse_scores = []

    for train_index, val_index in kf.split(X):
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_val_fold)
        rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
        rmse_scores.append(rmse)

    return np.mean(rmse_scores)


# <div style="background-color:#F0E3D2; color:#19180F; font-size:15px; font-family:Verdana; padding:10px; border: 2px solid #19180F; border-radius:10px"> 
# 📌<b> Optimizing hyperparams and fetching best params
#     </div>

# In[13]:


study_lgbm = optuna.create_study(direction='minimize')
study_lgbm.optimize(objective_lgbm, n_trials=1)

study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgb, n_trials=1)

best_params_lgbm = study_lgbm.best_params
best_model_lgbm = LGBMRegressor(**best_params_lgbm, random_state=42,n_jobs=-1)#, device="gpu")

best_params_xgb = study_xgb.best_params
best_model_xgb = XGBRegressor(**best_params_xgb, random_state=42,n_jobs=4)#, tree_method='gpu_hist')



# <div style="background-color:#F0E3D2; color:#19180F; font-size:15px; font-family:Verdana; padding:10px; border: 2px solid #19180F; border-radius:10px"> 
# 📌<b> Defining models for ensemble and fitting em
#     </div>

# In[14]:


ensemble = VotingRegressor(estimators=[('lgbm', best_model_lgbm), ('xgb', best_model_xgb)], n_jobs=-1)

ensemble.fit(X_train, y_train)


# <div style="background-color:#F0E3D2; color:#19180F; font-size:15px; font-family:Verdana; padding:10px; border: 2px solid #19180F; border-radius:10px"> 
# 📌<b> Making predictions on val set and calculating RMSE
#     </div>

# In[15]:


val_pred = ensemble.predict(X_val)

val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
print(f'Validation RMSE with Voting Ensemble: {val_rmse}')


# <div style="background-color:#F0E3D2; color:#19180F; font-size:15px; font-family:Verdana; padding:10px; border: 2px solid #19180F; border-radius:10px"> 
# 📌<b> Feature importances
#     </div>

# In[16]:


model1 = LGBMRegressor(random_state=42,n_jobs=-1)#, device="gpu")
model1.fit(X_train, y_train)
feature_importance_lgbm = model1.feature_importances_
feature_names = X_train.columns
feature_importance_lgbm_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance_lgbm})
feature_importance_lgbm_df = feature_importance_lgbm_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_lgbm_df)
plt.title('Feature Importance for LightGBM (in the ensemble)')
plt.show()


# <div style="background-color:#F0E3D2; color:#19180F; font-size:15px; font-family:Verdana; padding:10px; border: 2px solid #19180F; border-radius:10px"> 
# 📌<b> Generating predictions
#     </div>

# In[19]:


test_features = np.reshape(test_features,(6,12))


# In[20]:


test_predictions = ensemble.predict(test_features)


# <div style="background-color:#F0E3D2; color:#19180F; font-size:15px; font-family:Verdana; padding:10px; border: 2px solid #19180F; border-radius:10px"> 
# 📌<b> Generating submission file
#     </div>

# In[21]:


test_predictions


# In[22]:


submission = pd.DataFrame({'id': test_logs['id'], 'score': test_predictions})


# In[23]:


submission_grouped = submission.groupby('id')['score'].mean().reset_index()

submission_grouped.to_csv('/kaggle/working/submission.csv', index=False)



# In[24]:


submission_grouped


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




