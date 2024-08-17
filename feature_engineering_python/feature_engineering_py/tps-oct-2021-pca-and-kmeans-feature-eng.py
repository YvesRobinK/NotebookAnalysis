#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import cudf

from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_regression

get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

import shap


# In[2]:


get_ipython().run_cell_magic('time', '', 'train = cudf.read_csv(\'../input/tabular-playground-series-oct-2021/train.csv\', index_col=0)\ntest = cudf.read_csv(\'../input/tabular-playground-series-oct-2021/test.csv\', index_col=0)\n\nsample_submission = cudf.read_csv("../input/tabular-playground-series-oct-2021/sample_submission.csv").to_pandas()\n\nmemory_usage = train.memory_usage(deep=True) / 1024 ** 2\nstart_mem = memory_usage.sum()\n')


# In[3]:


feature_cols = [col for col in test.columns.tolist()]

cnt_features =[]
cat_features =[]

for col in feature_cols:
    if train[col].dtype=='float64':
        cnt_features.append(col)
    else:
        cat_features.append(col)
        

train[cnt_features] = train[cnt_features].astype('float32')
train[cat_features] = train[cat_features].astype('uint8')

test[cnt_features] = test[cnt_features].astype('float32')
test[cat_features] = test[cat_features].astype('uint8')

memory_usage = train.memory_usage(deep=True) / 1024 ** 2
end_mem = memory_usage.sum()

train = train.to_pandas()
test = test.to_pandas()


# In[4]:


print("Mem. usage decreased from {:.2f} MB to {:.2f} MB ({:.2f}% reduction)".format(start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))


# # KMeans

# In[5]:


get_ipython().run_cell_magic('time', '', 'useful_features = ["f22", "f179", "f69", "f58", "f214", "f78", "f136", "f156", "f8", "f3", "f77", "f200", "f92", "f185", "f142", "f115", "f284"]\nn_clusters = 6\ncd_feature = True # cluster distance instead of cluster number\ncluster_cols = [f"cluster{i+1}" for i in range(n_clusters)]\nkmeans = KMeans(n_clusters=n_clusters, n_init=50, max_iter=500, random_state=42)\n\nif cd_feature:\n    # train\n    X_cd = kmeans.fit_transform(train[useful_features])\n    X_cd = pd.DataFrame(X_cd, columns=cluster_cols, index=train.index)\n    train = train.join(X_cd)\n    # test\n    X_cd = kmeans.transform(test[useful_features])\n    X_cd = pd.DataFrame(X_cd, columns=cluster_cols, index=test.index)\n    test = test.join(X_cd)\n    \nelse:\n    # train\n    train["cluster"] = kmeans.fit_predict(train[useful_features])\n    # test\n    test["cluster"] = kmeans.predict(test[useful_features])\n    \n    # one-hot encode\n    ohe = OneHotEncoder()\n    X_ohe = ohe.fit_transform(np.array(train["cluster"]).reshape(-1,1)).toarray()\n    T_ohe = ohe.transform(np.array(test["cluster"]).reshape(-1,1)).toarray()\n\n    X_ohe = pd.DataFrame(X_ohe, columns=cluster_cols, index=train.index)\n    T_ohe = pd.DataFrame(T_ohe, columns=cluster_cols, index=test.index)\n\n    train = pd.concat([train, X_ohe],axis=1)\n    test = pd.concat([test, T_ohe],axis=1)\n\nfeature_cols += cluster_cols\ntrain.head()\n')


# In[6]:


fig = plt.figure(figsize = (10,5))

if cd_feature:
    sns.kdeplot(data=train[cluster_cols])
else:
    ax = sns.countplot(data=train, x='cluster', hue="target")
    for p in ax.patches:
        ax.annotate(f'\n{p.get_height()}', (p.get_x()+0.2, p.get_height()), ha='center', va='top', color='white', size=5)

plt.show()


# # PCA

# In[7]:


pca = PCA()
X_pca = pca.fit_transform(train[useful_features])
T_pca = pca.transform(test[useful_features])

pca_cols = [f"PC{i+1}" for i in range(X_pca.shape[1])]

X_pca = pd.DataFrame(X_pca, columns=pca_cols, index=train.index)
T_pca = pd.DataFrame(T_pca, columns=pca_cols, index=test.index)

train = pd.concat([train, X_pca], axis=1)
test = pd.concat([test, T_pca], axis=1)
train.head()


# In[8]:


loadings = pd.DataFrame(pca.components_, index=pca_cols, columns=train[useful_features].columns)
loadings.style.bar(align='mid', color=['#d65f5f', '#5fba7d'])


# In[9]:


feature_cols += ["PC11", "PC12"]


# # Add new features

# In[10]:


def add_feature(df):
    df["new_f1"] = df["f255"]*df["f249"]
    df["new_f2"] = (df["cluster1"]+df["cluster3"])/(df["cluster2"]+df["cluster4"])
    return df

new_features = ["new_f1", "new_f2"]
train = add_feature(train)
test = add_feature(test)
feature_cols += new_features
train.head()


# # Mutual Information

# In[11]:


get_ipython().run_cell_magic('time', '', 'x = train.iloc[:5000,:][feature_cols].copy()\ny = train.iloc[:5000,:][\'target\'].copy()\nmi_scores = mutual_info_regression(x, y)\nmi_scores = pd.Series(mi_scores, name="MI Scores", index=x.columns)\nmi_scores = mi_scores.sort_values(ascending=False)\n')


# In[12]:


top = 20
fig = px.bar(mi_scores, x=mi_scores.values[:top], y=mi_scores.index[:top])
fig.update_layout(
    title=f"Top {top} Strong Relationships Between Feature Columns and Target Column",
    xaxis_title="Relationship with Target",
    yaxis_title="Feature Columns",
    yaxis={'categoryorder':'total ascending'},
    colorway=["blue"]
)
fig.show()


# # KFold

# In[13]:


folds = 5
train["kfold"] = -1
kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

for fold, (train_indicies, valid_indicies) in enumerate(kf.split(train,train["target"])):
    train.loc[valid_indicies, "kfold"] = fold


# # XGBoost

# In[14]:


get_ipython().run_cell_magic('time', '', 'final_test_predictions = []\nscores = []\n\nfor fold in range(folds):\n    x_train = train[train.kfold != fold].copy()\n    x_valid = train[train.kfold == fold].copy()\n    x_test  = test[feature_cols].copy()\n    \n    y_train = x_train[\'target\']\n    y_valid = x_valid[\'target\']\n    \n    x_train = x_train[feature_cols]\n    x_valid = x_valid[feature_cols]\n\n    xgb_params = {\n        \'eval_metric\': \'auc\', \n        \'objective\': \'binary:logistic\', \n        \'tree_method\': \'gpu_hist\', \n        \'gpu_id\': 0, \n        \'predictor\': \'gpu_predictor\', \n        \'n_estimators\': 10000, \n        \'learning_rate\': 0.01063045229441343, \n        \'gamma\': 0.24652519525750877, \n        \'max_depth\': 4, \n        \'seed\': 42,       \n        \'min_child_weight\': 366, \n        \'subsample\': 0.6423040816299684, \n        \'colsample_bytree\': 0.7751264493218339, \n        \'colsample_bylevel\': 0.8675692743597421, \n        \'use_label_encoder\': False,\n        \'lambda\': 0, \n        \'alpha\': 10\n    }\n    \n    xgb_model = XGBClassifier(**xgb_params)\n    xgb_model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)\n    \n    preds_train = xgb_model.predict_proba(x_train)[:,1]\n    preds_valid = xgb_model.predict_proba(x_valid)[:,1]\n    auc_train = roc_auc_score(y_train, preds_train)\n    auc = roc_auc_score(y_valid, preds_valid)\n    print("Fold",fold,", train:", f"{auc_train:.6f}", ", valid:", f"{auc:.6f}")\n    scores.append(auc)\n    \n    preds_test = xgb_model.predict_proba(x_test)[:,1]\n    final_test_predictions.append(preds_test)\n    \n    \nprint("AVG AUC:",np.mean(scores))\n')


# # SHAP Values

# In[15]:


shap_values = shap.TreeExplainer(xgb_model).shap_values(x_valid)
shap.summary_plot(shap_values, x_valid)


# In[16]:


idx = 5
data_for_prediction = x_valid.iloc[idx]
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)


print(xgb_model.predict_proba(data_for_prediction_array))

shap.initjs()
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(data_for_prediction_array)
shap.force_plot(explainer.expected_value, shap_values, data_for_prediction)


# # Plot Prediction

# In[17]:


labels = [f'fold {i}' for i in range(folds)]

fig = ff.create_distplot(final_test_predictions, labels, bin_size=.3, show_hist=False, show_rug=False)
fig.show()


# In[18]:


sample_submission['target'] = np.mean(np.column_stack(final_test_predictions), axis=1)
sample_submission.to_csv("submission.csv", index=False)

