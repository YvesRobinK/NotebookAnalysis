#!/usr/bin/env python
# coding: utf-8

# # The Best Model for TPSAPR22 Without Neural Networks
# 
# This notebook shows how to solve TPSAPR22 with good feature engineering and a `HistGradientBoostingClassifier`. It furthermore shows how to cross-validate correctly without creating a data leak.
# 
# Some features have been inspired by C4rl05/V's [XGBoost notebook](https://www.kaggle.com/code/cv13j0/tps-apr-2022-xgboost-model).

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from cycler import cycler
from IPython.display import display
import datetime
import scipy.stats

from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier
from sklearn.pipeline import make_pipeline

pd.set_option("precision", 3)
plt.rcParams['axes.facecolor'] = '#0057b8' # blue
plt.rcParams['axes.prop_cycle'] = cycler(color=['#ffd700'] +
                                         plt.rcParams['axes.prop_cycle'].by_key()['color'][1:])


# # Reading the data
# 
# We read the data and pivot the training data so that we have a dataframe with one row per sequence.

# In[2]:


# Reading the data
train = pd.read_csv('../input/tabular-playground-series-apr-2022/train.csv')
train_labels = pd.read_csv('../input/tabular-playground-series-apr-2022/train_labels.csv')
test = pd.read_csv('../input/tabular-playground-series-apr-2022/test.csv')

sensors = [col for col in train.columns if 'sensor_' in col]

train_pivoted0 = train.pivot(index=['sequence', 'subject'], columns='step', values=sensors)
display(train_pivoted0)


# # Feature engineering
# 
# Let's keep it simple and calculate only the following features:
# - For every sensor, we calculate mean, standard deviation, interquartile range, standard deviation divided by mean, and kurtosis. This gives the first 5\*13=65 features.
# - For the special sensor_02, we count how many times it goes up or down.
# - For sensor_02, we calculate the sum of all upward / downward steps, the maximum of all upward / downward steps, and the mean of all upward / downward steps. 
# - For every subject, we count how many sequences belong to it, and we add this count as a feature to all sequences of the subject (the [EDA](https://www.kaggle.com/code/ambrosm/tpsapr22-eda-which-makes-sense) gives the motivation for this feature). 
# 
# Now we have 74 features. 

# In[3]:


# Feature engineering
def engineer(df):
    new_df = pd.DataFrame([], index=df.index)
    for sensor in sensors:
        new_df[sensor + '_mean'] = df[sensor].mean(axis=1)
        new_df[sensor + '_std'] = df[sensor].std(axis=1)
        new_df[sensor + '_iqr'] = scipy.stats.iqr(df[sensor], axis=1)
        new_df[sensor + '_sm'] = np.nan_to_num(new_df[sensor + '_std'] / 
                                               new_df[sensor + '_mean'].abs()).clip(-1e30, 1e30)
        new_df[sensor + '_kurtosis'] = scipy.stats.kurtosis(df[sensor], axis=1)
    new_df['sensor_02_up'] = (df.sensor_02.diff(axis=1) > 0).sum(axis=1)
    new_df['sensor_02_down'] = (df.sensor_02.diff(axis=1) < 0).sum(axis=1)
    new_df['sensor_02_upsum'] = df.sensor_02.diff(axis=1).clip(0, None).sum(axis=1)
    new_df['sensor_02_downsum'] = df.sensor_02.diff(axis=1) .clip(None, 0).sum(axis=1)
    new_df['sensor_02_upmax'] = df.sensor_02.diff(axis=1).max(axis=1)
    new_df['sensor_02_downmax'] = df.sensor_02.diff(axis=1).min(axis=1)
    new_df['sensor_02_upmean'] = np.nan_to_num(new_df['sensor_02_upsum'] / new_df['sensor_02_up'], posinf=40)
    new_df['sensor_02_downmean'] = np.nan_to_num(new_df['sensor_02_downsum'] / new_df['sensor_02_down'], neginf=-40)
    return new_df

train_pivoted = engineer(train_pivoted0)

train_shuffled = train_pivoted.sample(frac=1.0, random_state=1)
labels_shuffled = train_labels.reindex(train_shuffled.index.get_level_values('sequence'))
labels_shuffled = labels_shuffled[['state']].merge(train[['sequence', 'subject']].groupby('sequence').min(),
                                                   how='left', on='sequence')
labels_shuffled = labels_shuffled.merge(labels_shuffled.groupby('subject').size().rename('sequence_count'),
                                        how='left', on='subject')
train_shuffled['sequence_count_of_subject'] = labels_shuffled['sequence_count'].values

selected_columns = train_shuffled.columns
print(len(selected_columns))
#train_shuffled.columns


# To get a first impression of the usefulness of the 74 features, we plot how the target depends on every feature, i.e., a diagram of $P(y=1|x)$.  To get a meaningful plot, we apply two transformations:
# - The x axis is not the value of the feature, but its index (when sorted by feature value).
# - The y axis is not the target value (which can be only 0 or 1), but a rolling mean over 1000 targets.
# 
# The diagram shows bad features with an almost horizontal line (the probability of the positive target is 0.5 independently of the feature value) (e.g. sensor_05_std). Good features have a curve with high y_max - y_min (e.g. sensor_02_std). 

# In[4]:


# Plot dependence between every feature and the target
ncols = len(train_shuffled.columns) // 13
plt.subplots(15, ncols, sharey=True, sharex=True, figsize=(15, 40))
for i, col in enumerate(train_shuffled.columns):
    temp = pd.DataFrame({col: train_shuffled[col].values,
                         'state': labels_shuffled.state.values})
    temp = temp.sort_values(col)
    temp.reset_index(inplace=True)
    plt.subplot(15, ncols, i+1)
    plt.scatter(temp.index, temp.state.rolling(1000).mean(), s=2)
    plt.xlabel(col)
    plt.xticks([])
plt.show()


# # Feature selection
# 
# We don't need all 74 features. In a first step we drop 26 features which proved to be useless in a previous run of the notebook.

# In[5]:


# Drop some useless features
dropped_features = ['sensor_05_kurtosis', 'sensor_08_mean',
                    'sensor_05_std', 'sensor_06_kurtosis',
                    'sensor_06_std', 'sensor_03_std',
                    'sensor_02_kurtosis', 'sensor_03_kurtosis',
                    'sensor_09_kurtosis', 'sensor_03_mean',
                    'sensor_00_mean', 'sensor_02_iqr',
                    'sensor_05_mean', 'sensor_06_mean',
                    'sensor_07_std', 'sensor_10_iqr',
                    'sensor_11_iqr', 'sensor_12_iqr',
                    'sensor_09_mean', 'sensor_02_sm',
                    'sensor_03_sm', 'sensor_05_iqr', 
                    'sensor_06_sm', 'sensor_09_iqr', 
                    'sensor_07_iqr', 'sensor_10_mean']
selected_columns = [f for f in selected_columns if f not in dropped_features]
len(selected_columns)


# Now we select features sequentially. We start with zero features and add one feature after the other. In every step we select the feature which increases the model's validation score the most. In this example, we select all features, and the output tells us which features are useful and which aren't.
# 
# The same algorithm can be run backward by setting `backward` to `True`. It then starts with all features and repeatedly deletes the feature which adds the least value to the model's validation score.
# 
# The model is a `HistGradientBoostingClassifier`.

# In[6]:


# Sequential feature selection
# This code is a more verbose form of scikit-learn's SequentialFeatureSelector
estimator = HistGradientBoostingClassifier(learning_rate=0.05, max_leaf_nodes=25,
                                       max_iter=1000, min_samples_leaf=500,
                                       l2_regularization=1,
                                       max_bins=255,
                                       random_state=4, verbose=0)

X, y = train_shuffled[selected_columns], labels_shuffled.state
n_iterations, backward = 48, False

if n_iterations != 0:
    n_features = X.shape[1]
    current_mask = np.zeros(shape=n_features, dtype=bool)
    history = []
    for _ in range(n_iterations):
        candidate_feature_indices = np.flatnonzero(~current_mask)
        scores = {}
        for feature_idx in candidate_feature_indices:
            candidate_mask = current_mask.copy()
            candidate_mask[feature_idx] = True
            X_new = X.values[:, ~candidate_mask if backward else candidate_mask]
            scores[feature_idx] = cross_val_score(
                estimator,
                X_new,
                y,
                cv=GroupKFold(n_splits=5),
                groups=train_shuffled.index.get_level_values('subject'),
                scoring='roc_auc',
                n_jobs=-1,
            ).mean()
            #print(f"{str(X.columns[feature_idx]):30} {scores[feature_idx]:.3f}")
        new_feature_idx = max(scores, key=lambda feature_idx: scores[feature_idx])
        current_mask[new_feature_idx] = True
        history.append(scores[new_feature_idx])
        new = 'Deleted' if backward else 'Added'
        print(f'{new} feature: {str(X.columns[new_feature_idx]):30}'
              f' {scores[new_feature_idx]:.3f}')

    print()
    plt.figure(figsize=(12, 6))
    plt.scatter(np.arange(len(history)) + (0 if backward else 1), history)
    plt.ylabel('AUC')
    plt.xlabel('Features removed' if backward else 'Features added')
    plt.title('Sequential Feature Selection')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()

    if backward:
        current_mask = ~current_mask
    selected_columns = np.array(selected_columns)[current_mask]
    print(selected_columns)


# # Cross-validation
# 
# For cross-validation, we use a GroupKFold grouped on subjects. If we didn't group on subjects, we'd have a data leak (see the [EDA](https://www.kaggle.com/code/ambrosm/tpsapr22-eda-which-makes-sense) for an explanation).
# 
# The model is a `HistGradientBoostingClassifier`; I got the same cv score using an `XGBClassifier`.

# In[7]:


get_ipython().run_cell_magic('time', '', '# Cross-validation of the classifier\n\nprint(f"{len(selected_columns)} features")\nscore_list = []\nkf = GroupKFold(n_splits=5)\nfor fold, (idx_tr, idx_va) in enumerate(kf.split(train_shuffled, groups=train_shuffled.index.get_level_values(\'subject\'))):\n    X_tr = train_shuffled.iloc[idx_tr][selected_columns]\n    X_va = train_shuffled.iloc[idx_va][selected_columns]\n    y_tr = labels_shuffled.iloc[idx_tr].state\n    y_va = labels_shuffled.iloc[idx_va].state\n\n    model = HistGradientBoostingClassifier(learning_rate=0.05, max_leaf_nodes=25,\n                                           max_iter=1000, min_samples_leaf=500,\n                                           l2_regularization=1,\n                                           validation_fraction=0.05,\n                                           max_bins=63,\n                                           random_state=3, verbose=0)\n#     model = XGBClassifier(n_estimators=500, n_jobs=-1,\n#                           eval_metric=[\'logloss\'],\n#                           #max_depth=10,\n#                           colsample_bytree=0.8,\n#                           #gamma=1.4,\n#                           reg_alpha=6, reg_lambda=1.5,\n#                           tree_method=\'hist\',\n#                           learning_rate=0.03,\n#                           verbosity=1,\n#                           use_label_encoder=False, random_state=3)\n\n    if True or type(model) != XGBClassifier:\n        model.fit(X_tr.values, y_tr)\n    else:\n        model.fit(X_tr.values, y_tr, eval_set = [(X_va.values, y_va)], \n                  eval_metric = [\'auc\'], early_stopping_rounds=30, verbose=10)\n    try:\n        y_va_pred = model.decision_function(X_va.values) # HistGradientBoostingClassifier\n    except AttributeError:\n        try:\n            y_va_pred = model.predict_proba(X_va.values)[:,1] # XGBClassifier\n        except AttributeError:\n            y_va_pred = model.predict(X_va.values) # XGBRegressor\n    score = roc_auc_score(y_va, y_va_pred)\n    try:\n        print(f"Fold {fold}: n_iter ={model.n_iter_:5d}    AUC = {score:.3f}")\n    except AttributeError:\n        print(f"Fold {fold}:                  AUC = {score:.3f}")\n    score_list.append(score)\n    \nprint(f"OOF AUC:                       {np.mean(score_list):.3f}") # 0.944\n')


# # ROC curve
# 
# We plot the ROC curve just because it looks nice. The area under the red curve is the score of our model.

# In[8]:


# Plot the roc curve for the last fold
def plot_roc_curve(y_va, y_va_pred):
    plt.figure(figsize=(8, 8))
    fpr, tpr, _ = roc_curve(y_va, y_va_pred)
    plt.plot(fpr, tpr, color='r', lw=2)
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.gca().set_aspect('equal')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.show()

plot_roc_curve(y_va, y_va_pred)


# # Test predictions and submission
# 
# We create a submission file as follows:
# 1. We apply the same feature engineering to the test data as we did for the training data. Here it is important not to shuffle the test data so that the submission file is ordered correctly.
# 2. We retrain the `HistGradientBoostingClassifier` 100 times with different seeds on 95 % of the training data.
# 3. The decision functions of the 100 models can have different scales. To counter this, we convert the predictions to ranks using `scipy.stats.rankdata` and then submit the mean of the 100 ranks.

# In[9]:


# Feature engineering for test
test_pivoted0 = test.pivot(index=['sequence', 'subject'], columns='step', values=sensors)
test_pivoted = engineer(test_pivoted0)
sequence_count = test_pivoted.index.to_frame(index=False).groupby('subject').size().rename('sequence_count_of_subject')
#display(test_pivoted.head(2))
submission = pd.DataFrame({'sequence': test_pivoted.index.get_level_values('sequence')})
test_pivoted = test_pivoted.merge(sequence_count, how='left', on='subject')
test_pivoted.head(2)


# In[10]:


# Retrain, predict and write submission
print(f"{len(selected_columns)} features")

pred_list = []
for seed in range(100):
    X_tr = train_shuffled[selected_columns]
    y_tr = labels_shuffled.state

    model = HistGradientBoostingClassifier(learning_rate=0.05, max_leaf_nodes=25,
                                           max_iter=1000, min_samples_leaf=500,
                                           validation_fraction=0.05,
                                           l2_regularization=1,
                                           max_bins=63,
                                           random_state=seed, verbose=0)
    model.fit(X_tr.values, y_tr)
    pred_list.append(scipy.stats.rankdata(model.decision_function(test_pivoted[selected_columns].values)))
    print(f"{seed:2}", pred_list[-1])
print()
submission['state'] = sum(pred_list) / len(pred_list)
submission.to_csv('submission.csv', index=False)
submission


# In[ ]:




