#!/usr/bin/env python
# coding: utf-8

# # A Zoo of Models for Media Campaign Cost Prediction (PSS3E11)
# 
# This notebook shows how to create a strong and diverse ensemble for Episode 11 of the Playground Series.
# 
# The main points of the solution are:
# 
# 1. We create a zoo of 18 different, optimized models.
# 1. We make all models predict a transformed target, i.e. log1p(cost), so that we can use mean squared error as loss function. At the end of the notebook, we'll submit expm1(pred).
# 1. We use only a subset of the features (feature selection).
# 1. We add the original data to the training data.
# 1. We reduce training time massively by grouping the duplicates in the training data.
# 1. In some models, we treat the seemingly numerical `store_sqft` as categorical.
# 1. The final submission is a blend of 18 single models.
# 
# ## Structure of the notebook
# 
# The notebook doesn't start with a full EDA - you've seen enough of them. I'll discuss only two topics which have been somewhat neglected in the public EDAs: The duplicates and the categorical features. After this initial discussion, we'll have a conventional section (scikit-learn models and gradient boosting) and then a neural network section (Keras). At the end, we'll analyze the diversity of the resulting ensemble.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import os, gc, pickle, datetime, lightgbm, math, catboost, xgboost, warnings
from scipy.cluster.hierarchy import dendrogram
from category_encoders.target_encoder import TargetEncoder
from colorama import Fore, Back, Style

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, EarlyStopping
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.utils import plot_model
import keras_tuner

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from sklearn.model_selection import RepeatedKFold, KFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PolynomialFeatures, FunctionTransformer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression, LassoCV, LogisticRegression, SGDClassifier
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_log_error, mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from sklearn.inspection import PartialDependenceDisplay
from sklearn.cluster import AgglomerativeClustering

np.set_printoptions(linewidth=150, edgeitems=5)


# In[2]:


result_list = []
train = pd.read_csv('/kaggle/input/playground-series-s3e11/train.csv', index_col='id')
test = pd.read_csv('/kaggle/input/playground-series-s3e11/test.csv', index_col='id')
original = pd.read_csv('/kaggle/input/media-campaign-cost-prediction/train_dataset.csv')

print(f"Length of train:          {len(train)}")
print(f"Length of test:           {len(test)}")
print(f"Length of original_train: {len(original)}")
print()

print('Sample data from train:')
train.tail(3)


# # Feature engineering
# 
# We don't need much feature engineering: We just transform the target so that we can optimize for RMSE rather than RMSLE, and we merge the two almost identical features `salad_bar` and `prepared_food`. 

# In[3]:


for df in [train, original]:
    df['log_cost'] = np.log1p(df['cost'])
target = 'log_cost'
    
for df in [train, test, original]:
    df['salad'] = (df['salad_bar'] + df['prepared_food']) / 2


# # Feature selection
# 
# Throughout the notebook, we work with two subsets of the features:
# - `most_important_features` is the list of the eight most important features.
# - `relevant_features` is a nine-element list which additionally contains `unit_sales(in millions)`.
# 
# It turns out that best models use only the eight most important features, but the ensemble improves if a few nine-feature models are included.

# In[4]:


# Selection of eight features
most_important_features = ['total_children', 'num_children_at_home',
                           'avg_cars_at home(approx).1', 'store_sqft',
                           'coffee_bar', 'video_store', 'salad', 
                           'florist']

# Selection of nine features
relevant_features = ['unit_sales(in millions)'] + most_important_features


# # EDA part 1: Duplicates and the grouping trick
# 
# We know that in our dataset, not all features are relevant, and the relevant features have only very few unique values:

# In[5]:


print("Unique values:")
for f in relevant_features:
    print(f"{f:26} {np.unique(train[f])}")


# With a 360000-row dataset and so few unique feature values, we may expect quite some duplicated rows. Indeed, if we group the training data by the eight most important features, we get many groups with >1000 members:

# In[6]:


tg = train.groupby(most_important_features).log_cost.agg(['mean', 'std', 'count']).sort_values('count')
print(f"There are {len(tg)} groups.")
tg.tail(20)


# The large groups have means between 4.2 and 4.8 and standard deviations around 0.3. The small groups can have any mean and any standard deviation.

# In[7]:


# Mean vs. group size
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.scatter(tg['mean'], tg['count'], s=1, c=tg['std'], cmap='copper')
# low std = black, high std = copper
plt.xlabel('mean')
plt.ylabel('group size')

# Mean vs. standard deviation
plt.subplot(1, 3, 2)
# small = cyan, large = magenta
plt.scatter(tg['mean'], tg['std'], s=1, c=np.log(tg['count']), cmap='cool')
plt.xlabel('mean')
plt.ylabel('std')

# Group size vs. standard deviation
plt.subplot(1, 3, 3)
plt.scatter(tg['count'], tg['std'], s=1, c='k', cmap='cool')
plt.xlabel('group size')
plt.ylabel('std')
plt.show()


# There are many small groups and fewer large ones:

# In[8]:


plt.figure(figsize=(12, 3))
plt.subplot(1, 2, 1)
plt.title('train')
tg['count'].hist(bins=50)
plt.xlabel('group size')
plt.ylabel('count')
plt.subplot(1, 2, 2)
plt.title('both')
both = pd.concat([train, test], axis=0)
bg = both.groupby(most_important_features).total_children.agg(['count']).sort_values('count')
bg['count'].hist(bins=50)
plt.xlabel('group size')
plt.ylabel('count')
plt.show()


# **Insight: We can save a lot of training (and development) time if we group the training samples and present only the 3075 groups to the `fit()` functions rather than 360'336 individual samples.** Of course this trick only works with models that accept a `sample_weight` parameter in their `fit()` function.

# # EDA part 2: `store_sqft` is a categorical feature
# 
# If we train a simple random forest model and plot the partial dependence for the most important feature, store_sqft, we see a strange zig-zag line without a clear trend. Do you remember that store_sqft has only 20 unique values? The feature has only 20 unique values because the original data was gathered from 20 stores. The important insight now is that the target doesn't depend on the store size! The size of the store doesn't matter. The target depends on the store, but not on its size. We could as well number the stores from 1 to 20 or use their locations as feature because **the store (size) is a categorical feature**. 
# 
# If we accept that store_sqft is a categorical feature, we should treat it as such and for instance **try encoding it with OneHotEncoder or TargetEncoder**.
# 
# Note that treating the store as a categorical feature prevents us from making predictions for unseen stores, but luckily the test dataset stores are the same as the training stores.

# In[9]:


get_ipython().run_cell_magic('time', '', 'model = RandomForestRegressor(bootstrap=False, max_features=5, n_estimators=100, min_weight_fraction_leaf=3/288268, random_state=1)\nmodel.fit(train[most_important_features], train[\'log_cost\'])\n\nplt.figure(figsize=(6, 4))\nplt.suptitle(\'Partial Dependence\', y=1.0)\nPartialDependenceDisplay.from_estimator(model, train[most_important_features].sample(300),\n                                        [\'store_sqft\'],\n                                        percentiles=(0, 1),\n                                        pd_line_kw={"color": "blue"},\n                                        ice_lines_kw={"color": "lightblue"},\n                                        kind=\'both\',\n                                        ax=plt.gca())\nplt.show()\n')


# # Cross-validation
# 
# We define a function `score_model()`, which we'll use to cross-validate every model. In this function, you can see how the training samples are grouped to reduce the training time.

# In[10]:


def fit_model_grouped(model, train, features_used):
    """Group the duplicates in train and fit the model with the correct sample_weight"""
    train_grouped = train.groupby(features_used).log_cost.agg(['mean', 'count']).reset_index()
    X_tr = train_grouped[features_used]
    y_tr = train_grouped['mean']
    sample_weight_tr = train_grouped['count']
    if type(model) == Pipeline:
        sample_weight_name = model.steps[-1][0] + '__sample_weight'
    else:
        sample_weight_name = 'sample_weight'
    model.fit(X_tr, y_tr, **{sample_weight_name: sample_weight_tr})

def score_model(model, features_used, label=None, use_original=False):
    """Cross-validate a model"""
    start_time = datetime.datetime.now()
    score_list = []

    use_original = True

    oof = np.zeros_like(train[target], dtype=float)
    kf = KFold(shuffle=True, random_state=10)
    for fold, (idx_tr, idx_va) in enumerate(kf.split(train)):
        X_va = train.iloc[idx_va][features_used]
        y_va = train.iloc[idx_va][target]

        if use_original:
            fit_model_grouped(model, pd.concat([train.iloc[idx_tr], original], axis=0), features_used)
        else:
            fit_model_grouped(model, train.iloc[idx_tr], features_used)
            
        y_va_pred = model.predict(X_va)
        rmse = mean_squared_error(y_va, y_va_pred, squared=False)
        print(f"Fold {fold}: rmse = {rmse:.4f}")
        oof[idx_va] = y_va_pred
        score_list.append(rmse)

    rmse = sum(score_list) / len(score_list)
    execution_time = datetime.datetime.now() - start_time
    print(f"{Fore.GREEN}{Style.BRIGHT}Average rmse: {rmse:.5f} {label if label is not None else ''}{Style.RESET_ALL}")
    if label is not None:
        global result_list
        result_list.append((label, model, features_used, rmse, oof, execution_time, use_original))
        


# # Ridge regression
# 
# We start with two ridge regression models. Because all features are categorical, we one-hot encode them; and because the target depends on many feature interactions, we create interactions with the `PolynomialFeatures` transformer. 4th-degree polynomials give a better score than 3rd-degree polynomials:

# In[11]:


get_ipython().run_cell_magic('time', '', "model = make_pipeline(ColumnTransformer([('ohe', OneHotEncoder(drop='first'), \n                                          ['total_children',\n                                           'num_children_at_home', 'avg_cars_at home(approx).1',\n                                           'store_sqft'])],\n                                        remainder='passthrough'),\n                      PolynomialFeatures(3, interaction_only=True, include_bias=False),\n                      Ridge())\nscore_model(model, relevant_features, label=f'Onehot-Poly3-Ridge with unit_sales')\n")


# In[12]:


get_ipython().run_cell_magic('time', '', "model = make_pipeline(ColumnTransformer([('ohe', OneHotEncoder(drop='first'), \n                                          ['total_children',\n                                           'num_children_at_home', 'avg_cars_at home(approx).1',\n                                           'store_sqft'])],\n                                        remainder='passthrough'),\n                      PolynomialFeatures(4, interaction_only=True, include_bias=False),\n                      Ridge())\nscore_model(model, most_important_features, label=f'Onehot-Poly4-Ridge')\n")


# # Tree models
# 
# We start the tree model section with three random forests. They all use the same eight features, but the categorical store_sqft is either used as-is or one-hot encoded or target encoded:

# In[13]:


get_ipython().run_cell_magic('time', '', "score_model(RandomForestRegressor(bootstrap=False, max_features=5, n_estimators=400, min_weight_fraction_leaf=4.5/360336, random_state=44),\n            most_important_features, label=f'RF')\n")


# In[14]:


get_ipython().run_cell_magic('time', '', "score_model(make_pipeline(ColumnTransformer([('ohe', OneHotEncoder(drop='first', sparse='False'), ['store_sqft'])],\n                                            remainder='passthrough'),\n                          RandomForestRegressor(bootstrap=False, max_features=19, n_estimators=400,\n                                                min_weight_fraction_leaf=4.5/360336, random_state=44)\n                         ),\n            most_important_features, label=f'Onehot-RF')\n")


# In[15]:


get_ipython().run_cell_magic('time', '', "score_model(make_pipeline(TargetEncoder(verbose=30, cols=['store_sqft'], handle_unknown='error'),\n                          RandomForestRegressor(bootstrap=False, max_features=6, n_estimators=400,\n                                                min_weight_fraction_leaf=4.5/360336, random_state=44)\n                         ),\n            most_important_features, label=f'Target-RF')\n")


# We continue with three `ExtraTreesRegressor` for different feature subsets:

# In[16]:


get_ipython().run_cell_magic('time', '', "score_model(ExtraTreesRegressor(bootstrap=False, max_features=7, n_estimators=400, min_weight_fraction_leaf=4.5/360336, random_state=22),\n            most_important_features, label=f'ET')\n")


# In[17]:


get_ipython().run_cell_magic('time', '', "score_model(ExtraTreesRegressor(bootstrap=False, max_features=7, n_estimators=400, min_weight_fraction_leaf=4.5/360336, random_state=22),\n            most_important_features + ['unit_sales(in millions)'], label=f'ET with unit_sales')\n")


# In[18]:


get_ipython().run_cell_magic('time', '', "score_model(ExtraTreesRegressor(bootstrap=False, max_features=7, n_estimators=400, min_weight_fraction_leaf=4.5/360336, random_state=22),\n            most_important_features + ['store_sales(in millions)'], label=f'ET with store_sales')\n")


# In[19]:


get_ipython().run_cell_magic('time', '', "score_model(make_pipeline(TargetEncoder(verbose=30, cols=['store_sqft'], handle_unknown='error'),\n                          ExtraTreesRegressor(bootstrap=False, max_features=8, n_estimators=400,\n                                              min_weight_fraction_leaf=4.5/360336, random_state=22)\n                         ),\n            most_important_features, label=f'Target-ET')\n")


# With `HistGradientBoostingRegressor` we can use store_sqft either as-is, marked as a categorical feature or target-encoded:

# In[20]:


get_ipython().run_cell_magic('time', '', "score_model(HistGradientBoostingRegressor(max_iter=320, max_leaf_nodes=128, min_samples_leaf=2),\n        most_important_features, label=f'HGB A')\n")


# In[21]:


get_ipython().run_cell_magic('time', '', "score_model(make_pipeline(ColumnTransformer([('oe', OrdinalEncoder(), \n                                              ['store_sqft'])],\n                                            remainder='passthrough'),\n                          HistGradientBoostingRegressor(max_iter=320, max_leaf_nodes=128, min_samples_leaf=2,\n                                                        categorical_features=[0])),\n            most_important_features, label=f'HGB B')\n")


# In[22]:


get_ipython().run_cell_magic('time', '', "score_model(make_pipeline(TargetEncoder(cols=['store_sqft'], handle_unknown='error'),\n                          HistGradientBoostingRegressor(max_iter=320, max_leaf_nodes=128, min_samples_leaf=2)\n                         ),\n            most_important_features, label=f'Target-HGB')\n")


# To finish this section of the notebook, we train CatBoost in two variants, LightGBM, LightGBM-Dart and XGBoost:

# In[23]:


get_ipython().run_cell_magic('time', '', 'cb_params = {\'n_estimators\': 4000,\n             \'max_depth\': 12,\n             \'learning_rate\': 0.1,\n             \'verbose\': False,\n             \'boost_from_average\': True,\n            }\n\nscore_model(catboost.CatBoostRegressor(**cb_params),\n            most_important_features, f"CatBoost")\n')


# In[24]:


get_ipython().run_cell_magic('time', '', 'cb_params = {\'n_estimators\': 1500,\n             \'max_depth\': 12,\n             \'learning_rate\': 0.1,\n             \'verbose\': False,\n             \'boost_from_average\': True,\n            }\n\nscore_model(catboost.CatBoostRegressor(**cb_params),\n            relevant_features, f"CatBoost with unit_sales")\n    \n')


# In[25]:


get_ipython().run_cell_magic('time', '', 'lgbm_params = {\n    \'learning_rate\': 0.1,\n    \'n_estimators\': 450,\n    \'num_leaves\': 100,\n    \'min_child_samples\': 1,\n    \'min_child_weight\': 1e1,\n    \'categorical_feature\': [most_important_features.index(\'store_sqft\')],\n    \'random_state\': 1\n}\nscore_model(lightgbm.LGBMRegressor(**lgbm_params),\n            most_important_features, f"LightGBM")\n')


# In[26]:


get_ipython().run_cell_magic('time', '', 'dart_params = {\'boosting_type\': \'dart\',\n               \'learning_rate\': 0.3,\n               \'n_estimators\': 400,\n               \'num_leaves\': 200,\n               \'min_child_samples\': 1,\n               \'min_child_weight\': 10,\n              }\nscore_model(lightgbm.LGBMRegressor(**dart_params),\n            most_important_features, f"Dart")\n')


# In[27]:


get_ipython().run_cell_magic('time', '', 'xgb_params = {\'n_estimators\': 280,\n              \'learning_rate\': 0.05,\n              \'max_depth\': 10,\n              \'subsample\': 1.0,\n              \'colsample_bytree\': 1.0,\n              \'tree_method\': \'hist\',\n              \'enable_categorical\': True,\n              \'verbosity\': 1,\n              \'min_child_weight\': 3,\n              \'base_score\': 4.6,\n              \'random_state\': 1}\n\ndef cat_store_sqft(df):\n    df = df.copy()\n    df[\'store_sqft\'] = df[\'store_sqft\'].astype(\'category\')\n    return df\n\nmodel = make_pipeline(FunctionTransformer(cat_store_sqft), xgboost.XGBRegressor(**xgb_params))\nscore_model(model, most_important_features, label=f"XGBoost")\n')


# # Preprocessing for the neural network
# 
# 1. The neural network uses the same eight features as most other models.
# 1. We treat all features as categorical and one-hot-encode them.
# 

# In[28]:


nn_features = most_important_features

preprocessor = make_pipeline(ColumnTransformer([('ohe',
                                                 OneHotEncoder(drop='first', sparse=False),
                                                 ['total_children', 'num_children_at_home',
                                                  'avg_cars_at home(approx).1', 'store_sqft'])],
                                               remainder='passthrough'
                                              ),
                             StandardScaler())

X = preprocessor.fit_transform(train[nn_features]).astype(float)
y = train[target]
X_te = preprocessor.transform(test[nn_features]).astype(float)
X_or = preprocessor.transform(original[nn_features]).astype(float)
y_or = original[target]


# # The neural network
# 
# The neural network has a sequential architecture with four hidden layers having 256, 128, 64 and 64 neurons. Of all tested activation functions, relu gave the best result.

# In[29]:


LR_START = 1/128
BATCH_SIZE = 1024

best_hp = keras_tuner.HyperParameters()
best_hp.values = {'reg1': 8e-6,
                  'reg2': 2e-6,
                  'units1': 256,
                  'units2': 128,
                  'units3': 64,
                  'units4': 64,
                 }
    
def my_model(hp=best_hp, n_inputs=X.shape[1]):
    """Sequential neural network
    
    Returns a compiled instance of tensorflow.keras.models.Model.
    """
    activation = 'relu'
    reg1 = hp.Float("reg1", min_value=1e-8, max_value=1e-4, sampling="log")
    reg2 = hp.Float("reg2", min_value=1e-10, max_value=1e-5, sampling="log")
    
    inputs = Input(shape=(n_inputs, ))
    x0 = Dense(hp.Choice('units1', [64, 128, 256]), kernel_regularizer=tf.keras.regularizers.l2(reg1),
              activation=activation,
             )(inputs)
    x1 = Dense(hp.Choice('units2', [64, 128, 256]), kernel_regularizer=tf.keras.regularizers.l2(reg1),
              activation=activation,
             )(x0)
    x2 = Dense(hp.Choice('units3', [32, 64, 128, 256]), kernel_regularizer=tf.keras.regularizers.l2(reg1),
              activation=activation,
             )(x1)
    x3 = Dense(hp.Choice('units4', [32, 64, 128, 256]), kernel_regularizer=tf.keras.regularizers.l2(reg1),
              activation=activation,
             )(x2)
    x = Dense(1, kernel_regularizer=tf.keras.regularizers.l2(reg2),
             )(x3)
    regressor = Model(inputs, x)
    regressor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR_START),
#                       metrics=[tf.keras.metrics.RootMeanSquaredError()],
                      loss=tf.keras.losses.MeanSquaredError(),
                      steps_per_execution=32
                     )
    
    return regressor

display(plot_model(my_model(), show_layer_names=False, show_shapes=True, dpi=72))


# # Neural network cross-validation
# 
# The following loop trains the network twenty times and cross-validates it. When training makes no progress, we reduce the learning rate, and we apply early stopping.

# In[30]:


get_ipython().run_cell_magic('time', '', '# Cross-validation\nVERBOSE = 0 # set to 2 for more output, set to 0 for less output\nEPOCHS = 1000\nN_SPLITS = 10\nN_REPEATS = 2\n\nnp.random.seed(2)\ntf.random.set_seed(2)\n\nstart_time = datetime.datetime.now()\noof_preds = np.zeros(len(train))\ntest_preds = np.zeros(len(test))\nkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=1)\nscore_list = []\nfor fold, (idx_tr, idx_va) in enumerate(kf.split(train)):\n    model = None\n    gc.collect()\n    X_tr = X[idx_tr]\n    y_tr = y[idx_tr]\n    X_va = X[idx_va]\n    y_va = y[idx_va]\n\n    # Add the original data for training\n    X_tr = np.vstack([X_tr, X_or])\n    y_tr = pd.concat([y_tr, y_or])\n    \n    # Define the callbacks\n    lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, \n                           patience=4, verbose=VERBOSE)\n    es = EarlyStopping(monitor="val_loss",\n                       patience=12,\n                       min_delta=0.00002,\n                       verbose=0,\n                       mode="min", \n                       restore_best_weights=True)\n    callbacks = [lr, es, tf.keras.callbacks.TerminateOnNaN()]\n\n    # Construct and compile the model\n    model = my_model(best_hp, X_tr.shape[1])\n\n    # Train the model\n    history = model.fit(X_tr, y_tr, \n                        validation_data=(X_va, y_va), \n                        epochs=EPOCHS,\n                        verbose=VERBOSE,\n                        batch_size=BATCH_SIZE,\n                        shuffle=True,\n                        callbacks=callbacks)\n    del X_tr, y_tr\n    test_preds += model.predict(X_te, batch_size=65536, verbose=VERBOSE).ravel()\n    history = history.history\n    callbacks, lr = None, None\n    \n    # Validate the model\n    y_va_pred = model.predict(X_va, batch_size=len(X_va), verbose=VERBOSE).ravel()\n    rmse = mean_squared_error(y_va, y_va_pred, squared=False)\n    oof_preds[idx_va] += y_va_pred\n\n    print(f"{Fore.BLUE}{Style.BRIGHT}Fold {fold}: {es.stopped_epoch:3} epochs,"\n          f" rmse = {rmse:.4f}{Style.RESET_ALL}")\n    del es, X_va #, y_va, y_va_pred\n    score_list.append((None, rmse))\n\n# Save oof and test predictions\nassert np.isfinite(oof_preds).all()\noof_preds /= N_REPEATS\nwith open("oof_keras.pickle", \'wb\') as f: pickle.dump(oof_preds, f)\nkeras_preds = test_preds / N_SPLITS / N_REPEATS\n    \n# Show overall score\nscore_df = pd.DataFrame(score_list, columns=[\'none\', \'rmse\'])\nrmse = score_df[\'rmse\'].mean()\nprint(f"{Fore.GREEN}{Style.BRIGHT}Average  rmse = {rmse:.5f}{Style.RESET_ALL}")\nexecution_time = datetime.datetime.now() - start_time\nresult_list.append((\'Keras\', None, nn_features, rmse, oof_preds, execution_time, True))\n')


# # Final evaluation
# 
# Let's try to analyze the diversity of our model zoo. We can compute a distance matrix of the distances between (the oof predictions of) all models. The rows of the matrix are ordered by score, i.e. the best model is in the first row and the worst one in the last row. The biggest distances obviously occur between the best and the worst model. It is interesting, however, that the Keras model has a bigger distance (darker color) to the best model than the model below it. This means that the Keras model brings valuable diversity to the ensemble.
# 
# 

# In[31]:


# Create a dataframe of all results, ordered by score (best model comes first)
result_df = pd.DataFrame(result_list, columns=['label', 'model', 'features', 'rmse', 'oof', 'execution_time', 'use_original'])
result_df.drop_duplicates(subset='label', keep='last', inplace=True)
result_df = result_df[~result_df.label.str.contains('Mean')]
result_df.sort_values('rmse', inplace=True)
result_df.reset_index(drop=True, inplace=True)

# Display the distances between oof predictions
oof = np.column_stack(list(result_df.oof))
distances = euclidean_distances(oof.T, oof.T)
plt.figure(figsize=(10, 10))
plt.title('Distance between oof predictions')
sns.heatmap(distances, linewidth=0.1, fmt='.1f', 
            annot=True, annot_kws={'size': 8}, 
            cmap='YlOrRd',
            xticklabels=result_df.label,
            yticklabels=result_df.label,
           )
plt.show()

# Evaluate several unweighted blends and add them to the result list
r = result_df.set_index('label')
for i in range(2, len(r)+1):
    oof = np.column_stack(list(r.oof.iloc[:i]))
    oof = oof.mean(axis=1)
    rmse = mean_squared_error(train['log_cost'], oof, squared=False)
    result_list.append((f"Mean of {i}", None, None, rmse, oof, datetime.timedelta(), 'mean'))


# We now use ridge regression to compute the optimal weights for a weighted blend of all models. For simplicity, we don't cross-validate the regression but simply report its training score. The weights shown aren't very reliable, but we see that Keras and CatBoost with unit_sales should have higher weights than the other models.

# In[32]:


# Weighted blend (ridge regression)
oof = np.column_stack(list(r.oof))
optimum_blend = Ridge(positive=True, tol=1e-6, alpha=100)
optimum_blend.fit(oof, train.log_cost.values)
optimum_oof = optimum_blend.predict(oof)
trmse = mean_squared_error(train.log_cost.values, optimum_oof, squared=False)
print(f'Ridge regression blend training rmse: {trmse:.5f}')
display(pd.Series(optimum_blend.coef_.round(2), r.index, name='weight'))
result_list.append((f"Optimum blend", None, None, trmse, optimum_oof, datetime.timedelta(), 'mean'))


# The dendrogram gives another view onto the model zoo. The dendrogram represents a hierachical clustering of the models. Similar models end up in the same cluster. We can easily identify a cluster of gradient-boosting models - their predictions resemble each other. Another cluster consists of the models which were trained with the additional unit_sales feature:

# In[33]:


def plot_dendrogram(model, **kwargs):
    """Create linkage matrix and plot the dendrogram
    
    From https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
    """

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, color_threshold=0.6, orientation='left', **kwargs)
    plt.xticks(plt.xticks()[0], (np.arccos(1 - plt.xticks()[0]) / math.pi * 180).round(0).astype(int))


oof = np.column_stack(list(r.oof))

# Setting distance_threshold=0 ensures we compute the full tree.
# We use a cosine similarity metric; other metrics are possible.
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None,
                                linkage='average', affinity='cosine')
model = model.fit(oof.T - oof.T.mean(axis=0))
plt.title("Hierarchical Clustering of Models")
plot_dendrogram(model, labels=r.index)
plt.show()


# To end the evaluation, we show the scores and execution times of all models in two bar charts. The bar chart shows
# - all 18 single models,
# - unweighted blends of the k best single models for k between 2 and 18,
# - a weighted blend.
# 
# Some observations:
# 1. The ensemble score increases most when the 2nd model is added, when the 5th model (random forest) is added, with the 13th model (Keras), and with the 15th model (the first model based on the unit_sales feature). 
# 1. Diversity in the ensemble is better than a monoculture of gradient boosters.
# 1. The twenty Keras training runs take much more time than all other models together.
# 1. The optimally weighted blend isn't much better than a simple average of all models.

# In[34]:


# Recreate the result dataframe
result_df = pd.DataFrame(result_list, columns=['label', 'model', 'features', 'rmse', 'oof', 'execution_time', 'use_original'])
result_df.drop_duplicates(subset='label', keep='last', inplace=True)
result_df.sort_values('rmse', inplace=True)
result_df.reset_index(drop=True, inplace=True)

# Plot the scores as horizontal bar chart
plt.figure(figsize=(12, len(result_df) * 0.3 + 1))
plt.suptitle('Final comparison', fontsize=20)
plt.subplot(1, 2, 1)
color = result_df.use_original.map({True: 'orange', False: 'yellow', 'mean': 'brown'})
bars = plt.barh(result_df.index, result_df.rmse, color=color)
plt.gca().bar_label(bars, fmt='%.5f')
plt.gca().invert_yaxis()
plt.yticks(np.arange(len(result_df)), result_df.label)
plt.xlabel('RMSLE')
plt.xlim(0.292, 0.295)

# Plot the execution times as horizontal bar chart
plt.subplot(1, 2, 2)
bars = plt.barh(result_df.index, result_df.execution_time.dt.seconds, color='olive')
plt.gca().invert_yaxis()
plt.yticks(np.arange(len(result_df)), result_df.label)
plt.xlabel('Execution time (s)')
plt.xlim(0, 1000)
plt.tight_layout(w_pad=1)
plt.show()


# # Retraining and submission
# 
# We retrain the models on the full dataset and create the submission files. As a plausibility check, we plot histograms of the predictions. If these histograms were completely different from everybody else's histograms, we'd know that the code contains a bug somewhere.

# In[35]:


get_ipython().run_cell_magic('time', '', 'def submit(test_preds, path):\n    """Write expm1(test_preds) to a csv file at path."""\n    assert np.isfinite(test_preds).all()\n    submission = pd.Series(np.expm1(test_preds), index=test.index, name=\'cost\')\n    submission.to_csv(path)\n    with np.printoptions(precision=1, suppress=True):\n        print(submission.values[:10])\n    plt.figure(figsize=(10, 3))\n    plt.title(path, fontsize=20)\n    plt.hist(submission, bins=50)\n    plt.show()\n\ntest_preds = np.zeros((len(test), ), dtype=float)\nr[\'test_pred\'] = None\nfor i in range(len(r)):\n    print(f"Retraining {i:2} {r.index[i]} {\'with original data\' if r.iloc[i].use_original else \'\'}")\n    if r.index[i] != \'Keras\':\n        features = r.iloc[i].features\n        if r.iloc[i].use_original:\n            fit_model_grouped(r.iloc[i].model, pd.concat([train, original], axis=0), features)\n        else:\n            fit_model_grouped(r.iloc[i].model, train, features)\n        preds = r.iloc[i].model.predict(test[features])\n    else:\n        preds = keras_preds\n    r.at[r.index[i], \'test_pred\'] = preds\n    test_preds += preds\n    if i > 0:\n        submit(test_preds / (i+1), f\'submission_{i+1}.csv\')\n        \n# Optimum blend\ntest_preds = np.column_stack(list(r.test_pred))\ntest_preds = optimum_blend.predict(test_preds)\nsubmit(test_preds, \'submission_weighted.csv\')\n')

