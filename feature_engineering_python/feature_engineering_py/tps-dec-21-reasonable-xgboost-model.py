#!/usr/bin/env python
# coding: utf-8

# A reasonable XGBoost model
# ==
# 
# In this competition, I feel like like a lot of energy has been spent developing neural networks. My experince has been that I am getting more impressive results with gradient boosters, but the public notebooks that submit boosters don't have amazing results.
# 
# The disadvantage of boosters on this competition is that they're fairly slow. This notebook does not contain the best booster I've trained, but this one is a pretty good tradeoff between accuracy and performance, this notebook doesn't take too long to run. But you must run it with a GPU, otherwise it'll crash. On my gaming computer, the training loop itself will take 12-13 minutes here with the 5 folds used here, and it's only a little bit slower on kaggle. The lightgbm model I have that is better, takes around 1 hour on a 12-core CPU, so it's not fun to play around with it in kaggle notebooks, which only have 4 vCPU.
# 
# The booster parameters here are by *no means* optimal. I've been trying to spend a little energy doing hyper-parameter searches with optuna, but I've discovered that beyond some point, improved CV performance stops correlating with public LB performance. I'm trying to dig in to what that means. In the meantime, here's a booster that that's competetive with neural networks.

# In[1]:


import os
import gc
import random
import pandas as pd
import numpy as np
import seaborn as sns
import xgboost
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

sns.set(style='darkgrid', context='notebook', rc={'figure.figsize': (16, 12), 'figure.frameon': False})

np.random.seed(64)
random.seed(64)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=64)


# Next up, I'll load some data, using the pseudo-labeled examples from [this dataset](https://www.kaggle.com/remekkinas/tps12-pseudolabels) in addition. This boosts performance significantly for this booster. To me, it seems likely that it does so because the pseudo labels themselves have probably been primarily created by using NN outputs?
# 
# Loading data & pseudo labels
# ==
# 
# Anyway, let's load the datasets we'll use:
# 

# In[2]:


# This bit makes the notebook work on my local machine also
data_root = os.environ.get('KAGGLE_DIR', '../input')
train = pd.read_parquet(f'{data_root}/tpsdec2021parquet/train.pq')
pseudo = pd.read_csv(f'{data_root}/tps12-pseudolabels/tps12-pseudolabels_v2.csv', dtype=train.dtypes.to_dict())
test = pd.read_parquet(f'{data_root}/tpsdec2021parquet/test.pq')


# Next up, we'll put all of the data into one dataframe, so we can quickly do feature engineering on all the frames at the same time.
# 
# Feature engineering
# ==
# 
# Most of these are public in [this thread](https://www.kaggle.com/c/tabular-playground-series-dec-2021/discussion/293612). I believe I have only added a few, compared to those. These are identical to the features I've used to build my best performing model (public .95701), which is also a gradient booster using pseudo labels.
# 
# I also do not believe that we've found all the worthwhile features. Some of the ones I have are very similar -- eg. there are 4 variations of Aspect being included, yet I've found that performance drops if I exclude any of them.
# 

# In[3]:


all_df = pd.concat([train.assign(ds=0), pseudo.assign(ds=1), test.assign(ds=2)]).reset_index(drop=True).drop(
    columns=['Soil_Type7', 'Soil_Type15'] # 0 variance
)

def start_at_eps(series, eps=1e-10): return series - series.min() + eps

pos_h_hydrology = start_at_eps(all_df.Horizontal_Distance_To_Hydrology)
pos_v_hydrology = start_at_eps(all_df.Vertical_Distance_To_Hydrology)

wilderness = all_df.columns[all_df.columns.str.startswith('Wilderness')]
soil_type = all_df.columns[all_df.columns.str.startswith('Soil_Type')]
hillshade = all_df.columns[all_df.columns.str.startswith('Hillshade')]

all_df = pd.concat([
    all_df,

    all_df[wilderness].sum(axis=1).rename('Wilderness_Sum').astype(np.float32),
    all_df[soil_type].sum(axis=1).rename('Soil_Type_Sum').astype(np.float32),

    (all_df.Aspect % 360).rename('Aspect_mod_360'),
    (all_df.Aspect * np.pi / 180).apply(np.sin).rename('Aspect_sin').astype(np.float32),
    (all_df.Aspect - 180).where(all_df.Aspect + 180 > 360, all_df.Aspect + 180).rename('Aspect2'),

    (all_df.Elevation - all_df.Vertical_Distance_To_Hydrology).rename('Hydrology_Elevation'),
    all_df.Vertical_Distance_To_Hydrology.apply(np.sign).rename('Water_Vertical_Direction'),

    (pos_h_hydrology + pos_v_hydrology).rename('Manhatten_positive_hydrology').astype(np.float32),
    (all_df.Horizontal_Distance_To_Hydrology.abs() + all_df.Vertical_Distance_To_Hydrology.abs()).rename('Manhattan_abs_hydrology'),
    (pos_h_hydrology ** 2 + pos_v_hydrology ** 2).apply(np.sqrt).rename('Euclidean_positive_hydrology').astype(np.float32),
    (all_df.Horizontal_Distance_To_Hydrology ** 2 + all_df.Vertical_Distance_To_Hydrology ** 2).apply(np.sqrt).rename('Euclidean_hydrology'),

    all_df[hillshade].clip(lower=0, upper=255).add_suffix('_clipped'),
    all_df[hillshade].sum(axis=1).rename('Hillshade_sum'),

    (all_df.Horizontal_Distance_To_Roadways * all_df.Elevation).rename('road_m_elev'),
    (all_df.Vertical_Distance_To_Hydrology * all_df.Elevation).rename('vhydro_elevation'),
    (all_df.Elevation - all_df.Horizontal_Distance_To_Hydrology * .2).rename('elev_sub_.2_h_hydro').astype(np.float32),

    (all_df.Horizontal_Distance_To_Hydrology + all_df.Horizontal_Distance_To_Fire_Points).rename('h_hydro_p_fire'),
    (start_at_eps(all_df.Horizontal_Distance_To_Hydrology) + start_at_eps(all_df.Horizontal_Distance_To_Fire_Points)).rename('h_hydro_eps_p_fire').astype(np.float32),
    (all_df.Horizontal_Distance_To_Hydrology - all_df.Horizontal_Distance_To_Fire_Points).rename('h_hydro_s_fire'),
    (all_df.Horizontal_Distance_To_Hydrology + all_df.Horizontal_Distance_To_Roadways).abs().rename('abs_h_hydro_road'),
    (start_at_eps(all_df.Horizontal_Distance_To_Hydrology) + start_at_eps(all_df.Horizontal_Distance_To_Roadways)).rename('h_hydro_eps_p_road').astype(np.float32),

    (all_df.Horizontal_Distance_To_Fire_Points + all_df.Horizontal_Distance_To_Roadways).abs().rename('abs_h_fire_p_road'),
    (all_df.Horizontal_Distance_To_Fire_Points - all_df.Horizontal_Distance_To_Roadways).abs().rename('abs_h_fire_s_road'),
], axis=1)

types = {'Cover_Type': np.int8}
train = all_df.loc[all_df.ds == 0].astype(types).drop(columns=['ds'])
pseudo = all_df.loc[all_df.ds == 1].astype(types).drop(columns=['ds'])
test = all_df.loc[all_df.ds == 2].drop(columns=['Cover_Type', 'ds'])

del all_df
del pos_h_hydrology
del pos_v_hydrology
del wilderness
del soil_type
del hillshade

train.info()


# How to train the model?
# ==
# 
# At this point we iterate over our CV and record predictions and that's it. Right?
# 
# Well, actually, at this point we've got to make some choices. Because it turns out there are a number of ways to go about exploiting the pseudo labels. One way proceeding from here is to concatenate the train sets and pseudo sets on top of each other, then do folds over that. But I've actually had better performance when using the pseudo labels in _every fold_. How does that work? Well, I'm recording out of fold predictions / validations against only the original training set, but I'm letting each model see the pseudo labels. So let's do it like that, then.
# 
# For me, this has the bonus that the out of fold probabilites are easily compatible with my out of fold probabilities that did *not* use the pseudo labels, meaning it's easy for me to try blending this model with others.
# 

# In[4]:


label_encoder = LabelEncoder()

train = train.loc[train.Cover_Type != 5]

X = train.drop(columns=['Id', 'Cover_Type']).astype(np.float32).to_numpy()
y = label_encoder.fit_transform(train.Cover_Type)
feat_names = train.columns.drop(['Id', 'Cover_Type'])
del train
num_class = len(label_encoder.classes_)

X_pseudo = pseudo.drop(columns=['Id', 'Cover_Type']).astype(np.float32).to_numpy()
y_pseudo = label_encoder.transform(pseudo.Cover_Type)
del pseudo

X_test = test.drop(columns=['Id']).astype(np.float32).to_numpy()
del test

oof_proba = np.zeros((len(y), num_class), dtype=np.float32)
test_proba = np.zeros((len(X_test), num_class), dtype=np.float32)


# Most of the book-keeping setup for the loop over our folds is done now, so let's set up some xgboost parameters and write a little bit about what they mean.
# 
# XGBoost parameters review
# ==
# 

# In[5]:


params = {
    'num_class': num_class,
    'objective': 'multi:softprob',
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    'eval_metric': ['merror', 'mlogloss'],
    'learning_rate': .09,
    'max_depth': 0,
    'subsample': .15,
    'sampling_method': 'gradient_based',
    'seed': 64,
    'grow_policy': 'lossguide',
    'max_leaves': 255,
    'lambda': 100,
}


# We're setting this up as a `softprob` classifier, meaning we'll expect the booster to emit probabilities instead of classes. This is very useful to do soft voting ensembles, and it's also very useful to do analysis of which samples that are difficult and which ones that are not. I go for `softprob` for basically all multiclass classifiers with xgboost.
# 
# I've set xgboost up to use GPU here, if I can't use it with GPU, I'd rather use lightgbm. But the GPU enables a couple of interesting options that I always like _trying_ on tabular problems, and it makes the training speed of the booster competitive with options like neural nets and lightgbm.
# 
# As for the `'eval_metric'`, it's important to know that xgboost will only use the last metric for early stopping. When provided with multiple eval datasets, it'll use only the last data set. In this competition, I've found that I get better results when using logloss for early stopping than error, but I *tried both* and *you should too*.
# 
# The learning rate is the one xgboost parameter that is *always* worth playing around with. `.1` is not arbitrarily chosen, but I haven't done a big search to ensure it's a great value either. In general, the lower this number is, the more iterations you will need, and the higher the chances are that you will overfit. A low learning rate enables the booster to be very complex, so you may want to use strong regularization if you need a low learning rate.
# 
# `max_depth` is the maximum allowed depth of the decision trees the booster uses. Normally, a good value here is 3-11. But due to other options we're using here, we can safely allow an infinite `max_depth` (see discussion about `grow_policy` below), other limits will prevent xgboost from creating an infinitely deep tree.
# 
# `subsample` is set to .2 here, but we could probably go as low as .1. The meaning here is the fraction of the samples that will be used at each iteration of the booster. When using `sampling_method='gradient_based'`, you can set this really low, which is very helpful when playing with large-ish datasets like the one we have here. This is only possible with `tree_method='gpu_hist'`, though. It is useful to set this low, when you're able to -- it speeds up the training significantly to only look at ~10-20% of the data at each iteration.
# 
# I've set the seed for reproducible results, and `grow_policy='lossguide'`. What this does is that it'll make the trees grow leafwise instead of depth-wise, eg. it is no longer necessary to populate a whole level in the decision tree before descending further. This means we should set `max_leaves=255` to some value, here I've chosen 255, and remove the limit on `max_depth`, or set it really high.
# 
# `lambda` is for l2 regularization. We set it fairly high here, since `max_leaves` is high and our learning rate isn't super high. For this competition, I've found that it's important to regularize boosters to get good LB performance.
# 
# These settings I've chosen makes xgboost act a little bit like lightgbm, but since this runs on the GPU, it's quite fast, even on my GTX1660Ti on my laptop.
# 
# I treat these parameters as reasonable defaults, these are not params that I've tuned with a hyper parameter search and are by no means optimal. These are pretty close to what I'll try on the first time I'm seeing a new tabular problem. I'm certain that with some tinkering, we could get a better score. We probably have the most to gain by playing around with `learning_rate`, `lambda` and `max_leaves` *or* by "starting over" without `grow_policy='lossguide'` and `sampling_method='gradient_based'`.
# 
# If I were to introduce any new options, I might look at more regularization here, perhaps `gamma` and `colsample_bytree`. At that point, it might be necessary to run more iterations too, though.
# 
# Next, let's write a utility function that'll mix the pseudo labels into a train set and shuffle them, since we'll need that to train:

# In[6]:


def make_trainset(train_idx):
    ix = np.arange(len(train_idx) + len(y_pseudo))
    np.random.shuffle(ix)
    X_train = np.concatenate([X[train_idx], X_pseudo], axis=0)
    y_train = np.concatenate([y[train_idx], y_pseudo], axis=0)
    return xgboost.DMatrix(X_train[ix], label=y_train[ix])


# And we're ready to start training. With these settings, we can do a few hundred rounds of boosting per fold. I know this because I already ran this script, not because I'm a magician.
# 
# With xgboost, we need to remember to specify `iteration_range=` when predicting in order to use the best iteration of our booster. The other option is to predict with `booster[:booster.best_iteration]`. If you don't specify anything, it'll do predictions with all the iterations of the booster, in our case, maybe up to 30 iterations "past" where we wanted to predict from.
# 
# Let's go:

# In[7]:


get_ipython().run_cell_magic('time', '', "\nfor train_idx, val_idx in cv.split(X, y):\n    gc.collect()\n    dval = xgboost.DMatrix(X[val_idx], label=y[val_idx])\n    booster: xgboost.Booster = xgboost.train(\n        params=params,\n        dtrain=make_trainset(train_idx),\n        num_boost_round=600,\n        evals=[(dval, 'val')],\n        early_stopping_rounds=30,\n        verbose_eval=100,\n    )\n    oof_proba[val_idx] = booster.predict(\n        dval, iteration_range=(0, booster.best_iteration + 1)\n    )\n    test_proba += booster.predict(\n        xgboost.DMatrix(X_test), iteration_range=(0, booster.best_iteration + 1)\n    ) / cv.n_splits\n")


# Let's check if the last booster we trained liked any of our engineered features:
# 

# In[8]:


fscore = booster.get_fscore()
fscore = pd.DataFrame({'value': fscore.values()}, index=feat_names[np.arange(len(fscore))])
fscore['Wilderness_Sum':].plot.barh();


# In[9]:


fscore['Wilderness_Sum':]


# Not all of these are useful. Chances are pretty good that we haven't found the optimal features, and there are still more things we could try.
# 
# Let's check our out of fold performance:
# 

# In[10]:


y_pred = label_encoder.inverse_transform(oof_proba.argmax(axis=1))
y_true = label_encoder.inverse_transform(y)

print(classification_report(y_true, y_pred))


# I have been wondering, if perhaps I should drop the `Cover_Type=4` samples for some time. In half the cases where we predict that class, we're right, but we're only picking up 14% of them. This extra class costs a lot of extra training time, so we'd be able to iterate faster if we didn't have it -- and faster iteration enables more experimentation.

# In[11]:


accuracy_score(y_true, y_pred)


# .96276 is a reasonably good CV score. But at this level of CV score, I've found that the correlation to public LB is not fantastic. I've submitted many times with .9629 or better, but scored below .957 on public LB.
# 
# Let's check the confusion matrix, too. I will be zeroing out the diagonal because I don't think it's so interesting:
# 

# In[12]:


cm = confusion_matrix(y_true, y_pred)
ix = np.arange(cm.shape[0])
cm[ix, ix] = 0
col_names = [f'Cover_Type={cls}' for cls in label_encoder.classes_]
cm = pd.DataFrame(cm, columns=col_names, index=col_names)
cm


# In[13]:


sns.heatmap(cm, cmap='viridis', annot=True, fmt='d').set(title=f'{cm.sum().sum()} misses');


# I will be writing the out of fold probability predictions and the test probability predictions here, so that later, it is easy to try to soft-vote with this classifier:
# 

# In[14]:


pd.DataFrame(oof_proba, columns=col_names).to_parquet('reasonable_xgb_oof.pq')
pd.DataFrame(test_proba, columns=col_names).to_parquet('reasonable_xgb_test.pq')


# If you wanted to blend this classifier with another, you might want to just add the out of fold probabilities together first, to estimate the CV score. You might do that like this:
# 
# ```python
# import pandas as pd
# 
# xgb_proba = pd.read_parquet('../input/tps202112-reasonable-xgboost-model/reasonable_xgb_oof.pq').to_numpy()
# pred = (my_model_proba + xgb_proba).argmax(axis=1)
# ```
# 
# And now, let's submit:
# 

# In[15]:


y_pred = label_encoder.inverse_transform(test_proba.argmax(axis=1))
sub = pd.read_parquet(f'{data_root}/tpsdec2021parquet/test.pq', columns=['Id']).assign(Cover_Type=y_pred)
sub.head()


# In[16]:


sub.to_csv('submission.csv', index=False)


# # Thanks 

# The Original Notebook built Mr.@RKAVELAND I am very inspaire your notebook.

# In[ ]:




