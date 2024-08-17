#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
import multiprocessing as mp
from pathlib import Path

warnings.filterwarnings("ignore")
IN_KAGGLE_NOTEBOOK = False if mp.cpu_count() > 4 else True


# <div class="alert alert-warning">
# <b>🙋‍♂️ Platform Note:</b> The compute required for this feature selection process is large. Many things are computed off-Kaggle and read in as datasets. The flag <kbd>IN_KAGGLE_NOTEBOOK</kbd> acts as a switch for certain code. For example, on Kaggle, I cut down the dataset; off Kaggle I use the full dataset. The notebook is set up though so to show the off-Kaggle computed results.
# </div>

# # Feature Relevance based Unsupervised Feature Selection (FRUFS)
# 
# This notebook is based on a wonderful blog post published just a week ago [by the same name as above](https://www.deepwizai.com/projects/how-to-perform-unsupervised-feature-selection-using-supervised-algorithms) by the research collective DeepwizAI. The feature selection algorithm described in that post is great for the Ubiquant competition because in this competition the features are anonymized/obfuscated which makes feature engineering and manual feature selection difficult. FRUFS to the rescue!
# 
# First we install the package (https://github.com/atif-hassan/FRUFS).
# 

# In[2]:


get_ipython().run_cell_magic('capture', '', "\nif IN_KAGGLE_NOTEBOOK:\n    !pip install ../input/frufs-python-package/FRUFS-1.0.2-py3-none-any.whl\n    MODEL_PATH = Path('../input/frufs-ubiquant-models')\nelse:\n    !pip install FRUFS\n")


# So what is FRUFS? I'll do my best to summarize their excellent blog post in just a paragraph and a graphic I created.
# 
# (1) First thing...throw away the target! We don't need it (that's the **U** in FRUFS for unsupervised). (2) Next we iterate through each $j$ of the $m$ features. **We take a single feature $j$ as the target and try to predict it with a model $f$ using the remaining features.** In this model, the target is `X[sampled, j]`, and the features are `X[sampled, ~j]`. The "~j" means every feature not $j$. Since we need to iterate over all the (300) features, we sample the rows in **X** as well just to make fitting a little faster. We can use any model we like inside FRUFS; that's a nice point too: this method is **model agnostic**. (3) Then we take that fitted model for feature $j$ of $m$ and record the **feature importance** scores (again, the importance is model agnostic: for linear models we can use just the coefficients, in the case of gradient-boosted trees it could be split gain or split importance; more generally, for *any model*, it can be [LIME](https://github.com/marcotcr/lime), [SHAP](https://github.com/slundberg/shap) or [permutation importance](https://scikit-learn.org/stable/modules/permutation_importance.html), etc.!). The feature importances for the $m-1$ features tell you how redundant the information content of feature $j$ is. **The key idea in FRUFS is that a feature's skill in life is it's ability to predict other features!** If feature 1 is important to predict features 2, 3, and 4, then what do we need features 2, 3, and 4 for? Drop them. The final step (4) is to iterate through each feature $j$ of $m$ and average the feature importances across all the runs; we drop features that weren't important on average, keeping only $k$ features (this $k$ is a hyperparameter; if a float, it means the percentage to keep). This means we are dropping the most redundant information from the feature set which *should help generalization*. That's FRUFS. I made the following graphic to show what's going on. Our new feature matrix X', has $k$ columns (or in the case of a float value for $k$, has $k$ fraction of the original columns).
# 
# ![](https://i.imgur.com/w2Vt8lW.png)
# 
# Let's go!

# ## Imports

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
import joblib, gc
import lightgbm as lgb
import seaborn as sns

from sklearn.datasets import make_regression
from scipy.stats import pearsonr
from tqdm.notebook import trange, tqdm
from FRUFS import FRUFS
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# # Synthetic Data Example
# 
# Before we get into the Ubiquant data, I think it is helpful to see what FRUFS does on a dataset where we know the ground truth data generating process. Here I create a synthetic regression dataset and then run FRUFS on it. We know in advance which features are noise and which are redundant. Let' see if FRUFS can figure this out. I am using sklearn `make_regression`. All features are important but I set the matrix rank (`effective_rank`) of the data to be 2 while the number of features is 4. This means that the data will contain some redundant information. The `tail_strength` parameter is from 0 to 1 and is a dial to set how skewed the scree plot of the matrix would be (lower number means *more* skewed and thus *more* redundant information). I am making `tail_strength` as an exaggeration to be able to visually see the redudancy and hopefully understand what FRUFS is doing.

# In[4]:


X, y, true_coef = make_regression(
    n_samples=1_000,
    n_features=4,
    n_informative=4,
    n_targets=1,
    effective_rank=2,
    tail_strength=0.05,
    coef=True,
    random_state=17
)

def corrfunc(x, y, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    r, _ = pearsonr(x, y)
    ax = ax or plt.gca()
    ax.annotate(f'ρ = {r:.2f}', xy=(.1, .9), xycoords=ax.transAxes)


# In[5]:


sns.set(rc={'figure.figsize':(10, 10)})
sns.set(font_scale=1.1)
sns.set_style("whitegrid", {'axes.grid' : False})


g = sns.pairplot(
    pd.DataFrame(data=X).sample(frac=0.25),
    corner=True,
    kind='reg',
    plot_kws={
        'line_kws':{'color':'olive'},
        'scatter_kws': {'alpha': 0.2}
})
g.map_lower(corrfunc)
g.set(xticklabels=[])
g.set(yticklabels=[])
g.fig.suptitle('Synthetic Features vs Each Other');


# We see that feature 0 and feature 3 are highly redundant. Feature 0 is also a bit correlated with feature 2 and 1. I expect FRUFS will throw away either feature 0 or 3. It is important to note that per the `make_regression`, **all** the features are predictive of the target.

# In[6]:


model_frufs_generated = FRUFS(
        model_r=lgb.LGBMRegressor(random_state=42),
        k=3
    )
model_frufs_generated.fit(X)


# In[7]:


model_frufs_generated.feature_importance()


# That's encouraging! In this spoofed situation, the model recovers what we expect it to. Back to Ubiquant...

# In[8]:


if IN_KAGGLE_NOTEBOOK:
    train_cut = 600
    sample_frac = 0.025
else:
    train_cut = 0
    sample_frac = 0.20


# ## Load the data

# In[9]:


get_ipython().run_cell_magic('time', '', "\ntrain = (pd.read_parquet('../input/ubiquant-parquet/train_low_mem.parquet')\n           .sort_values(['time_id', 'investment_id'])\n           .query('time_id > @train_cut')\n           .drop(columns=['row_id', 'time_id', 'investment_id'])\n           .reset_index(drop=True));\n\ngc.collect()\n")


# In[10]:


train.head()


# ## Train and Test Baseline Model (full feature set)

# In[11]:


df_train, df_test = train_test_split(train, test_size=0.2, random_state=27)


# In[12]:


Y_train = df_train.pop('target')
Y_test = df_test.pop('target')


# In[13]:


mod_baseline = lgb.LGBMRegressor(device="gpu")


# In[14]:


get_ipython().run_cell_magic('time', '', "\nif IN_KAGGLE_NOTEBOOK:\n    mod_baseline = joblib.load(MODEL_PATH / 'mod_baseline.pkl')\nelse:\n    mod_baseline.fit(df_train, Y_train)\n\njoblib.dump(mod_baseline, 'mod_baseline.pkl')\n")


# In[15]:


preds = mod_baseline.predict(df_test)


# In[16]:


mean_squared_error(Y_test, preds)


# ## Run FRUFS Selection Process
# 
# FRUFS will need to iterate over each feature...300 LightGBM fits! Yikes. On a 20 core machine, FRUFS runs on the Ubiquant dataset, sampled to 20% of the rows, for a single fold with LightGBM on GPU at default hyperparameters in about 30 minutes. We actually want to tune the hyperparameter $k$ which means many FRUFS runs.

# In[17]:


get_ipython().run_cell_magic('time', '', '\nif IN_KAGGLE_NOTEBOOK:\n    model_frufs = joblib.load(MODEL_PATH / \'model_frufs.pkl\')\nelse:\n    model_frufs = FRUFS(\n        model_r=lgb.LGBMRegressor(random_state=42, device="gpu"),\n        k=0.98\n    )\n    model_frufs.fit(df_train.sample(frac=sample_frac))\n    \njoblib.dump(model_frufs, \'model_frufs.pkl\')\n')


# In[18]:


df_train_pruned = model_frufs.transform(df_train)
df_train_pruned.shape


# Let's take a look at what features are discarded.

# In[19]:


list(set(df_train.columns.tolist()) - set(df_train_pruned.columns.tolist()))


# In[20]:


df_test_pruned = model_frufs.transform(df_test)


# In[21]:


get_ipython().run_cell_magic('time', '', '\nif IN_KAGGLE_NOTEBOOK:\n    mod_k = joblib.load(MODEL_PATH / \'mod_k.pkl\')\nelse:\n    mod_k = lgb.LGBMRegressor(random_state=42, device="gpu")\n    mod_k.fit(df_train_pruned, Y_train)\n\njoblib.dump(mod_k, \'mod_k.pkl\')\n')


# In[22]:


preds_k = mod_k.predict(df_test_pruned)
mean_squared_error(Y_test, preds_k)


# We get some small improvement already. The FRUFS blogpost says we should tune the model, so let's do that...
# 
# ## FRUFS Hyperparameter Search
# 
# ![](https://optuna.org/assets/img/optuna-logo.png)
# 
# We want to tune $k$ **and** the hyperparameters of the Regressor that FRUFS uses (parameter `model_r`). One way to do this is to do a global hyperparamter tuning with everything at once. However, one FRUFS run takes ~30 minutes on my server and would take 7x that on Kaggle. As such, I am going to tune the hyperparameters in two stages. First I will tune the LightBGM model used inside FRUFS with **Optuna**, a Baysian hyperparameter tuning package. Then using the "optimal" regressor, I will sweep over 5 values of $k$.
# 
# To tune the FRUFS regressor, I use Optuna. The `objective(...)` function acts like the algorithm inside FRUFS: I pick a random feature, and fit a model to that feature with the remaining features. For each Optuna trial, I fit and predict individually for each of 60 randomly selected features of the 300 (I sample it down for speed purposes; even with this sampling, the tuning takes over 24 hours on my local machine). The trial returns the average MSE for each of those feature fits.

# In[23]:


all_columns = df_train.columns.tolist()

def objective(trial):

    n_feats = 60
    cumu_mse = 0
    
    for i in range(n_feats):
        # pick a random feature per FRUFS algorithm
        feature = np.random.choice(df_train.columns)
        X_columns = set(all_columns) - set([feature])

        train_sampled = df_train.sample(frac=0.10)

        train_x, valid_x, train_y, valid_y = train_test_split(
            train_sampled[X_columns], train_sampled[feature], test_size=0.25
        )

        dtrain = lgb.Dataset(train_x, label=train_y)
        dvalid = lgb.Dataset(valid_x, label=valid_y)

        param = {
            "device": "gpu",
            "force_col_wise": True,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "num_iterations": trial.suggest_int("num_iterations", 100, 300),
            "verbosity": 0,
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256, step=16),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 25, 200),
        }

        gbm = lgb.train(param, dtrain)

        preds = gbm.predict(valid_x)
        mse = mean_squared_error(valid_y, preds)
        cumu_mse = cumu_mse + mse
        
    return cumu_mse/n_feats


# In[24]:


get_ipython().run_cell_magic('time', '', '\nRUN_TUNE = False\n\nif IN_KAGGLE_NOTEBOOK or (RUN_TUNE == False):\n    study = joblib.load(MODEL_PATH / "optuna_study.pkl")\nelse:\n    study = optuna.create_study(\n        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),\n        direction="minimize"\n    )\n    study.optimize(objective, n_trials=100)\n\njoblib.dump(study, "optuna_study.pkl")\n')


# ## Optuna Study Inspection

# In[25]:


print("Number of finished trials: {}\n".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


# Optuna has some very nice built in visualizations as well. Let's check out some of them. The first is "Optimization History Plot" which shows the evolution of the best objective value across all trials.

# In[26]:


optuna.visualization.plot_optimization_history(study)


# I'm very happy to share the next plot. I've never appreciated "Parallel Coordinate" plots before. I always see them in advertisements for experiment tracking packages and websites but never stopped to think about what they are communicating. Working on this notebook though I finally see how informative they are. Take a look.

# In[27]:


optuna.visualization.plot_parallel_coordinate(study)


# You can see the darker the line the better the hyperparameter setting (it actually helps me to squint a little bit). Generally you don't want the optimal value to be near the edge of the range (because then there is a chance that a better value is outside the range). You can also get a sense of how sensitive 

# ## Re-Run FRUFS with the Optimal Hyperparameters

# In[28]:


model_frufs_fit1 = FRUFS(
    model_r=lgb.LGBMRegressor(random_state=42, **trial.params, device='gpu'),
    k=0.98
)


# In[29]:


if IN_KAGGLE_NOTEBOOK:
    model_frufs_fit1 = joblib.load(MODEL_PATH / 'model_frufs_fit1.pkl')
else:
    model_frufs_fit1.fit(df_train.sample(frac=sample_frac))

joblib.dump(model_frufs_fit1, 'model_frufs_fit1.pkl')


# In[30]:


df_train_pruned = model_frufs_fit1.transform(df_train)
df_train_pruned.shape


# In[31]:


dropped = list(set(df_train.columns.tolist()) - set(df_train_pruned.columns.tolist()))
dropped


# In[32]:


df_test_pruned = model_frufs_fit1.transform(df_test)

mod_k_1 = lgb.LGBMRegressor(random_state=42, device="gpu")


# In[33]:


if IN_KAGGLE_NOTEBOOK:
    mod_k_1 = joblib.load(MODEL_PATH / 'mod_k_1.pkl')
else:
    mod_k_1.fit(df_train_pruned, Y_train)

joblib.dump(mod_k_1, 'mod_k_1.pkl')


# In[34]:


preds_k = mod_k_1.predict(df_test_pruned)
mse_98 = mean_squared_error(Y_test, preds_k)
mse_98


# ## Search for the Optimal $k$
# 
# Sweep over different values of $k$.

# In[35]:


feat_imp_dict = dict(zip(model_frufs_fit1.columns_, model_frufs_fit1.feat_imps_))
feat_imp_dict = {k: v for k, v in sorted(feat_imp_dict.items(), key=lambda item: item[1], reverse=True)}


# In[36]:


feats = list(feat_imp_dict.keys())


# In[37]:


mse_trials = {0.98: mse_98}
feats_dropped = {0.98: dropped}

for k_trial in tqdm([0.75, 0.90, 0.95, 0.96, 0.97]):
    use_feats = feats[:-int(len(feats)*(1-k_trial))]
    
    feats_dropped[k_trial] = list(set(feats) - set(use_feats))
    df_train_pruned = df_train[use_feats]
    df_test_pruned = df_test[use_feats]
    
    if IN_KAGGLE_NOTEBOOK:
        mod_k_trial = joblib.load(MODEL_PATH / f'mod_k_trial_{int(k_trial*100)}.pkl')
    else:
        mod_k_trial = lgb.LGBMRegressor(random_state=42, device="gpu")
        mod_k_trial.fit(df_train_pruned, Y_train)
        
    joblib.dump(mod_k_trial, f'mod_k_trial_{int(k_trial*100)}.pkl') 
    
    preds_k = mod_k_trial.predict(df_test_pruned)
    mse_trials[k_trial] = mean_squared_error(Y_test, preds_k)


# In[38]:


mse_trials


# So the best setting is to keep **96%** of the features; this setting has the lowest MSE. Which features should we drop?

# In[39]:


feats_dropped[0.96]


# Good luck! Please lease questions/comments/criticism. Thank you for reading.
