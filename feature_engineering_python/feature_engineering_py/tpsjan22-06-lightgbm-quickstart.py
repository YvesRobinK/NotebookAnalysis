#!/usr/bin/env python
# coding: utf-8

# # LightGBM Quickstart
# 
# In this notebook, I want to create a good LightGBM model, applying the insight we have gained from [EDA](https://www.kaggle.com/ambrosm/tpsjan22-01-eda-which-makes-sense) and [linear model](https://www.kaggle.com/ambrosm/tpsjan22-03-linear-model).
# 
# LightGBM has major advantages compared to the linear model:
# - It finds all seasonal effects and fixed holidays from one single feature and doesn't need a Fourier transformation.
# - It finds all full-moon-dependent holidays from Easter to Pentecost from a single feature.
# 
# I still use a **transformed target, but with a new transformation**: I divide `num_sold` by the GDP\*\*1.212 and then take the logarithm. At the same time, I hide the year from the booster. With this trick, I avoid that the decision trees have to deal with year or GDP values they haven't seen in training. In other words, I take for granted that `num_sold` is proportional to the GDP\*\*1.212, and I want the model to not even try to find any alternative dependence between the GDP and `num_sold`. The derivation of this exponent can be found in the [linear model notebook](https://www.kaggle.com/ambrosm/tpsjan22-03-linear-model).
# 
# I have **changed the cross-validation scheme** from my earlier notebook: I'm no longer using the years 2015-2017 for training and 2018 for validation, but a full GroupKFold with the years as groups. I use the GroupKFold because after removing year and GDP from the features, the data is no longer a real time series. And we don't need to worry about using information from the future because the GDP is information from the future anyway.
# 
# Release notes:
# - V2: refit several times, tuned learning rate
# - V3: new hyperparameter optimization with visualization
# - V4: GDP exponent
# - V5: hyperparameter tuning, small feature modifications
# 

# In[1]:


import numpy as np
import pandas as pd
import lightgbm
import pickle
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
import dateutil.easter as easter


# In[2]:


original_train_df = pd.read_csv('../input/tabular-playground-series-jan-2022/train.csv', parse_dates=['date'])
original_test_df = pd.read_csv('../input/tabular-playground-series-jan-2022/test.csv', parse_dates=['date'])
gdp_df = pd.read_csv('../input/gdp-20152019-finland-norway-and-sweden/GDP_data_2015_to_2019_Finland_Norway_Sweden.csv',
                    index_col='year')

original_train_df.head(2)


# In[3]:


def smape_loss(y_true, y_pred):
    """SMAPE Loss"""
    return np.abs(y_true - y_pred) / (y_true + np.abs(y_pred)) * 200


# # Feature engineering

# In[4]:


# Feature engineering
gdp_exponent = 1.2121103201489674 # see https://www.kaggle.com/ambrosm/tpsjan22-03-linear-model for an explanation
def get_gdp(row):
    """Return the GDP based on row.country and row.date.year"""
    country = 'GDP_' + row.country
    return gdp_df.loc[row.date.year, country] ** gdp_exponent

le_dict = {feature: LabelEncoder().fit(original_train_df[feature]) for feature in ['country', 'product', 'store']}

def engineer(df):
    """Return a new dataframe with the engineered features"""
    
    new_df = pd.DataFrame({'gdp': df.apply(get_gdp, axis=1),
                           'dayofyear': df.date.dt.dayofyear,
                           'wd4': df.date.dt.weekday == 4, # Friday
                           'wd56': df.date.dt.weekday >= 5, # Saturday and Sunday
                          })

    new_df.loc[(df.date.dt.year != 2016) & (df.date.dt.month >=3), 'dayofyear'] += 1 # fix for leap years
    
    for feature in ['country', 'product', 'store']:
        new_df[feature] = le_dict[feature].transform(df[feature])
        
    # Easter
    easter_date = df.date.apply(lambda date: pd.Timestamp(easter.easter(date.year)))
    new_df['days_from_easter'] = (df.date - easter_date).dt.days.clip(-3, 59)
    new_df.loc[new_df['days_from_easter'].isin(range(12, 39)), 'days_from_easter'] = 12 # reduce overfitting
    #new_df.loc[new_df['days_from_easter'] == 59, 'days_from_easter'] = -3
    
    # Last Wednesday of June
    wed_june_date = df.date.dt.year.map({2015: pd.Timestamp(('2015-06-24')),
                                         2016: pd.Timestamp(('2016-06-29')),
                                         2017: pd.Timestamp(('2017-06-28')),
                                         2018: pd.Timestamp(('2018-06-27')),
                                         2019: pd.Timestamp(('2019-06-26'))})
    new_df['days_from_wed_jun'] = (df.date - wed_june_date).dt.days.clip(-5, 5)
    
    # First Sunday of November (second Sunday is Father's Day)
    sun_nov_date = df.date.dt.year.map({2015: pd.Timestamp(('2015-11-1')),
                                         2016: pd.Timestamp(('2016-11-6')),
                                         2017: pd.Timestamp(('2017-11-5')),
                                         2018: pd.Timestamp(('2018-11-4')),
                                         2019: pd.Timestamp(('2019-11-3'))})
    new_df['days_from_sun_nov'] = (df.date - sun_nov_date).dt.days.clip(-1, 9)
    
    return new_df

train_df = engineer(original_train_df)
train_df['date'] = original_train_df.date # used in GroupKFold
train_df['num_sold'] = original_train_df.num_sold.astype(np.float32)
train_df['target'] = np.log(train_df['num_sold'] / train_df['gdp'])
test_df = engineer(original_test_df)

features = test_df.columns.difference(['gdp'])
print(list(features))


# # Training and validation

# In[5]:


DIAGRAMS = True

params0 = {'objective': 'regression', # Manual optimization
           'force_row_wise': True,
           'max_bin': 400, # need more bins than days in a year
           'verbosity': -1,
           'seed': 1,
           'bagging_seed': 3,
           'feature_fraction_seed': 2,
           'learning_rate': 0.018,
           'lambda_l1': 0,
           'lambda_l2': 1e-2,
           'num_leaves': 18,
           'feature_fraction': 0.710344827586207,
           'bagging_fraction': 0.47931034482758617,
           'bagging_freq': 3,
           'min_child_samples': 20}

def fit_model(X_tr, X_va=None, run=0, fold=0, params=params0):
    """Scale the data, fit a model, plot the training history and validate the model"""
    start_time = datetime.now()

    # Preprocess the data
    X_tr_f = X_tr[features]
    y_tr = X_tr.target.values
    data_tr = lightgbm.Dataset(X_tr[features], label=y_tr,
                           categorical_feature=['country', 'product', 'store'])

    # Train the model
    model = lightgbm.train(params, data_tr, num_boost_round=2000,
                           categorical_feature=['country', 'product', 'store'])

    if X_va is not None:
        # Preprocess the validation data
        X_va_f = X_va[features]
        y_va = X_va.target.values
        data_va = lightgbm.Dataset(X_va[features], label=y_va, reference=data_tr)

        # Inference for validation
        y_va_pred = np.exp(model.predict(X_va_f)) * X_va['gdp']
        oof.update(y_va_pred)
        
        # Evaluation: Execution time and SMAPE
        smape_before_correction = np.mean(smape_loss(X_va.num_sold, y_va_pred))
        #y_va_pred *= LOSS_CORRECTION
        smape = np.mean(smape_loss(X_va.num_sold, y_va_pred))
        print(f"Fold {run}.{fold} | {str(datetime.now() - start_time)[-12:-7]}"
              f" | SMAPE: {smape:.5f}   (before correction: {smape_before_correction:.5f})")
        score_list.append(smape)
        
        # Plot y_true vs. y_pred
        if DIAGRAMS and fold == 0:
            plt.figure(figsize=(10, 10))
            plt.scatter(X_va.num_sold, y_va_pred, s=1, color='r')
            #plt.scatter(np.log(y_va), np.log(y_va_pred), s=1, color='g')
            plt.plot([plt.xlim()[0], plt.xlim()[1]], [plt.xlim()[0], plt.xlim()[1]], '--', color='k')
            plt.gca().set_aspect('equal')
            plt.xlabel('y_true')
            plt.ylabel('y_pred')
            plt.title('OOF Predictions')
            plt.show()
        
    else:
        smape = None
        
    return model, smape

# Plot all num_sold_true and num_sold_pred (five years) for one country-store-product combination
def plot_five_years_combination(engineer, country='Norway', store='KaggleMart', product='Kaggle Hat'):
    demo_df = pd.DataFrame({'row_id': 0,
                            'date': pd.date_range('2015-01-01', '2019-12-31', freq='D'),
                            'country': country,
                            'store': store,
                            'product': product})
    demo_df.set_index('date', inplace=True, drop=False)
    demo_df_e = engineer(demo_df)
    demo_df['num_sold'] = np.exp(model.predict(demo_df_e[features])) * demo_df.apply(get_gdp, axis=1)
    plt.figure(figsize=(20, 6))
    plt.plot(np.arange(len(demo_df)), demo_df.num_sold, label='prediction')
    train_subset = train_df[(original_train_df.country == country) & (original_train_df.store == store) & (original_train_df['product'] == product)]
    plt.scatter(np.arange(len(train_subset)), train_subset.num_sold, label='true', alpha=0.5, color='red', s=3)
    plt.legend()
    plt.title('Predictions and true num_sold for five years')
    plt.show()

oof = pd.Series(0, index=train_df.index)
score_list = []
kf = GroupKFold(n_splits=4)
for fold, (train_idx, val_idx) in enumerate(kf.split(train_df, groups=train_df.date.dt.year)):
    X_tr = train_df.iloc[train_idx]
    X_va = train_df.iloc[val_idx]
    model, smape = fit_model(X_tr, X_va, run=0, fold=fold)

print(f"Average SMAPE: {sum(score_list) / len(score_list):.5f}")
with open('oof.pickle', 'wb') as handle: pickle.dump(oof, handle)
    
if DIAGRAMS: plot_five_years_combination(engineer)


# # Forget Optuna
# There are two reasons not to use Optuna:
# 1. It overfits.
# 2. You learn more by plotting the effect of varying the parameters

# In[6]:


# Grid search for the best hyperparameter
def optimize_param(params, param_name, pmin, pmax, log=False, int_=False, n_steps=30):
    """Grid search for the best hyperparameter; updates params
    """
    score_list = []
    if int_:
        step_size = max(round((pmax-pmin) / n_steps), 1)
        w_array = np.arange(pmin, pmax+1, step_size) 
    elif log:
        w_array = np.logspace(np.log10(pmin), np.log10(pmax), n_steps) 
    else:
        w_array = np.linspace(pmin, pmax, n_steps) 
    for w in w_array:
        print(f"{param_name}: {w}")
        params[param_name] = w
        model, smape = fit_model(X_tr, X_va, run=0, fold=fold, params=params)
        score_list.append(smape)
    plt.figure(figsize=(12,4))
    plt.plot(w_array, score_list, label='measured')
    plt.scatter([w_array[np.argmin(np.array(score_list))]], [min(score_list)], color='b')
    poly = np.polynomial.polynomial.Polynomial.fit(w_array, score_list, deg=2)
    plt.plot(w_array, poly(w_array), 'g--', label='fit')
    plt.scatter([w_array[np.argmin(np.array(poly(w_array)))]], [min(poly(w_array))], color='g')
    plt.legend(loc='upper left')
    plt.ylabel('SMAPE')
    plt.xlabel(param_name)
    plt.title(f'Optimizing {param_name}')
    plt.show()

    best_w = w_array[np.argmin(np.array(poly(w_array)))]
    print(f"Best {param_name}: {best_w}    | Best SMAPE: {min(poly(w_array)):.5f}")
    params[param_name] = best_w

X_tr = train_df[train_df.date.dt.year < 2018]
X_va = train_df[train_df.date.dt.year == 2018]
params = params0
params['bagging_seed'] =10
#optimize_param(params, 'max_depth', 4, 14, int_=True, n_steps=12)
#optimize_param(params, 'lambda_l1', 0, 1e0, log=False, n_steps=30)
#optimize_param(params, 'lambda_l2', 1e-4, 0.02, log=True, n_steps=30)
optimize_param(params, 'bagging_freq', 1, 10, int_=True, n_steps=10)
optimize_param(params, 'bagging_fraction', 0.1, 0.6, log=False, n_steps=30)
optimize_param(params, 'feature_fraction', 0.4, 1.0, log=False, n_steps=30)
optimize_param(params, 'min_child_samples', 1, 40, int_=True, n_steps=30)
optimize_param(params, 'num_leaves', 6, 30, int_=True, n_steps=10)
optimize_param(params, 'learning_rate', 0.007, 0.04, n_steps=20)
params


# # Hyperparameter optimization with Optuna
# Let's skip this part...

# In[7]:


# Optuna
OPTUNA = False
if OPTUNA:
    import optuna

    def objective(trial):
        params = {'objective': 'mae', # 'regression' or 'mae'?
                  'force_row_wise': True,
                  'verbosity': -1,
                  'boosting_type': 'gbdt',
                  'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                  'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                  'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
                  'num_leaves': trial.suggest_int('num_leaves', 2, 256),
                  'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                  'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                  'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                  'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        }
        model, smape = fit_model(X_tr, X_va, run=0, fold=fold, params=params)
        return smape

    X_tr = train_df[train_df.date.dt.year < 2018]
    X_va = train_df[train_df.date.dt.year == 2018]
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)


# # Re-training, inference and submission

# In[8]:


# Refit the model on the complete training data several times with different seeds
test_pred_list = []
for i in range(25):
    params = params0
    params['seed'] = i
    params['bagging_seed'] = i+1
    params['feature_fraction_seed'] = i+2
    model, _ = fit_model(train_df, params=params)
    test_pred_list.append(np.exp(model.predict(test_df[features])) * test_df['gdp'].values)

#plot_five_years_combination(engineer) # Quick check for debugging
train_df['pred'] = np.exp(model.predict(train_df[features])) * train_df['gdp'].values
with open('train_pred.pickle', 'wb') as handle: pickle.dump(train_df.pred, handle) # save residuals for further analysis

if len(test_pred_list) > 0:
    # Create the submission file
    sub = original_test_df[['row_id']].copy()
    sub['num_sold'] = sum(test_pred_list) / len(test_pred_list)
    sub.to_csv('submission_lightgbm_quickstart.csv', index=False)

    # Plot the distribution of the test predictions
    plt.figure(figsize=(16,3))
    plt.hist(train_df['num_sold'], bins=np.linspace(0, 3000, 201),
             density=True, label='Training')
    plt.hist(sub['num_sold'], bins=np.linspace(0, 3000, 201),
             density=True, rwidth=0.5, label='Test predictions')
    plt.xlabel('num_sold')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


# In[9]:


sub


# In[10]:


# Create a rounded submission file
sub_rounded = sub.copy()
sub_rounded['num_sold'] = sub_rounded['num_sold'].round()
sub_rounded.to_csv('submission_lightgbm_quickstart_rounded.csv', index=False)
sub_rounded


# In[ ]:




