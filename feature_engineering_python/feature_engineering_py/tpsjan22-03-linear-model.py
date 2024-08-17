#!/usr/bin/env python
# coding: utf-8

# # Linear model for the January TPS
# 
# Scikit-learn doesn't offer SMAPE as a loss function. As a workaround, I'm training for Huber loss with a transformed target, apply a correction factor, and we'll see how far we'll get.
# 
# The **transformed target** for the regression is the log of the sales numbers. This has several benefits:
# - MAE of the log is a good approximation for SMAPE. Why? SMAPE is a relative error which can be approximated by abs(y_pred - y_true) / y_true = abs(y_pred/y_true - 1). For small errors, we may approximate abs(y_pred/y_true - 1) by abs(log(y_pred/y_true)) = abs(log(y_pred) - log(y_true)) = MAE(log()). A relative error resembles an absolute error of the logarithm of the target. In a [diagram](https://www.kaggle.com/c/tabular-playground-series-jan-2022/discussion/298473) of the two functions, the difference is barely noticeable.
# - While the regression can output negative predictions, exp(regression_output) is always positive.
# - With this transformed target, we can easily fit an exponential growth rate.
# - Most other effects will be multiplicative as well: hat sales on a Sunday in Finland will be average hat sales multiplied by a Sunday factor multiplied by a Finland factor
# 
# 
# We proceed in **two steps**:
# 1. We first create a simple model without holidays. Plotting the residuals of this simple model helps identify the holidays.
# 2. We then create an advanced model with features for all the holidays.
# 
# The notebook goes together with the [EDA notebook](https://www.kaggle.com/ambrosm/tpsjan22-01-eda-which-makes-sense), which visualizes the various seasonal effects and the differences in growth rate.
# 
# Bug reports: Please report all bugs in the comments section of the notebook.
# 
# Release notes:
# - V2: Modified yearly growth
# - V3: No growth from 2018 to 2019
# - V4: Added Easter feature
# - V5: Various optimizations
# - V6: Without loss correction factor
# - V7: More holidays
# - V8: Added GDP feature, thanks to [@carlmcbrideellis](https://www.kaggle.com/c/tabular-playground-series-jan-2022/discussion/298911)'s [dataset](https://www.kaggle.com/carlmcbrideellis/gdp-20152019-finland-norway-and-sweden)
# - V9: Added dayofdataset feature
# - V10: Taking the ceiling of all predictions as @remekkinas [suggests](https://www.kaggle.com/c/tabular-playground-series-jan-2022/discussion/299162)
# - V11: log of GDP makes more sense because we predict the log of the target
# - V14: Ridge ([proposed by @paddykb](https://www.kaggle.com/c/tabular-playground-series-jan-2022/discussion/299296#1641253)), rounding (as [proposed by @remekkinas](https://www.kaggle.com/c/tabular-playground-series-jan-2022/discussion/299162) and demonstrated in @fergusfindley's [ensembling and rounding techniques comparison](https://www.kaggle.com/fergusfindley/ensembling-and-rounding-techniques-comparison)), feature importance, different holiday length 
# - V15: GroupKFold, save oof and residuals, GDP exponent, shorter Fourier transformation
# - V16: more features, systematic residual analysis
# 

# In[1]:


import pandas as pd
import numpy as np
import pickle
import itertools
import gc
import math
import matplotlib.pyplot as plt
import dateutil.easter as easter
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, PercentFormatter
from datetime import datetime, date, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression, HuberRegressor, Ridge, Lasso


# In[2]:


original_train_df = pd.read_csv('../input/tabular-playground-series-jan-2022/train.csv')
original_test_df = pd.read_csv('../input/tabular-playground-series-jan-2022/test.csv')
gdp_df = pd.read_csv('../input/gdp-20152019-finland-norway-and-sweden/GDP_data_2015_to_2019_Finland_Norway_Sweden.csv')

gdp_df.set_index('year', inplace=True)

# The dates are read as strings and must be converted
for df in [original_train_df, original_test_df]:
    df['date'] = pd.to_datetime(df.date)
original_train_df.head(2)


# In[3]:


def smape_loss(y_true, y_pred):
    """SMAPE Loss"""
    return np.abs(y_true - y_pred) / (y_true + np.abs(y_pred)) * 200


# # Simple feature engineering (without holidays)
# 
# In this simple model, we consider the following:
# - country, store, product
# - weekdays
# - seasonal variations per product as a Fourier series with wavelengths from 1 year down to 18 days
# - country's GDP
# 
# The residuals of this simple model will permit us to understand the effect of holidays.

# In[4]:


# Feature engineering
def engineer(df):
    """Return a new dataframe with the engineered features"""
    
    def get_gdp(row):
        country = 'GDP_' + row.country
        return gdp_df.loc[row.date.year, country]
        
    new_df = pd.DataFrame({'gdp': np.log(df.apply(get_gdp, axis=1)),
                           'wd4': df.date.dt.weekday == 4, # Friday
                           'wd56': df.date.dt.weekday >= 5, # Saturday and Sunday
                          })

    # One-hot encoding (no need to encode the last categories)
    for country in ['Finland', 'Norway']:
        new_df[country] = df.country == country
    new_df['KaggleRama'] = df.store == 'KaggleRama'
    for product in ['Kaggle Mug', 'Kaggle Hat']:
        new_df[product] = df['product'] == product
        
    # Seasonal variations (Fourier series)
    # The three products have different seasonal patterns
    dayofyear = df.date.dt.dayofyear
    for k in range(1, 3):
        new_df[f'sin{k}'] = np.sin(dayofyear / 365 * 2 * math.pi * k)
        new_df[f'cos{k}'] = np.cos(dayofyear / 365 * 2 * math.pi * k)
        new_df[f'mug_sin{k}'] = new_df[f'sin{k}'] * new_df['Kaggle Mug']
        new_df[f'mug_cos{k}'] = new_df[f'cos{k}'] * new_df['Kaggle Mug']
        new_df[f'hat_sin{k}'] = new_df[f'sin{k}'] * new_df['Kaggle Hat']
        new_df[f'hat_cos{k}'] = new_df[f'cos{k}'] * new_df['Kaggle Hat']

    return new_df

train_df = engineer(original_train_df)
train_df['date'] = original_train_df.date
train_df['num_sold'] = original_train_df.num_sold.astype(np.float32)
test_df = engineer(original_test_df)

features = test_df.columns

for df in [train_df, test_df]:
    df[features] = df[features].astype(np.float32)
print(list(features))


# # Training the simple model (without holidays)
# 
# We train the model on the full training data. Cross-validation will come later; we first want to see the residual diagrams.
# 
# As a first quick check, we plot the predictions for the combination `country='Norway', store='KaggleMart', product='Kaggle Hat'`.
# 

# In[5]:


def fit_model(X_tr, X_va=None, outliers=False):
    """Scale the data, fit a model, plot the training history and validate the model"""
    start_time = datetime.now()

    # Preprocess the data
    X_tr_f = X_tr[features]
    preproc = StandardScaler()
    X_tr_f = preproc.fit_transform(X_tr_f)
    y_tr = X_tr.num_sold.values.reshape(-1, 1)
    
    # Train the model
    #model = LinearRegression()
    #model = HuberRegressor(epsilon=1.20, max_iter=500)
    model = Ridge()
    model.fit(X_tr_f, np.log(y_tr).ravel())

    if X_va is not None:
        # Preprocess the validation data
        X_va_f = X_va[features]
        X_va_f = preproc.transform(X_va_f)
        y_va = X_va.num_sold.values.reshape(-1, 1)

        # Inference for validation
        y_va_pred = np.exp(model.predict(X_va_f)).reshape(-1, 1)
        oof.update(pd.Series(y_va_pred.ravel(), index=X_va.index))
        
        # Evaluation: Execution time and SMAPE
        smape_before_correction = np.mean(smape_loss(y_va, y_va_pred))
        #y_va_pred *= LOSS_CORRECTION
        smape = np.mean(smape_loss(y_va, y_va_pred))
        print(f"Fold {run}.{fold} | {str(datetime.now() - start_time)[-12:-7]}"
              f" | SMAPE: {smape:.5f}   (before correction: {smape_before_correction:.5f})")
        score_list.append(smape)
        
        # Plot y_true vs. y_pred
        if fold == 0:
            plt.figure(figsize=(10, 10))
            plt.scatter(y_va, y_va_pred, s=1, color='r')
            #plt.scatter(np.log(y_va), np.log(y_va_pred), s=1, color='g')
            plt.plot([plt.xlim()[0], plt.xlim()[1]], [plt.xlim()[0], plt.xlim()[1]], '--', color='k')
            plt.gca().set_aspect('equal')
            plt.xlabel('y_true')
            plt.ylabel('y_pred')
            plt.title('OOF Predictions')
            plt.show()
        
    return preproc, model

preproc, model = fit_model(train_df)

# Plot all num_sold_true and num_sold_pred (five years) for one country-store-product combination
def plot_five_years_combination(engineer, country='Norway', store='KaggleMart', product='Kaggle Hat'):
    demo_df = pd.DataFrame({'row_id': 0,
                            'date': pd.date_range('2015-01-01', '2019-12-31', freq='D'),
                            'country': country,
                            'store': store,
                            'product': product})
    demo_df.set_index('date', inplace=True, drop=False)
    demo_df = engineer(demo_df)
    demo_df['num_sold'] = np.exp(model.predict(preproc.transform(demo_df[features])))
    plt.figure(figsize=(20, 6))
    plt.plot(np.arange(len(demo_df)), demo_df.num_sold, label='prediction')
    train_subset = train_df[(original_train_df.country == country) & (original_train_df.store == store) & (original_train_df['product'] == product)]
    plt.scatter(np.arange(len(train_subset)), train_subset.num_sold, label='true', alpha=0.5, color='red', s=3)
    plt.legend()
    plt.title('Predictions and true num_sold for five years')
    plt.show()

plot_five_years_combination(engineer)


# # Residuals of the simple model
# 
# Now we plot the residuals of the simple model. These diagrams show us where the holidays are:
# - End of year peak
# - Three movable holidays depending on the full moon (Easter)
# - First half of May, second half of May
# - Beginning of June, end of June
# - Beginning of November
# 

# In[6]:


train_df['pred'] = np.exp(model.predict(preproc.transform(train_df[features])))
by_date = train_df.groupby(train_df['date'])
residuals = (by_date.pred.sum() - by_date.num_sold.sum()) / (by_date.pred.sum() + by_date.num_sold.sum()) * 200

# Plot all residuals (four-year range, sum of all products)
def plot_all_residuals(residuals):
    plt.figure(figsize=(20,6))
    plt.scatter(residuals.index,
                residuals,
                s=1, color='k')
    plt.vlines(pd.date_range('2015-01-01', '2019-01-01', freq='M'),
               plt.ylim()[0], plt.ylim()[1], alpha=0.5)
    plt.vlines(pd.date_range('2015-01-01', '2019-01-01', freq='Y'),
               plt.ylim()[0], plt.ylim()[1], alpha=0.5)
    plt.title('Residuals for four years')
    plt.show()
    
plot_all_residuals(residuals)

# Plot residuals for interesting intervals
def plot_around(residuals, m, d, w):
    """Plot residuals in an interval of with 2*w around month=m and day=d"""
    plt.figure()
    plt.title(f"Residuals around m={m} d={d}")
    for y in np.arange(2015, 2020):
        d0 = pd.Timestamp(date(y, m, d))
        residual_range = residuals[(residuals.index > d0 - timedelta(w)) & 
                                   (residuals.index < d0 + timedelta(w))]
        plt.plot([(r - d0).days for r in residual_range.index], residual_range, label=str(y))
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) # only integer labels
    plt.legend()
    plt.show()

plot_around(residuals, 1, 1, 20) # end of year peak
plot_around(residuals, 5, 1, 50) # three moveable peaks depending on Easter
#plot_around(residuals, 5, 21, 10) # zoom-in
#plot_around(residuals, 5, 31, 15) # zoom-in
plot_around(residuals, 6, 10, 10) # first half of June (with overlay of Pentecost in 2017)
plot_around(residuals, 6, 30, 10) # moveable peak end of June
plot_around(residuals, 11, 5, 10) # moveable peak beginning of November


# # More feature engineering (advanced model)

# In[7]:


# Feature engineering for holidays
def engineer_more(df):
    """Return a new dataframe with more engineered features"""
    new_df = engineer(df)

    # End of year
    new_df = pd.concat([new_df,
                        pd.DataFrame({f"dec{d}":
                                      (df.date.dt.month == 12) & (df.date.dt.day == d)
                                      for d in range(24, 32)}),
                        pd.DataFrame({f"n-dec{d}":
                                      (df.date.dt.month == 12) & (df.date.dt.day == d) & (df.country == 'Norway')
                                      for d in range(24, 32)}),
                        pd.DataFrame({f"f-jan{d}":
                                      (df.date.dt.month == 1) & (df.date.dt.day == d) & (df.country == 'Finland')
                                      for d in range(1, 14)}),
                        pd.DataFrame({f"jan{d}":
                                      (df.date.dt.month == 1) & (df.date.dt.day == d) & (df.country == 'Norway')
                                      for d in range(1, 10)}),
                        pd.DataFrame({f"s-jan{d}":
                                      (df.date.dt.month == 1) & (df.date.dt.day == d) & (df.country == 'Sweden')
                                      for d in range(1, 15)})],
                       axis=1)
    
    # May
    new_df = pd.concat([new_df,
                        pd.DataFrame({f"may{d}":
                                      (df.date.dt.month == 5) & (df.date.dt.day == d) 
                                      for d in list(range(1, 10))}), #  + list(range(17, 25))
                        pd.DataFrame({f"may{d}":
                                      (df.date.dt.month == 5) & (df.date.dt.day == d) & (df.country == 'Norway')
                                      for d in list(range(19, 26))})],
                       axis=1)
    
    # June and July
    new_df = pd.concat([new_df,
                        pd.DataFrame({f"june{d}":
                                      (df.date.dt.month == 6) & (df.date.dt.day == d) & (df.country == 'Sweden')
                                      for d in list(range(8, 14))}),
                        #pd.DataFrame({f"june{d}":
                        #              (df.date.dt.month == 6) & (df.date.dt.day == d) & (df.country == 'Norway')
                        #              for d in list(range(22, 31))}),
                        #pd.DataFrame({f"july{d}":
                        #              (df.date.dt.month == 7) & (df.date.dt.day == d) & (df.country == 'Norway')
                        #              for d in list(range(1, 3))})],
                       ],
                       axis=1)
    
    # Last Wednesday of June
    wed_june_date = df.date.dt.year.map({2015: pd.Timestamp(('2015-06-24')),
                                         2016: pd.Timestamp(('2016-06-29')),
                                         2017: pd.Timestamp(('2017-06-28')),
                                         2018: pd.Timestamp(('2018-06-27')),
                                         2019: pd.Timestamp(('2019-06-26'))})
    new_df = pd.concat([new_df,
                        pd.DataFrame({f"wed_june{d}": 
                                      (df.date - wed_june_date == np.timedelta64(d, "D")) & (df.country != 'Norway')
                                      for d in list(range(-4, 6))})],
                       axis=1)
    
    # First Sunday of November
    sun_nov_date = df.date.dt.year.map({2015: pd.Timestamp(('2015-11-1')),
                                         2016: pd.Timestamp(('2016-11-6')),
                                         2017: pd.Timestamp(('2017-11-5')),
                                         2018: pd.Timestamp(('2018-11-4')),
                                         2019: pd.Timestamp(('2019-11-3'))})
    new_df = pd.concat([new_df,
                        pd.DataFrame({f"sun_nov{d}": 
                                      (df.date - sun_nov_date == np.timedelta64(d, "D")) & (df.country != 'Norway')
                                      for d in list(range(0, 9))})],
                       axis=1)
    
    # First half of December (Independence Day of Finland, 6th of December)
    new_df = pd.concat([new_df,
                        pd.DataFrame({f"dec{d}":
                                      (df.date.dt.month == 12) & (df.date.dt.day == d) & (df.country == 'Finland')
                                      for d in list(range(6, 14))})],
                       axis=1)

    # Easter
    easter_date = df.date.apply(lambda date: pd.Timestamp(easter.easter(date.year)))
    new_df = pd.concat([new_df,
                        pd.DataFrame({f"easter{d}": 
                                      (df.date - easter_date == np.timedelta64(d, "D"))
                                      for d in list(range(-2, 11)) + list(range(40, 48)) + list(range(50, 59))})],
                       axis=1)
    
    return new_df.astype(np.float32)

train_df = engineer_more(original_train_df)
train_df['date'] = original_train_df.date
train_df['num_sold'] = original_train_df.num_sold.astype(np.float32)
test_df = engineer_more(original_test_df)

features = list(test_df.columns)
print(list(features))


# # Residuals of the advanced model
# 
# The diagrams show that the residuals are much smaller now and that we have implemented the most important holidays correctly.

# In[8]:


preproc, model = fit_model(train_df)
train_df['pred'] = np.exp(model.predict(preproc.transform(train_df[features])))
with open('train_pred.pickle', 'wb') as handle: pickle.dump(train_df.pred, handle) # save residuals for further analysis
by_date = train_df.groupby(train_df['date'])
residuals = (by_date.pred.sum() - by_date.num_sold.sum()) / (by_date.pred.sum() + by_date.num_sold.sum()) * 200

# Plot all num_sold_true and num_sold_pred (five years) for one country-store-product combination
plot_five_years_combination(engineer_more)

# Plot all residuals (four-year range, sum of all products)
plot_all_residuals(residuals)

# Plot residuals for interesting intervals
plot_around(residuals, 1, 1, 20) # end of year peak
plot_around(residuals, 5, 1, 50) # three moveable peaks depending on Easter
#plot_around(residuals, 5, 21, 10) # zoom-in
#plot_around(residuals, 5, 31, 15) # zoom-in
plot_around(residuals, 6, 10, 10) # first half of June (with overlay of Pentecost in 2017)
plot_around(residuals, 6, 30, 10) # moveable peak end of June
plot_around(residuals, 11, 5, 10) # moveable peak beginning of November


# # Systematic analysis of the residuals
# 
# Looking at the residual plots has brought us quite far, but it's not enough. We'll now do a more systematic analysis of the residuals. We start by computing all the residuals, this time defined as the difference between the log of the prediction and the log of the true value.
# 
# We see that these residuals are normally distributed with center 0 and standard deviation 0.053.

# In[9]:


residuals = np.log(train_df.pred) - np.log(train_df.num_sold)
plt.figure(figsize=(18, 4))
plt.scatter(np.arange(len(residuals)), residuals, s=1)
plt.title('All residuals by row number')
plt.ylabel('residual')
plt.show()
plt.figure(figsize=(18, 4))
plt.hist(residuals, bins=200)
plt.title('Histogram of all residuals')
plt.show()
print(f"Standard deviation of log residuals: {residuals.std():.3f}")


# In the next step, we compute the mean residual for all the 366 days and search for days with unusually high residuals. But what is "unusually high"? In statistical tests we often aim at a significance level of 0.05. But the significance level corresponds to the rate of false positives, and when testing 366 days, I want to get at most 1 false positive holiday. At a significance level of 1/365, we need to find residuals which are higher than three times the standard deviation.
# 

# In[10]:


train_df['dayfix'] = train_df.date.dt.dayofyear
train_df.loc[(train_df.date.dt.year != 2016) & (train_df.date.dt.month >= 3), 'dayfix'] += 1

from scipy.stats import norm
print("Look for residuals beyond", norm.ppf([0.5/365, 364.5/365]))

rr = residuals.groupby(train_df.dayfix).mean()
rrstd = rr.std()
print(f"Standard deviation when grouped by dayofyear: {rrstd:.5f}")
rrdf = pd.DataFrame({'residual': rr, 'z_score': rr / rrstd, 'date': pd.date_range('2016-01-01', '2016-12-31')})
rrdf[rrdf.z_score.abs() > 3]


# The table above shows us one potential holiday: October 21. But wait: There may be country-specific holidays. Repeating the same procedure for residuals grouped by country and day gives some more candidates:

# In[11]:


# Candidate country-specific holidays
rr = residuals.groupby([original_train_df.country, train_df.dayfix]).mean()
rrstd = rr.std()
print(f"Standard deviation when grouped by country and dayofyear: {rrstd:.5f}")
rrdf = pd.DataFrame({'residual': rr, 'z_score': rr / rrstd, 'date': np.datetime64('2015-12-31') + pd.to_timedelta(rr.index.get_level_values(1), 'D')})
rrdf[rrdf.z_score.abs() > 3]


# Engineering the features for these holidays is a task for a future version of this notebook (or for your own model). And the other open task is the search for movable holidays (e.g. based on the first Sunday of November). We can apply the same statistical test to find movable holidays, we only need to correctly group the residuals.

# # Training with validation
# 
# We train on the years 2015 - 2017 and validate on 2018. For the validation, we show
# - The execution time and the SMAPE
# - A scatterplot y_true vs. y_pred (ideally all points should lie near the diagonal)
# 

# In[12]:


#%%time
RUNS = 1 # should be 1. increase the number of runs only if you want see how the result depends on the random seed
OUTLIERS = True
TRAIN_VAL_CUT = datetime(2018, 1, 1)
LOSS_CORRECTION = 1

# Make the results reproducible
np.random.seed(202100)

total_start_time = datetime.now()
oof = pd.Series(0.0, index=train_df.index)
score_list = []
for run in range(RUNS):
    kf = GroupKFold(n_splits=4)
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df, groups=train_df.date.dt.year)):
        X_tr = train_df.iloc[train_idx]
        X_va = train_df.iloc[val_idx]
        print(f"Fold {run}.{fold}")
        preproc, model = fit_model(X_tr, X_va)

print(f"Average SMAPE: {sum(score_list) / len(score_list):.5f}")
with open('oof.pickle', 'wb') as handle: pickle.dump(oof, handle)


# # Inference and submission
# 
# We are saving two submission files:
# - The real-valued predictions of the regression. You can use these values for blending.
# - The predictions rounded to the nearest integer.

# In[13]:


# Fit the model on the complete training data
train_idx = np.arange(len(train_df))
X_tr = train_df.iloc[train_idx]
preproc, model = fit_model(X_tr, None)

plot_five_years_combination(engineer_more) # Quick check for debugging

# Inference for test
test_pred_list = []
test_pred_list.append(np.exp(model.predict(preproc.transform(test_df[features]))) * LOSS_CORRECTION)

# Create the submission file
sub = original_test_df[['row_id']].copy()
sub['num_sold'] = sum(test_pred_list) / len(test_pred_list)
sub.to_csv('submission_linear_model.csv', index=False)

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

sub


# In[14]:


# Create a rounded submission file
sub_rounded = sub.copy()
sub_rounded['num_sold'] = sub_rounded['num_sold'].round()
sub_rounded.to_csv('submission_linear_model_rounded.csv', index=False)
sub_rounded


# # Feature importance
# 
# The coefficients (weights) of a linear model show us the importance of features. Let's start by displaying the weights of the days around Easter and the end of the year:
# 

# In[15]:


w = pd.Series(model.coef_, features) # weights of the linear regression
ws = w / preproc.scale_ # weight as it would be applied to the original feature (before scaling)

def plot_feature_weights_numbered(prefix):
    prefix_features = [f for f in features if f.startswith(prefix)]
    plt.figure(figsize=(12, 2))
    plt.bar([int(f[len(prefix):]) for f in prefix_features], ws[prefix_features])
    plt.title(f'Feature weights for {prefix}')
    plt.ylabel('weight')
    plt.xlabel('day')
    plt.show()
    
plot_feature_weights_numbered('easter')
plot_feature_weights_numbered('dec')
plot_feature_weights_numbered('jan')


# The weights have a clear real-world interpretation: If, for instance, the weekend feature `wd56` has a weight of $0.225$, this means that weekend sales are $e^{0.225} = 1.25$ times as high as non-weekend sales. Or sales on the 1st of January with a weight of $0.447$ are $e^{0.447} = 1.56$ times as high as if there were no end-of-year holiday.
# 
# The final display of this notebook shows the weights of the 30 most important features. Of course, you could as well show the 30 least important features and perhaps eliminate some of them.

# In[16]:


ws_sorted = ws.iloc[np.argsort(-np.abs(ws))]
ws_plot = ws_sorted.head(30)

plt.figure(figsize=(9, len(ws_plot) / 3))
plt.barh(np.arange(len(ws_plot)), ws_plot, color=ws_plot.apply(lambda ws: 'b' if ws >= 0 else 'y'))
plt.yticks(np.arange(len(ws_plot)), ws_plot.index)
plt.gca().invert_yaxis()
plt.title('Most important features')
plt.show()


# # GDP exponent
# 
# The weight of the GDP feature, which is 1.212, has a special significance: It means that the sales of every year are proportional to `GDP ** 1.212`. We'll use this exponent in the [LightGBM Quickstart notebook](https://www.kaggle.com/ambrosm/tpsjan22-06-lightgbm-quickstart).

# In[17]:


gdp_exponent = ws['gdp']
gdp_exponent


# In[ ]:




