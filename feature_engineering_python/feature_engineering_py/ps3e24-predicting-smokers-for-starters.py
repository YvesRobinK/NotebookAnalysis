#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Welcome to Binary Classification of Smoker Status competition! The problem in hands is that we have to predict smoker's status givven the health information in hand. The metric we will use is Area Under the ROC Curve. This notebook will cover the followings:
# - Loading Libraries and Datasets
# - Dataset Information
# - Adversarial Validation
# - Distribution of Numerical Features
# - Target Distribution
# - Correlation and Hierarchial Clustering
# - Machine Learning Preparation
# - Cross-Validation
# - Ensemble
# - Prediction and Submission
# 
# If you want to read the description of the original dataset, you can visit this page: https://www.kaggle.com/datasets/kukuroo3/body-signal-of-smoking/data

# # Loading Libraries and Datasets

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from category_encoders import OneHotEncoder, MEstimateEncoder, CatBoostEncoder
from sklearn import set_config
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from sklearn.preprocessing import FunctionTransformer, StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

sns.set_theme(style = 'white', palette = 'viridis')
pal = sns.color_palette('viridis')

pd.set_option('display.max_rows', 100)
set_config(transform_output = 'pandas')
pd.options.mode.chained_assignment = None


# In[2]:


train = pd.read_csv(r'/kaggle/input/playground-series-s3e24/train.csv', index_col = 'id')
test = pd.read_csv(r'/kaggle/input/playground-series-s3e24/test.csv', index_col = 'id')
orig_train = pd.read_csv(r'/kaggle/input/smoker-status-prediction/train_dataset.csv')


# # Dataset Info
# 
# Let's begin by taking a peek at our training dataset first

# In[3]:


train.head(10)


# In[4]:


desc = pd.DataFrame(index = list(train))
desc['type'] = train.dtypes
desc['count'] = train.count()
desc['nunique'] = train.nunique()
desc['%unique'] = desc['nunique'] / len(train) * 100
desc['null'] = train.isnull().sum()
desc['%null'] = desc['null'] / len(train) * 100
desc['min'] = train.min()
desc['max'] = train.max()
desc


# We can see that we have 159k rows and 23 columns, including our target here, which makes it 22 features.
# 
# Let's see the test dataset now.

# In[5]:


test.head(10)


# In[6]:


desc = pd.DataFrame(index = list(test))
desc['type'] = test.dtypes
desc['count'] = test.count()
desc['nunique'] = test.nunique()
desc['%unique'] = desc['nunique'] / len(test) * 100
desc['null'] = test.isnull().sum()
desc['%null'] = desc['null'] / len(test) * 100
desc['min'] = test.min()
desc['max'] = test.max()
desc


# On the test dataset, we have 106k rows. There is also no missing value on both.
# 
# Finally, let's try to see the original dataset.

# In[7]:


orig_train.head(10)


# In[8]:


desc = pd.DataFrame(index = list(orig_train))
desc['type'] = orig_train.dtypes
desc['count'] = orig_train.count()
desc['nunique'] = orig_train.nunique()
desc['%unique'] = desc['nunique'] / len(orig_train) * 100
desc['null'] = orig_train.isnull().sum()
desc['%null'] = desc['null'] / len(orig_train) * 100
desc['min'] = orig_train.min()
desc['max'] = orig_train.max()
desc


# Looks like everything perfect here. Now before moving on, if we see on all the tables above, we can find that there are four features that have consistent counts of unique values. Those are `hearing(left)`, `hearing(right)`, `Urine protein`, and `dental caries`. We can assume that those are actually categorical features, so let's try to group them for future convenience.

# In[9]:


categorical_features = ['hearing(left)', 'hearing(right)', 'Urine protein', 'dental caries']
numerical_features = list(test.drop(categorical_features, axis = 1))


# # Adversarial Validation
# 
# The purpose of adversarial validation is to check whether train and test dataset have similar distribution or not. If the validation gives ROC-AUC score of close to .5, we can say that both datasets are similar. However, if it's far from .5, both dataset have different distribution.
# 
# The reason we want to do this is to make sure that we can trust our CV score, since a trusted CV only comes from dataset with similar distribution.

# In[10]:


#thanks to @carlmcbrideellis
#https://www.kaggle.com/code/carlmcbrideellis/what-is-adversarial-validation

def adversarial_validation(dataset_1 = train, dataset_2 = test, label = 'Train-Test'):

    adv_train = dataset_1.drop('smoking', axis = 1)
    adv_test = dataset_2.copy()

    adv_train['is_test'] = 0
    adv_test['is_test'] = 1

    adv = pd.concat([adv_train, adv_test], ignore_index = True)

    adv_shuffled = adv.sample(frac = 1)

    adv_X = adv_shuffled.drop('is_test', axis = 1)
    adv_y = adv_shuffled.is_test

    skf = StratifiedKFold(n_splits = 5, random_state = 42, shuffle = True)

    val_scores = []
    predictions = np.zeros(len(adv))

    for fold, (train_idx, val_idx) in enumerate(skf.split(adv_X, adv_y)):
    
        adv_lr = XGBClassifier(random_state = 42)
        adv_lr.fit(adv_X.iloc[train_idx], adv_y.iloc[train_idx])
        
        val_preds = adv_lr.predict_proba(adv_X.iloc[val_idx])[:,1]
        predictions[val_idx] = val_preds
        val_score = roc_auc_score(adv_y.iloc[val_idx], val_preds)
        val_scores.append(val_score)
    
    fpr, tpr, _ = roc_curve(adv['is_test'], predictions)
    
    plt.figure(figsize = (10, 10), dpi = 300)
    sns.lineplot(x=[0, 1], y=[0, 1], linestyle="--", label="Indistinguishable Datasets")
    sns.lineplot(x=fpr, y=tpr, label="Adversarial Validation Classifier")
    plt.title(f'{label} Validation = {np.mean(val_scores):.5f}', weight = 'bold', size = 17)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()


# In[11]:


adversarial_validation()


# The result is very close to .5, therefore we can trust our CV.

# # Distribution of Numerical Features
# 
# Now that we have done taking a peek at the descriptive statistics of the datasets and doing adversarial validation, let's try to see the feature distribution this time.

# In[12]:


fig, ax = plt.subplots(6, 3, figsize = (10, 20), dpi = 300)
ax = ax.flatten()

for i, column in enumerate(numerical_features):
        
    sns.kdeplot(train[column], ax=ax[i], color=pal[0])
    sns.kdeplot(test[column], ax=ax[i], color=pal[2], warn_singular = False)
    sns.kdeplot(orig_train[column], ax=ax[i], color=pal[1])
    
    ax[i].set_title(f'{column} Distribution', size = 14)
    ax[i].set_xlabel(None)
    
fig.suptitle('Distribution of Feature\nper Dataset\n', fontsize = 24, fontweight = 'bold')
fig.legend(['Train', 'Test', 'Original Train'])
plt.tight_layout()


# It's safe to say that none of the features have normal distribution. You can try using PowerTransformer to normalize if you want.

# # Distribution of Categorical Features
# 
# Let's try to see the categorical features now.

# In[13]:


fig, ax = plt.subplots(4, 2, figsize = (16, 20), dpi = 300)
#ax = ax.flatten()

for i, column in enumerate(categorical_features):

    ax[i][0].pie(
        train[column].value_counts(), 
        shadow = True, 
        explode = [.1 for i in range(train[column].nunique())], 
        autopct = '%1.f%%',
        textprops = {'size' : 14, 'color' : 'white'}
    )

    sns.countplot(data = train, y = column, ax = ax[i][1], palette = 'viridis', order = train[column].value_counts().index)
    ax[i][1].yaxis.label.set_size(20)
    plt.yticks(fontsize = 12)
    ax[i][1].set_xlabel('Count in Train', fontsize = 15)
    ax[i][1].set_ylabel(f'{column}', fontsize = 15)
    plt.xticks(fontsize = 12)

fig.suptitle('Distribution of Categorical Features\nin Train Dataset\n\n\n\n', fontsize = 25, fontweight = 'bold')
plt.tight_layout()


# All categorical features above are heavily unbalanced, especially urine protein and hearing features. One category for each of those features have proportion of equal to or above 95%.

# # Target Distribution
# 
# We still need to check one last distribution: our target.

# In[14]:


fig, ax = plt.subplots(1, 2, figsize = (16, 5))
ax = ax.flatten()

ax[0].pie(
    train['smoking'].value_counts(), 
    shadow = True, 
    explode = [.1 for i in range(train.smoking.nunique())], 
    autopct = '%1.f%%',
    textprops = {'size' : 14, 'color' : 'white'}
)

sns.countplot(data = train, y = 'smoking', ax = ax[1], palette = 'viridis', order = train['smoking'].value_counts().index)
ax[1].yaxis.label.set_size(20)
plt.yticks(fontsize = 12)
ax[1].set_xlabel('Count', fontsize = 20)
ax[1].set_ylabel(None)
plt.xticks(fontsize = 12)

fig.suptitle('Smoking Distribution in Train Dataset', fontsize = 25, fontweight = 'bold')
plt.tight_layout()


# We can see above that the target distribution is quite balanced: 44% of total samples are smokers, while 56% aren't.

# # Correlation and Hierarchial Clustering
# 
# If we want to see the relationship between features, we can try calculating the correlation. If two features have negative correlation, it means that an increase of value in one feature will result in a decrease of value in another feature. On the other hand, positive correlation means that an increase of value in one feature will result in an increase of value in another. Let's try to take a look.

# In[15]:


def heatmap(dataset, label = None):
    corr = dataset.corr(method = 'spearman')
    plt.figure(figsize = (15, 15), dpi = 300)
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask = mask, cmap = 'viridis', annot = True, annot_kws = {'size' : 7})
    plt.title(f'{label} Dataset Correlation Matrix\n', fontsize = 25, weight = 'bold')
    plt.show()


# In[16]:


heatmap(train, 'Train')


# Seems like none of the features are duplicate of each others. Also, very few features are heavily correlated here. Let's try to cluster them now.

# In[17]:


def distance(data, label = ''):
    #thanks to @sergiosaharovsky for the fix
    corr = data.corr(method = 'spearman')
    dist_linkage = linkage(squareform(1 - abs(corr)), 'complete')
    
    plt.figure(figsize = (10, 8), dpi = 300)
    dendro = dendrogram(dist_linkage, labels=data.columns, leaf_rotation=90)
    plt.title(f'Feature Distance in {label} Dataset', weight = 'bold', size = 20)
    plt.show()


# In[18]:


distance(train, 'Train')


# # Machine Learning Preparation
# 
# This is where we start preparing everything if we want to start building machine learning models.

# In[19]:


X = pd.concat([train, orig_train])
y = X.pop('smoking')

seed = 42
splits = 5
skf = StratifiedKFold(n_splits = splits, random_state = seed, shuffle = True)
np.random.seed(seed)


# # Feature Engineering Function
# 
# This section builds up from the [topic discussed here](https://www.kaggle.com/competitions/playground-series-s3e24/discussion/450375). We know that age, weight, and height are actually binned numerical features, and that is why we will create a function that will fix any noises in the dataset.

# In[20]:


def categorize_numericals(x):
    x_copy = x.copy()
    for i in range(15, 90, 5):
        x_copy[(x_copy.age > i - 2.5) & (x_copy.age < i + 2.5)]['age'] = i
    for i in range(130, 195, 5):
        x_copy[
            (x_copy['height(cm)'] > i - 2.5) & (x_copy['height(cm)'] < i + 2.5)
        ]['height(cm)'] = i
    for i in range(30, 140, 5):
        x_copy[
            (x_copy['weight(kg)'] > i - 2.5) & (x_copy['weight(kg)'] < i + 2.5)
        ]['weight(kg)'] = i
    return x_copy

CategorizeNumericals = FunctionTransformer(categorize_numericals)


# We will also create a function that can fix the true value for blindness in eyesight features, though it probably won't help at all.

# In[21]:


def blindness_fix(x):
    x_copy = x.copy()
    x_copy[x_copy['eyesight(left)'] == 9.9]['eyesight(left)'] = 0
    x_copy[x_copy['eyesight(right)'] == 9.9]['eyesight(right)'] = 0
    return x_copy

BlindnessFix = FunctionTransformer(blindness_fix)


# # Cross Validation
# 
# Let's start by evaluating the performance of our model first. We will concatenate the original dataset only during the cross-validation process for robustness. All feature engineering function will be implemented inside the pipeline. We will also use Standard Scaler to improve the performance for parametric estimators such as Logistic Regression.

# In[22]:


def cross_val_score(estimator, cv = skf, label = '', include_original = False):
    
    X = train.copy()
    y = X.pop('smoking')
    
    #initiate prediction arrays and score lists
    val_predictions = np.zeros((len(X)))
    #train_predictions = np.zeros((len(sample)))
    train_scores, val_scores = [], []
    
    #training model, predicting prognosis probability, and evaluating metrics
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        
        model = clone(estimator)
        
        #define train set
        X_train = X.iloc[train_idx].reset_index(drop = True)
        y_train = y.iloc[train_idx].reset_index(drop = True)
        
        #define validation set
        X_val = X.iloc[val_idx].reset_index(drop = True)
        y_val = y.iloc[val_idx].reset_index(drop = True)
        
        if include_original:
            X_train = pd.concat([X_train, orig_train.drop('smoking', axis = 1)]).reset_index(drop = True)
            y_train = pd.concat([y_train, orig_train.smoking]).reset_index(drop = True)
        
        #train model
        model.fit(X_train, y_train)
        
        #make predictions
        train_preds = model.predict_proba(X_train)[:, 1]
        val_preds = model.predict_proba(X_val)[:, 1]
                  
        val_predictions[val_idx] += val_preds
        
        #evaluate model for a fold
        train_score = roc_auc_score(y_train, train_preds)
        val_score = roc_auc_score(y_val, val_preds)
        
        #append model score for a fold to list
        train_scores.append(train_score)
        val_scores.append(val_score)
    
    print(f'Val Score: {np.mean(val_scores):.5f} ± {np.std(val_scores):.5f} | Train Score: {np.mean(train_scores):.5f} ± {np.std(train_scores):.5f} | {label}')
    
    return val_scores, val_predictions


# In[23]:


score_list, oof_list = pd.DataFrame(), pd.DataFrame()

models = [
    ('log', LogisticRegression(random_state = seed, max_iter = 1000000, class_weight = 'balanced')),
    ('lda', LinearDiscriminantAnalysis()),
    ('gnb', GaussianNB()),
    ('bnb', BernoulliNB()),
    ('rf', RandomForestClassifier(random_state = seed)),
    ('et', ExtraTreesClassifier(random_state = seed)),
    ('xgb', XGBClassifier(random_state = seed)),
    ('lgb', LGBMClassifier(random_state = seed)),
    ('dart', LGBMClassifier(random_state = seed, boosting_type = 'dart')),
    ('cb', CatBoostClassifier(random_state = seed, verbose = 0)),
    ('gb', GradientBoostingClassifier(random_state = seed)),
    ('hgb', HistGradientBoostingClassifier(random_state = seed)),
]

for (label, model) in models:
    score_list[label], oof_list[label] = cross_val_score(
        make_pipeline(CategorizeNumericals, BlindnessFix, StandardScaler(), model),
        label = label,
        include_original = True
    )


# In[24]:


plt.figure(figsize = (8, 4), dpi = 300)
sns.barplot(data = score_list.reindex((-1 * score_list).mean().sort_values().index, axis = 1), palette = 'viridis', orient = 'h')
plt.title('Score Comparison', weight = 'bold', size = 20)
plt.show()


# We can see that CatBoost gives the best result out of all models.

# # Ensemble
# 
# Now let's try to define the weight of each model and then build a voting ensemble. We will use Ridge Classifier to define the weight by fitting it on OOF prediction and the true label.

# In[25]:


weights = RidgeClassifier(random_state = seed).fit(oof_list, train.smoking).coef_[0]
pd.DataFrame(weights, index = list(oof_list), columns = ['weight per model'])


# After defining the weight, we can start building a Voting Ensemble of our models.

# In[26]:


voter = VotingClassifier(models, weights = weights, voting = 'soft')
_ = cross_val_score(
    make_pipeline(StandardScaler(), voter),
    include_original = True
)


# # Prediction and Submission
# 
# Finally, let's train our chosen model on the whole train dataset and do prediction on the test dataset.

# In[27]:


model = make_pipeline(
    StandardScaler(),
    voter
)

model.fit(X, y)


# In[28]:


submission = test.copy()
submission['smoking'] = model.predict_proba(submission)[:, 1]

submission.smoking.to_csv('submission.csv')


# In[29]:


plt.figure(figsize = (15, 10), dpi = 300)
sns.kdeplot(submission.smoking, fill = True)
plt.title("Distribution of Smoking Probability", weight = 'bold', size = 25)
plt.show()


# Thanks for reading! Give upvote if you think it's helpful!
