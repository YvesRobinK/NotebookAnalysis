#!/usr/bin/env python
# coding: utf-8

# <h2><center> <span style = "font-family: Babas; font-size: 2em;"> CommonLit - Evaluate Student Summaries </span> </center></h2>
# <h4><center> <span style = "font-family: Babas; font-size: 2em; font-style: italic"> Starter Notebook (EDA + Baseline Modeling) with MultiOutputRegressor and RegressorChain </span> </center></h4>
# <h4><center> <span style = "font-family: Babas; font-size: 2em;"> Sugata Ghosh </span> </center></h4>

# ---
# ### Overview
# 
# The notebook serves as a starter notebook for the competition [**CommonLit - Evaluate Student Summaries**](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries).
# - We perform basic exploratory data analysis.
# - Relevant numerical features are extracted, and their relationships with the target variables are examined.
# - Some feature engineering steps are carried out with parameters that are set in the `config` dictionary towards the beginning of the notebook. One can experiment with these parameters and see which combination works best.
# - We consider two approaches to address the regression problem with multiple outputs: **direct multioutput regression** (implemented through the [**MultiOutputRegressor**](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html) class) and **chained multioutput regression** (implemented through the [**RegressorChain**](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.RegressorChain.html) class), the functionings of which are explained in the relevant sections.
# - We use a set of baseline algorithms for each approach and compare their performances via the cross-validation mean score of the evaluation metric, mean columnwise root mean squared error (MCRMSE), used in the competition.
# - The best regressor, thus obtained, is then fitted on the entire training set, and the fitted model is used to predict the targets in both the training set and the test set.
# ---

# ### Contents
# 
# - [**Data**](#Data)
# - [**Evaluation Metric**](#Evaluation-Metric)
# - [**Train-Test Split**](#Train-Test-Split)
# - [**Feature Extraction and Exploration**](#Feature-Extraction-and-Exploration)
# - [**Feature-Target Split**](#Feature-Target-Split)
# - [**Log Transformation**](#Log-Transformation)
# - [**Feature Scaling**](#Feature-Scaling)
# - [**Principal Component Analysis**](#Principal-Component-Analysis)
# - [**Direct Multioutput Regression**](#Direct-Multioutput-Regression)
# - [**Chained Multioutput Regression**](#Chained-Multioutput-Regression)
# - [**Prediction and Evaluation**](#Prediction-and-Evaluation)
# - [**Acknowledgement**](#Acknowledgement)
# - [**References**](#References)

# We use a configuration dictionary to determine certain steps in the feature engineering stage. In particular, we specify how we scale (normalize) the features and also the number of principal components in the PCA transformation of the features. We give the provision to not apply the specific steps as well.

# In[1]:


# Baseline configuration
config = {
    'scaling': 'standard', # 'none' or 'minmax' or 'standard'
    'pca_n_component': 4   # 'none' or integer between 0 and min(n_samples, n_features)
}


# In[2]:


# Importing libraries
import os, random, time, psutil, math
import numpy as np
from numpy import absolute, mean, std
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, RepeatedKFold
from sklearn.multioutput import MultiOutputRegressor, RegressorChain


# In[3]:


# Runtime and memory usage
start = time.time()
process = psutil.Process(os.getpid())


# In[4]:


# Setting random seeds
os.environ['PYTHONHASHSEED'] = '0'
random.seed(0)
np.random.seed(0)


# # Data

# The training prompts dataset contains four prompts with the following fields:
# - `prompt_id` - The ID of the prompt which links to the summaries file
# - `prompt_question` - The specific question the students are asked to respond to
# - `prompt_title` - A short-hand title for the prompt
# - `prompt_text` - The full prompt text

# In[5]:


# Importing the training prompts data
prompts = pd.read_csv('/kaggle/input/commonlit-evaluate-student-summaries/prompts_train.csv')
prompts.head()


# In[6]:


# Quick description of the training prompts data
prompts.info()


# In[7]:


# Example prompt
print(f"prompt_id: {prompts.loc[1]['prompt_id']}\n")
print(f"prompt_question: {prompts.loc[1]['prompt_question']}\n")
print(f"prompt_title: {prompts.loc[1]['prompt_title']}\n")
print(f"prompt_text: {prompts.loc[1]['prompt_text'][:500]}... (truncated to 500 characters)\n")
print(f"Number of characters in the full prompt text: {len(prompts.loc[1]['prompt_text'])}")


# The training summaries dataset contains four prompts with the following fields:
# - `student_id` - The ID of the student writer
# - `prompt_id` - The ID of the prompt which links to the prompt file
# - `text` - The full text of the student's summary
# - `content` - The content score for the summary (the first target)
# - `wording` - The wording score for the summary (the second target)

# In[8]:


# Importing the training summaries data
summaries = pd.read_csv('/kaggle/input/commonlit-evaluate-student-summaries/summaries_train.csv')
summaries.head()


# In[9]:


# Quick description of the training summaries data
summaries.info()


# In[10]:


# Example summary
print(f"student_id: {summaries.loc[2]['student_id']}\n")
print(f"text: {summaries.loc[2]['text']}\n")
print(f"content: {summaries.loc[2]['content']}\n")
print(f"wording: {summaries.loc[2]['wording']}")


# In[11]:


# Frequency comparison of prompts in the training summaries data
labels = summaries['prompt_id'].value_counts().index
sizes = summaries['prompt_id'].value_counts().values
plt.pie(sizes, labels = labels, autopct = '%1.1f%%')
center_circle = plt.Circle((0,0), 0.3, color = 'white')
p = plt.gcf()
p.gca().add_artist(center_circle)
plt.show()


# In[12]:


# Merging prompts and summaries data
merged = pd.merge(prompts, summaries, how = 'inner', on = 'prompt_id')
merged.head()


# In[13]:


# Shape of the datasets for sanity check
print(pd.Series({"Shape of training prompts set": prompts.shape,
                 "Shape of training summaries set": summaries.shape,
                 "Shape of the merged training set": merged.shape}).to_string())


# # Evaluation Metric

# Submissions are scored using mean columnwise root mean squared error (MCRMSE):
# 
# $$ \textrm{MCRMSE} = \frac{1}{n} \sum_{j=1}^n \left(\frac{1}{m} \sum_{i=1}^m \left(y_{ij} - \hat{y}_{ij}\right)^2\right)^{1/2}, $$
# 
# where $n$ is the number of target columns, $m$ is the number of observations, and $y_{ij}$ and $\hat{y}_{ij}$ are the actual and the predicted values, respectively, of the $i$th observation of the $j$th target column, for $i = 1, 2, \ldots, m$ and $j = 1, 2, \ldots, n$. In other words, it is simply the [**arithmatic mean**](https://en.wikipedia.org/wiki/Arithmetic_mean) of [**root mean squared error**](https://en.wikipedia.org/wiki/Root-mean-square_deviation) (RMSE) of $n$ target columns. In the present work, we have $n = 2$, as there are two target columns, namely `content` and `wording`.

# In[14]:


# Manual computation of MCRMSE
def mcrmse_score(Y_true, Y_pred):
    """
    Computes the MCRMSE score given true values and predicted values of the target columns
    Arg:
        Y_true (ndarray, shape (m, n)): array of true values of n target columns
        Y_pred (ndarray, shape (m, n)): array of predicted values of n target columns
    Returns:
        mcrmse (scalar): MCRMSE score for Y_true and Y_pred
    """
    Y_diff = Y_true - Y_pred
    mse_arr = (Y_diff * Y_diff).mean(axis = 0)
    mcrmse = np.sqrt(mse_arr).mean()
    return mcrmse

Y_true, Y_pred = np.array([[1, 5], [50, 100]]), np.array([[2, 4], [40, 110]])
print(f"MCRMSE: {mcrmse_score(Y_true, Y_pred)}")


# In[15]:


# Sanity check through sklearn.metrics.mean_squared_error
print(f"MCRMSE: {np.sqrt(mean_squared_error(Y_true, Y_pred))}")


# # Train-Test Split

# In[16]:


# Train-Test split
train, test = train_test_split(merged, test_size = 0.2, random_state = 0, shuffle = True)


# # Target Variables

# In[17]:


# Distributions of the target variables and their relationship (training set)
fig, ax = plt.subplots(1, 3, figsize = (15, 5))
c, w, num_bin = train['content'], train['wording'], math.floor(2 * (len(train)**(1/3))) + 1
ax[0].hist(c, bins = num_bin), ax[1].hist(w, bins = num_bin), ax[2].scatter(c, w, marker = '.')
ax[0].set_xlabel("Content"), ax[1].set_xlabel("Wording"), ax[2].set_xlabel("Content")
ax[0].set_ylabel("Count"), ax[1].set_ylabel("Count"), ax[2].set_ylabel("Wording")
t0 = f"Mean: {c.mean():.2f}, SD: {c.std():.2f}, Skew: {c.skew():.2f}, Kurt: {c.kurt():.2f}"
t1 = f"Mean: {w.mean():.2f}, SD: {w.std():.2f}, Skew: {w.skew():.2f}, Kurt: {w.kurt():.2f}"
t2 = f"Correlation Coefficient: {c.corr(w):.2f}"
ax[0].set_title(t0), ax[1].set_title(t1), ax[2].set_title(t2)
plt.tight_layout()
plt.show()


# In[18]:


# Distributions of the target variables and their relationship (test set)
fig, ax = plt.subplots(1, 3, figsize = (15, 5))
c, w, num_bin = test['content'], test['wording'], math.floor(2 * (len(test)**(1/3))) + 1
ax[0].hist(c, bins = num_bin), ax[1].hist(w, bins = num_bin), ax[2].scatter(c, w, marker = '.')
ax[0].set_xlabel("Content"), ax[1].set_xlabel("Wording"), ax[2].set_xlabel("Content")
ax[0].set_ylabel("Count"), ax[1].set_ylabel("Count"), ax[2].set_ylabel("Wording")
t0 = f"Mean: {c.mean():.2f}, SD: {c.std():.2f}, Skew: {c.skew():.2f}, Kurt: {c.kurt():.2f}"
t1 = f"Mean: {w.mean():.2f}, SD: {w.std():.2f}, Skew: {w.skew():.2f}, Kurt: {w.kurt():.2f}"
t2 = f"Correlation Coefficient: {c.corr(w):.2f}"
ax[0].set_title(t0), ax[1].set_title(t1), ax[2].set_title(t2)
plt.tight_layout()
plt.show()


# # Feature Extraction and Exploration

# In[19]:


# Distribution of a summary text attribute and its relationship with content and wording
def feature_plots(x, name):
    fig, ax = plt.subplots(1, 3, figsize = (15, 5))
    c, w, num_bin = train['content'], train['wording'], math.floor(2 * (len(x)**(1/3))) + 1
    ax[0].hist(x, bins = num_bin)
    ax[1].scatter(x, c, marker = '.'), ax[2].scatter(x, w, marker = '.')
    ax[0].set_xlabel(name), ax[1].set_xlabel(name), ax[2].set_xlabel(name)
    ax[0].set_ylabel("Count"), ax[1].set_ylabel("Content"), ax[2].set_ylabel("Wording")
    t0 = f"Mean: {x.mean():.2f}, SD: {x.std():.2f}, Skew: {x.skew():.2f}, Kurt: {x.kurt():.2f}"
    t1 = f"Correlation Coefficient: {x.corr(c):.2f}"
    t2 = f"Correlation Coefficient: {x.corr(w):.2f}"
    ax[0].set_title(t0), ax[1].set_title(t1), ax[2].set_title(t2)
    plt.tight_layout()
    plt.show()


# We consider several numerical attributes of the texts and check their relationship with the target variables `content` and `wording`.

# In[20]:


# Number of characters in summary texts
num_char_train = train['text'].str.len()
feature_plots(x = num_char_train, name = "Number of characters")


# In[21]:


# Number of words in summary texts
num_words_train = train['text'].str.split().map(lambda x: len(x))
feature_plots(x = num_words_train, name = "Number of words")


# In[22]:


# Ratio of stopwords in summary texts
stops = stopwords.words("english")
num_stopwords_train = train['text'].apply(lambda x: len(set(x.split()) & set(stops)))
ratio_stopwords_train = num_stopwords_train / num_words_train
feature_plots(x = ratio_stopwords_train, name = "Ratio of stopwords")


# In[23]:


# Ratio of specific parts of speech in summary texts
regexp = RegexpTokenizer("[\w']+")
def keep_pos(text):
    tokens = regexp.tokenize(text)
    tokens_tagged = nltk.pos_tag(tokens)
    keep_tags = ['NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'VBD', 'VBG', 'VBZ', 'WDT', 'WRB']
    keep_words = [x[0] for x in tokens_tagged if x[1] in keep_tags]
    return " ".join(keep_words)
num_pos_train = train['text'].apply(keep_pos).str.split().map(lambda x: len(x))
ratio_pos_train = num_pos_train / num_words_train
feature_plots(x = ratio_pos_train, name = "Ratio of selected POS")


# In[24]:


# Number of contractions in summary texts
contractions_url = 'https://raw.githubusercontent.com/sugatagh/E-commerce-Text-Classification/main/JSON/english_contractions.json'
contractions_dict = pd.read_json(contractions_url, typ = 'series')
contractions_list = list(contractions_dict.keys())
num_contractions_train = train['text'].apply(lambda x: len(set(x.split()) & set(contractions_list)))
feature_plots(x = num_contractions_train, name = "Number of contractions")


# In[25]:


# Average word-length in summary texts
word_length_train = train['text'].str.split().apply(lambda x : [len(i) for i in x])
avg_word_length_train = word_length_train.map(lambda x: np.mean(x))
feature_plots(x = avg_word_length_train, name = "Average word-length")


# In[26]:


# Training set
train_ = pd.DataFrame()
train_['num_words'] = num_words_train
train_['ratio_stopwords'] = ratio_stopwords_train
train_['num_contractions'] = num_contractions_train
train_['ratio_pos'] = ratio_pos_train
train_['content'], train_['wording'] = train['content'], train['wording']
train_.head()


# In[27]:


# Test set
test_ = pd.DataFrame()

num_words_test = test['text'].str.split().map(lambda x: len(x))
num_stopwords_test = test['text'].apply(lambda x: len(set(x.split()) & set(stops)))
ratio_stopwords_test = num_stopwords_test / num_words_test
num_contractions_test = test['text'].apply(lambda x: len(set(x.split()) & set(contractions_list)))
num_pos_test = test['text'].apply(keep_pos).str.split().map(lambda x: len(x))
ratio_pos_test = num_pos_test / num_words_test

test_['num_words'] = num_words_test
test_['ratio_stopwords'] = ratio_stopwords_test
test_['num_contractions'] = num_contractions_test
test_['ratio_pos'] = ratio_pos_test
test_['content'], test_['wording'] = test['content'], test['wording']
test_.head()


# # Feature-Target Split

# In[28]:


# Feature-targets split (training set)
X_train = train_.drop(['content', 'wording'], axis = 1)
y_train = train_[['content', 'wording']]
print(f"Feature shape: {X_train.shape}")
print(f"Target shape : {y_train.shape}")


# In[29]:


# Feature-targets split (test set)
X_test = test_.drop(['content', 'wording'], axis = 1)
y_test = test_[['content', 'wording']]
print(f"Feature shape: {X_test.shape}")
print(f"Target shape : {y_test.shape}")


# # Log Transformation

# We observe that the feature `num_words` has high positive skewness and apply the transformation $x \mapsto \log{x}$ to it.

# In[30]:


# Log transformation
def log_transform(X_train_in, X_test_in, cols_transform):
    """
    Selects skewed columns to transform
    Args:
        X_train_in (DataFrame, shape (m1, n)): Input training features
        X_test_in (DataFrame, shape (m2, n)) : Input test features
        cols_transform (list, shape (r,))    : List of columns to be transformed (r <= n)
            
    Returns:
        X_train_out (DataFrame, shape (m1, n)): output training features
        X_test_out (DataFrame, shape (m2, n)) : output test features
    """
    X_train_out, X_test_out = X_train_in.copy(deep = True), X_test_in.copy(deep = True)
    for col in cols_transform:
        assert col in X_train_out.columns, f"{col} is not found in the set of training features"
        X_train_out[col] = np.log(X_train_out[col])
        X_test_out[col] = np.log(X_test_out[col])
    return X_train_out, X_test_out


# In[31]:


# Log transformation of 'num_words'
X_train, X_test = log_transform(X_train, X_test, ['num_words'])
X_train.head()


# # Feature Scaling

# In[32]:


# Min-max normalization
def minmax_scaling(df_train_in, df_test_in):
    """
    Applies min-max scaling to selected columns
    Args:
        df_train_in (DataFrame, shape (m, n)): input training dataframe
        df_test_in (DataFrame, shape (m, n)) : input test dataframe
        
    Returns:
        df_train_out (DataFrame, shape (m, n)): output training dataframe
        df_test_out (DataFrame, shape (m, n)) : output test dataframe
    """
    df_train_out, df_test_out = df_train_in.copy(deep = True), df_test_in.copy(deep = True)
    cols = [col for col in df_train_in.columns if df_train_in[col].nunique() > 1]
    for col in cols:
        min_, max_ = df_train_out[col].min(), df_train_out[col].max()
        df_train_out[col] = (df_train_out[col] - min_) / (max_ - min_)
        df_test_out[col] = (df_test_out[col] - min_) / (max_ - min_)
    return df_train_out, df_test_out


# In[33]:


# Standardization
def standard_scaling(df_train_in, df_test_in):
    """
    Applies standardization to selected columns
    Args:
        df_train_in (DataFrame, shape (m, n)): input training dataframe
        df_test_in (DataFrame, shape (m, n)) : input test dataframe
        
    Returns:
        df_train_out (DataFrame, shape (m, n)): output training dataframe
        df_test_out (DataFrame, shape (m, n)) : output test dataframe
    """
    df_train_out, df_test_out = df_train_in.copy(deep = True), df_test_in.copy(deep = True)
    cols = [col for col in df_train_in.columns if df_train_in[col].nunique() > 1]
    for col in cols:
        mean_, std_ = df_train_out[col].mean(), df_train_out[col].std()
        df_train_out[col] = (df_train_out[col] - mean_) / std_
        df_test_out[col] = (df_test_out[col] - mean_) / std_
    return df_train_out, df_test_out


# If `config['scaling']` is set to `'minmax'`, we apply [**min-max normalization**](https://en.wikipedia.org/wiki/Feature_scaling#Methods). If it is set to `'standard'`, then we apply [**standardization**](https://en.wikipedia.org/wiki/Standard_score). Otherwise, we do not apply [**feature scaling**](https://en.wikipedia.org/wiki/Feature_scaling).

# In[34]:


# Feature scaling
if config['scaling'] == 'minmax':
    X_train, X_test = minmax_scaling(X_train, X_test)
elif config['scaling'] == 'standard':
    X_train, X_test = standard_scaling(X_train, X_test)
else:
    None

X_train.head()


# # Principal Component Analysis

# In[35]:


# Correlation matrix of training features
X_train.corr()


# If `config['pca_n_component']` is not set to `'none'`, we fit [**principal component analysis**](https://en.wikipedia.org/wiki/Principal_component_analysis) (PCA), with `config['pca_n_component']` number of components, on the training set and use it to transform both the training features and the test features.

# In[36]:


# PCA-transformation of training features and test features
if config['pca_n_component'] != 'none':
    pca = PCA(n_components = config['pca_n_component'])
    pca.fit(X_train)
    X_train = pd.DataFrame(pca.transform(X_train))
    X_test = pd.DataFrame(pca.transform(X_test))
    
X_train.head()


# In[37]:


# Correlation matrix of training features after PCA-transformation
X_train.corr()


# # Direct Multioutput Regression

# Let us consider a regression problem with $r$ output variables. The direct approach to solving this problem is to divide it into $r$ separate single-output regression problems:
# 
# - Problem $1$: Given input $(x_1, x_2, \ldots, x_n)$, predict $y_1$.
# - Problem $2$: Given input $(x_1, x_2, \ldots, x_n)$, predict $y_2$.
# - $\cdots\cdots\cdots\cdots\cdots\cdots\cdots\cdots\cdots\cdots\cdots\cdots\cdots\cdots\cdots$
# - Problem $r$: Given input $(x_1, x_2, \ldots, x_n)$, predict $y_r$.
# 
# The performance measures associated with the predictions in these two problems can then be combined in a suitable manner to produce a single aggregated value. In this work, we have a regression problem with two target variables $(r = 2)$: `content` and `wording`. Thus, we divide it into two separate problems, one for predicting `content` and the other for predicting `wording`.
# 
# This approach, while offering a simple solution to multiple-output regression problems, crucially assumes that the target variables are independent of each other. In practice, this may not be a feasible assumption. In fact, the strong positive correlation between `content` and `wording` already indicates against this supposition. Despite this limitation, this approach can produce effective predictions for a range of problems. Even when the target variables do not appear to be independent, the method can be applied to set up a baseline performance measure, as a benchmark for other approaches which consider the dependence structure among the targets. The approach is supported by the **MultiOutputRegressor** class in the scikit-learn library. The class takes a regression model as an argument and creates one instance of the input model for each output in the problem. The class is specifically built to extend models that do not natively support regression problems with multiple targets.

# In[38]:


# Repeated K-Fold cross validator and scoring
cv = RepeatedKFold(n_splits = 10, n_repeats = 10, random_state = 0)
scoring = 'neg_root_mean_squared_error'


# Note that we do not require the `MultiOutputRegressor` wrapper for quite a few models, such as **linear regression**, **k-nearest neighbors** and **decision tree**, as the corresponding algorithms are programmed to handle multiple targets. However, we feed all the candidate models to the `MultiOutputRegressor` wrapper in order to keep the same overall structure of the code.

# In[39]:


# Candidate models
models = {
    'Linear Regression'  : LinearRegression(),
    'k-Nearest Neighbors': KNeighborsRegressor(),
    'Decision Tree'      : DecisionTreeRegressor(random_state = 0),
    'Linear SVM'         : LinearSVR(max_iter = 2000),
    'Random Forest'      : RandomForestRegressor(random_state = 0),
    'SGD'                : SGDRegressor(random_state = 0),
    'Ridge'              : Ridge(random_state = 0),
    'XGBoost'            : XGBRegressor(random_state = 0),
    'AdaBoost'           : AdaBoostRegressor(random_state = 0),
    'ExtreTrees'         : ExtraTreesRegressor(random_state = 0)
}


# In[40]:


# Initial counter
best_regressor = MultiOutputRegressor(estimator = LinearRegression(), n_jobs = -1)
best_score = 100 # anything big


# In[41]:


# Performance of candidate models
for name, model in zip(models.keys(), models.values()):
    regressor = MultiOutputRegressor(model, n_jobs = -1)
    scores = cross_val_score(regressor, X_train, y_train, scoring = scoring, cv = cv, n_jobs = -1)
    mean_, std_ = (-1 * scores).mean(), (-1 * scores).std()
    if mean_ <= best_score:
        best_regressor = regressor
        best_score = mean_
    print(name)
    print(f"Average RMSE: {mean_:.3f} (SD: {std_:.3f})\n")


# # Chained Multioutput Regression

# We consider an approach that attempts to capture the dependence structure among the output variables. This is done by creating a linear sequence of single-output regression problems:
# 
# - Problem $1$: Given input $(x_1, x_2, \ldots, x_n)$, predict $y_1$.
# - Problem $2$: Given input $(x_1, x_2, \ldots, x_n)$ and $y_1$, predict $y_2$.
# - Problem $3$: Given input $(x_1, x_2, \ldots, x_n)$ and $(y_1, y_2)$, predict $y_3$.
# - $\cdots\cdots\cdots\cdots\cdots\cdots\cdots\cdots\cdots\cdots\cdots\cdots\cdots\cdots\cdots\cdots\cdots\cdots\cdots\cdots$
# - Problem $r$: Given input $(x_1, x_2, \ldots, x_n)$ and $(y_1, y_2, \ldots, y_{r-1})$, predict $y_r$.
# 
# One important aspect of this sequential scheme is the prediction order of the output variables. For instance, we can predict $y_1$ based on the input and then predict $y_2$ based on the input as well as the predicted value of $y_1$. Alternately, we can predict $y_2$ based on the input and then predict $y_1$ based on the input as well as the predicted value of $y_2$. Thus, we shall have to specify an ordered list of the output variables, which can be mapped from a permutation of $(0,1,\ldots,r-1)$, where $r$ is the number of output variables. In our problem we are required to predict `content` and `wording` given the inputs, thus $r = 2$. The task then can be partitioned into two dependent single-output regression problems as follows:
# 
# - Problem $1$: Given input data, predict `content`.
# - Problem $2$: Given input data and the predicted value of `content`, predict `wording`.
# 
# This can be achieved using the **RegressorChain** class in the scikit-learn library. The default order of the outputs is based on the order in which they appear in the dataset. It can also be manually specified through the `order` argument. For example, `order = [0, 1]` would first predict `content` and then `wording`, whereas `order = [1, 0]` would first predict `wording` and then `content`. In the latter case, the subproblems change as follows:
# 
# - Problem $1$: Given input data, predict `wording`.
# - Problem $2$: Given input data and the predicted value of `wording`, predict `content`.
# 
# Sometimes the ordering is trivial given the nature of the outputs, when one output depends on the other and not the other way around. For example, if we have a regression problem where we have to predict *marks in mathematics* and *marks in theoretical physics*, one can argue that we should predict *marks in mathematics* first and then *marks in theoretical physics*, as a student can use the knowledge in mathematics to score more in theoretical physics. The opposite direction, however, is not so apparent. In general, though, one may have to try different permutations and see which order works best for a given problem.

# In[42]:


# Performance of candidate models
for name, model in zip(models.keys(), models.values()):
    regressor = RegressorChain(model, order = [0, 1])
    scores = cross_val_score(regressor, X_train, y_train, scoring = scoring, cv = cv, n_jobs = -1)
    mean_, std_ = (-1 * scores).mean(), (-1 * scores).std()
    if mean_ <= best_score:
        best_regressor = regressor
        best_score = mean_
    print(name)
    print(f"Average RMSE: {mean_:.3f} (SD: {std_:.3f})\n")


# In[43]:


# Performance of candidate models
for name, model in zip(models.keys(), models.values()):
    regressor = RegressorChain(model, order = [1, 0])
    scores = cross_val_score(regressor, X_train, y_train, scoring = scoring, cv = cv, n_jobs = -1)
    mean_, std_ = (-1 * scores).mean(), (-1 * scores).std()
    if mean_ <= best_score:
        best_regressor = regressor
        best_score = mean_
    print(name)
    print(f"Average RMSE: {mean_:.3f} (SD: {std_:.3f})\n")


# # Prediction and Evaluation

# In direct multioutput regression and in each target permutation of chained multioutput regression, we observe the following:
# - The linear models (**linear regression**, **SGD regression**, and **Ridge regression**) perform better than other models.
# - The **decision tree** regressor performs poorly compared to the rest of the models.

# In[44]:


# Best regressor
display(best_regressor)
print(f"Best cross-validation score: {best_score}")


# In[45]:


# Prediction and evaluation
regressor = best_regressor
regressor.fit(X_train, y_train)
y_train_pred = regressor.predict(X_train)
y_test_pred = regressor.predict(X_test)
print(f"Training MCRMSE: {mcrmse_score(y_train, y_train_pred)}")
print(f"Test MCRMSE    : {mcrmse_score(y_test, y_test_pred)}")


# # Acknowledgement
# 
# - [**Alphabetical list of part-of-speech tags used in the Penn Treebank Project**](https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html)
# - [**CommonLit - Evaluate Student Summaries**](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries) competition
# - [**How to Develop Multi-Output Regression Models with Python**](https://machinelearningmastery.com/multi-output-regression-models-with-python/) by [**Jason Brownlee**](https://machinelearningmastery.com/about)

# # References
# 
# - [**Arithmatic mean**](https://en.wikipedia.org/wiki/Arithmetic_mean)
# - [**Feature scaling**](https://en.wikipedia.org/wiki/Feature_scaling)
# - [**Min-max normalization**](https://en.wikipedia.org/wiki/Feature_scaling#Methods)
# - [**MultiOutputRegressor**](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html)
# - [**Principal component analysis**](https://en.wikipedia.org/wiki/Principal_component_analysis)
# - [**RegressorChain**](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.RegressorChain.html)
# - [**Root mean squared error**](https://en.wikipedia.org/wiki/Root-mean-square_deviation)
# - [**Standardization**](https://en.wikipedia.org/wiki/Standard_score)

# In[46]:


# Runtime and memory usage
stop = time.time()
runtime = stop - start
memory_usage = process.memory_info()[0] / (1024*1024)
print(pd.Series({"Process runtime": "{:.2f} seconds".format(float(runtime)),
                 "Process memory usage": "{:.2f} MB".format(float(memory_usage))}).to_string())

