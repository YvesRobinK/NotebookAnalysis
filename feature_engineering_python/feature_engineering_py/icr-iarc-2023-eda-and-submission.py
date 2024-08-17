#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('wget http://bit.ly/3ZLyF82 -O CSS.css -q')
    
from IPython.core.display import HTML
with open('./CSS.css', 'r') as file:
    custom_css = file.read()

HTML(custom_css)


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#FF5C19; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #FF5C19">Libraries</p>

# In[2]:


import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm, skew, kurtosis
from scipy.special import expit

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from pandas.io.formats.style import Styler
import math

import warnings
warnings.filterwarnings('ignore')

tqdm.pandas()

rc = {
    "axes.facecolor": "#F8F8F8",
    "figure.facecolor": "#F8F8F8",
    "axes.edgecolor": "#000000",
    "grid.color": "#EBEBE7" + "30",
    "font.family": "serif",
    "axes.labelcolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "grid.alpha": 0.4
}

sns.set(rc=rc)
palette = ['#302c36', '#037d97', '#E4591E', '#C09741',
           '#EC5B6D', '#90A6B1', '#6ca957', '#D8E3E2']

from colorama import Style, Fore
blk = Style.BRIGHT + Fore.BLACK
gld = Style.BRIGHT + Fore.YELLOW
grn = Style.BRIGHT + Fore.GREEN
red = Style.BRIGHT + Fore.RED
blu = Style.BRIGHT + Fore.BLUE
res = Style.RESET_ALL

import xgboost as xgb
from xgboost.callback import EarlyStopping
import lightgbm as lgbm
from lightgbm import log_evaluation, early_stopping, record_evaluation
import catboost as cb
from catboost import Pool
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn import model_selection
from sklearn import metrics

from sklearn.metrics import confusion_matrix
from sklearn.metrics import RocCurveDisplay
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


# <p align="right">
#   <img src="https://drive.google.com/uc?export=view&id=1QK4yx-SMfv5OBPPLRHvyO5p5-d6jZu-s"/>
# </p>

# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#FF5C19; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #FF5C19">Intro</p>

# This Kaggle notebook is aimed at providing a comprehensive exploratory data analysis (EDA) for the given dataset, with the ultimate goal of making informed decisions and recommendations before diving into modeling. 
# >Through this EDA, we will gain a deeper understanding of the data structure, missing values, relationships between variables, and any patterns or anomalies that could impact our modeling process. By performing a thorough EDA, we can identify potential roadblocks and make necessary pre-processing decisions that will improve the performance and accuracy of our models. So, buckle up, and let's embark on this journey of discovering insights and valuable information from the data to drive better modeling decisions.

# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#FF5C19; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #FF5C19">Data</p>

# **The dataset** comprises over fifty anonymized health characteristics linked to three age-related conditions. Your goal is to predict whether a subject has or has not been diagnosed with one of these conditions -- a binary classification problem. The goal is to predict `1` or `0` (indicates the subject has been diagnosed with one of the three conditions respectively).
#  
# There are 56 independent variables (excluding `id`):
# 
# <ul>
# <li><strong>train.csv</strong> - The training set.<ul>
# <li><code>Id</code> Unique identifier for each observation.</li>
# <li><code>AB</code>-<code>GL</code> Fifty-six anonymized health characteristics. All are numeric except for <code>EJ</code>, which is categorical.</li>
# <li><code>Class</code> A binary target: <code>1</code> indicates the subject has been diagnosed with one of the three conditions, <code>0</code> indicates they have not.</li></ul></li>
# <li><strong>test.csv</strong> - The test set. Your goal is to predict the probability that a subject in this set belongs to each of the two classes.</li>
# <li><strong>greeks.csv</strong> - Supplemental metadata, only available for the training set.<ul>
# <li><code>Alpha</code> Identifies the type of age-related condition, if present.<ul>
# <li><code>A</code> No age-related condition. Corresponds to class <code>0</code>.</li>
# <li><code>B</code>, <code>D</code>, <code>G</code> The three age-related conditions. Correspond to class <code>1</code>.</li></ul></li>
# <li><code>Beta</code>, <code>Gamma</code>, <code>Delta</code> Three experimental characteristics.</li>
# <li><code>Epsilon</code> The date the data for this subject was collected. Note that all of the data in the test set was collected after the training set was collected.</li></ul></li>
# <li><strong>sample_submission.csv</strong> - A sample submission file in the correct format. See the <a target="_blank" href="https://www.kaggle.com/competitions/icr-identify-age-related-conditions/overview/code-requirements"><strong>Evaluation</strong></a> page for more details.</li>
# </ul>
# 
# Target varibale:
# * `Class`:  `0` conditions are not diagnosed, `1` conditions are diagnosed.
# 
# **Metrics**:
# * [LogLoss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html)

# In[3]:


PATH_TRAIN = '/kaggle/input/icr-identify-age-related-conditions/train.csv'
PATH_TEST = '/kaggle/input/icr-identify-age-related-conditions/test.csv'
PATH_SUB = '/kaggle/input/icr-identify-age-related-conditions/sample_submission.csv'

train =  pd.read_csv(PATH_TRAIN).drop(columns='Id')
test =   pd.read_csv(PATH_TEST).drop(columns='Id')

train['EJ'] = train['EJ'].map({'A': 0, 'B': 1})
test['EJ'] = test['EJ'].map({'A': 0, 'B': 1})


# In[4]:


print(f'{blk}[INFO] Shapes:'
      f'{blk}\n[+] train  -> {red}{train.shape}'
      f'{blk}\n[+] test   ->  {red}{test.shape}\n')

print(f'{blk}[INFO] Any missing values:'
      f'{blk}\n[+] train  -> {red}{train.isna().any().any()}'
      f'{blk}\n[+] test   -> {red}{test.isna().any().any()}')


# **Note**:
# 
# * There are some missing values in the train data, let's explore where and how many:

# In[5]:


missing = train.isna().sum().reset_index()
missing.columns = ['columns', 'missing_count']

print(f'{blk}[INFO] Any missing values:'
      f'\n\n{red}{missing[missing.missing_count > 0]}{res}')


# **Note**:
# * We are to figure out how to handle it later on (Mean/Median/Mode imputation or Forward/Backward fill or something else)

# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#FF5C19; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #FF5C19">Train</p>

# This notebook provides a code snippet of how effectively display dataframes in a tidy format using Pandas Styler class.
# 
# We are going to leverage CSS styling language to manipulate many parameters including colors, fonts, borders, background, format and make our tables interactive.
# 
# 
# **Reference**:
# * [Pandas Table Visualization](https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html).

# At this point, you will require a certain level of understanding in web development.
# 
# Primarily, you will have to modify the CSS of the `td`, `tr`, and `th` tags.
# 
# **You can refer** to the following materials to learn **HTML/CSS**:
# * [w3schools HTML Tutorial](https://www.w3schools.com/html/default.asp)
# * [w3schools CSS Reference](https://www.w3schools.com/cssref/index.php)

# In[6]:


def magnify(is_test: bool = False):
        base_color = '#FF5C19'
        if is_test:
            highlight_target_row = []
        else:
            highlight_target_row = [dict(selector='tr:last-child',
                                         props=[('background-color', f'{base_color}'+'20')])]
            
        return [dict(selector="th",
                     props=[("font-size", "11pt"),
                            ('background-color', f'{base_color}'),
                            ('color', 'white'),
                            ('font-weight', 'bold'),
                            ('border-bottom', '0.1px solid white'),
                            ('border-left', '0.1px solid white'),
                            ('text-align', 'right')]),

                dict(selector='th.blank.level0', 
                    props=[('font-weight', 'bold'),
                           ('border-left', '1.7px solid white'),
                           ('background-color', 'white')]),

                dict(selector="td",
                     props=[('padding', "0.5em 1em"),
                            ('text-align', 'right')]),

                dict(selector="th:hover",
                     props=[("font-size", "14pt")]),

                dict(selector="tr:hover td:hover",
                     props=[('max-width', '250px'),
                            ('font-size', '14pt'),
                            ('color', f'{base_color}'),
                            ('font-weight', 'bold'),
                            ('background-color', 'white'),
                            ('border', f'1px dashed {base_color}')]),
                
                 dict(selector="caption",
                      props=[(('caption-side', 'bottom'))])] + highlight_target_row

def stylize_min_max_count(pivot_table):
    """Waps the min_max_count pivot_table into the Styler.

        Args:
            df: |min_train| max_train |min_test |max_test |top5_counts_train |top_10_counts_train|

        Returns:
            s: the dataframe wrapped into Styler.
    """
    s = pivot_table
    # A formatting dictionary for controlling each column precision (.000 <-). 
    di_frmt = {(i if i.startswith('m') else i):
              ('{:.3f}' if i.startswith('m') else '{:}') for i in s.columns}

    s = s.style.set_table_styles(magnify(True))\
        .format(di_frmt)\
        .set_caption(f"The train and test datasets min, max, top5 values side by side (hover to magnify).")
    return s
  
    
def stylize_describe(df: pd.DataFrame, dataset_name: str = 'train', is_test: bool = False) -> Styler:
    """Applies .descibe() method to the df and wraps it into the Styler.
    
        Args:
            df: any dataframe (train/test/origin)
            dataset_name: default 'train'
            is_test: the bool parameter passed into magnify() function
                     in order to control the highlighting of the last row.
                     
        Returns:
            s: the dataframe wrapped into Styler.
    """
    s = df.describe().T.sort_values(by='mean')
    # A formatting dictionary for controlling each column precision (.000 <-). 
    di_frmt = {(i if i == 'count' else i):
              ('{:.0f}' if i == 'count' else '{:.3f}') for i in s.columns}
    
    s = s.style.set_table_styles(magnify(is_test))\
        .format(di_frmt)\
        .set_caption(f"The {dataset_name} dataset descriptive statistics (hover to magnify).")
    return s

def stylize_simple(df: pd.DataFrame, caption: str) -> Styler:
    """Waps the min_max_count pivot_table into the Styler.

        Args:
            df: any dataframe (train/test/origin)

        Returns:
            s: the dataframe wrapped into Styler.
    """
    s = df
    s = s.style.set_table_styles(magnify(True)).set_caption(f"{caption}")
    return s

display(stylize_simple(train.head(4), 'The train dataset 3 top rows (hover to magnify).'))
display(stylize_simple(test, 'The test dataset (hover to magnify).'))
display(stylize_describe(train))


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#FF5C19; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #FF5C19">Min Max and Counts</p>

# In[7]:


cm = sns.light_palette('#FF5C19', as_cmap=True)

counts_tr = pd.Series({ft: [train[ft].value_counts().round(3).iloc[:5].to_dict()] for ft in train.columns}, name='top_5_counts_train')
counts_te = pd.Series({ft: [test[ft].value_counts().round(3).iloc[:5].to_dict()] for ft in test.columns}, name='top_5_counts_test')
nunique_tr = train.nunique().rename('nunique_train')
nunique_te = test.nunique().rename('nunique_test')
nunique_te['Class'] = 0

min_max = train.describe().T[['min', 'max']].add_suffix('_train').join(test.describe().T[['min', 'max']].add_suffix('_test'))
stats_pivot = pd.concat([min_max, nunique_tr, nunique_te, counts_tr, counts_te], axis=1)
stylize_min_max_count(stats_pivot).background_gradient(cm, subset=['min_test', 'min_train', 'max_train', 'max_test'])


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#FF5C19; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #FF5C19">Distributions</p>

# In[8]:


# kudos to @datafan07 https://www.kaggle.com/code/datafan07/icr-simple-eda-baseline, please, upvote the author.
num_cols = train.columns.tolist()[1:-1]
cat_cols = 'EJ'
num_cols.remove(cat_cols)
features_std = train.loc[:,num_cols].apply(lambda x: np.std(x)).sort_values(
    ascending=False)
f_std = train[features_std.iloc[:20].index.tolist()]

with pd.option_context('mode.use_inf_as_na', True):
    features_skew = np.abs(train.loc[:,num_cols].apply(lambda x: np.abs(skew(x))).sort_values(
        ascending=False)).dropna()
skewed = train[features_skew.iloc[:20].index.tolist()]

with pd.option_context('mode.use_inf_as_na', True):
    features_kurt = np.abs(train.loc[:,num_cols].apply(lambda x: np.abs(kurtosis(x))).sort_values(
        ascending=False)).dropna()
kurt_f = train[features_kurt.iloc[:20].index.tolist()]

def feat_dist(df, cols, rows=3, columns=3, title=None, figsize=(30, 25)):
    
    fig, axes = plt.subplots(rows, columns, figsize=figsize, constrained_layout=True)
    axes = axes.flatten()

    for i, j in zip(cols, axes):
        sns.kdeplot(df, x=i, ax=j, hue='Class',  palette =palette[1:3], linewidth=1.5, linestyle='--')
        
        (mu, sigma) = norm.fit(df[i])
        
        xmin, xmax = j.get_xlim()[0], j.get_xlim()[1]
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, sigma)
        j.plot(x, p, 'k', linewidth=2)
        
        j.set_title('Dist of {0} Norm Fit: $\mu=${1:.2g}, $\sigma=${2:.2f}'.format(i, mu, sigma), weight='bold')
        j.legend(labels=[f'Class0_{i}', f'Class1_{i}', 'Normal Dist'])
        fig.suptitle(f'{title}', fontsize=24, weight='bold')


# In[9]:


feat_dist(train, f_std.columns.tolist(), rows=3, columns=3, title='Distribution of High Std Features', figsize=(30, 8))


# In[10]:


feat_dist(train, skewed.columns.tolist(), rows=3, columns=3, title='Distribution of Skewed Features', figsize=(30, 15))


# In[11]:


feat_dist(train, kurt_f.columns.tolist(), rows=3, columns=3, title='Distribution of High Kurtosis Features', figsize=(30, 15))


# **observations**:
# 
# * The majority of the feature distributions got the tail of the left or right skewed. 
# * The magnitude of the features varies widely, ranging from a mean of 0.026 to 1693. This suggests the your dataset has a high degree of variability.
# 
# **Notes**:
# * Skewed data can affect our classifier by violating the assumption of normality. 
# * It's important to identify the source of the skewness to determine the appropriate approach to handling it. Skewness can be caused by outliers, heavy-tailed distributions, or * the underlying process generating the data. If the skewness is due to outliers, removing those values may be appropriate. If the distribution is heavy-tailed, transformation techniques such as logarithmic or Box-Cox transformations can be used to normalize the data.

# In[12]:


features = test.columns
fig = plt.figure(figsize=(6*6, 45), dpi=130)
for idx, col in enumerate(features):
    ax = plt.subplot(19, 3, idx + 1)
    sns.kdeplot(
        data=train, hue='Class', fill=True,
        x=col, palette=palette[:2], legend=False
    )
            
    ax.set_ylabel(''); ax.spines['top'].set_visible(False), 
    ax.set_xlabel(''); ax.spines['right'].set_visible(False)
    ax.set_title(f'{col}', loc='right', 
                 weight='bold', fontsize=20)

fig.suptitle(f'Features vs Class\n\n\n', ha='center',  fontweight='bold', fontsize=25)
fig.legend([1, 0], loc='upper center', bbox_to_anchor=(0.5, 0.97), fontsize=25, ncol=3)
plt.tight_layout()
plt.show()


# **Observations**:
# 
# * By looking at the plots above we can already infer the class imbalance (the distributions of features against Class 1 are either extremely flat or dense).
# 
# Let's check this out:

# In[13]:


def plot_count(df: pd.core.frame.DataFrame, col_list: list, title_name: str='Train') -> None:
    """Draws the pie and count plots for categorical variables.
    
    Args:
        df: train or test dataframes
        col_list: a list of the selected categorical variables.
        title_name: 'Train' or 'Test' (default 'Train')
        
    Returns:
        subplots of size (len(col_list), 2)
    """
    f, ax = plt.subplots(len(col_list), 2, figsize=(10, 4))
    plt.subplots_adjust(wspace=0)
    
    s1 = df[col_list].value_counts()
    N = len(s1)

    outer_sizes = s1
    inner_sizes = s1/N

    outer_colors = [palette[0], palette[0], '#ff781f', '#ff9752', '#ff9752']
    inner_colors = [palette[1], palette[1], '#ffa66b']

    ax[0].pie(
        outer_sizes,colors=outer_colors, 
        labels=s1.index.tolist(), 
        startangle=90,frame=True, radius=1.3, 
        explode=([0.05]*(N-1) + [.3]),
        wedgeprops={ 'linewidth' : 1, 'edgecolor' : 'white'}, 
        textprops={'fontsize': 12, 'weight': 'bold'}
    )

    textprops = {
        'size':13, 
        'weight': 'bold', 
        'color':'white'
    }

    ax[0].pie(
        inner_sizes, colors=inner_colors,
        radius=1, startangle=90,
        autopct='%1.f%%',explode=([.1]*(N-1) + [.3]),
        pctdistance=0.8, textprops=textprops
    )

    center_circle = plt.Circle((0,0), .68, color='black', 
                               fc='white', linewidth=0)
    ax[0].add_artist(center_circle)

    x = s1
    y = [0, 1]
    sns.barplot(
        x=x, y=y, ax=ax[1],
        palette=palette[:2], orient='horizontal'
    )

    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].tick_params(
        axis='x',         
        which='both',      
        bottom=False,      
        labelbottom=False
    )

    for i, v in enumerate(s1):
        ax[1].text(v, i+0.1, str(v), color='black', 
                     fontweight='bold', fontsize=12)

#     plt.title(col_list)
    plt.setp(ax[1].get_yticklabels(), fontweight="bold")
    plt.setp(ax[1].get_xticklabels(), fontweight="bold")
    ax[1].set_xlabel(col_list, fontweight="bold", color='black')
    ax[1].set_ylabel('count', fontweight="bold", color='black')

    f.suptitle(f'{title_name} Dataset', fontsize=20, fontweight='bold')
    plt.tight_layout()    
#     plt.savefig('data/plot_count.png')
    plt.show()


# In[14]:


plot_count(train, ['Class'], 'Train')


# **Note**:
# * Alright, we got an imbalanced dataset. We are going to use [**StratifiedKFold**](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html) in order to preserve the percentage of samples for each class.

# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#FF5C19; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #FF5C19">Correlations</p>

# In[15]:


def plot_correlation_heatmap(df: pd.core.frame.DataFrame, title_name: str='Train correlation') -> None:
    """Draws the correlation heatmap plot.
    
    Args:
        df: train or test dataframes
        title_name: 'Train' or 'Test' (default 'Train correlation')
        
    Returns:
        subplots of size (len(col_list), 2)
    """

    corr = df.corr()
    fig, axes = plt.subplots(figsize=(10, 5))
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask=mask, linewidths=.5,  cmap=palette[5:][::-2] + palette[1:3], annot=False)
    plt.title(title_name)
    plt.show()

plot_correlation_heatmap(train, 'Train Dataset Correlation')


# **Notes:**
# 
# * There are some highly correlated features (> |0.75|). We might end up dropping some of them or create simple interations.
# 

# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#FF5C19; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #FF5C19">Hierarchical clustering</p>

# In[16]:


from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

hierarchy.set_link_color_palette(palette[1:2]*2)
fig, ax =  plt.subplots(1, 1, figsize=(14, 8), dpi=120)
correlations = train.corr()
converted_corr = 1 - abs(correlations)
Z = linkage(squareform(converted_corr), 'complete')

dn = dendrogram(Z, labels=train.columns,  ax=ax, above_threshold_color=palette[3], orientation='right')
hierarchy.set_link_color_palette(None)
plt.grid(axis='x')
plt.title('Hierarchical clustering, Dendrogram')
plt.show()


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#FF5C19; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #FF5C19">Base XGB Model</p>

# In[17]:


def show_confusion_logloss(y: pd.Series, oof: list) -> None:
    """Draws a confusion matrix and roc_curve with AUC score.
        
        Args:
            oof: predictions for each fold stacked. (list of tuples)
        
        Returns:
            None
    """
    
    f, ax = plt.subplots(1, 2, figsize=(11, 5))
    df = pd.concat([y, pd.Series(oof)], axis=1)
    df.columns = ['target', 'preds']
    
    # I don't remember the author of the cf_matrix code below, but it deserves kudos.
    oof_cb_rnd = np.where(oof > .50, 1, 0)
    cf_matrix = confusion_matrix(y, (oof_cb_rnd)) 
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap=['#037d97', '#90A6B1','#D8E3E2'], cbar=False, ax=ax[0])

    RocCurveDisplay.from_predictions(df.target, df.preds, color=palette[1], ax=ax[1])
    plt.tight_layout();

def log_loss(y_true, y_pred):
    return metrics.log_loss(y_true, y_pred)


# In[18]:


get_ipython().run_cell_magic('time', '', 'config = {\'SEED\': 42,\n          \'FOLDS\': 14,\n          \'N_SPLITS\': 3,\n          \'N_ESTIMATORS\': 900}\n\nxgb_params = {\n    \'colsample_bytree\': 0.5646751146007976,\n    \'gamma\': 7.788727238356553e-06,\n    \'learning_rate\': 0.1419865761603358,\n    \'max_bin\': 824, \'min_child_weight\': 1,\n    \'random_state\': 811996,\n    \'reg_alpha\': 1.6259583347890365e-07,\n    \'reg_lambda\': 2.110691851528507e-08,\n    \'subsample\': 0.879020578464637,\n    \'objective\': \'binary:logistic\',\n    \'eval_metric\': \'logloss\',\n    \'max_depth\': 3,\n    \'early_stopping_rounds\': 150,\n    \'n_jobs\': -1,\n    \'verbosity\': 0\n}\n\ntarget = [\'Class\']\nX, y = train.drop(columns=target), train[target[0]]\n\ncv = model_selection.RepeatedKFold(n_repeats=config[\'N_SPLITS\'], n_splits=config[\'FOLDS\'], random_state=config[\'SEED\'])\nfeature_importances_ = pd.DataFrame(index=X.columns)\nmetric = log_loss\neval_results_ = {}\nmodels_ = []\noof = np.zeros(len(X))\n\nfor fold, (fit_idx, val_idx) in enumerate(cv.split(X, y), start=1):\n\n    # Split the dataset according to the fold indexes.\n    X_fit = X.iloc[fit_idx]\n    X_val = X.iloc[val_idx]\n    y_fit = y.iloc[fit_idx]\n    y_val = y.iloc[val_idx]\n    \n    # XGB .train() requires xgboost.DMatrix.\n    # https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.DMatrix\n    fit_set = xgb.DMatrix(X_fit, y_fit)\n    val_set = xgb.DMatrix(X_val, y_val)\n    watchlist = [(fit_set, \'fit\'), (val_set, \'val\')]\n\n    # Training.\n    # https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.training\n    eval_results_[fold] = {}\n    model = xgb.train(\n        num_boost_round=config[\'N_ESTIMATORS\'],\n        params=xgb_params,\n        dtrain=fit_set,\n        evals=watchlist,\n        evals_result=eval_results_[fold],\n        verbose_eval=False,\n        callbacks=[\n            EarlyStopping(xgb_params[\'early_stopping_rounds\'],\n                          data_name=\'val\', save_best=True)],\n    )\n    \n    \n    val_preds = model.predict(val_set)\n    oof[val_idx] += val_preds / config[\'N_SPLITS\']\n\n    val_score = metric(y_val, val_preds)\n    best_iter = model.best_iteration\n    print(f\'Fold: {blu}{fold:>3}{res}| {metric.__name__}: {blu}{val_score:.5f}{res}\'\n          f\' | Best iteration: {blu}{best_iter:>4}{res}\')\n\n    # Stores the feature importances\n    feature_importances_[f\'gain_{fold}\'] = feature_importances_.index.map(model.get_score(importance_type=\'gain\'))\n    feature_importances_[f\'split_{fold}\'] = feature_importances_.index.map(model.get_score(importance_type=\'weight\'))\n\n    # Stores the model\n    models_.append(model)\n\nmean_cv_score = metric(y, oof)\nprint(f\'{"*" * 50}\\n{red}Mean{res} {metric.__name__}: {red}{mean_cv_score:.5f}\')\n')


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#FF5C19; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #FF5C19">Training Summary</p>

# In[19]:


logloss_folds = pd.DataFrame.from_dict(eval_results_).T
fit_logloss = logloss_folds.fit.apply(lambda x: x['logloss']).iloc[:config['FOLDS']+1]
val_logloss = logloss_folds.val.apply(lambda x: x['logloss']).iloc[:config['FOLDS']+1]

fig, axes = plt.subplots(math.ceil(config['FOLDS']/3), 3, figsize=(30, 30), dpi=150)
ax = axes.flatten()
for i, (f, v, m) in enumerate(zip(fit_logloss, val_logloss, models_[:config['FOLDS']+1])): 
    sns.lineplot(f, color='#B90000', ax=ax[i], label='fit')
    sns.lineplot(v, color='#048BA8', ax=ax[i], label='val')
    ax[i].legend()
    ax[i].spines['top'].set_visible(False);
    ax[i].spines['right'].set_visible(False)
    ax[i].set_title(f'Fold {i}', fontdict={'fontweight': 'bold'})
    
    color =  ['#048BA8', palette[-3]]
    best_iter = m.best_iteration
    span_range = [[0, best_iter], [best_iter + 10, best_iter + xgb_params['early_stopping_rounds']]]
    
    for idx, sub_title in enumerate([f'Best\nIteration: {best_iter}', f'Early\n Stopping: {xgb_params["early_stopping_rounds"]}']):
        ax[i].annotate(sub_title,
                    xy=(sum(span_range[idx])/2 , 0.5),
                    xytext=(0,0), textcoords='offset points',
                    va="center", ha="center",
                    color="w", fontsize=16, fontweight='bold',
                    bbox=dict(boxstyle='round4', pad=0.4, color=color[idx], alpha=0.6))
        ax[i].axvspan(span_range[idx][0]-0.4,span_range[idx][1]+0.4,  color=color[idx], alpha=0.07)
        
    ax[i].set_xlim(0, best_iter + 20 + xgb_params["early_stopping_rounds"])
    ax[i].legend(bbox_to_anchor=(0.95, 1), loc='upper right', title='LogLoss')

plt.tight_layout();


# **Observations**:
#     
# * During the training of the classifier some folds exhibit a greater number of iterations, while a few folds terminate at relatively early iterations.
# * The early terminated folds shows greated `log_loss`.
# 
# **Notes**:
# * The variation in the number of iterations required during training suggests that the complexity of the data may vary across different subsets of the data, and may require different hyperparameters and data preprocessing.

# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#FF5C19; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #FF5C19">Feature importances and OOF errors</p>

# **There are** [several types of importance](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster.get_score) in the Xgboost. It can be computed in several different ways. The default type is gain if you construct model with scikit-learn like API (docs). When you access Booster object and get the importance with get_score method, then default is weight. You can check the type of the importance with xgb.importance_type.
# * The `gain` shows the average gain across all splits the feature is used in.
# * The `weight` shows  the number of times a feature is used to split the data across all trees.

# In[20]:


fi = feature_importances_
fi_gain = fi[[col for col in fi.columns if col.startswith('gain')]].mean(axis=1)
fi_splt = fi[[col for col in fi.columns if col.startswith('split')]].mean(axis=1)

fig, ax = plt.subplots(2, 1, figsize=(20, 20), dpi=150)
ax = ax.flatten()
# Split fi.
data_splt = fi_splt.sort_values(ascending=False)
sns.barplot(x=data_splt.values, y=data_splt.index, 
            color=palette[1], linestyle="-", width=0.5, errorbar='sd',
            linewidth=0.5, edgecolor="black", ax=ax[0])
ax[0].set_title(f'Feature Importance "Split"', fontdict={'fontweight': 'bold'})
ax[0].set(xlabel=None)

for s in ['right', 'top']:
    ax[0].spines[s].set_visible(False)
ax[0]
# Gain fi.    
data_gain = fi_splt.sort_values(ascending=False)
sns.barplot(x=data_gain.values, y=data_gain.index,
            color=palette[-3], linestyle="-", width=0.5, errorbar='sd',
            linewidth=0.5, edgecolor="black", ax=ax[1])
ax[1].set_title(f'Feature Importance "Gain"', fontdict={'fontweight': 'bold'})
ax[1].set(xlabel=None)

for s in ['right', 'top']:
    ax[1].spines[s].set_visible(False)


# In[21]:


show_confusion_logloss(y, oof)


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#FF5C19; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #FF5C19">Platt and Isotonic Probability Calibration</p>

# **Few notes of why we might need to use probability calibration techniques:**
# * A well-calibrated classifier can help to prevent incorrect decisions based on the classifier's predictions, especially if those decisions are based on threshold values for the predicted probabilities.
# * Poorly calibrated classifiers can lead to incorrect decisions, which can be especially problematic in high-stakes situations.
# * Probability calibration techniques can adjust the predicted probabilities to better reflect the true probabilities of positive outcomes, leading to more accurate and reliable predictions.
# 
# **The process:**
# 
# To assess the calibration of a binary classifier, we can plot a calibration curve that shows the predicted probabilities on the x-axis and the proportion of true positive outcomes for each predicted probability bin on the y-axis. The calibration curve can also show the proportion of true negative outcomes for each predicted probability bin. A perfectly calibrated classifier would have a calibration curve that is close to the diagonal line, indicating that the predicted probabilities accurately reflect the true probabilities of both positive and negative outcomes.
# 
# If the probabilities are not calibrated we can build in a sense the second level model fitting our predictions into Logistic regression or any other chosen algorithm against the target.
# After that we repeat the process by plotting the calibration curve and returning the new score.
# 
# **Methods:**
# 
# To calibrate a binary classifier, the data set of scores and their corresponding binary outcomes is used. The aim is to find a function that can accurately estimate the relationship between the scores and the true probabilities as determined empirically in the calibration set. There are several methods of calibration that can be used, including:
# 
# * Platt Scaling
# * Isotonic Regression
# * Beta Calibration
# * SplineCalib
# 
# The choice of calibration method depends on factors such as the size of the calibration set, the complexity of the classifier, and the desired level of calibration accuracy.
# 
# **Let's plot the calibration plot and histogram first:**
# 

# In[22]:


oof_df = pd.concat([y, pd.Series(oof)], axis=1)
oof_df.columns = ['target', 'preds']

def probability_calibration_plot(y_true=oof_df.target,
                                 y_preds=oof_df.preds,
                                 y_cali=None,
                                 n_bins=30,
                                 yerr_c=0.4,
                                 xylim=1,
                                 tick=0.1,
                                 calib_method=''): 
    
    prob_true, prob_pred = calibration_curve(y_true, y_preds, n_bins=n_bins)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5), dpi=120)
    ax = ax.flatten()
    ax[0].errorbar(x=prob_pred, y=prob_true, yerr=abs(prob_true - prob_pred) * yerr_c, fmt=".k", label='Actual',
                   color=palette[1], capthick=0.5, capsize=3, elinewidth=0.7, ecolor=palette[1])

    sns.lineplot(x=np.linspace(0, xylim, 11), y=np.linspace(0, xylim, 11), color=palette[-3],
                 label='Perfectly calibrated', ax=ax[0], linestyle='dashed')
    
    if isinstance(y_cali, np.ndarray):
        prob_true_, prob_pred_ = calibration_curve(y_true, y_cali, n_bins=n_bins)
        sns.lineplot(x=prob_pred_, y=prob_true_, color=palette[-5],
                     label=f'{calib_method} Calibration', ax=ax[0], linestyle='solid')
    
    sns.histplot(y_preds, bins=n_bins*5, color=palette[1], ax=ax[1])
    for i, _ in enumerate(ax):
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].xaxis.grid(False)
        ax[i].yaxis.grid(True)

    ax[0].set_title(f'Probability calibration plot', fontdict={'fontweight': 'bold'})
    ax[1].set_title(f'Histogram of predictions', fontdict={'fontweight': 'bold'})

    ax[0].set_xticks(list(np.arange(0, xylim + tick, tick)))
    ax[0].set_yticks(list(np.arange(0, xylim + tick, tick)))
    ax[0].set(xlabel='predicted', ylabel='actual')
    fig.suptitle(f'Predictions in range {(0, xylim)}', ha='center',  fontweight='bold', fontsize=16)
    plt.tight_layout();
    

probability_calibration_plot()


# **Note**:
# * It might look the classifier is not well calibrated in a big range **~(0.2 < x < 0.8)** and we need to focus on that. But the important thing is to look at the histogram which shows that routhly **90%** of our predictions **< 0.1** and we need to zoom in:

# In[23]:


probability_calibration_plot(y_true=oof_df.query('preds < 0.1').target,
                             y_preds=oof_df.query('preds < 0.1').preds,
                             n_bins=300,
                             yerr_c=0.01,
                             xylim=0.1,
                             tick=0.1)


# Now we can see that our predictions < 0.1 are not well calibrated either.
# So let's try to fix it:
# 
# * We are going to apply Platt Scaling and Isotonic Regression:
# 
# >The **Platt Scaling** method assumes that there is a logistic relationship between the scores generated by a binary classifier and the true probability of positive outcomes. This means that it fits the two parameters of a logistic regression, just like in logistic regression models.
# >
# >This method has a very restrictive set of possible functions and requires very little data. It originated from the observation that a logistic regression is a suitable calibration method for Support Vector Machines, based on theoretical arguments.
# >
# >Platt, J. (1999) showed that using a logistic regression model to predict the probability of positive outcomes can improve the calibration of Support Vector Machines. The method involves training a logistic regression model on the scores generated by the binary classifier and their corresponding binary outcomes, and then using the logistic regression model to transform the predicted scores into calibrated probabilities.
# 
# >The **Isotonic Regression** method fits a piecewise constant, monotonically increasing function to map the scores generated by a binary classifier to probabilities. This method uses the Pool Adjacent Violators (PAV) algorithm and does not assume a particular parametric form.
# >
# >**Isotonic Regression** tends to perform better than Platt Scaling when there is sufficient data. However, it has a tendency **to overfit the calibration curve.**
# >
# > Zadrozny, B., & Elkan, C. (2001) proposed the use of Isotonic Regression for calibrating probability estimates from decision trees and naive Bayesian classifiers.

# In[24]:


lr = LogisticRegression(C=99999999999, solver='liblinear', max_iter=1000)
lr.fit(oof_df.preds.values.reshape(-1, 1), oof_df.target)
lr_preds_calibrated = lr.predict_proba(oof_df.preds.values.reshape(-1, 1))[:,1]
probability_calibration_plot(y_cali=lr_preds_calibrated, calib_method='Platt')

isor = IsotonicRegression(out_of_bounds='clip')
isor.fit(oof_df.preds.values.reshape(-1, 1), oof_df.target)
isor_preds_calibrated = isor.predict(oof_df.preds.values.reshape(-1, 1))

probability_calibration_plot(y_cali=isor_preds_calibrated, calib_method='Isotonic')
print(f'{gld}No calibration LogLoss:       {res}{log_loss(oof_df.target, oof_df.preds):.5f}')
print(f'{gld}Platt calibration LogLoss:    {red}{log_loss(oof_df.target, lr_preds_calibrated):.5f}{res}')
print(f'{gld}Isotonic calibration LogLoss: {grn}{log_loss(oof_df.target, isor_preds_calibrated):.5f}{res}\n\n')


# **Note:**
# 
# The **Isotonic Regression** has been found to yield better calibration performance than the **Platt Scaling**. The results showed that **Isotonic Regression** resulted in a lower logloss score than **Logistic Regression** after calibration, indicating more accurate probability estimates. However, the **Isotonic Regression tends to overfit** the calibration curve, which can result in unrealistic jumps. The Cross-validation techniques can be used to mitigate overfitting and ensure that the calibrated probabilities were reliable. Overall, the experiment demonstrated the potential benefits of using Isotonic Regression for calibrating binary classifiers (we are not going to apply it for the very first submissions).
# 
# **Future work**:
# 
# * Explore Beta Calibration yourself
# * Explore Spline Calibration yourself

# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#FF5C19; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #FF5C19">Submission</p>

# In[25]:


def predict(X):
    y = np.zeros(len(X))
    for model in tqdm(models_):
        y += model.predict(xgb.DMatrix(X))
    return y / len(models_)

predictions = predict(test)
sub = pd.read_csv(PATH_SUB)
sub['class_1'] = predictions
sub['class_0'] = 1 - predictions
# sub.to_csv('submission.csv',index=False)
sub.head(3)


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#FF5C19; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #FF5C19">A mixed blackbox ensemble</p>

# In[26]:


class BlackBoxCVModel:
    def __init__(self, cv,  **kwargs):
        self.cv = cv
        self.model_params = kwargs
        self.models_ = list()
        self.feature_importances_ = None
        self.eval_results_ = dict()
        self.oof = None
        self.metric = log_loss
        self.verbose_folds = False
        self.mean_cv_score = None
        self.general_config = None
        self.predictions = None
        self.target_name = None

    def fit(self, clf: str, X, y=None, **kwargs):
        self.oof = np.zeros(len(X))

        for fold, (fit_idx, val_idx) in enumerate(self.cv.split(X, y), start=1):
            
            # Split the dataset according to the fold indexes.
            X_fit = X.iloc[fit_idx]
            X_val = X.iloc[val_idx]
            y_fit = y.iloc[fit_idx]
            y_val = y.iloc[val_idx]

            
            if clf == 'xgb': 
                # XGB .train() requires DMatrix.
                # https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.DMatrix
                fit_set = xgb.DMatrix(X_fit, y_fit)
                val_set = xgb.DMatrix(X_val, y_val)
                watchlist = [(fit_set, 'fit'), (val_set, 'val')]

                # Training.
                # https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.training
                model = xgb.train(
                    params=self.model_params,
                    dtrain=fit_set,
                    evals=watchlist,
                    verbose_eval=False,
                    callbacks=[
                        EarlyStopping(self.model_params['early_stopping_rounds'],
                                      data_name='val', save_best=True)],
                    **kwargs
                )
    
                val_preds = model.predict(val_set)
            
            if clf == 'lgbm':
                # LGBM .train() requires lightgbm.Dataset.
                # https://lightgbm.readthedocs.io/en/latest/Python-API.html#lightgbm.Dataset
                fit_set = lgbm.Dataset(X_fit, y_fit)
                val_set = lgbm.Dataset(X_val, y_val)

                # Training.
                # https://lightgbm.readthedocs.io/en/latest/Python-API.html#lightgbm.train
                self.eval_results_[fold] = {}
                model = lgbm.train(
                    params=self.model_params,
                    train_set=fit_set,
                    valid_sets=[fit_set, val_set],
                    valid_names=['fit', 'val'],
                    callbacks=[
                        log_evaluation(0),
                        record_evaluation(self.eval_results_[fold]),
                        early_stopping(self.model_params['early_stopping_rounds'],
                                       verbose=False, first_metric_only=True)
                    ],
                    **kwargs
                )

                val_preds = model.predict(X_val)
                
            if clf == 'cbc':
                # CatBoost .fit() with Pool class for the datasets.
                # https://catboost.ai/en/docs/concepts/python-reference_catboost_fit
                fit_set = Pool(X_fit, y_fit)
                val_set = Pool(X_val, y_val)

                # Training.
                # https://catboost.ai/en/docs/concepts/python-quickstart
                self.eval_results_[fold] = {}
                model = cb.train(
                    params=self.model_params,
                    dtrain=fit_set,
                    eval_set=val_set,
                    verbose=False,
                    **kwargs
                )

                val_preds = expit(model.predict(val_set))
            
            if clf == 'hgb':
                model = HistGradientBoostingClassifier(**self.model_params)
                model.fit(X_fit, y_fit)
                val_preds = model.predict_proba(X_val)[:, 1]
                
            if clf == 'rf':
                model = RandomForestClassifier(**self.model_params)
                model.fit(X_fit, y_fit)
                val_preds = model.predict_proba(X_val)[:, 1]

            if clf == 'et':
                model = ExtraTreesClassifier(**self.model_params)
                model.fit(X_fit, y_fit)
                val_preds = model.predict_proba(X_val)[:, 1]
                
            self.oof[val_idx] += val_preds / self.general_config['N_REPEATS']
            
            if self.verbose_folds:
                val_score = self.metric(y_val, val_preds)
#                 best_iter = model.base_estimator_
                print(f'Fold: {blu}{fold:>3}{res}| {self.metric.__name__}: {blu}{val_score:.5f}{res}'
                      f' | Best iteration: {blu}{999:>4}{res}')

                # Stores the model
            self.models_.append(model)

        self.mean_cv_score = self.metric(y, self.oof)
        print(f'{"*" * 50}\n{red}Mean{res} {self.metric.__name__}: {red}{self.mean_cv_score:.5f}')


# In[27]:


target = ['Class']
X_train, y_train = train.drop(columns=target), train[target[0]]

config = {'SEED': 42,
          'FOLDS': 14,
          'N_REPEATS': 3,
          'N_ESTIMATORS': 1500}

# let me try some @takaito params before proceeding with optuna ones.
xgb_params = {
#     'colsample_bytree': 0.5646751146007976,
#     'gamma': 7.788727238356553e-06,
#     'learning_rate': 0.1419865761603358,
#     'max_bin': 824, 'min_child_weight': 1,
#     'random_state': 811996,
#     'reg_alpha': 1.6259583347890365e-07,
#     'reg_lambda': 2.110691851528507e-08,
#     'subsample': 0.879020578464637,
    'learning_rate': 0.005, 
    'max_depth': 4,
    'colsample_bytree': 0.50,
    'subsample': 0.80,
    'eta': 0.03,
    'gamma': 1.5,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
#     'max_depth': 3,
    'early_stopping_rounds': 150,
    'n_jobs': -1,
    'verbosity': 0
}

lgbm_params = {
#     'colsample_bytree': 0.5513118799561477,
#     'learning_rate': 0.1540292253902042,
#     'max_bin': 477,
#     'max_depth': 4,
#     'min_child_weight': 1,
#     'num_leaves': 8,
#     'random_state': 1000141,
#     'reg_alpha': 0.0090378117950824,
#     'reg_lambda': 0.0003886542452096,
#     'subsample': 0.3817791176893013,
     'learning_rate': 0.005,
     'num_leaves': 5,
     'feature_fraction': 0.50,
     'bagging_fraction': 0.80,
     'lambda_l1': 2, 
     'lambda_l2': 4,
    'n_jobs': -1,
    'objective': 'binary',
    'early_stopping_rounds': 150,
    'verbosity': -1,
    'metric': ['binary_logloss']
}

cbc_params = {
#     'colsample_bylevel': 0.562616402328728,
#     'depth': 5,
#     'l2_leaf_reg': 2.16082085701485,
#     'learning_rate': 0.0399690929293443,
#     'max_bin': 354,
#     'min_data_in_leaf': 25,
#     'random_seed': 623174,
#     'random_strength': 1.87644704742659,
#     'subsample': 0.980331194224351,
#     'grow_policy': 'Lossguide',
#     'max_leaves': 64,
#     'early_stopping_rounds': 200,
    'learning_rate': 0.005, 
  
    'depth': 4, 
    'colsample_bylevel': 0.50,
    'subsample': 0.80,
    'l2_leaf_reg': 3, 
    'eval_metric': 'Logloss',
    'loss_function': 'Logloss',
    'auto_class_weights': 'Balanced',
    'use_best_model': True,
    'bootstrap_type': 'Bernoulli',
    'thread_count': -1,
    'allow_writing_files': False
}


# In[28]:


xgb_cv = model_selection.RepeatedKFold(n_repeats=config['N_REPEATS'], n_splits=config['FOLDS'], random_state=config['SEED'])
xgb_model = BlackBoxCVModel(cv=xgb_cv, **xgb_params)
xgb_model.general_config = config
xgb_model.target_name = target[0]
xgb_model.fit('xgb', X_train, y_train, num_boost_round=config['N_ESTIMATORS'])


# In[29]:


lgbm_cv = model_selection.RepeatedStratifiedKFold(n_repeats=config['N_REPEATS'], n_splits=config['FOLDS'] + 1, random_state=config['SEED'])
lgbm_model = BlackBoxCVModel(cv=lgbm_cv, **lgbm_params)
lgbm_model.general_config = config
lgbm_model.target_name = target[0]
lgbm_model.fit('lgbm', X_train, y_train, num_boost_round=config['N_ESTIMATORS'])


# In[30]:


config = {'SEED': 42,
          'FOLDS': 15,
          'N_ESTIMATORS': 2000,
          'N_REPEATS': 1}


cbc_cv = model_selection.StratifiedKFold(n_splits=config['FOLDS'], shuffle=True, random_state=config['SEED'])
cbc_model = BlackBoxCVModel(cv=cbc_cv, **cbc_params)
cbc_model.general_config = config
cbc_model.target_name = target[0]
cbc_model.fit('cbc', X_train, y_train, num_boost_round=config['N_ESTIMATORS'])


# In[31]:


hgb_params = {
    'learning_rate': 0.02,
    'max_iter': 1200,
    'max_depth': 4,
    'l2_regularization': 25,
    'n_iter_no_change': 25
}

hgb_cv = model_selection.StratifiedKFold(n_splits=config['FOLDS'], shuffle=True, random_state=config['SEED'])
hgb_model = BlackBoxCVModel(cv=hgb_cv, **hgb_params)
hgb_model.general_config = config
hgb_model.target_name = target[0]
hgb_model.fit('hgb', X_train.fillna(-999), y_train)


# In[32]:


rf_params = {
    'n_estimators': 100,
    'max_depth': 5,
    'min_samples_split': 4,
    'min_samples_leaf': 1
}

rf_model = BlackBoxCVModel(cv=hgb_cv, **rf_params)
rf_model.general_config = config
rf_model.target_name = target[0]
rf_model.fit('rf', X_train.fillna(-999), y_train)


# In[33]:


et_params = {
    'random_state': 42,
    'n_jobs': -1,
    'max_depth': 5,
    'max_features': 0.92,
    'n_estimators': 200
}

et_model = BlackBoxCVModel(cv=hgb_cv, **et_params)
et_model.general_config = config
et_model.target_name = target[0]
et_model.fit('et', X_train.fillna(-999), y_train)


# In[34]:


oofs_bbox = pd.DataFrame()
oofs_bbox['xgb'] = xgb_model.oof
oofs_bbox['lgbm'] = lgbm_model.oof
oofs_bbox['cbc'] = cbc_model.oof
oofs_bbox['hgb'] = hgb_model.oof
oofs_bbox['rf'] = rf_model.oof
oofs_bbox['et'] = et_model.oof
oofs_bbox


# In[35]:


def plot_correlation_heatmap(df: pd.core.frame.DataFrame, title_name: str='Train correlation') -> None:
    """Draws the correlation heatmap plot.
    
    Args:
        df: train or test dataframes
        title_name: 'Train' or 'Test' (default 'Train correlation')
        
    Returns:
        subplots of size (len(col_list), 2)
    """
    corr = df.corr()
    fig, axes = plt.subplots(figsize=(14, 11), dpi=120)
    mask = np.zeros_like(corr)
    sns.heatmap(corr, linewidths=.5, cmap=palette[5:][::-2] + palette[1:3], annot=True)
    plt.title(title_name)
    plt.show()

plot_correlation_heatmap(oofs_bbox, 'OOF Correlation')


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#FF5C19; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #FF5C19">  Blackbox Submission</p>

# In[36]:


def predict(X, clf, models):
    y = np.zeros(len(X))
    for model in tqdm(models):
        if clf == 'xgb':
            y += model.predict(xgb.DMatrix(X))
        if clf == 'lgbm':
            y += model.predict(X.values)
        if clf == 'cbc':
            y += expit(model.predict(Pool(X)))
        # otherwise it throws unknown error during the submission
        if clf == 'hgb':
            y += model.predict_proba(X.fillna(-999))[:, 1]
        if clf == 'rf':
            y += model.predict_proba(X.fillna(-999))[:, 1]
    return y / len(models)

xgb_predictions = predict(test, 'xgb', xgb_model.models_)
lgbm_predictions = predict(test, 'lgbm', lgbm_model.models_)
cbc_predictions = predict(test, 'cbc', cbc_model.models_)
hgb_predictions = predict(test, 'hgb', hgb_model.models_)
rf_predictions = predict(test, 'rf', rf_model.models_)

sub = pd.read_csv(PATH_SUB)
blended_predictions = xgb_predictions*0.20 + lgbm_predictions*0.3 + cbc_predictions*0.35 + hgb_predictions*0.0999 + rf_predictions*0.0501
sub['class_1'] = blended_predictions
sub['class_0'] = 1 - blended_predictions
sub.to_csv('submission.csv',index=False)
sub


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#FF5C19; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #FF5C19">Acknowledgement</p>

# @jcaliz for .css and plotting ideas.

# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#FF5C19; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #FF5C19">Outro and future work</p>
# 
# The features engineering part is on the way. 
# I hope to continue working on the dataset.  Good luck in the competition!
