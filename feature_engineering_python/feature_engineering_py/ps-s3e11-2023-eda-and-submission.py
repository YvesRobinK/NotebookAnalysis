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
red = Style.BRIGHT + Fore.RED
blu = Style.BRIGHT + Fore.BLUE
res = Style.RESET_ALL

import xgboost as xgb
from xgboost.callback import EarlyStopping
from sklearn import model_selection
from sklearn import metrics


# <p align="right">
#   <img src="https://drive.google.com/uc?export=view&id=1a7o9FGic8-PrV16oia_oV2sw4myERhVu"/>
# </p>

# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#FF5C19; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #FF5C19">Intro</p>

# This Kaggle notebook is aimed at providing a comprehensive exploratory data analysis (EDA) for the given dataset, with the ultimate goal of making informed decisions and recommendations before diving into modeling. 
# >Through this EDA, we will gain a deeper understanding of the data structure, missing values, relationships between variables, and any patterns or anomalies that could impact our modeling process. By performing a thorough EDA, we can identify potential roadblocks and make necessary pre-processing decisions that will improve the performance and accuracy of our models. So, buckle up, and let's embark on this journey of discovering insights and valuable information from the data to drive better modeling decisions.

# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#FF5C19; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #FF5C19">Data</p>

# **The dataset** for this competition (both train and test) was generated from a deep learning model trained on the [Media Campaign Cost Prediction dataset](https://www.kaggle.com/datasets/gauravduttakiit/media-campaign-cost-prediction). Our task is to predict the cost of media campaigns in the food marts on the basis of the features provided.
# 
# >[Food Mart (CFM)](https://en.wikipedia.org/wiki/Convenient_Food_Mart) is a chain of convenience stores in the United States. The private company's headquarters are located in Mentor, Ohio, and currently, approximately 325 stores are located in the US. Convenient Food Mart operates on the franchise basis.
# 
#  
# There are 15 independent variables (including `id`):
# 
# * `store_sales`: Store sales in millions. 
# * `unit_sales`: Quantity of units sold.
# * `total_children`:  Total children in home.
# * `num_children_at_home`:  Total children at home as per customer filled details.
# * `avg_cars_at_home`: Average cars at home.
# * `gross_weight`: Gross weight of an item.
# * `recyclable_package`: If the package of the food item is recycleble `1` or not `0`.
# * `low_fat`: If an item is a low fat `1` or not `0`.
# * `units_per_case`: Units/case units available in each store shelves.
# * `store_sqft`: Store area available in sqft.
# * `coffee_bar`: If a store has a coffee bar available `1` or not `0`.
# * `video_store`: If a video store/gaming store is available `1` or not `0`.
# * `salad_bar`: if a salad bar is available in a store `1` or not `0`.
# * `prepared_food`: if a prepared food is available in a store `1` or not `0`.
# * `florist`: if flower shelves are available in a store `1` or not `0`.
# 
# Target varibale:
# * `cost`:  Cost on acquiring a customers in dollars.
# 
# **Metrics**:
# * [RMSLE](https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-log-error) (the sklearn mean_squared_log_error with squared=False).

# In[3]:


def fix_columns(df): 
    """Removes (in millions) and (approx).1 from names of columns."""
    df.columns = df.columns.str.replace('(in millions)', '', regex=False)
    df.columns = df.columns.str.replace(' home(approx).1', '_home', regex=False)
    return df

PATH_ORIGIN = '/kaggle/input/media-campaign-cost-prediction/train_dataset.csv'
PATH_TRAIN = '/kaggle/input/playground-series-s3e11/train.csv'
PATH_TEST = '/kaggle/input/playground-series-s3e11/test.csv'
PATH_SUB = '/kaggle/input/playground-series-s3e11/sample_submission.csv'

origin = fix_columns(pd.read_csv(PATH_ORIGIN))
train = fix_columns(pd.read_csv(PATH_TRAIN).drop(columns='id'))
test = fix_columns(pd.read_csv(PATH_TEST).drop(columns='id'))


# In[4]:


print(f'{blk}[INFO] Shapes:'
      f'{blk}\n[+] origin ->  {red}{origin.shape}'
      f'{blk}\n[+] train  -> {red}{train.shape}'
      f'{blk}\n[+] test   ->  {red}{test.shape}\n')

print(f'{blk}[INFO] Any missing values:'
      f'{blk}\n[+] origin -> {red}{origin.isna().any().any()}'
      f'{blk}\n[+] train  -> {red}{train.isna().any().any()}'
      f'{blk}\n[+] test   -> {red}{test.isna().any().any()}')


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#FF5C19; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #FF5C19">Interactive Dataframes Overview</p>

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

# In[5]:


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
           df: |min_train| max_train |min_test |max_test |top10_counts_train |top_10_counts_train|

       Returns:
           s: the dataframe wrapped into Styler.
   """
   s = pivot_table
   # A formatting dictionary for controlling each column precision (.000 <-). 
   di_frmt = {(i if i.startswith('m') else i):
             ('{:.3f}' if i.startswith('m') else '{:}') for i in s.columns}

   s = s.style.set_table_styles(magnify(True))\
       .format(di_frmt)\
       .set_caption(f"The train and test datasets min, max, top10 values side by side (hover to magnify).")
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
   s = df.describe().T
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
display(stylize_describe(train))
stylize_describe(test, 'test', is_test=True)


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#FF5C19; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #FF5C19">Min Max and Unique Counts</p>

# In[6]:


cm = sns.light_palette('#FF5C19', as_cmap=True)

counts_tr = pd.Series({ft: [train[ft].value_counts().round(3).iloc[:5].to_dict()] for ft in train.columns}, name='top_5_counts_train')
counts_te = pd.Series({ft: [test[ft].value_counts().round(3).iloc[:5].to_dict()] for ft in test.columns}, name='top_5_counts_test')
nunique_tr = train.nunique().rename('nunique_train')
nunique_te = test.nunique().rename('nunique_test')
nunique_te['Class'] = 0

min_max = train.describe().T[['min', 'max']].add_suffix('_train').join(test.describe().T[['min', 'max']].add_suffix('_test'))
stats_pivot = pd.concat([min_max, nunique_tr, nunique_te, counts_tr, counts_te], axis=1)
stylize_min_max_count(stats_pivot).background_gradient(cm, subset=['min_test', 'min_train', 'max_train', 'max_test'])


# **Observations**:
# 
# * 1. We got 7 binary features [`recyclable_package`, `low_fat`, `coffee_bar`, `video_store`, `salad_bar`, `prepared_food`, `florist`]
# * 2. `store_sqft` feature looks continuos but it got only 20 distinct values. It seems like we got 20 same stores.
# * 3. `units_per_case` feature got only 30 distinct values.
# * 4. `unit_sales`, `total_children`, `num_children_at_home`, `avg_cars_at_home` features can be treated as categorical since there are only 5-6 distinct values inside each of the them.
# 
# 
# **Takeaways:**
# * 1. Try to apply various encoders (one-hot, ordinal).
# * 2. Try to apply target encoders by binning the `cost`.

# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#FF5C19; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #FF5C19">Distributions</p>

# In[7]:


# kudos to @jcaliz
features = test.columns
n_bins = 50
histplot_hyperparams = {
    'kde':True,
    'alpha':0.4,
    'stat':'percent',
    'bins':n_bins
}

columns = features
n_cols = 3
n_rows = math.ceil(len(columns)/n_cols)
fig, ax = plt.subplots(n_rows, n_cols, figsize=(20, n_rows*4))
ax = ax.flatten()

for i, column in enumerate(columns):
    plot_axes = [ax[i]]
    sns.kdeplot(
        train[column], label='Train',
        ax=ax[i], color=palette[0]
    )
    
    sns.kdeplot(
        test[column], label='Test',
        ax=ax[i], color=palette[1]
    )
    
    sns.kdeplot(
        origin[column], label='Original',
        ax=ax[i], color=palette[2]
    )
    
    # titles
    ax[i].set_title(f'{column} Distribution');
    ax[i].set_xlabel(None)
    
    # remove axes to show only one at the end
    plot_axes = [ax[i]]
    handles = []
    labels = []
    for plot_ax in plot_axes:
        handles += plot_ax.get_legend_handles_labels()[0]
        labels += plot_ax.get_legend_handles_labels()[1]
        plot_ax.legend().remove()
    
for i in range(i+1, len(ax)):
    ax[i].axis('off')
    
fig.suptitle(f'Dataset Feature Distributions\n\n\n', ha='center',  fontweight='bold', fontsize=25)
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), fontsize=25, ncol=3)
plt.tight_layout()


# **Observations:**
# 
# * Feature distributions are close to, but not exactly the same, as the original.
# * The organizers [tell us](https://www.kaggle.com/competitions/playground-series-s3e11/data) to feel free to use the original dataset as part of this competition, both to explore differences as well as to see whether incorporating the original in training improves model performance.
# 
# **Things to remember:**
# >A synthetic dataset is a type of dataset created by generating new data that mimics the original data using various techniques. However, it is possible that the synthetic dataset features may not closely follow the original dataset distribution (our case). This can occur due to a variety of factors, such as using a different sampling technique, applying different data transformations, or introducing new features that were not present in the original dataset. When the synthetic dataset features do not closely follow the original dataset distribution, it can affect the performance of machine learning models trained on the origin data, as the models may not accurately capture the underlying patterns and relationships in the original data. Therefore, it is important to carefully evaluate the quality of both datasets before using them.
# 
# Let's take a look at the train dataset categorical features against the target variable:

# In[8]:


cat_features = ['recyclable_package', 'low_fat', 'coffee_bar',
                'video_store', 'salad_bar', 'prepared_food',
                'florist', 'unit_sales', 'total_children',
                'num_children_at_home', 'avg_cars_at_home']

fig, ax = plt.subplots(4, 3, figsize=(25, 15), dpi=150)
ax = ax.flatten()
for i, ft in enumerate(cat_features):
    sns.histplot(
        data=train,
        x="cost", hue=ft,
        multiple="stack",
        palette=palette,
        edgecolor=".3",
        linewidth=.5,
        log_scale=True,
        ax=ax[i],
    )
fig.suptitle(f'Train Categorical Features vs Price\n\n\n', ha='center',  fontweight='bold', fontsize=25)
plt.tight_layout()
plt.show()


# **Observations**:
# * 1. `salad_bar` and `prepared_food` looks almost the same.
# * 2. we cannot really see class 6 for `unit_sales`.
# 
# The the relative frequencies of the unique values might help us to understand the charts above better. (uncomment the code in the cell below to obtain the values) 
# 
# **Relative frequencies of the categorical features**:
#     
# * `recyclable_package` {1.0: 0.568, 0.0: 0.431}
# * `low_fat` {0.0: 0.672, 1.0: 0.328}
# * `coffee_bar` {1.0: 0.565, 0.0: 0.435}
# * `video_store` {0.0: 0.723, 1.0: 0.277}
# * `salad_bar` {1.0: 0.505, 0.0: 0.495}
# * `prepared_food` {1.0: 0.505, 0.0: 0.495}
# * `florist` {1.0: 0.503, 0.0: 0.496}
# * `unit_sales` {3.0: 0.487, 4.0: 0.264, 2.0: 0.214, 1.0: 0.019, 5.0: 0.016, 6.0: 0.0000}
# * `total_children` {1.0: 0.208, 2.0: 0.205, 3.0: 0.198, 4.0: 0.195, 0.0: 0.101, 5.0: 0.093}
# * `num_children_at_home` {0.0: 0.676, 1.0: 0.137, 2.0: 0.078, 3.0: 0.057, 4.0: 0.035, 5.0: 0.017}
# * `avg_cars_at_home` {2.0: 0.306, 3.0: 0.29, 1.0: 0.229, 4.0: 0.123, 0.0: 0.051}
# 
# **Takeaways**:
# * It seems `salad_bar` and `prepared_food` features are the same. We need to validate it. 
# * value `6.0` is almost non-existitent for `unit_sales` we might remove it or clip it to `5.0`.
# 
# Let's check if `salad_bar` and `prepared_food` features are equal:

# In[9]:


def validate_salad_eq_prepared(df: pd.DataFrame, df_name: str) -> None:
    percent_eq = (df['salad_bar'] == df['prepared_food']).sum()/df.shape[0]*100
    abs_ne = (df['salad_bar'] == df['prepared_food']).sum() - df.shape[0]
    print(f'{blk}[INFO] {df_name} dataset "salad_bar" equals to '
          f'"prepared_food" in {red}{percent_eq:.2f}%{res} {blk}observations. Not equal: {red}{abs_ne}')
          
validate_salad_eq_prepared(origin, 'Origin')
validate_salad_eq_prepared(train, 'Train ')
validate_salad_eq_prepared(test, 'Test  ')


# **Note**:
# * We might remove either `salad_bar` or `prepared_food` column and see how it works since they are almost the same.

# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#FF5C19; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #FF5C19">Correlations</p>

# In[10]:


def plot_correlation_heatmap(df: pd.core.frame.DataFrame, title_name: str='Train correlation') -> None:
    """Draws the correlation heatmap plot.
    
    Args:
        df: train or test dataframes
        title_name: 'Train' or 'Test' (default 'Train correlation')
        
    Returns:
        subplots of size (len(col_list), 2)
    """

    corr = df.corr()
    fig, axes = plt.subplots(figsize=(20, 15))
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask=mask, linewidths=.5, cmap=palette[5:][::-2] + palette[1:3], annot=True)
    plt.title(title_name)
    plt.show()

plot_correlation_heatmap(origin, 'Original Dataset Correlation')
plot_correlation_heatmap(train, 'Train Dataset Correlation')
plot_correlation_heatmap(train, 'Test Dataset Correlation')


# **Notes:**
# 
# * `salad_bar` and `prepared_food` features show correlation equal to 1 which supports already found peculiarity. 
# * There are some moderate positive correlations 0.5-0.6 between featrures, but overall features uncorrelated.

# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#FF5C19; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #FF5C19">Chasing Duplicates and Leaks</p>

# In[11]:


from itertools import combinations

columns = train.drop(columns=['cost', 'salad_bar']).columns

def get_columns_combinations(columns, tr=None):
    """Returns all combinations of columns
        Args:
            columns: array of column names
            tr: (int) num of columns. If None, default all columns.
        
        Returns:
            all_combs: (list of lists) all possible column combinations.
    """
    n_comb = len(columns)
    if tr:
        n_comb = len(columns[:tr])
    all_combs = []
    for i in range(13, n_comb+1):
        all_combs += list(map(list, combinations(columns, r=i)))
    return all_combs

print(f'{red}[INFO] Pseudo Train duplicates - num, col:\n\n')
all_combs = get_columns_combinations(columns)
for cols in all_combs:
    s = train[cols].duplicated().sum()
    if s > 5:
        print(f'{red}{s}{blk}, {cols}')
        
print(f'{red}\n[INFO] Pseudo Train-test duplicates - num, col:\n\n')
all_combs = get_columns_combinations(columns)
for cols in all_combs:
    s = pd.concat([train, test])[cols].duplicated().sum()
    if s > 5:
        print(f'{red}{s}{blk}, {cols}')


# **Notes:**
# 
# * There are a lot of pseudo duplicates. The reader might think to probe LB by assigning the values directly. The only problem is that the target is somewhat continuous (328 unique values in the train dataset) which might cause troubles. 

# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#FF5C19; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #FF5C19">Feature Engineering</p>

# In[12]:


class MediaDataProcessor:
    def __init__(self,
                 train_data=None,
                 test_data=None,
                 combined: bool = False,
                 verbose: bool = False):
        self.origin_data = None
        self.train_data = train_data
        self.test_data = test_data
        self.combined = combined
        self.verbose = verbose

        if self.verbose:
            print(f'{blk}[INFO] Shapes before feature engineering:'
                  f'{blk}\n[+] train  -> {red}{self.train_data.shape}'
                  f'{blk}\n[+] test   -> {red}{self.test_data.shape}\n')
            
    @staticmethod
    def fe(df):
        df.unit_sales = df.unit_sales.clip(0, 5)
        df['children_ratio'] = df['total_children']/df['num_children_at_home']
        df['children_ratio'] = df['children_ratio'].replace([np.inf, -np.inf], 10)
        return df

    def process_data(self):
        
        self.train_data = self.fe(self.train_data)
        self.test_data = self.fe(self.test_data)

        if self.combined:
            cols = self.train_data.columns
            self.origin_data = self.fe(self.origin_data)
            self.train_data = pd.concat([self.train_data, self.origin_data])
            self.train_data = self.train_data.drop_duplicates(subset=cols).reset_index(drop=True)

        if self.verbose:
            print(f'{blk}[INFO] Shapes after feature engineering:'
                  f'{blk}\n[+] train  -> {red}{self.train_data.shape}'
                  f'{blk}\n[+] test   -> {red}{self.test_data.shape}\n')

        return self.train_data, self.test_data
    
f_e = MediaDataProcessor(train, test, verbose=True)
train, test = f_e.process_data()


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#FF5C19; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #FF5C19">Base XGB Model</p>

# In[13]:


def rmsle(y_true, y_pred):
    return metrics.mean_squared_log_error(y_true, y_pred, squared=False)


# In[14]:


get_ipython().run_cell_magic('time', '', 'config = {\'SEED\': 42,\n          \'FOLDS\': 15,\n          \'N_ESTIMATORS\': 700}\n\n# kudos to @shashwatraman\nxgb_params = {\'objective\': \'reg:squarederror\',\n              \'eval_metric\': \'rmse\',\n              \'learning_rate\': 0.05,\n              \'max_depth\': 8,\n              \'early_stopping_rounds\': 200,\n              \'tree_method\': \'gpu_hist\',\n              \'subsample\': 1.0,\n              \'colsample_bytree\': 1.0,\n              \'verbosity\': 0,\n              \'random_state\': 42}\n\ncols_to_drop = [\'cost\', \'store_sales\', \'gross_weight\', \'unit_sales\', \'low_fat\',\n                \'recyclable_package\', \'salad_bar\',\'units_per_case\', ]\n\nX, y = train.drop(columns=cols_to_drop), train.cost\ny = np.log(y)\n\ncv = model_selection.KFold(n_splits=config[\'FOLDS\'], shuffle=True, random_state=config[\'SEED\'])\nfeature_importances_ = pd.DataFrame(index=X.columns)\nmetric = rmsle\neval_results_ = {}\nmodels_ = []\noof = np.zeros(len(X))\n\nfor fold, (fit_idx, val_idx) in enumerate(cv.split(X, y), start=1):\n\n    # Split the dataset according to the fold indexes.\n    X_fit = X.iloc[fit_idx]\n    X_val = X.iloc[val_idx]\n    y_fit = y.iloc[fit_idx]\n    y_val = y.iloc[val_idx]\n\n    # XGB .train() requires xgboost.DMatrix.\n    # https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.DMatrix\n    fit_set = xgb.DMatrix(X_fit, y_fit)\n    val_set = xgb.DMatrix(X_val, y_val)\n    watchlist = [(fit_set, \'fit\'), (val_set, \'val\')]\n\n    # Training.\n    # https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.training\n    eval_results_[fold] = {}\n    model = xgb.train(\n        num_boost_round=config[\'N_ESTIMATORS\'],\n        params=xgb_params,\n        dtrain=fit_set,\n        evals=watchlist,\n        evals_result=eval_results_[fold],\n        verbose_eval=False,\n        callbacks=[\n            EarlyStopping(xgb_params[\'early_stopping_rounds\'],\n                          data_name=\'val\', save_best=True)],\n    )\n\n    val_preds = model.predict(val_set)\n    oof[val_idx] = val_preds\n\n    val_score = metric(np.exp(y_val), np.exp(val_preds))\n    best_iter = model.best_iteration\n    print(f\'Fold: {blu}{fold:>3}{res}| {metric.__name__}: {blu}{val_score:.5f}{res}\'\n          f\' | Best iteration: {blu}{best_iter:>4}{res}\')\n\n    # Stores the feature importances\n    feature_importances_[f\'gain_{fold}\'] = feature_importances_.index.map(model.get_score(importance_type=\'gain\'))\n    feature_importances_[f\'split_{fold}\'] = feature_importances_.index.map(model.get_score(importance_type=\'weight\'))\n\n    # Stores the model\n    models_.append(model)\n\nmean_cv_score = metric(np.exp(y), np.exp(oof))\nprint(f\'{"*" * 50}\\n{red}Mean{res} {metric.__name__}: {red}{mean_cv_score:.5f}\')\n')


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#FF5C19; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #FF5C19">Training Summary</p>

# In[15]:


metric_score_folds = pd.DataFrame.from_dict(eval_results_).T
fit_rmsle = metric_score_folds.fit.apply(lambda x: x['rmse'])
val_rmsle = metric_score_folds.val.apply(lambda x: x['rmse'])

fig, axes = plt.subplots(math.ceil(config['FOLDS']/3), 3, figsize=(30, 30), dpi=150)
ax = axes.flatten()
for i, (f, v, m) in enumerate(zip(fit_rmsle, val_rmsle, models_)): 
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
    ax[i].legend(bbox_to_anchor=(0.95, 1), loc='upper right', title='RMSLE')

plt.tight_layout();


# **Note**:
# * The model reaches ~0.3 RMSLE in less than 100 iterations reaching plateu in improvement, although it does not trigger early stopping callback until ~3000 iterations. 

# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#FF5C19; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #FF5C19">Feature importances and OOF errors</p>

# **There are** [several types of importance](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.Booster.get_score) in the Xgboost. It can be computed in several different ways. The default type is gain if you construct model with scikit-learn like API (docs). When you access Booster object and get the importance with get_score method, then default is weight. You can check the type of the importance with xgb.importance_type.
# * The `gain` shows the average gain across all splits the feature is used in.
# * The `weight` shows  the number of times a feature is used to split the data across all trees.

# In[16]:


fi = feature_importances_
fi_gain = fi[[col for col in fi.columns if col.startswith('gain')]].mean(axis=1)
fi_splt = fi[[col for col in fi.columns if col.startswith('split')]].mean(axis=1)

fig, ax = plt.subplots(2, 1, figsize=(20, 10), dpi=150)
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


# In[17]:


oof_df = pd.DataFrame(np.vstack([oof, y]).T, columns=['cost_pred', 'cost'])

sort_idxs = np.argsort(oof)
oof_sorted = oof[sort_idxs]
y_true_sorted = train['cost'].iloc[sort_idxs]
y_true_sorted = pd.Series(y_true_sorted.values, index=oof_sorted)
y_roll_mean = np.log(y_true_sorted.rolling(80, center=True).mean())

fig, ax = plt.subplots(1, 2, figsize=(15, 5), dpi=120)
ax = ax.flatten()
sns.regplot(data=oof_df, x='cost_pred', y='cost', color=palette[1], scatter=False,
            line_kws={"color": "black", "linestyle": "--", "lw": 1.5}, ax=ax[0], label='Perfectly predicted')

sns.scatterplot(data=oof_df, x='cost_pred', y='cost', s=10, color=palette[1], ax=ax[0], label='Actual')
sns.scatterplot(x=y_roll_mean.index, y=y_roll_mean, color='red', s=1, ax=ax[0]);

ax[0].legend(bbox_to_anchor=(0.05, 1), loc='upper left')
ax[0].set(xlabel='cost predicted', ylabel='cost actual')

sns.histplot(oof_df.cost, color=palette[0], label='y_true', ax=ax[1])
sns.histplot(oof_df.cost_pred, color=palette[1], label='y_pred', ax=ax[1])
ax[1].legend(bbox_to_anchor=(0.95, 1), loc='upper right', title='RMSLE')
for i, _ in enumerate(ax):
    ax[i].spines['top'].set_visible(False)
    ax[i].spines['right'].set_visible(False)
    ax[i].xaxis.grid(False)
    ax[i].yaxis.grid(True)
    
ax[0].set_title(f'RegPlot of predictions', fontdict={'fontweight': 'bold'})
ax[1].set_title(f'Histogram of predictions', fontdict={'fontweight': 'bold'});


# It looks terrible. The predictions are way off target. There might be a problem with the data.
# 
# Let's check [R2 score](https://en.wikipedia.org/wiki/Coefficient_of_determination):

# In[18]:


r2 = metrics.r2_score(oof_df.cost, oof_df.cost_pred)
print(f'{blk}R_squared: {red}{r2:.3f}{res}')


# The original R2 score was **0.086**.
# R-squared (R2) score of **0.086** means that **only 8.6%** of the variance in the dependent variable is explained by the independent variables used in the model. In other words, the model is not a good fit for the data (or the data is not good), as it is only able to explain a small portion of the variability observed in the dependent variable.
# 
# We have improved!

# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#FF5C19; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #FF5C19">Submission</p>

# In[19]:


def predict(X):
    y = np.zeros(len(X))
    for model in tqdm(models_):
        y += np.exp(model.predict(xgb.DMatrix(X)))
    return y / len(models_)

predictions = predict(test.drop(columns=cols_to_drop[1:]))
sub = pd.read_csv(PATH_SUB)
sub.cost = predictions
sub.to_csv('submission.csv', index=False)
sub.head(3)


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#FF5C19; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #FF5C19">Acknowledgement</p>

# @jcaliz for .css and plotting ideas.

# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#FF5C19; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #FF5C19">Outro and future work</p>
# 
# Good luck in the competition!
