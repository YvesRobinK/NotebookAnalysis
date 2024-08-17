#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('wget http://bit.ly/3ZLyF82 -O CSS.css -q')
    
from IPython.core.display import HTML
with open('./CSS.css', 'r') as file:
    custom_css = file.read()

HTML(custom_css)


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#B90000; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #B90000">Libraries</p>

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import math

import warnings
warnings.filterwarnings('ignore')

tqdm.pandas()

rc = {
    "axes.facecolor": "#FFF9ED",
    "figure.facecolor": "#FFF9ED",
    "axes.edgecolor": "#000000",
    "grid.color": "#EBEBE7",
    "font.family": "serif",
    "axes.labelcolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "grid.alpha": 0.4
}

sns.set(rc=rc)

from colorama import Style, Fore
red = Style.BRIGHT + Fore.RED
blu = Style.BRIGHT + Fore.BLUE
mgt = Style.BRIGHT + Fore.MAGENTA
gld = Style.BRIGHT + Fore.YELLOW
res = Style.RESET_ALL

import xgboost as xgb
from xgboost.callback import EarlyStopping
from sklearn import model_selection
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error


# <p align="right">
#   <img src="https://drive.google.com/uc?export=view&id=1sFLl4_hs33s1S66S8rrl27UUqahScH6I"/>
# </p>

# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#B90000; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #B90000">Intro</p>

# This Kaggle notebook is aimed at providing a comprehensive exploratory data analysis (EDA) for the given dataset, with the ultimate goal of making informed decisions and recommendations before diving into modeling. 
# >Through this EDA, we will gain a deeper understanding of the data structure, missing values, relationships between variables, and any patterns or anomalies that could impact our modeling process. By performing a thorough EDA, we can identify potential roadblocks and make necessary pre-processing decisions that will improve the performance and accuracy of our models. So, buckle up, and let's embark on this journey of discovering insights and valuable information from the data to drive better modeling decisions.
# 
# **Disclaimer**: You are welcome to use this notebook the way you like. If you feel bored seeing your default jupyter notebook style, you are welcome to use one of the themes I compiled in [my github repo](https://github.com/SergeySakharovskiy/jupyter-themes-css). If you like this kernel or tips, your feedback will encourage me to share more.

# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#B90000; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #B90000">Data</p>

# **The dataset** for this competition (both train and test) was generated from a deep learning model trained on the [Gemstone Price Prediction dataset](https://www.kaggle.com/datasets/colearninglounge/gemstone-price-prediction). The goal is to predict `price`  the Price of the cubic zirconia.
#  
# There are 18 independent variables (including `id`):
# 
# * `Carat`: a weight of the cubic zirconia. A metric “carat” is defined as 200 milligrams.
# * `Cut`:  describes the cut quality of the cubic zirconia. Quality is increasing order Fair, Good, Very Good, Premium, Ideal.
# * `Color`:  refers to the color of the cubic zirconia. With D being the best and J the worst.
# * `Clarity`:  refers to the absence of the Inclusions and Blemishes. (In order from Best to Worst, FL = flawless, I3= level 3 inclusions) FL, IF, VVS1, VVS2, VS1, VS2, SI1, SI2, I1, I2, I3
# * `Depth`:  the `height` of a cubic zirconia, measured from the Culet to the table, divided by its average Girdle Diameter.
# * `Table`:  the `width` of the cubic zirconia's Table expressed as a Percentage of its Average Diameter.
# * `X`:  Length of the cubic zirconia in mm.
# * `Y`:  Width of the cubic zirconia in mm.
# * `Z`:  Height of the cubic zirconia in mm.
# 
# Target varibale:
# * `Price`:  the Price of the cubic zirconia.
# 
# **Metrics**:
# * [RMSE](https://en.wikipedia.org/wiki/Root-mean-square_deviation)

# In[3]:


PATH_ORIGIN = '/kaggle/input/gemstone-price-prediction/cubic_zirconia.csv'
PATH_TRAIN = '/kaggle/input/playground-series-s3e8/train.csv'
PATH_TEST = '/kaggle/input/playground-series-s3e8/test.csv'
PATH_SUB = '/kaggle/input/playground-series-s3e8/sample_submission.csv'

origin = pd.read_csv(PATH_ORIGIN).drop(columns='Unnamed: 0')
train = pd.read_csv(PATH_TRAIN).drop(columns='id')
test = pd.read_csv(PATH_TEST).drop(columns='id')

train.head(3)


# In[4]:


print(f'{gld}[INFO] Shapes:'
      f'{gld}\n[+] origin -> {red}{origin.shape}'
      f'{gld}\n[+] train -> {red}{train.shape}'
      f'{gld}\n[+] test -> {red}{test.shape}\n')

print(f'{gld}[INFO] Any missing values:'
      f'{gld}\n[+] origin -> {red}{origin.isna().any().any()}'
      f'{gld}\n[+] train -> {red}{train.isna().any().any()}'
      f'{gld}\n[+] test -> {red}{test.isna().any().any()}')


# ### <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#B90000; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #B90000">Train</p>

# In[5]:


pd.set_option('display.float_format', lambda x: '%.2f' % x)
train.iloc[:, :-1].describe().T.sort_values(by='std', ascending=False)\
                     .style.background_gradient(cmap='YlOrBr')\
                     .bar(subset=["mean",], color='green')\
                     .bar(subset=["max"], color='#BB0000')


# ### <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#B90000; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #B90000">Test</p>

# In[6]:


pd.set_option('display.float_format', lambda x: '%.2f' % x)
test.iloc[:, :-1].describe().T.sort_values(by='std', ascending=False)\
                     .style.background_gradient(cmap='YlOrBr')\
                     .bar(subset=["mean",], color='green')\
                     .bar(subset=["max"], color='#BB0000')


# ### <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#B90000; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #B90000">Min Max and Counts</p>

# In[7]:


counts_tr = pd.Series({ft: [train[ft].value_counts().round(3).iloc[:10].to_dict()] for ft in train.columns}, name='top10_counts_train')
counts_te = pd.Series({ft: [test[ft].value_counts().round(3).iloc[:10].to_dict()] for ft in test.columns}, name='top_10_counts_train')
min_max = train.describe().T[['min', 'max']].add_suffix('_train').join(test.describe().T[['min', 'max']].add_suffix('_test'))
stats_pivot = pd.concat([min_max, counts_tr, counts_te], axis=1)
stats_pivot.style.background_gradient(cmap='YlOrBr')


# ### <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#B90000; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #B90000">4Cs Evaluation Concept</p>
# 
# These criteria are used by gemologists to determine the quality and value of a diamond (in our case cubic zirconia).
# 
# "Then in the 1940s, Robert M. Shipley, the founder of GIA, coined the term 4Cs to help his students remember the four factors that characterize a faceted diamond: color, clarity, cut and carat weight. The concept was simple, but revolutionary. 
# 
# Today, the 4Cs of Diamond Quality is the universal method for assessing the quality of any diamond, anywhere in the world." [source](https://4cs.gia.edu/en-us/4cs-diamond-quality/#:~:text=Then%20in%20the%201940s%2C%20Robert,concept%20was%20simple%2C%20but%20revolutionary.)
# 
# Let's look at these attributes for better understanding:

# In[8]:


# Code snippet for generating the image with either cut quality, color or clarity counts.
# The illustration part was taken fro https://www.1215diamonds.com/blog/diamond-shape-cut-chart/
# P.s. I could not locate the original source for the clarity scale illustration, so you can easily find the same by googling.
# Uncomment below the line to reproduce. Change category (optional).

# train['dataset'] = 'train'
# test['dataset'] = 'test'
# train_test = pd.concat([train, test])

# catergory = 'cut'
# fig = plt.figure(figsize=(12, 5), dpi=120)
# s = sns.countplot(data=train_test, x=catergory, order=train_test[catergory].value_counts().index, hue='dataset', palette='Reds_r', width=0.6)
# fig.suptitle(f'Gem {catergory}', ha='center',  fontweight='bold', fontsize=14)
# for container in s.containers:
#     s.bar_label(container, c='black', size=12);
#     s.set_ylabel(''); s.spines['top'].set_visible(False), 
#     s.set_xlabel(''); s.spines['right'].set_visible(False),
#     s.spines['left'].set_visible(False)
#     plt.tick_params(labelleft=False)
#     plt.savefig('data/count_cuts.png'); 

# train.drop(columns='dataset', inplace=True)
# test.drop(columns='dataset', inplace=True)


# <p align="right">
#   <img src="https://drive.google.com/uc?export=view&id=1uZm4hBtHjFX3f4OWn5Lr9KwcvFAsBCsi"/>
# </p>

# <p align="right">
#   <img src="https://drive.google.com/uc?export=view&id=1iAEPc8wgj8kZjRe8xaiUC0b-IG0uyuLT"/>
# </p>

# <p align="right">
#   <img src="https://drive.google.com/uc?export=view&id=1KEeknUacuHTCaoZZ8SoPS9EpA2ZjythC"/>
# </p>

# <p align="right">
#   <img src="https://drive.google.com/uc?export=view&id=13WvFC-BlRTk5qnbuNjGKK1Sms4jaYnhm"/>
# </p>

# **Observations:**
# 
# * The majority of cubic zirconia stones in the dataset are less than 1 `carat` in weight.
# * The largest cubic zirconia stone in the dataset has a weight of 5 `carats`.
# * Most of the cubic zirconia stones in the dataset are colorless.
# * Ideal and premium `cut` cubic zirconia stones are prevalent in the dataset.
# * There are no flawless cubic zirconia stones in the dataset.
# * The largest groups of cubic zirconia stones in the dataset have `clarity` grades of SI1, VS1, and VS2. It means that the dataset stones are mostly of a good quality.

# ### <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#B90000; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #B90000">Crosstabs</p>

# In[9]:


pd.crosstab(train.color, train.clarity).style.background_gradient(cmap='YlOrRd')


# In[10]:


pd.crosstab(train.color, train.cut).style.background_gradient(cmap='YlOrRd')


# In[11]:


pd.crosstab(train.cut, train.clarity).style.background_gradient(cmap='YlOrRd')


# ### <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#B90000; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #B90000">Distributions</p>

# In[12]:


# kudos to @jcaliz
features = ['carat', 'depth', 'table', 'x', 'y', 'z']
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
        ax=ax[i], color='#9E3F00'
    )
    
    sns.kdeplot(
        test[column], label='Test',
        ax=ax[i], color='yellow'
    )
    
    sns.kdeplot(
        origin[column], label='Original',
        ax=ax[i], color='#20BEFF'
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
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.96), fontsize=25, ncol=3)
plt.tight_layout()


# In[13]:


fig = plt.figure(figsize=(6*6, 20), dpi=100)
for idx, col in enumerate(features):
    ax = plt.subplot(2, 3, idx + 1)
    sns.kdeplot(
        data=train.sample(20000), hue='price', fill=True,
        x=col, palette=['#9E3F00', 'red'], legend=False
    )
            
    ax.set_ylabel(''); ax.spines['top'].set_visible(False), 
    ax.set_xlabel(''); ax.spines['right'].set_visible(False)
    ax.set_title(f'{col}', loc='right', 
                 weight='bold', fontsize=20)

fig.suptitle(f'Features vs Price\n\n\n', ha='center',  fontweight='bold', fontsize=25)
plt.tight_layout()
plt.show()


# In[14]:


import matplotlib.gridspec as gridspec

# https://stackoverflow.com/questions/35042255/how-to-plot-multiple-seaborn-jointplot-in-subplot
class SeabornFig2Grid():

    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())


# In[15]:


g0 = sns.jointplot(data=train, x='carat', y='price', color='red')
g1 = sns.jointplot(data=train, x='depth', y='price', color='red')
g2 = sns.jointplot(data=train, x='table', y='price', color='red')
g3 = sns.jointplot(data=train, x='x', y='price', color='red')
g4 = sns.jointplot(data=train, x='y', y='price', color='red')
g5 = sns.jointplot(data=train, x='z', y='price', color='red')

fig = plt.figure(figsize=(20, 10), dpi=100)
gs = gridspec.GridSpec(2, 3)

mg0 = SeabornFig2Grid(g0, fig, gs[0])
mg1 = SeabornFig2Grid(g1, fig, gs[1])
mg2 = SeabornFig2Grid(g2, fig, gs[2])
mg4 = SeabornFig2Grid(g3, fig, gs[3])
mg5 = SeabornFig2Grid(g4, fig, gs[4])
mg6 = SeabornFig2Grid(g5, fig, gs[5])

gs.tight_layout(fig)
fig.suptitle(f'Features vs Price Join Plot\n\n\n', ha='center',  fontweight='bold', fontsize=25)
plt.show()


# In[16]:


fig, ax = plt.subplots(1, 3, figsize=(25, 7), dpi=100)
ax = ax.flatten()
for i, ft in enumerate(['cut', 'color', 'clarity']):
    sns.histplot(
        data=train,
        x="price", hue=ft,
        multiple="stack",
        palette="dark:red",
        edgecolor=".3",
        linewidth=.5,
        log_scale=True,
        ax=ax[i]
    )
fig.suptitle(f'Categorical Features vs Price\n\n\n', ha='center',  fontweight='bold', fontsize=25)
plt.tight_layout()
plt.show()


# **Observations:**
# * The distribution of the original dataset closely follows the synthetic one with minor deviations (although the reader might have a different oppinion).
# * The stacked charts supports the 4Cs concept of diamond evaluation, with the most valuable stones exhibiting the best clarity, largest weight, colorless composition, and ideal cut.

# ### <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#B90000; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #B90000">Correlations</p>

# In[17]:


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
    sns.heatmap(corr, mask=mask, linewidths=.5, cmap='Reds', annot=True)
    plt.title(title_name)
    plt.show()

plot_correlation_heatmap(origin, 'Original Dataset Correlation')
plot_correlation_heatmap(train, 'Train Dataset Correlation')
plot_correlation_heatmap(train, 'Test Dataset Correlation')


# **Comment**:
# 
# The correlation plots do not indicate any spectacular findings. The dimensions of a diamond (i.e., x, y, and z) show a strong correlation with its price and carat weight. As the dimensions of a diamond increase, so does its volume, which in turn affects its mass, as mass is equal to volume multiplied by density.

# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#B90000; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #B90000">Basic Feature Engineering</p>

# In[18]:


get_ipython().run_cell_magic('time', '', "#############################################################################################\n# CFG\n#############################################################################################\norigin = pd.read_csv(PATH_ORIGIN).drop(columns='Unnamed: 0')\ntrain = pd.read_csv(PATH_TRAIN).drop(columns='id')\ntest = pd.read_csv(PATH_TEST).drop(columns='id')\nCOMBINE = False\n\nif COMBINE:\n    train = pd.concat([train, origin]).reset_index(drop=True).dropna()\n    \nclass GemDataProcessor:\n    def __init__(self, train_data, test_data):\n        self.train_data = train_data\n        self.test_data = test_data\n        self.enc = OrdinalEncoder()\n        self.cats = ['cut', 'color', 'clarity']\n    \n    @staticmethod\n    def fe(df):\n        df['volume'] = df['x'] * df['y'] * df['z']\n        df['density'] = df['carat'] / (df['volume'] + 1e-6)\n        df['depth_per_volume'] = df['depth'] / (df['volume'] + 1e-6)\n        df['depth_per_density'] = df['depth'] / (df['density'] + 1e-6)\n        df['depth_per_table'] = df['depth'] / (df['table'] + 1e-6)\n        return df\n\n    def process_data(self):\n        self.train_data = self.fe(self.train_data)\n        self.test_data = self.fe(self.test_data)\n        \n        \n        self.train_data[self.cats] = self.enc.fit_transform(self.train_data[self.cats])\n        self.test_data[self.cats] = self.enc.transform(self.test_data[self.cats])\n        return self.train_data, self.test_data\n    \nf_e = GemDataProcessor(train, test)\ntrain, test = f_e.process_data()        \nprint(f'{gld}[INFO] Shapes after Feature Engineering Phase:'\n      f'{gld}\\n[+] train -> {red}{train.shape}'\n      f'{gld}\\n[+] test -> {red}{test.shape}\\n')\n")


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#B90000; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #B90000">Base XGB Model</p>

# In[19]:


def f_importance_plot(f_imp: pd.DataFrame, suffix: str):
    fig = plt.figure(figsize=(12, 0.20*len(f_imp)))
    plt.title(f'Feature importances {suffix}', size=16, y=1.05, 
              fontweight='bold', color='#444444')
    a = sns.barplot(data=f_imp, x='avg_imp', y='feature', 
                    palette='Reds_r', linestyle="-", 
                    linewidth=0.5, edgecolor="black")
    plt.xlabel('')
    plt.xticks([])
    plt.ylabel('')
    plt.yticks(size=11, color='#444444')
    
    for j in ['right', 'top', 'bottom']:
        a.spines[j].set_visible(False)
    for j in ['left']:
        a.spines[j].set_linewidth(0.5)
    plt.tight_layout()
    plt.show()
    
def get_mean_rmse(oof: np.array):
    """oof: ['val_idx', 'preds', 'target']"""
    oof = pd.DataFrame(np.concatenate(oof), columns=['id', 'preds', 'target']).set_index('id')
    oof.index = oof.index.astype(int)
    mean_val_rmse = mean_squared_error(oof.target, oof.preds, squared=False)
    return mean_val_rmse


# In[20]:


get_ipython().run_cell_magic('time', '', 'config = {\n        \'SEED\': 42,\n        \'FOLDS\': 5,\n        \'N_ESTIMATORS\': 4000,\n        \'COL_DROP\': [\n        ]\n    }\n\nxgb_params = {\n        \'max_depth\': 4,\n        \'learning_rate\': 0.2,\n        # \'random_state\': 42,\n        \'colsample_bytree\': 0.67,\n        \'n_jobs\': -1,\n        \'objective\': \'reg:squarederror\',\n        \'early_stopping_rounds\': 300,\n        \'verbosity\': 0,\n        \'eval_metric\': \'rmse\'\n    }\n\nX, y = train.drop(columns=[\'price\']), train.price\n\n\ncv = model_selection.KFold(n_splits=config[\'FOLDS\'], shuffle=True, random_state=config[\'SEED\'])\nfeature_importances_ = pd.DataFrame(index=test.columns)\neval_results_ = {}\nmodels_ = []\noof = []\n\nfor fold, (fit_idx, val_idx) in enumerate(cv.split(X, y)):\n    if (fold + 1) % 5 == 0 or (fold + 1) == 1:\n        print(f\'{"#" * 24} Training FOLD {fold + 1} {"#" * 24}\')\n\n    # Split the dataset according to the fold indexes.\n    X_fit = X.iloc[fit_idx]\n    X_val = X.iloc[val_idx]\n    y_fit = y.iloc[fit_idx]\n    y_val = y.iloc[val_idx]\n\n    # XGB .train() requires xgboost.DMatrix.\n    # https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.DMatrix\n    fit_set = xgb.DMatrix(X_fit, y_fit)\n    val_set = xgb.DMatrix(X_val, y_val)\n    watchlist = [(fit_set, \'fit\'), (val_set, \'val\')]\n\n    # Training.\n    # https://xgboost.readthedocs.io/en/stable/python/python_api.html#module-xgboost.training\n    eval_results_[fold] = {}\n    model = xgb.train(\n        num_boost_round=config[\'N_ESTIMATORS\'],\n        params=xgb_params,\n        dtrain=fit_set,\n        evals=watchlist,\n        evals_result=eval_results_[fold],\n        verbose_eval=False,\n        callbacks=[\n            EarlyStopping(xgb_params[\'early_stopping_rounds\'],\n                          data_name=\'val\', save_best=True)],\n    )\n\n    val_preds = model.predict(val_set)\n    val_score = mean_squared_error(y_val, val_preds, squared=False)\n    best_iter = model.best_iteration\n\n    idx_pred_target = np.vstack([val_idx, val_preds, y_val]).T  # shape(len(val_idx), 3)\n    print(f\'{" " * 20} RMSE:{blu}{val_score:.5f}{res} {" " * 6} best iteration  :{blu}{best_iter}{res}\')\n\n    oof.append(idx_pred_target)\n\n    # Stores the feature importances\n    feature_importances_[f\'gain_{fold}\'] = model.get_score(importance_type=\'gain\').values()\n    feature_importances_[f\'split_{fold}\'] = model.get_score(importance_type=\'weight\').values()\n\n    # Stores the model\n    models_.append(model)\n\nmean_val_rmse = get_mean_rmse(oof)\nprint(f\'{"*" * 45}\\n{red}Mean{res} RMSE: {red}{mean_val_rmse:.5f}\')\n')


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#B90000; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #B90000">Feature importances and OOF errors</p>

# In[21]:


f_imp_gain = feature_importances_[[col for col in feature_importances_.columns if col.startswith('gain')]].mean(axis=1)
f_imp_split = feature_importances_[[col for col in feature_importances_.columns if col.startswith('split')]].mean(axis=1)
f_imp_gain = f_imp_gain.reset_index().sort_values(by=0, ascending=False)
f_imp_gain.columns = ['feature', 'avg_imp']
f_imp_split = f_imp_split.reset_index().sort_values(by=0, ascending=False)
f_imp_split.columns = ['feature', 'avg_imp']
f_importance_plot(f_imp_gain, 'gain')
f_importance_plot(f_imp_split, 'weight')


# In[22]:


oof_df = pd.DataFrame(np.concatenate(oof), columns=['index', 'price_pred', 'price'])
fig = plt.figure(figsize=(12, 8), dpi=100)
sns.regplot(data=oof_df, x='price_pred', y='price', color='red',
            scatter=True, line_kws={"color": "black"}, scatter_kws={'s': 3})
plt.title('Price vs Price_pred');


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#B90000; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #B90000">Submission</p>

# In[23]:


def predict(X):
    y = np.zeros(len(X))
    for model in tqdm(models_):
        y += model.predict(xgb.DMatrix(X))
    return y / len(models_)

predictions = predict(test)
sub = pd.read_csv(PATH_SUB)
sub.price = predictions
sub.to_csv('first_submission', index=False)
sub.head(3)


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#B90000; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #B90000">Single Best XGB</p>
# 
# **Don't forget to upvote**.

# In[24]:


best_sub = pd.read_csv('/kaggle/input/s3e8-submissions/XGB_13.csv')
best_sub.to_csv('best_xgb_13.csv', index=False)
best_sub.head(3)


# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#B90000; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #B90000">Acknowledgement</p>
# 
# @jcaliz for .css and plotting ideas.

# ## <p style="font-family:JetBrains Mono; font-weight:normal; letter-spacing: 2px; color:#B90000; font-size:140%; text-align:left;padding: 0px; border-bottom: 3px solid #B90000">Outro</p>
# 
# The dataset does not seems extraordinal but I hope to continue working on it. Good luck in the competition!
