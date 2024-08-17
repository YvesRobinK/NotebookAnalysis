#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib.patches import ConnectionPatch
import numpy as np
import pandas as pd
import seaborn as sns
import datetime
import warnings


from scipy.stats import expon, reciprocal
from scipy.stats import randint
from scipy import stats

from pandas.plotting import scatter_matrix

from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding

from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering #kernel
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import FeatureAgglomeration
from sklearn.cluster import AffinityPropagation

from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture import GaussianMixture

from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, roc_auc_score

from xgboost import XGBClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import permutation_test_score
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.linear_model import LogisticRegression, RidgeClassifier

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler # mean0 std1
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning


from yellowbrick.cluster import KElbowVisualizer

get_ipython().run_line_magic('matplotlib', 'inline')

from  warnings import simplefilter
simplefilter("ignore", category=UserWarning)


# <p style= "background-color:#000000;font-family:Georgia;color:#FFFFFF;font-size:300%;text-align:center;border-radius:10px 10px;border-style:solid;border-width:3px;border-color:#000000;"><b>Titanic Data Science</b></p>
# 
# ## Please give me an UPVOTE if you can. Your UPVOTE will be a great encouragement to me

# <p style="background-color:#000000;font-family:Georgia;color:#FFFFFF;font-size:150%;text-align:center;border-radius:10px 10px;">Data Science Workflow</p>
# 
# 1. Understand the framework and big picture of the problem.
# 2. Analyze the data to develop a personal understanding.
# 3. Prepare data to make it easier for machine learning algorithms to find patterns in the data.
# 4. Try different models and narrow it down to the best few.
# 5. Fine-tune the models, use them as an ensemble, and combine them into a solution.
# 6. Present the solution.

# <p style="background-color:#000000;font-family:Georgia;color:#FFFFFF;font-size:150%;text-align:center;border-radius:10px 10px;">DATA Observation</p>

# In[2]:


main_test = pd.read_csv('../input/titanic/test.csv')
main_test.info()


# In[3]:


data = pd.read_csv('../input/titanic/train.csv')
data.info()


# In[4]:


cluster_data_c = data.copy()
cluster_main_test = main_test.copy()


# <p style="background-color:#000000;font-family:Georgia;color:#FFFFFF;font-size:150%;text-align:center;border-radius:10px 10px;">DATA Concatenation</p>

# >Merge the pre-divided data sets. This operation makes it possible to make the same changes during data conversion.

# In[5]:


cluster_data = pd.concat([cluster_data_c, cluster_main_test], axis=0)
cluster_data.describe().T.style.bar(subset=['mean'], color='#606ff2').background_gradient(subset=['std'], cmap='mako_r').background_gradient(subset=['50%'], cmap='mako_r')


# <p style="background-color:#000000;font-family:Georgia;color:#FFFFFF;font-size:300%;text-align:center;border-radius:10px 10px;">EDA</p>

# 1. Make a copy of the data for observation, or reduce it to the required size if the data is large.
# 2. examine the attributes of the data and their characteristics
#     - Missing values
#     - Categorical/textual/integer/floating point numbers
#     - Noise presence and type (stochastic, outlier, rounding error)
#     - Type of distribution (Gaussian, Uniform, Logarithmic)
# 3. visualize the data
# 4. examine correlation of features
# 5. Identify the transformation to be applied

# In[6]:


eda_data = data.reset_index().copy()


# In[7]:


with plt.style.context('fivethirtyeight'): # background color set rcPram
    fig, ax = plt.subplots(1, 2, figsize=(18, 6))
    eda_data['Survived'].value_counts().plot.pie(explode=[0, 0.18], autopct='%1.1f%%',
                                                 shadow=True, colors=['#682F2F', '#F3AB60'], startangle=70, ax=ax[0])
    
    age_bin = pd.qcut(eda_data['Age'], 10)
    age_counts = sns.barplot(x=age_bin.sort_index().value_counts().index, y=age_bin.value_counts().values,
                             linewidth=0.5, ec='black', zorder=3, palette='rocket', ax=ax[1])
    for i in age_counts.patches:
        values = f'{i.get_height():,.0f} | {i.get_height() / age_bin.shape[0]:,.1%}'
        x = i.get_x() + i.get_width() / 2
        y = i.get_y() + i.get_height() + 5
        age_counts.text(x, y, values, ha='center', va='center', fontsize=10, 
                        bbox=dict(facecolor='none', edgecolor='black', boxstyle='round', linewidth=0.2))
        ax[1].set_xlabel('ALL Age Bins')
        ax[1].set_ylabel('Counts')


# In[8]:


with plt.style.context('fivethirtyeight'): # background color set
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    fig.subplots_adjust(wspace=0)
    eda_data['Survived'].value_counts().plot.pie(explode=[0, 0.2], autopct='%1.1f%%',
                                                 shadow=True, colors=['#faf0e6', '#F3AB60'], startangle=70, ax=ax1)
    
    colors = ['#682f2f', '#774343', '#865858', '#956D6D', '#A48282',
              '#B39797', '#C2ABAB', '#D1C0C0', '#E0D5D5', '#EFEAEA']
    bar_per = eda_data['Survived'].groupby(age_bin).count()[::-1]
    bottom = 0
    for i in range(len(bar_per.values)):
        height = bar_per.values[i] / bar_per.values.sum()
        ax2.bar(-0.2, height=height, width=.2 ,bottom=bottom, color=colors[i], edgecolor='black')
        y = bottom + ax2.patches[i].get_height() / 2.5
        bottom += height
        values = f'{ax2.patches[i].get_height(): ,.0%} | {bar_per.values[i]: ,.0f}' # "%d%%" % (ax2.patches[i].get_height() *100)
        ax2.text(-0.2, y, values, ha='center',
                 bbox=dict(facecolor='none', edgecolor='black', boxstyle='round', linewidth=0.2))

    ax2.set_title('Surviver Age Per')
    # reversed legend
    ax2.legend(('0-13', '14-18', '19-21', '22-24', '25-27','28-30', '31-35', '36-40', '41-49', '50-80'), 
               title='38.4% -> 100%', loc='center', prop={'size':15})
    ax2.axis('off')
    ax2.set_xlim(-2.0 * .2, 2.0 * .2)

    theta1, theta2 = ax1.patches[0].theta1, ax1.patches[0].theta2
    center, r = ax1.patches[0].center, ax1.patches[0].r
    
    x = r * np.cos(np.pi / 173 * theta2) + center[0]
    y = r * np.sin(np.pi / 180 * theta2) + center[1]
    con = ConnectionPatch(xyA=(-.3, 0.01), xyB=(x, y), coordsA=ax2.transData, coordsB=ax1.transData, shrinkA=1, arrowstyle='<-')
    con.set_color('#e9967a')#e9967a
    con.set_linewidth(2)
    ax2.add_artist(con)

    x = r * np.cos(np.pi / 216 * theta1) + center[0]
    y = r * np.sin(np.pi / 180 * theta1) + center[1]
    con = ConnectionPatch(xyA=(-.3, 0.998), xyB=(x, y), coordsA=ax2.transData, coordsB=ax1.transData, shrinkA=1, arrowstyle='<-')
    con.set_color('#e9967a')#e9967a
    ax2.add_artist(con)
    con.set_linewidth(2)


# In[9]:


def ms_pair_plot(values, hue='Survived'):
    with plt.style.context('fivethirtyeight'):
        fig, ax = plt.subplots(1, 2, figsize=(18, 6))
        for i in range(len(values)):
            #fig, ax = plt.subplots(1, 2, figsize=(18, 6)) when plot_buplication is performed, inclusion 
            value = values[i]
            sns.histplot(x=value, hue=hue, data=eda_data, kde=True, palette='rocket', ax=ax[i])
            ax[i].axvline(x=eda_data[value].mean(), color='g', linestyle='--', linewidth=3)
            ax[i].axvline(x=eda_data[value].std(), color='c', linestyle=':', linewidth=3)
            ax[i].text(eda_data[value].mean(), eda_data[value].mean(), "<--Mean", horizontalalignment='left', size='small', color='black', weight='semibold')
            ax[i].text(eda_data[value].std(), eda_data[value].std(), "Std-->", horizontalalignment='right', size='small', color='black', weight='semibold')
            sns.despine()


# In[10]:


values = ['Age', 'Fare']
ms_pair_plot(values)


# In[11]:


values = ['SibSp', 'Parch']
ms_pair_plot(values)


# In[12]:


with plt.style.context('fivethirtyeight'):
    fig = plt.figure(figsize=(18, 6))
    sns.violinplot(x="Sex", y="Survived", data=eda_data, palette=['#682F2F', '#F3AB60'])
    #plt.ylim(-50, 200)


# In[13]:


survived_count = pd.crosstab(eda_data['Sex'], eda_data['Survived'])
survived_pct = survived_count.div(survived_count.sum(1), axis=0)
with plt.style.context('fivethirtyeight'):
    survived_pct.plot.barh(stacked=True, figsize=(19, 6), alpha=0.9, grid=False, color=['#682F2F', '#F3AB60'])


# In[14]:


survived_count = pd.crosstab(eda_data['SibSp'], eda_data['Survived'])
survived_pct = survived_count.div(survived_count.sum(1), axis=0)
with plt.style.context('fivethirtyeight'):
    survived_pct.plot.barh(stacked=True, figsize=(19, 6), alpha=0.9, grid=False, color=['#682F2F', '#F3AB60'])


# In[15]:


survived_count = pd.crosstab(eda_data['Parch'], eda_data['Survived'])
survived_pct = survived_count.div(survived_count.sum(1), axis=0)
with plt.style.context('fivethirtyeight'):
    survived_pct.plot.barh(stacked=True, figsize=(19, 6), alpha=0.9, grid=False, color=['#682F2F', '#F3AB60'])


# In[16]:


survived_count = pd.crosstab(eda_data['Fare'], eda_data['Survived'])
survived_pct = survived_count.div(survived_count.sum(1), axis=0)
with plt.style.context('fivethirtyeight'):
    survived_pct.plot.bar(stacked=True, alpha=0.9, figsize=(19, 6),
                          grid=False, use_index=None, logy=False,
                          rot=60, yticks=[], xticks=range(0, 251, 50),
                          title='Surviver Fare Pattern', color=['#682F2F', '#F3AB60'])


# In[17]:


def trimming_ax(ax, N):
    f_axs = ax.flat
    for ax in f_axs[N:]:
        ax.remove()
    return f_axs[:N]

def cluster_bar_plot(data, product_list, cols=3, figsize=(19, 6)):
    product_list.append('')
    length = len(product_list)
    product_list.remove('')
    if length % 2 == 0:
        rows = length // cols
    else:
        rows = length // cols + 1
    with plt.style.context('fivethirtyeight'):
        
        ax = plt.figure(figsize=figsize, constrained_layout=True).subplots(rows, cols)
        ax = trimming_ax(ax, length)
        for i, product in enumerate(product_list):
            cluster = data.query("Survived == {}".format(i))
            
            sns.barplot(x="Survived", y=product, data=data, palette='rocket', ax=ax[i])
            ax[i].legend(labels=['{}'.format(product)], title='P', loc=2, bbox_to_anchor=(1,1))
        
            sns.boxenplot(x="Survived", y="Fare", data=data, palette='rocket', ax=ax[-1])
            ax[-1].legend(labels=['Survived'], title='Survived_Number', loc=2, bbox_to_anchor=(1,1))


# In[18]:


product_List = ['Sex','Survived', 'Age', 'Fare', 'SibSp','Parch']


cluster_bar_plot(eda_data, product_List, figsize=(19, 7))


# In[19]:


def Survived_hist_plot(data, columns, cols=4, figsize=(10, 5)):
    #cols = cols
    rows = len(np.unique(data['Survived'])) // cols + 1
    bins = np.round(np.log(len(data)) + 1).astype(int) # Sturgess Formula : k=log2N+1
    ax = plt.figure(figsize=figsize, constrained_layout=True).subplots(rows, cols)
    ax = trimming_ax(ax, len(np.unique(data['Survived'])))
    with plt.style.context('fivethirtyeight'):
        for i in np.unique(data['Survived']):
            cluster = data.query("Survived == {}".format(i))
            # replace plot
            sns.histplot(x=columns, data=cluster.reset_index(), bins=bins, ax=ax[i])
            #sns.countplot(x='Age', data=cluster.reset_index(), ax=ax[i])
            ax[i].legend(labels=['{}'.format(i)], title='Survived', loc=2, bbox_to_anchor=(1,1))


# In[20]:


Survived_hist_plot(eda_data, columns='Age', figsize=(19, 7))


# In[21]:


Survived_hist_plot(eda_data, columns='Sex', figsize=(19, 7))


# <p style="background-color:#000000;font-family:Georgia;color:#FFFFFF;font-size:300%;text-align:center;border-radius:10px 10px;">Other Data Plot</p>
# 
# #### ・Data Plot & Normalization
# #### ・Dendrogram & HeatMap

# In[22]:


def density_plot(data):
    density_per_col = [stats.gaussian_kde(col) for col in data.values.T]
    x = np.linspace(np.min(data.values), np.max(data.values), 100)
    
    with plt.style.context('fivethirtyeight'):
        fig, ax = plt.subplots(figsize=(18, 6))
        for density in density_per_col:
            ax.plot(x, density(x))
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        fig.legend(data, loc='right')


# In[23]:


counts = data.copy()
counts.reset_index(inplace=True)
counts.drop(['Name', 'Cabin', 'Ticket', 'PassengerId', 'Sex', 'Embarked'], axis=1, inplace=True)
counts.fillna(0, inplace=True)
counts_nd = counts.values
log_counts = np.log(counts_nd + 1)
log_counts_pd = pd.DataFrame(log_counts, columns=counts.columns)


# In[24]:


density_plot(log_counts_pd)


# In[25]:


def quantile_norm(X):
    quantile = np.mean(np.sort(X, axis=0), axis=1) # quantile calculus
    rank = np.apply_along_axis(stats.rankdata, 0, X)
    rank_indices = rank.astype(np.int) - 1
    X_index = quantile[rank_indices]
    return X_index

def quantile_log(X):
    X_log = np.log(X + 1)
    Xi_log = quantile_norm(X_log)
    Xi_log_pd = pd.DataFrame(Xi_log, columns=X.columns)
    return Xi_log_pd


# In[26]:


count_normalized = quantile_log(counts)


# In[27]:


density_plot(count_normalized)


# In[28]:


def most_variable_rows(data, *arg):
    rowvar = np.var(data, axis=1)
    sort_indices = np.argsort(rowvar)
    variable_data = data[sort_indices, :]
    return variable_data

from scipy.cluster.hierarchy import linkage

def bicluster(data, linkage_method='average', distance_metric='correlation'):
    y_rows = linkage(data, method=linkage_method, metric=distance_metric)
    y_cols = linkage(data.T, method=linkage_method, metric=distance_metric)
    return y_rows, y_cols


# In[29]:


from scipy.cluster.hierarchy import dendrogram, leaves_list

def clear_spines(axes):
    for loc in ['left', 'right', 'top', 'bottom']:
        axes.spines[loc].set_visible(False)
    axes.set_xticks([])
    axes.set_yticks([])

def plot_bicluster(data, row_linkage, col_linkage, row_nclusters=10, col_nclusters=5):
    fig = plt.figure(figsize=(10, 10))
    
    ax1 = fig.add_axes([0.09, 0.1, 0.2, 0.6])
    
    threshold_r = (row_linkage[-row_nclusters, 2] +
                   row_linkage[-row_nclusters+1, 2]) / 2
    with plt.rc_context({'lines.linewidth': 0.75}):
        dendrogram(row_linkage, orientation='left',
                   color_threshold=threshold_r, ax=ax1)
    clear_spines(ax1)
    
    ax2 = fig.add_axes([0.3, 0.71, 0.6, 0.2]) 
    threshold_c = (col_linkage[-col_nclusters, 2] +
                   col_linkage[-col_nclusters+1, 2]) / 2
    with plt.rc_context({'lines.linewidth': 0.75}):
        dendrogram(col_linkage,
                   color_threshold=threshold_c, ax=ax2)
    clear_spines(ax2)
    
    ax = fig.add_axes([0.3, 0.1, 0.6, 0.6])
    
    idx_rows = leaves_list(row_linkage)
    data = data[idx_rows, :]
    idx_cols = leaves_list(col_linkage)
    data = data[:, idx_cols]
    
    im = ax.imshow(data, aspect='auto', origin='lower', cmap='YlGnBu_r')
    clear_spines(ax)
    
    ax.set_xlabel('Columns')
    ax.set_ylabel('Index', labelpad=150)
    axcolor = fig.add_axes([0.91, 0.1, 0.02, 0.6])
    plt.colorbar(im, cax=axcolor)


# In[30]:


count_log = np.log(counts_nd + 1)
count_var = most_variable_rows(count_log)

yr, yc = bicluster(count_var, linkage_method='ward', distance_metric='euclidean')

plot_bicluster(count_var, yr, yc)


# <p style="background-color:#000000;font-family:Georgia;color:#FFFFFF;font-size:300%;text-align:center;border-radius:10px 10px;">Pre-Processing & Feature Enginiering</p>

# # Pre-Processing & Feature Enginiering
# 1. **Data cleaning**
#     - Fix or remove outliers
#     - Fill missing values (0, mean, median)
# 3. **Feature engineering**
#     - Discretization of continuous value features
#     - Decomposition of features (categorical)
#     - Add transformations that are expected to be effective on features (log(x), sqrt(x), x**2)
#     - Aggregate the features to create new features
# 4. **Feature scaling**
#     - Standardization and normalization of features

# In[31]:


data.isnull().sum(), main_test.reset_index().isnull().sum()


# In[32]:


cluster_data['FamilySize'] = cluster_data['SibSp'] + cluster_data['Parch'] + 1
cluster_data['IsAlone'] = np.where(cluster_data['FamilySize'] <= 1, 1, 0)
cluster_data.loc[cluster_data['FamilySize'] > 0, 'travelled_alone'] = 'No'
cluster_data.loc[cluster_data['FamilySize'] == 0, 'travelled_alone'] = 'Yes'

cluster_data['Honorific'] = cluster_data.Name.str.split(',', -1,
                                                        expand=True)[1].str.split('.', 1,
                                                                                  expand=True)[0].str.strip().replace({'Mlle': 'Miss','Ms': 'Miss', 'Lady': 'Noble', 'Don': 'Noble',
                                                                                                                       'Jonkheer': 'Noble', 'the Countess': 'Noble', 'Sir': 'Noble',
                                                                                                                       'Countess': 'Noble',
                                                                                                                       'Mme': 'Mrs', 'Capt': 'Soldier', 'Major': 'Soldier', 'Col': 'Soldier', 
                                                                                                                       'Rev': 'Mr', 'Dr': 'Mr', 'Dona': 'Noble', 'Master': 'Soldier'})
cluster_data['FamilyName'] = cluster_data.Name.str.split(',', -1, expand=True)[0]
cluster_data['FullName'] = cluster_data.Name.str.split(',', -1, expand=True)[1].str.split('.', 1, expand=True)[1].str.strip()
cluster_data['FirstName'] = cluster_data.FullName.str.split(' ', 1, expand=True)[0].str.strip('(').str.strip(')')
cluster_data['NameLength'] = cluster_data.FullName.apply(lambda x: len(x))
cluster_data['Surname'] = cluster_data.Name.str.extract(r'([A-Za-z]+),', expand=False)
cluster_data['TicketPrefix'] = cluster_data.Ticket.str.extract(r'(.*\d)', expand=False)
cluster_data['SurnameTicket'] = cluster_data['Surname'] + cluster_data['TicketPrefix']
cluster_data['IsFamily'] = cluster_data.SurnameTicket.duplicated(keep=False).astype(int)
cluster_data['Child'] = cluster_data.Age.map(lambda x: 1 if x <=16 else 0)


bins = [0, 2, 12, 17, 60, np.inf]
labels = ['baby', 'child', 'teenager', 'adult', 'elderly']
age_groups = pd.cut(cluster_data.Age, bins, labels=labels)
cluster_data['AgeGroup'] = age_groups

#Cont_Features = ['Age', 'Fare']
#num_bins = 5
#for feature in Cont_Features:
#    bin_feature = feature + 'Bin'
#    cluster_data[bin_feature] = pd.qcut(cluster_data[feature], num_bins)
#    label = LabelEncoder()
#    cluster_data[bin_feature] = label.fit_transform(cluster_data[bin_feature])


cluster_data['Age*Class'] = cluster_data.Age * cluster_data.Pclass


FamilyWithChild = cluster_data[(cluster_data.IsFamily == 1) & (cluster_data.Child == 1)]['SurnameTicket'].unique()
cluster_data['FamilyId'] = 0
for ind, identifier in enumerate(FamilyWithChild):
    cluster_data.loc[cluster_data.SurnameTicket == identifier, ['FamilyId']] = ind + 1

cluster_data['FamilySurvival'] = 1
Survived_by_FamilyId = cluster_data.groupby('FamilyId').Survived.sum()
for i in range(1, len(FamilyWithChild)+1):
    if Survived_by_FamilyId[i] >= 1:
        cluster_data.loc[cluster_data.FamilyId == i, ['FamilySurvival']] = 2
    elif Survived_by_FamilyId[i] == 0:
        cluster_data.loc[cluster_data.FamilyId == i, ['FamilySurvival']] = 0


cluster_data.drop('Name', axis=1, inplace=True)
cabin_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 1, 'M': 8}
cluster_data['Deck'] = cluster_data['Cabin'].str[0].fillna('M').replace(cabin_map)
cluster_data['Ticket'] = cluster_data['Ticket'].str.split(' ').str.get(-1).str.get(0).str.replace('L', '2').astype(np.int64)


cat_features = cluster_data.columns[~cluster_data.columns.isin(['Survived', 'Fare', 'FamilySize', 'Pclass',
                                                                'PassengerId', 'Age', 'Ticket', 'Parch', 'Cabin',
                                                                'Embarked'])]
cluster_data = pd.get_dummies(cluster_data, columns=cat_features)
dummie_data = cluster_data.reset_index()

dummie = pd.get_dummies(dummie_data['Cabin'])
data_dummie = dummie_data.drop(['Cabin'], axis=1).join(pd.DataFrame(dummie.sum(axis=1), columns=['Cabin']))
data_rep = data_dummie.fillna({'Age': data_dummie['Age'].mean(), 'Embarked': data_dummie['Embarked'].fillna(method='ffill'), 'Fare': data_dummie['Fare'].mean()})
data_rep['Fare'] = data_rep['Fare'].round().astype(np.int64)
data_rep['Age'] = data_rep['Age'].astype(np.int64)
data_rep['Fare'].where((np.abs(data_rep['Fare']) < data_rep['Fare'].quantile(0.997, )), 500, inplace=True) # 300

data_rep_cluster = data_rep.drop('Survived', axis=1)
data_rep.drop(['index', 'PassengerId'], axis=1,  inplace=True)


# <p style="background-color:#000000;font-family:Georgia;color:#FFFFFF;font-size:200%;text-align:center;border-radius:10px 10px;">Data Splitting</p>

# ## Train & Test
# 
# >Divided into Train set used for Model training and data to make predictions.

# In[33]:


train = data_rep.iloc[cluster_data_c.index]
test = data_rep.iloc[cluster_main_test.index+cluster_data_c.shape[0]].drop(columns=['Survived'])
X = train.drop(columns='Survived', axis=1) # .to_numpy()
y = train['Survived'] # .to_numpy()

data_labels = y.to_numpy()


# In[34]:


data_num = X.drop(["Embarked"], axis=1)
number_attribs = list(data_num)


# In[35]:


categorie_attribs = ["Embarked"] 

number_pipeline = Pipeline([
    ('std_scaler', StandardScaler()),
])

cluster_test_pipeline = ColumnTransformer([
    ("number", number_pipeline, number_attribs),
    ("categorie", OneHotEncoder(), categorie_attribs),
])


# In[36]:


cluster_data_prepared = cluster_test_pipeline.fit_transform(X)
cluster_main_test = cluster_test_pipeline.transform(test)


# In[37]:


X_train, X_val, y_train, y_val = train_test_split(cluster_data_prepared, data_labels, test_size=0.25, stratify=data_labels)


# -----
# <p style="background-color:#000000;font-family:Georgia;color:#FFFFFF;font-size:300%;text-align:center;border-radius:10px 10px;">Model</p>
# 
# <p style="background-color:#000000;font-family:Georgia;color:#FFFFFF;font-size:200%;text-align:center;border-radius:10px 10px;">GridSearch RandomizedSearch</p>

# - A grid search is used to narrow down the rough range, and a random search is used to further understand the detailed values and obtain the hyperparameters of the model.

# -----
# <p style="background-color:#000000;font-family:Georgia;color:#FFFFFF;font-size:200%;text-align:center;border-radius:10px 10px;">Best Score Model</p>
# 
# # **GradientBoosting** 

# ## GridSearchCV

# In[38]:


#gb_model = GradientBoostingClassifier()
#gb_param_grid = {'learning_rate':[0.1, 0.01, 0.001], 'max_depth':[5, 10], 'n_estimators':[10, 100, 200, 300]}
#gb_model.get_params().keys()


# In[39]:


#gb_s_model = GridSearchCV(gb_model, gb_param_grid, cv=10, scoring='accuracy')
#gb_s_model.fit(X_train, y_train)


# In[40]:


#gb_s_model.best_params_
#{'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 100} samples=10000
gb_model = GradientBoostingClassifier(learning_rate=0.1, max_depth=5, n_estimators=100) # warm_start=True
gb_model.fit(X_train, y_train)


# In[41]:


#gb = gb_s_model.best_estimator_
y_pred_gb = gb_model.predict(X_val)


# In[42]:


print('Precision : {} / Recall : {}'.format(precision_score(y_val, y_pred_gb), recall_score(y_val, y_pred_gb)))
print(classification_report(y_val, y_pred_gb))
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_val, y_pred_gb), display_labels=gb_model.classes_)
disp.plot(cmap='Blues');


# In[43]:


main_pred_gb = gb_model.predict(cluster_main_test)


# In[44]:


#pd.DataFrame({'PassengerId': np.arange(892, 1310, 1), 'Survived': main_pred_gb.astype(int)}).to_csv('submission_test.csv', index=0) # 0.80143


# In[45]:


kfold_1 =StratifiedKFold(n_splits=5,shuffle=True) #random_state=42


# In[46]:


y_score = cross_val_predict(gb_model, X_train, y_train, cv=kfold_1, method='decision_function')


# ### PR_corve

# In[47]:


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precision[:-1], "b--", label="Precision") # [:-1]
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall") # [:-1]
    plt.legend()


# In[48]:


precision, recall, thresholds = precision_recall_curve(y_train, y_score)


# In[49]:


plot_precision_recall_vs_threshold(precision, recall, thresholds)


# In[50]:


threshold_90_precision = thresholds[np.argmax(precision >= 0.80)] # 0.85 # 0.80
threshold_90_precision


# In[51]:


y_train_pred_90 = (y_score >= threshold_90_precision)


# In[52]:


y_train_pred_90.shape # y_train


# In[53]:


gb_model.fit(X_train, y_train_pred_90)


# In[54]:


y_pred_90_gb = gb_model.predict(X_val)


# In[55]:


print('Precision : {} / Recall : {}'.format(precision_score(y_val, y_pred_90_gb), recall_score(y_val, y_pred_90_gb)))
print(classification_report(y_val, y_pred_90_gb))
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_val, y_pred_90_gb), display_labels=gb_model.classes_)
disp.plot(cmap='Blues');


# In[56]:


main_pred_90_gb = gb_model.predict(cluster_main_test)


# In[57]:


#pd.DataFrame({'PassengerId': np.arange(892, 1310, 1), 'Survived': main_pred_90_gb.astype(int)}).to_csv('submission_best.csv', index=0)


# <p style="background-color:#000000;font-family:Georgia;color:#FFFFFF;font-size:200%;text-align:center;border-radius:10px 10px;">early stopping</p>

# - One way to regularize a model
# - In stochastic gradient descent and mini-batch gradient descent methods, the curve is not as smooth as in batch gradient descent, and there is a risk of being caught in a local minimum. Therefore, if the verification error keeps rising above the minimum for a while, we can construct a syntax to roll back to the last minimum.
# - Early termination can also be used in boosting algorithms to find the optimal number of decision trees.

# In[58]:


errors = [precision_score(y_val, y_pred)
          for y_pred in gb_model.staged_predict(X_val)]


# In[59]:


bst_n_estimators = np.argmax(errors) + 1


# In[60]:


bst_n_estimators


# In[61]:


gb_best = GradientBoostingClassifier(learning_rate=0.1, max_depth=5, n_estimators=bst_n_estimators)


# In[62]:


gb_best.fit(X_train, y_train_pred_90)


# In[63]:


y_pred_90_gb_best = gb_model.predict(X_val)


# In[64]:


print('Precision : {} / Recall : {}'.format(precision_score(y_val, y_pred_90_gb_best), recall_score(y_val, y_pred_90_gb_best)))
print(classification_report(y_val, y_pred_90_gb_best))
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_val, y_pred_90_gb_best), display_labels=gb_best.classes_)
disp.plot(cmap='Blues');


# In[65]:


main_pred_90_gb_best = gb_best.predict(cluster_main_test)


# In[66]:


#pd.DataFrame({'PassengerId': np.arange(892, 1310, 1),
#              'Survived': pd.DataFrame(main_pred_90_gb_best).replace({False: 0, True :1}).to_numpy().T[0]}).to_csv('submission_01.csv', index=0)


# ## RandomizedSearchCV

# In[67]:


param_rand = {'learning_rate': reciprocal(0.001, 1), 
              'max_depth': randint(low=3, high=10)}


# In[68]:


gb_best_rd = GradientBoostingClassifier(subsample=0.25, n_estimators=bst_n_estimators, warm_start=True)


# In[69]:


rnd_search = RandomizedSearchCV(gb_best_rd, param_rand, n_iter=10, cv=kfold_1, scoring='f1')


# In[70]:


rnd_search.fit(X_train, y_train_pred_90)


# In[71]:


gb_best_p = GradientBoostingClassifier(learning_rate=0.12727932524008223, max_depth=3, subsample=0.25, n_estimators=bst_n_estimators, warm_start=True)


# In[72]:


gb_best_p.fit(X_train, y_train_pred_90)


# In[73]:


y_pred_90_gb_best_rnd = gb_best_p.predict(X_val)


# In[74]:


print('Precision : {} / Recall : {}'.format(precision_score(y_val, y_pred_90_gb_best_rnd), recall_score(y_val, y_pred_90_gb_best_rnd)))
print(classification_report(y_val, y_pred_90_gb_best_rnd))
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_val, y_pred_90_gb_best_rnd), display_labels=gb_best_p.classes_)
disp.plot(cmap='Blues');


# In[75]:


main_pred_90_gb_best_rnd = gb_best.predict(cluster_main_test)


# In[76]:


#pd.DataFrame({'PassengerId': np.arange(892, 1310, 1),'Survived': pd.DataFrame(main_pred_90_gb_best_rnd).replace({False: 0, True :1}).to_numpy().T[0]}).to_csv('submission_02.csv', index=0)


# In[77]:


precision_score(y_train, y_train_pred_90)


# In[78]:


recall_score(y_train, y_train_pred_90)


# In[79]:


fpr, tpr, thresholds_r = roc_curve(y_train_pred_90, y_score)


# In[80]:


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')


# In[81]:


plot_roc_curve(fpr, tpr)


# In[82]:


roc_auc_score(y_train_pred_90, y_score)


# In[83]:


y_train_pred = cross_val_predict(gb_model, X_train, y_train_pred_90, cv=3)


# <p style="background-color:#000000;font-family:Georgia;color:#FFFFFF;font-size:200%;text-align:center;border-radius:10px 10px;">Other Models</p>
# 
# - Publish to another notebook
# - Thank you for Reading. 
# - Please give me an UPVOTE if you can. Your UPVOTE will be a great encouragement to me
