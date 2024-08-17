#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This notebook is intended to extract useful insights for the datasets of ‘Tabular Playground Series - Feb 2021’ competition in Kaggle. For this competition, it is required to tackle the Regression problem to predict a continuous target based on a number of feature columns given in the data. All of the feature columns, cat0 - cat9 are categorical, and the feature columns cont0 - cont13 are continuous.
# 
# We are going to perform the complete and comprehensive EDA as follows
# -	Automate the generic aspects of EDA with AutoViz, one of the leading freeware Rapid EDA tools in Pythonic Data Science world
# -	Deep into the problem-specific advanced analytical questions/discoveries with the custom manual EDA routines programmed on top of standard capabilities of Plotly and Matplotlib
# 

# # Initial Preparations
# 
# We are going to start with the essential pre-requisites as follows
# 
# - importing the standard Python packages we need to use down the road
# - programming the useful automation routines for repeatable data visualizations we are going to draw in the Advance Analytical EDA trials down the road

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt
from typing import Tuple, List, Dict

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.offline


# read data
in_kaggle = True

def get_data_file_path(is_in_kaggle: bool) -> Tuple[str, str, str]:
    train_path = ''
    test_path = ''
    sample_submission_path = ''

    if is_in_kaggle:
        # running in Kaggle, inside the competition
        train_path = '../input/tabular-playground-series-feb-2021/train.csv'
        test_path = '../input/tabular-playground-series-feb-2021/test.csv'
        sample_submission_path = '../input/tabular-playground-series-feb-2021/sample_submission.csv'
    else:
        # running locally
        train_path = 'data/train.csv'
        test_path = 'data/test.csv'
        sample_submission_path = 'data/sample_submission.csv'

    return train_path, test_path, sample_submission_path

# cascatter implementation - reused from https://github.com/myrthings/catscatter/blob/master/catscatter.py
# (c) Myr Barnés, 2020
# More info about this function is available at
# - https://towardsdatascience.com/visualize-categorical-relationships-with-catscatter-e60cdb164395
# - https://github.com/myrthings/catscatter/blob/master/README.md
def catscatter(df,colx,coly,cols,color=['grey','black'],ratio=10,font='Helvetica',save=False,save_name='Default'):
    '''
    Goal: This function create an scatter plot for categorical variables. It's useful to compare two lists with elements in common.
    Input:
        - df: required. pandas DataFrame with at least two columns with categorical variables you want to relate, and the value of both (if it's just an adjacent matrix write 1)
        - colx: required. The name of the column to display horizontaly
        - coly: required. The name of the column to display vertically
        - cols: required. The name of the column with the value between the two variables
        - color: optional. Colors to display in the visualization, the length can be two or three. The two first are the colors for the lines in the matrix, the last one the font color and markers color.
            default ['grey','black']
        - ratio: optional. A ratio for controlling the relative size of the markers.
            default 10
        - font: optional. The font for the ticks on the matrix.
            default 'Helvetica'
        - save: optional. True for saving as an image in the same path as the code.
            default False
        - save_name: optional. The name used for saving the image (then the code ads .png)
            default: "Default"
    Output:
        No output. Matplotlib object is not shown by default to be able to add more changes.
    '''
    # Create a dict to encode the categeories into numbers (sorted)
    colx_codes=dict(zip(df[colx].sort_values().unique(),range(len(df[colx].unique()))))
    coly_codes=dict(zip(df[coly].sort_values(ascending=False).unique(),range(len(df[coly].unique()))))
    
    # Apply the encoding
    df[colx]=df[colx].apply(lambda x: colx_codes[x])
    df[coly]=df[coly].apply(lambda x: coly_codes[x])
    
    
    # Prepare the aspect of the plot
    plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True
    plt.rcParams['font.sans-serif']=font
    plt.rcParams['xtick.color']=color[-1]
    plt.rcParams['ytick.color']=color[-1]
    plt.box(False)

    
    # Plot all the lines for the background
    for num in range(len(coly_codes)):
        plt.hlines(num,-1,len(colx_codes)+1,linestyle='dashed',linewidth=1,color=color[num%2],alpha=0.5)
    for num in range(len(colx_codes)):
        plt.vlines(num,-1,len(coly_codes)+1,linestyle='dashed',linewidth=1,color=color[num%2],alpha=0.5)
        
    # Plot the scatter plot with the numbers
    plt.scatter(df[colx],
               df[coly],
               s=df[cols]*ratio,
               zorder=2,
               color=color[-1])
    
    # Change the ticks numbers to categories and limit them
    plt.xticks(ticks=list(colx_codes.values()),labels=colx_codes.keys(),rotation=90)
    plt.yticks(ticks=list(coly_codes.values()),labels=coly_codes.keys())
    plt.xlim(xmin=-1,xmax=len(colx_codes))
    plt.ylim(ymin=-1,ymax=len(coly_codes))
    
    # Save if wanted
    if save:
        plt.savefig(save_name+'.png')
        
# auxiliary function to build a tailored catscatter plot adapted to the current dataset
def build_catscatter_plot(
    data: pd.DataFrame,
    cat_x: str,
    cat_y: str,
    fig_size_x: int = 80,
    fig_size_y: int = 80,
    ratio: int = 10,
    font_size: int = 80
):
    # aggregate record counts by different labels of cat_x and cat_y
    agg_data = data.groupby([cat_x, cat_y]).size().reset_index(name='record_count')
    # define the color map
    colors=['blue', 'grey', 'green']
    
    # create the plot
    plt.figure(figsize=(80,80))
    catscatter(agg_data , cat_x, cat_y, 'record_count', font='Helvetica', color=colors, ratio=ratio)

    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.show()
    
# auxiliary function to build a tailored Plotly Express treemap plot adapted to the current dataset
def build_treemap(
    data: pd.DataFrame,
    path_cols: List[str],
    value_col: str,
    color_col: str,
    mid_point: float=0.5,
):
    prefix = 'Tree Map for Path: '
    separator = '-'
    plot_title = "".join([prefix, separator.join(path_cols)])
    
    plot_title = "".join([plot_title, " (areas sized by ", value_col, ", colored by ", color_col, ")"])
    
    fig = px.treemap(
        data, 
        path=path_cols, 
        values=value_col, 
        color=color_col, 
        color_continuous_midpoint=mid_point, 
        color_continuous_scale=px.colors.diverging.Portland,
        title=plot_title
    )
    fig.show()

# auxiliary function to build a tailored Plotly Express treemap plot adapted to the current dataset
def build_sunburst_plot(
    data: pd.DataFrame,
    path_cols: List[str],
    value_col: str,
):
    agg_df = data.groupby(path_cols).agg({value_col: 'count'}).reset_index()
    prefix = 'Sunburst Targets to Path: '
    separator = '-'
    plot_title = "".join([prefix, separator.join(path_cols)])
    fig = px.sunburst(agg_df,
                  path=path_cols,
                  values='target',
                  branchvalues='total', # other value: 'remainder'
                  height=600,
                  title=plot_title
                  )
    fig.update_layout(
        font_size=12,
        title_font_color="black",
    )
    fig.show()


# In[2]:


# main flow
start_time = dt.datetime.now()
print("Started at ", start_time)


# In[3]:


get_ipython().run_cell_magic('time', '', '# get the training set and labels\ntrain_set_path, test_set_path, sample_subm_path = get_data_file_path(in_kaggle)\n\ndf_train = pd.read_csv(train_set_path)\ndf_test = pd.read_csv(test_set_path)\n\nsubm = pd.read_csv(sample_subm_path)\n')


# # Basic Data Overview

# In[4]:


df_train.info()


# ## Express Analysis Insights
# 
# As we can see, the simple express EDA analysis (https://www.kaggle.com/gvyshnya/generic-express-eda-with-comprehensive-insights) yielded a lot of useful insights out of the box, in less then 20 minutes of the data crunching. Below are the key finding from the charts generated by *AutoViz* on a generic basis (see https://www.kaggle.com/gvyshnya/generic-express-eda-with-comprehensive-insights for more details).
# 
# ### Feature-to-Target Relations
# 
# We find that the training set data manifests the following relations between the *target* and feature variables
# 
# - It seems like the training set observations with *target* < 3.5 or/and cont5 < 0.1 could be clearly attributed to as outliers
# - Target variable is a little skewed to the right
# - There is no any numeric feature that is highly correlated with *target*
# - *target* distribution by the labels of the respective cat variables demonstrated that there is a relatively huge association of the *target* with *cat2, cat5, cat6, cat7, cat9*
# - The best association between the *target* values and the cat labels is demonstrated by cat7
# - In turn, there is a weaker association of the target variable with the rest of categorical features
# 
# 
# ### Numeric Feature Findings
# 
# It is demonstrated that
# 
# - There is a clear separation of the observations in the training and test sets into well-contained and well separable clusters by the values of *cont1* (6-8 clusters observed, subject to further clustering experiments)
# - Distribution of the continual variables is identic on both the training and testing sets (the details for each variables are provided below)
# - *cont3, cont4, cont5, cont6*, and *cont12* are highly skewed to the left 
# - *cont8, cont9, cont10, cont11*, and *cont13* have a polynomial distribution (binomial distribution, presumably)
# - *cont1* and *cont11* are skewed to the right
# - *cont0, cont2* have almost normal distribution
# - as per the review of the respective violin plots on the training set, it could be possible to use the extreme tail values of *cont0, cont5, cont6*, and *cont12* for the outlier removal when training the ML models on the training set
# - there are several quite highly correlated numeric feature pairs detected on the training and test sets (with the Pierson’s correlation coefficient >= 0.6): *cat5-cat8, cat5-cat9*, and *cat5-cat12* (among them, *cat5* has the highest absolute correlation with the target variable on the training set)
# 
# 
# ### Categorical Feature Findings
# 
# It has been detected that
# 
# - *cat0* is a two-label categorical variable, and it is unbalance by the label value distribution on the training set (‘A’ drastically predominates ‘B’)
# - *cat1* is a two-label categorical variable, and it is unbalanced a little (‘A’ vs. ‘B’)
# - *cat2* is a two-label categorical variable, and it is unbalance by the label value distribution on the training set (‘A’ drastically predominates ‘B’)
# - *cat3* is a four-label categorical variable, and two of its labels (‘C’, ‘A’) predominate the rest of the labels (the latter ones can be binned into a single category label ‘Other’, to reduce the dimensionality of the respective feature space)
# - *cat4* is a four-label categorical variable, and one of its labels (‘A’) predominates others (such labels can be binned into a single category label ‘Other’, to reduce the dimensionality of the respective feature space)
# - *cat5* is a four-label categorical variable, and two of its labels (‘B’, ‘D’) predominate the rest of the labels (the latter ones can be binned into a single category label ‘Other’, cont reduce the dimensionality of the respective feature space)
# - *cat6* is an eight-label categorical variable, and one of its labels (‘E’) predominates others (such labels can be binned into a single category label ‘Other’, to reduce the
# - *cat8* is a 7-label categorical variable, and 4 of its categories (‘C’, ‘E’, ‘G’, and ‘A’) predominate the rest of the categories on the training set (the latter ones can be binned into a single category label ‘Other’, to reduce the dimensionality of the respective feature space)
# - *cat9* is a 15-label categorical variable, and 3 of its labels (‘F’, ‘I’, and ‘L’) predominate others (the latter ones can be binned into a single category label ‘Other’, to reduce the dimensionality of the respective feature space)
# 
# ### Categorical-to-Numerical Feature Associations
# 
# There are quite strong associations found between the following categorical and numerical features on the training set
# 
# - cont1 by cat3
# - cont5 by cat3
# - cont6 by cat3
# - cont9 by cat3
# - cont10 by cat3
# - cont11 by cat3
# - cont12 by cat3
# - cont0 by cat4
# - cont5 by cat4
# - cont6 by cat4
# - cont8 by cat4
# - cont9 by cat4
# - cont10 by cat4
# - cont11 by cat4
# - cont12 by cat4
# - cont13 by cat4
# - all continual variables by cat5
# - all continual variables by cat6
# - all continual variables by cat7
# - all continual variables by cat8
# - cont0 by cat9
# - cont1 by cat9
# - cont2 by cat9
# - cont5 by cat9
# - cont6 by cat9
# - cont8 by cat9
# - cont9 by cat9
# - cont10 by cat9
# - cont11 by cat9
# - cont12 by cat9
# - cont13 by cat9
# - cont1 by cat1
# 
# The above-mentioned continual-to-categorical feature associations are also confirmed on the test set

# # Roadmap For Additional EDA Visualizations
# 
# The good insights we quickly got from the express EDA Analysis with *AutoViz* above were very helpful per se. However, they did not address all and every analytical issues we would like to address, when tackling the fundumantal question of what the impact of features on the *target* are.
# 
# Now we are going to undertake the additional manual EDA discoveries to review
# 
# - pair associations between the selective cat variables
# - multi-variative associations between selective cat variables, factored by the impact of such association on the conditional distributions of the *target* and numeric features on the training set
# - initial analysis on the optimal feature engineering steps to take in the ML experiments phase down the road
# 
# While doing it, we will be paying the most attention to the cat features highlighted in the express EDA analysis above. These are
# 
# - *cat2, cat5, cat6, cat7, and cat9* that have a good association with *target* on the training set
# - *cat8* that has good association with every feature variable both in the training and test sets

# # Multi-Variative Analysis of Cat Feature Associations
# 
# We are now going to investigate multi-cat relations among the features in the training set

# ## Multi-Variative Associations Between 'cat2', 'cat5', and 'cat6'

# In[5]:


path_cols = ['cat2', 'cat5', 'cat6']


# In[6]:


# build a sunburst plot with the record count per cat-to-cat buckets
build_sunburst_plot(
    data=df_train,
    path_cols=path_cols,
    value_col='target',
)


# We can see that the most of the records in the training set are withing the clusters in *'cat2'-'cat5'-'cat6'* as follows
# 
# - 'cat2'='A'-'cat5'='B'-'cat6'='A'
# - 'cat2'='A'-'cat5'='D'-'cat6'='A'

# In[7]:


agg_df = df_train.groupby(path_cols).agg({'target': ['mean', 'count']}).reset_index()
agg_df.columns = [path_cols[0], path_cols[1], path_cols[2], 'target_mean', 'target_count'] 


# In[8]:


build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col="target_count",
    color_col="target_mean",
    mid_point=7.0,
)


# We see that the individual strong associations of *target* with *'cat2','cat5', and 'cat6'* are well displayed in the multi-variate distribution of *target* of the subsets within the *'cat2'-'cat5'-'cat6'* dimensions. For instance,
# 
# - the subset of 'A-C-A' demonstrated the smallest mean *target* value within the training set (7.1399)
# - the subsets of 'A-B-C' and 'A-D-C' demonstrated the biggest mean *target* value within the training set (7.83+)
# - the rest of the subsets are also quite distinctive in terms of the mean *target* value for each of them
# 
# **Statistical Variability of Mean Numeric Feature Values Within 'cat2'-'cat5'-'cat6' Feature Space
# 
# Now we are going to see the variability of mean values for every numeric feature by training set clusters, when sliced by category labels within 'cat2'-'cat5'-'cat6' feature space

# In[9]:


agg_df = df_train.groupby(path_cols).agg({'cont0': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont0',
    mid_point=0.5,
)


# In[10]:


agg_df = df_train.groupby(path_cols).agg({'cont1': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont1',
    mid_point=0.5,
)


# In[11]:


agg_df = df_train.groupby(path_cols).agg({'cont2': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont2',
    mid_point=0.5,
)


# In[12]:


agg_df = df_train.groupby(path_cols).agg({'cont3': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont3',
    mid_point=0.5,
)


# In[13]:


agg_df = df_train.groupby(path_cols).agg({'cont4': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont4',
    mid_point=0.5,
)


# In[14]:


agg_df = df_train.groupby(path_cols).agg({'cont5': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont5',
    mid_point=0.5,
)


# In[15]:


agg_df = df_train.groupby(path_cols).agg({'cont6': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont6',
    mid_point=0.5,
)


# In[16]:


agg_df = df_train.groupby(path_cols).agg({'cont7': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont7',
    mid_point=0.5,
)


# In[17]:


agg_df = df_train.groupby(path_cols).agg({'cont8': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont8',
    mid_point=0.5,
)


# In[18]:


agg_df = df_train.groupby(path_cols).agg({'cont9': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont9',
    mid_point=0.5,
)


# In[19]:


agg_df = df_train.groupby(path_cols).agg({'cont9': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont9',
    mid_point=0.5,
)


# In[20]:


agg_df = df_train.groupby(path_cols).agg({'cont10': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont10',
    mid_point=0.5,
)


# In[21]:


agg_df = df_train.groupby(path_cols).agg({'cont11': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont11',
    mid_point=0.5,
)


# In[22]:


agg_df = df_train.groupby(path_cols).agg({'cont12': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont12',
    mid_point=0.5,
)


# In[23]:


agg_df = df_train.groupby(path_cols).agg({'cont13': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont13',
    mid_point=0.5,
)


# We find that
# 
# - there are statistically meaningful variations in the distribution of every numerical feature on the subsets of the training set within 'cat2'-'cat5'-'cat6' space
# - it is therefore not productive to bin the rare classes of 'cat5' and 'cat6' into the single categories as the significant intelligence could be lost
# - instead of it, we  can try to generate a number of new numeric features (means of each of the raw features, grouped by 'cat2', 'cat5', and 'cat6')

# ## Multi-Variative Associations Between 'cat2', 'cat5', and 'cat7'

# In[24]:


path_cols = ['cat2', 'cat5', 'cat7']


# In[25]:


# build a sunburst plot with the record count per cat-to-cat buckets
build_sunburst_plot(
    data=df_train,
    path_cols=path_cols,
    value_col='target',
)


# We can see that the largest clusters of the observations in the training set (within the space of *'cat2'-'cat5'-'cat7'*) are as follows
# 
# - 'cat2'=A -'cat5'=B - 'cat7'=E
# - 'cat2'=A -'cat5'=D - 'cat7'=E

# In[26]:


agg_df = df_train.groupby(path_cols).agg({'target': ['mean', 'count']}).reset_index()
agg_df.columns = [path_cols[0], path_cols[1], path_cols[2], 'target_mean', 'target_count'] 

build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col="target_count",
    color_col="target_mean",
    mid_point=7.0,
)


# We see that the individual strong associations of *target* with *'cat2','cat5', and 'cat7'* are well displayed in the multi-variate distribution of *target* of the subsets within the *'cat2'-'cat5'-'cat7'* dimensions. For instance,
# 
# - the subset of 'A-C' demonstrated the smallest mean *target* value within the training set (7.14+)
# - the subsets of 'B-D' and 'B-B' demonstrated the biggest mean *target* value within the training set (7.76+)
# - the rest of the subsets are also quite distinctive in terms of the mean *target* value for each of them
# 
# **Statistical Variability of Mean Numeric Feature Values Within 'cat2'-'cat5'-'cat7' Feature Space
# 
# Now we are going to see the variability of mean values for every numeric feature by training set clusters, when sliced by category labels within 'cat2'-'cat5'-'cat7' feature space

# In[27]:


agg_df = df_train.groupby(path_cols).agg({'cont0': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont0',
    mid_point=0.5,
)


# In[28]:


agg_df = df_train.groupby(path_cols).agg({'cont0': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont0',
    mid_point=0.5,
)


# In[29]:


agg_df = df_train.groupby(path_cols).agg({'cont1': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont1',
    mid_point=0.5,
)


# In[30]:


agg_df = df_train.groupby(path_cols).agg({'cont2': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont2',
    mid_point=0.5,
)


# In[31]:


agg_df = df_train.groupby(path_cols).agg({'cont3': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont3',
    mid_point=0.5,
)


# In[32]:


agg_df = df_train.groupby(path_cols).agg({'cont4': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont4',
    mid_point=0.5,
)


# In[33]:


agg_df = df_train.groupby(path_cols).agg({'cont5': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont5',
    mid_point=0.5,
)


# In[34]:


agg_df = df_train.groupby(path_cols).agg({'cont6': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont6',
    mid_point=0.5,
)


# In[35]:


agg_df = df_train.groupby(path_cols).agg({'cont7': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont7',
    mid_point=0.5,
)


# In[36]:


agg_df = df_train.groupby(path_cols).agg({'cont8': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont8',
    mid_point=0.5,
)


# In[37]:


agg_df = df_train.groupby(path_cols).agg({'cont9': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont9',
    mid_point=0.5,
)


# In[38]:


agg_df = df_train.groupby(path_cols).agg({'cont10': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont10',
    mid_point=0.5,
)


# In[39]:


agg_df = df_train.groupby(path_cols).agg({'cont11': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont11',
    mid_point=0.5,
)


# In[40]:


agg_df = df_train.groupby(path_cols).agg({'cont12': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont12',
    mid_point=0.5,
)


# We find that
# 
# - there are statistically meaningful variations in the distribution of every numerical feature on the subsets of the training set within 'cat2'-'cat5'-'cat7' space
# - it is therefore not productive to bin the rare classes of 'cat5', and 'cat7' into the single categories as the significant intelligence could be lost
# - instead of it, we  can try to generate a number of new numeric features (means of each of the raw features, grouped by 'cat2', 'cat5', and 'cat7')

# ## Multi-Variative Associations Between 'cat2', 'cat5', and 'cat8'

# In[41]:


path_cols = ['cat2', 'cat5', 'cat8']


# In[42]:


# build a sunburst plot with the record count per cat-to-cat buckets
build_sunburst_plot(
    data=df_train,
    path_cols=path_cols,
    value_col='target',
)


# We can see that the largest clusters of the observations in the training set (within the space of *'cat2'-'cat5'-'cat8'*) are as follows
# 
# - 'cat2'=A -'cat5'=B - 'cat8'=C
# - 'cat2'=A -'cat5'=B - 'cat8'=E
# - 'cat2'=A -'cat5'=D - 'cat8'=C

# In[43]:


agg_df = df_train.groupby(path_cols).agg({'target': ['mean', 'count']}).reset_index()
agg_df.columns = [path_cols[0], path_cols[1], path_cols[2], 'target_mean', 'target_count'] 

build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col="target_count",
    color_col="target_mean",
    mid_point=7.0,
)


# We see that the individual strong associations of *target* with *'cat2','cat5', and 'cat8'* are well displayed in the multi-variate distribution of *target* of the subsets within the *'cat2'-'cat5'-'cat8'* dimensions. For instance,
# 
# - the subset of 'A-C-G' demonstrated the smallest mean *target* value within the training set (7.07+)
# - the subsets of 'B-D-E' demonstrated the biggest mean *target* value within the training set (7.87+)
# - the rest of the subsets are also quite distinctive in terms of the mean *target* value for each of them
# 
# **Statistical Variability of Mean Numeric Feature Values Within 'cat2'-'cat5'-'cat8' Feature Space
# 
# Now we are going to see the variability of mean values for every numeric feature by training set clusters, when sliced by category labels within 'cat2'-'cat5'-'cat8' feature space

# In[44]:


agg_df = df_train.groupby(path_cols).agg({'cont0': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont0',
    mid_point=0.5,
)


# In[45]:


agg_df = df_train.groupby(path_cols).agg({'cont1': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont1',
    mid_point=0.5,
)


# In[46]:


agg_df = df_train.groupby(path_cols).agg({'cont2': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont2',
    mid_point=0.5,
)


# In[47]:


agg_df = df_train.groupby(path_cols).agg({'cont3': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont3',
    mid_point=0.5,
)


# In[48]:


agg_df = df_train.groupby(path_cols).agg({'cont4': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont4',
    mid_point=0.5,
)


# In[49]:


agg_df = df_train.groupby(path_cols).agg({'cont5': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont5',
    mid_point=0.5,
)


# In[50]:


agg_df = df_train.groupby(path_cols).agg({'cont6': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont6',
    mid_point=0.5,
)


# In[51]:


agg_df = df_train.groupby(path_cols).agg({'cont7': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont7',
    mid_point=0.5,
)


# In[52]:


agg_df = df_train.groupby(path_cols).agg({'cont8': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont8',
    mid_point=0.5,
)


# In[53]:


agg_df = df_train.groupby(path_cols).agg({'cont9': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont9',
    mid_point=0.5,
)


# In[54]:


agg_df = df_train.groupby(path_cols).agg({'cont10': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont10',
    mid_point=0.5,
)


# In[55]:


agg_df = df_train.groupby(path_cols).agg({'cont11': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont11',
    mid_point=0.5,
)


# In[56]:


agg_df = df_train.groupby(path_cols).agg({'cont12': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont12',
    mid_point=0.5,
)


# In[57]:


agg_df = df_train.groupby(path_cols).agg({'cont13': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont13',
    mid_point=0.5,
)


# We find that
# 
# - there are statistically meaningful variations in the distribution of every numerical feature on the subsets of the training set within 'cat2'-'cat5'-'cat8' space
# - it is therefore not productive to bin the rare classes of 'cat5', and 'cat8' into the single categories as the significant intelligence could be lost
# - instead of it, we  can try to generate a number of new numeric features (means of each of the raw features, grouped by 'cat2', 'cat5', and 'cat8')

# ## Multi-Variative Associations Between 'cat2', 'cat5', and 'cat9'

# In[58]:


path_cols = ['cat2', 'cat5', 'cat9']


# In[59]:


# build a sunburst plot with the record count per cat-to-cat buckets
build_sunburst_plot(
    data=df_train,
    path_cols=path_cols,
    value_col='target',
)


# We can see that the largest clusters of the observations in the training set (within the space of *'cat2'-'cat5'-'cat9'*) are as follows
# 
# - 'cat2'=A -'cat5'=B - 'cat9'=F
# - 'cat2'=A -'cat5'=B - 'cat9'=F

# In[60]:


agg_df = df_train.groupby(path_cols).agg({'target': ['mean', 'count']}).reset_index()
agg_df.columns = [path_cols[0], path_cols[1], path_cols[2], 'target_mean', 'target_count'] 

build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col="target_count",
    color_col="target_mean",
    mid_point=7.0,
)


# We see that the individual strong associations of *target* with *'cat2','cat5', and 'cat9'* are well displayed in the multi-variate distribution of *target* of the subsets within the *'cat2'-'cat5'-'cat9'* dimensions. For instance,
# 
# - the subset of 'A-C-A' demonstrated the smallest mean *target* value within the training set (6.97+)
# - the subsets of 'B-D-L' demonstrated the biggest mean *target* value within the training set (7.94+)
# - the rest of the subsets are also quite distinctive in terms of the mean *target* value for each of them
# 
# **Statistical Variability of Mean Numeric Feature Values Within 'cat2'-'cat5'-'cat9' Feature Space
# 
# Now we are going to see the variability of mean values for every numeric feature by training set clusters, when sliced by category labels within 'cat2'-'cat5'-'cat9' feature space

# In[61]:


agg_df = df_train.groupby(path_cols).agg({'cont0': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont0',
    mid_point=0.5,
)


# In[62]:


agg_df = df_train.groupby(path_cols).agg({'cont1': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont1',
    mid_point=0.5,
)


# In[63]:


agg_df = df_train.groupby(path_cols).agg({'cont2': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont2',
    mid_point=0.5,
)


# In[64]:


agg_df = df_train.groupby(path_cols).agg({'cont3': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont3',
    mid_point=0.5,
)


# In[65]:


agg_df = df_train.groupby(path_cols).agg({'cont4': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont4',
    mid_point=0.5,
)


# In[66]:


agg_df = df_train.groupby(path_cols).agg({'cont5': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont5',
    mid_point=0.5,
)


# In[67]:


agg_df = df_train.groupby(path_cols).agg({'cont6': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont6',
    mid_point=0.5,
)


# In[68]:


agg_df = df_train.groupby(path_cols).agg({'cont7': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont7',
    mid_point=0.5,
)


# In[69]:


agg_df = df_train.groupby(path_cols).agg({'cont8': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont8',
    mid_point=0.5,
)


# In[70]:


agg_df = df_train.groupby(path_cols).agg({'cont9': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont9',
    mid_point=0.5,
)


# In[71]:


agg_df = df_train.groupby(path_cols).agg({'cont10': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont10',
    mid_point=0.5,
)


# In[72]:


agg_df = df_train.groupby(path_cols).agg({'cont11': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont11',
    mid_point=0.5,
)


# In[73]:


agg_df = df_train.groupby(path_cols).agg({'cont12': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont12',
    mid_point=0.5,
)


# In[74]:


agg_df = df_train.groupby(path_cols).agg({'cont13': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont13',
    mid_point=0.5,
)


# We find that
# 
# - there are statistically meaningful variations in the distribution of every numerical feature on the subsets of the training set within 'cat2'-'cat5'-'cat9' space
# - it is therefore not productive to bin the rare classes of 'cat5', and 'cat9' into the single categories as the significant intelligence could be lost
# - instead of it, we  can try to generate a number of new numeric features (means of each of the raw features, grouped by 'cat2', 'cat5', and 'cat9')

# ## Multi-Variative Associations Between 'cat5', 'cat6', and 'cat7'

# In[75]:


path_cols = ['cat5', 'cat6', 'cat7']


# In[76]:


# build a sunburst plot with the record count per cat-to-cat buckets
build_sunburst_plot(
    data=df_train,
    path_cols=path_cols,
    value_col='target',
)


# We can see that the largest clusters of the observations in the training set (within the space of *'cat5'-'cat6'-'cat7'*) are as follows
# 
# - 'cat5'=B - 'cat6'=A - 'cat7'=E
# - 'cat5'=D - 'cat6'=A - 'cat7'=E

# In[77]:


agg_df = df_train.groupby(path_cols).agg({'target': ['mean', 'count']}).reset_index()
agg_df.columns = [path_cols[0], path_cols[1], path_cols[2], 'target_mean', 'target_count'] 

build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col="target_count",
    color_col="target_mean",
    mid_point=7.0,
)


# We see that the individual strong associations of *target* with *'cat5','cat6', and 'cat7'* are well displayed in the multi-variate distribution of *target* of the subsets within the *'cat5'-'cat6'-'cat7'* dimensions. For instance,
# 
# - the subset of 'C-A-E' demonstrated the smallest mean *target* value within the training set (7.14+)
# - the subsets of 'D-C-D' demonstrated the biggest mean *target* value within the training set (8.02+)
# - the rest of the subsets are also quite distinctive in terms of the mean *target* value for each of them
# 
# **Statistical Variability of Mean Numeric Feature Values Within 'cat5'-'cat6'-'cat7' Feature Space
# 
# Now we are going to see the variability of mean values for every numeric feature by training set clusters, when sliced by category labels within 'cat5'-'cat6'-'cat7' feature space

# In[78]:


agg_df = df_train.groupby(path_cols).agg({'cont0': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont0',
    mid_point=0.5,
)


# In[79]:


agg_df = df_train.groupby(path_cols).agg({'cont1': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont1',
    mid_point=0.5,
)


# In[80]:


agg_df = df_train.groupby(path_cols).agg({'cont2': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont2',
    mid_point=0.5,
)


# In[81]:


agg_df = df_train.groupby(path_cols).agg({'cont3': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont3',
    mid_point=0.5,
)


# In[82]:


agg_df = df_train.groupby(path_cols).agg({'cont4': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont4',
    mid_point=0.5,
)


# In[83]:


agg_df = df_train.groupby(path_cols).agg({'cont5': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont5',
    mid_point=0.5,
)


# In[84]:


agg_df = df_train.groupby(path_cols).agg({'cont6': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont6',
    mid_point=0.5,
)


# In[85]:


agg_df = df_train.groupby(path_cols).agg({'cont7': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont7',
    mid_point=0.5,
)


# In[86]:


agg_df = df_train.groupby(path_cols).agg({'cont8': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont8',
    mid_point=0.5,
)


# In[87]:


agg_df = df_train.groupby(path_cols).agg({'cont9': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont9',
    mid_point=0.5,
)


# In[88]:


agg_df = df_train.groupby(path_cols).agg({'cont10': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont10',
    mid_point=0.5,
)


# In[89]:


agg_df = df_train.groupby(path_cols).agg({'cont11': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont11',
    mid_point=0.5,
)


# In[90]:


agg_df = df_train.groupby(path_cols).agg({'cont12': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont12',
    mid_point=0.5,
)


# In[91]:


agg_df = df_train.groupby(path_cols).agg({'cont13': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont13',
    mid_point=0.5,
)


# We find that
# 
# - there are statistically meaningful variations in the distribution of every numerical feature on the subsets of the training set within 'cat5'-'cat6'-'cat7' space
# - it is therefore not productive to bin the rare classes of 'cat5', 'cat6', and 'cat7' into the single categories as the significant intelligence could be lost
# - instead of it, we  can try to generate a number of new numeric features (means of each of the raw features, grouped by 'cat5', 'cat6', and 'cat7')

# ## Multi-Variative Associations Between 'cat5', 'cat6', and 'cat8'

# In[92]:


path_cols = ['cat5', 'cat6', 'cat8']


# In[93]:


# build a sunburst plot with the record count per cat-to-cat buckets
build_sunburst_plot(
    data=df_train,
    path_cols=path_cols,
    value_col='target',
)


# We can see that the largest clusters of the observations in the training set (within the space of *'cat5'-'cat6'-'cat8'*) are as follows
# 
# - 'cat5'=B - 'cat6'=A - 'cat8'=C
# - 'cat5'=B - 'cat6'=A - 'cat8'=E
# - 'cat5'=D - 'cat6'=A - 'cat8'=C
# - 'cat5'=D - 'cat6'=A - 'cat8'=E

# In[94]:


agg_df = df_train.groupby(path_cols).agg({'target': ['mean', 'count']}).reset_index()
agg_df.columns = [path_cols[0], path_cols[1], path_cols[2], 'target_mean', 'target_count'] 

build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col="target_count",
    color_col="target_mean",
    mid_point=7.0,
)


# We see that the individual strong associations of *target* with *'cat5','cat6', and 'cat8'* are well displayed in the multi-variate distribution of *target* of the subsets within the *'cat5'-'cat6'-'cat8'* dimensions. For instance,
# 
# - the subset of 'C-A-G' demonstrated the smallest mean *target* value within the training set (7.08+)
# - the subsets of 'D-C-C' demonstrated the biggest mean *target* value within the training set (8.02+)
# - the rest of the subsets are also quite distinctive in terms of the mean *target* value for each of them
# 
# **Statistical Variability of Mean Numeric Feature Values Within 'cat5'-'cat6'-'cat8' Feature Space
# 
# Now we are going to see the variability of mean values for every numeric feature by training set clusters, when sliced by category labels within 'cat5'-'cat6'-'cat8' feature space

# In[95]:


agg_df = df_train.groupby(path_cols).agg({'cont0': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont0',
    mid_point=0.5,
)


# In[96]:


agg_df = df_train.groupby(path_cols).agg({'cont1': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont1',
    mid_point=0.5,
)


# In[97]:


agg_df = df_train.groupby(path_cols).agg({'cont2': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont2',
    mid_point=0.5,
)


# In[98]:


agg_df = df_train.groupby(path_cols).agg({'cont3': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont3',
    mid_point=0.5,
)


# In[99]:


agg_df = df_train.groupby(path_cols).agg({'cont4': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont4',
    mid_point=0.5,
)


# In[100]:


agg_df = df_train.groupby(path_cols).agg({'cont5': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont5',
    mid_point=0.5,
)


# In[101]:


agg_df = df_train.groupby(path_cols).agg({'cont6': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont6',
    mid_point=0.5,
)


# In[102]:


agg_df = df_train.groupby(path_cols).agg({'cont7': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont7',
    mid_point=0.5,
)


# In[103]:


agg_df = df_train.groupby(path_cols).agg({'cont8': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont8',
    mid_point=0.5,
)


# In[104]:


agg_df = df_train.groupby(path_cols).agg({'cont9': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont9',
    mid_point=0.5,
)


# In[105]:


agg_df = df_train.groupby(path_cols).agg({'cont10': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont10',
    mid_point=0.5,
)


# In[106]:


agg_df = df_train.groupby(path_cols).agg({'cont11': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont11',
    mid_point=0.5,
)


# In[107]:


agg_df = df_train.groupby(path_cols).agg({'cont12': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont12',
    mid_point=0.5,
)


# In[108]:


agg_df = df_train.groupby(path_cols).agg({'cont13': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont13',
    mid_point=0.5,
)


# We find that
# 
# - there are statistically meaningful variations in the distribution of every numerical feature on the subsets of the training set within 'cat5'-'cat6'-'cat8' space
# - it is therefore not productive to bin the rare classes of 'cat5', 'cat6', and 'cat8' into the single categories as the significant intelligence could be lost
# - instead of it, we  can try to generate a number of new numeric features (means of each of the raw features, grouped by 'cat5', 'cat6', and 'cat8')

# ## Multi-Variative Associations Between 'cat5', 'cat6', and 'cat9'

# In[109]:


path_cols = ['cat5', 'cat6', 'cat9']


# In[110]:


# build a sunburst plot with the record count per cat-to-cat buckets
build_sunburst_plot(
    data=df_train,
    path_cols=path_cols,
    value_col='target',
)


# We can see that the largest clusters of the observations in the training set (within the space of *'cat5'-'cat6'-'cat9'*) are as follows
# 
# - 'cat5'=B - 'cat6'=A - 'cat9'=F
# - 'cat5'=D - 'cat6'=A - 'cat9'=F

# In[111]:


agg_df = df_train.groupby(path_cols).agg({'target': ['mean', 'count']}).reset_index()
agg_df.columns = [path_cols[0], path_cols[1], path_cols[2], 'target_mean', 'target_count'] 

build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col="target_count",
    color_col="target_mean",
    mid_point=7.0,
)


# We see that the individual strong associations of *target* with *'cat5','cat6', and 'cat9'* are well displayed in the multi-variate distribution of *target* of the subsets within the *'cat5'-'cat6'-'cat9'* dimensions. For instance,
# 
# - the subset of 'C-A-I' demonstrated the smallest mean *target* value within the training set (7.00+)
# - the subsets of 'D-C-C' demonstrated the biggest mean *target* value within the training set (8.02+)
# - the rest of the subsets are also quite distinctive in terms of the mean *target* value for each of them
# 
# **Statistical Variability of Mean Numeric Feature Values Within 'cat5'-'cat6'-'cat9' Feature Space
# 
# Now we are going to see the variability of mean values for every numeric feature by training set clusters, when sliced by category labels within 'cat5'-'cat6'-'cat9' feature space

# In[112]:


agg_df = df_train.groupby(path_cols).agg({'cont0': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont0',
    mid_point=0.5,
)


# In[113]:


agg_df = df_train.groupby(path_cols).agg({'cont1': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont1',
    mid_point=0.5,
)


# In[114]:


agg_df = df_train.groupby(path_cols).agg({'cont2': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont2',
    mid_point=0.5,
)


# In[115]:


agg_df = df_train.groupby(path_cols).agg({'cont3': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont3',
    mid_point=0.5,
)


# In[116]:


agg_df = df_train.groupby(path_cols).agg({'cont4': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont4',
    mid_point=0.5,
)


# In[117]:


agg_df = df_train.groupby(path_cols).agg({'cont5': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont5',
    mid_point=0.5,
)


# In[118]:


agg_df = df_train.groupby(path_cols).agg({'cont6': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont6',
    mid_point=0.5,
)


# In[119]:


agg_df = df_train.groupby(path_cols).agg({'cont7': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont7',
    mid_point=0.5,
)


# In[120]:


agg_df = df_train.groupby(path_cols).agg({'cont8': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont8',
    mid_point=0.5,
)


# In[121]:


agg_df = df_train.groupby(path_cols).agg({'cont9': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont9',
    mid_point=0.5,
)


# In[122]:


agg_df = df_train.groupby(path_cols).agg({'cont10': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont10',
    mid_point=0.5,
)


# In[123]:


agg_df = df_train.groupby(path_cols).agg({'cont11': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont11',
    mid_point=0.5,
)


# In[124]:


agg_df = df_train.groupby(path_cols).agg({'cont12': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont12',
    mid_point=0.5,
)


# In[125]:


agg_df = df_train.groupby(path_cols).agg({'cont13': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont13',
    mid_point=0.5,
)


# We find that
# 
# - there are statistically meaningful variations in the distribution of every numerical feature on the subsets of the training set within 'cat5'-'cat6'-'cat9' space
# - it is therefore not productive to bin the rare classes of 'cat5', 'cat6', and 'cat9' into the single categories as the significant intelligence could be lost
# - instead of it, we  can try to generate a number of new numeric features (means of each of the raw features, grouped by 'cat5', 'cat6', and 'cat9')

# ## Multi-Variative Associations Between 'cat6', 'cat7', and 'cat8'

# In[126]:


path_cols = ['cat6', 'cat7', 'cat8']


# In[127]:


# build a sunburst plot with the record count per cat-to-cat buckets
build_sunburst_plot(
    data=df_train,
    path_cols=path_cols,
    value_col='target',
)


# We can see that the largest clusters of the observations in the training set (within the space of *'cat6'-'cat7'-'cat8'*) are as follows
# 
# - 'cat6'=A - 'cat7'=E - 'cat8'=E
# - 'cat6'=A - 'cat7'=E - 'cat8'=C

# In[128]:


agg_df = df_train.groupby(path_cols).agg({'target': ['mean', 'count']}).reset_index()
agg_df.columns = [path_cols[0], path_cols[1], path_cols[2], 'target_mean', 'target_count'] 

build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col="target_count",
    color_col="target_mean",
    mid_point=7.0,
)


# We see that the individual strong associations of *target* with *'cat6','cat7', and 'cat8'* are well displayed in the multi-variate distribution of *target* of the subsets within the *'cat6'-'cat7'-'cat8'* dimensions. For instance,
# 
# - the subset of 'A-E-G' demonstrated the smallest mean *target* value within the training set (7.30+)
# - the subsets of 'C' demonstrated the biggest mean *target* value within the training set (7.96+)
# - the rest of the subsets are also quite distinctive in terms of the mean *target* value for each of them
# 
# **Statistical Variability of Mean Numeric Feature Values Within 'cat6'-'cat7'-'cat8' Feature Space
# 
# Now we are going to see the variability of mean values for every numeric feature by training set clusters, when sliced by category labels within 'cat6'-'cat7'-'cat8' feature space

# In[129]:


agg_df = df_train.groupby(path_cols).agg({'cont0': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont0',
    mid_point=0.5,
)


# In[130]:


agg_df = df_train.groupby(path_cols).agg({'cont1': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont1',
    mid_point=0.5,
)


# In[131]:


agg_df = df_train.groupby(path_cols).agg({'cont2': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont2',
    mid_point=0.5,
)


# In[132]:


agg_df = df_train.groupby(path_cols).agg({'cont3': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont3',
    mid_point=0.5,
)


# In[133]:


agg_df = df_train.groupby(path_cols).agg({'cont4': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont4',
    mid_point=0.5,
)


# In[134]:


agg_df = df_train.groupby(path_cols).agg({'cont5': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont5',
    mid_point=0.5,
)


# In[135]:


agg_df = df_train.groupby(path_cols).agg({'cont6': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont6',
    mid_point=0.5,
)


# In[136]:


agg_df = df_train.groupby(path_cols).agg({'cont7': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont7',
    mid_point=0.5,
)


# In[137]:


agg_df = df_train.groupby(path_cols).agg({'cont8': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont8',
    mid_point=0.5,
)


# In[138]:


agg_df = df_train.groupby(path_cols).agg({'cont9': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont9',
    mid_point=0.5,
)


# In[139]:


agg_df = df_train.groupby(path_cols).agg({'cont10': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont10',
    mid_point=0.5,
)


# In[140]:


agg_df = df_train.groupby(path_cols).agg({'cont11': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont11',
    mid_point=0.5,
)


# In[141]:


agg_df = df_train.groupby(path_cols).agg({'cont12': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont12',
    mid_point=0.5,
)


# In[142]:


agg_df = df_train.groupby(path_cols).agg({'cont13': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont13',
    mid_point=0.5,
)


# We find that
# 
# - there are statistically meaningful variations in the distribution of every numerical feature on the subsets of the training set within 'cat6'-'cat7'-'cat8' space
# - it is therefore not productive to bin the rare classes of 'cat7' and 'cat8' into the single categories as the significant intelligence could be lost
# - instead of it, we  can try to generate a number of new numeric features (means of each of the raw features, grouped by 'cat6', 'cat7', and 'cat8')

# ## Multi-Variative Associations Between 'cat6', 'cat7', and 'cat9'

# In[143]:


path_cols = ['cat6', 'cat7', 'cat9']


# In[144]:


# build a sunburst plot with the record count per cat-to-cat buckets
build_sunburst_plot(
    data=df_train,
    path_cols=path_cols,
    value_col='target',
)


# We can see that the largest clusters of the observations in the training set (within the space of *'cat6'-'cat7'-'cat9'*) are as follows
# 
# - 'cat6'=A - 'cat7'=E - 'cat9'=F
# - 'cat6'=A - 'cat7'=E - 'cat9'=I
# - 'cat6'=A - 'cat7'=E - 'cat9'=L

# In[145]:


agg_df = df_train.groupby(path_cols).agg({'target': ['mean', 'count']}).reset_index()
agg_df.columns = [path_cols[0], path_cols[1], path_cols[2], 'target_mean', 'target_count'] 

build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col="target_count",
    color_col="target_mean",
    mid_point=7.0,
)


# In[146]:


agg_df = df_train.groupby(path_cols).agg({'cont0': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont0',
    mid_point=0.5,
)


# In[147]:


agg_df = df_train.groupby(path_cols).agg({'cont1': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont1',
    mid_point=0.5,
)


# In[148]:


agg_df = df_train.groupby(path_cols).agg({'cont2': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont2',
    mid_point=0.5,
)


# In[149]:


agg_df = df_train.groupby(path_cols).agg({'cont3': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont3',
    mid_point=0.5,
)


# In[150]:


agg_df = df_train.groupby(path_cols).agg({'cont4': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont4',
    mid_point=0.5,
)


# In[151]:


agg_df = df_train.groupby(path_cols).agg({'cont5': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont5',
    mid_point=0.5,
)


# In[152]:


agg_df = df_train.groupby(path_cols).agg({'cont6': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont6',
    mid_point=0.5,
)


# In[153]:


agg_df = df_train.groupby(path_cols).agg({'cont7': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont7',
    mid_point=0.5,
)


# In[154]:


agg_df = df_train.groupby(path_cols).agg({'cont8': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont8',
    mid_point=0.5,
)


# In[155]:


agg_df = df_train.groupby(path_cols).agg({'cont9': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont9',
    mid_point=0.5,
)


# In[156]:


agg_df = df_train.groupby(path_cols).agg({'cont10': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont10',
    mid_point=0.5,
)


# In[157]:


agg_df = df_train.groupby(path_cols).agg({'cont11': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont11',
    mid_point=0.5,
)


# In[158]:


agg_df = df_train.groupby(path_cols).agg({'cont12': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont12',
    mid_point=0.5,
)


# In[159]:


agg_df = df_train.groupby(path_cols).agg({'cont13': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont13',
    mid_point=0.5,
)


# We find that
# 
# - there are statistically meaningful variations in the distribution of every numerical feature on the subsets of the training set within 'cat6'-'cat7'-'cat9' space
# - it is therefore not productive to bin the rare classes of 'cat7' and 'cat9' into the single categories as the significant intelligence could be lost
# - instead of it, we  can try to generate a number of new numeric features (means of each of the raw features, grouped by 'cat6', 'cat7', and 'cat9')

# ## Multi-Variative Associations Between 'cat7', 'cat8', and 'cat9'

# In[160]:


path_cols = ['cat7', 'cat8', 'cat9']


# In[161]:


# build a sunburst plot with the record count per cat-to-cat buckets
build_sunburst_plot(
    data=df_train,
    path_cols=path_cols,
    value_col='target',
)


# We can see that the largest clusters of the observations in the training set (within the space of *'cat7'-'cat8'-'cat9'*) are as follows
# 
# - 'cat7'=E - 'cat8'=C - 'cat9'=F
# - 'cat7'=E - 'cat8'=E - 'cat9'=F
# 

# In[162]:


agg_df = df_train.groupby(path_cols).agg({'target': ['mean', 'count']}).reset_index()
agg_df.columns = [path_cols[0], path_cols[1], path_cols[2], 'target_mean', 'target_count'] 

build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col="target_count",
    color_col="target_mean",
    mid_point=7.0,
)


# In[163]:


agg_df = df_train.groupby(path_cols).agg({'cont0': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont0',
    mid_point=0.5,
)


# In[164]:


agg_df = df_train.groupby(path_cols).agg({'cont1': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont1',
    mid_point=0.5,
)


# In[165]:


agg_df = df_train.groupby(path_cols).agg({'cont2': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont2',
    mid_point=0.5,
)


# In[166]:


agg_df = df_train.groupby(path_cols).agg({'cont3': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont3',
    mid_point=0.5,
)


# In[167]:


agg_df = df_train.groupby(path_cols).agg({'cont4': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont4',
    mid_point=0.5,
)


# In[168]:


agg_df = df_train.groupby(path_cols).agg({'cont5': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont5',
    mid_point=0.5,
)


# In[169]:


agg_df = df_train.groupby(path_cols).agg({'cont6': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont6',
    mid_point=0.5,
)


# In[170]:


agg_df = df_train.groupby(path_cols).agg({'cont7': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont7',
    mid_point=0.5,
)


# In[171]:


agg_df = df_train.groupby(path_cols).agg({'cont8': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont8',
    mid_point=0.5,
)


# In[172]:


agg_df = df_train.groupby(path_cols).agg({'cont9': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont9',
    mid_point=0.5,
)


# In[173]:


agg_df = df_train.groupby(path_cols).agg({'cont10': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont10',
    mid_point=0.5,
)


# In[174]:


agg_df = df_train.groupby(path_cols).agg({'cont11': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont11',
    mid_point=0.5,
)


# In[175]:


agg_df = df_train.groupby(path_cols).agg({'cont12': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont12',
    mid_point=0.5,
)


# In[176]:


agg_df = df_train.groupby(path_cols).agg({'cont13': 'mean', 'target': 'count'}).reset_index()
build_treemap(
    data=agg_df,
    path_cols=path_cols,
    value_col='target',
    color_col='cont13',
    mid_point=0.5,
)


# We find that
# 
# - there are statistically meaningful variations in the distribution of every numerical feature on the subsets of the training set within 'cat7'-'cat8'-'cat9' space
# - it is therefore not productive to bin the rare classes of 'cat8' and 'cat9' into the single categories as the significant intelligence could be lost
# - instead of it, we  can try to generate a number of new numeric features (means of each of the raw features, grouped by 'cat7', 'cat8', and 'cat9')

# In[177]:


print('We are done. That is all, folks!')
finish_time = dt.datetime.now()
print("Finished at ", finish_time)
elapsed = finish_time - start_time
print("Elapsed time: ", elapsed)

