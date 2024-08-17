#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Hey, thanks for viewing my Kernel!
# 
# If you like my work, please, leave an upvote: it will be really appreciated and it will motivate me in offering more content to the Kaggle community ! :)

# In[1]:


import pandas as pd
import numpy as np
import warnings 

warnings.simplefilter("ignore")
train_ = pd.read_csv("../input/tabular-playground-series-apr-2022/train.csv")
test = pd.read_csv("../input/tabular-playground-series-apr-2022/test.csv")
train_labels = pd.read_csv("../input/tabular-playground-series-apr-2022/train_labels.csv")
sub = pd.read_csv("../input/tabular-playground-series-apr-2022/sample_submission.csv")

display(train_.head())
display(test.head())
display(train_labels.head())
display(sub.head())


# In[2]:


print("train: ", train_.shape, "- 401.77 MB")
print("test: ", test.shape, "- 189.32 MB")
print("train_labels: ", train_labels.shape, "- 196.65 kB")
print("sub: ", sub.shape, "- 97.76 kB")


# In[3]:


display(train_.isna().sum().sum())
display(test.isna().sum().sum())
display(train_labels.isna().sum().sum())


# In[4]:


display(train_.duplicated().sum())
display(test.duplicated().sum())
display(train_labels.duplicated().sum())


# In[5]:


train_.dtypes


# In[6]:


train_labels['sequence'].max(), test['sequence'].min()


# In[7]:


train_['subject'].max(), test['subject'].min()


# ### Insights 1: Sequence and Subject features are series

# In[8]:


train = train_.merge(train_labels, on='sequence', how='left')
train.shape


# In[9]:


from IPython.core.display import HTML
def value_counts_all(df, columns):
    pd.set_option('display.max_rows', 50)
    table_list = []
    for col in columns:
        table_list.append(pd.DataFrame(df[col].value_counts()))
    return HTML(
        f"<table><tr> {''.join(['<td>' + table._repr_html_() + '</td>' for table in table_list])} </tr></table>")


# In[10]:


value_counts_all(train, ['sequence', 'subject', 'step'])


# In[11]:


value_counts_all(test, ['sequence', 'subject', 'step'])


# # Distribution

# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
fig, ax = plt.subplots(figsize=(16, 8))
sns.countplot(data=train, x='state', ax=ax);


# In[13]:


fig, ax = plt.subplots(figsize=(16, 8))
sns.kdeplot(data=train, x='subject', hue='state', ax=ax);


# In[14]:


fig, ax = plt.subplots(figsize=(16, 8))
sns.countplot(data=train, x='step', hue='state', ax=ax);


# In[15]:


def target_kde_plot(df, columns, target, ncol=3, figsize=(16, 8)):
    nrow = round(len(columns) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    col, row = 0, 0
    for col_name in columns:
        if nrow <= 1:
            sns.kdeplot(data=df, x=col_name, hue=target, ax=axes[col])
            col += 1
        else:
            sns.kdeplot(data=df, x=col_name, hue=target, ax=axes[row][col])
            row += 1
            if row >= nrow:
                col += 1
                row = 0


# In[16]:


target_kde_plot(train, ['sequence', 'subject', 'step'], 'state', ncol=3, figsize=(16, 8))


# In[17]:


temp_df = pd.concat([train[['sequence', 'subject', 'step']], test[['sequence', 'subject', 'step']]])
temp_df.reset_index(inplace=True)
fig, axes = plt.subplots(1, 3, figsize=(16, 8))
sns.kdeplot(data=temp_df, x='sequence', ax=axes[0])
sns.kdeplot(data=temp_df, x='subject', ax=axes[1])
sns.kdeplot(data=temp_df, x='step', ax=axes[2]);


# ### Insights 2: Sequence feature has uniform distribution accornding to all data

# In[18]:


sensor_cols = ['sensor_'+'%02d'%i for i in range(1, 13)]
target_kde_plot(train, sensor_cols, 'state', ncol=3, figsize=(16, 24))


# ### Insights 3: Sensor_02 has a Bernoulli distribution. It has two different distributions

# In[19]:


def target_box_plot(df, columns, target, ncol=3, figsize=(16, 8)):
    nrow = round(len(columns) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    col, row = 0, 0
    for col_name in columns:
        if nrow <= 1:
            sns.boxplot(data=df, y=col_name, x=target, ax=axes[col])
            col += 1
        else:
            sns.kdeplot(data=df, y=col_name, x=target, ax=axes[row][col])
            row += 1
            if row >= nrow:
                col += 1
                row = 0


# In[20]:


target_box_plot(train, ['sequence', 'subject', 'step'], 'state', ncol=3, figsize=(16, 8))


# In[21]:


def SMAPE(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 200.0
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0.0
    return np.mean(diff)

def sensor_train_test_distribution(df_train, df_test, figsize=(16, 32)):
    df_group_train = df_train.groupby(['step'], as_index=False).mean()
    df_group_train['flag'] = 'train'
    df_group_test = df_test.groupby(['step'], as_index=False).mean()
    df_group_test['flag'] = 'test'
    
    sensor_cols = ['sensor_'+'%02d'%i for i in range(0, 13)]
    df_group = pd.concat([df_group_train, df_group_test])
    df_group.reset_index(inplace=True)
    fig, axes = plt.subplots(len(sensor_cols), figsize=figsize, sharex=True)
    for index, col in enumerate(sensor_cols):
        sns.lineplot(data=df_group, x='step', y=col, hue='flag', ax=axes[index])
        smape_score = SMAPE(df_group.loc[df_group['flag']=='train', col], 
                            df_group.loc[df_group['flag']=='test', col])
        axes[index].text(0.95, 0.9,'smape:'+str(round(smape_score, 2)), horizontalalignment='center', 
                         verticalalignment='center', transform = axes[index].transAxes)
        axes[index].legend(loc='lower right');


# In[22]:


sensor_train_test_distribution(train, test)


# ### Insights 4: Sensor_02, 01, 04, 12 are a better fit with the test set and, Sensor_10 is the worst fit with the test set

# # Correlations

# In[23]:


def display_p_values(df, columns, target, th=0.05, cut=False):
    from scipy.stats import pearsonr
    p_values_list = []
    for c in columns:
        p = round(pearsonr(train.loc[:,target], train.loc[:,c])[1], 4)
        p_values_list.append(p)

    p_values_df = pd.DataFrame(p_values_list, columns=[target], index=columns)
    def p_value_warning_background(cell_value):
        highlight = 'background-color: lightcoral;'
        default = ''
        if cell_value > th:
                return highlight
        return default
    
    if cut:
        p_values_df_high = p_values_df[p_values_df[target] > th]
    else:
        p_values_df_high = p_values_df.copy()
    display(p_values_df_high.style.applymap(p_value_warning_background))


# In[24]:


sensor_cols = ['sensor_'+'%02d'%i for i in range(0, 13)]
sensor_cols.append('step')
display_p_values(train, sensor_cols, 'state', th=0.05)


# In[25]:


corr = train.corr()
fig, ax = plt.subplots(figsize=(16, 16))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, ax=ax, annot=True, fmt='.2f');


# In[26]:


upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
similar_cols = [column for column in upper.columns if any(upper[column] > 0.2)]
similar_cols


# # Feature Engineering

# In[27]:


def create_new_features(df, aggregation_cols=['sequence'], prefix=''):
    df['sensor_02_num'] = df['sensor_02'] > -15
    df['sensor_02_num'] = df['sensor_02_num'].astype(int)
    df['sensor_sum1'] = (df['sensor_00'] + df['sensor_09'] + df['sensor_06'] + df['sensor_01'])
    df['sensor_sum2'] = (df['sensor_01'] + df['sensor_11'] + df['sensor_09'] + df['sensor_06'] + df['sensor_00'])
    df['sensor_sum3'] = (df['sensor_03'] + df['sensor_11'] + df['sensor_07'])
    df['sensor_sum4'] = (df['sensor_04'] + df['sensor_10'])
    
    agg_strategy = {
                    'sensor_00': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_01': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_02': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_03': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_04': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_05': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_06': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_07': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_08': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_09': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_10': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_11': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_12': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_02_num': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_sum1': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_sum2': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_sum3': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                    'sensor_sum4': ['mean', 'max', 'min', 'var', 'mad', 'sum', 'median'],
                   }
    
    group = df.groupby(aggregation_cols).aggregate(agg_strategy)
    group.columns = ['_'.join(col).strip() for col in group.columns]
    group.columns = [str(prefix) + str(col) for col in group.columns]
    group.reset_index(inplace = True)
    
    temp = (df.groupby(aggregation_cols).size().reset_index(name = str(prefix) + 'size'))
    group = pd.merge(temp, group, how = 'left', on = aggregation_cols,)
    return group


# In[28]:


train_fe = create_new_features(train, aggregation_cols=['sequence', 'subject'])
test_fe = create_new_features(test, aggregation_cols=['sequence', 'subject'])


# In[29]:


train_fe_subjects = create_new_features(train, aggregation_cols = ['subject'], prefix = 'subject_')
test_fe_subjects = create_new_features(test, aggregation_cols = ['subject'], prefix = 'subject_')


# In[30]:


train_fe = train_fe.merge(train_fe_subjects, on='subject', how='left')
train_fe = train_fe.merge(train_labels, on='sequence', how='left')
test_fe = test_fe.merge(test_fe_subjects, on='subject', how='left')


# In[31]:


print(train_fe.shape, test_fe.shape)


# In[32]:


def plot_umap(embedding, df, col, ax=None):
    colors = pd.factorize(df.loc[:, col])
    colors_dict = {}
    for index, label in enumerate(df[col].unique()):
        colors_dict[index] = label
    color_list = sns.color_palette(None, len(df[col].unique()))
    
    if ax == None:
        fig, ax = plt.subplots(figsize=(12,12))
        for color_key in colors_dict.keys():
            indexs = colors[0] == color_key
            temp_embedding = embedding[indexs, :]
            ax.scatter(temp_embedding[:, 0], temp_embedding[:, 1], 
                        c=color_list[color_key], 
                        edgecolor='none', 
                        alpha=0.80,
                        label=colors_dict[color_key],
                        s=10)
        ax.legend(bbox_to_anchor=(1, 1), fontsize="x-large", markerscale=2.)
        ax.set_title('UMAP - ' + col, fontsize=18);
    else:
        for color_key in colors_dict.keys():
            indexs = colors[0] == color_key
            temp_embedding = embedding[indexs, :]
            ax.scatter(temp_embedding[:, 0], temp_embedding[:, 1], 
                        c=color_list[color_key], 
                        edgecolor='none', 
                        alpha=0.80,
                        label=colors_dict[color_key],
                        s=10)
        ax.legend(bbox_to_anchor=(1, 1), fontsize="x-large", markerscale=2.)
        ax.set_title('UMAP - ' + col, fontsize=18);


# In[33]:


import umap

embedding = umap.UMAP(n_neighbors=10,
                      min_dist=0.3,
                      metric='correlation').fit_transform(train_fe.drop(['sequence', 'subject'], 1))


# In[34]:


fig, ax = plt.subplots(figsize=(16, 16))
plot_umap(embedding, train_fe.drop(['sequence', 'subject'], 1), "state", ax=ax)


# # Modeling

# In[35]:


from lightgbm import LGBMClassifier

X_test = test_fe.drop(['sequence', 'subject'], 1)
X_train = train_fe[X_test.columns]
y_train = train_fe[['state']]

model = LGBMClassifier()


# In[36]:


model.fit(X_train, y_train)


# In[37]:


sub['state'] = model.predict(X_test)
sub.to_csv('submission.csv', index=False)
sub['state'] = model.predict_proba(X_test)[:, 1]
sub.to_csv('submission_proba.csv', index=False)


# # Feature Importance

# In[38]:


def plot_feature_importance(importance,names,model_type, figsize=(10, 8)):
    
    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    
    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)
    
    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    
    #Define size of bar plot
    plt.figure(figsize=figsize)
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title(model_type + ' FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')


# In[39]:


plot_feature_importance(model.feature_importances_, X_train.columns, 'LGBM', figsize=(16, 48))

