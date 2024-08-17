#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Hey, thanks for viewing my Kernel!
# 
# If you like my work, please, leave an upvote: it will be really appreciated and it will motivate me in offering more content to the Kaggle community ! :)

# In[1]:


import pandas as pd
import numpy as np
import warnings

warnings.simplefilter("ignore")
train = pd.read_csv("../input/us-patent-phrase-to-phrase-matching/train.csv")
test = pd.read_csv("../input/us-patent-phrase-to-phrase-matching/test.csv")
sub = pd.read_csv("../input/us-patent-phrase-to-phrase-matching/sample_submission.csv")

display(train.head())
display(test.head())
display(sub.head())


# In[2]:


print('train shape: ', train.shape)
print('test shape: ', test.shape)
print('sub shape: ', sub.shape)


# In[3]:


display(train.isna().sum())
display(test.isna().sum())


# In[4]:


display(train.duplicated().sum())
display(test.duplicated().sum())


# In[5]:


from IPython.core.display import HTML
def value_counts_all(df, columns):
    pd.set_option('display.max_rows', 50)
    table_list = []
    for col in columns:
        table_list.append(pd.DataFrame(df[col].value_counts()))
    return HTML(
        f"<table><tr> {''.join(['<td>' + table._repr_html_() + '</td>' for table in table_list])} </tr></table>")


# In[6]:


value_counts_all(train, ['anchor', 'target', 'context', 'score'])


# In[7]:


value_counts_all(test, ['anchor', 'target', 'context'])


# # Contexts
# 
# A: Human Necessities <br />
# B: Operations and Transport <br />
# C: Chemistry and Metallurgy <br />
# D: Textiles <br />
# E: Fixed Constructions <br />
# F: Mechanical Engineering <br />
# G: Physics <br />
# H: Electricity <br />
# Y: Emerging Cross-Sectional Technologies <br />

# In[8]:


context_dict = {
    'A': 'Human Necessities',
    'B': 'Operations and Transport',
    'C': 'Chemistry and Metallurgy',
    'D': 'Textiles',
    'E': 'Fixed Constructions',
    'F': 'Mechanical Engineering',
    'G': 'Physics',
    'H': 'Electricity',
    'Y': 'Emerging Cross-Sectional Technologies'
}


# In[9]:


train['context'].str.len().max()


# # Feature Engineering

# In[10]:


cpc_codes_df = pd.read_csv("../input/cpc-codes/titles.csv")
cpc_codes_df.head(10)


# In[11]:


cpc_codes_df.shape


# In[12]:


def create_feature(df, cpc_codes_df):
    import fuzzywuzzy
    from fuzzywuzzy import fuzz
    from fuzzywuzzy import process
    
    df['section'] = df['context'].str[:1]
    df['class'] = df['context'].str[1:]
    
    df['anchor_len'] = df['anchor'].apply(lambda x: len(x.split(' ')))
    df['target_len'] = df['target'].apply(lambda x: len(x.split(' ')))
    
    pattern = '[0-9]'
    mask = df['anchor'].str.contains(pattern, na=False)
    df['num_anchor'] = mask
    mask = df['target'].str.contains(pattern, na=False)
    df['num_target'] = mask
    
    df['context_desc'] = df['context'].map(cpc_codes_df.set_index('code')['title']).str.lower()
    
    fuzzy_anchor_target_scores = []
    fuzzy_anchor_context_scores = []
    fuzzy_taget_context_scores = []
    for index, row in df.iterrows():
        fuzzy_anchor_target_scores.append(fuzz.ratio(row['anchor'], row['target']))
        fuzzy_anchor_context_scores.append(fuzz.ratio(row['anchor'], row['context_desc']))
        fuzzy_taget_context_scores.append(fuzz.ratio(row['context_desc'], row['target']))
    df['fuzzy_at_score'] = fuzzy_anchor_target_scores
    df['fuzzy_ac_score'] = fuzzy_anchor_context_scores
    df['fuzzy_tc_score'] = fuzzy_taget_context_scores
    df['fuzzy_c_score'] = df['fuzzy_ac_score'] + df['fuzzy_tc_score']
    df['fuzzy_total'] = df['fuzzy_at_score'] + df['fuzzy_c_score']
    
    df.drop(['context', 'fuzzy_ac_score', 'fuzzy_tc_score'], 1, inplace=True)
    
    return df


# In[13]:


new_train = create_feature(train.copy(), cpc_codes_df)
new_test = create_feature(test.copy(), cpc_codes_df)
new_train.head()


# # Distribution

# In[14]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
fig, ax = plt.subplots(figsize=(16, 8))
sns.countplot(data=new_train, x='section', ax=ax)
ax.set_xticklabels([context_dict['A'], context_dict['C'], context_dict['F'], context_dict['H'], context_dict['B'], 
                    context_dict['D'], context_dict['E'], context_dict['G']], rotation=45);


# In[15]:


fig, ax = plt.subplots(figsize=(16, 8))
sns.countplot(data=new_train, x='class', ax=ax);


# In[16]:


fig, ax = plt.subplots(figsize=(16, 8))
sns.kdeplot(data=new_train, x='anchor_len', ax=ax);


# In[17]:


fig, ax = plt.subplots(figsize=(16, 8))
sns.kdeplot(data=new_train, x='target_len', ax=ax);


# In[18]:


fig, ax = plt.subplots(figsize=(16, 8))
sns.countplot(data=new_train, x='num_anchor', ax=ax);
for container in ax.containers:
    ax.bar_label(container)


# In[19]:


fig, ax = plt.subplots(figsize=(16, 8))
sns.countplot(data=new_train, x='num_target', ax=ax);
for container in ax.containers:
    ax.bar_label(container)


# # Score Relationship

# In[20]:


temp_train = new_train.copy()
temp_train['score_jitter'] = new_train['score'] + np.random.normal(0, 0.1, size=len(new_train['score']))
temp_train['fuzzy_at_jitter'] = new_train['fuzzy_at_score'] + np.random.normal(0, 0.5, size=len(new_train['score']))


# In[21]:


def regplot_with_corr(df, x, y, ax=None):
    from matplotlib.offsetbox import AnchoredText
    if ax==None:
        fig, ax = plt.subplots(figsize=(16, 8))
        
    scatter_kws = dict(
                alpha=0.1,
                s=4,
            )
    line_kws = dict(color='red')
    corr = df[x].corr(df[y])
    sns.regplot(data=df, x=x, y=y, scatter_kws=scatter_kws,
                line_kws=line_kws, ax=ax)
    at = AnchoredText(
                f"{corr:.2f}",
                prop=dict(size="large"),
                frameon=True,
                loc="upper left",
            )
    at.patch.set_boxstyle("square, pad=0.0")
    ax.add_artist(at)


# In[22]:


regplot_with_corr(temp_train, 'score_jitter', 'fuzzy_at_jitter')


# In[23]:


regplot_with_corr(temp_train, 'score_jitter', 'fuzzy_c_score')


# In[24]:


regplot_with_corr(temp_train, 'score_jitter', 'fuzzy_total')


# In[25]:


fig, ax = plt.subplots(figsize=(16, 8))
temp_train['score_jitter'] = new_train['score'] + np.random.normal(0, 0.1, size=len(new_train['score']))
temp_train['fuzzy_at_jitter'] = new_train['fuzzy_at_score'] + np.random.normal(0, 0.5, size=len(new_train['score']))
sns.scatterplot(data=temp_train, x='score_jitter', y='fuzzy_at_jitter', ax=ax, alpha=0.5, s=10, hue='fuzzy_total');


# # Modeling

# In[26]:


numeric_cols = new_train.select_dtypes(include=np.number).columns.tolist()
object_cols = list(set(new_train.columns) - set(numeric_cols))
numeric_cols.remove("score")
ignore_cols = ['id']

print('numerical features: ', numeric_cols)
print('object features: ', object_cols)


# In[27]:


from catboost import CatBoostRegressor

cat_base = CatBoostRegressor(
    ignored_features=ignore_cols,
    cat_features=object_cols,
    eval_metric='MAE'
)


# In[28]:


X_train = new_train.drop(['score'], 1)
y_train = new_train['score']
cat_base.fit(X_train, y_train, silent=True)


# In[29]:


X_test = new_test.copy()
preds = pd.DataFrame(cat_base.predict(X_test), columns=['preds'])
preds.head()


# In[30]:


sub['score'] = preds['preds']
sub.to_csv('submission.csv', index=False)
sub.head()


# # Feature Importance

# In[31]:


def plot_feature_importance(importance,names,model_type):
    
    #Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    
    #Create a DataFrame using a Dictionary
    data={'feature_names':feature_names,'feature_importance':feature_importance}
    fi_df = pd.DataFrame(data)
    
    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)
    
    #Define size of bar plot
    plt.figure(figsize=(10,8))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title(model_type + 'FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')


# In[32]:


plot_feature_importance(cat_base.get_feature_importance(), X_train.columns, 'CATBOOST')


# In[33]:


import shap

explainer = shap.TreeExplainer(cat_base)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)


# In[34]:


shap.initjs()
shap.force_plot(explainer.expected_value, shap_values, X_test)

