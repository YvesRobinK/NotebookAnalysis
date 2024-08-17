#!/usr/bin/env python
# coding: utf-8

# # Linking Writing Processes to Writing Quality
# # Feature Engineering for beginners (日本語/EN)

# 各idにおける以下14個の特徴量を作成し、トレーニング可能なデータフレームを作成する。  
# Create the following 14 features in each id to create a trainable data frame.
# - 1） 各idにおけるaction_timeの合計（action_timeの合計値で作成）
#       Sum of action_time for each id (created with the total value of action_time)
# - 2） 各idにおけるスタートポーズ時間（down_timeの最小値で作成）
#       Start pause time for each id (created using the minimum value of down_time)
# - 3） 各idにおけるenter実行回数（down_eventのEnterのカウント合計値で作成）
#       Enter execution count in each id (created by the total count of Enter in down_event)
# - 4)　各idにおけるSpace実行回数（down_eventのSpaceのカウント合計値で作成）
#       Number of Space executions in each id (created by the total count of Space in down_event)
# - 5)　各idにおけるBackspace実行回数（down_eventのBackspaceのカウント合計値で作成）
#       Number of Backspace executions in each id (created by the total count of Backspace in down_event)
# - 6） 各idにおけるシンボルの長さ（cursor_positionの最大値で作成）
#       Symbol length in each id (created by the maximum value of cursor_position)
# - 7） 各idにおけるテキストの長さ（word_countの最大値で作成）
#       Length of text in each id (created by the maximum value of word_count)
# - 8） 各idにおける活動中の非生産行動回数（activityのNonproductionの平均値で作成）
#       Number of non-production actions during the activity in each id (created by the average value of nonproduction of activity)
# - 9） 各idにおける活動中のinput回数（activityのInputの平均値で作成）
#       Number of inputs during the activity in each id (created by the average value of Input in activity)
# - 10）各idにおける活動中のremove回数（activityのremove/Cutの平均値で作成）
#       Number of removals during activity for each id (created by the average value of remove/Cut of activity)
# - 11）各idにおける平均アクション時間（action_timeの平均値で作成）
#       Average action time for each id (created by the average value of action_time)
# - 12）各idにおける活動中の置き換え処理回数（activityのReplaceの要素数で作成）
#       Number of replace processes during the activity for each id (created by the number of elements in the activity's Replace)
# - 13）各idにおけるtext_changeのユニーク要素数（text_changeのユニーク要素数で作成）
#       Number of unique elements of text_change in each id (created by the number of unique elements of text_change)
# - 14）各idにおけるセンテンスの数（text_changeとdown_eventから要素数を抽出し作成）
#       Number of sentences in each id (created by extracting the number of elements from text_change and down_event)

# In[1]:


# ライブラリのインポート / Importing Libraries
import numpy as np
import pandas as pd


# In[2]:


# データ読み込み / data loading
df = pd.read_csv('/kaggle/input/linking-writing-processes-to-writing-quality/train_logs.csv')
train_scores = pd.read_csv('/kaggle/input/linking-writing-processes-to-writing-quality/train_scores.csv')
test_df = pd.read_csv('/kaggle/input/linking-writing-processes-to-writing-quality/test_logs.csv')


# In[3]:


# train data
df.head()


# In[4]:


# train data scores
train_scores.head()


# In[5]:


# test data
test_df.head()


# ### 各変数のユニーク数、数、型、欠損値の有無を確認
# ### Check each variable for uniqueness, number, type, and missing values

# In[6]:


def stats(data):
    
    maxx = []
    minn = []
    for i in data.columns:
        maxx.append(data[i].value_counts().max())
        minn.append(data[i].value_counts().min())

    return pd.DataFrame(
        {'nunique': data.nunique(),
         'len': len(data),
         'types':data.dtypes,
         'Nulls' : data.isna().sum(),
         "Value counts Max": maxx,
         'Value counts Min':minn 
        },
        columns = ['nunique', 'len','types','Nulls'
                   ,"Value counts Max",'Value counts Min']).\
        sort_values(by ='nunique',ascending = False)


# In[7]:


# train data
stats(df)


# In[8]:


# test data
stats(test_df)


# ### 1） 各idにおけるaction_timeの合計（action_timeの合計値で作成）
# ###     Sum of action_time for each id (created with the total value of action_time)
#         

# In[9]:


# 各idにおけるaction_timeの合計 / Total action_time for each id
new_df = df.groupby('id')['action_time'].sum().reset_index()
new_df = new_df.rename(columns={'action_time': 'summary_time'})
new_df.head()


# ### 2） 各idにおけるスタートポーズ時間（down_timeの最小値で作成）
# ###     Start pause time for each id (created using the minimum value of down_time)

# In[10]:


# スタートポーズ / starting pose
df_tmp = df.groupby('id')['down_time'].min().reset_index()
df_tmp = df_tmp.rename(columns={'down_time': 'start_pause'})
new_df = pd.merge(new_df, df_tmp, on='id', how='outer')
new_df.head()


# ### 3） 各idにおけるenter実行回数（down_eventのEnterのカウント合計値で作成）
# ###     Enter execution count in each id (created by the total count of Enter in down_event)

# In[11]:


# enter実行回数 / Number of times enter has been executed
copy_df = df
copy_df['enter_click'] = (copy_df['down_event'] == 'Enter')
copy_df = copy_df.groupby('id')['enter_click'].sum().reset_index()
new_df = pd.merge(new_df, copy_df, on='id', how='outer')
new_df.head()


# ### 4)　各idにおけるSpace実行回数（down_eventのSpaceのカウント合計値で作成）
# ###     Number of Space executions in each id (created by the total count of Space in down_event)

# In[12]:


# space実行回数 / Number of space execution
copy_df = df
copy_df['space_click'] = (copy_df['down_event'] == 'Space')
copy_df = copy_df.groupby('id')['space_click'].sum().reset_index()
new_df = pd.merge(new_df, copy_df, on='id', how='outer')
new_df.head()


# ### 5)　各idにおけるBackspace実行回数（down_eventのBackspaceのカウント合計値で作成）
# ###     Number of Backspace executions in each id (created by the total count of Backspace in down_event)

# In[13]:


# backspace実行回数 / Number of backspace executions
copy_df = df
copy_df['backspace_click'] = (copy_df['down_event'] == 'Backspace')
copy_df = copy_df.groupby('id')['backspace_click'].sum().reset_index()
new_df = pd.merge(new_df, copy_df, on='id', how='outer')
new_df.head()


# ### 6） 各idにおけるシンボルの長さ（cursor_positionの最大値で作成）
# ###     Symbol length in each id (created by the maximum value of cursor_position)

# In[14]:


# シンボル長さ / Symbol Length
df_tmp = df.groupby('id')['cursor_position'].max().reset_index()
df_tmp = df_tmp.rename(columns={'cursor_position': 'symbol_length'})
new_df = pd.merge(new_df, df_tmp, on='id', how='outer')
new_df.head()


# ### 7） 各idにおけるテキストの長さ（word_countの最大値で作成）
# ###     Length of text in each id (created by the maximum value of word_count)

# In[15]:


# テキスト長さ / text length
df_tmp = df.groupby('id')['word_count'].max().reset_index()
new_df = pd.merge(new_df, df_tmp, on='id', how='outer')
new_df.head()


# ### 8） 各idにおける活動中の非生産行動回数（activityのNonproductionの平均値で作成）
# ###     Number of non-production actions during the activity in each id (created by the average value of nonproduction of activity)

# In[16]:


# activityにおけるnonproductionの処理 / Processing of nonproduction in activity
df_tmp = df.groupby('id')['activity'].apply(lambda x: (x == 'Nonproduction').mean() * 100).reset_index()
df_tmp = df_tmp.rename(columns={'activity': 'nonproduction_feature'})
new_df = pd.merge(new_df, df_tmp, on='id', how='outer')
new_df.head()


# ### 9） 各idにおける活動中のinput回数（activityのInputの平均値で作成）
# ###     Number of inputs during the activity in each id (created by the average value of Input in activity)

# In[17]:


# activityにおけるinputの処理 / Processing of inputs in activity
df_tmp = df.groupby('id')['activity'].apply(lambda x: (x == 'Input').mean() * 100).reset_index()
df_tmp = df_tmp.rename(columns={'activity': 'input_feature'})
new_df = pd.merge(new_df, df_tmp, on='id', how='outer')
new_df.head()


# ### 10）各idにおける活動中のremove回数（activityのremove/Cutの平均値で作成）
# ###     Number of removals during activity for each id (created by the average value of remove/Cut of activity)

# In[18]:


# activityにおけるremoveの処理 / Processing of remove in activity
df_tmp = df.groupby('id')['activity'].apply(lambda x: (x == 'Remove/Cut').mean() * 100).reset_index()
df_tmp = df_tmp.rename(columns={'activity': 'remove_feature'})
new_df = pd.merge(new_df, df_tmp, on='id', how='outer')
new_df.head()


# ### 11）各idにおける平均アクション時間（action_timeの平均値で作成）
# ###     Average action time for each id (created by the average value of action_time)

# In[19]:


# 平均アクション時間 / Average Action Time
df_tmp = df.groupby('id')['action_time'].mean().reset_index()
df_tmp = df_tmp.rename(columns={'action_time': 'mean_action_time'})
new_df = pd.merge(new_df, df_tmp, on='id', how='outer')
new_df.head()


# ### 12）各idにおける活動中の置き換え処理回数（activityのReplaceの要素数で作成）
# ###     Number of replace processes during the activity for each id (created by the number of elements in the activity's Replace)

# In[20]:


# activityにおけるreplaceの処理 / Processing of replace in activity
df_tmp = df[df['activity'] == 'Replace'].groupby('id').size().reset_index(name='replace_feature')
new_df = pd.merge(new_df, df_tmp, on='id', how='outer')
new_df.head()


# ### 13）各idにおけるtext_changeのユニーク要素数（text_changeのユニーク要素数で作成）
# ###     Number of unique elements of text_change in each id (created by the number of unique elements of text_change)

# In[21]:


# text_changeのユニーク要素数 / Number of unique elements in text_change
df_tmp = df.groupby('id')['text_change'].nunique().reset_index()
df_tmp = df_tmp.rename(columns={'text_change': 'tch_unique'})
new_df = pd.merge(new_df, df_tmp, on='id', how='outer')
new_df.head()


# ### 14）各idにおけるセンテンスの数（text_changeとdown_eventから要素数を抽出し作成）
# ###     Number of sentences in each id (created by extracting the number of elements from text_change and down_event)

# In[22]:


# センテンスの数 / Number of sentences
df_tmp = df[(df['text_change'] == '.') & (df['down_event'] != 'Backspace')].groupby('id').size().reset_index(name = 'number_sentence')
new_df = pd.merge(new_df, df_tmp, on='id', how='outer')
new_df.head()


# ### 欠損値処理
# ### missing-value processing

# In[23]:


new_df.isnull().sum()


# In[24]:


# 欠損値をゼロで埋める / Fill in missing values with zeros
new_df['replace_feature'] = new_df['replace_feature'].fillna(0)
new_df['number_sentence'] = new_df['number_sentence'].fillna(0)


# In[25]:


new_df.head()


# ## まとめ（これまでの処理を関数化）
# ## Summary (Functionalize the process so far)

# In[26]:


def summary_time(df):
    result = df.groupby('id')['action_time'].sum().reset_index()
    result.rename(columns={'action_time': 'summary_time'}, inplace=True)
    return result
def start_pause(df):
    result = df.groupby('id')['down_time'].min().reset_index()
    result.rename(columns={'down_time': 'start_pause'}, inplace=True)
    return result
def enter_click(df):
    copy_df = df
    copy_df['enter_click'] = (copy_df['down_event'] == 'Enter')
    copy_df = copy_df.groupby('id')['enter_click'].sum().reset_index()
    return copy_df
def space_click(df):
    copy_df = df
    copy_df['space_click'] = (copy_df['down_event'] == 'Space')
    copy_df = copy_df.groupby('id')['space_click'].sum().reset_index()
    return copy_df
def backspace_click(df):
    copy_df = df
    copy_df['backspace_click'] = (copy_df['down_event'] == 'Backspace')
    copy_df = copy_df.groupby('id')['backspace_click'].sum().reset_index()
    return copy_df
def symbol_length(df):
    result = df.groupby('id')['cursor_position'].max().reset_index()
    result.rename(columns={'cursor_position': 'symbol_length'}, inplace=True)
    return result
def text_length(df):
    result = df.groupby('id')['word_count'].max().reset_index()
    return result
def nonproduction_feature(df):
    result = df.groupby('id')['activity'].apply(lambda x: (x == 'Nonproduction').mean() * 100).reset_index()
    result.rename(columns={'activity': 'nonproduction_feature'}, inplace=True)
    return result
def input_feature(df):
    result = df.groupby('id')['activity'].apply(lambda x: (x == 'Input').mean() * 100).reset_index()
    result.rename(columns={'activity': 'input_feature'}, inplace=True)
    return result
def remove_feature(df):
    result = df.groupby('id')['activity'].apply(lambda x: (x == 'Remove/Cut').mean() * 100).reset_index()
    result.rename(columns={'activity': 'remove_feature'}, inplace=True)
    return result
def mean_action_time(df):
    result = df.groupby('id')['action_time'].mean().reset_index()
    result.rename(columns={'action_time': 'mean_action_time'}, inplace=True)
    return result
def replace_feature(df):
    result = df[df['activity'] == 'Replace'].groupby('id').size().reset_index(name='replace_feature')
    return result
def text_change_unique(df):
    result = df.groupby('id')['text_change'].nunique().reset_index()
    result.rename(columns={'text_change': 'tch_unique'}, inplace=True)
    return result
def sentence_size_feature(df):
    result = df[(df['text_change'] == '.') & (df['down_event'] != 'Backspace')].groupby('id').size().reset_index(name = 'number_sentence')
    return result


# In[27]:


def getDataset(train_df):
    new_df = summary_time(train_df)

    functions = [
        start_pause, enter_click, space_click,
        backspace_click, symbol_length, text_length, nonproduction_feature,
        input_feature, remove_feature, mean_action_time,replace_feature,text_change_unique, sentence_size_feature
    ]

    for func in functions:
        result_df = func(train_df)
        new_df = pd.merge(new_df, result_df, on='id', how='outer')

    return new_df


# In[28]:


# 関数適用 / function application
df = getDataset(df)
test = getDataset(test_df)


# In[29]:


# 欠損値をゼロで埋める / Fill in missing values with zeros
df['replace_feature'] = df['replace_feature'].fillna(0)
df['number_sentence'] = df['number_sentence'].fillna(0)


# In[30]:


df.head()


# #### これで学習可能なデータフレームの完成です。 / This completes the trainable data frame.   
# #### 最後まで見ていただき、ありがとうございました！！ / Thank you for seeing it through to the end!!

# In[ ]:




