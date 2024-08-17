#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd#导入csv文件的库
import numpy as np#进行矩阵运算的库
import matplotlib.pyplot as plt#强大的绘图库
#设置随机种子,保证模型可以复现
import random
np.random.seed(2023)
random.seed(2023)


# In[2]:


train_logs=pd.read_csv("/kaggle/input/linking-writing-processes-to-writing-quality/train_logs.csv")
print(f"len(train_logs):{len(train_logs)}")
train_logs.head()


# <font size=4>Due to the fact that the user corresponds to multiple pieces of data, the data similar to action_time uses mean, variance, and quantity to extract features.</font>

# In[3]:


id=train_logs['action_time'].groupby([train_logs['id']]).mean().keys().values
mean_time=train_logs['action_time'].groupby([train_logs['id']]).mean().values
std_time=train_logs['action_time'].groupby([train_logs['id']]).std().values
count_time=train_logs['action_time'].groupby([train_logs['id']]).count().values


# <font size=4>For example,cursor_position、word_count is increasing, and features can be extracted using mean, variance, and maximum values</font>

# In[4]:


mean_cursor=train_logs['cursor_position'].groupby([train_logs['id']]).mean().values
std_cursor=train_logs['cursor_position'].groupby([train_logs['id']]).std().values
max_cursor=train_logs['cursor_position'].groupby([train_logs['id']]).max().values
mean_word_count=train_logs['word_count'].groupby([train_logs['id']]).mean().values
std_word_count=train_logs['word_count'].groupby([train_logs['id']]).std().values
max_word_count=train_logs['word_count'].groupby([train_logs['id']]).max().values


# <font size=4>When activity=="input", it indicates that the author is entering</font>

# In[5]:


train_logs['activity']=(train_logs['activity']=="input")#在输入状态
mean_activity=train_logs['activity'].groupby([train_logs['id']]).mean().values
std_activity=train_logs['activity'].groupby([train_logs['id']]).std().values
count_activity=train_logs['activity'].groupby([train_logs['id']]).count().values


# <font size=4>We found that columns down_event、up_event、text_change have 'q'.</font>

# In[6]:


train_logs['down_event']=(train_logs['down_event']=="q")
train_logs['up_event']=(train_logs['up_event']=="q")
train_logs['text_change']=(train_logs['text_change']=="q")

mean_down_event=train_logs['down_event'].groupby([train_logs['id']]).mean().values
std_down_event=train_logs['down_event'].groupby([train_logs['id']]).std().values
count_down_event=train_logs['down_event'].groupby([train_logs['id']]).count().values

mean_up_event=train_logs['up_event'].groupby([train_logs['id']]).mean().values
std_up_event=train_logs['up_event'].groupby([train_logs['id']]).std().values
count_up_event=train_logs['up_event'].groupby([train_logs['id']]).count().values

mean_text_change=train_logs['text_change'].groupby([train_logs['id']]).mean().values
std_text_change=train_logs['text_change'].groupby([train_logs['id']]).std().values
count_text_change=train_logs['text_change'].groupby([train_logs['id']]).count().values


# In[7]:


train_df=pd.DataFrame({"id":id,
                             "mean_time":mean_time,"std_time":std_time,"count_time":count_time,
                             'mean_cursor':mean_cursor,'std_cursor':std_cursor,'max_cursor':max_cursor,
                             'mean_word_count':mean_word_count,'std_word_count':std_word_count,'max_word_count':max_word_count,
                             'mean_activity':mean_activity,'std_activity':std_activity,'count_activity':count_activity,
                             'mean_down_event':mean_down_event,'std_down_event':std_down_event,'count_down_event':count_down_event,
                             'mean_up_event':mean_down_event,'std_up_event':std_down_event,'count_up_event':count_down_event,
                             'mean_text_change':mean_text_change,'std_text_change':std_text_change,'count_text_change':count_text_change,
                             'mean_time_count':mean_time*count_time,
                            })
train_df.head()


# In[8]:


train_scores=pd.read_csv("/kaggle/input/linking-writing-processes-to-writing-quality/train_scores.csv")
print(f"len(train_scores):{len(train_scores)}")
train_scores.head()


# In[9]:


train_df=pd.merge(train_df,train_scores,on="id",how="left")
train_df.drop(['id'],axis=1,inplace=True)
train_df.head()


# In[10]:


def deal_df(df):
    id=df['action_time'].groupby([df['id']]).mean().keys().values
    mean_time=df['action_time'].groupby([df['id']]).mean().values
    std_time=df['action_time'].groupby([df['id']]).std().values
    count_time=df['action_time'].groupby([df['id']]).count().values
    
    mean_cursor=df['cursor_position'].groupby([df['id']]).mean().values
    std_cursor=df['cursor_position'].groupby([df['id']]).std().values
    max_cursor=df['cursor_position'].groupby([df['id']]).max().values
    mean_word_count=df['word_count'].groupby([df['id']]).mean().values
    std_word_count=df['word_count'].groupby([df['id']]).std().values
    max_word_count=df['word_count'].groupby([df['id']]).max().values
    
    df['activity']=(df['activity']=="input")#在输入状态
    mean_activity=df['activity'].groupby([df['id']]).mean().values
    std_activity=df['activity'].groupby([df['id']]).std().values
    count_activity=df['activity'].groupby([df['id']]).count().values
    
    
    df['down_event']=(df['down_event']=="q")
    df['up_event']=(df['up_event']=="q")
    df['text_change']=(df['text_change']=="q")

    mean_down_event=df['down_event'].groupby([df['id']]).mean().values
    std_down_event=df['down_event'].groupby([df['id']]).std().values
    count_down_event=df['down_event'].groupby([df['id']]).count().values

    mean_up_event=df['up_event'].groupby([df['id']]).mean().values
    std_up_event=df['up_event'].groupby([df['id']]).std().values
    count_up_event=df['up_event'].groupby([df['id']]).count().values

    mean_text_change=df['text_change'].groupby([df['id']]).mean().values
    std_text_change=df['text_change'].groupby([df['id']]).std().values
    count_text_change=df['text_change'].groupby([df['id']]).count().values
    
    df=pd.DataFrame({"id":id,
                             "mean_time":mean_time,"std_time":std_time,"count_time":count_time,
                             'mean_cursor':mean_cursor,'std_cursor':std_cursor,'max_cursor':max_cursor,
                             'mean_word_count':mean_word_count,'std_word_count':std_word_count,'max_word_count':max_word_count,
                             'mean_activity':mean_activity,'std_activity':std_activity,'count_activity':count_activity,
                             'mean_down_event':mean_down_event,'std_down_event':std_down_event,'count_down_event':count_down_event,
                             'mean_up_event':mean_down_event,'std_up_event':std_down_event,'count_up_event':count_down_event,
                             'mean_text_change':mean_text_change,'std_text_change':std_text_change,'count_text_change':count_text_change,
                             'mean_time_count':mean_time*count_time,
                            })
    
    return df


# In[11]:


test_logs=pd.read_csv("/kaggle/input/linking-writing-processes-to-writing-quality/test_logs.csv")
print(f"len(test_logs):{len(test_logs)}")
test_df=deal_df(test_logs)
test_df.head()

