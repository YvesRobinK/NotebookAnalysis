#!/usr/bin/env python
# coding: utf-8

# <font size=4>Note:
#    This notebook is from <a href="https://www.kaggle.com/code/abdullahmeda/enter-ing-the-timeseries-space-sec-3-new-aggs">{ENTER}ing the TimeSeries {SPACE} Sec 3 + New Aggs</a>,I am just a learner of this Notebook.Thank you very much for making this code public for me to learn.I have carefully read each line of code and provided comments and explanations in areas I do not understand </font>

# <font size=4> Load Data.</font>

# In[1]:


import pandas as pd#导入csv文件的库
import numpy as np#进行矩阵运算的库
import gc#垃圾回收模块

import re#用于正则表达式提取
from collections import Counter#用于对一组元素计数
#设置随机种子,保证模型可以复现
import random
np.random.seed(2023)
random.seed(2023)
#import optuna#自动超参数优化软件框架
import warnings#避免一些可以忽略的报错
warnings.filterwarnings('ignore')#filterwarnings()方法是用于设置警告过滤器的方法，它可以控制警告信息的输出方式和级别。


# In[2]:


train_logs=pd.read_csv("/kaggle/input/linking-writing-processes-to-writing-quality/train_logs.csv")
print(f"len(train_logs):{len(train_logs)}")
train_logs['activity']=train_logs['activity'].apply(lambda x:"Move From" if x[:9]=="Move From" else x )

train_logs.head()


# In[3]:


train_scores=pd.read_csv("/kaggle/input/linking-writing-processes-to-writing-quality/train_scores.csv")
print(f"len(train_scores):{len(train_scores)}")
train_scores.head()


# In[4]:


train_df=pd.merge(train_logs,train_scores,on="id",how="left")
print(f"len(train_df):{len(train_df)}")
train_df.head()


# In[5]:


test_logs=pd.read_csv("/kaggle/input/linking-writing-processes-to-writing-quality/test_logs.csv")
print(f"len(test_logs):{len(test_logs)}")
test_logs['activity']=test_logs['activity'].apply(lambda x:"Move From" if x[:9]=="Move From" else x )
test_logs.head()


# In[6]:


#对一个id把down_time,up_time之类的变量打包,求个均值.
train_logs_agg_df = train_df.groupby("id")[['down_time', 'up_time', 'action_time', 'cursor_position', 'word_count', 'score']].mean().reset_index()
train_logs_agg_df.head()


# In[7]:


from collections import defaultdict#一个存在默认值的字典,访问不存在的值时抛出的是默认值

class Preprocessor:#数据预处理的一个类
    
    def __init__(self, seed):
        self.seed = seed#随机种子
        
        self.activities = ['Input', 'Remove/Cut', 'Nonproduction', 'Replace', 'Paste','Move From']#这是activity的一列
        self.events = ['q', 'Space', 'Backspace', 'Shift', 'ArrowRight', 'Leftclick', 'ArrowLeft', '.', ',', 
              'ArrowDown', 'ArrowUp', 'Enter', 'CapsLock', "'", 'Delete', 'Unidentified']#down_event中选出一些重要的
        self.text_changes = ['q', ' ', 'NoChange', '.', ',', '\n', "'", '"', '-', '?', ';', '=', '/', '\\', ':']#text_change中选出一些重要的
        self.punctuations = ['"', '.', ',', "'", '-', ';', ':', '?', '!', '<', '>', '/',
                        '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+']#down_event中的一些标点符号
        self.gaps = [1, 2, 3, 5, 10, 20, 50, 100]#滞后项
        
        #这里是用于存储每个activity的idf值
        self.idf = defaultdict(float)#创建了一个float类型的字典,如果访问不存在,默认值为0.0
    
    #统计df对象中activity的count
    def activity_counts(self, df):
        #对每个id的所有activity组合成一个列表
        tmp_df = df.groupby('id').agg({'activity': list}).reset_index()
        #创建一个空列表
        ret = list()
        for li in tmp_df['activity'].values:#取出一个人的activity列表
            items = list(Counter(li).items())#转成[(activity1:count1),(activity2:count2),……]
            di = dict()#一个空字典
            #每个activity初始化为0
            for k in self.activities:
                di[k] = 0
            #统计每个activity的count
            for item in items:
                k, v = item[0], item[1]#k:activity v:count
                if k in di:
                    di[k] = v
            #加上这个人的每个activity的count
            ret.append(di)
        #转成pandas类型
        ret = pd.DataFrame(ret)
        #给表格的每列换个名字
        cols = [f'activity_{i}_count' for i in range(len(ret.columns))]
        ret.columns = cols

        #每列元素求和,文章中出现的总次数
        cnts = ret.sum(1)

        #前面是词袋模型,这里转成tf-idf模型
        for col in cols:#activity_i_count
            if col in self.idf.keys():#字典里如果已经有这个key了
                idf = self.idf[col]
            else:#不在这个字典里
                #计算idf=log(数据量/(某列和+1))
                idf = np.log(df.shape[0] / (ret[col].sum() + 1))
                self.idf[col] = idf#将col的idf加入字典
            #ret[col] / cnts :给定文章的次数/在文章中出现的总次数,为什么取log再加1不知道
            ret[col] = 1 + np.log(ret[col] / cnts)
            ret[col] *= idf

        return ret#tf-idf

    #这个是event的tf-idf模型,这里可能有down_event和up_event,故colname单独设置
    def event_counts(self, df, colname):
        tmp_df = df.groupby('id').agg({colname: list}).reset_index()
        ret = list()
        for li in tmp_df[colname].values:
            items = list(Counter(li).items())
            di = dict()
            for k in self.events:
                di[k] = 0
            for item in items:
                k, v = item[0], item[1]
                if k in di:
                    di[k] = v
            ret.append(di)
        ret = pd.DataFrame(ret)
        cols = [f'{colname}_{i}_count' for i in range(len(ret.columns))]
        ret.columns = cols

        cnts = ret.sum(1)

        for col in cols:
            if col in self.idf.keys():
                idf = self.idf[col]
            else:
                idf = df.shape[0] / (ret[col].sum() + 1)
                idf = np.log(idf)
                self.idf[col] = idf
            
            ret[col] = 1 + np.log(ret[col] / cnts)
            ret[col] *= idf

        return ret

    #text_change的tf-idf模型
    def text_change_counts(self, df):
        tmp_df = df.groupby('id').agg({'text_change': list}).reset_index()
        ret = list()
        for li in tmp_df['text_change'].values:
            items = list(Counter(li).items())
            di = dict()
            for k in self.text_changes:
                di[k] = 0
            for item in items:
                k, v = item[0], item[1]
                if k in di:
                    di[k] = v
            ret.append(di)
        ret = pd.DataFrame(ret)
        cols = [f'text_change_{i}_count' for i in range(len(ret.columns))]
        ret.columns = cols

        cnts = ret.sum(1)

        for col in cols:
            if col in self.idf.keys():
                idf = self.idf[col]
            else:
                idf = df.shape[0] / (ret[col].sum() + 1)
                idf = np.log(idf)
                self.idf[col] = idf
            
            ret[col] = 1 + np.log(ret[col] / cnts)
            ret[col] *= idf
            
        return ret
    #统计标点之类的出现的次数,不过这次是直接将它们相加做统计的.(可能这样比tf-idf好?)
    def match_punctuations(self, df):
        tmp_df = df.groupby('id').agg({'down_event': list}).reset_index()
        ret = list()
        for li in tmp_df['down_event'].values:
            cnt = 0
            items = list(Counter(li).items())
            for item in items:
                k, v = item[0], item[1]
                if k in self.punctuations:#只要在这张表里,就相加
                    cnt += v
            ret.append(cnt)
        ret = pd.DataFrame({'punct_cnt': ret})
        return ret


    def get_input_words(self, df):
        #~是取反的布尔值 取出text_change 中不包含 => 且不是Nochange的
        tmp_df = df[(~df['text_change'].str.contains('=>'))&(df['text_change'] != 'NoChange')].reset_index(drop=True)
        #在drop掉包含 => 和Nochange之后 按id打包成列表
        tmp_df = tmp_df.groupby('id').agg({'text_change': list}).reset_index()
        #将列表连接成一个整体
        tmp_df['text_change'] = tmp_df['text_change'].apply(lambda x: ''.join(x))
        #用正则表达式子匹配一个或者多个'q'字符
        tmp_df['text_change'] = tmp_df['text_change'].apply(lambda x: re.findall(r'q+', x))
        #统计len,也就是统计text_change中有多少个有q的字符
        tmp_df['input_word_count'] = tmp_df['text_change'].apply(len)
        #求均值,方差,最大值,取到np.nan就设置为0
        tmp_df['input_word_length_mean'] = tmp_df['text_change'].apply(lambda x: np.mean([len(i) for i in x] if len(x) > 0 else 0))
        tmp_df['input_word_length_max'] = tmp_df['text_change'].apply(lambda x: np.max([len(i) for i in x] if len(x) > 0 else 0))
        tmp_df['input_word_length_std'] = tmp_df['text_change'].apply(lambda x: np.std([len(i) for i in x] if len(x) > 0 else 0))
        tmp_df.drop(['text_change'], axis=1, inplace=True)
        return tmp_df
    
    #对df做特征工程
    def make_feats(self, df):
        
        print("Starting to engineer features")
        #创建一个只有id一列的表格
        feats = pd.DataFrame({'id': df['id'].unique().tolist()})
        #做时序上的特征工程
        print("Engineering time data")
        for gap in self.gaps:
            print(f"-> for gap {gap}")
            #利用up_time的shift创造action_time_gap
            df[f'up_time_shift{gap}'] = df.groupby('id')['up_time'].shift(gap)
            df[f'action_time_gap{gap}'] = df['down_time'] - df[f'up_time_shift{gap}']
        df.drop(columns=[f'up_time_shift{gap}' for gap in self.gaps], inplace=True)

        #对cursor_position做特征工程,这个就是自己-自己
        print("Engineering cursor position data")
        for gap in self.gaps:
            print(f"-> for gap {gap}")
            df[f'cursor_position_shift{gap}'] = df.groupby('id')['cursor_position'].shift(gap)
            df[f'cursor_position_change{gap}'] = df['cursor_position'] - df[f'cursor_position_shift{gap}']
            #取了绝对值,鼠标向前移动也是移动了.
            df[f'cursor_position_abs_change{gap}'] = np.abs(df[f'cursor_position_change{gap}'])
        df.drop(columns=[f'cursor_position_shift{gap}' for gap in self.gaps], inplace=True)

        #对word_count做类似的特征工程,词数减少也是移动了.
        print("Engineering word count data")
        for gap in self.gaps:
            print(f"-> for gap {gap}")
            df[f'word_count_shift{gap}'] = df.groupby('id')['word_count'].shift(gap)
            df[f'word_count_change{gap}'] = df['word_count'] - df[f'word_count_shift{gap}']
            df[f'word_count_abs_change{gap}'] = np.abs(df[f'word_count_change{gap}'])
        df.drop(columns=[f'word_count_shift{gap}' for gap in self.gaps], inplace=True)
        
        print("Engineering statistical summaries for features")
        #需要对哪些特征做哪些统计变量,这些都是大佬统计好的,就不做修改了.
        feats_stat = [
            ('event_id', ['max']),
            ('up_time', ['max']),
            ('action_time', ['max', 'min', 'mean', 'std', 'quantile', 'sem', 'sum', 'skew', pd.DataFrame.kurt]),
            ('activity', ['nunique']),
            ('down_event', ['nunique']),
            ('up_event', ['nunique']),
            ('text_change', ['nunique']),
            ('cursor_position', ['nunique', 'max', 'quantile', 'sem', 'mean']),
            ('word_count', ['nunique', 'max', 'quantile', 'sem', 'mean'])]
        #滞后特征的统计变量用for循环进行添加
        for gap in self.gaps:
            feats_stat.extend([
                (f'action_time_gap{gap}', ['max', 'min', 'mean', 'std', 'quantile', 'sem', 'sum', 'skew', pd.DataFrame.kurt]),
                (f'cursor_position_change{gap}', ['max', 'mean', 'std', 'quantile', 'sem', 'sum', 'skew', pd.DataFrame.kurt]),
                (f'word_count_change{gap}', ['max', 'mean', 'std', 'quantile', 'sem', 'sum', 'skew', pd.DataFrame.kurt])
            ])
        
        pbar = feats_stat
        for item in pbar:
            colname, methods = item[0], item[1]#取出某列特征和需要进行的统计学的量'max'
            for method in methods:
                #转成能放入agg的方法
                if isinstance(method, str):
                    method_name = method
                else:
                    method_name = method.__name__
                #添加到feats里.
                tmp_df = df.groupby(['id']).agg({colname: method}).reset_index().rename(columns={colname: f'{colname}_{method_name}'})
                feats = feats.merge(tmp_df, on='id', how='left')

        #调用方法求activity的tf-idf
        print("Engineering activity counts data")
        tmp_df = self.activity_counts(df)
        feats = pd.concat([feats, tmp_df], axis=1)
        #调用方法求down_event和up_event的tf-idf
        print("Engineering event counts data")
        tmp_df = self.event_counts(df, 'down_event')
        feats = pd.concat([feats, tmp_df], axis=1)
        tmp_df = self.event_counts(df, 'up_event')
        feats = pd.concat([feats, tmp_df], axis=1)
        
        print("Engineering text change counts data")
        tmp_df = self.text_change_counts(df)
        feats = pd.concat([feats, tmp_df], axis=1)
        
        print("Engineering punctuation counts data")
        tmp_df = self.match_punctuations(df)
        feats = pd.concat([feats, tmp_df], axis=1)

        # input words
        print("Engineering input words data")
        tmp_df = self.get_input_words(df)
        feats = pd.merge(feats, tmp_df, on='id', how='left')

        # compare feats
        print("Engineering ratios data")
        feats['word_time_ratio'] = feats['word_count_max'] / feats['up_time_max']
        feats['word_event_ratio'] = feats['word_count_max'] / feats['event_id_max']
        feats['event_time_ratio'] = feats['event_id_max']  / feats['up_time_max']
        feats['idle_time_ratio'] = feats['action_time_gap1_sum'] / feats['up_time_max']
        
        print("Done!")
        return feats


# In[8]:


preprocessor = Preprocessor(seed=2023)
print("Engineering features for training data")

train_feats = preprocessor.make_feats(train_logs)
print("-"*25)
print("Engineering features for test data")
test_feats = preprocessor.make_feats(test_logs)


# In[9]:


#根据id对某些特征求了一些统计学变量
train_agg_fe_df = train_logs.groupby("id")[['down_time', 'up_time', 'action_time', 'cursor_position', 'word_count']].agg(['mean', 'std', 'min', 'max', 'last', 'first', 'sem', 'median', 'sum'])
#列名转成:"特征_统计学量"
train_agg_fe_df.columns = ['_'.join(x) for x in train_agg_fe_df.columns]
#给列名添加上前缀tmp_
train_agg_fe_df = train_agg_fe_df.add_prefix("tmp_")
#将行号设置为[0,1,2,3,……]
train_agg_fe_df.reset_index(inplace=True)
train_agg_fe_df.head()


# In[10]:


#对测试集做同样的操作
test_agg_fe_df = test_logs.groupby("id")[['down_time', 'up_time', 'action_time', 'cursor_position', 'word_count']].agg(['mean', 'std', 'min', 'max', 'last', 'first', 'sem', 'median', 'sum'])
test_agg_fe_df.columns = ['_'.join(x) for x in test_agg_fe_df.columns]
test_agg_fe_df = test_agg_fe_df.add_prefix("tmp_")
test_agg_fe_df.reset_index(inplace=True)
test_agg_fe_df.head()


# In[11]:


#将文件合并起来
train_feats = train_feats.merge(train_agg_fe_df, on='id', how='left')
test_feats = test_feats.merge(test_agg_fe_df, on='id', how='left')


# In[12]:


data = []

for logs in [train_logs, test_logs]:
    #up_time向后移动并且用down_time填充缺失的位置
    logs['up_time_lagged'] = logs.groupby('id')['up_time'].shift(1).fillna(logs['down_time'])
    #(down_time减上一个时刻的up_time) /1000是单位转换
    logs['time_diff'] = abs(logs['down_time'] - logs['up_time_lagged']) / 1000

    #按照id打包time_diff
    group = logs.groupby('id')['time_diff']
    #延迟时间的max,min,median
    largest_lantency = group.max()
    smallest_lantency = group.min()
    median_lantency = group.median()
    #down_time的first /1000是做单位转换吧
    initial_pause = logs.groupby('id')['down_time'].first() / 1000
    #分层次求和
    pauses_half_sec = group.apply(lambda x: ((x > 0.5) & (x <= 1)).sum())
    pauses_1_sec = group.apply(lambda x: ((x > 1) & (x <= 1.5)).sum())
    pauses_1_half_sec = group.apply(lambda x: ((x > 1.5) & (x <= 2)).sum())
    pauses_2_sec = group.apply(lambda x: ((x > 2) & (x <= 3)).sum())
    pauses_3_sec = group.apply(lambda x: (x > 3).sum())

    data.append(pd.DataFrame({
        'id': logs['id'].unique(),
        #延迟
        'largest_lantency': largest_lantency,
        'smallest_lantency': smallest_lantency,
        'median_lantency': median_lantency,
        'initial_pause': initial_pause,
        'pauses_half_sec': pauses_half_sec,
        'pauses_1_sec': pauses_1_sec,
        'pauses_1_half_sec': pauses_1_half_sec,
        'pauses_2_sec': pauses_2_sec,
        'pauses_3_sec': pauses_3_sec,
    }).reset_index(drop=True))

train_eD592674, test_eD592674 = data

gc.collect()


# In[13]:


#将延时的特征加入
train_feats = train_feats.merge(train_eD592674, on='id', how='left')
test_feats = test_feats.merge(test_eD592674, on='id', how='left')
#在训练的feature中加入标签
train_feats = train_feats.merge(train_scores, on='id', how='left')


# In[14]:


from sklearn import metrics#评估指标
from sklearn import model_selection#划分训练集和验证集的
from sklearn.preprocessing import LabelEncoder#对类别特征进行编码

import lightgbm as lgb#lgbm回归

le = LabelEncoder()

train_feats['score_class'] = le.fit_transform(train_feats['score'])#转换成了独热编码.

target_col = ['score']

drop_cols = ['id', 'score_class']

train_cols = [col for col in train_feats.columns if col not in target_col + drop_cols]#选出X的col

models_dict = {}
scores = []

test_predict_list = []
#这个估计也是大佬用optuna调参得到的.
best_params = {'reg_alpha': 0.007678095440286993, 
               'reg_lambda': 0.34230534302168353, 
               'colsample_bytree': 0.627061253588415, 
               'subsample': 0.854942238828458, 
               'learning_rate': 0.038697981947473245, 
               'num_leaves': 22, 
               'max_depth': 37, 
               'min_child_samples': 18}

for i in range(5): 
    kf = model_selection.KFold(n_splits=10, random_state=2023+ i, shuffle=True)#每次用不同的随机种子?

    oof_valid_preds = np.zeros(train_feats.shape[0], )#验证集的预测

    X_test = test_feats[train_cols]#选出X的几列


    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_feats)):

        print("==-"* 50)
        print("Fold : ", fold)

        X_train, y_train = train_feats.iloc[train_idx][train_cols], train_feats.iloc[train_idx][target_col]
        X_valid, y_valid = train_feats.iloc[valid_idx][train_cols], train_feats.iloc[valid_idx][target_col]


        params = {
            "objective": "regression",
            "metric": "rmse",
            'random_state': 2023,
            "n_estimators" : 12001,
            "verbosity": -1,
            **best_params
        }

        model = lgb.LGBMRegressor(**params)

        #只使用第一个评估指标
        early_stopping_callback = lgb.early_stopping(200, first_metric_only=True, verbose=False)
        verbose_callback = lgb.log_evaluation(100)#100次输出一个结果

        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],  
                  callbacks=[early_stopping_callback, verbose_callback],
        )

        valid_predict = model.predict(X_valid)
        oof_valid_preds[valid_idx] = valid_predict

        test_predict = model.predict(X_test)
        test_predict_list.append(test_predict)

        score = metrics.mean_squared_error(y_valid, valid_predict, squared=False)
        print("Fold RMSE Score : ", score)

        models_dict[f'{fold}_{i}'] = model

    oof_score = metrics.mean_squared_error(train_feats[target_col], oof_valid_preds, squared=False)
    scores.append(oof_score)
    print("OOF RMSE Score : ", oof_score)


# In[15]:


test_feats['score'] = np.mean(test_predict_list, axis=0)
test_feats[['id', 'score']].to_csv("submission.csv", index=False)

