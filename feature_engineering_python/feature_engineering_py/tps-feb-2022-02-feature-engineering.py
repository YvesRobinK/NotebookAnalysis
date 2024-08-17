#!/usr/bin/env python
# coding: utf-8

# <h1 div class='alert alert-success'><center> Feature Engineering </center></h1>
# 
# ![](https://storage.googleapis.com/kaggle-competitions/kaggle/26480/logos/header.png?t=2021-04-09-00-57-05)

# # <div class="alert alert-success">  OBJETIVO </div> 

# O objetivo neste notebook é criação novas variáveis (feature) que possam ajudar na identificação de novos padrões, com a finalidade de bater a baseline estabelecida no [notebook]() de 0.94297 com XGBoost utilizando scaler MaxAbsScaler  e **n_estimators** com 1000.

# # <div class="alert alert-success">  1. IMPORTAÇÕES </div> 

# ## 1.1. Instalações

# In[1]:


get_ipython().system(' pip install --q scikit-plot')


# In[2]:


#from google.colab import drive
#drive.mount('/content/drive')


# ## 1.2. Bibliotecas 

# In[3]:


import warnings
import random
import os
import gc
import torch
import math
import sklearn.exceptions


# In[4]:


import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt 
import seaborn           as sns
import joblib            as jb
import xgboost           as xgb
import scikitplot        as skplt


# In[5]:


from sklearn.model_selection import train_test_split,  KFold, StratifiedKFold
from sklearn.preprocessing   import StandardScaler, MinMaxScaler, RobustScaler 
from sklearn.preprocessing   import MaxAbsScaler, QuantileTransformer, LabelEncoder, normalize
from sklearn.impute          import SimpleImputer
from sklearn                 import metrics
from datetime                import datetime
from sklearn.cluster         import KMeans
from sklearn.decomposition   import PCA


# In[6]:


from yellowbrick.cluster        import KElbowVisualizer, SilhouetteVisualizer
from sklearn.utils.class_weight import compute_sample_weight
from scipy                      import stats
from scipy.cluster              import hierarchy as hc
from math                       import factorial
from scipy.stats                import mode


# ## 1.3. Funções
# Aqui centralizamos todas as funções desenvolvidas durante o projeto para melhor organização do código.

# In[7]:


def jupyter_setting():
    
    get_ipython().run_line_magic('matplotlib', 'inline')
      
    #os.environ["WANDB_SILENT"] = "true" 
    #plt.style.use('bmh') 
    #plt.rcParams['figure.figsize'] = [20,15]
    #plt.rcParams['font.size']      = 13
     
    pd.options.display.max_columns = None
    #pd.set_option('display.expand_frame_repr', False)

    warnings.filterwarnings(action='ignore')
    warnings.simplefilter('ignore')
    warnings.filterwarnings('ignore')
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
    warnings.filterwarnings("ignore", category= sklearn.exceptions.UndefinedMetricWarning)

    pd.set_option('display.max_rows', 200)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.max_colwidth', None)

    icecream = ["#00008b", "#960018","#008b00", "#00468b", "#8b4500", "#582c00"]
    #sns.palplot(sns.color_palette(icecream))
    
    colors = ["lightcoral", "sandybrown", "darkorange", "mediumseagreen",
          "lightseagreen", "cornflowerblue", "mediumpurple", "palevioletred",
          "lightskyblue", "sandybrown", "yellowgreen", "indianred",
          "lightsteelblue", "mediumorchid", "deepskyblue"]
    
    # Colors
    dark_red   = "#b20710"
    black      = "#221f1f"
    green      = "#009473"
    myred      = '#CD5C5C'
    myblue     = '#6495ED'
    mygreen    = '#90EE90'    
    color_cols = [myred, myblue,mygreen]
    
    return icecream, colors, color_cols

icecream, colors, color_cols = jupyter_setting()


# In[8]:


def reduce_memory_usage(df, verbose=True):
    
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    
    for col in df.columns:
        
        col_type = df[col].dtypes
        
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
        
    return df


# In[9]:


def missing_zero_values_table(df):
        mis_val         = df.isnull().sum()
        mis_val_percent = round(df.isnull().mean().mul(100), 2)
        mz_table        = pd.concat([mis_val, mis_val_percent], axis=1)
        mz_table        = mz_table.rename(columns = {df.index.name:'col_name', 
                                                     0 : 'Valores ausentes', 
                                                     1 : '% de valores totais'})
        
        mz_table['Tipo de dados'] = df.dtypes
        mz_table                  = mz_table[mz_table.iloc[:,1] != 0 ]. \
                                     sort_values('% de valores totais', ascending=False)
        
        msg = "Seu dataframe selecionado tem {} colunas e {} " + \
              "linhas. \nExistem {} colunas com valores ausentes."
            
        print (msg.format(df.shape[1], df.shape[0], mz_table.shape[0]))
        
        return mz_table.reset_index()


# In[10]:


def scaler_MaxAbsScaler_StandardScaler(df):    
    sc_mm = MaxAbsScaler()
    sc_st = StandardScaler()     
    col = df.columns
    df  = sc_mm.fit_transform(df)
    df  = pd.DataFrame(sc_st.fit_transform(df), columns=col)    
    return df


# In[11]:


def diff(t_a, t_b):
    from dateutil.relativedelta import relativedelta
    t_diff = relativedelta(t_b, t_a)  # later/end time comes first!
    return '{h}h {m}m {s}s'.format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)


# In[12]:


def free_gpu_cache():
    
    # https://www.kaggle.com/getting-started/140636
    #print("Initial GPU Usage")
    #gpu_usage()                             

    #cuda.select_device(0)
    #cuda.close()
    #cuda.select_device(0)   
    
    gc.collect()
    torch.cuda.empty_cache()


# ## 1.4. Criar estrutura de pasta 
# 

# In[13]:


paths = ['img', 'Data', 'Data/pkl', 'Data/submission', 'Data/tunning', 
         'model', 'model/preds', 'model/optuna','model/preds/test', 'model/mdl/',
         'model/preds/test/n1', 'model/preds/test/n2', 'model/preds/test/n3', 
         'model/preds/train', 'model/preds/train/n1', 'model/preds/train/n2', 
         'model/preds/train/n3', 'model/preds/param']

for path in paths:
    try:
        os.mkdir(path)
    except:
        pass  


# ## 1.5. Dataset

# ### 1.5.2. Carregar Dados

# In[14]:


path      = '/content/drive/MyDrive/kaggle/Tabular Playground Series/2022/02 - Fevereiro/'
path      = '../input/tabular-playground-series-feb-2022/'
path_data = '' 
target    = 'target'


# In[15]:


df1_train     = pd.read_csv(path + path_data + 'train.csv')
df1_test      = pd.read_csv(path + path_data + 'test.csv')
df_submission = pd.read_csv(path + path_data + 'sample_submission.csv')

df1_train.shape, df1_test.shape, df_submission.shape


# ### 1.5.3. Visualizar os dados 

# In[16]:


df1_train.head()


# In[17]:


df1_test.head()


# # <div class="alert alert-success"> 2. PROCESSAMENTO </div> 

# In[18]:


df2_train = df1_train.copy()
df2_test  = df1_test.copy()
df2_train.shape, df2_test.shape


# ## 2.1. Excluir variáveis

# In[19]:


df2_train.drop('row_id', axis=1, inplace=True)
df2_test.drop('row_id', axis=1, inplace=True)


# In[20]:


features = df2_train.columns[df2_train.columns!=target]


# ## 2.2. Duplicados 
# referencia: https://www.kaggle.com/ambrosm/tpsfeb22-02-postprocessing-against-the-mutants

# In[21]:


df2_train.duplicated().sum()


# In[22]:


vc = df2_train.value_counts()
dedup_train = pd.DataFrame([list(tup) for tup in vc.index.values], columns=df2_train.columns)
dedup_train['sample_weight'] = vc.values
dedup_train.shape


# In[23]:


(df2_train[features].values == dedup_train[features].iloc[0].values.reshape(1, -1)).all(axis=1).sum()


# In[24]:


df2_train = dedup_train.copy()


# ## 2.3. Redução dos datasets

# In[25]:


df2_train = reduce_memory_usage(df2_train)
df2_test  = reduce_memory_usage(df2_test)


# In[26]:


df2_train.shape, df2_test.shape


# # <div class="alert alert-success"> 3. FEATURE ENGINEERING </div> 
# Nesta parte do processo vamos criar diversas variávies com o intuito de ajudar o modelo a identificar novos padrões e consequentemente melhor o desempenho, como padrão vamos criar todas as variáveis com inicial **fe_**, a cada criação de novas variáveis vamos treinar o modelo __XGBoost__ e identificar se as novas variáveis ajudam a encontrar novos padrões.

# ## 3.1. Feature Descritivas 
# Nesta etapa vamos criar novar variárias com medidas estatísticas.

# In[27]:


feature_float = df2_test.select_dtypes(np.number).columns


# In[28]:


def feature_statistic(df, feature_float, feature_cat=None):
    
    df['fe_mean']         = df[feature_float].mean(axis=1)   
    df['fe_std']          = df[feature_float].std(axis=1)   
    df['fe_median']       = df[feature_float].median(axis=1)   
    df['fe_var']          = df[feature_float].var(axis=1) 
    df['fe_min']          = df[feature_float].min(axis=1)   
    df['fe_max']          = df[feature_float].max(axis=1)      
    df['fe_skew']         = df[feature_float].skew(axis=1)   
    df['fe_kurt']         = df[feature_float].kurt(axis=1)
    df['fe_quantile_25']  = df[feature_float].quantile(q=.25, axis=1)
    df['fe_quantile_50']  = df[feature_float].quantile(q=.5, axis=1)
    df['fe_quantile_75']  = df[feature_float].quantile(q=.75, axis=1)
    
    df['fe_range']        = df['fe_max'] - df['fe_min']
    df['fe_iqr']          = df['fe_quantile_75'] - df['fe_quantile_25']    
    df['fe_tails']        = df['fe_range'] / df['fe_iqr']    
    df['fe_dispersion_1'] = df['fe_median'] / df['fe_iqr']   
        
    if feature_cat is not None:
        df['fe_dammy_count'] = df[feature_cat].sum(axis=1)   
        
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(999, inplace=True)
    
    df = reduce_memory_usage (df, verbose=False)

    return df.dropna(axis=1)


# In[29]:


feature_new = list(df2_train.columns)


# In[30]:


df3_1_train = feature_statistic(df2_train.copy(), feature_new)  
df3_1_test  = feature_statistic(df2_test.copy(), feature_float)

df3_1_train.shape, df3_1_test.shape


# In[31]:


df3_1_train.filter(regex=r'fe').head().info()


# ## 3.1.1. Modelagem

# In[32]:


lb     = LabelEncoder()
X      = df3_1_train.drop([target], axis=1)
y      = pd.DataFrame(lb.fit_transform(df3_1_train[target]), columns=[target])
X_test = df3_1_test

X_train, X_valid, y_train, y_valid = train_test_split(X, y, 
                                                      test_size    = 0.2,
                                                      shuffle      = True, 
                                                      stratify     = y, 
                                                      random_state = 12359)

sample_weight_train = X_train['sample_weight']
sample_weight_valid = X_valid['sample_weight']

X_train.drop('sample_weight', axis=1, inplace=True)
X_valid.drop('sample_weight', axis=1, inplace=True)

X_train.shape, y_train.shape, X_valid.shape, y_valid.shape , X_test.shape


# In[33]:


get_ipython().run_cell_magic('time', '', '\nseed   = 12359\nparams = {"objective"     : \'multi:softmax\',    \n          \'eval_metric\'   : \'mlogloss\',         \n          \'n_estimators\'  : 1000,\n          \'random_state\'  : seed}\n\nif torch.cuda.is_available():           \n    params.update({\'tree_method\': \'gpu_hist\', \'predictor\': \'gpu_predictor\'})\n\nscaler     = RobustScaler()\nX_train_sc = pd.DataFrame(scaler.fit_transform(X_train) , columns=X_train.columns)\nX_valid_sc = pd.DataFrame(scaler.transform(X_valid), columns=X_valid.columns)\nX_test_sc  = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)\n\nmodel = xgb.XGBClassifier(**params)\nmodel.fit(X_train_sc, y_train, sample_weight_train)\n\ny_pred      = model.predict(X_valid_sc)\ny_pred_test = model.predict(X_test_sc)\n\nacc = metrics.accuracy_score(y_valid, y_pred, sample_weight=sample_weight_valid)\n\nprint (\'ACC: {:2.5f}\'.format(acc), end=\'\\n\\n\')\n')


# ### 3.1.2. Feature Importances  

# In[34]:


df_imp = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_})

plt.figure(figsize=(12, 7))
sns.barplot(x="importance", 
            y="feature", 
            data=df_imp.sort_values(by="importance", ascending=False).iloc[:25])

plt.title("XGB Feature Importance")
plt.tight_layout()

plt.show()


# <div class="alert alert-info" role="alert">
#     
# **`NOTA:`** <br>
# Podemos observar acima no gráfico de importância das variáveis, que temos 3 variáveis que criamos entre as 25 primeiras variáveis, vamos fazer uma validação cruzada e armazenar a importância das variáveis em cada fold, para termos uma ideia de como essas variáveis se comportam. 
#     
# </div>

# ### 3.1.2. Validação cruzada

# In[35]:


def save_data_model(model_, model_name_, path_, y_pred_train_prob_, y_pred_test_prob_, y_pred_test_, score_, seed_, level_='1', target_='target'):    
    
    level = 'n' + level_ + '/'

    if score_>.6:    
        path_name_param = path_ + 'model/preds/param/' + model_name_.format(score_, seed_) + '.pkl.z'
        path_name_train = path_ + 'model/preds/train/' + level + model_name_.format(score_, seed_)  + '.pkl.z'
        path_name_test  = path_ + 'model/preds/test/'  + level + model_name_.format(score_, seed_)  + '.pkl.z'   
        path_name_model = path_ + 'model/mdl/'         + model_name_.format(score_, seed_)  + '.pkl.z'   
        
        jb.dump(y_pred_train_prob_, path_name_train)
        jb.dump(y_pred_test_prob_, path_name_test)
        jb.dump(model_, path_name_model)


# In[36]:


def model_train_cv_fit(model_, X_, y_, X_test_, lb_, target_, model_name_, sc_=MinMaxScaler(), sc_second_=None, 
                       n_splits_=5, seed_=12359, save_sub_=True, path_='', save_predict_=False, level_='1'):
    
    taco              = 76 
    y_preds_test      = []
    y_preds_val_prob  = [] 
    y_preds_test_prob = []
    score             = []
    mdl               = []
    col_prob          = y_[target_].sort_values().unique()
    df_preds_prob     = pd.DataFrame()
    df_feature_imp    = pd.DataFrame()
    time_start        = datetime.now()    
    n_estimators      = model_.get_params()['n_estimators']
    dub_scaler        = '=> Double Scaler' if sc_second_!=None else ''
    
    print('='*taco)
    print('Scaler: {} - n_estimators: {} {}'.format(sc, n_estimators, dub_scaler))
    print('='*taco)

    folds = StratifiedKFold(n_splits=n_splits_, shuffle=True, random_state=seed_)

    for fold, (trn_idx, val_idx) in enumerate(folds.split(X_, y_, groups=y)):

        time_fold_start = datetime.now()
        
        # ----------------------------------------------------
        # Separar dados para treino 
        # ----------------------------------------------------
        X_trn, X_val, sample_weight_train = X_.iloc[trn_idx], X_.iloc[val_idx], X_.iloc[trn_idx]['sample_weight']
        y_trn, y_val, sample_weight_valid = y_.iloc[trn_idx], y_.iloc[val_idx], X_.iloc[val_idx]['sample_weight'] 
                
        # ----------------------------------------------------
        # Processamento
        # ----------------------------------------------------        
        X_trn.drop('sample_weight', axis=1, inplace=True)
        X_val.drop('sample_weight', axis=1, inplace=True)
        
        X_trn = pd.DataFrame(sc_.fit_transform(X_trn), columns=X_trn.columns)
        X_val = pd.DataFrame(sc_.transform(X_val), columns=X_val.columns)
        X_tst = pd.DataFrame(sc_.transform(X_test_), columns=X_test_.columns)

        if sc_second_ is not None: 
            X_trn = pd.DataFrame(sc_second_.fit_transform(X_trn), columns=X_trn.columns)
            X_val = pd.DataFrame(sc_second_.transform(X_val), columns=X_val.columns)
            X_tst = pd.DataFrame(sc_second_.transform(X_tst), columns=X_tst.columns)
                        
        # ---------------------------------------------------- 
        # Treinar o modelo 
        # ----------------------------------------------------     
        model_.fit(X_trn, 
                   y_trn,
                   sample_weight_train,
                   eval_set              = [(X_trn, y_trn), (X_val, y_val)],          
                   early_stopping_rounds = int(n_estimators*.1),
                   verbose               = False)

        # ---------------------------------------------------- 
        # Predição 
        # ----------------------------------------------------     
        y_pred_val       = model_.predict(X_val, ntree_limit=model_.best_ntree_limit)    
        y_pred_val_prob  = model_.predict_proba(X_val, ntree_limit=model_.best_ntree_limit) 
        y_pred_test_prob = model_.predict_proba(X_tst, ntree_limit=model_.best_ntree_limit)
        
        y_preds_test.append(model_.predict(X_tst))
        y_preds_test_prob.append(y_pred_test_prob)
       
        df_prob_temp    = pd.DataFrame(y_pred_val_prob, columns=col_prob)
        y_pred_pbro_max = df_prob_temp.max(axis=1)

        df_prob_temp['fold']    = fold+1
        df_prob_temp['id']      = val_idx        
        df_prob_temp['y_val']   = y_val.values
        df_prob_temp['y_pred']  = y_pred_val            
        df_prob_temp['y_proba'] = np.max(y_pred_val_prob, axis=1)
        
        df_preds_prob = pd.concat([df_preds_prob, df_prob_temp], axis=0)
        
        # ---------------------------------------------------- 
        # Score 
        # ---------------------------------------------------- 
        acc   = metrics.accuracy_score(y_val, y_pred_val, sample_weight=sample_weight_valid)
        f1    = metrics.f1_score(y_val, y_pred_val, average='macro') # weighted
        prec  = metrics.precision_score(y_val, y_pred_val, average='macro')

        score.append(acc)     

        # ---------------------------------------------------- 
        # Print resultado  
        # ---------------------------------------------------- 
        time_fold_end = diff(time_fold_start, datetime.now())
        msg = '[Fold {}] ACC: {:2.5f} - F1-macro: {:2.5f} - Precision: {:2.5f}  - {}'
        print(msg.format(fold+1, acc, f1, prec, time_fold_end))

        # ---------------------------------------------------- 
        # Feature Importance
        # ----------------------------------------------------             
        feat_imp = pd.DataFrame(index   = X_trn.columns,
                                data    = model_.feature_importances_,                            
                                columns = ['fold_{}'.format(fold+1)])

        feat_imp['acc_'+str(fold+1)] = acc
        df_feature_imp = pd.concat([df_feature_imp, feat_imp], axis=1)

        # ---------------------------------------------------- 
        # Salvar o modelo 
        # ---------------------------------------------------- 
        dic_model = {'scaler': sc, 'scaler_second': sc_second_,'fold': fold+1,'model': model_}
        mdl.append(dic_model)

        time_end = diff(time_start, datetime.now())   

        acc_mean = np.mean(score) 
        acc_std  = np.std(score)

    df_preds_prob.sort_values("id", axis=0, ascending=True, inplace=True)

    # ------------------------------
    # Pós-processamento
    # referencia: https://www.kaggle.com/ambrosm/tpsfeb22-02-postprocessing-against-the-mutants
    # -------------------------------        
    y_proba  = sum(y_preds_test_prob) / len(y_preds_test_prob)
    y_proba += np.array([0, 0, 0.01, 0.03, 0, 0, 0, 0, 0, 0])  
    
    y_pred_tuned      = lb.inverse_transform(np.argmax(y_proba, axis=1))
    y_pred_tuned_prob = np.max(y_proba, axis=1)

    if save_predict_:                 
        save_data_model(model_             = mdl, 
                        model_name_        = model_name_ +'_'+str(sc_second_).lower()[:4], 
                        path_              = path_, 
                        y_pred_train_prob_ = df_preds_prob['y_proba'], 
                        y_pred_test_prob_  = y_pred_tuned_prob, 
                        y_pred_test_       = y_pred_tuned,
                        score_             = acc_mean, 
                        seed_              = seed_, 
                        level_             = level_, 
                        target_            = target_
                        ) 

    print('-'*taco)
    print('[Mean Fold] ACC: {:2.5f} std: {:2.5f} - {}'.format(acc_mean, acc_std, time_end))    
    print('='*taco)
    print()

    if save_sub_:         
        df_submission[target_] = y_pred_tuned        
        name_file_sub          = model_name_ +'_'+str(sc_second_).lower()[:4]+'.csv'
        df_submission.to_csv(path_+'Data/submission/'+name_file_sub.format(acc_mean), index=False)
        
    del X_trn, X_val, y_trn, y_val, feat_imp

    return mdl, df_feature_imp, df_feature_imp , df_preds_prob


# In[37]:


lb     = LabelEncoder()
X      = df3_1_train.drop([target], axis=1)
y      = pd.DataFrame(lb.fit_transform(df3_1_train[target]), columns=[target])
X_test = df3_1_test


# In[38]:


get_ipython().run_cell_magic('time', '', '\nseed_       = 12359\nmdl         = []\ndf_trn_mdl  = []\ndf_fe_imp   = []\nscaler_list = [None]\n\nparams = {"objective"    : \'multi:softmax\',    \n          \'eval_metric\'  : \'mlogloss\',   \n          \'n_estimators\' : 1000,\n          \'random_state\' : seed_}\n\nif torch.cuda.is_available():           \n    params.update({\'tree_method\': \'gpu_hist\', \'predictor\': \'gpu_predictor\'})\n    \nfor sc in scaler_list:    \n    model, df_trn, df_feature_imp, df_preds_prob = \\\n    model_train_cv_fit(model_        = xgb.XGBClassifier(**params),\n                       model_name_   = \'xgb_fe_score_01_{:2.5f}\',\n                       X_            = X,\n                       y_            = y,\n                       X_test_       = X_test,\n                       lb_           = lb,\n                       target_       = target,\n                       sc_           = RobustScaler(), \n                       sc_second_    = sc,\n                       n_splits_     = 5,\n                       seed_         = seed_,\n                       save_sub_     = True,\n                       path_         = \'\', \n                       save_predict_ = True)\n\n    mdl.append(model)\n    df_trn_mdl.append(df_trn)\n    df_fe_imp.append(df_feature_imp)\n\ndel model, df_trn, df_feature_imp\n')


# 
# 
# ### 3.1.3. Feature Importances  CV 

# In[39]:


for i in range(len(df_trn_mdl)):
    plt.figure(figsize=(20,15))
    row = int(np.round(df_trn_mdl[i].filter(regex=r'fold').shape[1] / 3 +1))
    for fold, col in enumerate(df_trn_mdl[i].filter(regex=r'fold').columns):            
        col_acc = 'acc_' + str(fold+1)
        df_fi = df_trn_mdl[i].sort_values(by=col, ascending=False).reset_index().iloc[:25]
        df_fi = df_fi[['index', col, col_acc]]
        df_fi.columns = ['Feature', 'score', col_acc]
        plt.subplot(row,3, fold+1)
        sns.barplot(x='score', y='Feature', data=df_fi)    
        plt.title('Fold {} - score: {:2.5f}'.format(fold+1, df_fi[col_acc].mean()), 
                  fontdict={'fontsize':18})    
    
    plt.suptitle('Feature Importance XGB - {}'.format(scaler_list[i]), y=1.05, fontsize=24);
    plt.tight_layout(h_pad=3.0);   


# ## 3.2. Agregar valores ATGC individuais
# referencia: 

# In[40]:


def feature_aggregate_ATGC(df, cols):
    A = np.zeros(len(cols))
    T = np.zeros(len(cols))
    G = np.zeros(len(cols))
    C = np.zeros(len(cols))

    for i, x in enumerate(cols):
        A[i] = int(x.split('A')[1].split('T')[0])
        T[i] = int(x.split('T')[1].split('G')[0])
        G[i] = int(x.split('G')[1].split('C')[0])
        C[i] = int(x.split('C')[1])

    A /= 10
    T /= 10
    G /= 10
    C /= 10

    df['fe_A'] = np.matmul(df[cols].to_numpy(), A[np.newaxis].T) 
    df['fe_T'] = np.matmul(df[cols].to_numpy(), T[np.newaxis].T) 
    df['fe_G'] = np.matmul(df[cols].to_numpy(), G[np.newaxis].T) 
    df['fe_C'] = np.matmul(df[cols].to_numpy(), C[np.newaxis].T) 

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(999, inplace=True)
    
    df = reduce_memory_usage (df, verbose=False)

    return df


# In[41]:


df3_2_train = feature_aggregate_ATGC(df3_1_train.copy(),feature_float)
df3_2_test  = feature_aggregate_ATGC(df3_1_test.copy(), feature_float)

df3_2_train.shape, df3_2_train.shape


# In[42]:


df3_2_train.filter(regex=r'fe').head().info()


# ### 3.2.1. Modelagem

# In[43]:


lb     = LabelEncoder()
X      = df3_2_train.drop(target, axis=1)
y      = pd.DataFrame(lb.fit_transform(df3_1_train[target]), columns=[target])
X_test = df3_2_test

X_train, X_valid, y_train, y_valid = train_test_split(X, y, 
                                                      test_size    = 0.2,
                                                      shuffle      = True, 
                                                      stratify     = y, 
                                                      random_state = 12359)
sample_weight_train = X_train['sample_weight']
sample_weight_valid = X_valid['sample_weight']

X_train.drop('sample_weight', axis=1, inplace=True)
X_valid.drop('sample_weight', axis=1, inplace=True)

X_train.shape, y_train.shape, X_valid.shape, y_valid.shape , X_test.shape


# In[44]:


get_ipython().run_cell_magic('time', '', '\nseed   = 12359\nparams = {"objective"     : \'multi:softmax\',    \n          \'eval_metric\'   : \'mlogloss\',   \n          \'n_estimators\'  : 1000,\n          \'random_state\'  : seed}\n\nif torch.cuda.is_available():           \n    params.update({\'tree_method\': \'gpu_hist\', \'predictor\': \'gpu_predictor\'})\n\nscaler     = RobustScaler()\nX_train_sc = pd.DataFrame(scaler.fit_transform(X_train) , columns=X_train.columns)\nX_valid_sc = pd.DataFrame(scaler.transform(X_valid), columns=X_valid.columns)\n\nmodel = xgb.XGBClassifier(**params)\nmodel.fit(X_train_sc, y_train, sample_weight_train)\n\ny_pred = model.predict(X_valid_sc)\n\nacc = metrics.accuracy_score(y_valid, y_pred, sample_weight=sample_weight_valid)\n\nprint (\'ACC: {:2.5f}\'.format(acc), end=\'\\n\\n\')\n# ACC: 0.97355\n')


# In[45]:


df_imp = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_})

plt.figure(figsize=(14, 7))
sns.barplot(x="importance", 
            y="feature", 
            data=df_imp.sort_values(by="importance", ascending=False).iloc[:25])

plt.title("XGB Feature Importance")
plt.tight_layout()

plt.show()


# ### 3.2.2. Validação cruzada

# In[46]:


lb     = LabelEncoder()
X      = df3_2_train.drop([target], axis=1)
y      = pd.DataFrame(lb.fit_transform(df3_2_train[target]), columns=[target])
X_test = df3_2_test


# In[47]:


get_ipython().run_cell_magic('time', '', '\nseed_       = 12359\nmdl         = []\ndf_trn_mdl  = []\ndf_fe_imp   = []\nscaler_list = [None]\n\nparams = {"objective"    : \'multi:softmax\',    \n          \'eval_metric\'  : \'mlogloss\',      \n          \'n_estimators\' : 1000,\n          \'random_state\' : seed_}\n\nif torch.cuda.is_available():           \n    params.update({\'tree_method\': \'gpu_hist\', \'predictor\': \'gpu_predictor\'})\n    \nfor sc in scaler_list:    \n    model, df_trn, df_feature_imp, df_preds_prob = \\\n    model_train_cv_fit(model_        = xgb.XGBClassifier(**params),\n                       model_name_   = \'xgb_fe_score_02_{:2.5f}\',\n                       X_            = X,\n                       y_            = y,\n                       X_test_       = X_test,\n                       lb_           = lb,\n                       target_       = target,\n                       sc_           = RobustScaler(), \n                       sc_second_    = sc,\n                       n_splits_     = 5,\n                       seed_         = seed_,\n                       save_sub_     = True,\n                       path_         = \'\', \n                       save_predict_ = True)\n    \n    mdl.append(model)\n    df_trn_mdl.append(df_trn)\n    df_fe_imp.append(df_feature_imp)\n\ndel model, df_trn, df_feature_imp\n')


# <div class="alert alert-info" role="alert">
#     
# **`NOTA:`** <br>
# Como podemos observar acima, tivemos um aumento no score, isso é um indicativo que as novas variáveis estão ajudando na predições.
#     
# </div>

# ### 3.2.3. Feature Importances  CV 

# In[48]:


for i in range(len(df_trn_mdl)):
    plt.figure(figsize=(20,12))
    for fold, col in enumerate(df_trn_mdl[i].filter(regex=r'fold').columns):            
        col_acc = 'acc_' + str(fold+1)
        df_fi = df_trn_mdl[i].sort_values(by=col, ascending=False).reset_index().iloc[:25]
        df_fi = df_fi[['index', col, col_acc]]
        df_fi.columns = ['Feature', 'score', col_acc]
        plt.subplot(2,3, fold+1)
        sns.barplot(x='score', y='Feature', data=df_fi)    
        plt.title('Fold {} - score: {:2.5f}'.format(fold+1, df_fi[col_acc].mean()), 
                  fontdict={'fontsize':18})    
    
    plt.suptitle('Feature Importance XGB - {}'.format(scaler_list[i]), y=1.05, fontsize=24);
    plt.tight_layout(h_pad=3.0);   


# ## 3.3. Adicionando decâmeros
# 
# Conforme explicado na EDA do @AMBROSM, há um número diferente de decâmeros na amostra devido ao processo descrito no artigo "Analysis of Identification Method for Bacterial Species and Antibiotic Resistance Genes Using Optical Data From DNA Oligomers" (https://www.frontiersin.org/articles/10.3389/fmicb.2020.00257/full). Queremos adicionar também este recurso para testar se é informativo.
# 
# Consulte este notebook https://www.kaggle.com/ambrosm/tpsfeb22-01-eda-which-makes-sense para obter o código original e vote-o se achar útil.

# In[49]:


df3_3_train = df3_2_train.copy()
df3_3_test  = df3_2_test.copy()


# In[50]:


def bias(w, x, y, z):
    return factorial(10) / (factorial(w) * factorial(x) * factorial(y) * factorial(z) * 4**10)

def bias_of(s):
    w = int(s[1:s.index('T')])
    x = int(s[s.index('T')+1:s.index('G')])
    y = int(s[s.index('G')+1:s.index('C')])
    z = int(s[s.index('C')+1:])
    return factorial(10) / (factorial(w) * factorial(x) * factorial(y) * factorial(z) * 4**10)

def gcd_of_all(df_i, elements=feature_float):
    gcd = df_i[elements[0]]
    for col in elements[1:]:
        gcd = np.gcd(gcd, df_i[col])
    return gcd


# In[51]:


train_i = pd.DataFrame({col: ((df3_3_train[col]+bias_of(col))*1000000).round().astype(int) for col in feature_float})
test_i  = pd.DataFrame({col: ((df3_3_test[col]+bias_of(col))*1000000).round().astype(int) for col in feature_float})

df3_3_train['fe_gcd'] = gcd_of_all(train_i)
df3_3_test['fe_gcd']  = gcd_of_all(test_i)

del([train_i, test_i])


# In[52]:


df3_3_train.filter(regex=r'fe').head().info()


# ## 3.3.1. Modelagem

# In[53]:


lb     = LabelEncoder()
X      = df3_3_train.drop(target, axis=1)
y      = pd.DataFrame(lb.fit_transform(df3_3_train[target]), columns=[target])
X_test = df3_3_test


# ### 3.3.2. Validação cruzada

# In[54]:


get_ipython().run_cell_magic('time', '', '\nseed_       = 12359\nmdl         = []\ndf_trn_mdl  = []\ndf_fe_imp   = []\nscaler_list = [None]\n\nparams = {"objective"    : \'multi:softmax\',    \n          \'eval_metric\'  : \'mlogloss\',   \n          \'n_estimators\' : 1000,\n          \'random_state\' : seed_}\n\nif torch.cuda.is_available():           \n    params.update({\'tree_method\': \'gpu_hist\', \'predictor\': \'gpu_predictor\'})\n    \nfor sc in scaler_list:    \n    model, df_trn, df_feature_imp, df_preds_prob = \\\n    model_train_cv_fit(model_        = xgb.XGBClassifier(**params),\n                       model_name_   = \'xgb_fe_score_03_{:2.5f}\',\n                       X_            = X,\n                       y_            = y,\n                       X_test_       = X_test,\n                       lb_           = lb,\n                       target_       = target,\n                       sc_           = RobustScaler(), \n                       sc_second_    = sc,\n                       n_splits_     = 5,\n                       seed_         = seed_,\n                       save_sub_     = True,\n                       path_         = \'\', \n                       save_predict_ = True)\n    \n    mdl.append(model)\n    df_trn_mdl.append(df_trn)\n    df_fe_imp.append(df_feature_imp)\n\ndel model, df_trn, df_feature_imp\n')


# ### 3.3.3. Feature Importances  CV 

# In[55]:


for i in range(len(df_trn_mdl)):
    plt.figure(figsize=(20,12))
    for fold, col in enumerate(df_trn_mdl[i].filter(regex=r'fold').columns):            
        col_acc = 'acc_' + str(fold+1)
        df_fi = df_trn_mdl[i].sort_values(by=col, ascending=False).reset_index().iloc[:25]
        df_fi = df_fi[['index', col, col_acc]]
        df_fi.columns = ['Feature', 'score', col_acc]
        plt.subplot(2,3, fold+1)
        sns.barplot(x='score', y='Feature', data=df_fi)    
        plt.title('Fold {} - score: {:2.5f}'.format(fold+1, df_fi[col_acc].mean()), 
                  fontdict={'fontsize':18})    
    
    plt.suptitle('Feature Importance XGB - {}'.format(scaler_list[i]), y=1.05, fontsize=24);
    plt.tight_layout(h_pad=3.0);   


# ## 3.4. Gerar PCA
# Nesta etapa vamos utilizar a PCA para gerar novas variáveis para os modelos.
# 

# In[56]:


df3_4_train = df3_3_train.copy()
df3_4_test  = df3_3_test.copy()


# In[57]:


feature_pca     = feature_float
pca             = PCA(random_state=12359)
df3_4_train_pca = pca.fit_transform(df3_4_train[feature_pca])

skplt.decomposition.plot_pca_component_variance(pca, figsize=(8,6));


# In[58]:


features = range(pca.n_components_)

plt.figure(figsize=(8,4))
plt.bar(features[:15], pca.explained_variance_[:15], color='lightskyblue')
plt.xlabel('PCA feature')
plt.ylabel('Variance')
plt.xticks(features[:15])
plt.show()


# <div class="alert alert-info" role="alert">
#     
# **`NOTA:`** <br>
# Como podemos observar acima, o processo de PCA 27 componentes que repesentam 75% da variabilidade dos dados, sendo que vamos utilizar apenas as duas primeiras componentes, isso é, vamos cria duas novas variáveis. 
#     
# </div>

# In[59]:


n_components  = 4
pca           = PCA(n_components=n_components, random_state=123)
pca_feats     = [f'fe_pca_{i}' for i in range(n_components)]

df3_4_train[pca_feats] = pd.DataFrame(pca.fit_transform(df3_4_train[feature_float]), columns=pca_feats)
df3_4_test[pca_feats]  = pd.DataFrame(pca.transform(df3_4_test[feature_float]), columns=pca_feats)


# In[60]:


df3_4_train.filter(regex=r'fe').head().info()


# ## 3.4.1. Modelagem

# In[61]:


lb     = LabelEncoder()
X      = df3_4_train.drop(target, axis=1)
y      = pd.DataFrame(lb.fit_transform(df3_4_train[target]), columns=[target])
X_test = df3_4_test


# ### 3.4.2. Validação cruzada

# In[62]:


get_ipython().run_cell_magic('time', '', '\nseed_       = 12359\nmdl         = []\ndf_trn_mdl  = []\ndf_fe_imp   = []\nscaler_list = [None]\n\nparams = {"objective"    : \'multi:softmax\',    \n          \'eval_metric\'  : \'mlogloss\',  \n          \'n_estimators\' : 1000,\n          \'random_state\' : seed_}\n\nif torch.cuda.is_available():           \n    params.update({\'tree_method\': \'gpu_hist\', \'predictor\': \'gpu_predictor\'})\n    \nfor sc in scaler_list:    \n    model, df_trn, df_feature_imp, df_preds_prob = \\\n    model_train_cv_fit(model_        = xgb.XGBClassifier(**params),\n                       model_name_   = \'xgb_fe_score_04_{:2.5f}\',\n                       X_            = X,\n                       y_            = y,\n                       X_test_       = X_test,\n                       lb_           = lb,\n                       target_       = target,\n                       sc_           = RobustScaler(), \n                       sc_second_    = sc,\n                       n_splits_     = 5,\n                       seed_         = seed_,\n                       save_sub_     = True,\n                       path_         = \'\', \n                       save_predict_ = True)\n    \n    mdl.append(model)\n    df_trn_mdl.append(df_trn)\n    df_fe_imp.append(df_feature_imp)\n\ndel model, df_trn, df_feature_imp\n')


# <div class="alert alert-info" role="alert">
#     
# **`NOTA:`** <br>
# Tivemos um pequeno aumento com a criação das duas variaveis.  
#     
# </div>

# In[63]:


for i in range(len(df_trn_mdl)):
    plt.figure(figsize=(20,12))
    for fold, col in enumerate(df_trn_mdl[i].filter(regex=r'fold').columns):            
        col_acc = 'acc_' + str(fold+1)
        df_fi = df_trn_mdl[i].sort_values(by=col, ascending=False).reset_index().iloc[:25]
        df_fi = df_fi[['index', col, col_acc]]
        df_fi.columns = ['Feature', 'score', col_acc]
        plt.subplot(2,3, fold+1)
        sns.barplot(x='score', y='Feature', data=df_fi)    
        plt.title('Fold {} - score: {:2.5f}'.format(fold+1, df_fi[col_acc].mean()), 
                  fontdict={'fontsize':18})    
    
    plt.suptitle('Feature Importance XGB - {}'.format(scaler_list[i]), y=1.05, fontsize=24);
    plt.tight_layout(h_pad=3.0);   


# ## 3.5. Clustering

# In[64]:


df3_5_train = df3_4_train.copy()
df3_5_test  = df3_4_test.copy()

sc = StandardScaler()

df3_5_train_scaler = sc.fit_transform(df3_5_train[feature_float]) 
df3_5_test_scaler  = sc.transform(df3_5_test[feature_float]) 


# In[65]:


get_ipython().run_cell_magic('time', '', 'plt.figure(figsize=(12, 7))\nvisualizer_1 = KElbowVisualizer(KMeans(random_state=12359), k=(2,10))\nvisualizer_1.fit(df3_5_train_scaler)\nvisualizer_1.poof();\n')


# In[66]:


model_kmeans = KMeans(n_clusters=7, random_state=12359)
model_kmeans.fit(df3_5_train_scaler);

clusters_train = model_kmeans.predict(df3_5_train_scaler)
clusters_test  = model_kmeans.predict(df3_5_test_scaler)

df3_5_train['fe_cluster'] = clusters_train
df3_5_test['fe_cluster']  = clusters_test

#del df3_5_train_scaler, df3_5_test_scaler

df3_5_train.shape, df3_5_test.shape


# In[67]:


df3_5_train = pd.get_dummies(df3_5_train, columns=['fe_cluster'])
df3_5_test  = pd.get_dummies(df3_5_test, columns=['fe_cluster'])

df3_5_train.drop('fe_cluster_6', axis=1, inplace=True)
df3_5_test.drop('fe_cluster_6', axis=1, inplace=True)

df3_5_train.shape, df3_5_test.shape


# In[68]:


df3_5_train.filter(regex=r'fe').head().info()


# In[69]:


df3_5_train = reduce_memory_usage(df3_5_train)
df3_5_test  = reduce_memory_usage(df3_5_test)


# In[70]:


missing_zero_values_table(df3_5_train)


# In[71]:


jb.dump(df3_5_train, 'Data/pkl/df2_nb_02_train.pkl.z')
jb.dump(df3_5_test,  'Data/pkl/df2_nb_02_test.pkl.z')

gc.collect()


# # <div class="alert alert-success"> 6. Split Train/Test </div>

# In[72]:


df6_train     = jb.load('Data/pkl/df2_nb_02_train.pkl.z')
df6_test      = jb.load('Data/pkl/df2_nb_02_test.pkl.z')

df6_train.shape, df6_test.shape


# In[73]:


lb     = LabelEncoder()
X      = df6_train.drop(target, axis=1)
y      = pd.DataFrame(lb.fit_transform(df6_train[target]), columns=[target])
X_test = df6_test

X.shape , X_test.shape


# # <div class="alert alert-success"> 7. Modelagem </div>

# In[74]:


get_ipython().run_cell_magic('time', '', '\nseed_       = 12359\nmdl         = []\ndf_trn_mdl  = []\ndf_fe_imp   = []\nscaler_list = [None]\n\nparams = {"objective"    : \'multi:softmax\',    \n          \'eval_metric\'  : \'mlogloss\',       \n          \'n_estimators\' : 1000,\n          \'random_state\' : seed_}\n\nif torch.cuda.is_available():           \n    params.update({\'tree_method\': \'gpu_hist\', \'predictor\': \'gpu_predictor\'})\n    \nfor sc in scaler_list:    \n    model, df_trn, df_feature_imp, df_preds_prob = \\\n    model_train_cv_fit(model_        = xgb.XGBClassifier(**params),\n                       model_name_   = \'xgb_fe_score_05F_{:2.5f}\',\n                       X_            = X,\n                       y_            = y,\n                       X_test_       = X_test,\n                       lb_           = lb,\n                       target_       = target,\n                       sc_           = RobustScaler(), \n                       sc_second_    = sc,\n                       n_splits_     = 5,\n                       seed_         = seed_,\n                       save_sub_     = True,\n                       path_         = \'\', \n                      save_predict_  = True)\n\n    mdl.append(model)\n    df_trn_mdl.append(df_trn)\n    df_fe_imp.append(df_feature_imp)\n\ndel model, df_trn, df_feature_imp\n')


# <div class="alert alert-info" role="alert">
#     
# **`NOTA:`** <br>
# 
#     
# </div>

# ## 7.1. Feature Importances  CV 

# In[75]:


for i in range(len(df_trn_mdl)):
    plt.figure(figsize=(20,12))
    for fold, col in enumerate(df_trn_mdl[i].filter(regex=r'fold').columns):            
        col_acc = 'acc_' + str(fold+1)
        df_fi = df_trn_mdl[i].sort_values(by=col, ascending=False).reset_index().iloc[:25]
        df_fi = df_fi[['index', col, col_acc]]
        df_fi.columns = ['Feature', 'score', col_acc]
        plt.subplot(2,3, fold+1)
        sns.barplot(x='score', y='Feature', data=df_fi)    
        plt.title('Fold {} - score: {:2.5f}'.format(fold+1, df_fi[col_acc].mean()), 
                  fontdict={'fontsize':18})    
    
    plt.suptitle('Feature Importance XGB - {}'.format(scaler_list[i]), y=1.05, fontsize=24);
    plt.tight_layout(h_pad=3.0);   


# # <div class="alert alert-success"> 5. Conclusão </div>
# 
# <div class="alert alert-info" role="alert">    
# Neste notebook criamos novas variáveis utilizando a clusterização e variáveis estatísticas, com a finalidade de ajudar os modelos a identificar padrões no dados para melhora as previsões. <br>
#     
# <br> 
#     
# 
# <br>
#     
# </div>
