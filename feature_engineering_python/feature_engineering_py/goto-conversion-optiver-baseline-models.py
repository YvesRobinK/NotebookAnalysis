#!/usr/bin/env python
# coding: utf-8

# The novelty I have added is implementation of the goto_conversion (described in the link below). The rest of the work is done by @ravi20076.
# https://github.com/gotoConversion/goto_conversion/

# The favourite-longshot bias is not limited to gambling markets, it exists in stock markets too. Thus, we applied the original goto_conversion to stock markets by defining the zero_sum variant. Under the same philosophy as the original goto_conversion, zero_sum adjusts all predicted stock prices (e.g. weighted average price) by the same units of standard error to ensure all predicted stock prices relative to the index price (e.g. weighted average nasdaq price) sum to zero. This attempts to consider the favourite-longshot bias by utilising the wider standard errors implied for predicted stock prices with low trade volume and vice-versa.

# In[1]:


#Install and import goto_conversion
#%pip install goto-conversion
#import goto_conversion

#Installation fails, so we will copy and paste the __init__.py file for now
#source: https://github.com/gotoConversion/goto_conversion/blob/main/goto_conversion/__init__.py
def goto_conversion(listOfOdds, total = 1, eps = 1e-6, isAmericanOdds = False):

    #Convert American Odds to Decimal Odds
    if isAmericanOdds:
        for i in range(len(listOfOdds)):
            currOdds = listOfOdds[i]
            isNegativeAmericanOdds = currOdds < 0
            if isNegativeAmericanOdds:
                currDecimalOdds = 1 + (100/(currOdds*-1))
            else: #Is non-negative American Odds
                currDecimalOdds = 1 + (currOdds/100)
            listOfOdds[i] = currDecimalOdds

    #Error Catchers
    if len(listOfOdds) < 2:
        raise ValueError('len(listOfOdds) must be >= 2')
    if any(x < 1 for x in listOfOdds):
        raise ValueError('All odds must be >= 1, set isAmericanOdds parameter to True if using American Odds')

    #Computation
    listOfProbabilities = [1/x for x in listOfOdds] #initialize probabilities using inverse odds
    listOfSe = [pow((x-x**2)/x,0.5) for x in listOfProbabilities] #compute the standard error (SE) for each probability
    step = (sum(listOfProbabilities) - total)/sum(listOfSe) #compute how many steps of SE the probabilities should step back by
    outputListOfProbabilities = [min(max(x - (y*step),eps),1) for x,y in zip(listOfProbabilities, listOfSe)]
    return outputListOfProbabilities

def zero_sum(listOfPrices, listOfVolumes):
    listOfSe = [x**0.5 for x in listOfVolumes] #compute standard errors assuming standard deviation is same for all stocks
    step = sum(listOfPrices)/sum(listOfSe)
    outputListOfPrices = [x - (y*step) for x,y in zip(listOfPrices, listOfSe)]
    return outputListOfPrices


# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:black; font-size:120%; text-align:left;padding:3.0px; background: #cceeff; border-bottom: 8px solid #004466" > TABLE OF CONTENTS<br><div>  
# * [IMPORTS](#1)
# * [INTRODUCTION](#2)
# * [DATA PROCESSING](#3)
# * [MODEL TRAINING](#4) 
# * [MODEL INFERENCING](#5) 
# * [OUTRO](#6)  
#  

# <a id="1"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #003380; border-bottom: 10px solid #80ffff"> PACKAGE IMPORTS<br><div> 

# In[2]:


get_ipython().run_cell_magic('time', '', '\n# General library imports:-\nfrom IPython.display import display_html, clear_output, Markdown;\nfrom gc import collect;\n\nfrom copy import deepcopy;\nimport pandas as pd;\nimport numpy as np;\nimport joblib;\nfrom os import system, getpid, walk;\nfrom psutil import Process;\nimport ctypes;\nlibc = ctypes.CDLL("libc.so.6");\n\nfrom pprint import pprint;\nfrom colorama import Fore, Style, init;\nfrom warnings import filterwarnings;\nfilterwarnings(\'ignore\');\n\nfrom tqdm.notebook import tqdm;\n\nprint();\ncollect();\n')


# In[3]:


get_ipython().run_cell_magic('time', '', '\n# Model development:-\nfrom sklearn.model_selection import (RepeatedStratifiedKFold as RSKF, \n                                     StratifiedKFold as SKF,\n                                     KFold, \n                                     RepeatedKFold as RKF, \n                                     cross_val_score);\n\nfrom lightgbm import log_evaluation, early_stopping, LGBMRegressor as LGBMR;\nfrom xgboost import XGBRegressor as XGBR;\nfrom catboost import CatBoostRegressor as CBR;\nfrom sklearn.ensemble import HistGradientBoostingRegressor as HGBR;\nfrom sklearn.metrics import mean_absolute_error as mae, make_scorer;\n\nprint();\ncollect();\n')


# In[4]:


get_ipython().run_cell_magic('time', '', '\n# Defining global configurations and functions:-\n\n# Color printing    \ndef PrintColor(text:str, color = Fore.BLUE, style = Style.BRIGHT):\n    "Prints color outputs using colorama using a text F-string";\n    print(style + color + text + Style.RESET_ALL); \n    \ndef GetMemUsage():\n    """\n    This function defines the memory usage across the kernel. \n    Source-\n    https://stackoverflow.com/questions/61366458/how-to-find-memory-usage-of-kaggle-notebook\n    """;\n    \n    pid = getpid();\n    py = Process(pid);\n    memory_use = py.memory_info()[0] / 2. ** 30;\n    return f"RAM memory GB usage = {memory_use :.4}";\n\n# Making sklearn pipeline outputs as dataframe:-\nfrom sklearn import set_config; \nset_config(transform_output = "pandas");\npd.set_option(\'display.max_columns\', 50);\npd.set_option(\'display.max_rows\', 50);\n\nprint();\ncollect();\n')


# <a id="2"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #003380; border-bottom: 10px solid #80ffff"> INTRODUCTION<br><div> 

# <div class="alert alert-block alert-info" style = "font-family: Cambria Math;font-size: 115%; color: black; background-color: #e6f9ff; border: dashed black 1.0px; padding: 3.5px" >
# 1. This notebook is my first tryst with the Optiver challenge. This is a time series regression problem involving stock market trading data at the day's close auction book. <b>Mean Absolute Error metric</b> is used here <br>
# 2. This notebook aims to train a baseline model using a simple CV strategy from the memory reduced datasets created for the challenge. <br>
# 3. This is a continuation from my baseline data curation notebook and dataset. We continue the analysis herewith and train models to elicit a CV score. We then infer using these models here and make a submission<br>
# </div>

# <a id="2.1"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size: 90%; text-align:left;padding:4.0px; background: maroon; border-bottom: 5px solid black"> VERSION DETAILS<br><div> 

# | Version<br>Number | Version<br>Details | Preparation <br> date|LGBMR <br> CV|CBR <br> CV| XGBR <br> CV| HGBR <br> CV|Best LB <br>score|Single/<br> Ensemble|
# | :-: | --- | :-: |  :-: |:-: |:-: |:-: |:-: |:-: |
# |V1| * Baseline features <br> * No null treatments and scaling <br> * Simple ML models without tuning <br> * 5x1 K-fold CV <br> * Simple weighted ensemble| 22Sep2023|6.248286|6.25538|6.27198|6.266826|5.3702| Ensemble <br> LGBMR CBR|
# |V2| * Baseline features <br> * No null treatments and scaling <br> * Simple ML models without tuning with altered parameters <br> * 5x1 K-fold CV <br> * Simple weighted ensemble| 23Sep2023|6.23334|6.2535|||5.3728| Ensemble <br> LGBMR CBR|
# |V3| * Baseline features <br> * No null treatments and scaling <br> * ML models with V1 parameters <br> * 5x3 Repeated K-fold CV <br> * Simple weighted ensemble| 24Sep2023|6.248288| 6.25532||6.267036|5.3712|Ensemble <br> LGBMR CBR |

# <a id="2.2"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size: 90%; text-align:left;padding:4.0px; background: maroon; border-bottom: 5px solid black"> CONFIGURATION PARAMETERS<br><div> 

# | Parameter | Comments | Sample values|
# | :-: | --- | :-: |
# |version_nb | Version Number| integer value|
# |test_req| Are we testing the code?| Y/N|
# |test_frac| Test fraction for sampling and testing <br> Place small values for easy execution| float between 0 and 1|
# |load_tr_data| Are we loading the train data here? <br> If we are inferring only, this is not required | Y/N|
# |gpu_switch| Do we need a GPU here? |Y/N|
# |state| Random seed| integer|
# |target| Target column name| string value|
# |path| Data path for model training <br> I point this to my baseline data curation kernel| |
# |test_path| Relevant path for test data| Competition artefacts|
# |df_choice| Which data do I need for analysis? <br> Refer the baseline data prep kernel for details ||
# |mdl_path| Path to dump trained models with joblib||
# |inf_path| Appropriate path to extract the models for inference <br> I point to my baseline dataset with models trained as a starter||
# |methods| All trained model methods, choose 1-more based on the memory constraints <br> For inferencing, all trained methods need to be present|list |
# |ML| Do we need to do model training here? |Y/N |
# |n_splits| CV number of splits |integer value|
# |n_repeats| CV number of repetitions |integer value|
# |nbrnd_erly_stp| Number of early stopping rounds|integer value|
# |mdlcv_mthd| Model CV choice |KF, SKF, RSKF, RKF|
# |ensemble_req| Do we need an ensemble here? <br> Currently this is unused |Y/N|
# |enscv_mthd| Ensemble CV choice- used mostly with Optuna |KF, SKF, RSKF, RKF|
# |metric_obj| Based on the metric, do we wish to maximize/ minimize the function? |maximize/ minimize|
# |ntrials| Number of Optuna trials |integer value|
# |ens_weights| Weights if decided subjecively |list<br> apropos to number of trained methods|
# |inference_req| Do we need to infer here? |Y/N|

# In[5]:


get_ipython().run_cell_magic('time', '', '\n# Configuration class:-\nclass CFG:\n    """\n    Configuration class for parameters and CV strategy for tuning and training\n    Please use caps lock capital letters while filling in parameters\n    """;\n    \n    # Data preparation:-   \n    version_nb         = 1;\n    test_req           = "N";\n    test_frac          = 0.01;\n    load_tr_data       = "N";\n    gpu_switch         = "OFF"; \n    state              = 42;\n    target             = \'target\';\n    \n    path               = f"/kaggle/input/optiver-memoryreduction/";\n    test_path          = f"/kaggle/input/optiver-trading-at-the-close/example_test_files/test.csv";\n    df_choice          = f"XTrIntCmpNewFtre.parquet";\n    mdl_path           = f\'/kaggle/working/BaselineML/\';\n    inf_path           = f\'/kaggle/input/optiverbaselinemodels/\';\n     \n    # Model Training:-\n    methods            = ["LGBMR", "CBR"];\n    ML                 = "N";\n    n_splits           = 5;\n    n_repeats          = 1;\n    nbrnd_erly_stp     = 100 ;\n    mdlcv_mthd         = \'KF\';\n    \n    # Ensemble:-    \n    ensemble_req       = "N";\n    enscv_mthd         = "KF";\n    metric_obj         = \'minimize\';\n    ntrials            = 10 if test_req == "Y" else 200;\n    ens_weights        = [0.4, 0.6];\n    \n    # Inference:-\n    inference_req      = "Y";\n    \n    # Global variables for plotting:-\n    grid_specs = {\'visible\': True, \'which\': \'both\', \'linestyle\': \'--\', \n                  \'color\': \'lightgrey\', \'linewidth\': 0.75\n                 };\n    title_specs = {\'fontsize\': 9, \'fontweight\': \'bold\', \'color\': \'tab:blue\'};\n\nprint();\nPrintColor(f"--> Configuration done!\\n");\ncollect();\n\nPrintColor(f"\\n" + GetMemUsage(), color = Fore.RED);\n')


# In[6]:


get_ipython().run_cell_magic('time', '', '\n# Commonly used CV strategies for later usage:-\nall_cv= {\'KF\'  : KFold(n_splits= CFG.n_splits, shuffle = True, random_state= CFG.state),\n         \'RKF\' : RKF(n_splits= CFG.n_splits, n_repeats = CFG.n_repeats, random_state= CFG.state),\n         \'RSKF\': RSKF(n_splits= CFG.n_splits, n_repeats = CFG.n_repeats, random_state= CFG.state),\n         \'SKF\' : SKF(n_splits= CFG.n_splits, shuffle = True, random_state= CFG.state)\n        };\n\n# Defining the competition metric:-\ndef ScoreMetric(ytrue, ypred)-> float:\n    """\n    This function calculates the metric for the competition. \n    ytrue- ground truth array\n    ypred- predictions\n    returns - metric value (float)\n    """;\n    \n    return mae(ytrue, ypred);\n\n# Designing a custom scorer to use in cross_val_predict and cross_val_score:-\nmyscorer = make_scorer(ScoreMetric, greater_is_better = False, needs_proba=False,);\n\nprint();\ncollect();\n\nPrintColor(f"\\n" + GetMemUsage(), color = Fore.RED);\n')


# <div class="alert alert-block alert-info" style = "font-family: Cambria Math;font-size: 115%; color: black; background-color: #e6f9ff; border: dashed black 1.0px; padding: 3.5px" >
# <b>How to use these kernels</b> <br>
# 1. Use the memory reduction kernel input to curate features, reduce dataset memory and prepare essential datasets as input to this kernel. Else, use the starter dataset as input if features are already ready. Links are provided below. <br>
# <b> Baseline input features:-</b> https://www.kaggle.com/code/ravi20076/optiver-memoryreduction<br>
# <b> Baseline input dataset:-</b> https://www.kaggle.com/datasets/ravi20076/optiver-memoryreduceddatasets<br>
# 2. Design one's model framework here for a baseline and train models. It is advisable to train 1/2 models at a time to prevent memory overflow issues <br>
# 3. Store the model objects in the BaselineML directory in the working folder for inferencing <br>
# 4. It is advisable to infer and submit separately. This will surely not create a data memory overlow problem. In this case, please turn off training and do not load the training dataset. In this case, I have stored the model training artefacts in the link- https://www.kaggle.com/datasets/ravi20076/optiverbaselinemodels <br>
# 5.While inferencing, make sure to curate the same features as used in the training process. I shall make an improvement here and update the kernel shortly. <br>
# </div>

# <a id="3"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #003380; border-bottom: 10px solid #80ffff"> DATA PROCESSING<br><div> 

# <div class="alert alert-block alert-info" style = "font-family: Cambria Math;font-size: 115%; color: black; background-color: #e6f9ff; border: dashed black 1.0px; padding: 3.5px" >
# In this version, we choose the int-float compressed dataset with new features as per the reference notebook <br>
# </div>

# In[7]:


get_ipython().run_cell_magic('time', '', '\nif (CFG.load_tr_data == "Y" or CFG.ML == "Y") and CFG.test_req == "Y":\n    if isinstance(CFG.test_frac, float):\n        X = pd.read_parquet(CFG.path + CFG.df_choice).sample(frac = CFG.test_frac);\n    else:\n        X = pd.read_parquet(CFG.path + CFG.df_choice).sample(n = CFG.test_frac);\n        \n    y = pd.read_parquet(CFG.path + f"Ytrain.parquet").loc[X.index].squeeze();\n    PrintColor(f"---> Sampled train shapes for code testing = {X.shape} {y.shape}", \n               color = Fore.RED);\n    X.index, y.index = range(len(X)), range(len(y));\n\nelif CFG.load_tr_data == "Y" or CFG.ML == "Y":\n    X = pd.read_parquet(CFG.path + CFG.df_choice);\n    y = pd.read_parquet(CFG.path + f"Ytrain.parquet").squeeze();  \n    PrintColor(f"---> Train shapes for code testing = {X.shape} {y.shape}");\n\nelif CFG.load_tr_data != "Y" or CFG.inference_req == "Y":\n    PrintColor(f"---> Train data is not required as we are infering from the model");\n    \nprint();\ncollect();\nlibc.malloc_trim(0);\n\nPrintColor(f"\\n" + GetMemUsage(), color = Fore.RED);\n')


# <a id="4"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #003380; border-bottom: 10px solid #80ffff"> MODEL TRAINING AND CV<br><div> 

# In[8]:


get_ipython().run_cell_magic('time', '', '\n# Initializing model I-O:-\n\nif CFG.ML == "Y":\n    Mdl_Master = \\\n    {\'CBR\': CBR(**{\'task_type\'           : "GPU" if CFG.gpu_switch == "ON" else "CPU",\n                   \'objective\'           : "MAE",\n                   \'eval_metric\'         : "MAE",\n                   \'bagging_temperature\' : 0.5,\n                   \'colsample_bylevel\'   : 0.7,\n                   \'iterations\'          : 500,\n                   \'learning_rate\'       : 0.065,\n                   \'od_wait\'             : 25,\n                   \'max_depth\'           : 7,\n                   \'l2_leaf_reg\'         : 1.5,\n                   \'min_data_in_leaf\'    : 1000,\n                   \'random_strength\'     : 0.65, \n                   \'verbose\'             : 0,\n                   \'use_best_model\'      : True,\n                  }\n               ), \n\n      \'LGBMR\': LGBMR(**{\'device\'            : "gpu" if CFG.gpu_switch == "ON" else "cpu",\n                        \'objective\'         : \'regression_l1\',\n                        \'boosting_type\'     : \'gbdt\',\n                        \'random_state\'      : CFG.state,\n                        \'colsample_bytree\'  : 0.7,\n                        \'subsample\'         : 0.65,\n                        \'learning_rate\'     : 0.065,\n                        \'max_depth\'         : 6,\n                        \'n_estimators\'      : 500,\n                        \'num_leaves\'        : 150,  \n                        \'reg_alpha\'         : 0.01,\n                        \'reg_lambda\'        : 3.25,\n                        \'verbose\'           : -1,\n                       }\n                    ),\n\n      \'XGBR\': XGBR(**{\'tree_method\'        : "gpu_hist" if CFG.gpu_switch == "ON" else "hist",\n                      \'objective\'          : \'reg:absoluteerror\',\n                      \'random_state\'       : CFG.state,\n                      \'colsample_bytree\'   : 0.7,\n                      \'learning_rate\'      : 0.07,\n                      \'max_depth\'          : 6,\n                      \'n_estimators\'       : 500,                         \n                      \'reg_alpha\'          : 0.025,\n                      \'reg_lambda\'         : 1.75,\n                      \'min_child_weight\'   : 1000,\n                      \'early_stopping_rounds\' : CFG.nbrnd_erly_stp,\n                     }\n                  ),\n\n      "HGBR" : HGBR(loss              = \'squared_error\',\n                    learning_rate     = 0.075,\n                    early_stopping    = True,\n                    max_iter          = 200,\n                    max_depth         = 6,\n                    min_samples_leaf  = 1500,\n                    l2_regularization = 1.75,\n                    scoring           = myscorer,\n                    random_state      = CFG.state,\n                   )\n    };\n\nprint();\ncollect();\n\nPrintColor(f"\\n" + GetMemUsage(), color = Fore.RED);\n')


# In[9]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ML == "Y":\n    # Initializing the models from configuration class:-\n    methods = CFG.methods;\n\n    # Initializing a folder to store the trained and fitted models:-\n    system(\'mkdir BaselineML\');\n\n    # Initializing the model path for storage:-\n    model_path = CFG.mdl_path;\n\n    # Initializing the cv object:-\n    cv = all_cv[CFG.mdlcv_mthd];\n        \n    # Initializing score dataframe:-\n    Scores = pd.DataFrame(index = range(CFG.n_splits * CFG.n_repeats),\n                          columns = methods).fillna(0).astype(np.float32);\n    \n    FtreImp = pd.DataFrame(index = X.columns, columns = [methods]).fillna(0);\n\nprint();\ncollect();\nlibc.malloc_trim(0);\n\nPrintColor(f"\\n" + GetMemUsage(), color = Fore.RED);\n')


# In[10]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ML == "Y":\n    PrintColor(f"\\n{\'=\' * 25} ML Training {\'=\' * 25}\\n");\n    \n    # Initializing CV splitting:-       \n    for fold_nb, (train_idx, dev_idx) in tqdm(enumerate(cv.split(X, y)), \n                                              f"{CFG.mdlcv_mthd} CV {CFG.n_splits}x{CFG.n_repeats}"\n                                             ): \n        # Creating the cv folds:-    \n        Xtr  = X.iloc[train_idx];   \n        Xdev = X.iloc[dev_idx];\n        ytr  = y.iloc[train_idx];\n        ydev = y.iloc[dev_idx];\n        \n        PrintColor(f"-------> Fold{fold_nb} <-------");\n        # Fitting the models:- \n        for method in methods:\n            model = Mdl_Master[method];\n            if method == "LGBMR":\n                model.fit(Xtr, ytr, \n                          eval_set = [(Xdev, ydev)], \n                          verbose = 0, \n                          eval_metric = "mae",\n                          callbacks = [log_evaluation(0,), \n                                       early_stopping(CFG.nbrnd_erly_stp, verbose = False)], \n                         );\n\n            elif method == "XGBR":\n                model.fit(Xtr, ytr, \n                          eval_set = [(Xdev, ydev)], \n                          verbose = 0, \n                          eval_metric = "mae",\n                         );  \n\n            elif method == "CBR":\n                model.fit(Xtr, ytr, \n                          eval_set = [(Xdev, ydev)], \n                          verbose = 0, \n                          early_stopping_rounds = CFG.nbrnd_erly_stp,\n                         ); \n\n            else:\n                model.fit(Xtr, ytr);\n\n            #  Saving the model for later usage:-\n            joblib.dump(model, CFG.mdl_path + f\'{method}V{CFG.version_nb}Fold{fold_nb}.model\');\n            \n            # Creating OOF scores:-\n            score = ScoreMetric(ydev, model.predict(Xdev));\n            Scores.at[fold_nb, method] = score;\n            num_space = 6- len(method);\n            PrintColor(f"---> {method} {\' \'* num_space} OOF = {score:.5f}", \n                       color = Fore.MAGENTA);  \n            del num_space, score;\n            \n            # Collecting feature importances:-\n            FtreImp[method] = \\\n            FtreImp[method].values + (model.feature_importances_ / (CFG.n_splits * CFG.n_repeats));\n            collect();\n            \n        PrintColor(GetMemUsage());\n        print();\n        del Xtr, ytr, Xdev, ydev;\n        collect();\n    \n    clear_output();\n    PrintColor(f"\\n---> OOF scores across methods <---\\n");\n    Scores.index.name = "FoldNb";\n    Scores.index = Scores.index + 1;\n    display(Scores.style.format(precision = 5).\\\n            background_gradient(cmap = "Pastel1")\n           );\n    \n    PrintColor(f"\\n---> Mean OOF scores across methods <---\\n");\n    display(Scores.mean());\n    \n    FtreImp.to_csv(CFG.mdl_path + f"FtreImp_V{CFG.version_nb}.csv");\n    \ncollect();\nprint();\nlibc.malloc_trim(0);\n\nPrintColor(f"\\n" + GetMemUsage(), color = Fore.GREEN);\n')


# <a id="5"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #003380; border-bottom: 10px solid #80ffff"> MODEL INFERENCING AND SUBMISSION<br><div> 

# In[11]:


get_ipython().run_cell_magic('time', '', '\ndef MakeFtre(df : pd.DataFrame, prices: list) -> pd.DataFrame:\n    """\n    This function creates new features using the price columns. This was used in a baseline notebook as below-\n    https://www.kaggle.com/code/yuanzhezhou/baseline-lgb-xgb-and-catboost\n    \n    Inputs-\n    df:- pd.DataFrame -- input dataframe\n    cols:- price columns for transformation\n    \n    Returns-\n    df:- pd.DataFrame -- dataframe with extra columns\n    """;\n    \n    features = [\'seconds_in_bucket\', \'imbalance_buy_sell_flag\',\n               \'imbalance_size\', \'matched_size\', \'bid_size\', \'ask_size\',\n                \'reference_price\',\'far_price\', \'near_price\', \'ask_price\', \'bid_price\', \'wap\',\n                \'imb_s1\', \'imb_s2\'\n               ];\n    \n    df[\'imb_s1\'] = df.eval(\'(bid_size-ask_size)/(bid_size+ask_size)\').astype(np.float32);\n    df[\'imb_s2\'] = df.eval(\'(imbalance_size-matched_size)/(matched_size+imbalance_size)\').astype(np.float32);\n       \n    for i,a in enumerate(prices):\n        for j,b in enumerate(prices):\n            if i>j:\n                df[f\'{a}_{b}_imb\'] = df.eval(f\'({a}-{b})/({a}+{b})\');\n                features.append(f\'{a}_{b}_imb\'); \n                    \n    for i,a in enumerate(prices):\n        for j,b in enumerate(prices):\n            for k,c in enumerate(prices):\n                if i>j and j>k:\n                    max_ = df[[a,b,c]].max(axis=1);\n                    min_ = df[[a,b,c]].min(axis=1);\n                    mid_ = df[[a,b,c]].sum(axis=1)-min_-max_;\n\n                    df[f\'{a}_{b}_{c}_imb2\'] = ((max_-mid_)/(mid_-min_)).astype(np.float32);\n                    features.append(f\'{a}_{b}_{c}_imb2\');\n    \n    return df[features];\n\nprint();\ncollect();\n')


# In[12]:


get_ipython().run_cell_magic('time', '', '\n# Creating the testing environment:-\nif CFG.inference_req == "Y":\n    try: \n        del X, y;\n    except: \n        pass;\n        \n    prices = [\'reference_price\', \'far_price\', \'near_price\', \'bid_price\', \'ask_price\', \'wap\'];\n    \n    # Making the test environment for inferencing:-\n    import optiver2023;\n    try: \n        env = optiver2023.make_env();\n        iter_test = env.iter_test();\n        PrintColor(f"\\n---> Curating the inference environment");\n    except: \n        pass;\n    \n    # Collating a list of models to be used for inferencing:-\n    models = [];\n\n    # Loading the models for inferencing:-\n    if CFG.ML != "Y": \n        model_path = CFG.inf_path;\n        PrintColor(f"---> Loading models from the input data for the kernel\\n");\n    elif CFG.ML == "Y": \n        model_path = CFG.mdl_path;\n        PrintColor(f"---> Loading models from the working directory for the kernel\\n");\n    \n    # Loading the models from the models dataframe:-\n    mdl_lbl = [];\n    for _, _, filename in walk(model_path):\n        mdl_lbl.extend(filename);\n\n    models = [];\n    for filename in mdl_lbl:\n        models.append(joblib.load(model_path + f"{filename}"));\n        \n    mdl_lbl    = [m.replace(r".model", "") for m in mdl_lbl];\n    model_dict = {l:m for l,m in zip(mdl_lbl, models)};\n    PrintColor(f"\\n---> Trained models\\n");    \n    pprint(np.array(mdl_lbl), width = 100, indent = 10, depth = 1);  \n       \nprint();\ncollect();  \nlibc.malloc_trim(0);\nPrintColor(f"\\n" + GetMemUsage(), color = Fore.RED); \n')


# In[13]:


get_ipython().run_cell_magic('time', '', '\nif CFG.inference_req == "Y":\n    print();\n    counter = 0;\n    \n    for test, revealed_targets, sample_prediction in iter_test:\n        PrintColor(f"{counter + 1}. Inference", color = Fore.MAGENTA);\n        Xtest = MakeFtre(test, prices = prices);\n        \n        # Curating model predictions across methods and folds:-        \n        preds = pd.DataFrame(columns = CFG.methods, index = Xtest.index).fillna(0);\n        for method in CFG.methods:\n            for mdl_lbl, mdl in model_dict.items():\n                if mdl_lbl.startswith(f"{method}Fold"):\n                    print(mdl_lbl);\n                    preds[method] = preds[method] + mdl.predict(Xtest)/ (CFG.n_splits * CFG.n_repeats);\n        \n        # Curating the weighted average model predictions:-       \n        sample_prediction[\'target\'] = \\\n        np.average(preds.values, weights= CFG.ens_weights, axis=1);\n        \n        #My Novelty\n        sample_prediction[\'target\'] = zero_sum(sample_prediction[\'target\'], test.loc[:,\'bid_size\'] + test.loc[:,\'ask_size\'])\n        \n        try: \n            env.predict(sample_prediction);\n        except: \n            PrintColor(f"---> Submission did not happen as we have the file already");\n            pass;\n        \n        counter = counter+1;\n        collect();\n        \n    PrintColor(f"\\n---> Submission file\\n");\n    display(sample_prediction.head(10));\n            \nprint();\ncollect();  \nlibc.malloc_trim(0);\nPrintColor(f"\\n" + GetMemUsage(), color = Fore.RED); \n')


# <a id="6"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #003380; border-bottom: 10px solid #80ffff"> OUTRO<br><div> 

# <div class="alert alert-block alert-info" style = "font-family: Cambria Math;font-size: 115%; color: black; background-color: #e6f9ff; border: dashed black 1.0px; padding: 3.5px" >
# <b>Next steps</b> <br>
# 1. Better feature engineering. I shall perform a detailed EDA next <br>
# 2. Exploring better models and ensemble strategy <br>
# 3. Purging redundant features from the existing list of features <br>
# 4. Fostering improvements in the existing process based on public discussions and kernels<br>
# </div>

# <b>References</b> <br>
# 1. https://www.kaggle.com/code/yuanzhezhou/baseline-lgb-xgb-and-catboost

# <div class="alert alert-block alert-info" align = "center" style = "font-family: Calibri;font-size: 150%; color: black; background-color:#ccf2ff; border: solid black 2.5px; padding: 3.5px" >
#     <b>If you find this useful, please upvote the kernel and the input kernel too. <br> Best regards!</b>
# </div>
