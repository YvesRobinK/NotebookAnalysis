#!/usr/bin/env python
# coding: utf-8

# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0052cc; border-bottom: 8px solid #cc9966" > TABLE OF CONTENTS<br><div>  
# * [IMPORTS](#1)
# * [INTRODUCTION](#2)
#     * [CONFIGURATION](#2.1)
#     * [CONFIGURATION PARAMETERS](#2.2)    
#     * [COMPETITION DETAILS](#2.3)
# * [PREPROCESSING](#3)
# * [ADVERSARIAL CV](#4)
# * [EDA AND VISUALS](#5)
#     * [PAIR-PLOTS](#5.1)
#     * [CATEGORY COLUMN PLOTS](#5.2)
#     * [CONTINUOUS COLUMN ANALYSIS](#5.3)
#     * [DUPLICATES ANALYSIS](#5.4)
#     * [UNIVARIATE ANALYSIS AND FEATURE RELATIONS](#5.5)
#     * [OUTLIER ANALYSIS AND NULL ANALYSIS](#5.6)
# * [DATA TRANSFORMS](#6)
# * [MODEL TRAINING](#7)   
# * [ENSEMBLE AND SUBMISSION](#8)  
# * [PLANNED WAY FORWARD](#9)     

# <a id="1"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0052cc; border-bottom: 8px solid #cc9966" > IMPORTS<br> <div> 

# In[1]:


get_ipython().run_cell_magic('time', '', '\n# General library imports:-\n!pip install --upgrade scipy;\n\nimport pandas as pd;\nimport numpy as np;\nimport re;\nfrom scipy.stats import mode, kstest, normaltest, shapiro, anderson, jarque_bera;\nfrom collections import Counter;\nfrom itertools import product;\nfrom colorama import Fore, Style, init;\nfrom warnings import filterwarnings;\nfilterwarnings(\'ignore\');\n\nfrom tqdm.notebook import tqdm;\nfrom IPython.display import clear_output;\n\nimport seaborn as sns;\nimport matplotlib.pyplot as plt;\nfrom gc import collect;\nfrom pprint import pprint;\n\npd.set_option(\'display.max_columns\', 50);\npd.set_option(\'display.max_rows\', 50);\n\n# Setting rc parameters in seaborn for plots and graphs- \n# Reference - https://matplotlib.org/stable/tutorials/introductory/customizing.html:-\n# To alter this, refer to matplotlib.rcParams.keys()\n\nsns.set({"axes.facecolor"       : "#ffffff",\n         "figure.facecolor"     : "#ffffff",\n         "axes.edgecolor"       : "#000000",\n         "grid.color"           : "#ffffff",\n         "font.family"          : [\'Cambria\'],\n         "axes.labelcolor"      : "#000000",\n         "xtick.color"          : "#000000",\n         "ytick.color"          : "#000000",\n         "grid.linewidth"       : 0.85,  \n         "grid.linestyle"       : "--",\n         "axes.titlecolor"      : \'tab:blue\',\n         \'axes.titlesize\'       : 9.5,\n         \'axes.labelweight\'     : "bold",\n         \'legend.fontsize\'      : 7.0,\n         \'legend.title_fontsize\': 7.0,\n         \'font.size\'            : 8.0,\n         \'xtick.labelsize\'      : 7.5,\n         \'ytick.labelsize\'      : 7.5,        \n        });\n\nprint();\ncollect();\nclear_output();\n')


# In[2]:


get_ipython().run_cell_magic('time', '', '\n# Importing model and pipeline specifics:-\n\n# Pipeline specifics:-\nfrom sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, FunctionTransformer as FT;\nfrom sklearn.model_selection import (RepeatedStratifiedKFold as RSKF, \n                                     StratifiedKFold as SKF,\n                                     KFold, \n                                     RepeatedKFold as RKF, \n                                     cross_val_score);\nfrom sklearn.inspection import permutation_importance, PartialDependenceDisplay as PDD;\nfrom sklearn.feature_selection import mutual_info_regression, RFE;\nfrom sklearn.pipeline import Pipeline, make_pipeline;\nfrom sklearn.base import BaseEstimator, TransformerMixin;\nfrom sklearn.compose import ColumnTransformer;\nfrom sklearn.preprocessing import FunctionTransformer;\n\n# ML Model training:-\nfrom sklearn.metrics import mean_squared_error as mse, r2_score;\nfrom sklearn.svm import SVR;\nfrom xgboost import XGBRegressor;\nfrom lightgbm import LGBMClassifier, LGBMRegressor, log_evaluation;\nfrom catboost import CatBoostRegressor;\nfrom sklearn.ensemble import (RandomForestRegressor as RFR,\n                              ExtraTreesRegressor as ETR, \n                              GradientBoostingRegressor as GBR,\n                              HistGradientBoostingRegressor as HGBR);\nfrom sklearn.neighbors import KNeighborsRegressor as KNNR;\nfrom sklearn.linear_model import Ridge, Lasso;\n\nfrom sklearn import set_config; set_config(transform_output = "pandas");\n\n# Ensemble and tuning specifics:-\nimport optuna;\nfrom optuna import Trial, trial, create_study;\nfrom optuna.samplers import TPESampler;\noptuna.logging.set_verbosity = optuna.logging.ERROR;\n\nprint();\ncollect();\n')


# <a id="2"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0052cc; border-bottom: 8px solid #cc9966" > INTRODUCTION<br><div> 

# | Version<br>Number | Version Details | Best CV score| Single/ Ensemble|
# | :-: | --- | :-: | :-: |
# | **V1** |* EDA, plots and secondary features<br>* No scaling<br> * Used original data<br>* Baseline model-ML suite<br>* Optuna ensemble<br> | | |
# | **V2** |* EDA, plots and secondary features<br>* No scaling<br> * Used competition data<br>* Baseline model-ML suite<br>* Optuna ensemble<br> | | |
# | **V3** |* EDA, plots and secondary features<br>* No scaling<br> * Used original data in all folds<br>* Baseline model-ML suite<br>* Optuna ensemble<br> | | |
# 

# <a id="2.1"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > CONFIGURATION<br><div> 

# In[20]:


get_ipython().run_cell_magic('time', '', '\n# Configuration class:-\nclass CFG:\n    "Configuration class for parameters and CV strategy for tuning and training";\n    \n    # Data preparation:-   \n    version_nb         = 3;\n    gpu_switch         = "OFF"; \n    state              = 42;\n    target             = "xeout";\n    episode            = 15;\n    path               = f"/kaggle/input/playground-series-s3e{episode}/";\n    orig_path          = f"/kaggle/input/predicting-heat-flux/Data_CHF_Zhao_2020_ATE.csv";\n    adv_cv_req         = "Y";\n    ftre_plots_req     = "Y";\n    ftre_imp_req       = "Y";\n    \n    # Data transforms and scaling:-    \n    conjoin_orig_data  = "Y";\n    sec_ftre_req       = "Y";\n    imp_catcols        = "Y";\n    OH_enc_req         = "Y";\n    scale_req          = "N";\n    # NOTE---Keep a value here even if scale_req = N, this is used for linear models:-\n    scl_method         = "Z"; \n    \n    # Model Training:- \n    baseline_req       = "N";\n    ML                 = "Y";\n    use_orig_allfolds  = "Y";\n    n_splits           = 10 ;\n    n_repeats          = 2 ;\n    nbrnd_erly_stp     = 200 ;\n    mdlcv_mthd         = \'RKF\';\n    \n    # Ensemble with optuna:-   \n    ensemble_req       = "Y";\n    direction          = \'minimize\';\n    n_ens_trials       = 500;\n          \n    # Global variables for plotting:-\n    grid_specs = {\'visible\': True, \'which\': \'both\', \'linestyle\': \'--\', \n                           \'color\': \'lightgrey\', \'linewidth\': 0.75};\n    title_specs = {\'fontsize\': 9, \'fontweight\': \'bold\', \'color\': \'tab:blue\'};\n\n# Color printing    \ndef PrintColor(text:str, color = Fore.BLUE, style = Style.BRIGHT):\n    "Prints color outputs using colorama using a text F-string";\n    print(style + color + text + Style.RESET_ALL); \n\n# Scaler to be used for continuous columns:- \nall_scalers = {\'Robust\': RobustScaler(), \n               \'Z\': StandardScaler(), \n               \'MinMax\': MinMaxScaler()\n              };\nscaler      = all_scalers.get(CFG.scl_method);\n\n# Commonly used CV strategies for later usage:-\nall_cv= {\'KF\'  : KFold(n_splits= CFG.n_splits, shuffle = True, random_state= CFG.state),\n         \'RKF\' : RKF(n_splits= CFG.n_splits, n_repeats = CFG.n_repeats, random_state= CFG.state),\n         \'RSKF\': RSKF(n_splits= CFG.n_splits, n_repeats = CFG.n_repeats, random_state= CFG.state),\n         \'SKF\' : SKF(n_splits= CFG.n_splits, shuffle = True, random_state= CFG.state)\n        };\nprint();\n\nPrintColor(f"--> Configuration done!");\ncollect();\n')


# <a id="2.2"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > CONFIGURATION PARAMETERS<br><div> 
# 
# 
# | Parameter         | Description                                             | Possible value choices|
# | ---               | ---                                                     | :-:                   |
# |  version_nb       | Version Number                                          | integer               |
# |  gpu_switch       | GPU switch                                              | ON/OFF                |
# |  state            | Random state for most purposes                          | integer               |
# |  target           | Target column name                                      | yield                 |
# |  episode          | Episode Number                                          | integer               |
# |  path             | Path for input data files                               |                       |
# |  orig_path        | Path for input original data files                      |                       |
# |  adv_cv_req       | Adversarial CV required                                 | Y/N                   |
# |  ftre_plots_req   | Feature plots required                                  | Y/N                   |
# |  ftre_imp_req     | Feature importance required                             | Y/N                   |
# |  conjoin_orig_data| Conjoin original data                                   | Y/N                   |
# |  sec_ftre_req     | Secondary features required                             | Y/N                   |
# |  imp_catcols      | Impute category columns                                 | Y/N                   |
# |  OH_enc_req       | One-hot encoding required                               | Y/N                   |
# |  scale_req        | Scaling required                                        | Y/N                   |
# |  scl_method       | Scaling method                                          | Z/ Robust/ MinMax     |
# |  baseline_req     | Baseline model required                                 | Y/N                   |
# |  ML               | Machine Learning Models                                 | Y/N                   |
# |  GAM              | GAM required                                            | Y/N                   |
# |  use_orig_allfolds| Use original data across all folds                      | Y/N                   |
# |  n_splits         | Number of CV splits                                     | integer               |
# |  n_repeats        | Number of CV repeats                                    | integer               |
# |  nbrnd_erly_stp   | Number of early stopping rounds                         | integer               |
# |  mdl_cv_mthd      | Model CV method name                                    | RKF/ RSKF/ SKF/ KFold |
# |  ensemble_req     | Ensemble Required                                       | Y/N                   |
# |  direction        | Optuna objective direction                              | RKminimize/ maximize  |
# |  n_ens_trials     | Ensemble trials                                         | integer               |

# <a id="2.3"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > COMPETITION DETAILS<br><div>
#        
# **Competition details and notebook objectives**<br>
# 1. This is a regression challenge to predict null features in the dataset column named **x_e_out [-]**. We don't have a test set here. **All rows with null x_e_out [-] are the test set**. The competition metric is RMSE<br>
# 2. In this starter notebook, we start the assignment with a detailed EDA, feature plots, interaction effects, adversarial CV analysis and develop starter models to initiate the challenge. We will also incorporate other opinions and approaches as we move along the challenge. 

# <a id="3"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0052cc; border-bottom: 8px solid #cc9966" > PREPROCESSING<br><div> 

# In[4]:


get_ipython().run_cell_magic('time', '', '\nPrintColor(f"\\n---------- Data Preprocessing ---------- \\n", color = Fore.MAGENTA);\n\n# Reading the datasets:-\ntrain    = pd.read_csv(CFG.path + f"data.csv", index_col = \'id\');\noriginal = pd.read_csv(CFG.orig_path, index_col = \'id\');\nsub_fl   = pd.read_csv(CFG.path + f"sample_submission.csv");\n\n# Renaming columns:-\nPrintColor(f"Train columns before transforms");\ndisplay(train.columns);\nstrt_ftre = train.columns.str.replace(r"[\\W+]|\\s+|_", "").str.lower();\ntrain.columns = strt_ftre;\noriginal.columns = strt_ftre;\nPrintColor(f"\\nTrain columns after transforms");\ndisplay(strt_ftre);\n\n# Creating dataset information:\nPrintColor(f"\\nTrain information\\n");\ndisplay(train.info());\n\nPrintColor(f"\\nOriginal data information\\n")\ndisplay(original.info());\nprint();\n\n# Calculating desciptions:-\nPrintColor(f"\\nTrain description\\n");\ndisplay(train.describe(percentiles= [0.05, 0.25, 0.50, 0.75, 0.9, 0.95, 0.99]).\\\n        transpose().\\\n        drop(columns = [\'count\'], errors = \'ignore\').\\\n        drop([CFG.target], axis=0, errors = \'ignore\').\\\n        style.format(precision = 2));\n\nPrintColor(f"\\nOriginal description\\n");\ndisplay(original.describe(percentiles= [0.05, 0.25, 0.50, 0.75, 0.9, 0.95, 0.99]).\\\n        transpose().\\\n        drop(columns = [\'count\'], errors = \'ignore\').\\\n        drop([CFG.target], axis=0, errors = \'ignore\').\\\n        style.format(precision = 2));\n\n# Creating a copy of the datasets for further use:-\ntrain_copy, orig_copy = train.copy(deep= True), original.copy(deep = True);\n\n# Dislaying the unique values across the datasets:-\nPrintColor(f"\\nUnique values\\n");\n_ = pd.concat([train.nunique(), original.nunique()], axis=1);\n_.columns = [\'Train\', \'Original\'];\ndisplay(_.style.background_gradient(cmap = \'Blues\').format(formatter = \'{:,.0f}\'));\n\n# Analyzing nulls in the dataset:-\nPrintColor(f"\\nNull values across data set\\n");\n_ = pd.concat([train.isna().sum(axis=0),original.isna().sum(axis=0)], axis=1).\\\nrename(columns = {0: "Nb_Nulls_Train", 1: "Nb_Nulls_Orig"});\n_ = _.assign(Null_Pct_Train = _.Nb_Nulls_Train/ len(train));\ndisplay(_.style.\\\n        format(formatter = {\'Nb_Nulls_Train\': \'{:,.0f}\',\n                            \'Null_Pct_Train\': \'{:.2%}\'}\n              ).background_gradient(cmap = \'Blues\')\n       );\n\nprint();\ncollect();\n')


# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > INFERENCES<br> <div>

# <div style= "font-family: Cambria; letter-spacing: 0px; color:#000000; font-size:110%; text-align:left;padding:3.0px; background: #f2f2f2" >
# 1. We have 2 categorical columns and remaining numerical columns<br>
# 2. We have nulls across all columns, with our target column having 32% nulls. Column chfexpmwm2 has no nulls. Other columns have between 14-18% nulls<br>
# 4. The synthetic data is nearly 17 times the original data. Duplicate handling could be a key challenge in this case<br>
# 5. We have few unique values (discrete values) in quite a few columns. These could be considered as categorical encoded columns and then suitable encoding options could also be ensued on these attributes as well<br>
# </div>

# <a id="4"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0052cc; border-bottom: 8px solid #cc9966" > ADVERSARIAL CV<br><div>
#     
# We perform adversarial CV with the train-original data. We drop nulls in the train data while doing the adversarial CV.

# In[5]:


get_ipython().run_cell_magic('time', '', '\n# Performing adversarial CV between the 2 specified datasets:-\ndef Do_AdvCV(df1:pd.DataFrame, df2:pd.DataFrame, source1:str, source2:str):\n    "This function performs an adversarial CV between the 2 provided datasets if needed by the user";\n    \n    # Adversarial CV per column:-\n    ftre = train.drop(columns = [\'id\', CFG.target, "Source"], errors = \'ignore\').columns[2:];\n    adv_cv = {};\n\n    for col in ftre:\n        PrintColor(f"---> Current feature = {col}", style = Style.NORMAL);\n        shuffle_state = np.random.randint(low = 10, high = 100, size= 1);\n\n        full_df = \\\n        pd.concat([df1[[col]].assign(Source = source1), df2[[col]].assign(Source = source2)], \n                  axis=0, ignore_index = True).\\\n        dropna().\\\n        sample(frac = 1.00, random_state = shuffle_state);\n\n        full_df = full_df.assign(Source_Nb = full_df[\'Source\'].eq(source2).astype(np.int8));\n\n        # Checking for adversarial CV:-\n        model = LGBMClassifier(random_state = CFG.state, max_depth = 3, learning_rate = 0.05);\n        cv    = all_cv[\'RSKF\'];\n        score = np.mean(cross_val_score(model, \n                                        full_df[[col]], \n                                        full_df.Source_Nb, \n                                        scoring= \'roc_auc\', \n                                        cv= cv)\n                       );\n        adv_cv.update({col: round(score, 4)});\n        collect();\n    \n    del ftre;\n    \n    PrintColor(f"\\nResults\\n");\n    pprint(adv_cv, indent = 5, width = 20, depth = 1);\n    collect();\n    \n    fig, ax = plt.subplots(1,1,figsize = (12, 5));\n    pd.Series(adv_cv).plot.bar(color = \'tab:blue\', ax = ax);\n    ax.axhline(y = 0.60, color = \'red\', linewidth = 2.75);\n    ax.grid(**CFG.grid_specs); \n    plt.yticks(np.arange(0.0, 0.81, 0.05));\n    plt.show();\n    \n# Implementing the adversarial CV:-\nif CFG.adv_cv_req == "Y":\n    PrintColor(f"\\n---------- Adversarial CV - Train vs Original ----------\\n", \n               color = Fore.MAGENTA);\n    Do_AdvCV(df1 = train, df2 = original, source1 = \'Train\', source2 = \'Original\');\n      \nif CFG.adv_cv_req == "N":\n    PrintColor(f"\\nAdversarial CV is not needed\\n", color = Fore.RED);\n    \ncollect();\nprint();\n')


# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > INFERENCE<br><div>

# <div style= "font-family: Cambria; letter-spacing: 0px; color:#000000; font-size:110%; text-align:left;padding:3.0px; background: #f2f2f2" >
# We need to further check the train-original distribution further, adversarial validation results indicate that we may not use the original dataset<br>
# </div>

# <a id="5"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0052cc; border-bottom: 8px solid #cc9966" > VISUALS AND EDA <br><div> 
#  

# <a id="5.1"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > PAIRPLOTS<br><div>

# In[6]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ftre_plots_req == "Y":\n    PrintColor(f"\\nTrain data- pair plots\\n");\n    _ = sns.pairplot(data = train, \n                     diag_kind = \'kde\', markers= \'o\', plot_kws= {\'color\': \'tab:blue\'}               \n                    );\n\nprint();\ncollect();\n')


# In[7]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ftre_plots_req == "Y":\n    PrintColor(f"\\nOriginal data- pair plots\\n");\n    _ = sns.pairplot(data = original, \n                     diag_kind = \'kde\', markers= \'o\', plot_kws= {\'color\': \'tab:blue\'}               \n                    );\nprint();\ncollect();\n')


# <a id="5.2"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > CATEGORY COLUMN PLOTS<br><div>

# In[8]:


get_ipython().run_cell_magic('time', '', '\ncat_cols = strt_ftre[0:2];\n\nif CFG.ftre_plots_req == "Y":\n    fig, axes = plt.subplots(2, 2, figsize = (12, len(cat_cols)* 4),\n                             sharey = True,\n                             gridspec_kw = {\'wspace\': 0.2, \'hspace\': 0.5}, \n                             width_ratios = [0.75, 0.25]\n                            );\n\n    for i, col in enumerate(cat_cols):\n        ax = axes[0, i];\n        train[col].value_counts(normalize = True).sort_index().plot.bar(color = \'#0059b3\', ax = ax);\n        ax.set_title(f"{col}_Train", **CFG.title_specs);\n        ax.grid(**CFG.grid_specs);\n        ax.set(xlabel = \'\', ylabel = \'\');\n        ax.set_yticks(np.arange(0, 0.81, 0.05));\n        \n        ax = axes[1, i];\n        original[col].value_counts(normalize = True).sort_index().plot.bar(color = \'#0099e6\', ax = ax);\n        ax.set_title(f"{col}_Original", **CFG.title_specs);\n        ax.grid(**CFG.grid_specs);\n        ax.set(xlabel = \'\', ylabel = \'\');   \n        ax.set_yticks(np.arange(0, 0.81, 0.05));\n\n    plt.tight_layout();\n    plt.show();\n    \nprint();\ncollect();\n')


# <a id="5.3"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > CONTINUOUS COLUMN PLOTS<br><div>

# In[9]:


get_ipython().run_cell_magic('time', '', '\ncont_cols = strt_ftre[2:];\n\nif CFG.ftre_plots_req == "Y":\n    df = pd.concat([train[cont_cols].assign(Source = \'Train\'), \n                    original[cont_cols].assign(Source = \'Original\')], \n                   axis=0, ignore_index = True\n                  );\n    \n    fig, axes = plt.subplots(3,len(cont_cols), figsize = (30, 8), \n                             gridspec_kw = {\'hspace\': 0.35, \'wspace\': 0.2}, \n                             height_ratios = [0.70, 0.15, 0.15]\n                            );\n    \n    for i,col in enumerate(cont_cols):\n        ax = axes[0,i];\n        sns.kdeplot(data = df[[col, \'Source\']], x = col, hue = \'Source\', palette = [\'blue\', \'red\'], \n                    ax = ax, linewidth = 2.25\n                   );\n        ax.set_title(f"\\n{col}", **CFG.title_specs);\n        ax.grid(**CFG.grid_specs);\n        ax.set(xlabel = \'\', ylabel = \'\');\n        \n        ax = axes[1,i];\n        sns.boxplot(data = df.loc[df.Source == \'Train\', [col]], x = col, width = 0.25,\n                    color = \'#008ae6\', saturation = 0.90, linewidth = 1.25, fliersize= 2.25,\n                    ax = ax);\n        ax.set(xlabel = \'\', ylabel = \'\');\n        ax.set_title(f"{col}- Train", **CFG.title_specs);\n        \n        ax = axes[2,i];\n        sns.boxplot(data = df.loc[df.Source == \'Original\', [col]], x = col, width = 0.25, fliersize= 2.25,\n                    color = \'#00aaff\', saturation = 0.6, linewidth = 1.25, \n                    ax = ax); \n        ax.set(xlabel = \'\', ylabel = \'\');\n        ax.set_title(f"{col}- Original", **CFG.title_specs);\n              \n    plt.suptitle(f"\\nDistribution analysis- continuous columns\\n", **CFG.title_specs);\n    plt.tight_layout();\n    plt.show();\n    \nprint();\ncollect();\n')


# In[10]:


get_ipython().run_cell_magic('time', '', '\ndef MakeGrpPlot(df: pd.DataFrame, contcol: str):\n    """\n    This function makes kde plots per continuous column per category column\n    """;\n    \n    global cont_cols, cat_cols;\n    fig, axes = plt.subplots(1,2, figsize = (20, 3.75), \n                             gridspec_kw = {\'hspace\': 0.2, \'wspace\': 0.25});\n    \n    for i, catcol in enumerate(cat_cols):\n        ax = axes[i];\n        sns.kdeplot(data      = df[[contcol, catcol]], \n                    x         = contcol, \n                    hue       = catcol, \n                    palette   = \'rainbow\', \n                    ax        = ax, \n                    linewidth = 2.5,\n                   );\n        ax.set_title(f"Hue = {catcol}", **CFG.title_specs);\n        ax.grid(**CFG.grid_specs);\n        ax.set(xlabel = \'\', ylabel = \'\');\n        \n    plt.tight_layout();\n    plt.show();\n\n# Implementing the grouped distribution analysis:-\nif CFG.ftre_plots_req == "Y":\n    for contcol in cont_cols:\n        PrintColor(f"\\n{\'-\'* 30} {contcol.capitalize()} distribution analysis {\'-\'* 30}\\n");\n        MakeGrpPlot(train, contcol);\n        print();\n    \nprint();\ncollect();\n')


# <a id="5.4"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > DUPLICATES ANALYSIS<br><div>

# In[11]:


get_ipython().run_cell_magic('time', '', '\n# Displaying duplicates by model and test datasets:-\nif CFG.ftre_plots_req == "Y":\n    fig, ax = plt.subplots(1,1, figsize = (10,2));\n    _ = train.loc[train.duplicated(keep = \'first\')];\n    _.groupby(_[CFG.target].isna()).size().plot.barh(ax = ax, color = \'tab:blue\');\n    ax.set_title(f"\\nDuplicates in the train-test model data\\n", **CFG.title_specs);\n    ax.set_yticks([True, False], [\'Test\', \'Train\'], rotation = 45, fontsize = 7.5);\n    plt.xticks(range(0, 261,10));\n    plt.tight_layout();\n    plt.show();\n    \ncollect();\nprint();\n')


# <a id="5.5"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > FEATURE INTERACTION AND UNIVARIATE RELATIONS<br><div>
#     
# We aim to do the below herewith<br>
# 1. Correlation<br>
# 2. Mutual information<br>
# 3. Leave One Feature Out model<br>

# In[12]:


get_ipython().run_cell_magic('time', '', '\ndef MakeCorrPlot(df: pd.DataFrame, data_label:str, figsize = (30, 9)):\n    """\n    This function develops the correlation plots for the given dataset\n    """;\n    \n    fig, axes = plt.subplots(1,2, figsize = figsize, gridspec_kw = {\'hspace\': 0.2, \'wspace\': 0.2},\n                             sharey = True\n                            );\n    \n    for i, method in enumerate([\'pearson\', \'spearman\']):\n        corr_ = df.drop(columns = [\'id\', \'Source\'], errors = \'ignore\').corr(method = method);\n        ax = axes[i];\n        sns.heatmap(data = corr_,  annot= True,fmt= \'.2f\', cmap = \'viridis\',\n                    annot_kws= {\'fontweight\': \'bold\',\'fontsize\': 7.5}, \n                    linewidths= 1.5, linecolor=\'white\', cbar= False, mask= np.triu(np.ones_like(corr_)),\n                    ax= ax\n                   );\n        ax.set_title(f"\\n{method.capitalize()} correlation- {data_label}\\n", **CFG.title_specs);\n        \n    collect();\n    print();\n')


# In[13]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ftre_imp_req == "Y":\n    \n    # Implementing correlation analysis:-\n    MakeCorrPlot(df = train, data_label = \'Train\', figsize = (12, 4));\n    MakeCorrPlot(df = original, data_label = \'Original\', figsize = (12, 4));\n\n    # Calculating mutual information after dropping nulls:-\n    MutInfoSum = {};\n\n    for col in strt_ftre[3:]:\n        if col == CFG.target: pass\n        else:\n            df = train[[CFG.target, col]].dropna();\n            MutInfoSum[col] = np.round(mutual_info_regression(df[[col]], df[CFG.target], \n                                                              random_state = CFG.state)[0], 5);\n            del df;\n\n    fig, axes = plt.subplots(1,2, figsize = (16,4), sharey = True,\n                             gridspec_kw = {\'hspace\': 0.2, \'wspace\': 0.2});\n    ax = axes[0];\n    pd.Series(MutInfoSum).plot.barh(ax = ax, color = \'tab:blue\');\n    ax.set_title(f"\\nMutual information\\n", **CFG.title_specs);\n \n    # Calculating leave-one-feature-out analysis:-    \n    model = LGBMRegressor(max_depth     = 6, \n                          random_state  = CFG.state,\n                          objective     = \'regression\', \n                          metric        = \'rmse\',\n                          num_leaves    = 100,\n                          n_estimators  = 5000, \n                          reg_alpha     = 0.001,\n                          reg_lambda    = 0.85,\n                          verbosity     = -1,\n                         );\n\n    X, y   = train.iloc[:, 3:].dropna().drop(columns = CFG.target), train.iloc[:, 3:].dropna()[CFG.target];\n    LOOSum = {};\n    cv     = all_cv[\'KF\'];\n    Scores = {};\n    \n    for col in tqdm(strt_ftre[3:]):\n        if col == CFG.target: pass\n        else:\n            scores = [];\n            cols   = [c for c in strt_ftre[3:] if c not in [CFG.target, col]];\n            \n            for fold_nb, (train_idx, dev_idx) in enumerate(cv.split(X, y)): \n                Xtr  = X.iloc[train_idx][cols];   \n                Xdev = X.iloc[dev_idx][cols];\n                ytr  = y.loc[y.index.isin(Xtr.index)];\n                ydev = y.loc[y.index.isin(Xdev.index)];\n\n                # Fitting the model:-    \n                model.fit(Xtr, ytr, eval_set = [(Xdev, ydev)], \n                          verbose = 0,\n                          eval_metric = \'rmse\',\n                          early_stopping_rounds = CFG.nbrnd_erly_stp\n                         ); \n                dev_preds = model.predict(Xdev);\n                scores.append(mse(ydev, dev_preds, squared = False));\n                \n            Scores[col] = np.round(np.mean(scores), 5); \n            del scores;\n            \n    ax = axes[1];\n    pd.Series(Scores).plot.barh(color = \'tab:blue\', ax = ax);\n    ax.set_title(f"\\nLeave one out analysis- numerical columns\\n", **CFG.title_specs);\n    ax.set_xticks(ticks = np.arange(0, 0.091,0.005), \n                  labels = np.arange(0, 0.091,0.005),\n                  fontsize = 7, rotation = 45\n                 );\n    \n    plt.tight_layout();\n    plt.show();\n    \n    del Scores, X,y, cv, LOOSum;\n\nprint();\ncollect();\n')


# <a id="5.6"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > OUTLIER AND NULL ANALYSIS<br><div>
#     
# We need to pay special attention to outliers as RMSE is used as a metric here. The plots above indicates outliers across all numeric columns. <br>
# Let's analyse them further, assuming a homogenous train-test set

# In[14]:


get_ipython().run_cell_magic('time', '', '\n\nfig, axes = plt.subplots(1,2, figsize = (18, 10), sharey = True,\n                         gridspec_kw= {\'wspace\': 0.25}\n                        ); \ncols = list(strt_ftre[2:]);\ncols.remove(CFG.target);\n\n_ = train[cols].describe().T[[\'25%\', \'75%\']];\n_[\'iqr\'] = _[\'75%\'] - _[\'25%\'];\n_[\'otl_ub\'], _[\'otl_lb\'] = _[\'75%\'] + (1.5 * _[\'iqr\']), _[\'25%\'] - (1.5 * _[\'iqr\']);\n_ = _.iloc[:, -2:];\n\nax = axes[0];\nOtl_Prf = np.clip((train[cols] > _[\'otl_ub\'].values).astype(np.int8) + \\\n                  (train[cols] < _[\'otl_lb\'].values).astype(np.int8), \n                  a_min = 0, a_max =1\n                 ); \nsns.heatmap(data = Otl_Prf,\n            cbar = None, \n            ax   = ax, \n            cmap = [\'#e6f9ff\', \'#1a1aff\']\n           );\nax.set_title(f"\\nOutlier analysis and location\\n", **CFG.title_specs);\n\nax = axes[1];\nsns.heatmap(train[strt_ftre].isna(), cmap= [\'#e6f9ff\', \'#003366\'], ax = ax, cbar= False);\nax.set_title(f"\\nNull location across the competition data\\n", **CFG.title_specs);\nax.set(xlabel = \'\', ylabel = \'\');\n\nplt.tight_layout();\nplt.show();\ndel fig, ax, axes;\n\n# Studying outliers in conjunction with null values in the target:-\nOtl_Prf[CFG.target] = train[CFG.target].values;\nOtl_NullTgt = {col: \n               len(Otl_Prf.loc[(Otl_Prf[CFG.target].isna()) & (Otl_Prf[col] >0)]) for col in cols};\n\nfig, ax = plt.subplots(1,1, figsize = (8,4), sharey = True, gridspec_kw= {\'wspace\': 0.25}); \npd.Series(Otl_NullTgt).plot.barh(ax = ax, color = \'tab:blue\');\nax.set_title(f"\\nOutliers and null target per column\\n", **CFG.title_specs);\nax.set_xticks(range(0, 1501,50), range(0, 1501,50), rotation = 90, fontsize = 7);\n\nplt.tight_layout();\nplt.show();\ndel fig, ax;\n    \n# Analyzing nulls across rows in the complete data:-\ntrain[\'Nb_Null_Ftre\']     = train.drop(columns = [CFG.target]).isna().sum(axis=1).astype(np.int8);\ntrain[\'Nb_Null_CatFtre\']  = train[cat_cols].isna().sum(axis=1).astype(np.int8);\ntrain[\'Nb_Null_ContFtre\'] = train[cont_cols].isna().sum(axis=1).astype(np.int8);\n\nfig, axes = plt.subplots(1,3, figsize = (27, 7), sharey = True, gridspec_kw = {\'wspace\': 0.20});\n\nfor i, col in tqdm(enumerate(train.columns[-3:])):\n    train[col].\\\n    value_counts().\\\n    sort_index().\\\n    plot.bar(ax = axes[i], color = \'tab:blue\');\n    axes[i].set_title(f"\\n{col.upper()}\\n", **CFG.title_specs);\n    axes[i].set_yticks(range(0, 22501, 800), range(0, 22501, 800), fontsize = 7,);\n\nplt.suptitle(f"Number of nulls across rows\\n", color = \'tab:blue\', \n             fontsize = 14, fontweight = \'bold\');\nplt.tight_layout();\nplt.show();\n\nPrintColor(f"\\nNulls across features-rows\\n");\npprint(Counter(train[\'Nb_Null_Ftre\']), depth = 1, width = 10, indent = 5);\n\nprint();\ncollect();\n')


# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > INFERENCES<br> <div>

# <div style= "font-family: Cambria; letter-spacing: 0px; color:#000000; font-size:110%; text-align:left;padding:3.0px; background: #f2f2f2" >
# 1. Nearly 10,000 rows in the data have null target values and they are the test set elements.Also, approximately 9600 rows do not have any nulls.<br>
# 2. 2 categorical columns and the remaining numerical columns are all important towards the target<br>
# 3. Feature correlation is pronounced and can be used for regression<br>
# 4. Imputation of categorical columns is separate from the numerical columns<br>
# 5. The last column (chfexpmwm2) is of importance. It has no nulls and has additional importance than the other columns in the numerical column group<br>
# 6. Lots of outliers lie in the data. Since our metric is RMSE, we need to pay attention to them. Outliers are likely to influence the nulls in the target<br>
# 7. Duplicates are also present in the test set, handling them is again important.<br>
# 
# </div>

# <a id="6"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0052cc; border-bottom: 8px solid #cc9966" > DATA TRANSFORMS <br><div> 
#     
# Thanks to- https://www.kaggle.com/code/mohammadrazeghi/p03e15-preprocess-pipeline-baseline-xgboost
#     
# **Process outline-**<br>
# 1. Split the data into train and test tables<br>
# 2. Combine train and original data if needed<br>
# 3. Make secondary features if necessary<br>
# 4. Encode categorical columns and impute numerical columns<br>
# 5. Ensue centering and scaling if needed<br>

# In[15]:


get_ipython().run_cell_magic('time', '', '\n# Developing secondary features:-\ndef SecFtreMaker(df: pd.DataFrame):\n    "This function creates secondary features if required by the user";\n    global num_cols;\n\n    if CFG.sec_ftre_req == "Y":\n        df[\'sa\']  = df[\'demm\'] * df[\'lengthmm\'];\n        df[\'sdr\'] = df[\'demm\'] / df[\'dhmm\'];\n        \n    num_cols = list(df.drop(columns = [\'Source\'], errors = \'ignore\').columns[3:]);\n    return df;\n\n# Developing imputation and scaling:-\nclass ImpScl(BaseEstimator, TransformerMixin):\n    """\n    This class develops imputation for the baseline models.\n    It considers mode imputation and one-hot encoding for character columns and mean imputation for numeric columns\n    """;\n    \n    def __init__(self): pass\n    \n    def fit(self, X, y= None):\n        """\n        This method creates parameters for imputation and scaling operations\n        """;\n        \n        self.cat_cols = list(X.iloc[:, 0:2].columns);\n        self.num_cols = list(X.select_dtypes(include = np.number).\\\n                             drop(columns= [CFG.target, \'id\', \'Source\'], errors = \'ignore\').\\\n                             columns\n                            );\n        \n        self.num_imptr = \\\n        X.loc[X.Source == \'Competition\'].\\\n        groupby(self.cat_cols)[self.num_cols].mean().\\\n        drop(CFG.target, axis=1, errors = \'ignore\').\\\n        add_suffix("_mu").\\\n        reset_index();\n        \n        self.cat_imptr = X[cat_cols].mode().values[0].tolist();\n        \n        self.params          = X[self.num_cols].describe().transpose().rename({\'50%\':\'median\'}, axis=1);\n        self.params[\'iqr\']   = self.params[\'75%\'] - self.params[\'25%\'];\n        self.params          = self.params[[\'mean\', \'std\', \'max\', \'min\', \'iqr\', \'median\']].astype(np.float32);\n    \n        return self;\n    \n    def transform(self, X, y = None):\n        """\n        This function creates the imputed dataset and scales the data if requested by the user\n        We can also create one-hot encoder for categorical columns\n        """;\n        \n        df = X.copy();\n        \n        if CFG.imp_catcols == "Y":\n            for i, col in enumerate(self.cat_cols): df[col] = df[col].fillna(self.cat_imptr[i]);\n            \n        df = df.merge(self.num_imptr, how= \'left\', on = self.cat_cols);\n        \n        for col in self.num_cols: \n            df[col] = df[col].fillna(df[f"{col}_mu"][0]);\n            df.drop(columns = [f"{col}_mu"], errors = \'ignore\', inplace = True);\n               \n        if CFG.scale_req == "Y":\n            if CFG.scl_method == "Z":\n                for col in self.num_cols:\n                    df[col] = (df[col] - self.params.at[col, \'mean\']) / self.params.at[col, \'std\'];\n            elif CFG.scl_method == "Robust":\n                for col in self.num_cols:\n                    df[col] = (df[col] - self.params.at[col, \'median\']) / self.params.at[col, \'iqr\'];\n            elif CFG.scl_method == "MinMax":\n                for col in self.num_cols:\n                    df[col] = (df[col] - self.params.at[col, \'min\']) / (self.params.at[col, \'max\'] - self.params.at[col, \'min\']);  \n                    \n        \n        if CFG.OH_enc_req == "Y":\n            df = pd.concat([df, pd.get_dummies(df[self.cat_cols])], axis = 1).\\\n            drop(columns = self.cat_cols, errors = \'ignore\');\n            \n        PrintColor("\\nNulls after transform\\n");\n        display(df.isna().sum(axis=0));\n                                \n        return df;\n    \nprint();\ncollect();\n')


# In[33]:


get_ipython().run_cell_magic('time', '', '\nPrintColor(f"\\n{\'-\'* 30} Data Transformations {\'-\'* 30}\\n", color = Fore.MAGENTA);\n\n_ = train.loc[train[CFG.target].notnull()];\nXtrain, ytrain = _.drop(columns = [CFG.target], errors = \'ignore\').assign(Source = \'Competition\'), _[CFG.target];\nXtest = train.loc[train[CFG.target].isna()].drop(columns = [CFG.target]).assign(Source = \'Competition\');\ndel _;\n\nPrintColor(f"---> Train-test shapes = {Xtrain.shape} {Xtest.shape}");\n\nif CFG.conjoin_orig_data == "Y":\n    Xtrain = pd.concat([Xtrain, \n                        original.\\\n                        drop(columns = [CFG.target]).\\\n                        assign(Nb_Null_Ftre = 0, Nb_Null_CatFtre = 0, Nb_Null_ContFtre = 0, Source ="Original")\n                       ], axis=0, ignore_index = True);\n    ytrain = pd.concat([ytrain, original[CFG.target]], axis=0, ignore_index = True);\n    PrintColor(f"---> Shape of X,y train with original = {Xtrain.shape} {ytrain.shape}");\n\nxform = Pipeline(steps = [(\'SecFtre\', FunctionTransformer(SecFtreMaker)),(\'Imp\', ImpScl())]);\nxform.fit(Xtrain, ytrain);\nPrintColor(f"\\n---> Pipeline transform - Train");\nXtrain = xform.transform(Xtrain); \nPrintColor(f"\\n---> Pipeline transform - Test");\nXtest  = xform.transform(Xtest);\n\n# Setting indices between ytrain and Xtrain:-\nytrain.index = Xtrain.index\n\nPrintColor(f"\\n---> Shape after transforms = {Xtrain.shape} {Xtest.shape} {ytrain.shape}\\n");\n\nprint();\ncollect();\n')


# <a id="7"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0052cc; border-bottom: 8px solid #cc9966" > MODEL TRAINING <br><div> 
#     
# We commence our model assignment with a simple ensemble of tree-based and linear models and then shall proceed with the next steps

# In[34]:


get_ipython().run_cell_magic('time', '', '\n# Initializing model I-O:-\n\nMdl_Master = \\\n{\'CBR\': CatBoostRegressor(**{\'task_type\'           : "GPU" if CFG.gpu_switch == "ON" else "CPU",\n                             \'loss_function\'       : \'RMSE\',\n                             \'eval_metric\'         : \'RMSE\',\n                             \'objective\'           : \'RMSE\',\n                             \'random_state\'        : CFG.state,\n                             \'bagging_temperature\' : 1.0,\n                             \'colsample_bylevel\'   : 0.3,\n                             \'iterations\'          : 12_000,\n                             \'learning_rate\'       : 0.04,\n                             \'od_wait\'             : 25,\n                             \'max_depth\'           : 6,\n                             \'l2_leaf_reg\'         : 1.20,\n                             \'min_data_in_leaf\'    : 20,\n                             \'random_strength\'     : 0.45, \n                             \'max_bin\'             : 400,\n                             \'use_best_model\'      : True, \n                           }\n                        ),\n\n \'LGBMR\': LGBMRegressor(**{\'device\'            : "gpu" if CFG.gpu_switch == "ON" else "cpu",\n                           \'objective\'         : \'regression\',\n                           \'metric\'            : \'rmse\',\n                           \'boosting_type\'     : \'gbdt\',\n                           \'random_state\'      : CFG.state,\n                           \'feature_fraction\'  : 0.875,\n                           \'learning_rate\'     : 0.0555,\n                           \'max_depth\'         : 5,\n                           \'n_estimators\'      : 12_000,\n                           \'num_leaves\'        : 120,                    \n                           \'reg_alpha\'         : 0.00001,\n                           \'reg_lambda\'        : 1.25,\n                           \'verbose\'           : -1,\n                         }\n                      ),\n\n \'XGBR\': XGBRegressor(**{\'objective\'          : \'reg:squarederror\',\n                         \'eval_metric\'        : \'rmse\',\n                         \'random_state\'       : CFG.state,\n                         \'tree_method\'        : "gpu_hist" if CFG.gpu_switch == "ON" else "hist",\n                         \'colsample_bytree\'   : 0.95,\n                         \'subsample\'          : 0.65,\n                         \'learning_rate\'      : 0.015,\n                         \'max_depth\'          : 6,\n                         \'n_estimators\'       : 10_000,                         \n                         \'reg_alpha\'          : 0.000001,\n                         \'reg_lambda\'         : 3.75,\n                         \'min_child_weight\'   : 30,\n                        }\n                     ),\n \n \'RFR\' : RFR(**{\'n_estimators\'            : 300,\n                \'criterion\'               : \'squared_error\',\n                \'max_depth\'               : 8,\n                \'min_samples_leaf\'        : 15,\n                \'oob_score\'               : True,\n                \'bootstrap\'               : True,\n                \'n_jobs\'                  : -1,\n                \'random_state\'            : CFG.state,\n               }\n            ),\n \n \'ETR\' : ETR(**{\'n_estimators\'            : 400,\n                \'criterion\'               : \'squared_error\',\n                \'max_depth\'               : 10,\n                \'min_samples_leaf\'        : 10,\n                \'oob_score\'               : True,\n                \'bootstrap\'               : True,\n                \'n_jobs\'                  : -1,\n                \'random_state\'            : CFG.state,\n                \'min_samples_split\'       : 6,\n               }\n            ),\n \n \'GBR\': GBR(**{ \'loss\'            : \'squared_error\',\n                \'learning_rate\'   : 0.0525,\n                \'n_estimators\'    : 800,\n                \'min_samples_leaf\': 30,\n                \'max_depth\'       : 6,\n                \'random_state\'    : CFG.state,\n                \'alpha\'           : 0.25,\n                \'n_iter_no_change\': 50,\n                \'tol\'             : 0.0001, \n              }\n           ),\n \n \'HBGR\': HGBR(loss              = \'squared_error\',\n              learning_rate     = 0.05215,\n              max_iter          = 800,\n              max_depth         = 7,\n              min_samples_leaf  = 20,\n              l2_regularization = 1.25,\n              max_bins          = 255,\n              n_iter_no_change  = 75,\n              tol               = 1e-04,\n              verbose           = 0,\n              random_state      = CFG.state\n             )\n};\n\n# Selecting relevant columns for the train and test sets:-\nsel_cols = \\\n[\'Source\', \n \'pressurempa\', \'massfluxkgm2s\', \'demm\', \'dhmm\', \'lengthmm\',\n \'chfexpmwm2\', \'Nb_Null_Ftre\', \n \'sa\', \'sdr\', \n \'author_Weatherhead\', \'geometry_annulus\'\n];\nXtrain, Xtest = Xtrain[sel_cols], Xtest[sel_cols];\n\n# Initializing output tables for the models:-\nmethods   = list(Mdl_Master.keys());\nOOF_Preds = pd.DataFrame(columns = methods);\nMdl_Preds = pd.DataFrame(index = Xtest.index, columns = methods);\nFtreImp   = pd.DataFrame(index = Xtrain.drop(columns = [\'id\', CFG.target, \'Source\', \'Label\'],\n                                             errors = \'ignore\').columns, \n                         columns = methods\n                        );\nScores    = pd.DataFrame(columns = methods);\n\n# Defining the competition metric:-\ndef ScoreMetric(ytrue, ypred) -> np.float32:\n    "This function defines the competition metric for subsequent models";\n    return np.round(mse(ytrue, ypred, squared = False), decimals = 6);\n\nprint();\ncollect();\n')


# In[35]:


get_ipython().run_cell_magic('time', '', '\ndef TrainMdl(method:str):\n    "This function trains the regression models and collates the scores and predictions";\n    \n    global Mdl_Master, Mdl_Preds, OOF_Preds, all_cv, FtreImp, Xtrain, ytrain; \n    \n    model     = Mdl_Master.get(method); \n    cols_drop = [\'id\', \'Source\', \'Label\'];\n    scl_cols  = [col for col in Xtrain.columns if col not in cols_drop];\n    cv        = all_cv.get(CFG.mdlcv_mthd);\n    Xt        = Xtest.copy(deep = True);\n    \n    if CFG.scale_req == "N" and method.upper() in [\'RIDGE\', \'LASSO\', \'SVR\', \'KNNR\']:\n        X, y        = Xtrain, ytrain;\n        scaler      = all_scalers[\'Z\'];\n        X[scl_cols] = scaler.fit_transform(X[scl_cols]);\n        Xt[scl_cols]= scaler.transform(Xt[scl_cols]);\n        PrintColor(f"--> Scaling the data for {method} model");\n\n    if CFG.use_orig_allfolds == "Y":\n        X    = Xtrain.query("Source == \'Competition\'");\n        y    = ytrain.loc[ytrain.index.isin(X.index)]; \n        Orig = pd.concat([Xtrain, ytrain], axis=1).query("Source == \'Original\'");\n        \n    elif CFG.use_orig_allfolds != "Y":\n        X,y = Xtrain.copy(deep = True), ytrain.copy(deep = True);\n                \n    # Initializing I-O for the given seed:-        \n    test_preds = 0;\n    oof_preds  = pd.DataFrame(); \n    scores     = [];\n    ftreimp    = 0;\n          \n    for fold_nb, (train_idx, dev_idx) in enumerate(cv.split(X, y)): \n        Xtr  = X.iloc[train_idx].drop(columns = cols_drop, errors = \'ignore\');   \n        Xdev = X.iloc[dev_idx].loc[X.Source == "Competition"].\\\n        drop(columns = cols_drop, errors = \'ignore\'); \n        ytr  = y.loc[y.index.isin(Xtr.index)];\n        ydev = y.loc[y.index.isin(Xdev.index)];\n\n        if CFG.use_orig_allfolds == "Y":\n            Xtr = pd.concat([Xtr, Orig.drop(columns = [CFG.target, \'Source\'], errors = \'ignore\')], \n                            axis = 0, ignore_index = True);\n            ytr = pd.concat([ytr, Orig[CFG.target]], axis = 0, ignore_index = True);\n          \n        # Fitting the model:- \n        if method.upper() in [\'CBR\', \'LGBMR\', \'XGBR\']:     \n            model.fit(Xtr, ytr, \n                      eval_set = [(Xdev, ydev)], \n                      verbose = 0,\n                      early_stopping_rounds = CFG.nbrnd_erly_stp\n                     );   \n        else:\n            model.fit(Xtr, ytr);\n            \n        # Collecting predictions and scores and post-processing OOF:-\n        dev_preds = model.predict(Xdev);\n        score     = ScoreMetric(ydev, dev_preds);\n        scores.append(score); \n        Scores.loc[fold_nb, method] = np.round(score, decimals= 6);\n        oof_preds = pd.concat([oof_preds,\n                               pd.DataFrame(index   = Xdev.index, \n                                            data    = dev_preds,\n                                            columns = [method])\n                              ],axis=0, ignore_index= False\n                             );  \n    \n        oof_preds = pd.DataFrame(oof_preds.groupby(level = 0)[method].mean());\n        oof_preds.columns = [method];\n        \n        test_preds = test_preds + model.predict(Xt.drop(columns = cols_drop, errors = \'ignore\')); \n        \n        try: \n            ftreimp += model.feature_importances_;\n        except:\n            ftreimp = 0;\n            \n    num_space = 20 - len(method);\n    PrintColor(f"--> {method} {\'-\' * num_space} CV mean = {np.mean(scores):.6f}", \n               color = Fore.MAGENTA);\n    del num_space;\n    \n    OOF_Preds[f\'{method}\'] = oof_preds.values.flatten();\n    Mdl_Preds[f\'{method}\'] = test_preds.flatten()/ (CFG.n_splits * CFG.n_repeats); \n    FtreImp[method]        = ftreimp / (CFG.n_splits * CFG.n_repeats);\n    collect(); \n      \ncollect();\nprint();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Implementing the ML models:-\nif CFG.ML == "Y": \n    for method in tqdm(methods, "ML models----"): TrainMdl(method);\n    clear_output();\n    \n    PrintColor(f"\\nCV scores across methods\\n");\n    display(pd.concat([Scores.mean(axis = 0), Scores.std(axis = 0)], axis=1).\\\n            rename(columns = {0: \'Mean\', 1: \'Std\'}).\\\n            style.format(precision = 6).\\\n            background_gradient(cmap = \'viridis\')\n           );\nelse:\n    PrintColor(f"\\nML models are not needed\\n", color = Fore.RED);\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ML == "Y":\n    OOF_Preds[CFG.target] = \\\n    pd.concat([Xtrain, ytrain], axis=1).\\\n    query("Source == \'Competition\'")[CFG.target].values; \n    \n    try:\n        all_mthd = methods;\n        fig, axes = plt.subplots(len(all_mthd), 2, figsize = (22, len(all_mthd)* 7.0), \n                                 gridspec_kw = {\'wspace\': 0.2, \'hspace\': 0.25},\n                                 width_ratios = [0.75, 0.25])\n        for i, method in enumerate(all_mthd):\n            ax = axes[i, 0];\n            FtreImp[method].plot.barh(color = \'#008ae6\', ax = ax);\n            ax.set_title(f"{method} - importances", **CFG.title_specs);\n\n            ax = axes[i,1];\n            rsq = r2_score(OOF_Preds[CFG.target].values, OOF_Preds[method].values);\n            \n            sns.regplot(data= OOF_Preds[[method, CFG.target]], \n                        y = method, x = f"{CFG.target}", \n                        seed= CFG.state, color = \'#4dffff\', marker = \'o\',\n                        line_kws= {\'linewidth\': 2.25, \'linestyle\': \'--\', \'color\': \'#0000b3\'},\n                        label = f"{method}",\n                        ax = ax,\n                       );\n            ax.set_title(f"{method} RSquare = {rsq:.2%}", **CFG.title_specs);\n            ax.set(ylabel = \'Predictions\', xlabel = \'Actual\')\n            del rsq;\n        \n        plt.tight_layout();\n        plt.show();\n        del all_mthd;\n    \n    except:\n        pass\n    \n    collect();\n    print(); \n')


# <a id="8"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0052cc; border-bottom: 8px solid #cc9966" > ENSEMBLE AND SUBMISSION<br> <div> 

# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndef Objective(trial):\n    "This function defines the objective for the optuna ensemble using variable models";\n    \n    global OOF_Preds, all_cv, ScoreMetric;\n    \n    # Define the weights for the predictions from each model:-\n    opt_ens_mdl = list(OOF_Preds.drop(columns = [CFG.target], errors = \'ignore\').columns);\n    weights  = [trial.suggest_float(f"M{n}", 0.005, 0.995, step = 0.001) \\\n                for n in range(len(opt_ens_mdl))];\n    weights  = np.array(weights)/ sum(weights);\n    \n    # Calculating the CV-score for the weighted predictions on the competition data only:-\n    scores = [];  \n    cv     = all_cv[CFG.mdlcv_mthd];\n    X,y    = OOF_Preds[opt_ens_mdl], OOF_Preds[CFG.target];\n    \n    for fold_nb, (train_idx, dev_idx) in enumerate(cv.split(X,y)):\n        Xtr, Xdev = X.iloc[train_idx], X.iloc[dev_idx];\n        ytr, ydev = y.loc[Xtr.index], y.loc[Xdev.index];\n        scores.append(ScoreMetric(ydev, np.dot(Xdev, weights)));\n        \n    collect();\n    clear_output();\n    return np.mean(scores);\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ensemble_req == "Y":\n    PrintColor(f"{\'-\'* 30} Ensemble with optuna {\'-\'* 30}", color = Fore.MAGENTA);\n    \n    opt_ens_mdl = list(OOF_Preds.drop(columns = [CFG.target], errors = \'ignore\').columns);\n    study = optuna.create_study(direction  = CFG.direction, \n                                study_name = "OptunaEnsemble", \n                                sampler    = TPESampler(seed = CFG.state)\n                               );\n    study.optimize(Objective, \n                   n_trials          = CFG.n_ens_trials, \n                   gc_after_trial    = True,\n                   show_progress_bar = True);\n    \n    weights       = study.best_params;\n    final_weights = np.array(list(weights.values()));\n    final_weights = final_weights/ sum(final_weights);\n    clear_output();\n    \n    PrintColor(f"\\n--> Post ensemble weights\\n");\n    pprint(final_weights, indent = 5, width = 10, depth = 1);\n    \n    PrintColor(f"\\n--> Best ensemble CV score = {study.best_value :.5f}\\n");\n    del weights;\n    \n    # Making weighted predictions on the test set:-  \n    sub_fl[sub_fl.columns[-1]] = np.dot(Mdl_Preds[opt_ens_mdl], final_weights);\n    \n    PrintColor(f"\\n--> Post ensemble test-set predictions\\n");\n    display(sub_fl.head(5).style.format(precision = 4));  \n    \n    sub_fl.to_csv(f"submission_V{CFG.version_nb}.csv", index = None);\n         \ncollect();\nprint();    \n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ML == "Y":  \n    OOF_Preds.add_prefix(f"V{CFG.version_nb}_").to_csv(f"OOF_Preds_V{CFG.version_nb}.csv");\n    Mdl_Preds.index = sub_fl.index;\n    Mdl_Preds.add_prefix(f"V{CFG.version_nb}_").to_csv(f"Mdl_Preds_V{CFG.version_nb}.csv");\n    Scores.to_csv(f"Scores_V{CFG.version_nb}.csv");\n    \ncollect();\nprint();\n')


# <a id="9"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0052cc; border-bottom: 8px solid #cc9966" > NEXT STEPS<br> <div> 

# <div style= "font-family: Cambria; letter-spacing: 0px; color:#000000; font-size:110%; text-align:left;padding:3.0px; background: #f2f2f2" >
# 1. Better feature engineering. This competition is driven by feature engineering almost entirely<br>
# 2. Better experiments with scaling, encoding with categorical columns. This seems to have some promise<br>
# 3. Better model tuning<br>
# 4. Better emphasis on secondary features<br>
# 5. Adding more algorithms and new methods to the model suite<br>
# </div>
