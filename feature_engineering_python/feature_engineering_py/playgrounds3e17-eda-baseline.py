#!/usr/bin/env python
# coding: utf-8

# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0052cc; border-bottom: 8px solid #cc9966" > TABLE OF CONTENTS<br><div>  
# * [IMPORTS](#1)
# * [INTRODUCTION](#2)
#     * [CONFIGURATION](#2.1)
#     * [CONFIGURATION PARAMETERS](#2.2)    
#     * [DATASET COLUMNS](#2.3)
# * [PREPROCESSING](#3)
# * [ADVERSARIAL CV](#4)
# * [EDA AND VISUALS](#5)
#     * [DUPLICATES ANALYSIS](#5.1)
#     * [TARGET PLOTS](#5.2) 
#     * [PAIR-PLOTS](#5.3)
#     * [CATEGORY COLUMN PLOTS](#5.4)
#     * [CONTINUOUS COLUMN ANALYSIS](#5.5)
#     * [PRODUCT ID ANALYSIS](#5.6)
#     * [UNIVARIATE RELATIONS](#5.7)
#     * [BINARY COLUMN ANALYSIS](#5.8)
#     * [INFERENCES](#5.9)    
# * [DATA TRANSFORMS](#6)
# * [MODEL TRAINING](#7)    
# * [ENSEMBLE AND SUBMISSION](#8)  
# * [PLANNED WAY FORWARD](#9)     

# <a id="1"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0052cc; border-bottom: 8px solid #cc9966" > IMPORTS<br> <div> 

# In[2]:


get_ipython().run_cell_magic('time', '', "\n# General library imports:-\n!pip install --upgrade scipy;\n\nimport pandas as pd;\nimport numpy as np;\nfrom scipy.stats import mode, kstest, normaltest, shapiro, anderson, jarque_bera;\nfrom collections import Counter;\nfrom itertools import product;\nfrom colorama import Fore, Style, init;\nfrom warnings import filterwarnings;\nfilterwarnings('ignore');\n\nfrom tqdm.notebook import tqdm;\nfrom IPython.display import clear_output;\n\nimport seaborn as sns;\nimport matplotlib.pyplot as plt;\nfrom gc import collect;\nfrom pprint import pprint;\n\npd.set_option('display.max_columns', 50);\npd.set_option('display.max_rows', 50);\n\nprint();\ncollect();\nclear_output();\n")


# In[3]:


get_ipython().run_cell_magic('time', '', '\n# Importing model and pipeline specifics:-\n\n!pip install -q category_encoders;\nfrom category_encoders import OrdinalEncoder, OneHotEncoder;\n\n# Pipeline specifics:-\nfrom sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler;\nfrom sklearn.model_selection import (RepeatedStratifiedKFold as RSKF, \n                                     StratifiedKFold as SKF,\n                                     KFold, \n                                     RepeatedKFold as RKF, \n                                     cross_val_score);\nfrom sklearn.inspection import permutation_importance, PartialDependenceDisplay as PDD;\nfrom sklearn.feature_selection import mutual_info_classif, RFE;\nfrom sklearn.pipeline import Pipeline;\nfrom sklearn.base import BaseEstimator, TransformerMixin;\nfrom sklearn.compose import ColumnTransformer;\n\n# ML Model training:-\nfrom sklearn.calibration import CalibrationDisplay as Clb;\nfrom sklearn.metrics import roc_auc_score, auc;\nfrom sklearn.svm import SVC;\nfrom xgboost import XGBClassifier, XGBRegressor;\nfrom lightgbm import LGBMClassifier, LGBMRegressor, log_evaluation;\nfrom catboost import CatBoostRegressor, CatBoostClassifier;\nfrom sklearn.ensemble import (RandomForestRegressor as RFR,\n                              ExtraTreesRegressor as ETR,\n                              GradientBoostingRegressor as GBR,\n                              HistGradientBoostingRegressor as HGBR,\n                              RandomForestClassifier as RFC,\n                              ExtraTreesClassifier as ETC,\n                              GradientBoostingClassifier as GBC,\n                              HistGradientBoostingClassifier as HGBC,\n                             );\n\n# Ensemble and tuning:-\nimport optuna;\nfrom optuna import Trial, trial, create_study;\nfrom optuna.samplers import TPESampler, CmaEsSampler;\noptuna.logging.set_verbosity = optuna.logging.ERROR;\n\nfrom sklearn import set_config; \nset_config(transform_output = "pandas");\n\nclear_output();\nprint();\ncollect();\n')


# In[4]:


get_ipython().run_cell_magic('time', '', '\n# Setting rc parameters in seaborn for plots and graphs- \n# Reference - https://matplotlib.org/stable/tutorials/introductory/customizing.html:-\n# To alter this, refer to matplotlib.rcParams.keys()\n\nsns.set({"axes.facecolor"       : "#ffffff",\n         "figure.facecolor"     : "#ffffff",\n         "axes.edgecolor"       : "#000000",\n         "grid.color"           : "#ffffff",\n         "font.family"          : [\'Cambria\'],\n         "axes.labelcolor"      : "#000000",\n         "xtick.color"          : "#000000",\n         "ytick.color"          : "#000000",\n         "grid.linewidth"       : 0.75,  \n         "grid.linestyle"       : "--",\n         "axes.titlecolor"      : \'#0099e6\',\n         \'axes.titlesize\'       : 8.5,\n         \'axes.labelweight\'     : "bold",\n         \'legend.fontsize\'      : 7.0,\n         \'legend.title_fontsize\': 7.0,\n         \'font.size\'            : 7.5,\n         \'xtick.labelsize\'      : 7.5,\n         \'ytick.labelsize\'      : 7.5,        \n        });\n\n# Color printing    \ndef PrintColor(text:str, color = Fore.BLUE, style = Style.BRIGHT):\n    "Prints color outputs using colorama using a text F-string";\n    print(style + color + text + Style.RESET_ALL); \n    \nprint();\ncollect();\n')


# <a id="2"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0052cc; border-bottom: 8px solid #cc9966" > INTRODUCTION<br><div> 

# | Version<br>Number | Version Details | Best CV score| Single/ Ensemble|
# | :-: | --- | :-: | :-: |
# | **V1** |* EDA, plots and secondary features<br>* No scaling<br> * Used original data<br>* Tree based ML models and Optuna ensemble<br>* Post-processing predictions|0.966460|Optuna ensemble |
# | **V2** |* EDA, plots and secondary features<br>* No scaling<br> * Used only training data<br>* Tree based ML models and Optuna ensemble<br>* Post-processing predictions|0.966408|Optuna ensemble|
# | **V3** |* EDA, plots and secondary features<br>* No scaling<br> * Used original data entirely in all folds<br>* Tree based ML models and Optuna ensemble<br>* Post-processing predictions|0.967403|Optuna ensemble|
# | **V4** |* EDA, plots and secondary features<br>* No scaling<br> * Used original data entirely in all folds<br>* Tree based ML models and Optuna ensemble<br>* Post-processing predictions<br>* Blended my submission with other public notebooks||Optuna ensemble|
# 
# 

# <a id="2.1"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > CONFIGURATION<br><div> 

# In[5]:


get_ipython().run_cell_magic('time', '', '\n# Configuration class:-\nclass CFG:\n    "Configuration class for parameters and CV strategy for tuning and training";\n    \n    # Data preparation:-   \n    version_nb         = 4;\n    test_req           = "N";\n    gpu_switch         = "OFF"; \n    state              = 42;\n    target             = "Machinefailure";\n    episode            = 17;\n    path               = f"/kaggle/input/playground-series-s3e{episode}/";\n    orig_path          = f"/kaggle/input/machine-failure-predictions/machine failure.csv";\n    adv_cv_req         = "Y";\n    ftre_plots_req     = "Y";\n    ftre_imp_req       = "Y";\n    \n    # Data transforms and scaling:-    \n    conjoin_orig_data  = "Y";\n    sec_ftre_req       = "Y";\n    scale_req          = "N";\n    # NOTE---Keep a value here even if scale_req = N, this is used for linear models:-\n    scl_method         = "Z"; \n    enc_method         = \'Label\';\n    \n    # Model Training:- \n    baseline_req       = "N";\n    pstprcs_oof        = "Y";\n    pstprcs_train      = "Y";\n    ML                 = "Y";\n    use_orig_allfolds  = "Y";\n    n_splits           = 5 ;\n    n_repeats          = 1 ;\n    nbrnd_erly_stp     = 200 ;\n    mdlcv_mthd         = \'RSKF\';\n    \n    # Ensemble:-    \n    ensemble_req       = "Y";\n    enscv_mthd         = "RSKF";\n    metric_obj         = \'maximize\';\n    ntrials            = 10 if test_req == "Y" else 250;\n    \n    # Global variables for plotting:-\n    grid_specs = {\'visible\': True, \'which\': \'both\', \'linestyle\': \'--\', \n                           \'color\': \'lightgrey\', \'linewidth\': 0.75};\n    title_specs = {\'fontsize\': 9, \'fontweight\': \'bold\', \'color\': \'tab:blue\'};\n\nprint();\nPrintColor(f"--> Configuration done for version number = {CFG.version_nb}");\n\nif CFG.test_req == "Y":\n    PrintColor(f"--> This is a test-run\\n", color = Fore.RED); \n\ncollect();\n')


# In[6]:


get_ipython().run_cell_magic('time', '', '\n# Defining functions to be used throughout the code for common tasks:-\n\n# Scaler to be used for continuous columns:- \nall_scalers = {\'Robust\': RobustScaler(), \n               \'Z\': StandardScaler(), \n               \'MinMax\': MinMaxScaler()\n              };\nscaler      = all_scalers.get(CFG.scl_method);\n\n# Commonly used CV strategies for later usage:-\nall_cv= {\'KF\'  : KFold(n_splits= CFG.n_splits, shuffle = True, random_state= CFG.state),\n         \'RKF\' : RKF(n_splits= CFG.n_splits, n_repeats = CFG.n_repeats, random_state= CFG.state),\n         \'RSKF\': RSKF(n_splits= CFG.n_splits, n_repeats = CFG.n_repeats, random_state= CFG.state),\n         \'SKF\' : SKF(n_splits= CFG.n_splits, shuffle = True, random_state= CFG.state)\n        };\n\n# Defining the competition metric:-\ndef ScoreMetric(ytrue, ypred)-> float:\n    """\n    This function calculates the metric for the competition. \n    ytrue- ground truth array\n    ypred- predictions\n    returns - metric value (float)\n    """;\n    return roc_auc_score(ytrue, ypred);\n\ndef PostProcessPred(preds, post_process = "Y"):\n    """\n    We limit the prediction values between 0.00001 and 0.99999 herewith. \n    This is specifically useful for regressions that could produce out-of-range predictions\n    """;\n    \n    if post_process == "Y": return np.clip(preds, a_min = 0.00001, a_max = 0.999999);\n    else: return preds;\n    \ncollect();\nprint();\n')


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
# |  scale_req        | Scaling required                                        | Y/N                   |
# |  scl_method       | Scaling method                                          | Z/ Robust/ MinMax     |
# |  baseline_req     | Baseline model required                                 | Y/N                   |
# |  pstprcs_oof      | Post-process OOF after model training                   | Y/N                   |
# |  pstprcs_train    | Post-process OOF during model training for dev-set      | Y/N                   |
# |  ML               | Machine Learning Models                                 | Y/N                   |
# |  use_orig_allfolds| Use original data across all folds                      | Y/N                   |
# |  n_splits         | Number of CV splits                                     | integer               |
# |  n_repeats        | Number of CV repeats                                    | integer               |
# |  nbrnd_erly_stp   | Number of early stopping rounds                         | integer               |
# |  mdl_cv_mthd      | Model CV method name                                    | RKF/ RSKF/ SKF/ KFold |

# <a id="2.3"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > DATASET AND COMPETITION DETAILS<br><div>
#     
# **Data columns**<br>
# This is referred from the link - https://www.kaggle.com/competitions/playground-series-s3e17/discussion/416765
#     
# 
# 
# | Column              | Description                                             | 
# | ---                 | ---                                                     |
# | UDI                 | Device identifier                                       |
# | Production Id       | Production ID                                           |
# | Type                | Type of product/device (L/M/H)                          |
# | Air Temperature     | Temperature                                             |
# | Process Temperature | Process temperature                                     |
# | Rotational Speed    | Rotational speed                                        |
# | Torque              | Extent of torque                                        |
# | Tool Wear           | Time unit needed to wear down the product/tool          |
# | **Machine Failure** | Binary feature- Machine failure - **target**            |
# | TWF                 | Tool wear failure                                       |
# | HDF                 | Heat dissipation failure                                |
# | PWF                 | Power Failure                                           |
# | OSF                 | Over-stain                                              | 
# | RNF                 | Random failure                                          |
# 
# **Competition details and notebook objectives**<br>
# 1. This is a binary classification challenge to predict machine failures using the provided features. **GINI** is the metric for the challenge<br>
# 2. In this starter notebook, we start the assignment with a detailed EDA, feature plots, interaction effects, adversarial CV analysis and develop starter models to initiate the challenge. We will also incorporate other opinions and approaches as we move along the challenge. 

# <a id="3"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0052cc; border-bottom: 8px solid #cc9966" > PREPROCESSING<br><div> 

# In[7]:


get_ipython().run_cell_magic('time', '', '\nPrintColor(f"\\n{\'-\'*20} Data Preprocessing {\'-\'*20}\\n", \n           color = Fore.MAGENTA);\n\n# Reading the train-test datasets:-\ntrain    = pd.read_csv(CFG.path + f"train.csv", index_col = \'id\');\ntest     = pd.read_csv(CFG.path + f"test.csv", index_col = \'id\');\noriginal = pd.read_csv(CFG.orig_path).drop(columns = [\'UDI\'], errors = \'ignore\');\n\n# Importing the sample submission file:-\nsub_fl = pd.read_csv(CFG.path + f"sample_submission.csv");\n\n# Replacing spaces in column names:-\nfor df in [train, test, original]: \n    df.columns = df.columns.str.replace(r"\\s+|\\[\\D+", \'\');\n\nif CFG.test_req == "Y":\n    PrintColor(f"---> We are testing the code with 5% data sample", color = Fore.RED);\n    train     = train.groupby([CFG.target]).sample(frac = 0.05);\n    original  = original.groupby([CFG.target]).sample(frac = 0.05);\n    test      = test.sample(frac = 0.05);\n    sub_fl    = sub_fl.loc[sub_fl.id.isin(test.index)];\n    \nelse: \n    PrintColor(f"---> We are not testing the code- this is an actual code run", color = Fore.RED);\n    \noriginal.index += max(test.index) + 1;\noriginal.index.name = \'id\';\n\ntrain[\'Source\'], test[\'Source\'], original[\'Source\'] = "Competition", "Competition", "Original";\nPrintColor(f"\\nData shapes- [train, test, original]-- {train.shape} {test.shape} {original.shape}\\n");\n\n# Creating dataset information:\nPrintColor(f"\\nTrain information\\n");\ndisplay(train.info());\nPrintColor(f"\\nTest information\\n")\ndisplay(test.info());\nPrintColor(f"\\nOriginal data information\\n")\ndisplay(original.info());\nprint();\n\n# Displaying column description:-\nPrintColor(f"\\nTrain description\\n");\ndisplay(train.describe(percentiles= [0.05, 0.25, 0.50, 0.75, 0.9, 0.95, 0.99]).\\\n        transpose().\\\n        drop(columns = [\'count\'], errors = \'ignore\').\\\n        drop([CFG.target], axis=0, errors = \'ignore\').\\\n        style.format(precision = 2));\n\nPrintColor(f"\\nTest description\\n");\ndisplay(test.describe(percentiles= [0.05, 0.25, 0.50, 0.75, 0.9, 0.95, 0.99]).\\\n        transpose().\\\n        drop(columns = [\'count\'], errors = \'ignore\').\\\n        drop([CFG.target], axis=0, errors = \'ignore\').\\\n        style.format(precision = 2));\n\nPrintColor(f"\\nOriginal description\\n");\ndisplay(original.describe(percentiles= [0.05, 0.25, 0.50, 0.75, 0.9, 0.95, 0.99]).\\\n        transpose().\\\n        drop(columns = [\'count\'], errors = \'ignore\').\\\n        drop([CFG.target], axis=0, errors = \'ignore\').\\\n        style.format(precision = 2));\n\n# Collating the column information:-\nstrt_ftre = test.columns;\nPrintColor(f"\\nStarting columns\\n");\ndisplay(strt_ftre);\n\n# Creating a copy of the datasets for further use:-\ntrain_copy, test_copy, orig_copy = \\\ntrain.copy(deep= True), test.copy(deep = True), original.copy(deep = True);\n\n# Dislaying the unique values across train-test-original:-\nPrintColor(f"\\nUnique values\\n");\n_ = pd.concat([train.nunique(), test.nunique(), original.nunique()], axis=1);\n_.columns = [\'Train\', \'Test\', \'Original\'];\ndisplay(_.style.background_gradient(cmap = \'Blues\').format(formatter = \'{:,.0f}\'));\n\n# Normality check:-\ncols = list(strt_ftre[3:-1]);\nPrintColor(f"\\nShapiro Wilk normality test analysis\\n");\npprint({col: [np.round(shapiro(train[col]).pvalue,decimals = 4), \n              np.round(shapiro(test[col]).pvalue,4) if col != CFG.target else np.NaN,\n              np.round(shapiro(original[col]).pvalue,4)] for col in cols\n       }, indent = 5, width = 100, depth = 2, compact= True);\n\nPrintColor(f"\\nNormal-test normality test analysis\\n");\npprint({col: [np.round(normaltest(train[col]).pvalue,decimals = 4), \n              np.round(normaltest(test[col]).pvalue,4) if col != CFG.target else np.NaN,\n              np.round(normaltest(original[col]).pvalue,4)] for col in cols\n       }, indent = 5, width = 100, depth = 2, compact= True);\n\nPrintColor(f"\\nK-S normality test analysis\\n");\npprint({col: [np.round(kstest(train[col], cdf = \'norm\').pvalue,decimals = 4), \n              np.round(kstest(test[col], cdf = \'norm\').pvalue,4) if col != CFG.target else np.NaN,\n              np.round(kstest(original[col], cdf = \'norm\').pvalue,4)] for col in cols\n       }, indent = 5, width = 100, depth = 2, compact= True);\n\n# Enlisting category and numeric columns:- \ncat_cols = [ \'Type\', \'TWF\', \'HDF\', \'PWF\', \'OSF\',\'RNF\'];\ncont_cols= [col for col in strt_ftre[2:-1] if col not in cat_cols];\n\nPrintColor(f"\\nCategory and continuous columns\\n");\npprint(cat_cols, depth = 1, width = 10, indent = 5);\nprint();\npprint(cont_cols, depth = 1, width = 10, indent = 5);\n\nprint();\n')


# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > INFERENCES<br> <div>

# <div style= "font-family: Cambria; letter-spacing: 0px; color:#000000; font-size:110%; text-align:left;padding:3.0px; background: #f2f2f2" >
# 1. All the columns are numerical, except for product-type and product-ID. These need to studied better to encode appropriately<br>
# 2. We do not have any nulls in the data<br>
# 3. All columns are non-normal. Torque shows up normality chracteristics on a couple of tests<br>
# 4. The synthetic data is nearly 22.7 times the original data, creating a potential quasi-duplicate row issue. Duplicate handling could be a key challenge in this case<br>
# 5. Binary target is imbalanced. Handling class imbalance is key to success
# </div>

# <a id="4"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0052cc; border-bottom: 8px solid #cc9966" > ADVERSARIAL CV<br><div>

# In[38]:


get_ipython().run_cell_magic('time', '', '\n# Performing adversarial CV between the 2 specified datasets:-\ndef Do_AdvCV(df1:pd.DataFrame, df2:pd.DataFrame, source1:str, source2:str):\n    "This function performs an adversarial CV between the 2 provided datasets if needed by the user";\n    \n    # Adversarial CV per column:-\n    ftre = test.select_dtypes(include = np.number).\\\n    drop(columns = [\'id\', CFG.target, "Source"], errors = \'ignore\').columns;\n    adv_cv = {};\n\n    for col in ftre:\n        PrintColor(f"---> Current feature = {col}", style = Style.NORMAL);\n        shuffle_state = np.random.randint(low = 10, high = 100, size= 1);\n\n        full_df = \\\n        pd.concat([df1[[col]].assign(Source = source1), df2[[col]].assign(Source = source2)], \n                  axis=0, ignore_index = True).\\\n        sample(frac = 1.00, random_state = shuffle_state);\n\n        full_df = full_df.assign(Source_Nb = full_df[\'Source\'].eq(source2).astype(np.int8));\n\n        # Checking for adversarial CV:-\n        model = LGBMClassifier(random_state = CFG.state, max_depth = 6, learning_rate = 0.05);\n        cv    = all_cv[\'SKF\'];\n        score = np.mean(cross_val_score(model, \n                                        full_df[[col]], \n                                        full_df.Source_Nb, \n                                        scoring= \'roc_auc\', \n                                        cv= cv)\n                       );\n        adv_cv.update({col: round(score, 4)});\n        collect();\n    \n    del ftre;\n    \n    PrintColor(f"\\nResults\\n");\n    pprint(adv_cv, indent = 5, width = 20, depth = 1);\n    collect();\n    \n    fig, ax = plt.subplots(1,1,figsize = (12, 5));\n    pd.Series(adv_cv).plot.bar(color = \'tab:blue\', ax = ax);\n    ax.axhline(y = 0.60, color = \'red\', linewidth = 2.75);\n    ax.grid(**CFG.grid_specs); \n    plt.yticks(np.arange(0.0, 0.81, 0.05));\n    plt.show();\n    \n# Implementing the adversarial CV:-\nif CFG.adv_cv_req == "Y":\n    PrintColor(f"\\n---------- Adversarial CV - Train vs Original ----------\\n", \n               color = Fore.MAGENTA);\n    Do_AdvCV(df1 = train, df2 = original, source1 = \'Train\', source2 = \'Original\');\n    \n    PrintColor(f"\\n---------- Adversarial CV - Train vs Test ----------\\n", \n               color = Fore.MAGENTA);\n    Do_AdvCV(df1 = train, df2 = test, source1 = \'Train\', source2 = \'Test\');\n    \n    PrintColor(f"\\n---------- Adversarial CV - Original vs Test ----------\\n", \n               color = Fore.MAGENTA);\n    Do_AdvCV(df1 = train, df2 = test, source1 = \'Original\', source2 = \'Test\');   \n    \nif CFG.adv_cv_req == "N":\n    PrintColor(f"\\nAdversarial CV is not needed\\n", color = Fore.RED);\n    \ncollect();\nprint();\n')


# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > INFERENCES<br><div>

# <div style= "font-family: Cambria; letter-spacing: 0px; color:#000000; font-size:110%; text-align:left;padding:3.0px; background: #f2f2f2" >
# 1. Train-test belong to the same distribution, we can perhaps rely on the CV score<br>
# 2. We need to further check the train-original distribution further, adversarial validation results indicate that we can use the original dataset<br>
# </div>

# <a id="5"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0052cc; border-bottom: 8px solid #cc9966" > VISUALS AND EDA <br><div> 
#  

# <a id="5.1"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > DUPLICATES ANALYSIS<br><div>

# In[39]:


get_ipython().run_cell_magic('time', '', '\nPrintColor(f"\\n-------- Duplicates analysis --------\\n", color = Fore.MAGENTA);\n\ntry: del _;\nexcept Exception as e: pass;\n\n_ = train.loc[train[strt_ftre[0:-1]].duplicated(keep = \'first\')];\nPrintColor(f"Train set duplicates = {len(_)} rows");\n\n_ = test.loc[test[strt_ftre[0:-1]].duplicated(keep = \'first\')];\nPrintColor(f"Test set duplicates = {len(_)} rows");\n\n_ = pd.concat([train[strt_ftre[0:-1]], test[strt_ftre[0:-1]]], axis= 0, ignore_index = False);\n_ = len(_.loc[_.duplicated(keep = \'first\')]);\nPrintColor(f"Train-Test set combined duplicates = {_} rows");\n\ndup_df = test.reset_index().merge(train.drop(columns = [\'Source\'], errors = \'ignore\').reset_index(), \n                                  how     = \'inner\', \n                                  on      = list(strt_ftre[0:-1]),\n                                  suffixes = {\'\',\'_train\'}\n                                 )[[\'id\', \'id_train\', CFG.target]];\n\nPrintColor(f"\\nDuplicated rows between train and test-\\n")\ndisplay(dup_df.head(5).style.format(precision = 2))\n\nprint();\ncollect();\n')


# <a id="5.2"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > TARGET PLOT<br><div>

# In[40]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ftre_plots_req == "Y":\n    fig, axes = plt.subplots(1,2, figsize = (12, 5), sharey = True, gridspec_kw = {\'wspace\': 0.25});\n    \n    for i, df in tqdm(enumerate([train, original]), "Target balance ---> "):\n        ax= axes[i];\n        a = df[CFG.target].value_counts(normalize = True);\n        _ = ax.pie(x = a , labels = a.index.values, \n                   explode      = [0.0, 0.3], \n                   startangle   = 40, \n                   shadow       = True, \n                   colors       = [\'#3377ff\', \'#66ffff\'], \n                   textprops    = {\'fontsize\': 7, \'fontweight\': \'bold\', \'color\': \'black\'},\n                   pctdistance  = 0.60, \n                   autopct = \'%1.1f%%\'\n                  );\n        df_name = \'Train\' if i == 0 else "Original";\n        _ = ax.set_title(f"\\n{df_name} data- target\\n", **CFG.title_specs);\n\n    plt.tight_layout();\n    plt.show();\n    \ncollect();\nprint();\n')


# <a id="5.3"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > PAIRPLOTS<br><div>

# In[41]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ftre_plots_req == "Y":\n    PrintColor(f"\\n{\'-\' * 20} Train data- pair plots {\'-\' * 20}\\n");\n    _ = sns.pairplot(data = train.drop(columns = [\'id\',\'Source\', CFG.target], errors = \'ignore\'), \n                     diag_kind = \'kde\', \n                     markers   = \'o\', \n                     plot_kws  = {\'color\': \'#33bbff\'},\n                     corner    = True\n                    );\n\nprint();\ncollect();\n')


# In[42]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ftre_plots_req == "Y":\n    PrintColor(f"\\n{\'-\' * 20} Original data- pair plots {\'-\' * 20}\\n");\n    _ = sns.pairplot(data = original.drop(columns = [\'id\',\'Source\', CFG.target, "UDI"], errors = \'ignore\'), \n                     diag_kind = \'kde\', \n                     markers   = \'o\', \n                     plot_kws  = {\'color\': \'#4d88ff\'},\n                     corner    = True\n                    );\nprint();\ncollect();\n')


# <a id="5.4"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > CATEGORY COLUMN PLOTS<br><div>

# In[43]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ftre_plots_req == "Y":\n    fig, axes = plt.subplots(len(cat_cols), 2, figsize = (15, len(cat_cols)* 5.5), \n                             gridspec_kw = {\'wspace\': 0.2, \'hspace\': 0.2});\n\n    for i, col in enumerate(cat_cols):\n        ax = axes[i, 0];\n        a = train[col].value_counts();\n        ax.pie(x = a , \n               labels        = a.index.values, \n               explode       = [0.15]*len(a), \n               colors        = sns.color_palette(\'pastel\'),\n               startangle    = 40, \n               shadow        = True, \n               textprops     = {\'fontsize\': 6.5, \'fontweight\': \'bold\', \'color\': \'black\'},\n               pctdistance   = 0.5, \n               labeldistance = 1.25,\n               autopct       = \'%1.2f%%\'\n              );\n        ax.set_title(f"{col}_Train", **CFG.title_specs);\n\n        ax = axes[i, 1];\n        a = test[col].value_counts();\n        ax.pie(x = a , \n               labels        = a.index.values, \n               explode       = [0.15]*len(a), \n               colors        = sns.color_palette(\'pastel\'),\n               startangle    = 40, \n               shadow        = True, \n               textprops     = {\'fontsize\': 6.5, \'fontweight\': \'bold\', \'color\': \'black\'},\n               pctdistance   = 0.5, \n               labeldistance = 1.20,\n               autopct       = \'%1.2f%%\'\n              );\n        ax.set_title(f"{col}_Test", **CFG.title_specs);\n        del a;\n\n    plt.tight_layout();\n    plt.show();\n    \nprint();\ncollect();\n')


# <a id="5.5"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > CONTINUOUS COLUMN PLOTS<br><div>

# In[44]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ftre_plots_req == "Y":\n    df = pd.concat([train[cont_cols].assign(Source = \'Train\'), \n                    test[cont_cols].assign(Source = \'Test\')], \n                   axis=0, ignore_index = True\n                  );\n    \n    fig, axes = plt.subplots(len(cont_cols), 3 ,figsize = (12, len(cont_cols) * 4.2), \n                             gridspec_kw = {\'hspace\': 0.35, \'wspace\': 0.3, \'width_ratios\': [0.80, 0.20, 0.20]});\n    \n    for i,col in enumerate(cont_cols):\n        ax = axes[i,0];\n        sns.kdeplot(data = df[[col, \'Source\']], x = col, hue = \'Source\', \n                    palette = [\'#0039e6\', \'#ff5500\'], \n                    ax = ax, linewidth = 2.25\n                   );\n        ax.set_title(f"\\n{col}", **CFG.title_specs);\n        ax.grid(**CFG.grid_specs);\n        ax.set(xlabel = \'\', ylabel = \'\');\n        \n        ax = axes[i,1];\n        sns.boxplot(data = df.loc[df.Source == \'Train\', [col]], y = col, width = 0.25,\n                    color = \'#33ccff\', saturation = 0.90, linewidth = 0.90, \n                    fliersize= 2.25,\n                    ax = ax);\n        ax.set(xlabel = \'\', ylabel = \'\');\n        ax.set_title(f"Train", **CFG.title_specs);\n        \n        ax = axes[i,2];\n        sns.boxplot(data = df.loc[df.Source == \'Test\', [col]], y = col, width = 0.25, fliersize= 2.25,\n                    color = \'#80ffff\', saturation = 0.6, linewidth = 0.90, \n                    ax = ax); \n        ax.set(xlabel = \'\', ylabel = \'\');\n        ax.set_title(f"Test", **CFG.title_specs);\n              \n    plt.suptitle(f"\\nDistribution analysis- continuous columns\\n", **CFG.title_specs, \n                 y = 0.92, x = 0.57);\n    plt.tight_layout();\n    plt.show();\n    \nprint();\ncollect();\n')


# In[45]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ftre_plots_req == "Y":\n    fig, axes = plt.subplots(len(cont_cols), len(cat_cols), \n                             figsize = (len(cat_cols) * 5, len(cont_cols) * 5), \n                             gridspec_kw = {\'hspace\': 0.30, \'wspace\': 0.25}\n                            );  \n\n    for i, col in enumerate(cont_cols):\n        for j, c in enumerate(cat_cols):\n            ax = axes[i, j];\n            sns.kdeplot(data      = train, \n                        x         = col, \n                        hue       = c, \n                        palette   = \'inferno\', \n                        ax        = ax, \n                        linewidth = 2.5,\n                       );\n\n            ax.set_title(f"{col} - {c}", **CFG.title_specs);\n            ax.grid(**CFG.grid_specs);\n            ax.set(xlabel = \'\', ylabel = \'\');\n\n    plt.tight_layout();\n    plt.suptitle(f"Continuous columns versus category column distribution", \n                 y = 0.90, **CFG.title_specs);\n    plt.show();\n    \ncollect();\nprint();\n')


# <a id="5.6"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > PRODUCT ID ANALYSIS<br><div>
# 
# We could split product ID into 2 components- the second and third digits and the rest. The rest could be perhaps summed up to create fewer categories perhaps.

# In[46]:


get_ipython().run_cell_magic('time', '', '\ndef AnalysePID(X:pd.DataFrame, df_label:str):\n    """\n    This function analyzes the product ID, makes secondary features and checks the distribution\n    """;\n    \n    df = X.copy(deep = True);\n    \n    df[\'ProductID_Comp1\'], df["ProductID_Comp2"] = \\\n    (df[\'ProductID\'].str[1:3].astype(np.int8), \n    df[\'ProductID\'].str[3].astype(np.int8) + df[\'ProductID\'].str[4].astype(np.int8) + df[\'ProductID\'].str[5].astype(np.int8));\n\n    df1 = df.groupby([\'Type\', \'ProductID_Comp1\']).size().reset_index();\n    df2 = df.groupby([\'Type\', \'ProductID_Comp2\']).size().reset_index();\n\n    fig, axes = plt.subplots(2,3, figsize = (18, 10), gridspec_kw = {\'hspace\': 0.2, "wspace": 0.2})\n    for i, t in enumerate(df[\'Type\'].unique()):\n        ax = axes[0, i];\n        df1.loc[df1.Type == t].plot.bar(x = \'ProductID_Comp1\', ax = ax, color = \'#00ace6\');\n        ax.legend(\'\');\n        ax.set_xlabel(\'\');\n        ax.set_title(f" Comp1 {t}", **CFG.title_specs);\n\n        ax = axes[1, i];\n        df2.loc[df2.Type == t].plot.bar(x = \'ProductID_Comp2\', ax = ax, color = \'#80aaff\');\n        ax.legend(\'\');\n        ax.set_xlabel(\'\');\n        ax.set_title(f"Comp2 {t}", **CFG.title_specs);   \n\n    plt.tight_layout();\n    plt.suptitle(f"{df_label} product ID analysis", \n                 y= 0.96, \n                 color = \'#007acc\', fontsize = 14, fontweight = \'bold\');\n    plt.show();\n    collect();\n    print(\'\\n\');\n    \n# Analyzing the product ID column:-\nAnalysePID(train, "Train");\nAnalysePID(test, "Test");\nAnalysePID(original, "Original");\n\ncollect();  \nprint();\n')


# <a id="5.7"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > FEATURE INTERACTION AND UNIVARIATE RELATIONS<br><div>
#     
# We aim to do the below herewith<br>
# 1. Correlation<br>
# 2. Mutual information<br>
# 3. Leave One Out model<br>

# In[47]:


get_ipython().run_cell_magic('time', '', '\ndef MakeCorrPlot(df: pd.DataFrame, data_label:str, figsize = (30, 9)):\n    """\n    This function develops the correlation plots for the given dataset\n    """;\n    \n    fig, axes = plt.subplots(1,2, figsize = figsize, gridspec_kw = {\'hspace\': 0.2, \'wspace\': 0.1},\n                             sharey = True\n                            );\n    \n    for i, method in enumerate([\'pearson\', \'spearman\']):\n        corr_ = df.drop(columns = [\'id\', \'Source\'], errors = \'ignore\').corr(method = method);\n        ax = axes[i];\n        sns.heatmap(data = corr_,  \n                    annot= True,\n                    fmt= \'.2f\', \n                    cmap = \'Blues\',\n                    annot_kws= {\'fontweight\': \'bold\',\'fontsize\': 7.5}, \n                    linewidths= 1.5, \n                    linecolor=\'white\', \n                    cbar= False, \n                    mask= np.triu(np.ones_like(corr_)),\n                    ax= ax\n                   );\n        ax.set_title(f"\\n{method.capitalize()} correlation- {data_label}\\n", **CFG.title_specs);\n        \n    collect();\n    print();\n\n# Implementing correlation analysis:-\nMakeCorrPlot(df = train, data_label = \'Train\', figsize = (15, 4.5));\nMakeCorrPlot(df = original, data_label = \'Original\', figsize = (15, 4.5));\nMakeCorrPlot(df = test, data_label = \'Test\', figsize = (15, 4.5));\n\nprint();\ncollect();\n')


# In[48]:


get_ipython().run_cell_magic('time', '', '\nMutInfoSum = {};\nfor i, df in enumerate([train, original]):\n    MutInfoSum.update({\'Train\' if i == 0 else \'Original\': \n                       mutual_info_classif(df[strt_ftre[2:-1]], df[CFG.target],random_state = CFG.state)\n                      });\n\nMutInfoSum = pd.DataFrame(MutInfoSum, index = strt_ftre[2:-1]);\n\nfig, axes = plt.subplots(1,2, figsize = (14, 4), gridspec_kw = {\'wspace\': 0.2});\nfor i in range(2):\n    MutInfoSum.iloc[:, i].plot.bar(ax = axes[i], color = \'#4080bf\');\n    axes[i].set_title(f"{MutInfoSum.columns[i]}- Mutual Information", **CFG.title_specs);\n\nplt.tight_layout();\nplt.show();\nprint();\ncollect();\n')


# <a id="5.8"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > BINARY COLUMN ANALYSIS<br><div>

# In[49]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ftre_plots_req == "Y":\n    fig, axes = plt.subplots(1, 5, figsize = (20, 15), \n                             gridspec_kw = {\'wspace\': 0.2, \'hspace\': 0.2}, \n                             sharey = True);\n\n    for i, col in enumerate(strt_ftre[-6:-1]):\n        ax = axes[i];\n        _ = train[[col, CFG.target]].groupby(col)[CFG.target].agg([np.sum, np.size]);\n        _ = _.assign(HitRate = _[\'sum\']/ _[\'size\']);\n        \n        ax.pie(x             = _[\'HitRate\'].values,\n               labels        = _.index.values,\n               shadow        = True,\n               colors        = [\'#ccf2ff\', \'#9999ff\'],\n               startangle    = 40, \n               textprops     = {\'fontsize\': 6.5, \'fontweight\': \'bold\', \'color\': \'black\'},\n               pctdistance   = 0.5, \n               labeldistance = 1.20,\n               autopct       = \'%1.2f%%\',\n               explode       = [0.15, 0.15]\n              )\n        ax.set_title(f"{col}", **CFG.title_specs);\n\n    plt.tight_layout();\n    plt.suptitle(f"Binary columns and their relation with target counts", \n                 **CFG.title_specs, \n                 y = 0.65);\n    plt.show();\n    collect();\n    print();\n    del _;   \n    \n')


# <a id="5.9"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > INFERENCES<br> <div>

# <div style= "font-family: Cambria; letter-spacing: 0px; color:#000000; font-size:110%; text-align:left;padding:3.0px; background: #f2f2f2" >
# 1. All columns seem to be significant. Inter-column correlations seem to be important and worthy of further analysis<br>
# 2. We will pay special attention to the binary features. TWF = 1 seems to always coincide with machine failure = 1. Other binary columns are also perhaps consistent with the target = 1<br>
# 3. Original data could be included in the model. We will still test this with 2 versions and assess the CV score impact.<br>
# 4. We have a few outliers in the continuous columns as seen in the box plot. We may have to treat them in a future run and then proceed with the CV score impact<br>
# 5. All the columns are lowly-moderately correlated. Perhaps some dimensionality reduction methods may also work here. Impact of this on the CV score may be assessed<br>
# 6. Patterns in the product ID could be used for secondary feature generation. Unique values in this column appropos to Type could be used as secondary features<br>
# </div>

# <a id="6"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0052cc; border-bottom: 8px solid #cc9966" > DATA TRANSFORMS <br><div> 
#     
# This section aims at creating secondary features, scaling and if necessary, conjoining the competition training and original data tables<br>
# Reference (in part) - https://www.kaggle.com/code/reymaster/97137-eda-feature-engineering-ensemble-baseline

# In[8]:


get_ipython().run_cell_magic('time', '', '\n# Data transforms:-\nclass Xformer(TransformerMixin, BaseEstimator):\n    """\n    This class is used to create secondary features from the existing data\n    """;\n    \n    def __init__(self): pass\n    \n    def fit(self, X, y= None, **params):\n        self.ip_cols = X.columns;\n        return self;\n    \n    def transform(self, X, y= None, **params):\n        """\n        This function does the below-\n        1. Corrects the 0 height values with the nearest non-zero value. Height = 0 is meaningless.\n        2. It creates secondary features if requested. Else it passes through.\n        3. It drops the product ID column\n        """;\n        \n        global strt_ftre;\n        df    = X.copy();      \n      \n        if CFG.sec_ftre_req == "Y":\n            df[\'PID1\'] = df[\'ProductID\'].str[1:3].astype(np.int8);\n            df[\'PID2\'] = \\\n            df[\'ProductID\'].str[3].astype(np.int8) + \\\n            df[\'ProductID\'].str[4].astype(np.int8) + \\\n            df[\'ProductID\'].str[5].astype(np.int8);\n            df[\'PID3\'] = df[\'ProductID\'].str[3:].astype(np.int32);\n            \n            df["Power"]           = np.log1p(df["Torque"] * df["Rotationalspeed"]);\n            df["TempRatio"]       = np.log1p(df["Processtemperature"]) - np.log1p(df["Airtemperature"]);\n            df[\'TotFail\']         = df[\'TWF\'] + df[\'HDF\'] + df[\'PWF\'] + df[\'OSF\'] + df[\'RNF\'];\n            df["ToolWrRotSp"]     = np.log1p(df["Toolwear"] * df["Rotationalspeed"]);\n            df["TorquexWear"]     = np.log1p(df["Torque"] * df["Toolwear"]);\n            df[\'TorqueDivWear\']   = np.log1p(df["Torque"]) - np.log1p(df["Toolwear"]);\n            df[\'Rotationalspeed\'] = np.log1p(df[\'Rotationalspeed\']);\n\n        if CFG.sec_ftre_req != "Y": \n            PrintColor(f"Secondary features are not required", color = Fore.RED);\n        \n        df = df.drop(columns = [\'ProductID\'], errors = \'ignore\');\n            \n        self.op_cols = df.columns;  \n        return df;\n    \n    def get_feature_names_in(self, X, y=None, **params): \n        return self.ip_cols;    \n    \n    def get_feature_names_out(self, X, y=None, **params): \n        return self.op_cols;\n     \n# Scaling:-\nclass Scaler(TransformerMixin, BaseEstimator):\n    """\n    This class aims to create scaling for the provided dataset\n    """;\n    \n    def __init__(self, scl_method: str, scale_req: str):\n        self.scl_method = scl_method;\n        self.scale_req  = CFG.scale_req;\n        \n    def fit(self,X, y=None, **params):\n        "This function calculates the train-set parameters for scaling";\n        \n        self.scl_cols        = X.drop(columns = [\'id\', \'Source\', CFG.target], errors = \'ignore\').columns;\n        self.params          = X[self.scl_cols].describe(percentiles = [0.25, 0.50, 0.75]).drop([\'count\'], axis=0).T;\n        self.params[\'iqr\']   = self.params[\'75%\'] - self.params[\'25%\'];\n        self.params[\'range\'] = self.params[\'max\'] - self.params[\'min\'];\n        \n        return self;\n    \n    def transform(self,X, y=None, **params):  \n        "This function transform the relevant scaling columns";\n        \n        df = X.copy();\n        if self.scale_req == "Y":\n            if CFG.scl_method == "Z":\n                df[self.scl_cols] = (df[self.scl_cols].values - self.params[\'mean\'].values) / self.params[\'std\'].values;\n            elif CFG.scl_method == "Robust":\n                df[self.scl_cols] = (df[self.scl_cols].values - self.params[\'50%\'].values) / self.params[\'iqr\'].values;\n            elif CFG.scl_method == "MinMax":\n                df[self.scl_cols] = (df[self.scl_cols].values - self.params[\'min\'].values) / self.params[\'range\'].values;\n        else:\n            PrintColor(f"Scaling is not needed", color = Fore.RED);\n    \n        return df;\n    \ncollect();\nprint();\n')


# In[13]:


get_ipython().run_cell_magic('time', '', '\nPrintColor(f"\\n{\'-\'* 20} Data transforms and encoding {\'-\'* 20}", \n           color = Fore.MAGENTA);\nPrintColor(f"\\nTrain data shape before transforms = {train.shape}");\n\n# Conjoining the train and original data:-\nif CFG.conjoin_orig_data == "Y":\n    train = pd.concat([train, original], axis=0, ignore_index = True);\n    PrintColor(f"We are using the training and original data", color = Fore.RED);\n    \n# Adjusting the data for duplicates:-\nif CFG.conjoin_orig_data == "N":\n    PrintColor(f"We are using the training data only", color = Fore.RED);\n    \nPrintColor(f"Train data shape after conjoining = {train.shape}\\n");\ndisplay(train.columns);\n\n# Data transforms:--\nXtrain, ytrain = train.drop([CFG.target], axis=1), train[CFG.target];\n\nenc = OrdinalEncoder() if CFG.enc_method == "Label" else OneHotEncoder();\npipe = \\\nColumnTransformer([(\'E\', enc, \'Type\')], verbose_feature_names_out= False,\n                  remainder = Pipeline(steps = [(\'T\', Xformer())]));\npipe.fit(Xtrain, ytrain);\nXtrain, Xtest = pipe.transform(Xtrain), pipe.transform(test);\nPrintColor(f"Xtrain-Xtest data shape after all transforms = {Xtrain.shape} {Xtest.shape}");\n\n# Adjusting the indices after transforms:-\nXtrain.index = range(len(Xtrain));\nytrain.index = Xtrain.index;\n\nprint();\ncollect();\n')


# In[14]:


get_ipython().run_cell_magic('time', '', '\n# Displaying the transformed data descriptions for infinite/ null values:-\nPrintColor(f"\\n---- Transformed data description for distribution analysis ----\\n",\n          color = Fore.MAGENTA);\n\nPrintColor(f"\\nTrain data\\n");\ndisplay(Xtrain.describe(percentiles = [0.05, 0.9, 0.95]).T.\\\n        drop(columns = [\'count\', \'mean\', \'std\']).\\\n        style.format(formatter = \'{:,.2f}\').\\\n        background_gradient(cmap = \'Blues\')\n       );\n\nPrintColor(f"\\nTest data\\n");\ndisplay(Xtest.describe(percentiles = [0.05, 0.9, 0.95]).T.\\\n        drop(columns = [\'count\', \'mean\', \'std\']).\\\n        style.format(formatter = \'{:,.2f}\').\\\n        background_gradient(cmap = \'Blues\')\n       );\n\nprint();\ncollect();\n')


# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > FEATURE ANALYSIS AFTER TRANSFORMS<br> <div>
#     
# We aim to do the below herewith- <br>
# 1. Correlation<br>
# 2. Mutual Information<br>
# 3. Comparison of all features with random noise<br>
#     
# Source idea for point 3 - https://www.kaggle.com/competitions/playground-series-s3e14/discussion/406873

# In[53]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ftre_imp_req == "Y":\n    PrintColor(f"\\n{\'-\'* 30} Feature analysis after data transforms {\'-\'* 30}\\n", \n              color = Fore.MAGENTA);\n\n    # Correlation:-\n    MakeCorrPlot(df = Xtrain, data_label = "Train", figsize = (28, 8));\n    MakeCorrPlot(df = Xtest,  data_label = "Test",  figsize = (28, 8));\n\n    # Making univariate specific interim tables:-\n    X = Xtrain.loc[Xtrain.Source == \'Competition\'].drop(columns = [\'Source\', \'id\'], errors = \'ignore\');\n    X[\'noise\'] = np.random.normal(0.0, 3.0, len(X));\n    y = ytrain.loc[X.index];\n    Unv_Snp = pd.DataFrame(index = X.columns,columns = [\'MutInfo\']);\n\n    # Mutual information:-\n    Unv_Snp[\'MutInfo\'] = mutual_info_classif(X,y,random_state = CFG.state);\n\n    fig, ax = plt.subplots(1,1, figsize = (10, 4.5), \n                             gridspec_kw = {\'wspace\': 0.2, \'hspace\': 0.25});\n    Unv_Snp[\'MutInfo\'].plot.bar(ax = ax, color = \'tab:blue\');\n    ax.set_title(f"Mutual Information", **CFG.title_specs);\n    ax.set_yticks(np.arange(0.0, 0.07, 0.003));\n    \n    plt.tight_layout();\n    plt.show();\n\nelse:\n    PrintColor(f"\\nFeature importance is not needed\\n", color = Fore.RED);\n    \nprint();\ncollect();\n')


# <a id="7"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0052cc; border-bottom: 8px solid #cc9966" > MODEL TRAINING <br><div> 
#     
# We commence our model assignment with a simple ensemble of tree-based and linear models and then shall proceed with the next steps<br>
#     
# **Note**-<br>
# GINI metric is a rank based metric that does not necessitate the usage of classifiers only. Regressors could be used to good effect appropos to their contribution to the CV score. 

# In[54]:


get_ipython().run_cell_magic('time', '', '\n# Initializing model I-O:-\n\nMdl_Master = \\\n{\'CBR\': CatBoostRegressor(**{\'task_type\'           : "GPU" if CFG.gpu_switch == "ON" else "CPU",\n                             \'loss_function\'       : \'RMSE\',\n                             \'eval_metric\'         : \'RMSE\',\n                             \'bagging_temperature\' : 0.45,\n                             \'colsample_bylevel\'   : 0.75,\n                             \'iterations\'          : 4500,\n                             \'learning_rate\'       : 0.085,\n                             \'od_wait\'             : 40,\n                             \'max_depth\'           : 7,\n                             \'l2_leaf_reg\'         : 0.75,\n                             \'min_data_in_leaf\'    : 35,\n                             \'random_strength\'     : 0.2, \n                             \'max_bin\'             : 256,\n                             \'verbose\'             : 0,\n                           }\n                        ),\n \n \'CBC\': CatBoostClassifier(**{\'task_type\'           : "GPU" if CFG.gpu_switch == "ON" else "CPU",\n                              \'objective\'           : \'Logloss\',\n                              \'loss_function\'       : \'Logloss\',\n                              \'eval_metric\'         : \'AUC\',\n                              \'bagging_temperature\' : 0.425,\n                              \'colsample_bylevel\'   : 0.8,\n                              \'iterations\'          : 4_000,\n                              \'learning_rate\'       : 0.045,\n                              \'od_wait\'             : 32,\n                              \'max_depth\'           : 6,\n                              \'l2_leaf_reg\'         : 0.45,\n                              \'min_data_in_leaf\'    : 22,\n                              \'random_strength\'     : 0.15, \n                              \'max_bin\'             : 200,\n                              \'verbose\'             : 0,\n                           }\n                         ), \n\n \'LGBMR\': LGBMRegressor(**{\'device\'            : "gpu" if CFG.gpu_switch == "ON" else "cpu",\n                           \'objective\'         : \'regression\',\n                           \'metric\'            : \'rmse\',\n                           \'boosting_type\'     : \'gbdt\',\n                           \'random_state\'      : CFG.state,\n                           \'colsample_bytree\'  : 0.67,\n                           \'feature_fraction\'  : 0.70,\n                           \'learning_rate\'     : 0.06,\n                           \'max_depth\'         : 8,\n                           \'n_estimators\'      : 5000,\n                           \'num_leaves\'        : 120,                    \n                           \'reg_alpha\'         : 1.25,\n                           \'reg_lambda\'        : 3.5,\n                           \'verbose\'           : -1,\n                         }\n                      ),\n \n  \'LGBMC\': LGBMClassifier(**{\'device\'            : "gpu" if CFG.gpu_switch == "ON" else "cpu",\n                             \'objective\'         : \'binary\',\n                             \'metric\'            : \'auc\',\n                             \'boosting_type\'     : \'gbdt\',\n                             \'random_state\'      : CFG.state,\n                             \'colsample_bytree\'  : 0.62,\n                             \'subsample\'         : 0.925,\n                             \'scale_pos_weight\'  : 0.925,\n                             \'feature_fraction\'  : 0.70,\n                             \'learning_rate\'     : 0.045,\n                             \'max_depth\'         : 9,\n                             \'n_estimators\'      : 4000,\n                             \'num_leaves\'        : 90,                    \n                             \'reg_alpha\'         : 0.0001,\n                             \'reg_lambda\'        : 1.5,\n                             \'verbose\'           : -1,\n                         }\n                      ),\n\n \'XGBR\': XGBRegressor(**{\'objective\'          : \'reg:squarederror\',\n                         \'eval_metric\'        : \'rmse\',\n                         \'random_state\'       : CFG.state,\n                         \'tree_method\'        : "gpu_hist" if CFG.gpu_switch == "ON" else "hist",\n                         \'colsample_bytree\'   : 0.75,\n                         \'learning_rate\'      : 0.0125,\n                         \'max_depth\'          : 8,\n                         \'n_estimators\'       : 5000,                         \n                         \'reg_alpha\'          : 1.25,\n                         \'reg_lambda\'         : 1e-05,\n                         \'min_child_weight\'   : 40,\n                        }\n                     ),\n \n  \'XGBC\': XGBClassifier(**{\'tree_method\'        : "gpu_hist" if CFG.gpu_switch == "ON" else "hist",\n                           \'objective\'          : \'binary:logistic\',\n                           \'eval_metric\'        : \'auc\',\n                           \'random_state\'       : CFG.state,\n                           \'scale_pos_weight\'   : 0.95,\n                           \'colsample_bytree\'   : 0.25,\n                           \'learning_rate\'      : 0.015,\n                           \'max_depth\'          : 9,\n                           \'n_estimators\'       : 4000,                         \n                           \'reg_alpha\'          : 0.0001,\n                           \'reg_lambda\'         : 2.25,\n                           \'min_child_weight\'   : 50,\n                        }\n                       ),\n \n  \'HGBC\': HGBC(learning_rate    = 0.075,\n               max_iter         = 2000,\n               max_depth        = 8,\n               min_samples_leaf = 25,\n               l2_regularization= 1.25,\n               max_bins         = 200,\n               n_iter_no_change = 50,\n               random_state     = CFG.state,\n              ),\n};\n\nprint();\ncollect();\n')


# In[55]:


get_ipython().run_cell_magic('time', '', '\n# Selecting relevant columns for the train and test sets:-\nif CFG.enc_method == "Label":\n    sel_cols = [\'Source\',\n                \'Type\', \'Airtemperature\', \'Rotationalspeed\',\n                \'Torque\', \'Toolwear\', \'TWF\', \'HDF\', \'PWF\', \'OSF\', \n                \'PID1\', \'Power\', \'ToolWrRotSp\',\'TorquexWear\'\n               ];\nelse:\n    sel_cols = [\'Source\',\n                \'Type_M\', \'Type_L\', \n                \'Airtemperature\', \'Rotationalspeed\',\n                \'Torque\', \'Toolwear\', \'TWF\', \'HDF\', \'PWF\', \'OSF\', \n                \'PID1\', \'Power\', \'ToolWrRotSp\',\'TorquexWear\'\n               ];    \ntry: \n    Xtrain, Xtest = Xtrain[sel_cols], Xtest[sel_cols];\nexcept: \n    PrintColor(f"\\n---> Check the columns selected\\n---> Selected columns-", color = Fore.RED);\n    pprint(Xtest.columns, depth = 1, width = 10, indent = 5);\n\n# Initializing output tables for the models:-\nmethods   = list(Mdl_Master.keys());\nOOF_Preds = pd.DataFrame(columns = methods);\nMdl_Preds = pd.DataFrame(index = sub_fl[\'id\'], columns = methods);\nFtreImp   = pd.DataFrame(index = Xtrain.drop(columns = [CFG.target, \'Source\', \'Label\'],\n                                             errors = \'ignore\').columns, \n                         columns = methods\n                        );\nScores    = pd.DataFrame(columns = methods);\n\nPrintColor(f"\\n---> Selected model options- ");\npprint(methods, depth = 1, width = 100, indent = 5);\n\nprint();\ncollect();\n')


# In[56]:


get_ipython().run_cell_magic('time', '', '\ndef TrainMdl(method:str):\n    \n    global Mdl_Master, Mdl_Preds, OOF_Preds, all_cv, FtreImp, Xtrain, ytrain; \n    \n    model     = Mdl_Master.get(method); \n    cols_drop = [\'id\', \'Source\', \'Label\'];\n    scl_cols  = [col for col in Xtrain.columns if col not in cols_drop];\n    cv        = all_cv.get(CFG.mdlcv_mthd);\n    Xt        = Xtest.copy(deep = True);\n    \n    if CFG.scale_req == "N" and method.upper() in [\'RIDGE\', \'LASSO\', \'SVR\']:\n        X, y        = Xtrain, ytrain;\n        scaler      = all_scalers[CFG.scl_method];\n        X[scl_cols] = scaler.fit_transform(X[scl_cols]);\n        Xt[scl_cols]= scaler.transform(Xt[scl_cols]);\n        PrintColor(f"--> Scaling the data for {method} model");\n\n    if CFG.use_orig_allfolds == "Y":\n        X    = Xtrain.query("Source == \'Competition\'");\n        y    = ytrain.loc[ytrain.index.isin(X.index)]; \n        Orig = pd.concat([Xtrain, ytrain], axis=1).query("Source == \'Original\'");\n        \n    elif CFG.use_orig_allfolds != "Y":\n        X,y = Xtrain.copy(deep = True), ytrain.copy(deep = True);\n                \n    # Initializing I-O for the given seed:-        \n    test_preds = 0;\n    oof_preds  = pd.DataFrame(); \n    scores     = [];\n    ftreimp    = 0;\n          \n    for fold_nb, (train_idx, dev_idx) in enumerate(cv.split(X, y)): \n        Xtr  = X.iloc[train_idx].drop(columns = cols_drop, errors = \'ignore\');   \n        Xdev = X.iloc[dev_idx].loc[X.Source == "Competition"].\\\n        drop(columns = cols_drop, errors = \'ignore\'); \n        ytr  = y.loc[y.index.isin(Xtr.index)];\n        ydev = y.loc[y.index.isin(Xdev.index)];\n\n        if CFG.use_orig_allfolds == "Y":\n            Xtr = pd.concat([Xtr, Orig.drop(columns = [CFG.target, \'Source\'], errors = \'ignore\')], \n                            axis = 0, ignore_index = True);\n            ytr = pd.concat([ytr, Orig[CFG.target]], axis = 0, ignore_index = True);\n            \n        # Fitting the model:- \n        if method in [\'LGBMR\', \'LGBMC\', \'CBC\', \'CBR\', \'XGBR\', \'XGBC\']:    \n            model.fit(Xtr, ytr, \n                      eval_set = [(Xdev, ydev)], \n                      verbose = 0,\n                      early_stopping_rounds = CFG.nbrnd_erly_stp\n                     ); \n        else: \n            model.fit(Xtr, ytr); \n            \n        # Collecting predictions and scores and post-processing OOF based on model method:-\n        if method.upper().endswith(\'R\'):\n            dev_preds = PostProcessPred(model.predict(Xdev), \n                                        post_process= CFG.pstprcs_train);\n            test_preds = test_preds + \\\n            PostProcessPred(model.predict(Xt.drop(columns = cols_drop, errors = \'ignore\')),\n                            post_process= CFG.pstprcs_train); \n        else:\n            dev_preds = model.predict_proba(Xdev)[:,1];\n            test_preds = test_preds + \\\n            model.predict_proba(Xt.drop(columns = cols_drop, errors = \'ignore\'))[:,1];            \n        \n        score = ScoreMetric(ydev.values.flatten(), dev_preds);\n        scores.append(score); \n \n        Scores.loc[fold_nb, method] = np.round(score, decimals= 6);\n        oof_preds = pd.concat([oof_preds,\n                               pd.DataFrame(index   = Xdev.index, \n                                            data    = dev_preds,\n                                            columns = [method])\n                              ],axis=0, ignore_index= False\n                             );  \n    \n        oof_preds = pd.DataFrame(oof_preds.groupby(level = 0)[method].mean());\n        oof_preds.columns = [method];\n        \n        try: ftreimp += model.feature_importances_;\n        except: ftreimp = 0;\n            \n    num_space = 10 - len(method);\n    PrintColor(f"--> {method}{\'-\' * num_space} CV = {np.mean(scores):.6f}");\n    del num_space;\n    \n    OOF_Preds[f\'{method}\'] = PostProcessPred(oof_preds.values.flatten(), CFG.pstprcs_train);\n    \n    if CFG.mdlcv_mthd in [\'KF\', \'SKF\']:\n        Mdl_Preds[f\'{method}\'] = test_preds.flatten()/ CFG.n_splits; \n        FtreImp[method]        = ftreimp / CFG.n_splits;\n    else:\n        Mdl_Preds[f\'{method}\'] = test_preds.flatten()/ (CFG.n_splits * CFG.n_repeats); \n        FtreImp[method]        = ftreimp / (CFG.n_splits * CFG.n_repeats);\n    \n    collect(); \n    \n# Implementing the ML models:-\nif CFG.ML == "Y": \n    PrintColor(f"\\n{\'-\' * 25} ML model training {\'-\' * 25}\\n", color = Fore.MAGENTA);\n    for method in tqdm(methods, "ML models----"): \n        TrainMdl(method);\n    clear_output();\n    \n    PrintColor(f"\\n{\'-\' * 20} OOF CV scores across methods {\'-\' * 20}\\n", \n               color = Fore.MAGENTA);\n    display(pd.concat([Scores.mean(axis = 0), Scores.std(axis = 0)], axis=1).\\\n            rename(columns = {0: \'Mean\', 1: \'Std\'}).T.\\\n            style.format(precision = 6).\\\n            background_gradient(cmap = \'cubehelix\', axis=1)\n           );    \nelse:\n    PrintColor(f"\\nML models are not needed\\n", color = Fore.RED);\n    \ncollect();\nprint();\n')


# In[57]:


get_ipython().run_cell_magic('time', '', '\n# Analysing the model results and feature importances and calibration curves:-\nif CFG.ML == "Y":\n    fig, axes = plt.subplots(len(methods), 2, figsize = (25, len(methods) * 6),\n                             gridspec_kw = {\'hspace\': 0.2, \'wspace\': 0.2}, \n                             width_ratios= [0.7, 0.3],\n                            );\n\n    for i, col in enumerate(methods):\n        ax = axes[i,0];\n        FtreImp[col].plot.barh(ax = ax, color = \'#0073e6\');\n        ax.set_title(f"{col} Importances", **CFG.title_specs);\n        ax.set(xlabel = \'\', ylabel = \'\');\n\n        ax = axes[i,1];\n        Clb.from_predictions(ytrain[0:len(OOF_Preds)], OOF_Preds[col], n_bins= 20, ref_line = True,\n                             **{\'color\': \'#0073e6\', \'linewidth\': 1.2, \n                                \'markersize\': 3.75, \'marker\': \'o\', \'markerfacecolor\': \'#cc7a00\'},\n                             ax = ax);\n        ax.set_title(f"{col} Calibration", **CFG.title_specs);\n        ax.set(xlabel = \'\', ylabel = \'\',);\n        ax.set_yticks(np.arange(0,1.01, 0.05), labels = np.round(np.arange(0,1.01, 0.05), 2), fontsize = 7.0);\n        ax.set_xticks(np.arange(0,1.01, 0.05), \n                      labels = np.round(np.arange(0,1.01, 0.05), 2), \n                      fontsize = 7.0, \n                      rotation = 90\n                     );\n        ax.legend(\'\');\n\n    plt.tight_layout();\n    plt.show();\n    \ncollect();\nprint();\n')


# <a id="8"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0052cc; border-bottom: 8px solid #cc9966" > ENSEMBLE AND SUBMISSION<br> <div> 
#     
# We will conclude our baseline model with a simple optuna based ensemble<br>
# We also engender rule based post-processing involving-<br>
# 1. Replacing duplicate train-test observations with the values from the train data<br>
# 2. Replacing the target with 1 where TWF = 1<br>

# In[58]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ensemble_req == "Y":\n    def Objective(trial):\n        "This function defines the objective for the optuna ensemble using variable models";\n\n        global OOF_Preds, all_cv, ytrain, methods;\n\n        # Define the weights for the predictions from each model:-\n        weights  = [trial.suggest_float(f"M{n}", 0.0001, 0.9999, step = 0.001) \\\n                    for n in range(len(OOF_Preds[methods].columns))\n                   ];\n\n        # Calculating the CV-score for the weighted predictions on the competition data only:-\n        scores = [];  \n        cv     = all_cv[CFG.enscv_mthd];\n        X,y    = OOF_Preds[methods], ytrain[0: len(OOF_Preds)];\n\n        for fold_nb, (train_idx, dev_idx) in enumerate(cv.split(X,y)):\n            Xtr, Xdev = X.iloc[train_idx], X.iloc[dev_idx];\n            ytr, ydev = y.loc[Xtr.index],  y.loc[Xdev.index];\n            scores.append(ScoreMetric(ydev, np.average(Xdev, axis=1, weights = weights)));\n\n        collect();\n        clear_output();\n        return np.mean(scores);\n    \nclear_output();\nprint();\ncollect();\n')


# In[59]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ensemble_req == "Y":   \n    PrintColor(f"\\n{\'-\' * 20} Creating an Optuna Ensemble {\'-\' * 20}\\n", \n               color = Fore.MAGENTA);   \n    \n    study = optuna.create_study(direction  = CFG.metric_obj, \n                                study_name = "OptunaEnsemble", \n                                sampler    = TPESampler(seed = CFG.state)\n                               );\n    study.optimize(Objective, \n                   n_trials          = CFG.ntrials, \n                   gc_after_trial    = True,\n                   show_progress_bar = True\n                  );\n    weights       = study.best_params;\n    clear_output();\n    \n    PrintColor(f"\\n--> Post ensemble weights\\n");\n    pprint(weights, indent = 5, width = 10, depth = 1);\n    PrintColor(f"\\n--> Best ensemble CV score = {study.best_value :.6f}\\n");\n    \n    # Making weighted predictions on the test set:-\n    sub_fl[\'Optuna\'] = np.average(Mdl_Preds[methods], \n                                  weights = list(weights.values()),\n                                  axis=1);\n    \n    PrintColor(f"\\n--> Post ensemble test-set predictions\\n");\n    display(sub_fl.head(5).style.format(precision = 5));   \n    \n    # Implementing rule-based processing after predicting with the model:-\n    _ = \\\n    pd.concat([sub_fl.set_index(\'id\')[[\'Optuna\']], \n               test[[\'TWF\']],\n               dup_df.drop_duplicates(subset = [\'id\']).set_index(\'id\')[[CFG.target]]\n              ], axis=1).fillna(-1).\\\n    rename(columns = {CFG.target : "Dup_Train"});\n\n    # Creating the final submission file:-\n    sub_fl[sub_fl.columns[1]] = np.select([_.TWF == 1, _.Dup_Train > -1.0], [1, _.Dup_Train], _.Optuna).flatten();\n    sub_fl.iloc[:, 0:2].to_csv(f"Submission_V{CFG.version_nb}.csv", index = None);\n    del _;\n    \n    # Creating the histogram of predictions:-\n    fig, axes = plt.subplots(1,2, figsize = (18, 4), gridspec_kw = {\'hspace\': 0.2, \'wspace\': 0.2});\n    _ = pd.DataFrame(np.arange(0, 1.01, 0.03), dtype = np.float16);\n\n    ax = axes[0];\n    sns.histplot(sub_fl[\'Optuna\'], color = \'#0039e6\', bins = \'auto\', ax = ax);\n    ax.set_title(f"\\nEnsemble predictions before post-processing\\n", **CFG.title_specs);\n    ax.set(xlabel = \'\', ylabel = \'\');\n    ax.set_xticks(_.values.flatten(), labels = np.round(_.values.flatten(),2), rotation = 90);\n\n    ax = axes[1];\n    sns.histplot(sub_fl[sub_fl.columns[1]], color = \'#00ace6\', bins = \'auto\', ax = ax);\n    ax.set_title(f"\\nEnsemble predictions after post-processing\\n", **CFG.title_specs);\n    ax.set(xlabel = \'\', ylabel = \'\');\n    ax.set_xticks(_.values.flatten(), labels = np.round(_.values.flatten(),2), rotation = 90);\n    \n    plt.tight_layout();\n    plt.show();\n\n    del _;\n      \ncollect();\nprint();\n')


# In[ ]:


get_ipython().run_line_magic('time', '')

if CFG.ensemble_req == "Y": 
    # Blending my submissions with public notebooks:-
    


# In[60]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ML == "Y":  \n    OOF_Preds.add_prefix(f"V{CFG.version_nb}_").to_csv(f"OOF_Preds_V{CFG.version_nb}.csv");\n    Mdl_Preds.add_prefix(f"V{CFG.version_nb}_").to_csv(f"Mdl_Preds_V{CFG.version_nb}.csv"); \n    if isinstance(Scores, pd.DataFrame) == True:\n        Scores.to_csv(f"Scores_V{CFG.version_nb}.csv");\n        \ndup_df.to_csv(f"Duplicates_V{CFG.version_nb}.csv");\n\n    \ncollect();\nprint();\n')


# <a id="9"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0052cc; border-bottom: 8px solid #cc9966" > NEXT STEPS<br> <div> 

# <div style= "font-family: Cambria; letter-spacing: 0px; color:#000000; font-size:110%; text-align:left;padding:3.0px; background: #f2f2f2" >
# 1. Better feature engineering- this includes feature importance assessments, decision to include/ exclude features and new feature creation<br>
# 2. Better experiments with scaling, encoding with categorical columns. This seems to have some promise<br>
# 3. Better model tuning<br>
# 4. Model calibration- this may not be absolutely necessary with a rank-metric like GINI<br>
# 5. Adding more algorithms and new methods to the model suite<br>
# 6. Better ensemble strategy<br>
# </div>
