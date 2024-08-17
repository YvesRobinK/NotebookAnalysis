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
#     * [TARGET PLOTS](#5.1) 
#     * [PAIR-PLOTS](#5.2)
#     * [CATEGORY COLUMN PLOTS](#5.3)
#     * [CONTINUOUS COLUMN ANALYSIS](#5.4)
#     * [DUPLICATES ANALYSIS](#5.5)
#     * [UNIVARIATE ANALYSIS AND FEATURE RELATIONS](#5.6)
# * [DATA TRANSFORMS](#6)
# * [MODEL TRAINING](#7) 
#     * [BASELINE MODEL](#7.1)    
#     * [ML MODELS](#7.2)    
# * [ENSEMBLE AND SUBMISSION](#8)  
# * [PLANNED WAY FORWARD](#9)     

# <a id="1"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0052cc; border-bottom: 8px solid #cc9966" > IMPORTS<br> <div> 

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# General library imports:-\n!pip install --upgrade scipy;\n\nimport pandas as pd;\nimport numpy as np;\nfrom scipy.stats import mode, kstest, normaltest, shapiro, anderson, jarque_bera;\nfrom collections import Counter;\nfrom itertools import product;\nfrom colorama import Fore, Style, init;\nfrom warnings import filterwarnings;\nfilterwarnings(\'ignore\');\n\nfrom tqdm.notebook import tqdm;\nfrom IPython.display import clear_output;\n\nimport seaborn as sns;\nimport matplotlib.pyplot as plt;\nfrom gc import collect;\nfrom pprint import pprint;\n\npd.set_option(\'display.max_columns\', 50);\npd.set_option(\'display.max_rows\', 50);\n\n# Setting rc parameters in seaborn for plots and graphs- \n# Reference - https://matplotlib.org/stable/tutorials/introductory/customizing.html:-\n# To alter this, refer to matplotlib.rcParams.keys()\n\nsns.set({"axes.facecolor"       : "#ffffff",\n         "figure.facecolor"     : "#ffffff",\n         "axes.edgecolor"       : "#000000",\n         "grid.color"           : "#ffffff",\n         "font.family"          : [\'Cambria\'],\n         "axes.labelcolor"      : "#000000",\n         "xtick.color"          : "#000000",\n         "ytick.color"          : "#000000",\n         "grid.linewidth"       : 0.85,  \n         "grid.linestyle"       : "--",\n         "axes.titlecolor"      : \'tab:blue\',\n         \'axes.titlesize\'       : 9.5,\n         \'axes.labelweight\'     : "bold",\n         \'legend.fontsize\'      : 7.0,\n         \'legend.title_fontsize\': 7.0,\n         \'font.size\'            : 8.0,\n         \'xtick.labelsize\'      : 7.5,\n         \'ytick.labelsize\'      : 7.5,        \n        });\n\nprint();\ncollect();\nclear_output();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Importing model and pipeline specifics:-\n\n# Pipeline specifics:-\nfrom sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler;\nfrom sklearn.model_selection import (RepeatedStratifiedKFold as RSKF, \n                                     StratifiedKFold as SKF,\n                                     KFold, \n                                     RepeatedKFold as RKF, \n                                     cross_val_score);\nfrom sklearn.inspection import permutation_importance, PartialDependenceDisplay as PDD;\nfrom sklearn.feature_selection import mutual_info_regression, RFE;\nfrom sklearn.pipeline import Pipeline;\nfrom sklearn.base import BaseEstimator, TransformerMixin;\n\n# ML Model training:-\nfrom sklearn.metrics import mean_absolute_error as mae, r2_score;\nfrom sklearn.svm import SVR;\nfrom xgboost import XGBRegressor;\nfrom lightgbm import LGBMClassifier, LGBMRegressor, log_evaluation;\nfrom catboost import CatBoostRegressor;\nfrom sklearn.ensemble import RandomForestRegressor as RFR,ExtraTreesRegressor as ETR;\nfrom sklearn.linear_model import Ridge, Lasso;\n\nprint();\ncollect();\n')


# <a id="2"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0052cc; border-bottom: 8px solid #cc9966" > INTRODUCTION<br><div> 

# | Version<br>Number | Version Details | Best CV score| Single/ Ensemble|
# | :-: | --- | :-: | :-: |
# | **V1** |* EDA, plots and secondary features<br>* Standard scaler<br> * Used original data<br>* Baseline model for feature selection<br>* ML models and simple ensemble<br> |343.9214 | Single<br>CatBoost|
# | **V2** |* EDA, plots and secondary features<br>* Standard scaler<br> * Used only training data<br>* Baseline model for feature selection<br>* ML models and simple ensemble<br> | 345.2313| Single<br>CatBoost|
# | **V3** |* EDA, plots and secondary features<br>* Standard scaler<br> * Used original data in all folds<br>* Baseline model for feature selection<br>* ML models and simple ensemble<br> |344.6269 |Single<br>CatBoost |
# | **V4** |* EDA, plots and secondary features<br>* Standard scaler<br> * Used original data<br>* Used permutation importance with noise feature as a benchmark<br>* Used regression_l1 objective function for LGBM<br>* Tree based ML models and simple weighted ensemble<br>* Introduced post-processing by rounding to nearest target value |340.8826 |Single<br>LightGBM |
# | **V5** |* Same as V4 without the original data | 342.0471|Single<br>LightGBM |
# | **V6** |* Same as V4 with original data in all folds |341.0515 |Single<br>LightGBM |
# | **V7** |* Same as V4 without post-processing |340.9798 |Single<br>LightGBM |
# | **V8** |* Same as V4 with small parameter adjustments<br>* Blended results with top public submissions |**340.7079** |Single<br>LightGBM  |

# <a id="2.1"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > CONFIGURATION<br><div> 

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Configuration class:-\nclass CFG:\n    "Configuration class for parameters and CV strategy for tuning and training";\n    \n    # Data preparation:-   \n    version_nb         = 8;\n    gpu_switch         = "OFF"; \n    state              = 42;\n    target             = "yield";\n    episode            = 14;\n    path               = f"/kaggle/input/playground-series-s3e{episode}/";\n    orig_path          = f"/kaggle/input/wild-blueberry-yield-prediction-dataset/WildBlueberryPollinationSimulationData.csv";\n    adv_cv_req         = "N";\n    ftre_plots_req     = "Y";\n    ftre_imp_req       = "Y";\n    \n    # Data transforms and scaling:-    \n    conjoin_orig_data  = "Y";\n    sec_ftre_req       = "Y";\n    scale_req          = "Y";\n    # NOTE---Keep a value here even if scale_req = N, this is used for linear models:-\n    scl_method         = "Z"; \n    \n    # Model Training:- \n    baseline_req       = "N";\n    predpstprcs_req    = "Y";\n    ML                 = "Y";\n    GAM                = "N";\n    use_orig_allfolds  = "N";\n    n_splits           = 10 ;\n    n_repeats          = 10 ;\n    nbrnd_erly_stp     = 300 ;\n    mdlcv_mthd         = \'RKF\';\n          \n    # Global variables for plotting:-\n    grid_specs = {\'visible\': True, \'which\': \'both\', \'linestyle\': \'--\', \n                           \'color\': \'lightgrey\', \'linewidth\': 0.75};\n    title_specs = {\'fontsize\': 9, \'fontweight\': \'bold\', \'color\': \'tab:blue\'};\n\n# Color printing    \ndef PrintColor(text:str, color = Fore.BLUE, style = Style.BRIGHT):\n    "Prints color outputs using colorama using a text F-string";\n    print(style + color + text + Style.RESET_ALL); \n\n# Scaler to be used for continuous columns:- \nall_scalers = {\'Robust\': RobustScaler(), \n               \'Z\': StandardScaler(), \n               \'MinMax\': MinMaxScaler()\n              };\nscaler      = all_scalers.get(CFG.scl_method);\n\n# Commonly used CV strategies for later usage:-\nall_cv= {\'KF\'  : KFold(n_splits= CFG.n_splits, shuffle = True, random_state= CFG.state),\n         \'RKF\' : RKF(n_splits= CFG.n_splits, n_repeats = CFG.n_repeats, random_state= CFG.state),\n         \'RSKF\': RSKF(n_splits= CFG.n_splits, n_repeats = CFG.n_repeats, random_state= CFG.state),\n         \'SKF\' : SKF(n_splits= CFG.n_splits, shuffle = True, random_state= CFG.state)\n        };\nprint();\n\nPrintColor(f"--> Configuration done!");\ncollect();\n')


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
# |  ML               | Machine Learning Models                                 | Y/N                   |
# |  GAM              | GAM required                                            | Y/N                   |
# |  use_orig_allfolds| Use original data across all folds                      | Y/N                   |
# |  n_splits         | Number of CV splits                                     | integer               |
# |  n_repeats        | Number of CV repeats                                    | integer               |
# |  nbrnd_erly_stp   | Number of early stopping rounds                         | integer               |
# |  mdl_cv_mthd      | Model CV method name                                    | RKF/ RSKF/ SKF/ KFold |

# <a id="2.3"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > DATASET AND COMPETITION DETAILS<br><div>
#     
# **Data columns**
#     
# * Clonesize m2 -The average blueberry clone size in the field
# * Honeybee bees/m2/min -Honeybee density in the field
# * Bumbles bees/m2/min -Bumblebee density in the field
# * Andrena bees/m2/min -Andrena bee density in the field
# * Osmia bees/m2/min -Osmia bee density in the field
# * MaxOfUpperTRange ℃ -The highest record of the upper band daily air temperature during the bloom season
# * MinOfUpperTRange ℃ -The lowest record of the upper band daily air temperature
# * AverageOfUpperTRange ℃ -The average of the upper band daily air temperature
# * MaxOfLowerTRange ℃ -The highest record of the lower band daily air temperature
# * MinOfLowerTRange ℃ -The lowest record of the lower band daily air temperature
# * AverageOfLowerTRange ℃ -The average of the lower band daily air temperature
# * RainingDays Day -The total number of days during the bloom season, each of which has precipitation larger than zero
# * AverageRainingDays Day- The average of raining days of the entire bloom season
#     
# **Competition details and notebook objectives**<br>
# 1. This is a regression challenge to predict the yield for blueberries using the above features. **Mean absolute error** is the metric for the challenge<br>
# 2. In this starter notebook, we start the assignment with a detailed EDA, feature plots, interaction effects, adversarial CV analysis and develop starter models to initiate the challenge. We will also incorporate other opinions and approaches as we move along the challenge. 

# <a id="3"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0052cc; border-bottom: 8px solid #cc9966" > PREPROCESSING<br><div> 

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nPrintColor(f"\\n---------- Data Preprocessing ---------- \\n", color = Fore.MAGENTA);\n\n# Reading the train-test datasets:-\ntrain    = pd.read_csv(CFG.path + f"train.csv");\ntest     = pd.read_csv(CFG.path + f"test.csv");\noriginal = pd.read_csv(CFG.orig_path).drop(\'Row#\', axis=1);\noriginal.insert(0, \'id\', range(len(original)));\noriginal[\'id\'] = original[\'id\'] + test[\'id\'].max() + 1;\n\ntrain[\'Source\'], test[\'Source\'], original[\'Source\'] = "Competition", "Competition", "Original";\nPrintColor(f"\\nData shapes- [train, test, original]-- {train.shape} {test.shape} {original.shape}\\n");\n\n# Creating dataset information:\nPrintColor(f"\\nTrain information\\n");\ndisplay(train.info());\nPrintColor(f"\\nTest information\\n")\ndisplay(test.info());\nPrintColor(f"\\nOriginal data information\\n")\ndisplay(original.info());\nprint();\n\n# Displaying column description:-\nPrintColor(f"\\nTrain description\\n");\ndisplay(train.describe(percentiles= [0.05, 0.25, 0.50, 0.75, 0.9, 0.95, 0.99]).\\\n        transpose().\\\n        drop(columns = [\'count\'], errors = \'ignore\').\\\n        drop([CFG.target], axis=0, errors = \'ignore\').\\\n        style.format(precision = 2));\n\nPrintColor(f"\\nTest description\\n");\ndisplay(test.describe(percentiles= [0.05, 0.25, 0.50, 0.75, 0.9, 0.95, 0.99]).\\\n        transpose().\\\n        drop(columns = [\'count\'], errors = \'ignore\').\\\n        drop([CFG.target], axis=0, errors = \'ignore\').\\\n        style.format(precision = 2));\n\nPrintColor(f"\\nOriginal description\\n");\ndisplay(original.describe(percentiles= [0.05, 0.25, 0.50, 0.75, 0.9, 0.95, 0.99]).\\\n        transpose().\\\n        drop(columns = [\'count\'], errors = \'ignore\').\\\n        drop([CFG.target], axis=0, errors = \'ignore\').\\\n        style.format(precision = 2));\n\n# Collating the column information:-\nstrt_ftre = test.columns;\nPrintColor(f"\\nStarting columns\\n");\ndisplay(strt_ftre);\n\n# Creating a copy of the datasets for further use:-\ntrain_copy, test_copy, orig_copy = \\\ntrain.copy(deep= True), test.copy(deep = True), original.copy(deep = True);\n\n# Dislaying the unique values across train-test-original:-\nPrintColor(f"\\nUnique values\\n");\n_ = pd.concat([train.nunique(), test.nunique(), original.nunique()], axis=1);\n_.columns = [\'Train\', \'Test\', \'Original\'];\ndisplay(_.style.background_gradient(cmap = \'Blues\').format(formatter = \'{:,.0f}\'));\n\n# Normality check:-\ncols = list(strt_ftre[1:-1]);\nPrintColor(f"\\nShapiro Wilk normality test analysis\\n");\npprint({col: [np.round(shapiro(train[col]).pvalue,decimals = 4), \n              np.round(shapiro(test[col]).pvalue,4) if col != CFG.target else np.NaN,\n              np.round(shapiro(original[col]).pvalue,4)] for col in cols\n       }, indent = 5, width = 100, depth = 2, compact= True);\n\nPrintColor(f"\\nNormal-test normality test analysis\\n");\npprint({col: [np.round(normaltest(train[col]).pvalue,decimals = 4), \n              np.round(normaltest(test[col]).pvalue,4) if col != CFG.target else np.NaN,\n              np.round(normaltest(original[col]).pvalue,4)] for col in cols\n       }, indent = 5, width = 100, depth = 2, compact= True);\n\nPrintColor(f"\\nK-S normality test analysis\\n");\npprint({col: [np.round(kstest(train[col], cdf = \'norm\').pvalue,decimals = 4), \n              np.round(kstest(test[col], cdf = \'norm\').pvalue,4) if col != CFG.target else np.NaN,\n              np.round(kstest(original[col], cdf = \'norm\').pvalue,4)] for col in cols\n       }, indent = 5, width = 100, depth = 2, compact= True);\nprint();\n')


# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > INFERENCES<br> <div>

# <div style= "font-family: Cambria; letter-spacing: 0px; color:#000000; font-size:110%; text-align:left;padding:3.0px; background: #f2f2f2" >
# 1. All the columns are numerical<br>
# 2. We do not have any nulls in the data<br>
# 3. All columns are non-normal<br>
# 4. The synthetic data is nearly 32.8 times the original data, creating a potential duplicate row issue. Duplicate handling is a key challenge in this case<br>
# 5. We have few unique values (discrete values) in quite a few columns. These could be considered as categorical encoded columns and then suitable encoding options could also be ensued on these attributes as well.<br>
# </div>

# <a id="4"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0052cc; border-bottom: 8px solid #cc9966" > ADVERSARIAL CV<br><div>

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Performing adversarial CV between the 2 specified datasets:-\ndef Do_AdvCV(df1:pd.DataFrame, df2:pd.DataFrame, source1:str, source2:str):\n    "This function performs an adversarial CV between the 2 provided datasets if needed by the user";\n    \n    # Adversarial CV per column:-\n    ftre = train.drop(columns = [\'id\', CFG.target, "Source"], errors = \'ignore\').columns;\n    adv_cv = {};\n\n    for col in ftre:\n        PrintColor(f"---> Current feature = {col}", style = Style.NORMAL);\n        shuffle_state = np.random.randint(low = 10, high = 100, size= 1);\n\n        full_df = \\\n        pd.concat([df1[[col]].assign(Source = source1), df2[[col]].assign(Source = source2)], \n                  axis=0, ignore_index = True).\\\n        sample(frac = 1.00, random_state = shuffle_state);\n\n        full_df = full_df.assign(Source_Nb = full_df[\'Source\'].eq(source2).astype(np.int8));\n\n        # Checking for adversarial CV:-\n        model = LGBMClassifier(random_state = CFG.state, max_depth = 3, learning_rate = 0.05);\n        cv    = all_cv[\'RSKF\'];\n        score = np.mean(cross_val_score(model, \n                                        full_df[[col]], \n                                        full_df.Source_Nb, \n                                        scoring= \'roc_auc\', \n                                        cv= cv)\n                       );\n        adv_cv.update({col: round(score, 4)});\n        collect();\n    \n    del ftre;\n    \n    PrintColor(f"\\nResults\\n");\n    pprint(adv_cv, indent = 5, width = 20, depth = 1);\n    collect();\n    \n    fig, ax = plt.subplots(1,1,figsize = (12, 5));\n    pd.Series(adv_cv).plot.bar(color = \'tab:blue\', ax = ax);\n    ax.axhline(y = 0.60, color = \'red\', linewidth = 2.75);\n    ax.grid(**CFG.grid_specs); \n    plt.yticks(np.arange(0.0, 0.81, 0.05));\n    plt.show();\n    \n# Implementing the adversarial CV:-\nif CFG.adv_cv_req == "Y":\n    PrintColor(f"\\n---------- Adversarial CV - Train vs Original ----------\\n", \n               color = Fore.MAGENTA);\n    Do_AdvCV(df1 = train, df2 = original, source1 = \'Train\', source2 = \'Original\');\n    \n    PrintColor(f"\\n---------- Adversarial CV - Train vs Test ----------\\n", \n               color = Fore.MAGENTA);\n    Do_AdvCV(df1 = train, df2 = test, source1 = \'Train\', source2 = \'Test\');\n    \n    PrintColor(f"\\n---------- Adversarial CV - Original vs Test ----------\\n", \n               color = Fore.MAGENTA);\n    Do_AdvCV(df1 = train, df2 = test, source1 = \'Original\', source2 = \'Test\');   \n    \nif CFG.adv_cv_req == "N":\n    PrintColor(f"\\nAdversarial CV is not needed\\n", color = Fore.RED);\n    \ncollect();\nprint();\n')


# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > INFERENCES<br><div>

# <div style= "font-family: Cambria; letter-spacing: 0px; color:#000000; font-size:110%; text-align:left;padding:3.0px; background: #f2f2f2" >
# 1. Train-test belong to the same distribution, we can perhaps rely on the CV score<br>
# 2. We need to further check the train-original distribution further, adversarial validation results indicate that we can use the original dataset<br>
# </div>

# <a id="5"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0052cc; border-bottom: 8px solid #cc9966" > VISUALS AND EDA <br><div> 
#  

# <a id="5.1"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > TARGET PLOT<br><div>

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ftre_plots_req == "Y":\n    fig, axes = plt.subplots(1,2, figsize = (12, 4),gridspec_kw = {\'wspace\': 0.25});\n    \n    for i, df in tqdm(enumerate([train, original]), "Target plot ---"):\n        ax = axes[i];\n        sns.histplot(data = df[[CFG.target]], x = CFG.target, color = \'#008ae6\', bins = 50, ax = ax);\n        df_name = \'Train\' if i == 0 else "Original";\n        _ = ax.set_title(f"\\n{df_name} data- target\\n", **CFG.title_specs);\n\n    plt.tight_layout();\n    plt.show();\n    \ncollect();\nprint();\n')


# <a id="5.2"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > PAIRPLOTS<br><div>

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ftre_plots_req == "Y":\n    PrintColor(f"\\nTrain data- pair plots\\n");\n    _ = sns.pairplot(data = train.drop(columns = [\'id\',\'Source\', CFG.target], errors = \'ignore\'), \n                     diag_kind = \'kde\', markers= \'o\', plot_kws= {\'color\': \'tab:blue\'}               \n                    );\n\nprint();\ncollect();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ftre_plots_req == "Y":\n    PrintColor(f"\\nOriginal data- pair plots\\n");\n    _ = sns.pairplot(data = original.drop(columns = [\'id\',\'Source\', CFG.target], errors = \'ignore\'), \n                     diag_kind = \'kde\', markers= \'o\', plot_kws= {\'color\': \'tab:blue\'}               \n                    );\nprint();\ncollect();\n')


# <a id="5.3"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > CATEGORY COLUMN PLOTS<br><div>
#     
# Quite a few columns appear to be categorical with very few unique values. We could choose to analyse them as categorical columns and possibly consider measures like one-hot encoding to improve the overall efficacy.<br>

# In[ ]:


get_ipython().run_cell_magic('time', '', '\ncat_cols = train[strt_ftre[1:-1]].nunique()[0:-3].index;\n\nif CFG.ftre_plots_req == "Y":\n    fig, axes = plt.subplots(len(cat_cols), 2, figsize = (18, len(cat_cols)* 5.5), \n                             gridspec_kw = {\'wspace\': 0.2, \'hspace\': 0.5});\n\n    for i, col in enumerate(cat_cols):\n        ax = axes[i,0];\n        train[col].value_counts(normalize = True).sort_index().plot.bar(color = \'tab:blue\', ax = ax);\n        ax.set_title(f"{col}_Train", **CFG.title_specs);\n        ax.grid(**CFG.grid_specs);\n        ax.set(xlabel = \'\', ylabel = \'\');\n\n        ax = axes[i,1];\n        test[col].value_counts(normalize = True).sort_index().plot.bar(color = \'#1aa3ff\', ax = ax);\n        ax.set_title(f"{col}_Test", **CFG.title_specs);\n        ax.grid(**CFG.grid_specs);  \n        ax.set(xlabel = \'\', ylabel = \'\'); \n\n    plt.tight_layout();\n    plt.show();\n    \nprint();\ncollect();\n')


# <a id="5.4"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > CONTINUOUS COLUMN PLOTS<br><div>

# In[ ]:


get_ipython().run_cell_magic('time', '', '\ncont_cols = [col for col in strt_ftre[1:-1] if col not in cat_cols];\n\nif CFG.ftre_plots_req == "Y":\n    df = pd.concat([train[cont_cols].assign(Source = \'Train\'), \n                    test[cont_cols].assign(Source = \'Test\')], \n                   axis=0, ignore_index = True\n                  );\n    \n    fig, axes = plt.subplots(3,3, figsize = (18, 8.0), \n                             gridspec_kw = {\'hspace\': 0.35, \'wspace\': 0.2, \'height_ratios\': [0.8, 0.2, 0.2]});\n    \n    for i,col in enumerate(cont_cols):\n        ax = axes[0,i];\n        sns.kdeplot(data = df[[col, \'Source\']], x = col, hue = \'Source\', palette = [\'blue\', \'red\'], \n                    ax = ax, linewidth = 2.25\n                   );\n        ax.set_title(f"\\n{col}", **CFG.title_specs);\n        ax.grid(**CFG.grid_specs);\n        ax.set(xlabel = \'\', ylabel = \'\');\n        \n        ax = axes[1,i];\n        sns.boxplot(data = df.loc[df.Source == \'Train\', [col]], x = col, width = 0.25,\n                    color = \'#008ae6\', saturation = 0.90, linewidth = 1.25, fliersize= 2.25,\n                    ax = ax);\n        ax.set(xlabel = \'\', ylabel = \'\');\n        ax.set_title(f"{col}- Train", **CFG.title_specs);\n        \n        ax = axes[2,i];\n        sns.boxplot(data = df.loc[df.Source == \'Test\', [col]], x = col, width = 0.25, fliersize= 2.25,\n                    color = \'#00aaff\', saturation = 0.6, linewidth = 1.25, \n                    ax = ax); \n        ax.set(xlabel = \'\', ylabel = \'\');\n        ax.set_title(f"{col}- Test", **CFG.title_specs);\n              \n    plt.suptitle(f"\\nDistribution analysis- continuous columns\\n", **CFG.title_specs);\n    plt.tight_layout();\n    plt.show();\n    \nprint();\ncollect();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndef MakeGrpPlot(df: pd.DataFrame, contcol: str):\n    """\n    This function makes kde plots per continuous column per category column\n    """;\n    \n    global cont_cols, cat_cols;\n    fig, axes = plt.subplots(7,2, figsize = (18, 30), \n                             gridspec_kw = {\'hspace\': 0.2, \'wspace\': 0.25});\n    \n    for i, catcol in enumerate(cat_cols):\n        ax = axes[i // 2, i % 2];\n        sns.kdeplot(data      = df[[contcol, catcol]], \n                    x         = contcol, \n                    hue       = catcol, \n                    palette   = \'viridis\', \n                    ax        = ax, \n                    linewidth = 2.5,\n                   );\n        ax.set_title(f"Hue = {catcol}", **CFG.title_specs);\n        ax.grid(**CFG.grid_specs);\n        ax.set(xlabel = \'\', ylabel = \'\');\n        \n    plt.tight_layout();\n    plt.show();\n\n# Implementing the grouped distribution analysis:-\nif CFG.ftre_plots_req == "Y":\n    for contcol in cont_cols:\n        PrintColor(f"\\n{\'-\'* 30} {contcol.capitalize()} distribution analysis {\'-\'* 30}\\n");\n        MakeGrpPlot(train, contcol);\n        print();\n    \nprint();\ncollect();\n')


# <a id="5.5"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > DUPLICATES ANALYSIS<br><div>
#     
# We could simply place the training set predictions for the duplicate rows across all columns and complete the process<br>
# If we consider feature subsets, we could obtain higher number of duplicates. We may have to work with specific duplicate handling thereby<br>

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nPrintColor(f"\\n-------- Duplicates analysis --------\\n", color = Fore.MAGENTA);\n\n_ = train.loc[train[strt_ftre[1:-1]].duplicated(keep = \'first\')];\nPrintColor(f"Train set duplicates = {len(_)} rows");\n\n_ = test.loc[test[strt_ftre[1:-1]].duplicated(keep = \'first\')];\nPrintColor(f"Test set duplicates = {len(_)} rows");\n\n_ = pd.concat([train[strt_ftre[1:-1]], test[strt_ftre[1:-1]]], axis= 0, ignore_index = False);\n_ = len(_.loc[_.duplicated(keep = \'first\')]);\nPrintColor(f"Train-Test set combined duplicates = {_} rows");\n\ndup_df = test.merge(train.drop(columns = [\'Source\'], errors = \'ignore\'), \n                               how     = \'inner\', \n                               on      = list(strt_ftre[1:-1]),\n                    suffixes = {\'\',\'_train\'})[[\'id\', \'id_train\', CFG.target]];\n\nPrintColor(f"\\nDuplicated rows between train and test-\\n")\ndisplay(dup_df.head(5).style.format(precision = 2))\n\nprint();\ncollect();\n')


# <a id="5.6"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > FEATURE INTERACTION AND UNIVARIATE RELATIONS<br><div>
#     
# We aim to do the below herewith<br>
# 1. Correlation<br>
# 2. Mutual information<br>
# 3. Leave One Out model<br>

# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndef MakeCorrPlot(df: pd.DataFrame, data_label:str, figsize = (30, 9)):\n    """\n    This function develops the correlation plots for the given dataset\n    """;\n    \n    fig, axes = plt.subplots(1,2, figsize = figsize, gridspec_kw = {\'hspace\': 0.2, \'wspace\': 0.1},\n                             sharey = True\n                            );\n    \n    for i, method in enumerate([\'pearson\', \'spearman\']):\n        corr_ = df.drop(columns = [\'id\', \'Source\'], errors = \'ignore\').corr(method = method);\n        ax = axes[i];\n        sns.heatmap(data = corr_,  annot= True,fmt= \'.2f\', cmap = \'viridis\',\n                    annot_kws= {\'fontweight\': \'bold\',\'fontsize\': 7.5}, \n                    linewidths= 1.5, linecolor=\'white\', cbar= False, mask= np.triu(np.ones_like(corr_)),\n                    ax= ax\n                   );\n        ax.set_title(f"\\n{method.capitalize()} correlation- {data_label}\\n", **CFG.title_specs);\n        \n    collect();\n    print();\n\n# Implementing correlation analysis:-\nMakeCorrPlot(df = train, data_label = \'Train\', figsize = (26, 8));\nMakeCorrPlot(df = original, data_label = \'Original\', figsize = (26, 8));\nMakeCorrPlot(df = test, data_label = \'Test\', figsize = (26, 8));\n\nprint();\ncollect();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nMutInfoSum = {};\nfor i, df in enumerate([train, original]):\n    MutInfoSum.update({\'Train\' if i == 0 else \'Original\': \n                       mutual_info_regression(df[strt_ftre[1:-1]], df[CFG.target],random_state = CFG.state)\n                      });\n\nMutInfoSum = pd.DataFrame(MutInfoSum, index = strt_ftre[1:-1]);\n\nfig, axes = plt.subplots(1,2, figsize = (14, 4.5), gridspec_kw = {\'wspace\': 0.2});\nfor i in range(2):\n    MutInfoSum.iloc[:, i].plot.bar(ax = axes[i], color = \'tab:blue\');\n    axes[i].set_title(f"{MutInfoSum.columns[i]}- Mutual Information", **CFG.title_specs);\n\nplt.tight_layout();\nplt.show();\nprint();\ncollect();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '  \nif CFG.ftre_imp_req == "Y":\n    model = LGBMRegressor(max_depth     = 8, \n                          random_state  = CFG.state, \n                          num_leaves    = 120,\n                          n_estimators  = 1800, \n                          objective     = \'mae\', \n                          metric        = \'mean_absolute_error\',\n                          reg_alpha     = 0.000012,\n                          reg_lambda    = 0.45,\n                         );\n    \n    X = train.loc[train.Source == "Competition"].drop(columns = [\'id\', \'Source\']);\n    X[\'noise\'] = np.random.normal(0, 3.0, len(X));\n    y,X = X[CFG.target], X.drop(columns = [CFG.target]);\n  \n    fig, ax = plt.subplots(1,1, figsize = (10, 4.5), gridspec_kw = {\'hspace\': 0.2});\n    model.fit(X, y);\n    PermFtreImp = \\\n    permutation_importance(model, X, y, \n                           scoring = \'neg_mean_absolute_error\', \n                           n_repeats = 3,\n                           random_state = CFG.state,\n                          );\n    PermFtreImp = pd.Series(np.mean(PermFtreImp[\'importances\'], axis=1), index = X.columns);\n    noise_level = PermFtreImp.loc[\'noise\'];\n    PermFtreImp.plot.bar(ax = ax, color = \'tab:blue\');\n    ax.axhline(y = noise_level, color = \'red\', lw = 2.0, ls = \'--\');\n    ax.set_title(f"Permutation Importance", **CFG.title_specs);\n    plt.tight_layout();\n    plt.show();\n    del noise_level, X,y, PermFtreImp;\n\nif CFG.ftre_imp_req != "Y":\n    PrintColor(f"\\nFeature importance is not required\\n", color = Fore.RED);\n\ncollect();\nprint();\n')


# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > INFERENCES<br> <div>

# <div style= "font-family: Cambria; letter-spacing: 0px; color:#000000; font-size:110%; text-align:left;padding:3.0px; background: #f2f2f2" >
# 1. The last 3 continuous columns are highly important for the model. Other categorical columns are all equally, but lowly important.<br>
# 2. We will pay special attention to categorical features with importance lower than noise<br>
# 3. Original data could be included in the model. We will still test this with 2 versions and assess the CV score impact.<br>
# 4. We have a few outliers in the continuous columns as seen in the box plot. We may have to treat them in a future run and then proceed with the CV score impact<br>
# 5. All the temperature range columns are extremely highly correlated. It will be better to drop some of these columns for the model.<br>
# 6. We cannot remove a set of features just because their permutation importances are lower than random noise. Permutation importance highlights the relative importance of a feature in a given model and seldom gives a complete picture of the predictability of the feature. We will try several feature combinations and check the CV score. That is the real indicator of model performance.
# </div>

# <a id="6"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0052cc; border-bottom: 8px solid #cc9966" > DATA TRANSFORMS <br><div> 
#     
# This section aims at creating secondary features, scaling and if necessary, conjoining the competition training and original data tables<br>
# We also change the values in the raining days column to reflect similarity with the original data- <br>
# Referred from https://www.kaggle.com/code/paddykb/ps-s3e14-flaml-bfi-be-bop-a-blueberry-do-dahhttps://www.kaggle.com/code/paddykb/ps-s3e14-flaml-bfi-be-bop-a-blueberry-do-dah

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Data transforms:-\nclass Xformer(TransformerMixin, BaseEstimator):\n    """\n    This class is used to create secondary features from the existing data\n    """;\n    \n    def __init__(self): pass\n    \n    def fit(self, X, y= None, **params):\n        self.ip_cols = X.columns;\n        return self;\n    \n    def transform(self, X, y= None, **params):\n        global strt_ftre;\n        df    = X.copy();\n        \n        df.loc[df[\'RainingDays\'] == 26, \'RainingDays\'] = 24;\n        df.loc[df[\'RainingDays\'] == 33, \'RainingDays\'] = 34;       \n      \n        if CFG.sec_ftre_req == "Y":\n            df[\'frtst_mult_frtms\'] = df[\'fruitset\'] * df[\'fruitmass\'];\n            df[\'frtst_div_frtms\']  = df[\'fruitset\'] / df[\'fruitmass\'];\n            df[\'frtst_sum_frtms\']  = df[\'fruitset\'] + df[\'fruitmass\'];\n\n            df[\'frtst_mult_seeds\'] = df[\'fruitset\'] * df[\'seeds\'];\n            df[\'frtst_div_seeds\']  = df[\'fruitset\'] / df[\'seeds\'];\n            df[\'frtst_sum_seeds\']  = df[\'fruitset\'] + df[\'seeds\'];   \n\n            df[\'frtms_mult_seeds\'] = df[\'fruitmass\'] * df[\'seeds\'];\n            df[\'frtms_div_seeds\']  = df[\'fruitmass\'] / df[\'seeds\'];\n            df[\'frtms_sum_seeds\']  = df[\'fruitmass\'] + df[\'seeds\']; \n            \n            df[\'temp_sum\']         = df[strt_ftre[strt_ftre.str.endswith(\'TRange\')]].sum(axis=1);\n            df[\'temp_range\']       = df[\'MaxOfUpperTRange\'] - df[\'MinOfLowerTRange\'];\n            df[\'rain_rate\']        = df[\'AverageRainingDays\'] / df[\'RainingDays\'];\n            \n        if CFG.sec_ftre_req != "Y": \n            PrintColor(f"Secondary features are not required", color = Fore.RED);\n            \n        self.op_cols = df.columns;  \n        return df;\n    \n    def get_feature_names_in(self, X, y=None, **params): \n        return self.ip_cols;    \n    \n    def get_feature_names_out(self, X, y=None, **params): \n        return self.op_cols;\n     \n# Scaling:-\nclass Scaler(TransformerMixin, BaseEstimator):\n    """\n    This class aims to create scaling for the provided dataset\n    """;\n    \n    def __init__(self, scl_method: str, scale_req: str):\n        self.scl_method = scl_method;\n        self.scale_req  = CFG.scale_req;\n        \n    def fit(self,X, y=None, **params):\n        "This function calculates the train-set parameters for scaling";\n        \n        self.scl_cols        = X.drop(columns = [\'id\', \'Source\', CFG.target], errors = \'ignore\').columns;\n        self.params          = X[self.scl_cols].describe(percentiles = [0.25, 0.50, 0.75]).drop([\'count\'], axis=0).T;\n        self.params[\'iqr\']   = self.params[\'75%\'] - self.params[\'25%\'];\n        self.params[\'range\'] = self.params[\'max\'] - self.params[\'min\'];\n        \n        return self;\n    \n    def transform(self,X, y=None, **params):  \n        "This function transform the relevant scaling columns";\n        \n        df = X.copy();\n        if self.scale_req == "Y":\n            if CFG.scl_method == "Z":\n                df[self.scl_cols] = (df[self.scl_cols].values - self.params[\'mean\'].values) / self.params[\'std\'].values;\n            elif CFG.scl_method == "Robust":\n                df[self.scl_cols] = (df[self.scl_cols].values - self.params[\'50%\'].values) / self.params[\'iqr\'].values;\n            elif CFG.scl_method == "MinMax":\n                df[self.scl_cols] = (df[self.scl_cols].values - self.params[\'min\'].values) / self.params[\'range\'].values;\n        else:\n            PrintColor(f"Scaling is not needed", color = Fore.RED);\n    \n        return df;\n    \ncollect();\nprint();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nPrintColor(f"\\n{\'-\'* 20} Data transforms, encoding and scaling {\'-\'* 20}", \n           color = Fore.MAGENTA);\nPrintColor(f"\\nTrain data shape before transforms = {train.shape}");\n\n# Conjoining the train and original data:-\nif CFG.conjoin_orig_data == "Y":\n    train = pd.concat([train, original], axis=0, ignore_index = True);\n    PrintColor(f"We are using the training and original data", color = Fore.RED);\n    \n# Adjusting the data for duplicates:-\nif CFG.conjoin_orig_data == "N":\n    PrintColor(f"We are using the training data only", color = Fore.RED);\n    \nPrintColor(f"Train data shape after conjoining = {train.shape}\\n");\ndisplay(train.columns);\n\nPrintColor(f"\\nTrain data shape before de-dup = {train.shape}");\n_     = list(train.drop(columns = [\'id\', CFG.target], errors = \'ignore\').columns);\ntrain = train.groupby(_).agg({CFG.target : np.mean}).reset_index();\nPrintColor(f"Train data shape after de-dup = {train.shape}");\ndel _;\n\n# Part 1:- Data transforms:--\npipe = Pipeline(steps = [(\'Xform\', Xformer()), \n                         (\'Scl\',Scaler(scl_method = CFG.scl_method, scale_req = CFG.scale_req))\n                        ]);\npipe.fit(train.drop(columns = [CFG.target]), train[CFG.target]);\ntrain, Xtest = pipe.transform(train), pipe.transform(test);\nPrintColor(f"Train-test data shape after scaling = {train.shape} {Xtest.shape}");\n\n# Part 2:- Creating Xtrain and ytrain:--\nXtrain, ytrain = train.drop([CFG.target], axis=1), train[CFG.target];\nPrintColor(f"Train-test data shape after all transforms = {Xtrain.shape} {Xtest.shape}");\n\nprint();\ncollect();\n')


# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > FEATURE ANALYSIS AFTER TRANSFORMS<br> <div>
#     
# We aim to do the below herewith- <br>
# 1. Correlation<br>
# 2. Mutual Information<br>
# 3. Comparison of all features with random noise<br>
#     
# Source idea for point 3 - https://www.kaggle.com/competitions/playground-series-s3e14/discussion/406873

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ftre_imp_req == "Y":\n    PrintColor(f"\\n{\'-\'* 30} Feature analysis after data transforms {\'-\'* 30}\\n");\n\n    # Correlation:-\n    MakeCorrPlot(df = Xtrain, data_label = "Train", figsize = (32, 9));\n    MakeCorrPlot(df = Xtest, data_label = "Test", figsize = (32, 9));\n\n    # Making univariate specific interim tables:-\n    X = Xtrain.loc[Xtrain.Source == \'Competition\'].drop(columns = [\'Source\', \'id\'], errors = \'ignore\');\n    X[\'noise\'] = np.random.normal(0.0, 3.0, len(X));\n    y = ytrain.loc[X.index];\n    Unv_Snp = pd.DataFrame(index = X.columns,columns = [\'MutInfo\']);\n\n    # Mutual information:-\n    Unv_Snp[\'MutInfo\'] = mutual_info_regression(X,y,random_state = CFG.state);\n\n    fig, axes = plt.subplots(1,2, figsize = (20, 4.5), \n                             gridspec_kw = {\'wspace\': 0.2, \'hspace\': 0.25});\n    ax = axes[0];\n    Unv_Snp[\'MutInfo\'].plot.bar(ax = ax, color = \'tab:blue\');\n    ax.set_title(f"Mutual Information", **CFG.title_specs);\n\n    # Permutation Importance:-\n    model = LGBMRegressor(max_depth     = 8, \n                          random_state  = CFG.state, \n                          num_leaves    = 120,\n                          n_estimators  = 5_000, \n                          objective     = \'mae\', \n                          metric        = \'mean_absolute_error\',\n                          reg_alpha     = 0.001,\n                          reg_lambda    = 0.35,\n                         );\n\n    model.fit(X, y);\n    PermFtreImp = \\\n    permutation_importance(model, X, y, \n                           scoring = \'neg_mean_absolute_error\', \n                           n_repeats = 3,\n                           random_state = CFG.state,\n                          );\n    Unv_Snp = \\\n    pd.concat([Unv_Snp, \n               pd.Series(np.mean(PermFtreImp[\'importances\'], axis=1), index = X.columns)],\n              axis=1).rename(columns = {0: \'PermFtreImp\'});\n\n    ax = axes[1];\n    Unv_Snp[\'PermFtreImp\'].plot.bar(ax = ax, color = \'tab:blue\');\n    ax.set_title(f"Permutation Importance", **CFG.title_specs);\n\n    plt.tight_layout();\n    plt.show();\n\n    PrintColor(f"\\nUnivariate Snapshot\\n");\n    display(Unv_Snp.\\\n            sort_values(by = [\'PermFtreImp\'], ascending = False).\\\n            style.\\\n            format(precision = 3)\n           );\n\n    del X,y;\n    \nelse:\n    PrintColor(f"\\nFeature importance is not needed\\n", color = Fore.RED);\n    \nprint();\ncollect();\n')


# <a id="7"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0052cc; border-bottom: 8px solid #cc9966" > MODEL TRAINING <br><div> 
#     
# We commence our model assignment with a simple ensemble of tree-based and linear models and then shall proceed with the next steps

# <a id="7.1"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > BASELINE MODEL<br><div>
#     
# We aim to create a simple baseline model with all features and assess the feature importance herewith. We can then make a suitable decision regarding model choices and associated feature selection to proceed with the model training.<br>
#     
# **We analyze the requisite features as below**<br>
# 1. Take multiple feature subsets from the training data features, keeping the continuous columns always. This is a trial-error technique<br>
# 2. Make a simgle model for all of these elements and assess the CV impact<br>
# 3. Analyse the feature importance plot/ partial dependence plot for the model candidate<br>
# 4. Consider the model with the best feature CV as the best feature subset for the next step<br>

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nif CFG.baseline_req == "Y":\n    PrintColor(f"\\n{\'-\' * 30} Baseline model- different features {\'-\' * 30}\\n");\n    filterwarnings(\'ignore\');\n    ftre_ss = \\\n    [\n    [\'Source\', \'fruitset\', \'fruitmass\', \'seeds\', \n     \'frtst_div_frtms\', \'frtst_sum_frtms\',\n     \'clonesize\', \'honeybee\', \'bumbles\', \'MinOfUpperTRange\',\'AverageRainingDays\'\n    ],\n\n    [\'Source\', \'fruitset\', \'fruitmass\', \'seeds\', \n     \'frtst_mult_frtms\', \'frtst_sum_seeds\', \n     \'bumbles\', \'andrena\', \'osmia\',\'MaxOfUpperTRange\',\'RainingDays\', \n    ], \n\n    [\'Source\', \'fruitset\', \'fruitmass\', \'seeds\',  \n     \'clonesize\', \'honeybee\',\'AverageOfUpperTRange\',\'AverageRainingDays\', \n     \'frtst_div_seeds\', \'frtms_sum_seeds\'\n    ], \n\n    [\'Source\', \'fruitset\', \'fruitmass\', \'seeds\',  \n     \'clonesize\', \'honeybee\',\'AverageOfUpperTRange\',\'AverageRainingDays\'\n    ], \n\n    [\'Source\', \'frtst_mult_frtms\', \'frtst_sum_seeds\',\'frtms_mult_seeds\',  \n     \'clonesize\', \'honeybee\', \'osmia\', \'rain_rate\'    \n    ], \n        \n    [\'Source\', \'fruitset\', \'fruitmass\', \'seeds\', \n     \'clonesize\', \'honeybee\', \'osmia\',\n     \'AverageOfUpperTRange\',\'RainingDays\'    \n    ],  \n        \n    [\'Source\', \'clonesize\', \'honeybee\', \'bumbles\', \'andrena\', \'osmia\',\n     \'MaxOfUpperTRange\', \'MinOfUpperTRange\', \'AverageOfUpperTRange\',\n     \'MaxOfLowerTRange\', \'MinOfLowerTRange\', \'AverageOfLowerTRange\',\n     \'RainingDays\', \'AverageRainingDays\', \'fruitset\', \'fruitmass\', \'seeds\' \n    ],\n        \n    [\'Source\', \'honeybee\', \'bumbles\', \'osmia\',\n     \'AverageOfUpperTRange\',\'AverageOfLowerTRange\',\n     \'RainingDays\', \'fruitset\', \'fruitmass\', \'seeds\' \n    ],   \n    \n    [\'Source\', \'clonesize\', \'honeybee\', \'bumbles\',\n     \'AverageOfUpperTRange\',\'AverageOfLowerTRange\',\n     \'RainingDays\', \'fruitset\', \'fruitmass\', \'seeds\', \'frtst_div_seeds\'\n    ],   \n        \n    [\'Source\', \'fruitset\', \'fruitmass\', \'seeds\', \'bumbles\'] ,\n        \n    [\'Source\', \'fruitset\', \'fruitmass\', \'seeds\', \n     \'clonesize\', \'honeybee\', \'bumbles\', \n     \'MaxOfLowerTRange\', \'AverageRainingDays\' ] ,\n        \n    [\'Source\', \'fruitset\', \'fruitmass\', \'seeds\', \n     \'rain_rate\', \'temp_range\', \'bumbles\', \'frtst_mult_seeds\']     \n    ];\n    \n    for i, cols in enumerate(ftre_ss):\n        model = LGBMRegressor(n_estimators = 10_000, \n                              random_state = CFG.state, \n                              max_depth = 8, \n                              num_leaves = 90,\n                              learning_rate = 0.075, \n                              reg_lambda = 1.25,\n                              metric = \'mae\',\n                              colsample_bytree = 0.65,\n                             );\n        Scores = cross_val_score(model, Xtrain[cols[1:]], ytrain, \n                                 scoring   = \'neg_mean_absolute_error\', \n                                 cv        = CFG.n_splits, \n                                 n_jobs    = -1,\n                                 verbose   = 0,\n                                 fit_params= {\'callbacks\': [log_evaluation(period = 0)]},\n                                );\n        if i == 0: print();\n        PrintColor(f"---> Subset{i}. {\' \' if i <= 9 else \'\'} Mean CV score = {-1 * np.mean(Scores):,.3f}", \n                   color = Fore.GREEN);\n        collect();\n        \nelse:\n    PrintColor(f"We do not need a baseline", color = Fore.RED);\n    \nprint();\ncollect();  \n')


# <a id="7.2"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > ML MODEL SUITE<br><div>
#     
# We will test the impact of the discussion for post-processing on OOF and test predictions as below<br>
# https://www.kaggle.com/competitions/playground-series-s3e14/discussion/407327 
#     
# Please peruse [Paddy KB notebook](https://www.kaggle.com/code/paddykb/ps-s3e14-flaml-bfi-be-bop-a-blueberry-do-dah) for more details

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nUnq_Tgt = train_copy[CFG.target].unique();\ndef PostProcessPred(preds, post_process = CFG.predpstprcs_req.upper()):\n    """\n    This function is inspired by the discussion link above-\n    1. We read the unique target values in the competition train data\n    2. We then compare these values to the predictions\n    3. We round off the predicted values (OOF and test) to the nearest value of the unique target value\n    """;\n    \n    global Unq_Tgt;\n    \n    if post_process == "Y":\n        return np.array([min(Unq_Tgt, key = lambda x: abs(x - pred)) for pred in preds]);\n    else:\n        return preds;\n\nprint();\ncollect();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Initializing model I-O:-\n\nMdl_Master = \\\n{\'CBR\': CatBoostRegressor(**{\'task_type\'           : "GPU" if CFG.gpu_switch == "ON" else "CPU",\n                             \'loss_function\'       : \'MAE\',\n                             \'eval_metric\'         : \'MAE\',\n                             \'bagging_temperature\' : 2.5,\n                             \'colsample_bylevel\'   : 0.8,\n                             \'iterations\'          : 12_000,\n                             \'learning_rate\'       : 0.067,\n                             \'od_wait\'             : 25,\n                             \'max_depth\'           : 8,\n                             \'l2_leaf_reg\'         : 2.25,\n                             \'min_data_in_leaf\'    : 47,\n                             \'random_strength\'     : 0.30, \n                             \'max_bin\'             : 180,\n                           }\n                        ),\n\n \'LGBMR\': LGBMRegressor(**{\'device\'            : "gpu" if CFG.gpu_switch == "ON" else "cpu",\n                           \'objective\'         : \'regression_l1\',\n                           \'metric\'            : \'mean_absolute_error\',\n                           \'boosting_type\'     : \'gbdt\',\n                           \'random_state\'      : CFG.state,\n                           \'feature_fraction\'  : 0.875,\n                           \'learning_rate\'     : 0.05556,\n                           \'max_depth\'         : 8,\n                           \'n_estimators\'      : 12_000,\n                           \'num_leaves\'        : 145,                    \n                           \'reg_alpha\'         : 0.00001,\n                           \'reg_lambda\'        : 1.25,\n                           \'verbose\'           : -1,\n                         }\n                      ),\n\n \'XGBR\': XGBRegressor(**{\'objective\'          : \'reg:squarederror\',\n                         \'eval_metric\'        : \'mae\',\n                         \'random_state\'       : CFG.state,\n                         \'tree_method\'        : "gpu_hist" if CFG.gpu_switch == "ON" else "hist",\n                         \'colsample_bytree\'   : 0.825,\n                         \'learning_rate\'      : 0.0625,\n                         \'max_depth\'          : 8,\n                         \'n_estimators\'       : 10_000,                         \n                         \'reg_alpha\'          : 0.000001,\n                         \'reg_lambda\'         : 3.75,\n                         \'min_child_weight\'   : 39,\n                        }\n                     )\n};\n\n# Selecting relevant columns for the train and test sets:-\nsel_cols = [\'clonesize\', \'honeybee\', \'bumbles\', \'andrena\', \'osmia\',\n            \'MaxOfUpperTRange\', \'RainingDays\',  \'fruitset\', \'fruitmass\', \'seeds\',\n            \'Source\'\n           ];\nXtrain, Xtest = Xtrain[sel_cols], Xtest[sel_cols];\n\n# Initializing output tables for the models:-\nmethods   = list(Mdl_Master.keys());\nOOF_Preds = pd.DataFrame(columns = methods);\nMdl_Preds = pd.DataFrame(index = test.id, columns = methods);\nFtreImp   = pd.DataFrame(index = Xtrain.drop(columns = [\'id\', CFG.target, \'Source\', \'Label\'],\n                                             errors = \'ignore\').columns, \n                         columns = methods\n                        );\nScores    = pd.DataFrame(columns = methods);\n\nprint();\ncollect();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndef TrainMdl(method:str):\n    \n    global Mdl_Master, Mdl_Preds, OOF_Preds, all_cv, FtreImp, Xtrain, ytrain; \n    \n    model     = Mdl_Master.get(method); \n    cols_drop = [\'id\', \'Source\', \'Label\'];\n    scl_cols  = [col for col in Xtrain.columns if col not in cols_drop];\n    cv        = all_cv.get(CFG.mdlcv_mthd);\n    Xt        = Xtest.copy(deep = True);\n    \n    if CFG.scale_req == "N" and method.upper() in [\'RIDGE\', \'LASSO\', \'SVR\']:\n        X, y        = Xtrain, ytrain;\n        scaler      = all_scalers[\'Z\'];\n        X[scl_cols] = scaler.fit_transform(X[scl_cols]);\n        Xt[scl_cols]= scaler.transform(Xt[scl_cols]);\n        PrintColor(f"--> Scaling the data for linear {method} model");\n\n    if CFG.use_orig_allfolds == "Y":\n        X    = Xtrain.query("Source == \'Competition\'");\n        y    = ytrain.loc[ytrain.index.isin(X.index)]; \n        Orig = pd.concat([Xtrain, ytrain], axis=1).query("Source == \'Original\'");\n        \n    elif CFG.use_orig_allfolds != "Y":\n        X,y = Xtrain.copy(deep = True), ytrain.copy(deep = True);\n                \n    # Initializing I-O for the given seed:-        \n    test_preds = 0;\n    oof_preds  = pd.DataFrame(); \n    scores     = [];\n    ftreimp    = 0;\n          \n    for fold_nb, (train_idx, dev_idx) in enumerate(cv.split(X, y)): \n        Xtr  = X.iloc[train_idx].drop(columns = cols_drop, errors = \'ignore\');   \n        Xdev = X.iloc[dev_idx].loc[X.Source == "Competition"].\\\n        drop(columns = cols_drop, errors = \'ignore\'); \n        ytr  = y.loc[y.index.isin(Xtr.index)];\n        ydev = y.loc[y.index.isin(Xdev.index)];\n\n        if CFG.use_orig_allfolds == "Y":\n            Xtr = pd.concat([Xtr, Orig.drop(columns = [CFG.target, \'Source\'], errors = \'ignore\')], \n                            axis = 0, ignore_index = True);\n            ytr = pd.concat([ytr, Orig[CFG.target]], axis = 0, ignore_index = True);\n            \n        # Fitting the model:- \n        try:     \n            model.fit(Xtr, ytr, \n                      eval_set = [(Xdev, ydev)], \n                      verbose = 0,\n                      early_stopping_rounds = CFG.nbrnd_erly_stp\n                     ); \n        except: \n            model.fit(Xtr, ytr) \n            \n        # Collecting predictions and scores and post-processing OOF:-\n        dev_preds = PostProcessPred(model.predict(Xdev));\n        score     = mae(ydev, dev_preds);\n        scores.append(score); \n        Scores.loc[fold_nb, method] = np.round(score, decimals= 6);\n        oof_preds = pd.concat([oof_preds,\n                               pd.DataFrame(index   = Xdev.index, \n                                            data    = dev_preds,\n                                            columns = [method])\n                              ],axis=0, ignore_index= False\n                             );  \n    \n        oof_preds = pd.DataFrame(oof_preds.groupby(level = 0)[method].mean());\n        oof_preds.columns = [method];\n        \n        test_preds = test_preds + model.predict(Xt.drop(columns = cols_drop, errors = \'ignore\')); \n        \n        try: \n            ftreimp += model.feature_importances_;\n        except:\n            ftreimp = 0;\n            \n    num_space = 20 - len(method);\n    PrintColor(f"--> {method}{\'-\' * num_space} {np.mean(scores):.3f}", \n               color = Fore.MAGENTA);\n    del num_space;\n    \n    OOF_Preds[f\'{method}\'] = PostProcessPred(oof_preds.values.flatten());\n    Mdl_Preds[f\'{method}\'] = test_preds.flatten()/ (CFG.n_splits * CFG.n_repeats); \n    FtreImp[method]        = ftreimp / (CFG.n_splits * CFG.n_repeats);\n    collect(); \n    \n# Implementing the ML models:-\nif CFG.ML == "Y": \n    for method in tqdm(methods, "ML models----"): \n        TrainMdl(method);\n    clear_output();\n    \n    if CFG.predpstprcs_req == "Y":\n        PrintColor(f"\\nCV scores across methods with post-processing\\n");\n    else:\n        PrintColor(f"\\nCV scores across methods without post-processing\\n");\n    display(pd.concat([Scores.mean(axis = 0), Scores.std(axis = 0)], axis=1).\\\n            rename(columns = {0: \'Mean\', 1: \'Std\'}).\\\n            style.format(precision = 4).background_gradient(cmap = \'viridis\')\n           );\nelse:\n    PrintColor(f"\\nML models are not needed\\n", color = Fore.RED);\n    \ncollect();\nprint();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ML == "Y":\n    OOF_Preds[CFG.target] = \\\n    pd.concat([Xtrain, ytrain], axis=1).\\\n    query("Source == \'Competition\'")[CFG.target].values; \n    \n    try:\n        all_mthd = FtreImp.drop(columns = [\'RIDGE\', \'LASSO\', \'SVR\'], errors = \'ignore\').columns;\n        fig, axes = plt.subplots(len(all_mthd), 2, figsize = (25, len(all_mthd)*5.5), \n                                 gridspec_kw = {\'wspace\': 0.2, \'hspace\': 0.25},\n                                 width_ratios = [0.75,0.25])\n        for i, method in enumerate(all_mthd):\n            ax = axes[i, 0];\n            FtreImp[method].plot.barh(color = \'#33bbff\', ax = ax);\n            ax.set_title(f"{method} - importances", **CFG.title_specs);\n\n            ax = axes[i,1];\n            rsq = r2_score(OOF_Preds[CFG.target].values, OOF_Preds[method].values);\n            \n            sns.regplot(data= OOF_Preds[[method, CFG.target]], \n                        y = method, x = f"{CFG.target}", \n                        seed= CFG.state, color = \'#b3ecff\', marker = \'o\',\n                        line_kws= {\'linewidth\': 2.25, \'linestyle\': \'--\', \'color\': \'#0000b3\'},\n                        label = f"{method}",\n                        ax = ax,\n                       );\n            ax.set_title(f"{method} RSquare = {rsq:.2%}", **CFG.title_specs);\n            ax.set(ylabel = \'Predictions\', xlabel = \'Actual\')\n            del rsq;\n        \n        plt.tight_layout();\n        plt.show();\n        del all_mthd;\n    \n    except:\n        pass\n    \n    collect();\n    print(); \n')


# <a id="8"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0052cc; border-bottom: 8px solid #cc9966" > ENSEMBLE AND SUBMISSION<br> <div> 
#     
# We shall take up the top 2-3 models from our current model set and then blend with the best 3-4 public notebooks for submission.

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ML == "Y":  \n    # Blending with best public notebooks-\n    all_subs = pd.read_csv("/kaggle/input/playgrounds3e7ensemble/S3E14_PublicModels.csv", index_col = \'id\');\n    \n    all_subs[\'M5\'] = (Mdl_Preds[\'CBR\'] * 0.04 + Mdl_Preds[\'LGBMR\'] * 0.95 + Mdl_Preds[\'XGBR\'] * 0.01).values;\n    all_subs[CFG.target] = PostProcessPred(np.average(all_subs, weights = [0.45, 0.30, 0.10, 0.10, 0.05], axis=1));\n    all_subs[[CFG.target]].reset_index().to_csv(f"Submission_V{CFG.version_nb}.csv", index = None);\n    all_subs[CFG.target] = PostProcessPred(all_subs[CFG.target].values);\n    \n    # Saving other datasets:-\n    OOF_Preds.add_prefix(f"V{CFG.version_nb}_").to_csv(f"OOF_Preds_V{CFG.version_nb}.csv");\n    Mdl_Preds.add_prefix(f"V{CFG.version_nb}_").to_csv(f"Mdl_Preds_V{CFG.version_nb}.csv");\n    Scores.to_csv(f"Scores_V{CFG.version_nb}.csv");\n    \n    PrintColor(f"Final submission file");\n    display(all_subs[[CFG.target]].head(5).style.format(precision = 3));\n    \ncollect();\nprint();\n')


# <a id="9"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0052cc; border-bottom: 8px solid #cc9966" > NEXT STEPS<br> <div> 

# <div style= "font-family: Cambria; letter-spacing: 0px; color:#000000; font-size:110%; text-align:left;padding:3.0px; background: #f2f2f2" >
# 1. Better feature engineering<br>
# 2. Better experiments with scaling, encoding with categorical columns. This seems to have some promise<br>
# 3. Better model tuning<br>
# 4. Better emphasis on secondary features<br>
# 5. Adding more algorithms and new methods to the model suite<br>
# </div>

# **References**<br>
# 1. https://www.kaggle.com/competitions/playground-series-s3e14/discussion/407327<br>
# 2. https://www.kaggle.com/code/paddykb/ps-s3e14-flaml-bfi-be-bop-a-blueberry-do-dah<br>
# 3. https://www.kaggle.com/code/tetsutani/ps3e14-eda-various-models-ensemble-baseline<br>
# 4. https://www.kaggle.com/code/sergiosaharovskiy/ps-s3e14-2023-eda-and-submission<br>
