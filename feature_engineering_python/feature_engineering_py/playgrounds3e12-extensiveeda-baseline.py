#!/usr/bin/env python
# coding: utf-8

# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:130%; text-align:left;padding:2.5px; border-bottom: 7px solid #ffc266; background:#004d99" > TABLE OF CONTENTS<br>  
# * [IMPORTS](#1)
# * [INTRODUCTION](#2)
#     * [CONFIGURATION](#2.1) 
#     * [EXECUTIVE SUMMARY](#2.2)
# * [PREPROCESSING](#3)
#     * [INFERENCES](#3.1)
# * [ADVERSARIAL CV](#4)
#     * [INFERENCES](#4.1)
# * [EDA- VISUALS](#5)
#     * [TARGET BALANCE](#5.1)
#     * [PAIRPLOTS](#5.2)
#     * [DISTRIUTION PLOTS](#5.3)
#     * [INFERENCES](#5.4)
# * [UNIVARIATE FEATURE IMPORTANCE](#6)
#     * [INFERENCES](#6.1)    
# * [DATA TRANSFORMS](#7)
# * [MODEL TRAINING- BASELINE](#8)
# * [ENSEMBLE](#9)
# * [OUTRO](#10)

# <a id="1"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:130%; text-align:left;padding:2.5px; border-bottom: 7px solid #ffc266; background:#004d99" > IMPORTS
# <div>

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# General library imports:-\n\nimport pandas as pd;\nimport numpy as np;\nfrom scipy.stats import mode, kstest, normaltest, shapiro, anderson, jarque_bera;\nfrom collections import Counter;\nfrom itertools import product;\nfrom termcolor import colored;\nfrom colorama import Fore, Style, init;\nfrom warnings import filterwarnings;\nfilterwarnings(\'ignore\');\n\nfrom tqdm.notebook import tqdm;\nfrom IPython.display import clear_output;\n\nimport seaborn as sns;\nimport matplotlib.pyplot as plt;\nfrom gc import collect;\nfrom pprint import pprint;\n\npd.set_option(\'display.max_columns\', 50);\npd.set_option(\'display.max_rows\', 50);\n\n# Setting rc parameters in seaborn for plots and graphs- \n# Reference - https://matplotlib.org/stable/tutorials/introductory/customizing.html:-\n# To alter this, refer to matplotlib.rcParams.keys()\n\nsns.set({"axes.facecolor"       : "#ffffff",\n         "figure.facecolor"     : "#ffffff",\n         "axes.edgecolor"       : "#000000",\n         "grid.color"           : "#ffffff",\n         "font.family"          : [\'Cambria\'],\n         "axes.labelcolor"      : "#000000",\n         "xtick.color"          : "#000000",\n         "ytick.color"          : "#000000",\n         "grid.linewidth"       : 0.90,  \n         "grid.linestyle"       : "--",\n         "axes.titlecolor"      : \'tab:blue\',\n         \'axes.titlesize\'       : 10,\n         \'axes.labelweight\'     : "bold",\n         \'legend.fontsize\'      : 7.0,\n         \'legend.title_fontsize\': 7.0,\n         \'font.size\'            : 8.0,\n         \'xtick.labelsize\'      : 7.5,\n         \'ytick.labelsize\'      : 7.5,        \n        });\n\nprint();\ncollect();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Importing model and pipeline specifics:-\n\n# Pipeline specifics:-\nfrom sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler;\nfrom sklearn.model_selection import (RepeatedStratifiedKFold as RSKF, \n                                     StratifiedKFold as SKF,\n                                     KFold, \n                                     RepeatedKFold as RKF, \n                                     cross_val_score);\nfrom sklearn.inspection import permutation_importance, PartialDependenceDisplay as PDD;\nfrom sklearn.feature_selection import mutual_info_classif;\nfrom sklearn.inspection import permutation_importance;\nfrom sklearn.pipeline import Pipeline;\nfrom sklearn.base import BaseEstimator, TransformerMixin;\n\n# ML Model training:-\nfrom sklearn.metrics import roc_auc_score;\nfrom sklearn.linear_model import LogisticRegression;\nfrom sklearn.svm import SVC;\nfrom xgboost import XGBClassifier, XGBRegressor;\nfrom lightgbm import LGBMClassifier, LGBMRegressor;\nfrom catboost import CatBoostClassifier, CatBoostRegressor;\nfrom sklearn.ensemble import (ExtraTreesClassifier as ETC, \n                              RandomForestClassifier as RFC, \n                              RandomForestRegressor as RFR,\n                              ExtraTreesRegressor as ETR\n                             );\n\nprint();\ncollect();\n')


# <a id="2"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:130%; text-align:left;padding:2.5px; border-bottom: 7px solid #ffc266; background:#004d99" > INTRODUCTION
# <div>

# | Version<br>Number | Version Details | Best CV score| Public LB score|
# | :-: | --- | :-: | :-: |
# | **V1** |* Extensive EDA with appropriate configuration class<br>* No scaling<br>* No extra features<br>* Baseline XGB, XGBR, RFC, ETR, ETC, Logistic models<br>* Simple ensemble with average |0.809739 |0.85866 |
# | **V2** |* Better feature choices<br>* 10 ML models with better plots and hand tuning<br>* Weighted average ensemble |0.822832 | 0.85006|
# | **V3** |* Better feature choices- refer AmbrosM's post<br>* 11 ML models with better plots and hand tuning<br>* Weighted average ensemble with selected models |0.821121 | 0.85200|
# | **V4** |* Configuration class description<br>* Slight adjustment of features (secondary features)<br>* Partial dependency plots in model training || |

# <a id="2.1"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:black; font-size:120%; text-align:left;padding:2.5px; border-bottom: 6px solid black; background: #e6f2ff"> CONFIGURATION PARAMETERS
# <div>
#     
# |Section          | Parameter                   | Description                                    | Intended values |
# |---              | :-:                         | ---                                            | :-:|
# |Data-preparation | gpu_switch                  | Turns the GPU ON/ OFF- here it os OFF as the data is too small| OFF/ON|
# |Data-preparation | state                       | Random seed integer                                   | integer value|
# |Data-preparation | adv_cv_req                  | Checks if adversarial CV is needed        | (Y/N)|
# |Data-preparation | ftre_plots_req              | Checks if plots and visuals are needed in EDA | (Y/N)|
# |Data-preparation | ftre_imp_req                | Checks if plots and visuals are needed for feature importance after transforms |  (Y/N)|  
# |Data-transforms  | conjoin_orig_data           | Checks if original data needs to be appended to training data | (Y/N)|
# |Data-transforms  | sec_ftre_req                | Checks if we need any secondary feature |(Y/N)|
# |Data-transforms  | scale_req                   | Checks if we need any scaling method |(Y/N)|
# |Data-transforms  | scl_method                  | Puts up a scaling method - **keep a value here even if scale_req == N**|(Robust/ Z/ MinMax) |
# |Model training   | ML                          | Checks if ML models (except for GAM) are needed, for EDA only, keep this as N| (Y/N)|
# |Model training   | n_splits                    | Provides the number of data splits in CV strategy|integer value, usually between 3 and 15|
# |Model training   | n_repeats                   | Provides the number of data repeats in CV strategy| integer value|
# |Model training   | nbrnd_erly_stp              | Provides the number of early-stopping rounds in ensemble tree models to reduce overfitting| integer value|
# |Model training   | prtldepplot_req             | Plots partial dependency plots from model estimator on training| (Y/N)|
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Configuration class:-\nclass CFG:\n    "Configuration class for parameters and CV strategy for tuning and training";\n    \n    # Data preparation:-   \n    version_nb         = 4;\n    gpu_switch         = "OFF"; \n    state              = 42;\n    target             = "target";\n    episode            = 12;\n    path               = f"/kaggle/input/playground-series-s3e{episode}/";\n    orig_path          = f"/kaggle/input/kidney-stone-prediction-based-on-urine-analysis/kindey stone urine analysis.csv";\n    adv_cv_req         = "Y";\n    ftre_plots_req     = "Y";\n    ftre_imp_req       = "Y";\n    \n    # Data transforms and scaling:-    \n    conjoin_orig_data  = "N";\n    sec_ftre_req       = "Y";\n    scale_req          = "Y";\n    scl_method         = "Robust";\n    \n    # Model Training:-  \n    ML                 = "Y";\n    n_splits           = 10;\n    n_repeats          = 5;\n    nbrnd_erly_stp     = 50;\n    prtldepplot_req    = "Y";\n        \n    # Global variables for plotting:-\n    grid_specs = {\'visible\': True, \'which\': \'both\', \'linestyle\': \'--\', \n                           \'color\': \'lightgrey\', \'linewidth\': 0.75};\n    title_specs = {\'fontsize\': 9, \'fontweight\': \'bold\', \'color\': \'tab:blue\'};\n\n# Color printing    \ndef PrintColor(text:str, color = Fore.BLUE, style = Style.BRIGHT):\n    "Prints color outputs using colorama using a text F-string";\n    print(colored(style + color + text + Style.RESET_ALL)); \n\n# Scaler to be used for continuous columns:- \nall_scalers = {\'Robust\': RobustScaler(), \n               \'Z\': StandardScaler(), \n               \'MinMax\': MinMaxScaler()\n              };\nscaler      = all_scalers.get(CFG.scl_method);\n\n# Commonly used CV strategies for later usage:-\nall_cv= {\'KF\'  : KFold(n_splits= CFG.n_splits, shuffle = True, random_state= CFG.state),\n         \'RKF\' : RKF(n_splits= CFG.n_splits, n_repeats = CFG.n_repeats, random_state= CFG.state),\n         \'RSKF\': RSKF(n_splits= CFG.n_splits, n_repeats = CFG.n_repeats, random_state= CFG.state),\n         \'SKF\' : SKF(n_splits= CFG.n_splits, shuffle = True, random_state= CFG.state)\n        };\nprint();\n\nPrintColor(f"--> Configuration done!");\ncollect();\n')


# <a id="2.2"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:black; font-size:120%; text-align:left;padding:2.5px; border-bottom: 6px solid black; background: #e6f2ff"> EXECUTIVE SUMMARY
# <div>
# 
# This notebook is starter for the **Playground Series 3- Episode 12**. This is a binary classifier from a synthetic dataset created from the llink below<br>
# https://www.kaggle.com/datasets/vuppalaadithyasairam/kidney-stone-prediction-based-on-urine-analysis
# 
# The evaluation metric is **ROC-AUC**<br>
# 
# **Column description**
#     
# | Column Name                 | Description                                    | 
# | :-:                         | ---                                            | 
# | specific gravity            | Density of materials in the urine              |
# | pH                          | Acidity of urine                               |
# | osmolarity                  | Molecule concentration                         |
# | conductivity                | Concentration of charged ions in the sample    |
# | urea concentration          | Concentration of urea in milli-moles/ litre    |
# | calcium concentration       | Concentration of calcium in milli-moles/ litre |
# | **target**                  | Binary target variable                         |

# <a id="3"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:130%; text-align:left;padding:2.5px; border-bottom: 7px solid #ffc266; background:#004d99" > PREPROCESSING
# <div>

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nPrintColor(f"\\n---------- Data Preprocessing ---------- \\n", color = Fore.MAGENTA);\n\n# Reading the train-test datasets:-\ntrain    = pd.read_csv(CFG.path + f"train.csv");\ntest     = pd.read_csv(CFG.path + f"test.csv");\noriginal = pd.read_csv(CFG.orig_path);\noriginal.insert(0, \'id\', range(len(original)));\noriginal[\'id\'] = original[\'id\'] + test[\'id\'].max() + 1;\n\ntrain[\'Source\'], test[\'Source\'], original[\'Source\'] = "Competition", "Competition", "Original";\nPrintColor(f"\\nData shapes- [train, test, original]-- {train.shape} {test.shape} {original.shape}\\n");\n\n# Creating dataset information:\nPrintColor(f"\\nTrain information\\n");\ndisplay(train.info());\nPrintColor(f"\\nTest information\\n")\ndisplay(test.info());\nPrintColor(f"\\nOriginal data information\\n")\ndisplay(original.info());\nprint();\n\n# Displaying column description:-\nPrintColor(f"\\nTrain description\\n");\ndisplay(train.describe(percentiles= [0.05, 0.25, 0.50, 0.75, 0.9, 0.95, 0.99]).\\\n        transpose().\\\n        drop(columns = [\'count\'], errors = \'ignore\').\\\n        drop([CFG.target], axis=0, errors = \'ignore\').\\\n        style.format(precision = 2));\n\nPrintColor(f"\\nTest description\\n");\ndisplay(test.describe(percentiles= [0.05, 0.25, 0.50, 0.75, 0.9, 0.95, 0.99]).\\\n        transpose().\\\n        drop(columns = [\'count\'], errors = \'ignore\').\\\n        drop([CFG.target], axis=0, errors = \'ignore\').\\\n        style.format(precision = 2));\n\nPrintColor(f"\\nOriginal description\\n");\ndisplay(original.describe(percentiles= [0.05, 0.25, 0.50, 0.75, 0.9, 0.95, 0.99]).\\\n        transpose().\\\n        drop(columns = [\'count\'], errors = \'ignore\').\\\n        drop([CFG.target], axis=0, errors = \'ignore\').\\\n        style.format(precision = 2));\n\n# Collating the column information:-\nstrt_ftre = test.columns;\nPrintColor(f"\\nStarting columns\\n");\ndisplay(strt_ftre);\n\n# Creating a copy of the datasets for further use:-\ntrain_copy, test_copy, orig_copy = \\\ntrain.copy(deep= True), test.copy(deep = True), original.copy(deep = True);\n\n# Dislaying the unique values across train-test-original:-\nPrintColor(f"\\nUnique values\\n");\n_ = pd.concat([train.nunique(), test.nunique(), original.nunique()], axis=1);\n_.columns = [\'Train\', \'Test\', \'Original\'];\ndisplay(_.style.background_gradient(cmap = \'Blues\').format(formatter = \'{:,.0f}\'));\n\n# Normality check:-\ncols = list(strt_ftre[1:-1]);\nPrintColor(f"\\nShapiro Wilk normality test analysis\\n");\npprint({col: [np.round(shapiro(train[col]).pvalue,decimals = 4), \n              np.round(shapiro(test[col]).pvalue,4) if col != CFG.target else np.NaN,\n              np.round(shapiro(original[col]).pvalue,4)] for col in cols\n       }, indent = 5, width = 100, depth = 2, compact= True);\n\nPrintColor(f"\\nNormal-test normality test analysis\\n");\npprint({col: [np.round(normaltest(train[col]).pvalue,decimals = 4), \n              np.round(normaltest(test[col]).pvalue,4) if col != CFG.target else np.NaN,\n              np.round(normaltest(original[col]).pvalue,4)] for col in cols\n       }, indent = 5, width = 100, depth = 2, compact= True);\n\nPrintColor(f"\\nK-S normality test analysis\\n");\npprint({col: [np.round(kstest(train[col], cdf = \'norm\').pvalue,decimals = 4), \n              np.round(kstest(test[col], cdf = \'norm\').pvalue,4) if col != CFG.target else np.NaN,\n              np.round(kstest(original[col], cdf = \'norm\').pvalue,4)] for col in cols\n       }, indent = 5, width = 100, depth = 2, compact= True);\nprint();\n')


# <a id="3.1"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:black; font-size:120%; text-align:left;padding:2.5px; border-bottom: 6px solid black; background: #e6f2ff"> INFERENCES
# <div>
# 
# 1. Train and original data appear to have few outliers<br>
# 2. Some columns are close to being normally distributed<br>
# 3. We don't have any nulls in the data at all<br>
# 4. We have a completely numeric dataset<br>
#     
# ### **Side-note -- Interpreting normality tests:-**
# 1. We are using Shapiro-Wilk, NormalTest and 1-sample Kolmogorov Smirnov tests for normality with the p-value evaluator<br>
# 2. p-value illustrates the area under the tail region of the statistics test. Here, we may ensue the below-<br>
# a. Null hypothesis- Data is non-normal<br>
# b. Alternative hypothesis- Data is normally distributed<br>
# c. p-value illustrates the tail area. If p-value is lower than the determined threshold, we reject the null hypothesis.<br>
# d. Herewith, our p-values are usually lesser than 5%, a common threshold for statistical tests throughout. In some cases, the p-value crosses the 5% threshold too.<br>
# e. Wherever the p-value is more than 5%, we cannot reject the null hypothesis (we have insufficient evidence to reject the null hypothesis). We can infer that the data in such cases is normally distributed.<br> 

# <a id="4"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:130%; text-align:left;padding:2.5px; border-bottom: 7px solid #ffc266; background:#004d99" > ADVERSARIAL CV
# <div>
#     
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Performing adversarial CV between the 2 specified datasets:-\ndef Do_AdvCV(df1:pd.DataFrame, df2:pd.DataFrame, source1:str, source2:str):\n    "This function performs an adversarial CV between the 2 provided datasets if needed by the user";\n    \n    # Adversarial CV per column:-\n    ftre = train.drop(columns = [\'id\', CFG.target, "Source"], errors = \'ignore\').columns;\n    adv_cv = {};\n\n    for col in ftre:\n        PrintColor(f"---> Current feature = {col}", style = Style.NORMAL);\n        shuffle_state = np.random.randint(low = 10, high = 100, size= 1);\n\n        full_df = \\\n        pd.concat([df1[[col]].assign(Source = source1), df2[[col]].assign(Source = source2)], \n                  axis=0, ignore_index = True).\\\n        sample(frac = 1.00, random_state = shuffle_state);\n\n        full_df = full_df.assign(Source_Nb = full_df[\'Source\'].eq(source2).astype(np.int8));\n\n        # Checking for adversarial CV:-\n        model = LGBMClassifier(random_state = CFG.state, max_depth = 3, learning_rate = 0.05);\n        cv    = all_cv[\'RSKF\'];\n        score = np.mean(cross_val_score(model, \n                                        full_df[[col]], \n                                        full_df.Source_Nb, \n                                        scoring= \'roc_auc\', \n                                        cv= cv)\n                       );\n        adv_cv.update({col: round(score, 4)});\n        collect();\n    \n    del ftre;\n    \n    PrintColor(f"\\nResults\\n");\n    pprint(adv_cv, indent = 5, width = 20, depth = 1);\n    collect();\n    \n    fig, ax = plt.subplots(1,1,figsize = (12, 5));\n    pd.Series(adv_cv).plot.bar(color = \'tab:blue\', ax = ax);\n    ax.axhline(y = 0.60, color = \'red\', linewidth = 2.75);\n    ax.grid(**CFG.grid_specs); \n    plt.yticks(np.arange(0.0, 0.81, 0.05));\n    plt.show();\n    \n# Implementing the adversarial CV:-\nif CFG.adv_cv_req == "Y":\n    PrintColor(f"\\n---------- Adversarial CV - Train vs Original ----------\\n", \n               color = Fore.MAGENTA);\n    Do_AdvCV(df1 = train, df2 = original, source1 = \'Train\', source2 = \'Original\');\n    \n    PrintColor(f"\\n---------- Adversarial CV - Train vs Test ----------\\n", \n               color = Fore.MAGENTA);\n    Do_AdvCV(df1 = train, df2 = test, source1 = \'Train\', source2 = \'Test\');\n    \n    PrintColor(f"\\n---------- Adversarial CV - Original vs Test ----------\\n", \n               color = Fore.MAGENTA);\n    Do_AdvCV(df1 = train, df2 = test, source1 = \'Original\', source2 = \'Test\');   \n    \nif CFG.adv_cv_req == "N":\n    PrintColor(f"\\nAdversarial CV is not needed\\n", color = Fore.RED);\n    \ncollect();\nprint();\n')


# <a id="4.1"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:black; font-size:120%; text-align:left;padding:2.5px; border-bottom: 6px solid black; background: #e6f2ff"> INFERENCES
# <div>
# 
# 1. We need to investigate the train-original distribution as the adversarial GINI is quite different from 50%<br>
# 2. Train-test belong to the same distribution, we can perhaps rely on the CV score

# <a id="5"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:130%; text-align:left;padding:2.5px; border-bottom: 7px solid #ffc266; background:#004d99" > EDA AND VISUALS
# <div>

# <a id="5.1"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:black; font-size:120%; text-align:left;padding:2.5px; border-bottom: 6px solid black; background: #e6f2ff"> TARGET BALANCE
# <div>

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ftre_plots_req == "Y":\n    fig, axes = plt.subplots(1,2, figsize = (12, 5), sharey = True, gridspec_kw = {\'wspace\': 0.25});\n    \n    for i, df in tqdm(enumerate([train, original]), "Target balance ---> "):\n        ax= axes[i];\n        a = df[CFG.target].value_counts(normalize = True);\n        _ = ax.pie(x = a , labels = a.index.values, \n                   explode      = [0.0, 0.25], \n                   startangle   = 30, \n                   shadow       = True, \n                   colors       = [\'#004d99\', \'#ac7339\'], \n                   textprops    = {\'fontsize\': 8, \'fontweight\': \'bold\', \'color\': \'white\'},\n                   pctdistance  = 0.50, autopct = \'%1.2f%%\'\n                  );\n        df_name = \'Train\' if i == 0 else "Original";\n        _ = ax.set_title(f"\\n{df_name} data- target\\n", **CFG.title_specs);\n\n    plt.tight_layout();\n    plt.show();\n    \ncollect();\nprint();\n')


# <a id="5.2"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:black; font-size:120%; text-align:left;padding:2.5px; border-bottom: 6px solid black; background: #e6f2ff"> PAIR-PLOTS
# <div>

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ftre_plots_req == "Y":\n    PrintColor(f"\\nTrain data- pair plots\\n");\n    _ = sns.pairplot(data = train.drop(columns = [\'id\',\'Source\', CFG.target], errors = \'ignore\'), \n                     diag_kind = \'kde\', markers= \'o\', plot_kws= {\'color\': \'tab:blue\'}               \n                    );\n\nprint();\ncollect();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'if CFG.ftre_plots_req == "Y":\n    PrintColor(f"\\nOriginal data- pair plots\\n");\n    _ = sns.pairplot(data = original.drop(columns = [\'id\',\'Source\', CFG.target], errors = \'ignore\'), \n                     diag_kind = \'kde\', markers= \'o\', plot_kws= {\'color\': \'tab:blue\'}               \n                    );\nprint();\ncollect();\n')


# <a id="5.3"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:black; font-size:120%; text-align:left;padding:2.5px; border-bottom: 6px solid black; background: #e6f2ff"> DISTRIBUTION PLOTS
# <div>

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Violin plots for numeric columns:-\nif CFG.ftre_plots_req == "Y":\n    PrintColor(f"\\nDistribution plots- numerical columns\\n");\n    num_cols = strt_ftre[1:-1];\n    fig, axes = plt.subplots(2, len(num_cols) , figsize = (36, 16),\n                             gridspec_kw= {\'wspace\': 0.2, \'hspace\': 0.25}, \n                             sharex = True);\n    for i, col in enumerate(num_cols):\n        ax = axes[0, i];\n        sns.violinplot(data = train[col], linewidth= 2.5,color= \'#0073e6\', ax = ax);       \n        ax.set_title(f"\\n{col}_Train\\n", **CFG.title_specs);\n        ax.grid(**CFG.grid_specs);\n\n        ax = axes[1, i];\n        sns.violinplot(data = original[col], linewidth= 2.5,color= \'#004d4d\', ax = ax);       \n        ax.set_title(f"\\n{col}_Original\\n", **CFG.title_specs);\n        ax.grid(**CFG.grid_specs); \n\n    plt.tight_layout();\n    plt.show();\n    del num_cols;\n    \ncollect();\nprint();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Distribution analysis by target:-\ndef AnalyzeDist(df:pd.DataFrame):\n    "Plots the KDE plot by the binary target";\n    \n    fig, axes = plt.subplots(2,3, figsize = (18, 7.5),\n                             gridspec_kw= {\'wspace\': 0.2, \'hspace\': 0.3});\n\n    for i, col in enumerate(strt_ftre[1:-1]):\n        ax = axes[i//3, i%3];\n        sns.kdeplot(data      = df[[col, CFG.target]], \n                    x         = col, \n                    hue       = CFG.target, \n                    palette   = [\'#005c99\', \'#e63900\'], \n                    shade     = False, \n                    linewidth = 2.50,\n                    ax        = ax\n                   );\n        ax.set_title(f"\\n{col}\\n", **CFG.title_specs);\n        ax.grid(**CFG.grid_specs);\n        ax.set(xlabel = \'\', ylabel = \'\');\n\n    plt.tight_layout();\n    plt.show();\n    collect();\n    \n# Implementing the feature plots:-\nif CFG.ftre_plots_req.upper() == "Y":\n    PrintColor(f"\\nTrain features\\n");\n    AnalyzeDist(df = train);\n    print();\n    PrintColor(f"\\nOriginal features\\n");\n    AnalyzeDist(df = original);   \n    \nelif CFG.ftre_plots_req.upper() != "Y":\n    PrintColor(f"\\nFeature plots are not needed\\n", color = Fore.RED);\n    \ncollect();\nprint();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Distribution analysis by dataset:-\n\nfull_df = \\\npd.concat([train[strt_ftre[1:-1]].assign(Source = "Train"),\n           test[strt_ftre[1:-1]].assign(Source = "Test"),\n           original[strt_ftre[1:-1]].assign(Source = "Original")\n          ], ignore_index = True\n         );\n\nfig, axes = plt.subplots(2,3, figsize = (18, 8), \n                         gridspec_kw = {\'wspace\': 0.25, \'hspace\': 0.30});\nfor i, col in enumerate(strt_ftre[1:-1]):\n    ax = axes[i//3, i%3];\n    sns.kdeplot(data      = full_df[[\'Source\', col]], \n                x         = col, \n                hue       = \'Source\',\n                palette   = [\'#006bb3\', \'#e63900\', \'#00cc44\'], \n                shade     = None, \n                ax        = ax,\n                linewidth = 2.50,\n               );\n    ax.set_title(f"\\n{col}\\n", **CFG.title_specs);\n    ax.set(xlabel = \'\', ylabel = \'\');\n    ax.grid(**CFG.grid_specs);\n\nplt.suptitle(f"\\nFeature distribution analysis across datasets\\n", \n             fontsize = 12, color = \'#005266\', \n             fontweight = \'bold\');\nplt.tight_layout();\nplt.show();\n\nprint();\ncollect();\n')


# <a id="5.4"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:black; font-size:120%; text-align:left;padding:2.5px; border-bottom: 6px solid black; background: #e6f2ff"> INFERENCES
# <div>
# 
# 1. Target is balanced, no need to use imbalanced techniques<br>
# 2. Meak feature interactions are seen across some columns, linear interactions are seen in the original data<br>
# 3. pH is an important column, where the target values do not cause a significant change of distribution<br>

# <a id="6"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:130%; text-align:left;padding:2.5px; border-bottom: 7px solid #ffc266; background:#004d99" > UNIVARIATE FEATURE IMPORTANCE
# <div>
#     
# **We will do a leave-one-out analysis and singular feature analysis to check the strength of relationship with the target**

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nUnv_Prf_Sum = pd.DataFrame(data = None, columns = [\'Unv_Clsf\', \'LOO\'], \n                           index = strt_ftre[1:-1]\n                          );\nmodel       = LGBMClassifier(random_state   = CFG.state, \n                             max_depth      = 3, \n                             learning_rate  = 0.085,\n                             num_leaves     = 80, \n                            );\n\nfor i, ftre in tqdm(enumerate(strt_ftre[1:-1]), "Univariate Analysis ---- "):\n    # Initiating single feature relationship analysis:-\n    score = \\\n    cross_val_score(model, train[[ftre]], train[CFG.target], \n                    cv      = all_cv[\'RSKF\'],\n                    scoring = \'roc_auc\', \n                    n_jobs  = -1,\n                    verbose = 0\n                   );\n    Unv_Prf_Sum.loc[ftre, \'Unv_Clsf\'] = np.mean(score);\n    del score;\n    \n    # Initiating LOO:-\n    cols  = [col for col in strt_ftre[1:-1] if col != ftre];\n    score = \\\n    cross_val_score(model, train[cols], train[CFG.target], \n                    cv      = all_cv[\'RSKF\'],\n                    scoring = \'roc_auc\', \n                    n_jobs  = -1,\n                    verbose = 0,\n                   );\n    Unv_Prf_Sum.loc[ftre, \'LOO\'] = np.mean(score);\n    del score, cols;\n    collect();\n    \n# Plotting the feature analysis:-\nfig, axes = plt.subplots(1,2, figsize = (13, 4.5), \n                         sharey       = True, \n                         gridspec_kw  = {\'hspace\': 0.4});\n\nfor i, col in enumerate(Unv_Prf_Sum.columns):\n    ax = axes[i];\n    Unv_Prf_Sum.loc[:, col].plot.bar(color = \'#0059b3\', ax = ax);\n    ax.set_title(f"{col}", **CFG.title_specs);\n    ax.grid(**CFG.grid_specs);\n\nplt.yticks(np.arange(0.0, 0.9, 0.05));\nplt.suptitle(f"Univariate performance\\n", \n             color      = \'#005266\', \n             fontsize   = 12, \n             fontweight = \'bold\'\n            );\nplt.tight_layout();\nplt.show();\ncollect();\nprint();\n')


# <a id="6.1"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:black; font-size:120%; text-align:left;padding:2.5px; border-bottom: 6px solid black; background: #e6f2ff"> INFERENCES
# <div>
# 
# 1. **Calc** seems like the most important column<br>
# 2. **pH** seems like the lowest important column<br>

# <a id="7"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:130%; text-align:left;padding:2.5px; border-bottom: 7px solid #ffc266; background:#004d99" > DATA TRANSFORMS
# <div>
#     
# We will incorporate some inputs from -<br>
# 1. https://www.kaggle.com/code/oscarm524/ps-s3-ep12-eda-modelinghttps://www.kaggle.com/code/oscarm524/ps-s3-ep12-eda-modeling
# 2. https://www.kaggle.com/competitions/playground-series-s3e12/discussion/400152https://www.kaggle.com/competitions/playground-series-s3e12/discussion/400152

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Making secondary features:-\nclass SecFtreMaker(BaseEstimator, TransformerMixin):\n    "Makes encoded features and other secondary features for the data transformation step";\n    \n    def __init__(self): pass\n        \n    def fit(self, X, y= None, **fit_params):\n        return self;\n    \n    def transform(self, X, y = None, **transform_params):\n        df = X.copy(deep = True);  \n        \n        if CFG.sec_ftre_req == "Y":\n            df[\'calc\']             = df[\'calc\'].clip(None, 8.00);\n            df[\'gravity\']          = df.gravity.clip(None, 1.03);\n            \n            df[\'Cond_Gvty_Rt\']     = df[\'cond\'] / df[\'gravity\'];\n            df[\'Urea_Osmo\']        = df[\'urea\'] * df[\'osmo\'];\n            df[\'Calc_Gvty\']        = df[\'calc\'] * df[\'gravity\'];\n            df[\'Osmo_Gvty\']        = df[\'osmo\'] * df[\'gravity\'];\n            df[\'Calc_Urea_Rt\']     = df[\'calc\'] / df[\'urea\'];\n            df[\'Calc_Gvty_Rt\']     = df[\'calc\'] / df[\'gravity\'];\n            \n            df[\'RF_gravity\']       = np.where(df[\'gravity\'] >= 1.030, 1,0).astype(np.int8);\n            df[\'RF_osmo\']          = np.where(df[\'osmo\'] >= 1200, 1,0).astype(np.int8);\n            df[\'RF_calc\']          = np.where(df[\'calc\'] >= 7.50, 1,0).astype(np.int8); \n            df[\'RF_total\']         = df[\'RF_gravity\'] + df[\'RF_osmo\'] + df[\'RF_calc\'];\n          \n            df[\'Sq_cond\']          = df[\'cond\'] ** 2;\n            df[\'Sq_calc\']          = df[\'calc\'] ** 2;  \n            df[\'Cond_Calc\']        = df[\'cond\'] * df[\'calc\'];\n            df[\'Cond_Calc_Rt\']     = df[\'cond\'] / df[\'calc\'];\n            df[\'Cond_Calc_Tot\']    = df[\'cond\'] + df[\'calc\'];         \n            df[\'Sq_Cond_Calc_Dif\'] = (df[\'cond\'] - df[\'calc\']) ** 2; \n            \n        self.op_cols = df.columns;\n        return df;\n    \n    def get_feature_names_in(self, X,y):\n        return X.columns;\n    \n    def get_feature_names_out(self, X, y):\n        return self.op_cols;\n        \n# Scaling the data if needed:-   \nclass DataScaler(BaseEstimator, TransformerMixin):\n    "Scales the data columns based on the method specified";\n    \n    def __init__(self): pass\n    \n    def fit(self, X, y = None, **fit_params):\n        "Calculates the metrics for scaling";\n        \n        self.scl_cols = \\\n        [col for col in X.drop(columns = [CFG.target, "Source", \'id\'], errors = \'ignore\').columns \n         if col.startswith("RF") == False];\n        \n        df      = X[self.scl_cols]; \n        self.mu = df.mean().values;\n        self.std= df.std().values;\n        self.M  = df.max().values;\n        self.m  = df.min().values;        \n        self.q1 = np.percentile(df, axis = 0, q = 25);\n        self.q3 = np.percentile(df, axis = 0, q = 75);\n        self.IQR= self.q3 - self.q1;\n        self.q2 = np.percentile(df, axis = 0, q = 50);\n        return self;\n    \n    def transform(self, X, y = None, **transform_params):\n        "Scales the data according to the method chosen";\n        \n        df = X.copy();\n        \n        if CFG.scale_req == "Y" and CFG.scl_method == "Robust":\n            df[self.scl_cols] = (df[self.scl_cols].values - self.q2)/ self.IQR;\n            \n        elif CFG.scale_req == "Y" and CFG.scl_method == "Z":       \n            df[self.scl_cols] = (df[self.scl_cols].values - self.mu)/ self.IQR;\n        \n        elif CFG.scale_req == "Y" and CFG.scl_method == "MinMax":\n            df[self.scl_cols] = (df[self.scl_cols].values - self.m)/ (self.M - self.m);\n            \n        else:\n            PrintColor(f"--> Scaling is not needed", color = Fore.RED);\n        \n        self.op_cols = df.columns;\n        return df;\n        \n    def get_feature_names_in(self, X,y):\n        return X.columns;\n    \n    def get_feature_names_out(self, X, y):\n        return self.op_cols;        \n\nprint();\ncollect();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Implementing the transform pipeline:-\nPrintColor(f"\\n---------- Data transformation pipeline ----------\\n", color = Fore.MAGENTA);\n\nPrintColor(f"--> Shape before transform (train, test, original) = {train.shape} {test.shape} {original.shape}");\n\nif CFG.conjoin_orig_data == "Y":\n    train = pd.concat([train, original], \n                      axis = 0, ignore_index = True);\n    PrintColor(f"--> Shape after adding original data (train) = {train.shape}");\n\nelse:\n    PrintColor(f"--> Original data is not needed", color = Fore.RED);\n\ntrain  = train.drop_duplicates();\nPrintColor(f"--> Shape after removing duplicates = {train.shape}");\n\nytrain = train[CFG.target];\nxform  = Pipeline(steps = [(\'Xform\', SecFtreMaker()), (\'S\', DataScaler())]);\nxform.fit(train.drop(CFG.target, axis=1, errors = \'ignore\'), ytrain);\nXtrain = xform.transform(train.drop(CFG.target, axis=1, errors = \'ignore\'));\nXtest  = xform.transform(test);\n\nPrintColor(f"--> Shape after transform (Xtrain, test, ytrain) = {Xtrain.shape} {test.shape} {ytrain.shape}");\n\nPrintColor(f"\\n--> Data after transform\\n");\ndisplay(Xtrain.head(5).style.format(precision = 2));\n\nprint(\'\\n\\n\');\ndisplay(Xtest.head(5).style.format(precision = 2));\n\ncollect();\nprint();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ftre_imp_req.upper() == "Y":\n    fig, axes = plt.subplots(3,2, figsize = (25, 28), sharex = True,\n                             gridspec_kw = {\'wspace\': 0.2, \n                                            \'hspace\': 0.25, \n                                            \'height_ratios\': [0.6, 0.35, 0.4]\n                                           }\n                            );\n\n    # Train- feature correlations:-\n    corr_ = Xtrain.iloc[:, 1:].corr();\n    ax = axes[0,0];\n    sns.heatmap(data = corr_, cmap = "Blues", linewidth = 1.8, linecolor = \'white\',\n                annot = True, fmt = \'.2f\', annot_kws = {\'fontsize\': 7, \'fontweight\': \'bold\'},\n                mask = np.triu(np.ones_like(corr_)),\n                cbar = None, ax = ax\n               );\n    ax.set_title(f"\\nTrain Correlations\\n", **CFG.title_specs);\n\n    # Test-feature correlations:-\n    ax = axes[0,1];\n    corr_ = Xtest.iloc[:, 1:].corr();\n    sns.heatmap(data = corr_, cmap = "Blues", linewidth = 1.8, linecolor = \'white\',\n                annot = True, fmt = \'.2f\', annot_kws = {\'fontsize\': 7, \'fontweight\': \'bold\'},\n                mask = np.triu(np.ones_like(corr_)),\n                cbar = None, ax = ax\n               );\n    ax.set_title(f"\\nTest Correlations\\n", **CFG.title_specs);\n\n    # Target- feature correlations:-\n    ax = axes[1,0];\n    corr_ = pd.concat([Xtrain, ytrain], axis=1).corr()[CFG.target].\\\n    drop([CFG.target, \'id\'], axis=0);\n    corr_.plot.bar(ax = ax, color = \'tab:blue\');\n    ax.set_title(f"\\nTarget Correlations\\n", **CFG.title_specs);\n    ax.grid(**CFG.grid_specs);\n    ax.set_yticks(np.arange(-1.0, 1.01, 0.10));\n\n    # Mutual information:-\n    ax = axes[1,1];\n    pd.Series(data = mutual_info_classif(Xtrain.drop([\'id\', \'Source\'], axis=1, errors = \'ignore\'), ytrain),\n              index = Xtrain.drop([\'id\', \'Source\'], axis=1, errors = \'ignore\').columns).\\\n    plot.bar(ax = ax, color = \'tab:blue\')\n    ax.set_title(f"\\nMutual information\\n", **CFG.title_specs);\n    ax.grid(**CFG.grid_specs);\n\n    # Permutation importance:-\n    ax = axes[2,0];\n    model = LGBMClassifier(random_state = CFG.state);\n    model.fit(Xtrain.drop([\'id\', \'Source\'], axis=1, errors = \'ignore\'), ytrain);\n    pd.Series(data = np.mean(permutation_importance(model, Xtrain.drop([\'id\', \'Source\'], \n                                                                       axis=1, errors = \'ignore\'), \n                                                    ytrain,\n                                                    scoring= \'neg_log_loss\', n_repeats= 10,\n                                                    n_jobs= -1, random_state= CFG.state).\\\n                             get(\'importances\'), axis=1),\n              index = Xtrain.drop([\'id\', \'Source\'], axis=1, errors = \'ignore\').columns\n             ).plot.bar(color = \'tab:blue\', ax = ax);\n    ax.set_title(f"\\nPermutation Importance\\n", **CFG.title_specs);\n    ax.grid(**CFG.grid_specs);\n\n    # Univariate classification:-\n    ax = axes[2,1];\n    all_cols = Xtrain.drop([\'id\',\'Source\'], axis=1, errors = \'ignore\').columns;\n    scores = [];\n    for col in all_cols:\n        model = LGBMClassifier(random_state = CFG.state, max_depth = 4, \n                               learning_rate = 0.85, num_leaves = 90);\n        score = \\\n        cross_val_score(model, Xtrain[[col]], ytrain, \n                        scoring = \'roc_auc\',\n                        cv      = all_cv[\'SKF\'], \n                        n_jobs  = -1, \n                        verbose = 0\n                       );\n        scores.append(np.mean(score));\n    \n    pd.Series(scores, index = all_cols).plot.bar(color = \'tab:blue\', ax = ax);\n    ax.set_title(f"\\nUnivariate classification\\n", **CFG.title_specs);\n    ax.grid(**CFG.grid_specs);\n    ax.set_yticks(np.arange(0.0, 0.96, 0.05));\n\n    plt.tight_layout();\n    plt.show();\n\n    del corr_, score, scores, model, all_cols;\n    \nelse:\n    PrintColor(f"\\nFeature importance plots are not required\\n", color = Fore.RED);\n    \nPrintColor(f"\\nAll transformed features after data pipeline---> \\n")\ndisplay(Xtest.columns);\n\nprint();\ncollect();\n')


# <a id="8"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:130%; text-align:left;padding:2.5px; border-bottom: 7px solid #ffc266; background:#004d99" > BASELINE MODELS
# <div>
#    
# **Key notes**<br>
# 1. Complex models are unlikely to work here, so we stick to the simplest ones<br>
# 2. We will tune minimal parameters as we are unsure of the private data<br>
# 3. We will not build models for the public LB, it has 56-57 samples and is unrealible<br>

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Initializing baseline parameters:-\nMdl_Master = \\\n{\n \'XGB\'  : XGBClassifier(**{\'objective\'       : \'binary:logistic\',\n                           \'tree_method\'     : "gpu_hist" if CFG.gpu_switch == "ON" else "hist", \n                           \'eval_metric\'     : \'auc\',\n                           \'random_state\'    : CFG.state,\n                           \'colsample_bytree\': 0.95,\n                           \'learning_rate\'   : 0.095,\n                           \'min_child_weight\': 3,\n                           \'max_depth\'       : 4,\n                           \'n_estimators\'    : 1200,\n                           \'reg_lambda\'      : 4.5,\n                           \'reg_alpha\'       : 4.0,                      \n                       }\n                     ),\n    \n \'XGBR\' : XGBRegressor(**{\'colsample_bytree\' : 0.95,\n                           \'learning_rate\'   : 0.035,\n                           \'max_depth\'       : 3,\n                           \'min_child_weight\': 11,\n                           \'n_estimators\'    : 1000,\n                           \'objective\'       : \'reg:squarederror\',\n                           \'tree_method\'     : "gpu_hist" if CFG.gpu_switch == "ON" else "hist", \n                           \'eval_metric\'     : \'rmse\',\n                           \'random_state\'    : CFG.state,\n                           \'reg_lambda\'      : 0.25,\n                           \'reg_alpha\'       : 5.5,\n                       }\n                     ),\n    \n \'RFC\'  : RFC(n_estimators      = 100, \n              max_depth         = 3, \n              min_samples_leaf  = 3, \n              min_samples_split = 13, \n              random_state      = CFG.state,\n             ),\n    \n \'RFR\'  : RFR(n_estimators      = 200, \n              max_depth         = 3, \n              min_samples_leaf  = 10, \n              min_samples_split = 14, \n              random_state      = CFG.state,\n             ),\n\n \'ETR\'  : ETR(n_estimators = 180, max_depth= 3, min_samples_leaf= 4, min_samples_split = 12, random_state= CFG.state,),\n    \n \'ETC\'  : ETR(n_estimators = 140, max_depth= 3, min_samples_leaf= 4, min_samples_split = 14, random_state= CFG.state,),\n        \n \'LREG\' : LogisticRegression(max_iter     = 5000, \n                             penalty      = \'l2\', \n                             solver       = \'saga\', \n                             C            = 2.5,\n                             random_state = CFG.state,\n                             tol          = 0.001,\n                            ),\n        \n \'LGBM\' : LGBMClassifier(random_state      = CFG.state,\n                         max_depth         = 3,\n                         learning_rate     = 0.075,\n                         num_leaves        = 45,\n                         min_child_samples = 3,\n                         reg_alpha         = 3.5,\n                         reg_lambda        = 8.5,\n                         metric            = \'auc\',\n                         objective         = "binary",\n                         n_estimators      = 1000,\n                        ),\n    \n \'LGBMR\': LGBMRegressor(random_state      = CFG.state,\n                        max_depth         = 3,\n                        num_leaves        = 80,\n                        learning_rate     = 0.065,\n                        reg_alpha         = 0.5,\n                        reg_lambda        = 5.5,\n                        metric            = \'rmse\',\n                        objective         = "regression",\n                        min_child_samples = 10,\n                        n_estimators      = 1000,                       \n                        ),\n       \n \'CB\'   : CatBoostClassifier(iterations         = 1000, \n                             max_depth          = 4, \n                             eval_metric        = "AUC",\n                             random_strength    = 0.6,\n                             min_data_in_leaf   = 4,\n                             learning_rate      = 0.08,\n                             verbose            = 0,\n                             l2_leaf_reg        = 5.5,\n                             bagging_temperature= 1.6,\n                            ),\n    \n \'CBR\'  : CatBoostRegressor( iterations         = 1000, \n                             max_depth          = 3, \n                             eval_metric        = "RMSE",\n                             loss_function      = "RMSE",\n                             random_strength    = 0.5,\n                             min_data_in_leaf   = 5,\n                             learning_rate      = 0.065,\n                             verbose            = 0,\n                             l2_leaf_reg        = 1.25,\n                             bagging_temperature= 0.75,\n                             od_wait            = 7,\n                             random_seed        = CFG.state,\n                            ),\n    \n \'SVC\'  : SVC(random_state= CFG.state, C = 5.5, kernel = \'rbf\',probability= True, tol = 0.0001)\n    \n};\n\n# Shortlisted model features:-\nsel_ftre  = [\'calc\', \'Cond_Calc_Rt\', \'Cond_Calc\',\n             "Source"\n            ];\ncols      = Mdl_Master.keys();\nMdl_Preds = pd.DataFrame(index = test.id, columns = cols, \n                         data = np.zeros((len(test.id), len(cols)))\n                        );\nOOF_Preds = pd.DataFrame(index = train.id, columns = cols,\n                         data = np.zeros((len(train.id), len(cols)))\n                        );\nScores    = pd.DataFrame(columns = cols);\nFtreImp   = pd.DataFrame(columns = cols, \n                         index   = [col for col in sel_ftre if col not in [\'Source\']]);\ncv        = all_cv[\'RSKF\'];\n\nprint();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Making the partial dependence plot:-\ndef MakePrtlDepPlot(model, method, ftre, X):\n    "Makes the partial dependence plot if necessary";\n    \n    fig, axes = plt.subplots(1, len(ftre), figsize = (len(ftre) * 6, 3.0), sharey = True, \n                             gridspec_kw= {\'wspace\': 0.15, \'hspace\': 0.25});\n    plt.suptitle(f\'\\n{method}- partial dependence\\n\', y= 1.0, \n                 color = \'tab:blue\', fontsize = 8.5, fontweight = \'bold\');\n\n    PDD.from_estimator(model, X[ftre],ftre,\n                       pd_line_kw   = {"color": "#0047b3", \'linewidth\': 1.50},\n                       ice_lines_kw = {"color": "#ccffff"},\n                       kind         = \'both\',\n                       ax           = axes.ravel()[: len(ftre)],\n                       random_state = CFG.state\n                      );\n\n    for i, ax in enumerate(axes.ravel()[: len(ftre)]):\n        ax.set(ylabel = \'\', xlabel = ftre[i], title = \'\');\n        ax.grid(**CFG.grid_specs);\n\n    plt.tight_layout(h_pad=0.3, w_pad=0.5);\n    plt.show();   \n    collect();\n    \nprint();\ncollect();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Training the ML models:-\ndef TrainMdl(method:str):\n    \n    global Mdl_Master, Mdl_Preds, OOF_Preds, cv, Scores, FtreImp, Xtrain, ytrain, Xtest, sel_ftre; \n    \n    model  = Mdl_Master.get(method); \n    X,y,Xt = Xtrain[sel_ftre], ytrain, Xtest[sel_ftre]; \n    ftre   = X.drop([\'id\', \'Source\', CFG.target, \'Label\'], axis=1, errors = \'ignore\').columns;\n \n    # Initializing I-O for the given seed:-        \n    scores     = [];\n    oof_preds  = pd.DataFrame();\n    test_preds = 0;\n    ftre_imp   = 0;\n    \n    PrintColor(f"--------------------- {method.upper()} model ---------------------");\n    \n    for fold_nb, (train_idx, dev_idx) in enumerate(cv.split(X,y)): \n\n        Xtr  = X.iloc[train_idx].drop(columns = [\'id\',"Source", "Label"], errors = \'ignore\');   \n        Xdev = X.iloc[dev_idx].loc[X.Source == "Competition"].\\\n        drop(columns = [\'id\',"Source", "Label"], errors = \'ignore\'); \n        ytr  = y.loc[y.index.isin(Xtr.index)];\n        ydev = y.loc[y.index.isin(Xdev.index)];\n\n        if method.upper() in [\'XGB\', \'LGBM\', \'CB\', \'CBC\',\'XGBR\', \'LGBMR\', \'CBR\']:\n            model.fit(Xtr, ytr, eval_set = [(Xdev, ydev)], verbose = 0,\n                      early_stopping_rounds = CFG.nbrnd_erly_stp); \n        else:\n            model.fit(Xtr, ytr);\n                \n        # Collecting predictions and scores:-\n        if method in [\'XGB\', \'LGBM\', \'CB\', \'CBC\',\'RFC\', \'GBC\', \'LREG\', \'SVC\']:\n            dev_preds = np.clip(model.predict_proba(Xdev)[:,1], \n                                a_max = 1.0, a_min = 0.0); \n            t_preds   = np.clip(model.predict_proba(Xt.drop(columns = [\'id\', "Source", "Label"], \n                                              errors = \'ignore\'))[:,1],\n                                a_max = 1.0, a_min = 0.0);\n            \n        else:\n            dev_preds = np.clip(model.predict(Xdev), \n                                a_max = 1.0, a_min = 0.0); \n            t_preds   = np.clip(model.predict(Xt.drop(columns = [\'id\', "Source", "Label"], \n                                              errors = \'ignore\')),\n                                a_max = 1.0, a_min = 0.0);\n            \n        score = roc_auc_score(ydev, dev_preds);       \n        Scores.loc[fold_nb, method] = np.round(score,6);\n        scores.append(score);\n        oof_preds = pd.concat([oof_preds,\n                               pd.DataFrame(index   = Xdev.index, \n                                            data    = dev_preds,\n                                            columns = [f\'{method}\'])\n                              ],axis=0, ignore_index= False\n                             );  \n        test_preds = test_preds + t_preds/ (CFG.n_splits * CFG.n_repeats);\n\n        if method not in [\'LASSO\', \'RIDGE\', \'LREG\', \'SVC\', \'SVR\']:\n            ftre_imp += model.feature_importances_ / (CFG.n_splits * CFG.n_repeats);\n             \n    #  Collating results:-\n    mean_score = np.mean(scores);\n    print(Style.BRIGHT + Fore.BLUE + f"Mean CV score = "+ f"{\' \'* 2}" + Fore.YELLOW + Style.BRIGHT + f"{mean_score:.5f}");\n        \n    oof_preds              = pd.DataFrame(oof_preds.groupby(level = 0)[f\'{method}\'].mean());\n    oof_preds.columns      = [f\'{method}\'];\n    OOF_Preds[f\'{method}\'] = OOF_Preds[f\'{method}\'].values.flatten() + oof_preds.values.flatten();\n    Mdl_Preds[f\'{method}\'] = Mdl_Preds[f\'{method}\'].values.flatten() + test_preds ;   \n    FtreImp[method]        = ftre_imp;\n    \n    # Plotting the partial dependence plot:- \n    if CFG.prtldepplot_req == "Y": MakePrtlDepPlot(model, method,ftre, X);\n        \n    collect();\n    print();\n    \ncollect();\nprint();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nPrintColor(f"Training model suite:-");\ndisplay(cols);\nprint(\'\\n\\n\');\n\n# Implementing the training functions:-\nif CFG.ML.upper() == "Y":\n    for method in tqdm(cols, "ML training --- "): \n        TrainMdl(method = method);   \n    \nif CFG.ML.upper() != "Y": \n    PrintColor(f"\\nML models are not required\\n", color = Fore.RED);\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ML == "Y":\n    fig, axes = plt.subplots(1,2, figsize = (25,6));\n    \n    # Plotting the mean CV scores on a scattergram:-\n    ax = axes[0];\n    sns.scatterplot(x = Scores.mean(), y = Scores.columns, color = \'blue\',\n                    markers = True, s = 360, marker = \'o\',\n                    ax = ax);\n    ax.set_title(f"\\nMean CV scores across all ML models trained\\n", **CFG.title_specs);\n    ax.grid(**CFG.grid_specs);\n\n    ax = axes[1];\n    sns.violinplot(Scores, palette = \'pastel\', linewidth= 1.75, inner = \'point\', \n                   saturation = 0.999, width = 0.4, \n                   ax = ax\n                  );\n    ax.set_title(f"\\nMetric distribution across folds\\n", **CFG.title_specs);\n    ax.grid(**CFG.grid_specs);\n    ax.set_yticks(np.arange(0.50, 1.01, 0.02));\n    \n    plt.tight_layout();\n    plt.show();\n    print();\n    \n    # Plotting feature importance:-    \n    n_cols    = int(np.ceil(len(cols)/3));\n    fig, axes = plt.subplots(3, n_cols, figsize = (20, len(cols) * 0.75), sharey = True,\n                             gridspec_kw= {\'wspace\': 0.2, \'hspace\': 0.35}\n                            );\n    plt.suptitle(f"\\nFeature importance across all models", **CFG.title_specs);\n\n    for i, method in enumerate(FtreImp.columns):\n        ax = axes[i // n_cols, i % n_cols];\n        FtreImp[method].plot.barh(ax = ax, color = \'#0086b3\');\n        ax.set_title(f"{method}", **CFG.title_specs);\n        ax.grid(**CFG.grid_specs);\n\n    plt.tight_layout();\n    plt.show();\n    \n    PrintColor(f"\\n\\nPredictions and OOF results after training\\n");\n    display(OOF_Preds.head(5).style.format(precision = 2));\n    print();\n    display(Mdl_Preds.head(5).style.format(precision = 2));\n\nelse:\n    PrintColor(f"\\nPlots are not required as models are not trained\\n", \n               color = Fore.RED);\n\nprint();\ncollect();\n')


# <a id="9"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:130%; text-align:left;padding:2.5px; border-bottom: 7px solid #ffc266; background:#004d99" > ENSEMBLE
# <div>

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ML == "Y":\n    sub_fl = pd.read_csv(CFG.path + f"sample_submission.csv");\n    sub_fl[CFG.target] = \\\n    np.clip((Mdl_Preds[\'CB\']  * 0.00 + Mdl_Preds[\'ETC\'] * 1.00  + Mdl_Preds[\'ETR\']  * 0.00  + \\\n             Mdl_Preds[\'RFC\'] * 0.00 + Mdl_Preds[\'LGBM\']* 0.00  + Mdl_Preds[\'XGB\']  * 0.00 + \\\n             Mdl_Preds[\'XGBR\']* 0.00 + Mdl_Preds[\'RFR\'] * 0.00  + Mdl_Preds[\'LGBMR\']* 0.00 + \\\n             Mdl_Preds[\'LREG\']* 0.00 + Mdl_Preds[\'SVC\'] * 0.00  + Mdl_Preds[\'CBR\']  * 0.00\n            ).values, \n            a_min = 0.0001, a_max = 0.9999\n           );\n    \n    sub_fl.to_csv(f"EnsSub_{CFG.version_nb}.csv", index = None);\n    display(sub_fl.head(5).style.format(precision = 3));\n    \n    Mdl_Preds.to_csv(f"Mdl_Preds_{CFG.version_nb}.csv");\n    OOF_Preds.to_csv(f"OOF_Preds_{CFG.version_nb}.csv");\n    \n    PrintColor(f"\\nMean scores across methods\\n");\n    display(pd.concat([Scores.mean(), Scores.std()], axis=1).\\\n            rename({0:\'Mean_CV\', 1: \'Std_CV\'}, axis=1).\\\n            sort_values([\'Mean_CV\', \'Std_CV\'], ascending = [False, True]).\\\n            style.bar(color = \'#b3ecff\').\\\n            format(formatter = \'{:.4%}\')\n           );\n\nelse:\n    PrintColor(f"\\nNo need to save anything as we are not training any models\\n", \n               color = Fore.RED);\n    \ncollect();\nprint();\n')


# <a id="10"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:130%; text-align:left;padding:2.5px; border-bottom: 7px solid #ffc266; background:#004d99" > KEY INFERENCES
# <div>

# 
# <div class= "alert alert-block alert-info" align = "left" style = "color: #004d99;background-color: #f2f2f2; font-size: 110%; font-family: Cambria; font-weight: bold">
# 1. Cond and Calc are magic features- using them well could add value to the CV score<br>
# 2. Model tuning is almost futile here, a small level of hand tuning is sufficient<br>
# 3. It is advisable to make an ensemble carefully, as such smaller datasets could result in overfitting risk. The CV scores also have lots of dispersion, an added risk of unreliability<br>
# 4. We do have some hidden leaks that could be exploited as part of post-processing steps<br>
# 5. We need to drop needless features. Less is more for this assignment<br>
# 6. I have not yet explored calibrated classifiers as yet, but this could be a future activity in the days to come. I am not sure if this will add value as the metric is rank-based though.<br>
# 7. Expect a huge churn in the leaderboard at the end of the assignment. The public leaderboard is not reliable at all. 
# </div>

# <div class= "alert alert-block alert-info" align = "center" style = "color: #ffc266;background-color: #004d99; font-size: 140%; font-family: Cambria; font-weight: bold">
# Wishing you all the best for the challenge!
# </div>
