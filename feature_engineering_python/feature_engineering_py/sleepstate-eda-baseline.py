#!/usr/bin/env python
# coding: utf-8

# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #003380; border: 10px solid #80ffff"> TABLE OF CONTENTS<br><div>  
# 
# * [IMPORTS](#1)
# * [INTRODUCTION](#2)
# * [PREPROCESSING](#3)
# * [EDA AND VISUALS](#4)    
# * [DATA TRANSFORMS](#5)
# * [ML MODELS](#6)
#     * [MODEL INITIATION](#6.1)
#     * [I-O](#6.2)
#     * [TRAINING](#6.3)
# * [SUBMISSION](#7)
# * [NEXT STEPS AND OUTRO](#8)
#     

# <a id="1"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #003380; border-bottom: 10px solid #80ffff"> PACKAGE IMPORTS<br><div> 

# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# General library imports:-\nfrom IPython.display import display_html, clear_output, Markdown;\nfrom gc import collect;\nfrom itertools import groupby as grp_by;\n\nfrom copy import deepcopy;\nimport pandas as pd;\nimport polars as pl;\nimport polars.selectors as cs;\nimport pandas.api.types;\nfrom typing import Dict, List, Tuple;\nimport numpy as np;\nfrom datetime import datetime;\n\nfrom pprint import pprint;\nimport os;\nfrom functools import partial;\nfrom collections import Counter;\nfrom itertools import product;\nfrom colorama import Fore, Style, init;\nfrom warnings import filterwarnings;\nfilterwarnings('ignore');\n\nfrom tqdm.notebook import tqdm;\nimport seaborn as sns;\n\nimport matplotlib.pyplot as plt;\n%matplotlib inline\n\nclear_output();\nprint();\n")


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Importing model and pipeline specifics:-\n\n# Data engineering and pipeline development:-\nfrom sklearn.model_selection import (RepeatedStratifiedKFold as RSKF, \n                                     StratifiedKFold as SKF,\n                                     KFold, \n                                     RepeatedKFold as RKF, \n                                     cross_val_score\n                                    );\n\nfrom sklearn.pipeline import Pipeline, make_pipeline;\nfrom sklearn.base import BaseEstimator, TransformerMixin;\nfrom sklearn.compose import ColumnTransformer;\nfrom sklearn.metrics import average_precision_score as aps;\nfrom sklearn.calibration import CalibrationDisplay as Clb;\n\n# Model development:-\nfrom xgboost import XGBClassifier;\nfrom lightgbm import LGBMClassifier, log_evaluation, early_stopping;\nfrom catboost import CatBoostClassifier;\nfrom sklearn.ensemble import (RandomForestClassifier as RFC,\n                              ExtraTreesClassifier as ETC,\n                              GradientBoostingClassifier as GBC,\n                              HistGradientBoostingClassifier as HGBC,\n                             );\n\n# Ensemble and tuning:-\nimport optuna;\nfrom optuna import Trial, trial, create_study;\nfrom optuna.samplers import TPESampler, CmaEsSampler;\noptuna.logging.set_verbosity = optuna.logging.ERROR;\n\nclear_output();\nprint();\ncollect();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Defining global configurations and functions:-\n\n# Setting rc parameters in seaborn for plots and graphs- \n# Reference - https://matplotlib.org/stable/tutorials/introductory/customizing.html:-\n# To alter this, refer to matplotlib.rcParams.keys()\n\nsns.set({"axes.facecolor"       : "#ffffff",\n         "figure.facecolor"     : "#ffffff",\n         "axes.edgecolor"       : "#000000",\n         "grid.color"           : "#ffffff",\n         "font.family"          : [\'Cambria\'],\n         "axes.labelcolor"      : "#000000",\n         "xtick.color"          : "#000000",\n         "ytick.color"          : "#000000",\n         "grid.linewidth"       : 0.75,  \n         "grid.linestyle"       : "--",\n         "axes.titlecolor"      : \'#0099e6\',\n         \'axes.titlesize\'       : 8.5,\n         \'axes.labelweight\'     : "bold",\n         \'legend.fontsize\'      : 7.0,\n         \'legend.title_fontsize\': 7.0,\n         \'font.size\'            : 7.5,\n         \'xtick.labelsize\'      : 7.5,\n         \'ytick.labelsize\'      : 7.5,        \n        });\n\n# Color printing    \ndef PrintColor(text:str, color = Fore.BLUE, style = Style.BRIGHT):\n    "Prints color outputs using colorama using a text F-string";\n    print(style + color + text + Style.RESET_ALL); \n\n# Making sklearn pipeline outputs as dataframe:-\nfrom sklearn import set_config; \nset_config(transform_output = "pandas");\npd.set_option(\'display.max_columns\', 50);\npd.set_option(\'display.max_rows\', 50);\n\n# Setting global configurations for polars:-\npl.Config.activate_decimals(True).set_tbl_hide_column_data_types(True);\npl.Config(**dict(tbl_formatting = \'ASCII_FULL_CONDENSED\',\n                 tbl_hide_column_data_types = True,\n                 tbl_hide_dataframe_shape = True,\n                 fmt_float = "mixed",\n                 tbl_cell_alignment = \'CENTER\',\n                 tbl_hide_dtype_separator = True,\n                 tbl_cols = 100,\n                 tbl_rows = 100,\n                 fmt_str_lengths = 100,\n                )\n         );\n\nPrintColor(f"\\n---> Defining the global configuration for polars for all further steps\\n");\npprint(pl.Config.state(), indent = 10, width = 50);\n\nprint();\ncollect();\n')


# <a id="2"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #003380; border-bottom: 10px solid #80ffff"> INTRODUCTION<br><div> 

# | Version<br>Number | Version Details | Best CV score| Single/ Ensemble|LB score|
# | :-: | --- | :-: | :-: |:-:|
# | **V1** |* Used Dr. Carl's **binary classifier** dataset<br> * EDA, plots and secondary features <br>* No scaling <br>* Curated secondary date and rolling features from public work||Ensemble <br> Optuna ||
# | **V2** |* Used Dr. Carl's **multiclass classifier** dataset<br>* No scaling <br>* Curated secondary date and rolling features from public work <br> * Used Random Forest without CV|| ||
# | **V3** |* Used Dr. Carl's **multiclass classifier** dataset<br>* No scaling <br>* Curated secondary date and rolling features from public work <br> * Used Catboost classifier without CV|| ||

# <a id="2.1"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > CONFIGURATION PARAMETERS<br><div> 
# 
# 
# | Parameter         | Description                                             | Possible value choices|
# | ---               | ---                                                     | :-:                   |
# |  version_nb       | Version Number                                          | integer               |
# |  gpu_switch       | GPU switch                                              | ON/OFF                |
# |  state            | Random state for most purposes                          | integer               |
# |  target           | Target column name                                      | awake                 |
# |  path             | Path for input data files                               |                       |
# |  alt_path         | Path for alternative data file                           |                       |
# |  dtl_preproc_req  | Proprocessing required                                  | Y/N                   |    
# |  ftre_plots_req   | Feature plots required                                  | Y/N                   |
# |  sec_ftre_req     | Secondary features required                             | Y/N                   |
# |  scale_req        | Scaling required                                        | Y/N                   |
# |  scl_method       | Scaling method                                          | Z/ Robust/ MinMax     |
# |  enc_method       | Encoding method                                         | -                     |
# |  roll_nper        | Periods for roll-windows                                | integer               |
# |  pstprcs_oof      | Post-process OOF after model training                   | Y/N                   |
# |  pstprcs_train    | Post-process OOF during model training for dev-set      | Y/N                   |
# |  ML               | Machine Learning Models                                 | Y/N                   |
# |  n_splits         | Number of CV splits                                     | integer               |
# |  n_repeats        | Number of CV repeats                                    | integer               |
# |  nbrnd_erly_stp   | Number of early stopping rounds                         | integer               |
# |  mdl_cv_mthd      | Model CV method name                                    | RKF/ RSKF/ SKF/ KFold |

# <a id="2.2"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > ASSIGNMENT SPECIFICS<br><div> 
#     
# **In this assignment, we entail the below-**<br>
# 1. This is a time series classifier for sleep state detection<br>
# 2. We are provided acceleromater data for the train set and are supposed to make predictions for the sleep state in the test set<br>
# 3. The target is converted to a binary state in the alternative dataset<br>
# 4. Evaluation metric is derived from the adjutant notebook and is **average precision** over timestamp error tolerance thresholds, averaged over event classes.<br>

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Defining a Configuration class:-\nclass CFG:\n    "This is a Configuration class for parameters and CV strategy for tuning and training";\n    \n    # Data preparation:-   \n    version_nb         = 2;\n    \n    # Used during code proto-typing and syntax testing:-    \n    test_req           = "N";\n    n_samples          = 1000;\n    \n    gpu_switch         = "OFF"; \n    state              = 42;\n    target             = "awake";\n    path               = f"/kaggle/input/child-mind-institute-detect-sleep-states/";\n    alt_path           = f"/kaggle/input/zzzs-lightweight-training-dataset-target/Zzzs_train_multi.parquet";\n    \n    dtl_preproc_req    = "Y";\n    ftre_plots_req     = "Y";\n    \n    # Data transforms and scaling:-    \n    sec_ftre_req       = "Y";\n    scale_req          = "N";\n    # NOTE---Keep a value here even if scale_req = N, this is used for linear models:-\n    scl_method         = "Z"; \n    enc_method         = \'Label\';\n    roll_nper          = 50;\n    \n    # Model Training:- \n    pstprcs_oof        = "N";\n    pstprcs_train      = "N";\n    ML                 = "Y";\n    methods            = [\'CBC\', \'LGBMC\', \'XGBC\', \'RFC\', \'HGBC\'];\n    method             = "RFC";\n    n_splits           = 5 ;\n    n_repeats          = 1 ;\n    nbrnd_erly_stp     = 200 ;\n    mdlcv_mthd         = \'RSKF\';\n    \n    # Ensemble:-    \n    ensemble_req       = "Y";\n    enscv_mthd         = "RSKF";\n    metric_obj         = \'maximize\';\n    ntrials            = 10 if test_req == "Y" else 75;\n    \n    # Global variables for plotting:-\n    grid_specs = {\'visible\': True, \'which\': \'both\', \'linestyle\': \'--\', \n                           \'color\': \'lightgrey\', \'linewidth\': 0.75};\n    title_specs = {\'fontsize\': 9, \'fontweight\': \'bold\', \'color\': \'tab:blue\'};\n\nprint();\nPrintColor(f"--> Configuration done!\\n");\ncollect();\n')


# <a id="2.3"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > SCORING METRIC AND ADJUTANT FUNCTIONS<br><div> 

# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndef ScoreMetric(ytrue: np.array, ypred: np.array) -> np.float32:\n    """\n    This function defines the competition evaluation metric\n    This is a temporary metric till I incorporate the competition evaluation metric into the notebook\n    """;\n    return aps(ytrue, ypred);\n\ndef ScoreLGBM(ytrue: np.array, ypred: np.array):\n    """\n    This function defines the interim metric as per LGBM requirements and classifier type\n    """;\n    return (\'AvgPrecision\', aps(ytrue, ypred), True);\n\ndef PostProcessPred(ypred: np.array, pp_req : str = "N") -> np.array:\n    "This is an optional post-processing function required in certain assignments";\n    return ypred;\n\n# Defining commonly used CV strategies for later usage:-\nall_cv= {\'KF\'  : KFold(n_splits= CFG.n_splits, shuffle = True, random_state= CFG.state),\n         \'RKF\' : RKF(n_splits= CFG.n_splits, n_repeats = CFG.n_repeats, random_state= CFG.state),\n         \'RSKF\': RSKF(n_splits= CFG.n_splits, n_repeats = CFG.n_repeats, random_state= CFG.state),\n         \'SKF\' : SKF(n_splits= CFG.n_splits, shuffle = True, random_state= CFG.state)\n        };\n\ncollect();\nprint();\n')


# <a id="3"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #003380; border-bottom: 10px solid #80ffff"> PREPROCESSING<br><div> 

# In[ ]:


get_ipython().run_line_magic('time', '')

class Preprocessor():
    """
    This class aims to do the below-
    1. Read the datasets and change the datatype of the target column in the alternative data
    3. Check information and description
    4. Check unique values
    5. Collate starting features 
    6. Curate secondary features based on public work
    """;
    
    def __init__(self):
        """
        We do the below here-
        1. Read in the train data from the alternative data source
        2. Read in the test and submission file from the competition source
        3. We sample the training data if necessary upon testing the code semantics
        """;
           
        self.test     = \
        pl.scan_parquet(CFG.path + f"test_series.parquet").\
        with_columns(pl.col('timestamp').\
                     str.strptime(pl.Datetime)
                    );
        self.sub_fl   = pl.read_csv(CFG.path + f"sample_submission.csv");
        
        self.target       = CFG.target;
        self.test_req     = CFG.test_req;
        self.sec_ftre_req = CFG.sec_ftre_req;
        self.nper         = CFG.roll_nper;
       
        self.train    = \
        pl.scan_parquet(CFG.alt_path).\
        with_columns(pl.col(CFG.target).cast(pl.Int8), pl.col('timestamp').\
                     str.strptime(pl.Datetime)
                    );
        
        if self.test_req == "Y":
            PrintColor(f"---> We are testing the code with {CFG.n_samples} data samples per series and target", 
                       color = Fore.RED);
            self.train = \
            self.train.groupby(['series_id', self.target]).\
            agg(pl.all().sample(CFG.n_samples)).\
            explode(['step', 'timestamp', 'anglez', 'enmo']);
           
        else:
            PrintColor(f"---> We are not testing the code- this is an actual and complete run", color = Fore.RED);           
                       
    def PreprocessDF(self):
        """
        This method ensues basic preprocessing steps for the submission file and collates the starting features
        """;
                
        PrintColor(f"\nSubmission file head", color = Fore.GREEN);
        display(self.sub_fl.head(5));   
        
        self.strt_ftre = self.train.columns;
        try:  self.strt_ftre.remove(CFG.target); 
        except: pass;
        
        PrintColor(f"Starting features");
        pprint(self.strt_ftre);
                   
        return self; 
    
    def _XformDF(self, X: pl.DataFrame):
        """
        This method prepares secondary features if requested by the user.
        We use the date time column and rolling statistics from public work
        """
        
        # Curating date features:-        
        df = \
        X.with_columns([pl.col("timestamp").dt.hour().cast(pl.Int8).alias("hour_nb"),
                        pl.col("timestamp").dt.day().cast(pl.Int8).alias("day_nb"),
                        pl.col("timestamp").dt.weekday().cast(pl.Int8).alias("weekday_nb"),
                        pl.col("timestamp").dt.week().cast(pl.Int8).alias("week_nb"),
                        pl.col("timestamp").dt.month().cast(pl.Int8).alias("month_nb"),
                        pl.col("timestamp").dt.year().cast(pl.Int16).alias("year_nb"),
                        (pl.col("anglez") * pl.col("enmo")).alias("anglez_enmo")
                       ]);
        
        # Curating rolling features and window results:-
        df = \
        df.\
        with_columns([pl.col("anglez").diff(self.nper).over("series_id").\
                      fill_null(strategy = 'backward').alias("d_anglez"),
                      pl.col("enmo").diff(self.nper).over("series_id").\
                      fill_null(strategy = 'backward').alias("d_enmo"),
                      pl.col("anglez").rolling_mean(self.nper).over("series_id").\
                      fill_null(strategy = 'backward').fill_null(strategy = 'forward').alias('ma_anglez'),
                      pl.col("enmo").rolling_mean(self.nper).over("series_id").\
                      fill_null(strategy = 'backward').fill_null(strategy = 'forward').alias('ma_enmo')
                     ]
                    ).\
        with_columns([pl.col('d_anglez').rolling_mean(self.nper).over("series_id").\
                      fill_null(strategy = 'backward').fill_null(strategy = 'forward').alias('ma_d_anglez'),
                      pl.col('d_enmo').rolling_mean(self.nper).over("series_id").\
                      fill_null(strategy = 'backward').fill_null(strategy = 'forward').alias('ma_d_enmo')
                     ]
                    );
            
        return df;
    
    def CurateSecFtre(self):
        "This method curates the dataframe with secondary features and returns a modified dataframe";
        
        if self.sec_ftre_req == "Y":
            PrintColor(f"Secondary features are required");
            return self._XformDF(self.train), self._XformDF(self.test);
        else:
            PrintColor(f"Secondary features are not required", color = Fore.RED);
            return self.train, self.test;
               
collect();
print();


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Preprocessinng the data:-\nPrintColor(f"\\n{\'=\'*30} Data Pre-processing {\'=\'*30}\\n", Fore.MAGENTA);\npp = Preprocessor();\npp.PreprocessDF();\n\ncollect();\nprint();\n')


# <a id="4"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #003380; border-bottom: 10px solid #80ffff"> EDA AND VISUALS<br><div>  

# <div class="alert alert-block alert-info" style = "font-family: Cambria Math;font-size: 115%; color: black; background-color: #e6f9ff; border: dashed black 1.0px; padding: 3.5px" >
# <b>Note</b> <br>
# This section is work in progress. I shall update it in a couple of days with relevant insights
# </div>

# <a id="5"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #003380; border-bottom: 10px solid #80ffff"> DATA TRANSFORMS<br><div>

# <div class="alert alert-block alert-info" style = "font-family: Cambria Math;font-size: 115%; color: black; background-color: #e6f9ff; border: dashed black 1.0px; padding: 3.5px" >
# <b>Note</b> <br>
# We have ensued static transforms in the previous section. We shall implement this herewith
# </div>

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nif CFG.dtl_preproc_req == "Y":\n    # Implementing secondary features:-\n    PrintColor(f"\\n{\'=\'*30} Data Transformation {\'=\'*30}\\n", Fore.MAGENTA);\n\n    train, test = pp.CurateSecFtre();\n    print();\n    PrintColor(f"\\n--> Train data query plan <---\\n");\n    display(train.show_graph(figsize = (14, 8)));\n\n    # Creating the train-test datasets:-\n    PrintColor(f"\\n\\n--> Collating the train-test datasets <---");\n    train, test = train.collect(), test.collect();\n\n    PrintColor(f"\\n---> Train set after pre-processing\\n");\n    display(train.head(5));\n    PrintColor(f"\\n---> Test set after pre-processing\\n");\n    display(test.head(5));\n\n    PrintColor(f"\\n---> Train set description after pre-processing\\n");\n    df = train.describe(percentiles = [0.01, 0.05, 0.90, 0.95, 0.99]).to_pandas();\n    cols = df.iloc[:,0].values.tolist();\n    df = df.iloc[:, 1:].transpose();\n    df.columns = cols;\n    display(df.drop(columns = [\'count\']).style.format(precision = 2));\n    del df, cols;\n\nprint();\ncollect();\n')


# <a id="6"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #003380; border-bottom: 10px solid #80ffff"> ML MODELS<br><div>

# <div class="alert alert-block alert-info" style = "font-family: Cambria Math;font-size: 115%; color: black; background-color: #e6f9ff; border: dashed black 1.0px; padding: 3.5px" >
# <b>Strategy</b> <br>
# 1. We initialize basic model parameters for ensemble tree and other ML models to start off<br>
# 2. We build data structures to store the model predictions and OOF predictions too<br>
# 3. We train the models one by one with the chosen strategy<br>
# 4. We collate predictions and store them in the prescribed data structures<br>
# </div>

# <a id="6.1"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > MODEL INITIALIZATION<br><div> 

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Initializing model I-O:-\n\nMdl_Master = \\\n{\'CBC\': CatBoostClassifier(**{\'task_type\'           : "GPU" if CFG.gpu_switch == "ON" else "CPU",\n                              \'random_state\'        : CFG.state,\n                              \'classes_count\'       : 3,\n                              \'bagging_temperature\' : 0.425,\n                              \'colsample_bylevel\'   : 0.667,\n                              \'iterations\'          : 1000,\n                              \'learning_rate\'       : 0.05,\n                              \'od_wait\'             : 32,\n                              \'max_depth\'           : 6,\n                              \'l2_leaf_reg\'         : 0.45,\n                              \'min_data_in_leaf\'    : 300,\n                              \'random_strength\'     : 0.15, \n                              \'max_bin\'             : 200,\n                              \'verbose\'             : 0,                        \n                           }\n                         ), \n \n  \'LGBMC\': LGBMClassifier(**{\'device\'            : "gpu" if CFG.gpu_switch == "ON" else "cpu",\n                             \'objective\'         : \'multiclass\',\n                             \'metric\'            : \'none\',\n                             \'boosting_type\'     : \'gbdt\',\n                             \'random_state\'      : CFG.state,\n                             \'colsample_bytree\'  : 0.675,\n                             \'subsample\'         : 0.925,\n                             \'learning_rate\'     : 0.065,\n                             \'max_depth\'         : 7,\n                             \'n_estimators\'      : 2000,\n                             \'num_leaves\'        : 220,\n                             \'min_child_samples\' : 350,\n                             \'reg_alpha\'         : 0.01,\n                             \'reg_lambda\'        : 1.5,\n                             \'verbose\'           : -1,\n                             "is_unbalance"      : True,\n                         }\n                      ),\n\n  \'XGBC\': XGBClassifier(**{\'tree_method\'        : "gpu_hist" if CFG.gpu_switch == "ON" else "hist",\n                           \'objective\'          : \'multi:logistic\',\n                           \'random_state\'       : CFG.state,\n                           \'colsample_bytree\'   : 0.45,\n                           \'learning_rate\'      : 0.05,\n                           \'max_depth\'          : 6,\n                           \'n_estimators\'       : 1500,                         \n                           \'reg_alpha\'          : 0.01,\n                           \'reg_lambda\'         : 1.25,\n                           \'min_child_samples\'  : 350,\n                        }\n                       ),\n \n  \'HGBC\': HGBC(learning_rate    = 0.055,\n               max_iter         = 2000,\n               max_depth        = 7,\n               min_samples_leaf = 300,\n               l2_regularization= 1.25,\n               max_bins         = 200,\n               n_iter_no_change = 100,\n               random_state     = CFG.state,\n              ),\n \n  \'RFC\' : RFC(n_estimators     = 125,\n              criterion        = \'gini\',\n              max_depth        = 6,\n              min_samples_leaf = 350,\n              max_features     = "log2",\n              n_jobs           = -1,\n              random_state     = CFG.state,\n              verbose          = 0,\n             ),\n \n  "GBC" : GBC(n_estimators       = 50,\n              max_depth          = 8,\n              min_samples_leaf   = 350,\n              random_state       = CFG.state,\n             )\n};\n\nprint();\ncollect();\n')


# <a id="6.2"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > MODEL I-O INITIALIZATION<br><div> 

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ML == "Y":\n    PrintColor(f"\\n{\'=\'*30} Initializing model I-O {\'=\'*30}\\n", Fore.MAGENTA);\n\n    # Initializing output tables for the models:-\n    try: \n        methods   = CFG.methods;\n    except: \n        methods = list(Mdl_Master.keys());\n        PrintColor(f"---> Check the methods in the configuration class", Fore.RED);\n\n    # Converting the polars train set to pandas Xtrain, ytrain with selected features:-\n    Xtrain, ytrain = train.drop([CFG.target]).to_pandas().iloc[:, 3:], train[CFG.target].to_pandas();\n    Xtest = test.to_pandas().iloc[:, 3:];\n\n    # Dropping the train polars data to conserve memory:-\n    if CFG.test_req != "Y": \n        del train;\n    else: \n        PrintColor(f"---> Polars train dataframe are not dropped in code-testing", Fore.RED);\n\n    PrintColor(f"\\n---> Selected model options- ");\n    pprint(methods, depth = 1, width = 100, indent = 5);\n    \n    PrintColor(f"\\n---> Dtypes of Xtrain, ytrain = {type(Xtrain)} {type(ytrain)}");\n    \n    PrintColor(f"\\n---> Train set features -");\n    pprint(Xtrain.iloc[0:2,:].columns);\n    \n    PrintColor(f"\\n---> Test set features -");\n    pprint(Xtest.iloc[0:2,:].columns);   \n\ncollect();\nprint();\n')


# <a id="6.3"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > BASELINE MODEL TRAINING<br><div> 

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Implementing the ML models:-\nif CFG.ML == "Y":           \n    PrintColor(f"\\n{\'=\' * 20} ML training {CFG.method} {\'=\' * 20}\\n", color = Fore.MAGENTA);\n    model = Mdl_Master[CFG.method];\n    model.fit(Xtrain, ytrain);\n    \n    del Xtrain, ytrain;\n    \nelse:\n    PrintColor(f"\\nML models are not needed\\n", color = Fore.RED);\n    \nprint();\ncollect();\n')


# <a id="7"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #003380; border-bottom: 10px solid #80ffff"> SUBMISSION<br><div>

# <div class="alert alert-block alert-info" style = "font-family: Cambria Math;font-size: 115%; color: black; background-color: #e6f9ff; border: dashed black 1.0px; padding: 3.5px" >
# <b>Note</b> <br>
# We follow the submission file curation from the reference notebook.<br>
# We use pandas for the same, considering the small size of the test set.
# </div>

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ML == "Y":\n    \n    PrintColor(f"\\n{\'=\' * 20} Submission file preparation {\'=\' * 20}\\n", \n               color = Fore.MAGENTA); \n    \n    # Curating the test set scores:- \n    if isinstance(test, pd.DataFrame) == True:\n        PrintColor(f"---> Test data is a pandas dataframe");\n        sub_fl = test.copy(deep = True);\n    else:\n        PrintColor(f"---> Converting the polars test data into a pandas dataframe");\n        sub_fl = test.to_pandas();\n    \n    sub_fl[\'not_awake\'] = model.predict_proba(Xtest)[:,0];\n    sub_fl[\'awake\']     = model.predict_proba(Xtest)[:,1];\n    \n    smoothing_length = 2 * 225;\n    sub_fl["score"]  = \\\n    sub_fl["awake"].\\\n    rolling(smoothing_length, center=True).mean().\\\n    fillna(method="bfill").\\\n    fillna(method="ffill");\n\n    sub_fl["smooth"] = \\\n    sub_fl[f"not_{CFG.target}"].\\\n    rolling(smoothing_length, center= True).\\\n    mean().\\\n    fillna(method="bfill").\\\n    fillna(method="ffill");\n\n    # Re-binarizing the target:-\n    sub_fl["smooth"] = sub_fl["smooth"].round();\n\n    def get_event(df):\n        lstCV = zip(df.series_id, df.smooth);\n        lstPOI = [];\n        for (c, v), g in grp_by(lstCV, lambda cv: \n                                (cv[0], cv[1]!=0 and not pd.isnull(cv[1]))):\n            llg = sum(1 for item in g);\n            if v is False: \n                lstPOI.extend([0]*llg);\n            else: \n                lstPOI.extend([\'onset\']+(llg-2)*[0]+[\'wakeup\'] if llg > 1 else [0]);\n        return lstPOI;\n\n    sub_fl["event"] = get_event(sub_fl);\n\n    # Curating the requisite submission file:-\n    sub_fl = \\\n    sub_fl.loc[sub_fl["event"] != 0][["series_id","step","event","score"]].\\\n    copy().\\\n    reset_index(drop=True).\\\n    reset_index(names="row_id");\n    sub_fl.to_csv(f"submission.csv", index = None);\n\n    PrintColor(f"---> Final submission file");\n    display(sub_fl.head(10).style.format(precision = 2));\n    \nelse:\n    PrintColor(f"---> Final submission file is not needed", Fore.RED);\n\nprint();\ncollect();\n')


# <a id="8"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #003380; border-bottom: 10px solid #80ffff"> OUTRO<br><div>

# <div class="alert alert-block alert-info" style = "font-family: Cambria Math;font-size: 115%; color: black; background-color: #e6f9ff; border: dashed black 1.0px; padding: 3.5px" >
# <b>References</b> <br>
# 1. https://www.kaggle.com/code/carlmcbrideellis/zzzs-random-forest-model-starter <br>
# 2. https://www.kaggle.com/datasets/carlmcbrideellis/zzzs-lightweight-training-dataset-target<br>
# 3. https://www.kaggle.com/code/satheeshbhukya1/detect-sleep-states<br>
# 
# <br>
# <b>Next steps:-</b> <br>   
# 1. Incorporating the competition metric for a correct evaluation <br>
# 2. We will try other models, especially NNs <br>
# 3. Data cleaning and EDA is required to elicit trends and patterns in the existing series <br>
# 4. We will try and explore the multi-class dataset too and develop models thereby<br>
# <br>
# </div>
