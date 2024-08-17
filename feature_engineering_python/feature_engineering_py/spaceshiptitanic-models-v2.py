#!/usr/bin/env python
# coding: utf-8

# #### <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:black; font-size:150%; text-align:left;padding:3.0px; background: #cceeff; border-bottom: 8px solid #004466" > TABLE OF CONTENTS<br><div>  
# * [IMPORTS](#1)
# * [INTRODUCTION](#2)
#     * [CONFIGURATION](#2.1)
#     * [CONFIGURATION PARAMETERS](#2.2)    
#     * [DATASET COLUMNS](#2.3)
# * [PREPROCESSING](#3)
# * [MODEL TRAINING](#4)      
# * [PLANNED WAY FORWARD](#5) 

# <a id="1"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:black; font-size:120%; text-align:left;padding:3.0px; background: #cceeff; border-bottom: 8px solid #004466" > IMPORTS<br> <div> 

# In[1]:


get_ipython().run_cell_magic('time', '', '\n# Installing select libraries:-\nfrom gc import collect;\nfrom warnings import filterwarnings;\nfilterwarnings(\'ignore\');\nfrom IPython.display import display_html, clear_output;\n\n! python -m pip install --no-index \\\n--find-links=/kaggle/input/packageinstallation \\\n-r /kaggle/input/packageinstallation/requirements.txt -q;\nclear_output();\n\nimport xgboost as xgb, lightgbm as lgb, catboost as cb, sklearn as sk;\nprint(f"---> XGBoost = {xgb.__version__} | LightGBM = {lgb.__version__} | Catboost = {cb.__version__}");\nprint(f"---> Sklearn = {sk.__version__}\\n\\n");\ncollect();\n')


# In[2]:


get_ipython().run_cell_magic('time', '', "\n# General library imports:-\nfrom copy import deepcopy;\nimport pandas as pd;\nimport numpy as np;\nimport re;\nfrom scipy.stats import mode, kstest, normaltest, shapiro, anderson, jarque_bera;\nfrom collections import Counter;\nfrom itertools import product;\nfrom colorama import Fore, Style, init;\nfrom warnings import filterwarnings;\nfilterwarnings('ignore');\nimport joblib;\nimport os;\n\nfrom tqdm.notebook import tqdm;\nimport seaborn as sns;\nimport matplotlib.pyplot as plt;\nfrom matplotlib.colors import ListedColormap as LCM;\n%matplotlib inline\n\nfrom pprint import pprint;\nfrom functools import partial;\n\nprint();\ncollect();\nclear_output();\n")


# In[3]:


get_ipython().run_cell_magic('time', '', '\n# Importing model and pipeline specifics:-\nfrom category_encoders import OrdinalEncoder, OneHotEncoder;\n\n# Pipeline specifics:-\nfrom sklearn.preprocessing import (RobustScaler, \n                                   MinMaxScaler, \n                                   StandardScaler, \n                                   FunctionTransformer as FT,\n                                   PowerTransformer,\n                                  );\nfrom sklearn.impute import SimpleImputer as SI;\nfrom sklearn.model_selection import (RepeatedStratifiedKFold as RSKF, \n                                     StratifiedKFold as SKF,\n                                     KFold, \n                                     RepeatedKFold as RKF, \n                                     cross_val_score, cross_val_predict\n                                    );\nfrom sklearn.inspection import permutation_importance;\nfrom sklearn.feature_selection import mutual_info_classif, RFE;\nfrom sklearn.pipeline import Pipeline, make_pipeline;\nfrom sklearn.base import BaseEstimator, TransformerMixin;\nfrom sklearn.compose import ColumnTransformer;\n\n# ML Model training:-\nfrom sklearn.metrics import log_loss, brier_score_loss, accuracy_score, roc_curve, make_scorer;\nfrom xgboost import DMatrix, XGBClassifier as XGBC;\nfrom lightgbm import log_evaluation, early_stopping, LGBMClassifier as LGBMC;\nfrom catboost import CatBoostClassifier as CBC, Pool;\nfrom sklearn.ensemble import (HistGradientBoostingClassifier as HGBC, \n                              RandomForestClassifier as RFC,\n                              ExtraTreesClassifier as ETC,\n                              GradientBoostingClassifier as GBC,\n                             );\nfrom sklearn.linear_model import LogisticRegression as LC;\nfrom sklearn.neighbors import KNeighborsClassifier as KNNC;\n\n# Calibration:-\nfrom sklearn.isotonic import IsotonicRegression as ITRC;\nfrom sklearn.calibration import CalibrationDisplay as Clb;\n\n# Ensemble and tuning:-\nimport optuna;\nfrom optuna import Trial, trial, create_study;\nfrom optuna.pruners import HyperbandPruner;\nfrom optuna.samplers import TPESampler, CmaEsSampler;\noptuna.logging.set_verbosity = optuna.logging.CRITICAL;\n\nclear_output();\nprint();\ncollect();\n')


# In[4]:


get_ipython().run_cell_magic('time', '', '\n# Setting rc parameters in seaborn for plots and graphs- \n# Reference - https://matplotlib.org/stable/tutorials/introductory/customizing.html:-\n# To alter this, refer to matplotlib.rcParams.keys()\n\nsns.set({"axes.facecolor"       : "#ffffff",\n         "figure.facecolor"     : "#ffffff",\n         "axes.edgecolor"       : "#000000",\n         "grid.color"           : "#ffffff",\n         "font.family"          : [\'Cambria\'],\n         "axes.labelcolor"      : "#000000",\n         "xtick.color"          : "#000000",\n         "ytick.color"          : "#000000",\n         "grid.linewidth"       : 0.75,  \n         "grid.linestyle"       : "--",\n         "axes.titlecolor"      : \'#0099e6\',\n         \'axes.titlesize\'       : 8.5,\n         \'axes.labelweight\'     : "bold",\n         \'legend.fontsize\'      : 7.0,\n         \'legend.title_fontsize\': 7.0,\n         \'font.size\'            : 7.5,\n         \'xtick.labelsize\'      : 7.5,\n         \'ytick.labelsize\'      : 7.5,        \n        });\n\n# Color printing    \ndef PrintColor(text:str, color = Fore.BLUE, style = Style.BRIGHT):\n    "Prints color outputs using colorama using a text F-string";\n    print(style + color + text + Style.RESET_ALL); \n\n# Making sklearn pipeline outputs as dataframe:-\nfrom sklearn import set_config; \nset_config(transform_output = "pandas");\npd.set_option(\'display.max_columns\', 50);\npd.set_option(\'display.max_rows\', 50);\n\nprint();\ncollect();\n')


# <a id="2"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:black; font-size:120%; text-align:left;padding:3.0px; background: #cceeff; border-bottom: 8px solid #004466" > INTRODUCTION<br><div> 

# | Version<br>Number | Version Details | CV score| Single/ Ensemble|Public LB Score|
# | :-: | --- | :-: | :-: |:-:|
# | **V1** |* Used the FE component from Arun Klein's notebook <br> * Attached my FE dataset <br> * Used my training pipeline <br> * No Isotonic regression calibration <br> * Optuna ensemble, 10x1 ML model CV <br> * Blended with good public work |0.82503|Optuna <br> Ensemble|0.82066|
# | **V2** |* Similar to V1 <br> * Used pseudo labels||Optuna <br> Ensemble||

# <a id="2.1"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > CONFIGURATION<br><div> 

# In[5]:


get_ipython().run_cell_magic('time', '', '\n# Configuration class:-\nclass CFG:\n    """\n    Configuration class for parameters and CV strategy for tuning and training\n    Some parameters may be unused here as this is a general configuration class\n    """;\n    \n    # Data preparation:-   \n    version_nb         = 2;\n    test_req           = "N";\n    test_sample_frac   = 0.025;\n    gpu_switch         = "OFF"; \n    state              = 42;\n    target             = \'Transported\';\n    data_path          = f"/kaggle/input/spacetitanicfe";\n    path               = f"/kaggle/input/spaceship-titanic";\n    ftre_imp_req       = "Y";\n    scl_method         = "Robust";\n        \n    # Model Training:- \n    pstprcs_oof        = "N";\n    pstprcs_train      = "N";\n    pstprcs_test       = "N";\n    ML                 = "Y";\n    ist_reg_req        = "N";\n    \n    pseudo_lbl_req     = "Y";\n    pseudolbl_up       = 0.975;\n    pseudolbl_low      = 0.025;\n    n_splits           = 3 if test_req == "Y" else 10;\n    n_repeats          = 1 ;\n    nbrnd_erly_stp     = 100;\n    mdlcv_mthd         = \'RSKF\';\n    \n    # Ensemble:-    \n    ensemble_req       = "Y";\n    hill_climb_req     = "N";\n    optuna_req         = "Y";\n    LAD_req            = "N";\n    enscv_mthd         = "RSKF";\n    metric_obj         = \'maximize\';\n    ntrials            = 10 if test_req == "Y" else 200;\n    \n    # Global variables for plotting:-\n    grid_specs = {\'visible\': True, \'which\': \'both\', \'linestyle\': \'--\', \n                           \'color\': \'lightgrey\', \'linewidth\': 0.75};\n    title_specs = {\'fontsize\': 9, \'fontweight\': \'bold\', \'color\': \'tab:blue\'};\n\n\nprint();\nPrintColor(f"--> Configuration done!\\n");\ncollect();\n')


# <a id="2.2"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > CONFIGURATION PARAMETERS<br><div> 
# 
# 
# | Parameter         | Description                                             | Possible value choices|
# | ---               | ---                                                     | :-:                   |
# |  version_nb       | Version Number                                          | integer               |
# |  test_req         | Are we testing syntax here?                             | Y/N                   |  
# |  test_sample_frac | Sample size for syntax test                             | float(0-1)/ int       |     
# |  gpu_switch       | GPU switch                                              | ON/OFF                |
# |  state            | Random state for most purposes                          | integer               |
# |  target           | Target column name                                      | yield                 |
# |  path             | Path for input data files                               |                       |  
# |  ftre_imp_req     | Feature importance required                             | Y/N                   |      
# |  pstprcs_oof      | Post-process OOF after model training                   | Y/N                   |
# |  pstprcs_train    | Post-process OOF during model training for dev-set      | Y/N                   |
# |  pstprcs_test     | Post-process test after training                        | Y/N                   |
# |  ML               | Machine Learning Models                                 | Y/N                   |
# |  ist_reg_req      | Isotonic Regression required                            | Y/N                   |    
# |  n_splits         | Number of CV splits                                     | integer               |
# |  n_repeats        | Number of CV repeats                                    | integer               |
# |  nbrnd_erly_stp   | Number of early stopping rounds                         | integer               |
# |  mdl_cv_mthd      | Model CV method name                                    | RKF/ RSKF/ SKF/ KFold |
# |  ensemble_req     | Ensemble required                                       | Y/N                   | 
# |  hill_climb_req   | Ensemble hill climb required                            | Y/N                   |  
# |  optuna_req       | Ensemble Optuna required                                | Y/N                   | 
# |  LAD_req          | Ensemble LAD required                                   | Y/N                   | 
# |  enscv_mthd       | Ensemble CV method                                      | RKF/ RSKF/ SKF/ KFold | 
# |  metric_obj       | Metric objective                                        | minimize/ maximize    |
# |  ntrials          | Number of trials                                        | int                   |
#     

# <a id="2.3"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > DATASET AND COMPETITION DETAILS<br><div>
#     
# **Data columns**<br>
# This is available in the original data description as below in the Spaceship Titanic competition at the link below<br>
# https://www.kaggle.com/competitions/spaceship-titanic <br>
# 
# <br>**Competition details and notebook objectives**<br>
# 1. This is a binary classification challenge to predict the probability of transporting the targets to an alternative dimension. **Brier Score** is the metric for the challenge<br>
# 2. In this starter notebook, we start the assignment with a detailed EDA, feature plots, interaction effects, adversarial CV analysis and develop starter models to initiate the challenge. We will also incorporate other opinions and approaches as we move along the challenge.<br>
# 3. Thanks to the notebook by Arun Klein for feature engineering, link is as below- <br>
# https://www.kaggle.com/code/arunklenin/space-titanic-eda-advanced-feature-engineering
# <br>
# **Model strategy** <br>
# We start off with simple tree based ML models and an Optuna ensemble to create sample inputs for the submission. <br>
# 

# <a id="3"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:black; font-size:120%; text-align:left;padding:3.0px; background: #cceeff; border-bottom: 8px solid #004466" > PREPROCESSING<br><div> 

# In[6]:


get_ipython().run_line_magic('time', '')

class Preprocessor():
    """
    This class aims to do the below-
    1. Read the datasets
    2. Check information and description
    3. Check unique values and nulls
    4. Collate starting features 
    """;
    
    def __init__(self):
        self.train    = pd.read_parquet(os.path.join(CFG.data_path,"Train.parquet"));
        self.test     = pd.read_parquet(os.path.join(CFG.data_path ,"Test.parquet"));
        self.target   = CFG.target ;
        self.test_req = CFG.test_req;
        
        self.ytrain   = \
        pd.read_csv(os.path.join(CFG.path, "train.csv"), usecols = [self.target]).squeeze();
        self.ytrain   = pd.Series(np.uint8(~self.ytrain), name = self.target);
        
        self.sub_fl   = pd.read_csv(os.path.join(CFG.path, "sample_submission.csv"));
        
        self.train = self.train.drop(columns = [self.target], axis=1, errors = "ignore");
        PrintColor(f"Data shapes - train-test = {self.train.shape} {self.test.shape}");
        
        for tbl in [self.train, self.test]:
            tbl.columns = tbl.columns.str.replace(r"\(|\)|\s+","", regex = True);
            
        PrintColor(f"\nTrain set head", color = Fore.CYAN);
        display(self.train.head(5).style.format(precision = 3));
        PrintColor(f"\nTest set head", color = Fore.CYAN);
        display(self.test.head(5).style.format(precision = 3));
        
        self.strt_ftre = [c for c in self.test.columns if c not in ['id', "Source", "Label"]];
                              
    def _CollateInfoDesc(self):
        PrintColor(f"\n{'-'*20} Information and description {'-'*20}\n", color = Fore.MAGENTA);

        # Creating dataset information and description:
        for lbl, df in {'Train': self.train, 'Test': self.test}.items():
            PrintColor(f"\n{lbl} description\n");
            display(df.describe(percentiles= [0.05, 0.25, 0.50, 0.75, 0.9, 0.95, 0.99]).\
                    transpose().\
                    drop(columns = ['count'], errors = 'ignore').\
                    drop([CFG.target], axis=0, errors = 'ignore').\
                    style.format(formatter = '{:,.2f}').\
                    background_gradient(cmap = 'Blues')
                   );

            PrintColor(f"\n{lbl} information\n");
            display(df.info());
            collect();
        return self;
    
    @staticmethod
    def ReduceMem(df: pd.DataFrame):
        "This method reduces memory for numeric columns in the dataframe";
        
        numerics  = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = df.memory_usage().sum() / 1024**2;
        
        for col in df.columns:
            col_type = df[col].dtypes
            
            if col_type in numerics:
                c_min = df[col].min();
                c_max = df[col].max();

                if "int" in str(col_type):
                    if c_min >= np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min >= np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min >= np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min >= np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                else:
                    if c_min >= np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    if c_min >= np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)  

        end_mem = df.memory_usage().sum() / 1024**2
    
        PrintColor(f"Start - end memory:- {start_mem:5.2f} - {end_mem:5.2f} Mb");
        return df;
    
    def _CollateUnqNull(self): 
        # Dislaying the unique values across train-test-original:-
        PrintColor(f"\nUnique and null values\n");
        _ = pd.concat([self.train[self.strt_ftre].nunique(), 
                       self.test[self.strt_ftre].nunique(), 
                       self.train[self.strt_ftre].isna().sum(axis=0),
                       self.test[self.strt_ftre].isna().sum(axis=0),
                      ], 
                      axis=1);
        _.columns = ['Train_Nunq', 'Test_Nunq', 'Train_Nulls', 'Test_Nulls'];

        display(_.T.style.background_gradient(cmap = 'Blues', axis=1).\
                format(formatter = '{:,.0f}')
               );
            
        return self;
       
    def DoPreprocessing(self):
        print();
        for df in [self.train, self.test]: 
            df = self.ReduceMem(df);
        self._CollateInfoDesc();
        self._CollateUnqNull(); 
        print();
        
        return self.train, self.ytrain, self.test; 
          
collect();
print();


# In[7]:


get_ipython().run_cell_magic('time', '', '\npp = Preprocessor();\nXtrain, ytrain, Xtest = pp.DoPreprocessing();\n\nprint();\ncollect();\n')


# <a id="4"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:black; font-size:120%; text-align:left;padding:3.0px; background: #cceeff; border-bottom: 8px solid #004466" > MODEL TRAINING <br><div> 
#    

# In[8]:


get_ipython().run_cell_magic('time', '', '\ndef MakeIntPreds(y_valid, y_pred_valid, if_cutoff: bool = False):\n    """\n    Source- https://www.kaggle.com/code/arunklenin/space-titanic-eda-advanced-feature-engineering\n    This is a global function to find out the cut-off for the accuracy score\n    \n    Inputs:-\n    y_valid, y_pred_valid:- true and prediction probabilities \n    if_cutoff:- Do we need a cutoff value returned?\n    \n    Returns:-\n    y_pred_valid:- integer array based on cutoff\n    cutoff :- threshold value for cutoff (float)\n    """;\n    \n    try:\n        y_valid      = np.array(y_valid);\n        y_pred_valid = np.array(y_pred_valid);\n    except:\n        pass;\n        \n    fpr, tpr, threshold = roc_curve(y_valid, y_pred_valid);\n    pred_valid          = pd.DataFrame({\'label\': y_pred_valid});\n    thresholds          = np.array(threshold);\n    pred_labels         = (pred_valid[\'label\'].values > thresholds[:, None]).astype(int);\n    \n    acc_scores          = (pred_labels == y_valid).mean(axis = 1);\n    acc_df              = pd.DataFrame({\'threshold\': threshold, \'test_acc\': acc_scores});\n    \n    acc_df.sort_values(by = \'test_acc\', ascending = False, inplace = True);\n    cutoff = float(acc_df.iloc[0, 0]);\n    \n    y_pred_valid = np.where(y_pred_valid < cutoff,0,1);\n    \n    if if_cutoff:\n        return y_pred_valid, cutoff;\n    else:\n        return y_pred_valid;\n\nprint();\ncollect();\n')


# In[9]:


get_ipython().run_cell_magic('time', '', '\nclass OptunaEnsembler:\n    """\n    This is the Optuna ensemble class-\n    Source- https://www.kaggle.com/code/arunklenin/ps3e26-cirrhosis-survial-prediction-multiclass\n    """;\n    \n    def __init__(self):\n        self.study        = None;\n        self.weights      = None;\n        self.random_state = CFG.state;\n        self.n_trials     = CFG.ntrials;\n        self.direction    = CFG.metric_obj;\n        \n    def ScoreMetric(self, ytrue, ypred):\n        """\n        This is the metric function for the competition\n        """;\n        return accuracy_score(ytrue, MakeIntPreds(ytrue, ypred, False));\n\n    def _objective(self, trial, y_true, y_preds):\n        """\n        This method defines the objective function for the ensemble\n        """;\n        \n        if isinstance(y_preds, pd.DataFrame) or isinstance(y_preds, np.ndarray):\n            weights = [trial.suggest_float(f"weight{n}", 0, 1) for n in range(y_preds.shape[-1])];\n            axis = 1;\n        elif isinstance(y_preds, list):\n            weights = [trial.suggest_float(f"weight{n}", 0, 1) for n in range(len(y_preds))];\n            axis = 0;\n\n        # Calculating the weighted prediction:-\n        weighted_pred  = np.average(np.array(y_preds), axis = axis, weights = weights);\n        score          = self.ScoreMetric(y_true, weighted_pred);\n        return score;\n\n    def fit(self, y_true, y_preds):\n        "This method fits the Optuna objective on the fold level data";\n        \n        optuna.logging.set_verbosity = optuna.logging.ERROR;\n        self.study = \\\n        optuna.create_study(sampler    = TPESampler(seed = self.random_state), \n                            pruner     = HyperbandPruner(),\n                            study_name = "Ensemble", \n                            direction  = self.direction,\n                           );\n        \n        obj = partial(self._objective, y_true = y_true, y_preds = y_preds);\n        self.study.optimize(obj, n_trials = self.n_trials);\n               \n        if isinstance(y_preds, list):\n            self.weights = [self.study.best_params[f"weight{n}"] for n in range(len(y_preds))];\n        else:\n            self.weights = [self.study.best_params[f"weight{n}"] for n in range(y_preds.shape[-1])];\n        clear_output();\n\n    def predict(self, y_preds):\n        "This method predicts using the fitted Optuna objective";\n        \n        assert self.weights is not None, \'OptunaWeights error, must be fitted before predict\';\n        \n        if isinstance(y_preds, list):\n            weighted_pred = np.average(np.array(y_preds), axis=0, weights = self.weights);\n        else:\n            weighted_pred = np.average(np.array(y_preds), axis=1, weights = self.weights);\n        return weighted_pred;\n\n    def fit_predict(self, y_true, y_preds):\n        """\n        This method fits the Optuna objective on the fold data, then predicts the test set\n        """;\n        self.fit(y_true, y_preds);\n        return self.predict(y_preds);\n    \n    def weights(self):\n        return self.weights;\n    \nprint();\ncollect();\n')


# In[10]:


get_ipython().run_cell_magic('time', '', '\nclass MdlDeveloper(CFG):\n    """\n    This class implements the training pipeline elements-\n    1. Initializes the Model predictions\n    2. Trains and infers models\n    3. Returns the OOF and model test set predictions\n    """;\n    \n    def __init__(self, Xtrain, ytrain, Xtest, sel_cols, **kwarg):\n        """\n        In this method, we initialize the below-\n        1. Train-test data, selected columns\n        2. Metric, custom scorer, model and cv object\n        3. Output tables for score and predictions\n        """;\n        \n        self.Xtrain      = Xtrain;\n        self.ytrain      = ytrain;\n        self.y_grp       = ytrain;\n        self.Xtest       = Xtest;\n        self.sel_cols    = sel_cols;\n     \n        self._DefineModels();\n        self.cv          = self.all_cv[self.mdlcv_mthd];\n        self.methods     = list(self.Mdl_Master.keys());\n        self.OOF_Preds   = pd.DataFrame();\n        self.Mdl_Preds   = pd.DataFrame();\n        self.Scores      = pd.DataFrame(columns = self.methods + ["Ensemble"], \n                                        index = range(self.n_splits * self.n_repeats)\n                                       ); \n        self.mdlscorer   = make_scorer(self.ScoreMetric, \n                                       greater_is_better = False,\n                                       needs_proba       = True,\n                                       needs_threshold   = False,\n                                      );  \n        \n        PrintColor(f"\\n---> Selected model options-");\n        pprint(self.methods, depth = 1, width = 100, indent = 5);\n              \n    def _DefineModels(self):\n        """\n        This method initiliazes models for the analysis\n        It also initializes the CV methods and class-weights that could be tuned going ahead.\n        """;\n        \n        # Commonly used CV strategies for later usage:-\n        self.all_cv = \\\n        {\'KF\'  : KFold(n_splits= CFG.n_splits, shuffle = True, random_state= CFG.state),\n         \'RKF\' : RKF(n_splits= CFG.n_splits, n_repeats = CFG.n_repeats, random_state= CFG.state),\n         \'RSKF\': RSKF(n_splits= CFG.n_splits, n_repeats = CFG.n_repeats, random_state= CFG.state),\n         \'SKF\' : SKF(n_splits= CFG.n_splits, shuffle = True, random_state= CFG.state)\n        };\n               \n        self.Mdl_Master = \\\n        {                      \n         \'XGB1C\': XGBC(**{\'tree_method\'           : "gpu_hist" if CFG.gpu_switch == "ON" else "hist",\n                          \'objective\'             : \'binary:logistic\',\n                          \'eval_metric\'           : "logloss",\n                          \'random_state\'          : self.state,\n                          \'colsample_bytree\'      : 0.25,\n                          \'learning_rate\'         : 0.035,\n                          \'max_depth\'             : 8,\n                          \'n_estimators\'          : 1100,                         \n                          \'reg_alpha\'             : 0.09,\n                          \'reg_lambda\'            : 0.70,\n                          \'min_child_weight\'      : 12,\n                          \'early_stopping_rounds\' : CFG.nbrnd_erly_stp,\n                          \'verbosity\'             : 0,\n                         }\n                      ),\n            \n         \'XGB2C\': XGBC(**{\'tree_method\'           : "gpu_hist" if CFG.gpu_switch == "ON" else "hist",\n                          \'objective\'             : \'binary:logistic\',\n                          \'eval_metric\'           : "logloss",\n                          \'random_state\'          : self.state,\n                          \'colsample_bytree\'      : 0.40,\n                          \'learning_rate\'         : 0.02,\n                          \'max_depth\'             : 9,\n                          \'n_estimators\'          : 2500,                         \n                          \'reg_alpha\'             : 0.12,\n                          \'reg_lambda\'            : 0.8,\n                          \'min_child_weight\'      : 15,\n                          \'early_stopping_rounds\' : CFG.nbrnd_erly_stp,\n                          \'verbosity\'             : 0,\n                         }\n                      ),\n\n         \'XGB3C\': XGBC(**{\'tree_method\'           : "gpu_hist" if CFG.gpu_switch == "ON" else "hist",\n                          \'objective\'             : \'binary:logistic\',\n                          \'eval_metric\'           : "logloss",\n                          \'random_state\'          : self.state,\n                          \'colsample_bytree\'      : 0.5,\n                          \'learning_rate\'         : 0.04,\n                          \'max_depth\'             : 8,\n                          \'n_estimators\'          : 3000,                         \n                          \'reg_alpha\'             : 0.2,\n                          \'reg_lambda\'            : 0.6,\n                          \'min_child_weight\'      : 16,\n                          \'early_stopping_rounds\' : CFG.nbrnd_erly_stp,\n                          \'verbosity\'             : 0,\n                         }\n                      ),\n            \n         \'XGB4C\': XGBC(**{\'tree_method\'           : "gpu_hist" if CFG.gpu_switch == "ON" else "hist",\n                          \'objective\'             : \'binary:logistic\',\n                          \'eval_metric\'           : "logloss",\n                          \'random_state\'          : self.state,\n                          \'colsample_bytree\'      : 0.80,\n                          \'learning_rate\'         : 0.055,\n                          \'max_depth\'             : 6,\n                          \'n_estimators\'          : 2000,                         \n                          \'reg_alpha\'             : 0.005,\n                          \'reg_lambda\'            : 0.95,\n                          \'min_child_weight\'      : 16,\n                          \'early_stopping_rounds\' : CFG.nbrnd_erly_stp,\n                          \'verbosity\'             : 0,\n                          \'class_weight\'          : "balanced",\n                         }\n                      ),\n              \n         \'LGBM1C\':LGBMC(**{\'device\'              : "gpu" if CFG.gpu_switch == "ON" else "cpu",\n                           \'objective\'           : \'binary\',\n                           \'boosting_type\'       : \'gbdt\',\n                           \'random_state\'        : self.state,\n                           \'colsample_bytree\'    : 0.56,\n                           \'subsample\'           : 0.35,\n                           \'learning_rate\'       : 0.025,\n                           \'max_depth\'           : 8,\n                           \'n_estimators\'        : 3000,\n                           \'num_leaves\'          : 100,\n                           \'reg_alpha\'           : 0.14,\n                           \'reg_lambda\'          : 0.85,\n                           \'verbosity\'           : -1,\n                          }\n                       ),\n            \n         \'LGBM2C\':LGBMC(**{\'device\'              : "gpu" if CFG.gpu_switch == "ON" else "cpu",\n                           \'objective\'           : \'binary\',\n                           \'boosting_type\'       : \'gbdt\',\n                           \'data_sample_strategy\': "goss",\n                           \'random_state\'        : self.state,\n                           \'colsample_bytree\'    : 0.20,\n                           \'subsample\'           : 0.25,\n                           \'learning_rate\'       : 0.018,\n                           \'max_depth\'           : 9,\n                           \'n_estimators\'        : 3000,\n                           \'num_leaves\'          : 85, \n                           \'reg_alpha\'           : 0.15,\n                           \'reg_lambda\'          : 0.90,\n                           \'verbosity\'           : -1,\n                          }\n                       ),\n            \n         \'LGBM3C\':LGBMC(**{\'device\'              : "gpu" if CFG.gpu_switch == "ON" else "cpu",\n                           \'objective\'           : \'binary\',\n                           \'boosting_type\'       : \'gbdt\',\n                           \'random_state\'        : self.state,\n                           \'colsample_bytree\'    : 0.45,\n                           \'subsample\'           : 0.45,\n                           \'learning_rate\'       : 0.05,\n                           \'max_depth\'           : 6,\n                           \'n_estimators\'        : 3000,\n                           \'num_leaves\'          : 80, \n                           \'reg_alpha\'           : 0.05,\n                           \'reg_lambda\'          : 0.95,\n                           \'verbosity\'           : -1,\n                          }\n                       ), \n            \n         \'LGBM4C\':LGBMC(**{\'device\'              : "gpu" if CFG.gpu_switch == "ON" else "cpu",\n                           \'objective\'           : \'binary\',\n                           \'boosting_type\'       : \'gbdt\',\n                           \'random_state\'        : self.state,\n                           \'colsample_bytree\'    : 0.55,\n                           \'subsample\'           : 0.55,\n                           \'learning_rate\'       : 0.04,\n                           \'max_depth\'           : 9,\n                           \'n_estimators\'        : 3000,\n                           \'num_leaves\'          : 76, \n                           \'reg_alpha\'           : 0.08,\n                           \'reg_lambda\'          : 0.995,\n                           \'verbosity\'           : -1,\n                          }\n                       ),\n            \n        "CB1C" :  CBC(**{\'task_type\'           : "GPU" if CFG.gpu_switch == "ON" else "CPU",\n                         \'objective\'           : \'Logloss\',\n                         \'bagging_temperature\' : 0.1,\n                         \'colsample_bylevel\'   : 0.88,\n                         \'iterations\'          : 3000,\n                         \'learning_rate\'       : 0.065,\n                         \'od_wait\'             : 12,\n                         \'max_depth\'           : 7,\n                         \'l2_leaf_reg\'         : 1.75,\n                         \'min_data_in_leaf\'    : 20,\n                         \'random_strength\'     : 0.1, \n                         \'max_bin\'             : 100,\n                         \'verbose\'             : 0,\n                         \'use_best_model\'      : True,\n                        }\n                     ),\n            \n        "CB2C" :  CBC(**{\'task_type\'           : "GPU" if CFG.gpu_switch == "ON" else "CPU",\n                         \'objective\'           : \'Logloss\',\n                         \'bagging_temperature\' : 0.5,\n                         \'colsample_bylevel\'   : 0.50,\n                         \'iterations\'          : 2500,\n                         \'learning_rate\'       : 0.0525,\n                         \'od_wait\'             : 24,\n                         \'max_depth\'           : 8,\n                         \'l2_leaf_reg\'         : 1.235,\n                         \'min_data_in_leaf\'    : 13,\n                         \'random_strength\'     : 0.35, \n                         \'max_bin\'             : 160,\n                         \'verbose\'             : 0,\n                         \'use_best_model\'      : True,\n                        }\n                     ),\n            \n        "RFC" : RFC(n_estimators     = 300,\n                    criterion        = \'log_loss\',\n                    max_depth        = 7,\n                    min_samples_leaf = 17,\n                    max_features     =\'sqrt\',\n                    bootstrap        = True,\n                    oob_score        = False,\n                    n_jobs           = -1,\n                    random_state     = self.state,\n                    verbose          = 0,\n                   ),\n        \n        "GBC" : GBC(loss                = \'log_loss\',\n                    learning_rate       = 0.05,\n                    n_estimators        = 850,\n                    subsample           = 0.6,\n                    min_samples_leaf    = 15,\n                    max_depth           = 7,\n                    random_state        = self.state,\n                    verbose             = 0,\n                    validation_fraction = 0.1,\n                    n_iter_no_change    = 25,\n                   )  \n        };\n        return self;\n    \n    def ScoreMetric(self, ytrue, ypred):\n        """\n        This is the metric function for the competition scoring\n        """;\n        return accuracy_score(ytrue, MakeIntPreds(ytrue, ypred, False));\n       \n    def PostProcessPred(self, ypred):\n        """\n        This is an optional post-processing method\n        """;\n        return np.clip(ypred, a_min = 0, a_max = 1);\n    \n    def TrainMdl(self, test_preds_req: str = "Y"):\n        """\n        This method trains and infers from the model suite and returns the predictions and scores\n        It optionally predicts the test set too, if desired by the user\n        """;\n\n        # Initializing I-O:- \n        X,y, Xt    = self.Xtrain[self.sel_cols], self.ytrain.copy(deep = True), self.Xtest[self.sel_cols];\n        cols_drop  = [\'Source\', "id"];\n        ens        = OptunaEnsembler();\n        \n        self.FtreImp = pd.DataFrame(columns = self.methods, \n                                    index   = [c for c in self.sel_cols if c not in cols_drop]\n                                   ).fillna(0);\n        \n        # Making CV folds:-        \n        for fold_nb, (train_idx, dev_idx) in tqdm(enumerate(self.cv.split(X, self.y_grp))):\n            Xtr  = X.iloc[train_idx].drop(columns = cols_drop, errors = \'ignore\');  \n            Xdev = X.iloc[dev_idx].drop(columns = cols_drop, errors = \'ignore\');    \n            ytr  = y.loc[y.index.isin(Xtr.index)];\n            ydev = y.loc[y.index.isin(Xdev.index)];\n                   \n            # Initializing the OOF and test set predictions:-            \n            oof_preds = pd.DataFrame(columns = self.methods, index = Xdev.index);\n            mdl_preds = pd.DataFrame(columns = self.methods, index = Xt.index);\n            \n            PrintColor(f"\\n{\'=\' * 5} FOLD {fold_nb + 1} {\'=\' * 5}\\n");\n            # Initializing models across methods:-\n            for method in tqdm(self.methods):\n                model = Pipeline(steps = [("M", self.Mdl_Master.get(method))]); \n\n                # Fitting the model:-          \n                if "CB" in method:    \n                    model.fit(Xtr, ytr, \n                              M__eval_set = [(Xdev, ydev)], \n                              M__verbose = 0,\n                              M__early_stopping_rounds = CFG.nbrnd_erly_stp,\n                             ); \n\n                elif "LGBM" in method:\n                    model.fit(Xtr, ytr, \n                              M__eval_set = [(Xdev, ydev)], \n                              M__callbacks = [log_evaluation(0), \n                                              early_stopping(stopping_rounds = CFG.nbrnd_erly_stp, \n                                                             verbose = False,),\n                                             ],\n                             ); \n\n                elif "XGB" in method:\n                     model.fit(Xtr, ytr, \n                               M__eval_set = [(Xdev, ydev)], \n                               M__verbose  = 0,\n                              );            \n\n                else: \n                    model.fit(Xtr, ytr);\n                    \n                # Collating feature importance:-\n                try: \n                    self.FtreImp[method] += model["M"].feature_importances_;\n                except: \n                    pass;\n                    \n                # Collecting predictions and scores and post-processing OOF based on model method:-\n                dev_preds    = model.predict_proba(Xdev)[:,1];\n                train_preds  = model.predict_proba(Xtr)[:,1];\n                tr_score     = self.ScoreMetric(ytr.values.flatten(),train_preds);\n                score        = self.ScoreMetric(ydev.values.flatten(),dev_preds);\n\n                PrintColor(f"Score:- OOF = {score:.5f} | Train = {tr_score:.5f}| {method}", \n                           color = Fore.CYAN\n                          );  \n                oof_preds[method] = dev_preds;\n\n                # Integrating the predictions and scores:-               \n                self.Scores.at[fold_nb, method] = np.round(score, decimals= 6);\n                \n                if test_preds_req == "Y": \n                    test_preds = model.predict_proba(Xt.drop(columns = cols_drop, errors = "ignore"));\n                    if self.ist_reg_req == "Y": \n                        mdl_preds[method] = clb.predict(np.reshape(test_preds, -1));\n                    else:\n                        mdl_preds[method] = test_preds;\n                    del test_preds;\n                \n            try:\n                del dev_preds, train_preds, tr_score, score, clb_score, clb_tr_preds, clb_dev_preds;\n            except:\n                pass;\n                \n            # Ensembling the predictions:-\n            oof_preds["Ensemble"]          = ens.fit_predict(ydev, oof_preds[self.methods]);\n            clb_score                      = self.ScoreMetric(ydev, oof_preds["Ensemble"].values);\n            self.OOF_Preds                 = pd.concat([self.OOF_Preds, oof_preds], axis = 0, ignore_index = False);\n            self.Scores.at[fold_nb, "Ensemble"] = np.round(clb_score,6);\n            \n            if test_preds_req == "Y": \n                mdl_preds["Ensemble"] = ens.predict(mdl_preds[self.methods]);\n                self.Mdl_Preds        = pd.concat([self.Mdl_Preds, mdl_preds], axis = 0, ignore_index = False);\n                \n        # Averaging the predictions after all folds:-       \n        self.OOF_Preds = self.OOF_Preds.groupby(level = 0).mean();\n        if test_preds_req == "Y": \n            self.Mdl_Preds = self.Mdl_Preds[self.methods + ["Ensemble"]].groupby(level=0).mean();\n            \n        return self.OOF_Preds, self.Mdl_Preds, self.Scores;\n    \n    def DisplayAdjTbl(self, *args):\n        """\n        This function displays pandas tables in an adjacent manner, sourced from the below link-\n        https://stackoverflow.com/questions/38783027/jupyter-notebook-display-two-pandas-tables-side-by-side\n        """;\n\n        html_str = \'\';\n        for df in args:\n            html_str += df.to_html();\n        display_html(html_str.replace(\'table\',\'table style="display:inline"\'),raw=True);\n         \n    def DisplayScores(self):\n        "This method displays the scores and their means";\n        \n        PrintColor(f"\\n---> OOF score across all methods and folds\\n",color = Fore.LIGHTMAGENTA_EX);\n        display(self.Scores.style.format(precision = 5).\\\n                background_gradient(cmap = "Pastel2", subset = self.methods).\\\n                background_gradient(cmap = "mako", subset = ["Ensemble"]).\\\n                set_caption(f"\\nOOF scores across methods and folds\\n")\n               );\n        \n        PrintColor(f"\\n---> Mean OOF score across all methods and folds\\n",\n                   color = Fore.RED\n                  );\n        \n        display(self.Scores.mean().to_frame().\\\n                transpose().\\\n                style.format(precision = 5).\\\n                background_gradient(cmap = "terrain", subset = self.methods + ["Ensemble"], axis=1)\n               );\n        print();\n      \n    def MakePseudoLbl(self, up_cutoff: float, low_cutoff: float, **kwarg):\n        """\n        This method makes pseudo-labels using confident test set predictions to add to the training data\n        """;\n        \n        # Locating confident test-set predictions:-        \n        df = \\\n        self.Mdl_Preds.loc[(self.Mdl_Preds.Ensemble >= up_cutoff) | (self.Mdl_Preds.Ensemble <= low_cutoff), \n                           "Ensemble"\n                          ];\n        PrintColor(f"---> Pseudo Label additions from test set = {df.shape[0]:,.0f}", color = Fore.RED);\n        df = df.astype(np.uint8);\n        \n        #  Integrating new Xtrain and ytrain based on pseudo-labels:- \n        new_ytrain       = pd.concat([self.ytrain, df], axis=0, ignore_index = True);\n        new_ytrain.index = range(len(new_ytrain));\n        new_Xtrain       = pd.concat([self.Xtrain, self.Xtest.loc[df.index]], axis=0, ignore_index = True);\n        new_Xtrain.index = range(len(new_Xtrain));\n        \n        #  Verifying the additions:-\n        PrintColor(f"---> Revised train set shapes after pseudo labels = {new_Xtrain.shape} {new_ytrain.shape}");\n        return new_Xtrain, new_ytrain;\n    \n    def MakeMLPlots(self):\n        """\n        This method makes plots for the ML models, including feature importance and calibration curves\n        """;\n        \n        fig, axes = plt.subplots(len(self.methods), 2, figsize = (30, len(self.methods) * 9),\n                                 gridspec_kw = {\'hspace\': 0.475, \'wspace\': 0.2}, \n                                 width_ratios= [0.7, 0.3],\n                                 sharex = False,\n                                );\n    \n        for i, col in enumerate(self.methods):\n            try: \n                ax = axes[i,0];\n            except: \n                ax = axes[0];\n                \n            self.FtreImp[col].plot.bar(ax = ax, color = \'#0073e6\');\n            ax.set_title(f"{col} Importances", **CFG.title_specs);\n            ax.set(xlabel = \'\', ylabel = \'\');\n\n            try:\n                ax = axes[i,1];\n            except:\n                ax = axes[1];\n\n            Clb.from_predictions(self.ytrain[0:len(self.OOF_Preds)], \n                                 self.OOF_Preds[col], \n                                 n_bins = 20, \n                                 ref_line = True,\n                                 **{\'color\': \'#0073e6\', \'linewidth\': 1.2, \n                                    \'markersize\': 3.75, \'marker\': \'o\', \'markerfacecolor\': \'#cc7a00\'},\n                                 ax = ax\n                                );\n            ax.set_title(f"{col} Calibration", **CFG.title_specs);\n            ax.set(xlabel = \'\', ylabel = \'\',);\n            ax.set_yticks(np.arange(0,1.01, 0.05), \n                          labels = np.round(np.arange(0,1.01, 0.05), 2), fontsize = 7.0);\n            ax.set_xticks(np.arange(0,1.01, 0.05), \n                          labels = np.round(np.arange(0,1.01, 0.05), 2), \n                          fontsize = 7.0, \n                          rotation = 90\n                         );\n            ax.legend(\'\');\n\n        plt.tight_layout();\n        plt.show();\n                   \nprint();\ncollect();\n')


# In[11]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ML == "Y":\n    sel_cols = np.array(Xtrain.columns);\n    PrintColor(f"\\n---> Selected model columns");\n    with np.printoptions(linewidth = 120):\n        pprint(sel_cols);\n        \nprint();\ncollect();\n')


# In[12]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ML == "Y":\n    # Training the models with a CV analysis:-  \n    md = MdlDeveloper(Xtrain, ytrain, Xtest, sel_cols = sel_cols);\n    OOF_Preds, Mdl_Preds, Scores = md.TrainMdl(test_preds_req = "Y");\n    \n    PrintColor(f"\\n{\'=\' * 20} ML MODELS TRAINING AND CV {\'=\' * 20}\\n", color = Fore.MAGENTA);\n    md.DisplayScores();\n    collect();\n    \nprint();\ncollect();\n')


# In[13]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ML == "Y" and CFG.pseudo_lbl_req == "Y":\n    Xtrain, ytrain = md.MakePseudoLbl(up_cutoff = CFG.pseudolbl_up, low_cutoff = CFG.pseudolbl_low);\n    \n    # Re-initializing the developer class with the new training set:-  \n    md = MdlDeveloper(Xtrain, ytrain, Xtest, sel_cols = sel_cols);\n    OOF_Preds, Mdl_Preds, Scores = md.TrainMdl(test_preds_req = "Y");\n    \n    PrintColor(f"\\n{\'=\' * 20} ML MODELS TRAINING AND CV AFTER PSEUDO-LABELS {\'=\' * 20}\\n", \n               color = Fore.MAGENTA\n              );\n    md.DisplayScores();\n    collect();    \n    \nelse:\n    PrintColor(f"---> Pseudo Labels are not needed", Fore.RED);\n\nprint();\ncollect();\n')


# In[14]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ML == "Y":\n    md.MakeMLPlots();\n    \nprint();\ncollect();\n')


# In[15]:


get_ipython().run_cell_magic('time', '', '\n# Saving the datasets:-\nif CFG.ML == "Y":\n    sub_fl = pp.sub_fl.copy();\n    sub_fl[CFG.target] = Mdl_Preds["Ensemble"].values.flatten();\n    sub_fl[CFG.target] = np.where(sub_fl[CFG.target] > 0.596, True, False);\n    \n    PrintColor(f"\\nTarget distribution- models\\n");\n    display(sub_fl[CFG.target].value_counts());\n    print();\n    \n    # Blending with good public notebooks:-\n    sub1 = pd.read_csv(f"/kaggle/input/space-titanic-eda-advanced-feature-engineering/submission.csv");\n    sub2 = pd.read_csv(f"/kaggle/input/space-titanic/XGB_best.csv");\n    sub_fl[CFG.target] = sub_fl[CFG.target] | sub1[CFG.target] | sub2[CFG.target];\n    sub_fl.to_csv(f"Submission_V{CFG.version_nb}.csv", index = None);\n    \n    PrintColor(f"\\nTarget distribution- after blending\\n");\n    display(sub_fl[CFG.target].value_counts());\n    print();\n    \n    OOF_Preds.to_csv(f"OOF_Preds_V{CFG.version_nb}.csv");\n    Mdl_Preds.to_csv(f"Mdl_Preds_V{CFG.version_nb}.csv");\n    \n    PrintColor(f"\\nFinal submission file\\n");\n    display(sub_fl.head(10).style.format(precision = 3));\n    \ncollect();\nprint(); \n')


# 
#    

# <a id="5"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0052cc; border-bottom: 8px solid #cc9966" > NEXT STEPS<br> <div> 

# <div style= "font-family: Cambria; letter-spacing: 0px; color:#000000; font-size:110%; text-align:left;padding:3.0px; background: #f2f2f2" >
# 1. We will perform a detailed EDA and elicit feature interactions and other relevant insights<br>
# 2. Model tuning<br>
# 3. Including other models in the ensemble <br>
# 4. Better ensemble strategy <br>
# 5. Any other discussion/ public work based insights <br>
# 6. Better calibration <br>
# </div>
