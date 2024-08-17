#!/usr/bin/env python
# coding: utf-8

# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:black; font-size:120%; text-align:left;padding:3.0px; background: #cceeff; border-bottom: 8px solid #004466" > TABLE OF CONTENTS<br><div>  
# * [IMPORTS](#1)
# * [INTRODUCTION](#2)
#     * [CONFIGURATION](#2.1)
#     * [CONFIGURATION PARAMETERS](#2.2)    
#     * [DATASET COLUMNS](#2.3)
# * [PREPROCESSING](#3)
# * [ADVERSARIAL CV](#4)
# * [EDA AND VISUALS](#5) 
# * [DATA TRANSFORMS](#6)
# * [MODEL TRAINING](#7)    
# * [ENSEMBLE AND SUBMISSION](#8)  
# * [PLANNED WAY FORWARD](#9)     

# <a id="1"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:black; font-size:120%; text-align:left;padding:3.0px; background: #cceeff; border-bottom: 8px solid #004466" > IMPORTS<br> <div> 

# In[1]:


get_ipython().run_cell_magic('time', '', "\n# Installing select libraries:-\nfrom gc import collect;\nfrom warnings import filterwarnings;\nfilterwarnings('ignore');\nfrom IPython.display import clear_output;\n\n!pip install -q --upgrade scipy;\n!pip install -q category_encoders;\n\nclear_output();\nprint();\ncollect();\n")


# In[2]:


get_ipython().run_cell_magic('time', '', "\n# General library imports:-\nfrom copy import deepcopy;\nimport pandas as pd;\nimport numpy as np;\nimport re;\nfrom scipy.stats import mode, kstest, normaltest, shapiro, anderson, jarque_bera;\nfrom collections import Counter;\nfrom itertools import product;\nfrom colorama import Fore, Style, init;\nfrom warnings import filterwarnings;\nfilterwarnings('ignore');\n\nfrom tqdm.notebook import tqdm;\nimport seaborn as sns;\nimport matplotlib.pyplot as plt;\nfrom matplotlib.colors import ListedColormap as LCM;\n%matplotlib inline\n\nfrom pprint import pprint;\n\nprint();\ncollect();\nclear_output();\n")


# In[3]:


get_ipython().run_cell_magic('time', '', '\n# Importing model and pipeline specifics:-\nfrom category_encoders import OrdinalEncoder, OneHotEncoder;\n\n# Pipeline specifics:-\nfrom sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler;\nfrom sklearn.impute import SimpleImputer as SI;\nfrom sklearn.model_selection import (RepeatedStratifiedKFold as RSKF, \n                                     StratifiedKFold as SKF,\n                                     KFold, \n                                     RepeatedKFold as RKF, \n                                     cross_val_score);\nfrom sklearn.inspection import permutation_importance;\nfrom sklearn.feature_selection import mutual_info_classif, RFE;\nfrom sklearn.pipeline import Pipeline, make_pipeline;\nfrom sklearn.base import BaseEstimator, TransformerMixin;\nfrom sklearn.compose import ColumnTransformer;\n\n# ML Model training:-\nfrom sklearn.metrics import f1_score, confusion_matrix, make_scorer;\nfrom xgboost import DMatrix, XGBClassifier;\nfrom lightgbm import LGBMClassifier, log_evaluation, early_stopping;\nfrom catboost import CatBoostClassifier, Pool;\nfrom sklearn.ensemble import (RandomForestClassifier as RFC, \n                              ExtraTreesClassifier as ETC,\n                              AdaBoostClassifier as ABC,\n                              BaggingClassifier as BC,\n                              HistGradientBoostingClassifier as HGBC\n                             );\nfrom sklearn.linear_model import LogisticRegression as LC;\n\n# Ensemble and tuning:-\nimport optuna;\nfrom optuna import Trial, trial, create_study;\nfrom optuna.samplers import TPESampler, CmaEsSampler;\noptuna.logging.set_verbosity = optuna.logging.ERROR;\n\nclear_output();\nprint();\ncollect();\n')


# In[4]:


get_ipython().run_cell_magic('time', '', '\n# Setting rc parameters in seaborn for plots and graphs- \n# Reference - https://matplotlib.org/stable/tutorials/introductory/customizing.html:-\n# To alter this, refer to matplotlib.rcParams.keys()\n\nsns.set({"axes.facecolor"       : "#ffffff",\n         "figure.facecolor"     : "#ffffff",\n         "axes.edgecolor"       : "#000000",\n         "grid.color"           : "#ffffff",\n         "font.family"          : [\'Cambria\'],\n         "axes.labelcolor"      : "#000000",\n         "xtick.color"          : "#000000",\n         "ytick.color"          : "#000000",\n         "grid.linewidth"       : 0.75,  \n         "grid.linestyle"       : "--",\n         "axes.titlecolor"      : \'#0099e6\',\n         \'axes.titlesize\'       : 8.5,\n         \'axes.labelweight\'     : "bold",\n         \'legend.fontsize\'      : 7.0,\n         \'legend.title_fontsize\': 7.0,\n         \'font.size\'            : 7.5,\n         \'xtick.labelsize\'      : 7.5,\n         \'ytick.labelsize\'      : 7.5,        \n        });\n\n# Color printing    \ndef PrintColor(text:str, color = Fore.BLUE, style = Style.BRIGHT):\n    "Prints color outputs using colorama using a text F-string";\n    print(style + color + text + Style.RESET_ALL); \n\n# Making sklearn pipeline outputs as dataframe:-\nfrom sklearn import set_config; \nset_config(transform_output = "pandas");\npd.set_option(\'display.max_columns\', 50);\npd.set_option(\'display.max_rows\', 50);\n\nprint();\ncollect();\n')


# <a id="2"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:black; font-size:120%; text-align:left;padding:3.0px; background: #cceeff; border-bottom: 8px solid #004466" > INTRODUCTION<br><div> 

# | Version<br>Number | Version Details | CV score| Single/ Ensemble|Public LB Score|
# | :-: | --- | :-: | :-: |:-:|
# | **V1** |* EDA, plots and secondary features and encoding<br>* No scaling<br> * Used original data<br>* Tree based ML models and basic ensemble|0.71275|Simple blend |0.79268|
# | **V2** |* Same EDA as V1<br> * Incorporated custom metric callables in GBM<br> * Dropped 6 features|0.70755 |Simple blend |0.79878|
# | **V3** |* Same as V2<br> * Removed early stopping in GBM models|0.71058|Simple blend |0.79878|
# | **V4** |* Same as V2 **without the original data**<br> * Removed early stopping in GBM models|0.70283|Simple blend ||
# | **V5** |* Same as V2 <br> * Removed early stopping in GBM models <br> * Added more ML models to the suite|0.70051|Simple blend |0.76219|
# | **V6** |* Same as V2 <br> * Re-organized the score collation dataframe and OOF predictions|0.70051|Simple blend |0.76219|
# | **V7** |* Analysed hospital number and lesion features for signals <br> * Created OH encoding and Label encoding across features to analyse them better <br> * Curated tree based ML models only|0.73270|Simple blend|0.76829|
# | **V8** |* Same as V7 without post-processing in the final step|0.73270|Simple blend|0.78048|
# | **V9** |* Split the lesion feature into components <br> * Same as V8 otherwise |0.71099|Simple blend|0.79268|
# | **V10** |* Same as V9 <br> * Used post-processing in the last step |0.71099|Simple blend|0.78048|
# | **V11** |* Discarded the original dataset and same as V10 otherwise |0.70829|Simple blend||

# <a id="2.1"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > CONFIGURATION<br><div> 

# In[5]:


get_ipython().run_cell_magic('time', '', '\n# Configuration class:-\nclass CFG:\n    "Configuration class for parameters and CV strategy for tuning and training";\n    \n    # Data preparation:-   \n    version_nb         = 11;\n    test_req           = "N";\n    gpu_switch         = "OFF"; \n    state              = 42;\n    target             = \'outcome\';\n    episode            = 22;\n    path               = f"/kaggle/input/playground-series-s3e{episode}/";\n    orig_path          = f"/kaggle/input/horse-survival-dataset/horse.csv";\n    \n    dtl_preproc_req    = "Y";\n    adv_cv_req         = "N";\n    ftre_plots_req     = \'N\';\n    ftre_imp_req       = "N";\n    \n    # Data transforms and scaling:-    \n    conjoin_orig_data  = "Y";\n    sec_ftre_req       = "Y";\n    scale_req          = "N";\n    # NOTE---Keep a value here even if scale_req = N, this is used for linear models:-\n    scl_method         = "Z"; \n    enc_method         = \'Label\';\n    OH_cols            = [\'pain\', \'mucous_membrane\'];\n    tgt_mapper         = {"lived": 2, "euthanized": 1, "died": 0};\n    drop_tr_idx        = "Y";\n    \n    # Model Training:- \n    baseline_req       = "N";\n    pstprcs_oof        = "N";\n    pstprcs_train      = "N";\n    pstprcs_test       = "Y";\n    ML                 = "Y";\n    n_splits           = 5 ;\n    n_repeats          = 5 ;\n    nbrnd_erly_stp     = 50 ;\n    mdlcv_mthd         = \'RSKF\';\n    \n    # Ensemble:-    \n    ensemble_req       = "Y";\n    enscv_mthd         = "RSKF";\n    metric_obj         = \'maximize\';\n    ntrials            = 10 if test_req == "Y" else 200;\n    \n    # Global variables for plotting:-\n    grid_specs = {\'visible\': True, \'which\': \'both\', \'linestyle\': \'--\', \n                           \'color\': \'lightgrey\', \'linewidth\': 0.75};\n    title_specs = {\'fontsize\': 9, \'fontweight\': \'bold\', \'color\': \'tab:blue\'};\n\nprint();\nPrintColor(f"--> Configuration done!\\n");\ncollect();\n')


# In[6]:


get_ipython().run_cell_magic('time', '', '\n# Defining functions to be used throughout the code for common tasks:-\n\n# Scaler to be used for continuous columns:- \nall_scalers = {\'Robust\': RobustScaler(), \n               \'Z\': StandardScaler(), \n               \'MinMax\': MinMaxScaler()\n              };\nscaler      = all_scalers.get(CFG.scl_method);\n\n# Commonly used CV strategies for later usage:-\nall_cv= {\'KF\'  : KFold(n_splits= CFG.n_splits, shuffle = True, random_state= CFG.state),\n         \'RKF\' : RKF(n_splits= CFG.n_splits, n_repeats = CFG.n_repeats, random_state= CFG.state),\n         \'RSKF\': RSKF(n_splits= CFG.n_splits, n_repeats = CFG.n_repeats, random_state= CFG.state),\n         \'SKF\' : SKF(n_splits= CFG.n_splits, shuffle = True, random_state= CFG.state)\n        };\n\n# Defining the competition metric:-\ndef ScoreMetric(ytrue, ypred)-> float:\n    """\n    This function calculates the metric for the competition. \n    ytrue- ground truth array\n    ypred- predictions\n    returns - metric value (float)\n    \n    This function necessitates that the predictions array should be flattened.\n    """;\n    \n    return f1_score(ytrue, ypred, average = "micro");\n\ndef PostProcessPred(preds, post_process = "N"):\n    """\n    This is an optional post-processing function. \n    We correct predictions based on the insights in hospital numbers and lesions\n    """;\n    return preds;\n\n# Defining the scoring for LightGBM, XGBoost and Catboost:-\ndef ScoreLGBM(ytrue: np.array, ypred: np.array):\n    "Defines the custom metric for light GBM classifier";\n    \n    # Reshaping the prediction array to facilitate appropriate observations and flattening problem:-    \n    y_pred = ypred.reshape(ytrue.shape[0], -1);\n    return (\'MicroF1\', f1_score(ytrue, np.argmax(y_pred, axis=1), average = "micro"), True);\n\ndef ScoreXGB(ypred: np.array, dtrain: DMatrix):\n    "This function returns the custom metric according to the XGBoost requirements";  \n    \n    ytrue  = dtrain.get_label();\n    y_pred = ypred.reshape(ytrue.shape[0], -1);\n    return ("MicroF1", f1_score(ytrue, np.argmax(y_pred, axis= 1), average = "micro"));\n\nclass ScoreCB:\n    """\n    This class defines the custom scoring metric for the catboost classifier\n    Source - https://stackoverflow.com/questions/65462220/how-to-create-custom-eval-metric-for-catboost\n    """;\n    \n    def is_max_optimal(self):\n        "This method confirms if greater is better";\n        return True;\n\n    def evaluate(self, approxes, target, weight):    \n        y_true = np.array(target);\n        y_pred = np.array(approxes).argmax(0).reshape(y_true.shape[0], -1);\n        return (f1_score(y_true, np.argmax(y_pred, axis=1), average = "micro"), 1)\n\n    def get_final_error(self, error, weight):\n        return error;\n    \n# Designing a custom scorer to use in cross_val_predict and cross_val_score:-\nmyscorer = make_scorer(ScoreMetric, greater_is_better = True, needs_proba=False,);\n\ncollect();\nprint();\n')


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
# |  dtl_preproc_req  | Proprocessing required                                  | Y/N                   |    
# |  adv_cv_req       | Adversarial CV required                                 | Y/N                   |
# |  ftre_plots_req   | Feature plots required                                  | Y/N                   |
# |  ftre_imp_req     | Feature importance required                             | Y/N                   |
# |  conjoin_orig_data| Conjoin original data                                   | Y/N                   |
# |  sec_ftre_req     | Secondary features required                             | Y/N                   |
# |  scale_req        | Scaling required                                        | Y/N                   |
# |  scl_method       | Scaling method                                          | Z/ Robust/ MinMax     |
# |  enc_method       | Encoding method                                         |-                      |
# |  OH_cols          | Onehot columns                                          |list                   |
# |  tgt_mapper       | Target mapper                                           | dict                  |
# |  drop_tr_req      | Drop extra training elements not in test                | Y/N                   |
# |  baseline_req     | Baseline model required                                 | Y/N                   |
# |  pstprcs_oof      | Post-process OOF after model training                   | Y/N                   |
# |  pstprcs_train    | Post-process OOF during model training for dev-set      | Y/N                   |
# |  pstprcs_test     | Post-process test after training                        | Y/N                   |
# |  ML               | Machine Learning Models                                 | Y/N                   |
# |  n_splits         | Number of CV splits                                     | integer               |
# |  n_repeats        | Number of CV repeats                                    | integer               |
# |  nbrnd_erly_stp   | Number of early stopping rounds                         | integer               |
# |  mdl_cv_mthd      | Model CV method name                                    | RKF/ RSKF/ SKF/ KFold |

# <a id="2.3"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > DATASET AND COMPETITION DETAILS<br><div>
#     
# **Data columns**<br>
# This is available in my disscussion post as below<br>
# https://www.kaggle.com/competitions/playground-series-s3e22/discussion/438603https://www.kaggle.com/competitions/playground-series-s3e22/discussion/438603<br>
# <br>
# **Competition details and notebook objectives**<br>
# 1. This is a multi-class classification challenge to predict horse survival using the provided features. **F1-micro** is the metric for the challenge<br>
# 2. In this starter notebook, we start the assignment with a detailed EDA, feature plots, interaction effects, adversarial CV analysis and develop starter models to initiate the challenge. We will also incorporate other opinions and approaches as we move along the challenge.<br>

# <a id="3"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:black; font-size:120%; text-align:left;padding:3.0px; background: #cceeff; border-bottom: 8px solid #004466" > PREPROCESSING<br><div> 

# In[7]:


get_ipython().run_line_magic('time', '')

class Preprocessor():
    """
    This class aims to do the below-
    1. Read the datasets
    2. In this case, process the original data
    3. Check information and description
    4. Check unique values and nulls
    5. Collate starting features 
    6. Conjoin train-original data if requested based on Adversarial CV results
    """;
    
    def __init__(self):
        self.train    = pd.read_csv(CFG.path + f"train.csv", index_col = 'id');
        self.test     = pd.read_csv(CFG.path + f"test.csv", index_col = 'id');
        self.target   = CFG.target ;
        self.original = pd.read_csv(CFG.orig_path);
        self.conjoin_orig_data = CFG.conjoin_orig_data;
        self.dtl_preproc_req = CFG.dtl_preproc_req;
        
        self.sub_fl   = pd.read_csv(CFG.path + f"sample_submission.csv");
        
        PrintColor(f"Data shapes - train-test-original = {self.train.shape} {self.test.shape} {self.original.shape}");
        
        PrintColor(f"\nTrain set head", color = Fore.GREEN);
        display(self.train.head(5).style.format(precision = 3));
        PrintColor(f"\nTest set head", color = Fore.GREEN);
        display(self.test.head(5).style.format(precision = 3));
        PrintColor(f"\nOriginal set head", color = Fore.GREEN);
        display(self.original.head(5).style.format(precision = 3));
                 
        # Resetting original data index:-
        self.original.index = range(len(self.original));
        self.original.index+= max(self.test.index) + 1;
        self.original.index.name = 'id';
        
        #  Changing original data column order to match the competition column structure:-
        self.original = self.original.reindex(self.train.columns, axis=1);
  
    def _AddSourceCol(self):
        self.train['Source'] = "Competition";
        self.test['Source']  = "Competition";
        self.original['Source'] = 'Original';
        
        self.strt_ftre = self.test.columns;
        return self;
    
    def _CollateInfoDesc(self):
        if self.dtl_preproc_req == "Y":
            PrintColor(f"\n{'-'*20} Information and description {'-'*20}\n", color = Fore.MAGENTA);

            # Creating dataset information and description:
            for lbl, df in {'Train': self.train, 'Test': self.test, 'Original': self.original}.items():
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
    
    def _CollateUnqNull(self):
        
        if self.dtl_preproc_req == "Y":
            # Dislaying the unique values across train-test-original:-
            PrintColor(f"\nUnique and null values\n");
            _ = pd.concat([self.train[self.strt_ftre].nunique(), 
                           self.test[self.strt_ftre].nunique(), 
                           self.original[self.strt_ftre].nunique(),
                           self.train[self.strt_ftre].isna().sum(axis=0),
                           self.test[self.strt_ftre].isna().sum(axis=0),
                           self.original[self.strt_ftre].isna().sum(axis=0)
                          ], 
                          axis=1);
            _.columns = ['Train_Nunq', 'Test_Nunq', 'Original_Nunq', 
                         'Train_Nulls', 'Test_Nulls', 'Original_Nulls'
                        ];

            display(_.T.style.background_gradient(cmap = 'Blues', axis=1).\
                    format(formatter = '{:,.0f}')
                   );
            
        return self;
       
    def DoPreprocessing(self):
        self._AddSourceCol();
        self._CollateInfoDesc();
        self._CollateUnqNull();
        
        return self; 
        
    def ConjoinTrainOrig(self):
        if self.conjoin_orig_data == "Y":
            PrintColor(f"Train shape before conjoining with original = {self.train.shape}");
            train = pd.concat([self.train, self.original], axis=0, ignore_index = True);
            PrintColor(f"Train shape after conjoining with original= {train.shape}");
            
            train = train.drop_duplicates();
            PrintColor(f"Train shape after de-duping = {train.shape}");
            
            train.index = range(len(train));
            train.index.name = 'id';
        
        else:
            PrintColor(f"We are using the competition training data only");
            train = self.train;
        return train;
          
collect();
print();


# In[8]:


get_ipython().run_cell_magic('time', '', '\npp = Preprocessor();\npp.DoPreprocessing();\n\nprint();\ncollect();\n')


# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > INFERENCES<br> <div>

# <div style= "font-family: Cambria; letter-spacing: 0px; color:#000000; font-size:110%; text-align:left;padding:3.0px; background: #f2f2f2" >
# 1. We have numerical, categorical and object columns<br>
# 2. We may ensue null imputation in this challenge<br>
# 3. The dataset is very small risking a shakeup<br>
# </div>

# <a id="4"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:black; font-size:120%; text-align:left;padding:3.0px; background: #cceeff; border-bottom: 8px solid #004466" > ADVERSARIAL CV<br><div>

# In[9]:


get_ipython().run_cell_magic('time', '', '\n# Performing adversarial CV between the 2 specified datasets:-\ndef Do_AdvCV(df1:pd.DataFrame, df2:pd.DataFrame, source1:str, source2:str):\n    "This function performs an adversarial CV between the 2 provided datasets if needed by the user";\n    \n    # Adversarial CV per column:-\n    ftre = pp.test.select_dtypes(include = np.number).\\\n    drop(columns = [\'id\', "Source"], errors = \'ignore\').columns;\n    adv_cv = {};\n\n    for col in ftre:\n        shuffle_state = np.random.randint(low = 10, high = 100, size= 1);\n\n        full_df = \\\n        pd.concat([df1[[col]].assign(Source = source1), df2[[col]].assign(Source = source2)], \n                  axis=0, ignore_index = True).\\\n        sample(frac = 1.00, random_state = shuffle_state);\n\n        full_df = full_df.assign(Source_Nb = full_df[\'Source\'].eq(source2).astype(np.int8));\n\n        # Checking for adversarial CV:-\n        model = LGBMClassifier(random_state = CFG.state, max_depth = 6, learning_rate = 0.05);\n        cv    = all_cv[\'SKF\'];\n        score = np.mean(cross_val_score(model, \n                                        full_df[[col]], \n                                        full_df.Source_Nb, \n                                        scoring= \'roc_auc\', \n                                        cv     = cv)\n                       );\n        adv_cv.update({col: round(score, 4)});\n        collect();\n    \n    del ftre;\n    collect();\n    \n    fig, ax = plt.subplots(1,1,figsize = (12, 5));\n    pd.Series(adv_cv).plot.bar(color = \'tab:blue\', ax = ax);\n    ax.axhline(y = 0.60, color = \'red\', linewidth = 2.75);\n    ax.grid(**CFG.grid_specs); \n    plt.yticks(np.arange(0.0, 0.81, 0.05));\n    plt.show();\n    \n# Implementing the adversarial CV:-\nif CFG.adv_cv_req == "Y":\n    PrintColor(f"\\n---------- Adversarial CV - Train vs Original ----------\\n", \n               color = Fore.MAGENTA);\n    Do_AdvCV(df1 = pp.train, df2 = pp.original, source1 = \'Train\', source2 = \'Original\');\n    \n    PrintColor(f"\\n---------- Adversarial CV - Train vs Test ----------\\n", \n               color = Fore.MAGENTA);\n    Do_AdvCV(df1 = pp.train, df2 = pp.test, source1 = \'Train\', source2 = \'Test\');\n    \n    PrintColor(f"\\n---------- Adversarial CV - Original vs Test ----------\\n", \n               color = Fore.MAGENTA);\n    Do_AdvCV(df1 = pp.original, df2 = pp.test, source1 = \'Original\', source2 = \'Test\');   \n    \nif CFG.adv_cv_req == "N":\n    PrintColor(f"\\nAdversarial CV is not needed\\n", color = Fore.RED);\n    \ncollect();\nprint();\n')


# In[10]:


get_ipython().run_cell_magic('time', '', '\nprint();\ntrain, test, strt_ftre = pp.ConjoinTrainOrig(), pp.test.copy(deep = True), deepcopy(pp.strt_ftre);\ncat_cols  = test.select_dtypes(include = \'object\').columns[:-1];\ncont_cols = \\\ntest.drop(columns = [\'lesion_1\', \'lesion_2\', \'lesion_3\', \'hospital_number\'], \n          errors = \'ignore\'\n         ).\\\nselect_dtypes(exclude = \'object\').columns;\n\nPrintColor(f"\\nCategory columns\\n");\ndisplay(cat_cols);\nPrintColor(f"\\nContinuous columns\\n");\ndisplay(np.array(cont_cols));\nPrintColor(f"\\nAll columns\\n");\ndisplay(strt_ftre);\n\nprint();\ncollect();\n')


# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > INFERENCES<br><div>

# <div style= "font-family: Cambria; letter-spacing: 0px; color:#000000; font-size:110%; text-align:left;padding:3.0px; background: #f2f2f2" >
# 1. Train-test belong to the same distribution, we can perhaps rely on the CV score<br>
# 2. We need to further check the train-original distribution further, adversarial validation results indicate that we cannot use the original dataset based on a couple of features<br>
# </div>

# <a id="5"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:black; font-size:120%; text-align:left;padding:3.0px; background: #cceeff; border-bottom: 8px solid #004466" > VISUALS AND EDA <br><div> 
#  

# <a id="5.2"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > TARGET PLOT<br><div>

# In[11]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ftre_plots_req == "Y":\n    \n    fig, axes = plt.subplots(1,2, figsize = (12, 5), sharey = True, gridspec_kw = {\'wspace\': 0.35});\n\n    for i, df in tqdm(enumerate([pp.train, pp.original]), "Target balance ---> "):\n        ax= axes[i];\n        a = df[CFG.target].value_counts(normalize = True);\n        _ = ax.pie(x = a , labels = a.index.values, \n                   explode      = [0.0, 0.2, 0.2], \n                   startangle   = 40, \n                   shadow       = True, \n                   colors       = [\'#3377ff\', \'#66ffff\',\'#809fff\'], \n                   textprops    = {\'fontsize\': 7, \'fontweight\': \'bold\', \'color\': \'black\'},\n                   pctdistance  = 0.60, \n                   autopct = \'%1.1f%%\'\n                  );\n        df_name = \'Train\' if i == 0 else "Original";\n        _ = ax.set_title(f"\\n{df_name} data\\n", **CFG.title_specs);\n\n    plt.tight_layout();\n    plt.show();\n        \n        \n    \ncollect();\nprint();\n')


# In[12]:


get_ipython().run_cell_magic('time', '', '\n# Assessing target interactions:-\nif CFG.ftre_plots_req == "Y":\n    fig, axes = plt.subplots(1,2, figsize = (12, 4), gridspec_kw = {\'wspace\': 0.2});\n    \n    for i, (lbl, df) in enumerate({"Train": pp.train, "Original": pp.original}.items()):\n        ax = axes[i];\n        c = [\'#3377ff\', \'#6699cc\'];\n        df.groupby(CFG.target).size().plot.bar(ax = ax, color = c[i]);\n        ax.set_title(f"Target interaction - {lbl} set", **CFG.title_specs);\n        ax.set(xlabel = "");\n        \n    plt.tight_layout();\n    plt.show()\n        \n')


# <a id="5.4"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > CATEGORY COLUMN PLOTS<br><div>

# In[13]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ftre_plots_req == "Y":\n    fig, axes = plt.subplots(len(cat_cols), 3, figsize = (20, len(cat_cols)* 4.5), \n                             gridspec_kw = {\'wspace\': 0.25, \'hspace\': 0.3});\n\n    for i, col in enumerate(cat_cols):\n        ax = axes[i, 0];\n        a = pp.train[col].value_counts(normalize = True);\n        a.sort_index().plot.barh(ax = ax, color = \'#007399\');\n        ax.set_title(f"{col}_Train", **CFG.title_specs);\n        ax.set_xticks(np.arange(0.0, 0.9, 0.05), \n                      labels = np.round(np.arange(0.0, 0.9, 0.05),2), \n                      rotation = 90\n                     );\n        ax.set(xlabel = \'\', ylabel = \'\');\n        del a;\n\n        ax = axes[i, 1];\n        a = pp.test[col].value_counts(normalize = True);\n        a.sort_index().plot.barh(ax = ax, color = \'#0088cc\');\n        ax.set_title(f"{col}_Test", **CFG.title_specs);\n        ax.set_xticks(np.arange(0.0, 0.9, 0.05), \n                      labels = np.round(np.arange(0.0, 0.9, 0.05),2), \n                      rotation = 90\n                     );\n        ax.set(xlabel = \'\', ylabel = \'\');\n        del a;\n        \n        ax = axes[i, 2];\n        a = pp.original[col].value_counts(normalize = True);\n        a.sort_index().plot.barh(ax = ax, color = \'#0047b3\');\n        ax.set_title(f"{col}_Original", **CFG.title_specs);\n        ax.set_xticks(np.arange(0.0, 0.9, 0.05), \n                      labels = np.round(np.arange(0.0, 0.9, 0.05),2), \n                      rotation = 90\n                     );\n        ax.set(xlabel = \'\', ylabel = \'\');\n        del a;       \n    \n    plt.suptitle(f"Category column plots", **CFG.title_specs, y= 0.90);\n    plt.tight_layout();\n    plt.show();\n    \nprint();\ncollect();\n')


# <a id="5.5"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > CONTINUOUS COLUMN PLOTS<br><div>

# In[14]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ftre_plots_req == "Y":\n    df = pd.concat([pp.train[cont_cols].assign(Source = \'Train\'), \n                    pp.test[cont_cols].assign(Source = \'Test\'),\n                    pp.original[cont_cols].assign(Source = "Original")\n                   ], \n                   axis=0, ignore_index = True\n                  );\n    \n    fig, axes = plt.subplots(len(cont_cols), 4 ,figsize = (16, len(cont_cols) * 4.2), \n                             gridspec_kw = {\'hspace\': 0.35, \'wspace\': 0.3, \'width_ratios\': [0.80, 0.20, 0.20, 0.20]});\n    \n    for i,col in enumerate(cont_cols):\n        ax = axes[i,0];\n        sns.kdeplot(data = df[[col, \'Source\']], x = col, hue = \'Source\', \n                    palette = [\'#0039e6\', \'#ff5500\', \'#00b300\'], \n                    ax = ax, linewidth = 2.1\n                   );\n        ax.set_title(f"\\n{col}", **CFG.title_specs);\n        ax.grid(**CFG.grid_specs);\n        ax.set(xlabel = \'\', ylabel = \'\');\n        \n        ax = axes[i,1];\n        sns.boxplot(data = df.loc[df.Source == \'Train\', [col]], y = col, width = 0.25,\n                    color = \'#33ccff\', saturation = 0.90, linewidth = 0.90, \n                    fliersize= 2.25,\n                    ax = ax);\n        ax.set(xlabel = \'\', ylabel = \'\');\n        ax.set_title(f"Train", **CFG.title_specs);\n        \n        ax = axes[i,2];\n        sns.boxplot(data = df.loc[df.Source == \'Test\', [col]], y = col, width = 0.25, fliersize= 2.25,\n                    color = \'#80ffff\', saturation = 0.6, linewidth = 0.90, \n                    ax = ax); \n        ax.set(xlabel = \'\', ylabel = \'\');\n        ax.set_title(f"Test", **CFG.title_specs);\n        \n        ax = axes[i,3];\n        sns.boxplot(data = df.loc[df.Source == \'Original\', [col]], y = col, width = 0.25, fliersize= 2.25,\n                    color = \'#99ddff\', saturation = 0.6, linewidth = 0.90, \n                    ax = ax); \n        ax.set(xlabel = \'\', ylabel = \'\');\n        ax.set_title(f"Original", **CFG.title_specs);\n              \n    plt.suptitle(f"\\nDistribution analysis- continuous columns\\n", **CFG.title_specs, \n                 y = 0.905, x = 0.50\n                );\n    plt.tight_layout();\n    plt.show();\n    \nprint();\ncollect();\n')


# <a id="5.7"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > FEATURE INTERACTION AND UNIVARIATE RELATIONS<br><div>
#     
# We aim to start off with a simple correlation analysis to determine feature relations

# In[15]:


get_ipython().run_cell_magic('time', '', '\ndef MakeCorrPlot(df: pd.DataFrame, data_label:str, figsize = (30, 9)):\n    """\n    This function develops the correlation plots for the given dataset\n    """;\n    \n    fig, axes = plt.subplots(1,2, figsize = figsize, gridspec_kw = {\'hspace\': 0.2, \'wspace\': 0.1},\n                             sharey = True\n                            );\n    \n    for i, method in enumerate([\'pearson\', \'spearman\']):\n        corr_ = df.drop(columns = [\'id\', \'Source\'], errors = \'ignore\').corr(method = method);\n        ax = axes[i];\n        sns.heatmap(data = corr_,  \n                    annot= True,\n                    fmt= \'.2f\', \n                    cmap = \'Blues\',\n                    annot_kws= {\'fontweight\': \'bold\',\'fontsize\': 6.75}, \n                    linewidths= 1.5, \n                    linecolor=\'white\', \n                    cbar= False, \n                    mask= np.triu(np.ones_like(corr_)),\n                    ax= ax\n                   );\n        ax.set_title(f"\\n{method.capitalize()} correlation- {data_label}\\n", **CFG.title_specs);\n        \n    collect();\n    print();\n\n# Implementing correlation analysis:-\nif CFG.ftre_imp_req == "Y":\n    for lbl, df in {"Train": pp.train[cont_cols], \n                    "Test": pp.test[cont_cols], \n                    "Original": pp.original[cont_cols]\n                   }.items():\n        MakeCorrPlot(df = df, data_label = lbl, figsize = (16,5));\n\nprint();\ncollect();\n')


# <a id="5.9"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > SURVIVAL ANALYSIS WITH CATEGORY COLUMNS<br> <div>

# In[16]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ftre_plots_req == "Y":\n    \n    # Analyzing survival chances across category columns:-\n    fig, axes = plt.subplots(len(cat_cols), 4, gridspec_kw = {\'hspace\': 0.35, \'wspace\': 0.35},\n                             figsize = (20, 4 * len(cat_cols)), \n                             sharex = True\n                            );\n    for lbl, mdl_df in tqdm({\'Train\': pp.train, "Original": pp.original}.items()):\n        for i, col in tqdm(enumerate(cat_cols)):\n            df = pd.crosstab(mdl_df[col], mdl_df[CFG.target]);\n            df[\'Sum_C\'] = np.sum(df, axis=1);\n            df1 = df.apply(lambda x: x/ x[\'Sum_C\'], axis=1);\n\n            if lbl == "Train": j = 0;\n            else: j = 2;\n\n            ax = axes[i,j];\n            sns.heatmap(df.iloc[:, :-1], cmap = \'winter\', fmt= \',.0f\', annot = True, \n                        cbar = False, linewidths= 1.5, linecolor=\'white\',\n                        annot_kws= {\'fontweight\': \'bold\',\'fontsize\': 6.75},\n                        ax = ax\n                       );\n            ax.set(xlabel = \'\', ylabel = \'\');\n            ax.set_title(f"{col} {lbl}", **CFG.title_specs);\n\n            ax = axes[i,j+1];\n            sns.heatmap(df1.iloc[:, :-1], cmap = \'icefire\', fmt= \',.2%\', annot = True, \n                        cbar = False, linewidths= 1.5, linecolor=\'white\',\n                        annot_kws= {\'fontweight\': \'bold\',\'fontsize\': 6.75},\n                        ax = ax\n                       );\n            ax.set(xlabel = \'\', ylabel = \'\');\n            ax.set_title(f"{col}_pct {lbl}", **CFG.title_specs);\n\n            del df, df1;\n\n    plt.suptitle(f"Survival analysis with category columns", **CFG.title_specs, y = 0.90);\n    plt.tight_layout();   \n    plt.show();\n\nprint();\ncollect();\n')


# <a id="5.10"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > HOSPITAL NUMBER FEATURE ANALYSIS<br> <div>

# In[17]:


get_ipython().run_cell_magic('time', '', '\n# Curating a complete dataset with all hospital numbers:-\ndf = \\\npd.concat([pp.train[[\'hospital_number\', CFG.target]].assign(Source = "Train"), \n           pp.test[[\'hospital_number\']].assign(Source = "Test"), \n           pp.original[[\'hospital_number\', CFG.target]].assign(Source = "Original")\n          ], axis=0, ignore_index = False);\n\n# Analyzing numbers in train + original only:-\nhnb_only_tr = \\\n[hnb for hnb in df.loc[df.Source != "Test", \'hospital_number\'].unique()\n if hnb not in df.loc[df.Source == "Test", \'hospital_number\'].unique()];\n\n# Analyzing numbers in the test only:-\nhnb_only_test = \\\n[hnb for hnb in df.loc[df.Source == "Test", \'hospital_number\'].unique()\n if hnb not in df.loc[df.Source != "Test", \'hospital_number\'].unique()];\n\nPrintColor(f"\\n---> Hospital numbers in the train + original data but not in test");\npprint(np.array(hnb_only_tr));\nPrintColor(f"\\n---> Hospital numbers in the test data but not in train + original");\npprint(np.array(hnb_only_test));\n\nif CFG.ftre_plots_req == "Y":\n    print(\'\\n\'*2)\n    # Plotting the relevant hospital number with the target:-\n    _ = df.loc[df.hospital_number.isin(hnb_only_tr), ["hospital_number", CFG.target]];\n    _ = pd.crosstab(_.hospital_number, _[CFG.target]);\n    _[\'Total\'] = np.sum(_, axis=1).values;\n\n    fig, axes = plt.subplots(1,2, figsize = (25, 6), \n                           gridspec_kw = {"hspace": 0.2, "width_ratios": [0.8, 0.2]}\n                          );\n    ax = axes[0];\n    _.drop(columns = [\'Total\']).\\\n    plot.\\\n    bar(stacked = True, color = sns.color_palette(\'icefire\', n_colors = 3), ax = ax);\n    ax.set_title(f"\\nTrain data hospital-number not in test\\n", **CFG.title_specs);\n    ax.legend(bbox_to_anchor = (0.5,-0.2), fontsize = 10);\n    ax.set_yticks(range(0,10), np.int8(range(0,10)));\n    ax.set(xlabel = \'\');\n    \n    ax = axes[1];\n    np.sum(_.drop(columns = [\'Total\']), axis=0).\\\n    plot.bar(color = sns.color_palette(\'icefire\', n_colors = 3), ax = ax);\n    ax.set_title(f"\\nTarget analysis\\n", **CFG.title_specs);\n    ax.set(xlabel = \'\');\n\n    plt.tight_layout();\n    plt.show();\n\ncollect();\nprint();\n')


# In[18]:


get_ipython().run_cell_magic('time', '', '\ntry: del _, HNB_Prf, df; \nexcept: PrintColor(f"\\n---> Intended dataframe to delete does not exist", color = Fore.RED); \n\n# Analyzing targets per hospital number between the train-test common ones:-\ndf = train.loc[~train.hospital_number.isin(hnb_only_tr), [\'hospital_number\', CFG.target]];\ndf = pd.crosstab(df[\'hospital_number\'], df[CFG.target])\ndf[\'Total\'] = np.sum(df, axis=1);\n\ndf = \\\npd.concat([df, df.apply(lambda x: x/ x[\'Total\'], axis=1).\\\n           drop(columns = [\'Total\']).add_suffix(f"_pct")], axis=1);\n\nHNB_tgt = \\\n{col: df.loc[df[f"{col}_pct"] == 1.0].index for col in df.columns[0:3]};\nPrintColor(f"\\n---> Special hospital numbers for target post-processing\\n");\npprint(HNB_tgt, indent = 2, depth = 2, width = 100);\n\ndf.to_csv(f"HNB_V{CFG.version_nb}.csv");\ncollect();\nprint();\n')


# <a id="5.11"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > LESSION FEATURE ANALYSIS<br> <div>

# In[19]:


get_ipython().run_cell_magic('time', '', '\n# Analyzing lession 1 and lession 2 only as lession 3 is 0 in the test set:-\n\n# Curating a complete dataset with all lession1 numbers:-\ndf = \\\npd.concat([pp.train[[\'lesion_1\', CFG.target]].assign(Source = "Train"), \n           pp.test[[\'lesion_1\']].assign(Source = "Test"), \n           pp.original[[\'lesion_1\', CFG.target]].assign(Source = "Original")\n          ], axis=0, ignore_index = False);\n\n# Analyzing numbers in train + original only:-\nl1_only_tr = \\\n[l1 for l1 in df.loc[df.Source != "Test", \'lesion_1\'].unique()\n if l1 not in df.loc[df.Source == "Test", \'lesion_1\'].unique()];\n\n# Analyzing numbers in the test only:-\nl1_only_test = \\\n[hnb for hnb in df.loc[df.Source == "Test", \'lesion_1\'].unique()\n if hnb not in df.loc[df.Source != "Test", \'lesion_1\'].unique()];\n\nPrintColor(f"\\n---> Lesion1 numbers in the train + original data but not in test");\npprint(np.array(l1_only_tr));\nPrintColor(f"\\n---> Lesion1 numbers in the test data but not in train + original");\npprint(np.array(l1_only_test));\n\nif CFG.ftre_plots_req == "Y":\n    print(\'\\n\'*2)\n    # Plotting the relevant lesion number with the target:-\n    _ = df.loc[df[\'lesion_1\'].isin(l1_only_tr), ["lesion_1", CFG.target]];\n    _ = pd.crosstab(_[\'lesion_1\'], _[CFG.target]);\n    _[\'Total\'] = np.sum(_, axis=1).values;\n\n    fig, axes = plt.subplots(1,2, figsize = (25, 6), \n                           gridspec_kw = {"hspace": 0.2, "width_ratios": [0.8, 0.2]}\n                          );\n    ax = axes[0];\n    _.drop(columns = [\'Total\']).\\\n    plot.\\\n    bar(stacked = True, color = sns.color_palette(\'icefire\', n_colors = 3), ax = ax);\n    ax.set_title(f"\\nTrain data lesion-number not in test\\n", **CFG.title_specs);\n    ax.legend(bbox_to_anchor = (0.5,-0.2), fontsize = 10);\n    ax.set_yticks(range(0,10), np.int8(range(0,10)));\n    ax.set(xlabel = \'\');\n    \n    ax = axes[1];\n    np.sum(_.drop(columns = [\'Total\']), axis=0).\\\n    plot.bar(color = sns.color_palette(\'icefire\', n_colors = 3), ax = ax);\n    ax.set_title(f"\\nTarget analysis\\n", **CFG.title_specs);\n    ax.set(xlabel = \'\');\n\n    plt.suptitle(f"Analysis - Lesion 1", **CFG.title_specs, y = 1.0);\n    plt.tight_layout();\n    plt.show();\n    \ncollect();\nprint();\n')


# In[20]:


get_ipython().run_cell_magic('time', '', '\n# Splitting the lesion1 values:-\n_ = \\\ntrain[\'lesion_1\'].astype(str).str.split(\'\', expand = True).add_prefix("lesion_1_").\\\ndrop(columns = [\'lesion_1_0\', "lesion_1_6"])\n_ = _.applymap(lambda x: 0 if x in ["", None, np.NaN] else x).astype(np.int8);\n_[CFG.target] = train[CFG.target].values.flatten();\n\nfig, axes = plt.subplots(2,5, figsize = (25, 12), \n                         gridspec_kw = {"hspace": 0.2, "wspace": 0.15,"height_ratios": [0.4, 0.6],}\n                        );\n\nfor i, col in enumerate(_.columns[0:-1]):\n    j = i+1;\n    ax = axes[0, i];\n    n_colors = _[f\'lesion_1_{j}\'].unique().shape[0];\n    _[f\'lesion_1_{j}\'].\\\n    value_counts().sort_index().\\\n    plot.bar(ax = ax, color = sns.color_palette(\'winter\', n_colors = n_colors, desat = 0.45));\n    ax.set_title(f\'lesion_1_{j}\', **CFG.title_specs);\n    ax.set(xlabel = \'\', ylabel = "");\n    \n    ax = axes[1,i];\n    sns.heatmap(pd.crosstab(_[f\'lesion_1_{j}\'], _[CFG.target]), \n                annot = True, cbar = None, fmt=\'.0f\',\n                linewidths = 2.5, linecolor =\'white\', \n                cmap= "Blues",\n                annot_kws= {"fontsize": 11, "fontweight": "bold"},\n                ax = ax\n               );\n    ax.set(xlabel = \'\', ylabel = "");\n\nplt.suptitle(f"Lesion1 split analysis", **CFG.title_specs, y= 0.95)\nplt.tight_layout();\nplt.show();\n    \n')


# In[21]:


get_ipython().run_cell_magic('time', '', '\n# Curating a complete dataset with all lession2 numbers:-\ndf = \\\npd.concat([pp.train[[\'lesion_2\', CFG.target]].assign(Source = "Train"), \n           pp.test[[\'lesion_2\']].assign(Source = "Test"), \n           pp.original[[\'lesion_2\', CFG.target]].assign(Source = "Original")\n          ], axis=0, ignore_index = False);\n\n# Analyzing numbers in train + original only:-\nl2_only_tr = \\\n[l1 for l1 in df.loc[df.Source != "Test", \'lesion_2\'].unique()\n if l1 not in df.loc[df.Source == "Test", \'lesion_2\'].unique()];\n\n# Analyzing numbers in the test only:-\nl2_only_test = \\\n[hnb for hnb in df.loc[df.Source == "Test", \'lesion_2\'].unique()\n if hnb not in df.loc[df.Source != "Test", \'lesion_2\'].unique()];\n\nPrintColor(f"\\n---> Lesion2 numbers in the train + original data but not in test");\npprint(np.array(l2_only_tr));\nPrintColor(f"\\n---> Lesion2 numbers in the test data but not in train + original");\npprint(np.array(l2_only_test));\n\ncollect();\nprint();\n')


# In[22]:


get_ipython().run_cell_magic('time', '', '\n# Creating a post-processing dataset for later usage:-\ntry: del _, df; \nexcept: PrintColor(f"\\n---> Intended dataframe to delete does not exist", color = Fore.RED); \n\n# Analyzing targets per lesion1 number between the train-test common ones:-\ndf = train.loc[~train[\'lesion_1\'].isin(l1_only_tr), [\'lesion_1\', CFG.target]];\ndf = pd.crosstab(df[\'lesion_1\'], df[CFG.target])\ndf[\'Total\'] = np.sum(df, axis=1);\n\ndf = \\\npd.concat([df, df.apply(lambda x: x/ x[\'Total\'], axis=1).\\\n           drop(columns = [\'Total\']).add_suffix(f"_pct")], axis=1);\n\nL1_tgt = \\\n{col: df.loc[df[f"{col}_pct"] == 1.0].index for col in df.columns[0:3]};\nPrintColor(f"\\n---> Special lesion 1 numbers for target post-processing\\n");\npprint(L1_tgt, indent = 2, depth = 2, width = 100);\n\ndf.to_csv(f"L1_V{CFG.version_nb}.csv");\n\n# Analyzing targets per lesion2 number between the train-test common ones:-\ndel df;\ndf = train.loc[~train[\'lesion_2\'].isin(l2_only_tr), [\'lesion_2\', CFG.target]];\ndf = pd.crosstab(df[\'lesion_2\'], df[CFG.target])\ndf[\'Total\'] = np.sum(df, axis=1);\n\ndf = \\\npd.concat([df, df.apply(lambda x: x/ x[\'Total\'], axis=1).\\\n           drop(columns = [\'Total\']).add_suffix(f"_pct")], axis=1);\n\nL2_tgt = \\\n{col: df.loc[df[f"{col}_pct"] == 1.0].index for col in df.columns[0:3]};\nPrintColor(f"\\n---> Special lesion 2 numbers for target post-processing\\n");\npprint(L2_tgt, indent = 2, depth = 2, width = 100);\n\ndf.to_csv(f"L2_V{CFG.version_nb}.csv");\n\ncollect();\nprint();\n')


# <a id="5.12"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > INFERENCES<br> <div>

# <div style= "font-family: Cambria; letter-spacing: 0px; color:#000000; font-size:110%; text-align:left;padding:3.0px; background: #f2f2f2" >
# 1. Feature selection is a very important part of the assignment. We have lots of features and feature selection will be a differentiator<br>
# 2. Quite a few features have outliers. Outlier handling may be another differentiator in this challenge. Certain categorical features have label outliers that are handled while encoding<br>
# 3. Columns are not highly correlated. Dimensionality reduction may help here<br>
# 4. We will need to encode lots of object columns with different encoder types<br>
# 5. Certain key inferences regarding columns (survival) are as below-<br>
#     - Adults<br>
#     - Non-surgical treatments<br>
#     - Normal values in health risk indicators<br>
# 6. Lesion 3 is a quasi-constant feature and is superfluous as it is 0 in the test set<br>
# 7. Hospital number may provide insights for feature creation perhaps<br>
# 8. Lesions are categorical columns as encoded in the subsequent section<br>
# 9. We may choose to exclude the training set hospital numbers and lesion numbers that do not appear in the test set perhaps
# </div>

# <a id="6"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:black; font-size:120%; text-align:left;padding:3.0px; background: #cceeff; border-bottom: 8px solid #004466" > DATA TRANSFORMS <br><div> 
#     
# This section aims at creating secondary features, scaling and if necessary, conjoining the competition training and original data tables<br>
# 

# In[23]:


get_ipython().run_cell_magic('time', '', '\n# Data transforms:-\nclass Xformer(TransformerMixin, BaseEstimator):\n    """\n    This class is used to create secondary features from the existing data.\n    We correct the hospital number in the test set to equal the nearest training data number too\n    """;\n    \n    def __init__(self, hnb_only_test:list, hnb_only_train: list): \n        self.sec_ftre_req = CFG.sec_ftre_req;\n        self.hnb_only_test = hnb_only_test;\n        self.hnb_only_train = hnb_only_train;\n    \n    def fit(self, X, y= None, **params):\n        "This method adjusts the only-test hospital number to the nearest training set number";\n        \n        hnb_diff = \\\n        {nb: abs(nb - base_val) for nb in X[\'hospital_number\'].unique() for base_val in self.hnb_only_test};\n        self.hnb_mapper = {self.hnb_only_test[0]: min(hnb_diff, key= hnb_diff.get)}\n    \n        self.ip_cols = X.columns;\n        return self;\n    \n    def transform(self, X, y= None, **params):       \n        global strt_ftre;\n        df    = X.copy();  \n        \n        # Correcting the test-only hospital number with the nearest training set number:-\n        try:\n            df[\'hospital_number\'] = df[\'hospital_number\'].astype(np.int64).map(self.hnb_mapper).astype(np.int64);\n        except:\n            PrintColor(f"---> Check the hospital number assignment", Fore.RED);\n            df.loc[df[\'hospital_number\'] == 528338, "hospital_number"] = 528355;\n        \n        # Splitting the lesion 1 feature into its components:-\n        l1_prf = \\\n        df[\'lesion_1\'].astype(np.int64).\\\n        astype(str).str.split(\'\', expand = True).\\\n        applymap(lambda x: 0 if x in [\'\', None, np.NaN] else x).astype(np.int8).\\\n        add_prefix("lesion_1_").\\\n        drop(columns = [\'lesion_1_0\', "lesion_1_6"], errors = "ignore");\n        \n        df = pd.concat([df, l1_prf], axis=1);\n        del l1_prf;\n\n        if self.sec_ftre_req == "Y":\n            df[\'rectal_temp_risk\'] = np.where(df.rectal_temp >= 37.8,1,0).astype(np.int8);\n            df[\'pulse_risk\']       = np.where(df.pulse >= 40,1,0).astype(np.int8);\n            df[\'cell_vol_risk\']    = np.where(df.packed_cell_volume >= 50, 1,0).astype(np.int8);\n            df[\'protein_risk\']     = np.where(df.total_protein >= 7.5, 1,0).astype(np.int8);\n                \n        if CFG.sec_ftre_req != "Y": \n            PrintColor(f"Secondary features are not required", color = Fore.RED);    \n        \n        self.op_cols = df.columns;  \n        return df;\n    \n    def get_feature_names_in(self, X, y=None, **params): \n        return self.ip_cols;    \n    \n    def get_feature_names_out(self, X, y=None, **params): \n        return self.op_cols;\n    \ncollect();\nprint();\n')


# In[24]:


get_ipython().run_cell_magic('time', '', '    \n# Encoding the categorical columns with domain based encoding:-\nclass Encoder(TransformerMixin, BaseEstimator):\n    """\n    This class is used to create encoded features from the existing object data\n    """; \n    \n    def __init__(self): pass;\n    \n    def fit(self, X, y= None, **params):\n        self.ip_cols = X.columns;\n        return self;\n    \n    def transform(self, X, y= None, **params):\n        """\n        This method performs manual label encoding for the category columns and lesion columns.\n        This is done as per the original data instructions- refer the original metadata page for details\n        """;\n        \n        df = X.copy();\n        df[\'surgery\'] = df[\'surgery\'].map({\'no\': 1, \'yes\': 0}).astype(np.int8);\n        df[\'age\']     = df[\'age\'].map({\'adult\': 0, \'young\': 1}).astype(np.int8);\n        df[\'temp_of_extremities\'] = \\\n        df[\'temp_of_extremities\'].map({\'None\': 0, "normal":1, "warm": 2, "cool":3, "cold":4}).astype(np.int8);\n        df[\'peripheral_pulse\'] = \\\n        df[\'peripheral_pulse\'].map({\'NA\': 0, "None": 0, \n                                      "normal":1, "increased": 2, "reduced":3, "absent":4}\n                                  ).astype(np.int8);\n        df[\'mucous_membrane\'] = \\\n        df[\'mucous_membrane\'].map({\'NA\': 0, "None": 0, "normal":1, "normal_pink":1, "pink": 2, "bright" : 3,\n                                     "bright_pink":3, "pale_pink":4 , "pale_cyanotic": 5, \n                                     "bright_red":6, "injected": 6, "dark_cyanotic": 7\n                                    }\n                                 ).astype(np.int8);\n        df[\'capillary_refill_time\'] = \\\n        df[\'capillary_refill_time\'].map({\'NA\': 0, "None": 0, "less_3_sec":1, "3": 2, "more_3_sec": 2}).astype(np.int8); \n        df[\'pain\'] = \\\n        df[\'pain\'].map({"NA": 0, "None": 0, "alert" : 1, "no_pain": 2, "depressed": 3, \n                        "mild_pain": 4, \'slight\': 3, "moderate": 4, "severe_pain": 5, "extreme_pain": 6\n                       }\n                      ).astype(np.int8);\n        df[\'peristalsis\'] = \\\n        df[\'peristalsis\'].map({"NA": 0, "None": 0, "hypermotile": 1, \'distend_small\':1, \n                               "normal": 2,"hypomotile": 3, "absent": 4}\n                             ).astype(np.int8);\n        df[\'abdominal_distention\'] = \\\n        df[\'abdominal_distention\'].map({"NA": 0, "none": 1, "slight": 2, "moderate": 3, "severe": 4}).astype(np.int8);\n        df[\'nasogastric_tube\'] = \\\n        df[\'nasogastric_tube\'].map({"NA": 0, "none": 1, "slight": 2, "significant": 3}).astype(np.int8);\n        df[\'nasogastric_reflux\'] = \\\n        df[\'nasogastric_reflux\'].map({"NA": 0, "none": 1, \'slight\':2, "less_1_liter": 2, "more_1_liter": 3}).astype(np.int8);\n        df[\'rectal_exam_feces\'] = \\\n        df[\'rectal_exam_feces\'].map({"NA": 0, "None": 0, \n                                     "normal": 1, "increased": 3, "decreased": 4, "absent": 5, \'serosanguious\':6}\n                                   ).astype(np.int8);\n        df[\'abdomen\'] = \\\n        df[\'abdomen\'].map({"NA": 0, "None":0, \n                           "normal": 1, "other": 2, "firm": 3, "distend_small": 4, "distend_large": 5}\n                         ).astype(np.int8);\n        df[\'abdomo_appearance\'] = \\\n        df[\'abdomo_appearance\'].map({"NA": 0, "None":0,"clear": 1, "cloudy": 2, "serosanguious": 3}).astype(np.int8);\n        df[\'surgical_lesion\'] = df[\'surgical_lesion\'].map({"no": 1, "yes": 0}).astype(np.int8);\n        df[\'cp_data\'] = df[\'cp_data\'].map({"no": 1, "yes": 0}).astype(np.int8);\n        \n        #  Encoding the lesion 2 as 0/ non-zero:-\n        df[\'lesion_2\'] = df[\'lesion_2\'].clip(0, 1).astype(np.int8);\n        \n        df[[\'hospital_number\', \'lesion_1\', \'pain\']] = \\\n        df[[\'hospital_number\', \'lesion_1\', \'pain\']].astype(int);\n                  \n        self.op_cols = df.columns; \n        return df;\n    \n    def get_feature_names_in(self, X, y=None, **params): \n        return self.ip_cols;    \n    \n    def get_feature_names_out(self, X, y=None, **params): \n        return self.op_cols;       \n    \n\ncollect();\nprint();\n')


# In[25]:


get_ipython().run_cell_magic('time', '', '\n# Scaling:-\nclass Scaler(TransformerMixin, BaseEstimator):\n    """\n    This class aims to create scaling for the provided dataset\n    """;\n    \n    def __init__(self, scl_method: str, scale_req: str, scl_cols):\n        self.scl_method = scl_method;\n        self.scale_req  = scale_req;\n        self.scl_cols   = scl_cols;\n        \n    def fit(self,X, y=None, **params):\n        "This function calculates the train-set parameters for scaling";\n        \n        self.params          = X[self.scl_cols].describe(percentiles = [0.25, 0.50, 0.75]).drop([\'count\'], axis=0).T;\n        self.params[\'iqr\']   = self.params[\'75%\'] - self.params[\'25%\'];\n        self.params[\'range\'] = self.params[\'max\'] - self.params[\'min\'];\n        \n        return self;\n    \n    def transform(self,X, y=None, **params):  \n        "This function transform the relevant scaling columns";\n        \n        df = X.copy();\n        if self.scale_req == "Y":\n            if CFG.scl_method == "Z":\n                df[self.scl_cols] = (df[self.scl_cols].values - self.params[\'mean\'].values) / self.params[\'std\'].values;\n            elif CFG.scl_method == "Robust":\n                df[self.scl_cols] = (df[self.scl_cols].values - self.params[\'50%\'].values) / self.params[\'iqr\'].values;\n            elif CFG.scl_method == "MinMax":\n                df[self.scl_cols] = (df[self.scl_cols].values - self.params[\'min\'].values) / self.params[\'range\'].values;\n        else:\n            PrintColor(f"Scaling is not needed", color = Fore.RED);\n    \n        return df;\n    \n')


# In[26]:


get_ipython().run_cell_magic('time', '', '\nPrintColor(f"\\n{\'=\'* 20} Data transformation {\'=\'* 20} \\n");\n\n# Dropping extra train elements:-\ndrop_idx = \\\nlist(set(train.loc[(train[\'hospital_number\'].isin(hnb_only_tr))].index.to_list()));\n\n# Implementing the pipeline:-\nif CFG.drop_tr_idx == "Y":\n    PrintColor(f"---> Train shape before dropping extra indices = {train.shape}");\n    Xtrain = \\\n    train.drop(CFG.target, axis=1, errors = \'ignore\').\\\n    drop(drop_idx, axis=0,errors = \'ignore\');\n    ytrain = train[CFG.target].map(CFG.tgt_mapper).astype(np.int8).drop(drop_idx, axis=0,errors = \'ignore\');\n    PrintColor(f"---> Train shape after dropping extra indices = {Xtrain.shape} {ytrain.shape}");\n    \n    Xtrain.index = range(len(Xtrain));\n    ytrain.index = range(len(ytrain));\n    \nelse:\n    Xtrain, ytrain = train.drop(CFG.target, axis=1, errors = \'ignore\'), train[CFG.target].map(CFG.tgt_mapper).astype(np.int8);\n\n# Transforming the data:-\nsteps = [("Imp", ColumnTransformer([("CImp", SI(strategy = "most_frequent"), cat_cols.to_list() + [\'Source\'])],\n                                   remainder = SI(strategy = \'mean\'),\n                                   verbose_feature_names_out = False\n                                  )\n         ),\n         (\'Xform\', Xformer(hnb_only_test, hnb_only_tr)), \n         (\'Enc\', Encoder()),\n         (\'OHE\', OneHotEncoder(cols = CFG.OH_cols, drop_invariant = True, use_cat_names = True))\n        ];\ntry: \n    xform = Pipeline(steps = steps, verbose = False);\n    PrintColor(f"\\n---> Data pipeline is initialized fully without issues");\nexcept: \n    xform = Pipeline(steps = steps[0:-1], verbose = False);\n    PrintColor(f"\\n---> Data pipeline has an issue in one-hot stage, please check", color = Fore.RED);\n\nPrintColor(f"\\n---> Data pipeline structure\\n");\ndisplay(xform);\n\nPrintColor(f"\\n---> Post pipeline datasets\\n");\nXtrain = xform.fit_transform(Xtrain, ytrain);\nXtest  = xform.transform(test);\n\nXtrain.columns = [re.sub(r"\\.[0-9]*", \'\', col) for col in Xtrain.columns];\nXtest.columns = [re.sub(r"\\.[0-9]*", \'\', col) for col in Xtest.columns];\n\nPrintColor(f"\\n---> Train data\\n");\ndisplay(Xtrain.head(5).style.format(precision = 2));\nPrintColor(f"\\n---> Test data\\n");\ndisplay(Xtest.head(5).style.format(precision = 2));\n\nPrintColor(f"\\n---> Train data columns after data pipeline\\n");\npprint(Xtrain.columns);\n\nPrintColor(f"\\n---> Test data columns after data pipeline\\n");\npprint(Xtest.columns);\n\nPrintColor(f"\\n---> Train-test shape after pipeline = {Xtrain.shape} {Xtest.shape}");\n\nprint();\ncollect();\n')


# <a id="7"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:black; font-size:120%; text-align:left;padding:3.0px; background: #cceeff; border-bottom: 8px solid #004466" > MODEL TRAINING <br><div> 
#    

# In[27]:


get_ipython().run_cell_magic('time', '', '\n# Initializing model I-O:-\n\nMdl_Master = \\\n{\'CBC\': CatBoostClassifier(**{\'task_type\'           : "GPU" if CFG.gpu_switch == "ON" else "CPU",\n                              \'objective\'           : "MultiClass",\n                              \'eval_metric\'         : "Accuracy",\n                              \'classes_count\'       : 3,\n                              \'bagging_temperature\' : 0.10,\n                              \'colsample_bylevel\'   : 0.75,\n                              \'iterations\'          : 1000,\n                              \'learning_rate\'       : 0.075,\n                              \'od_wait\'             : 3,\n                              \'max_depth\'           : 4,\n                              \'l2_leaf_reg\'         : 0.85,\n                              \'min_data_in_leaf\'    : 6,\n                              \'random_strength\'     : 0.65, \n                              \'max_bin\'             : 80,\n                              \'verbose\'             : 0,\n                              \'use_best_model\'      : True,\n                           }\n                         ), \n\n  \'LGBMC\': LGBMClassifier(**{\'device\'            : "gpu" if CFG.gpu_switch == "ON" else "cpu",\n                             \'objective\'         : \'multiclass\',\n                             \'metric\'            : \'none\',\n                             \'boosting_type\'     : \'gbdt\',\n                             \'random_state\'      : CFG.state,\n                             \'colsample_bytree\'  : 0.5,\n                             \'subsample\'         : 0.65,\n                             \'learning_rate\'     : 0.08,\n                             \'max_depth\'         : 4,\n                             \'n_estimators\'      : 1000,\n                             \'num_leaves\'        : 72,                    \n                             \'reg_alpha\'         : 0.01,\n                             \'reg_lambda\'        : 1.75,\n                             \'verbose\'           : -1,\n                         }\n                      ),\n\n  \'XGBC\': XGBClassifier(**{\'tree_method\'        : "gpu_hist" if CFG.gpu_switch == "ON" else "hist",\n                           \'objective\'          : \'multi:softprob\',\n                           \'random_state\'       : CFG.state,\n                           \'colsample_bytree\'   : 0.7,\n                           \'learning_rate\'      : 0.07,\n                           \'max_depth\'          : 4,\n                           \'n_estimators\'       : 1100,                         \n                           \'reg_alpha\'          : 0.025,\n                           \'reg_lambda\'         : 1.75,\n                           \'min_child_weight\'   : 5,\n                           \'early_stopping_rounds\' : CFG.nbrnd_erly_stp,\n                        }\n                       ),\n \n   \'RFC\' : RFC(n_estimators     = 150, \n               criterion        = \'gini\',\n               max_depth        = 4,\n               min_samples_leaf = 5,\n               max_features     = \'log2\',\n               bootstrap        = True,\n               oob_score        = True,\n               random_state     = CFG.state,\n               verbose          =0,\n              ), \n \n  "HGBC" : HGBC(loss              = \'categorical_crossentropy\',\n                learning_rate     = 0.075,\n                early_stopping    = True,\n                max_iter          = 200,\n                max_depth         = 4,\n                min_samples_leaf  = 5,\n                l2_regularization = 1.75,\n                scoring           = myscorer,\n                random_state      = CFG.state,\n               )\n};\n\nprint();\ncollect();\n')


# In[28]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ML == "Y":\n\n    # Selecting relevant columns for the train and test sets:-\n    PrintColor(f"\\n{\'=\'* 20} Model I-O initialization {\'=\'* 20} \\n");\n\n    drop_cols = [\'rectal_temp_risk\', \'pulse_risk\', \'cell_vol_risk\', \'protein_risk\',\n                 \'lesion_1\', \'lesion_2\', \'lesion_3\',\n                 \'pain_1\', \'lesion_1_5\', \n                 \'lesion_2_0\', \'lesion_2_1400\',\'lesion_2_3111\', \n                 \'lesion_2_3112\', \'lesion_2_7111\', \'lesion_2_6112\', \'lesion_2_4300\', \n                ];\n    print(); \n\n    try: \n        Xtrain, Xtest = \\\n        Xtrain.drop(drop_cols, axis=1,errors = \'ignore\'), \\\n        Xtest.drop(drop_cols, axis=1, errors = \'ignore\');\n\n        PrintColor(f"---> Selected columns for model-");\n        pprint(Xtest.columns, depth = 1, width = 10, indent = 5);\n\n    except: \n        PrintColor(f"\\n---> Check the columns selected\\n---> Selected columns-", color = Fore.RED);\n        pprint(Xtest.columns, depth = 1, width = 10, indent = 5);\n\n    # Initializing output tables for the models:-\n    methods   = list(Mdl_Master.keys());\n    OOF_Preds = pd.DataFrame(columns = [\'Method\'] + [f"Class{i}" for i in range(3)]);\n    Mdl_Preds = pd.DataFrame(index = pp.sub_fl[\'id\'], columns = [f"Class{i}" for i in range(3)],\n                             data = np.zeros((len(Xtest),3))\n                            );\n    FtreImp   = pd.DataFrame(index = Xtrain.drop(columns = [\'Source\'], errors = \'ignore\').columns,\n                             columns = methods,\n                             data = np.zeros((len(Xtrain.drop(columns = [\'Source\'], errors = \'ignore\').columns),\n                                              len(methods)\n                                             )\n                                            )\n                            );\n    Scores = pd.DataFrame(index = range(CFG.n_splits * CFG.n_repeats), \n                          columns = methods + [\'OOF\', "Train"]);\n\n    PrintColor(f"\\n---> Selected model options- ");\n    pprint(methods, depth = 1, width = 100, indent = 5);\n    \n    cat_ftre  = [c for c in Xtest.columns if c not in cont_cols.to_list() + [\'Source\']];\n    PrintColor(f"\\n---> Selected category columns- ");\n    pprint(np.array(cat_ftre), depth = 1, width = 100, indent = 5);  \n\nprint();\ncollect();\n')


# 
#    

# In[29]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ML == "Y":\n    PrintColor(f"\\n{\'=\'* 20} Model Training and CV {\'=\'* 20} \\n");\n    \n    cols_drop = [\'id\', \'Source\', \'Label\'];\n    cv        = all_cv.get(CFG.mdlcv_mthd);\n    Xt        = Xtest.copy(deep = True);\n    X,y       = Xtrain.copy(deep = True), ytrain.copy(deep = True);\n             \n    # Initializing CV splitting:-       \n    for fold_nb, (train_idx, dev_idx) in tqdm(enumerate(cv.split(X, y)), \n                                              f"{CFG.mdlcv_mthd} CV {CFG.n_splits}x{CFG.n_repeats}"\n                                             ): \n        Xtr  = X.iloc[train_idx].drop(columns = cols_drop, errors = \'ignore\');   \n        Xdev = X.iloc[dev_idx].drop(columns = cols_drop, errors = \'ignore\'); \n        ytr  = y.loc[y.index.isin(Xtr.index)];\n        ydev = y.loc[y.index.isin(Xdev.index)];\n        \n        oof_preds   = np.zeros((len(Xdev), 3));\n        mdl_preds   = np.zeros((len(Xt), 3));\n        train_preds = np.zeros((len(Xtr), 3));\n       \n        # Fitting the models:- \n        for method in methods:     \n            if method in [\'LC\', "SVC"]:\n                model = Pipeline(steps = [("S", all_scalers[CFG.scl_method]),\n                                          ("M", Mdl_Master[method])]\n                                );\n                model.fit(Xtr, ytr);\n                                                  \n            if method in [\'CBR\', \'CBC\']: \n                model = Pipeline(steps = [(\'M\', Mdl_Master[method])]);\n                model.fit(Xtr, ytr, \n                          M__eval_set = [(Xdev, ydev)], \n                          M__verbose = 0,\n                          M__cat_features = cat_ftre,\n                          M__early_stopping_rounds = CFG.nbrnd_erly_stp,\n                         ); \n\n            elif method in [\'LGBMR\', \'LGBMC\']: \n                model = Pipeline(steps = [(\'M\', Mdl_Master[method])]);\n                model.fit(Xtr, ytr, \n                          M__eval_set = [(Xdev, ydev)],\n                          M__eval_metric = ScoreLGBM,\n                          M__verbose = -1,\n                          M__categorical_feature = cat_ftre,\n                          M__callbacks = [log_evaluation(0), \n                                          early_stopping(CFG.nbrnd_erly_stp, verbose = 0)\n                                         ], \n                         );\n\n            elif method in [\'XGBR\', \'XGBC\']: \n                model = Pipeline(steps = [(\'M\', Mdl_Master[method])]);                                 \n                model.fit(Xtr, ytr, \n                          M__eval_set = [(Xdev, ydev)], \n                          M__verbose = 0,\n                          M__eval_metric = ScoreXGB,\n                         );            \n\n            else: \n                model = Pipeline(steps = [(\'M\', Mdl_Master[method])]); \n                model.fit(Xtr, ytr); \n                \n            # Collecting predictions and scores and post-processing OOF based on model method:-  \n            dev_preds   = model.predict_proba(Xdev);\n            oof_preds   = oof_preds + dev_preds;\n            mdl_preds   = mdl_preds   + model.predict_proba(Xt.drop(columns = cols_drop, errors = \'ignore\'));\n            train_preds = train_preds + model.predict_proba(Xtr);\n            \n            try: \n                FtreImp[method] = FtreImp[method] + model[\'M\'].feature_importances_;\n            except: \n                pass;\n            \n            # Integrating the scores:-\n            Scores.at[fold_nb, method] = ScoreMetric(ydev, np.argmax(dev_preds, axis=1));\n            dev_preds = \\\n            pd.DataFrame(dev_preds, index = Xdev.index, columns = [f"Class{i}" for i in range(3)]);\n            dev_preds.insert(0, \'Method\', method); \n            OOF_Preds = pd.concat([OOF_Preds, dev_preds], axis = 0,ignore_index = False);\n            del dev_preds;\n            \n        Mdl_Preds = Mdl_Preds + mdl_preds;      \n        \n        # Calculating the fold-level score metric:-        \n        Scores.at[fold_nb, "OOF"]   = ScoreMetric(ydev, np.argmax(oof_preds,   axis=1));\n        Scores.at[fold_nb, "Train"] = ScoreMetric(ytr, np.argmax(train_preds, axis=1));\n        collect(); \n    \n    for col in range(3): \n        Mdl_Preds[f"Class{col}"] = Mdl_Preds[f"Class{col}"]/ (CFG.n_splits * CFG.n_repeats * len(methods));\n     \n    clear_output();\n    PrintColor(f"\\n{\'=\'* 20} CV results {\'=\'* 20} \\n");\n    Scores.index.name = "CVFolds";\n    \n    display(Scores.\\\n            style.\\\n            format(formatter = "{:.5f}").\\\n            background_gradient(cmap = LCM(sns.color_palette("Pastel1",\n                                                             desat = 0.9,\n                                                             n_colors = len(methods)\n                                                            )), \n                                subset = methods\n                               ).\\\n            highlight_max(subset = [\'OOF\', "Train"], \n                          props = """font-weight: bold; \n                          font-family: Arial; \n                          color : #1a1aff;\n                          font-size: 110%; \n                          background-color:  #adebeb;\n                          border: 3px solid #004466\n                          """\n                         ).\\\n            highlight_min(subset = [\'OOF\', "Train"], \n                          props = """font-weight: bold; \n                          font-family: Arial; \n                          color : #e64d00;\n                          font-size: 110%; \n                          background-color:  #ffff66; \n                          border: 3px solid #004466\n                          """\n                         )          \n           );\n    \n    PrintColor(f"\\n---> Mean-Stddev OOF CV = {Scores[\'OOF\'].mean(): .5f} +- {Scores[\'OOF\'].std(): .5f}");\n    PrintColor(f"---> Mean-Stddev Train CV = {Scores[\'Train\'].mean(): .5f} +- {Scores[\'Train\'].std(): .5f}\\n");\n           \ncollect();\nprint();\n')


# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > INFERENCES<br> <div>

# <div style= "font-family: Cambria; letter-spacing: 0px; color:#000000; font-size:110%; text-align:left;padding:3.0px; background: #f2f2f2" >
# 1. Considering the distinct differences between the training and OOF set, we can gather that we have some overfitting risk<br>
# 2. We have to drop some features to ensue better CV results<br>
# 3. We need to fine-tune some model parameters too<br>
# 4. We could try other model methods too, perhaps some simpler ones will be good enough <br>
# </div>

# In[30]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ML == "Y":\n\n    # Analyzing the Ml model results:-\n    fig, axes = plt.subplots(len(methods), 1, figsize = (25, len(methods)* 5), \n                             sharex = True, gridspec_kw= {\'hspace\': 0.3}\n                            );\n    for i, method in enumerate(methods):\n        ax = axes[i];\n        FtreImp[method].plot.bar(ax = ax, color = \'tab:blue\');\n        ax.set_title(f"Feature Importance - {method}", **CFG.title_specs);\n        ax.set(xlabel = \'\', ylabel = \'\');\n\n    plt.xticks(rotation = 45);\n    plt.tight_layout();\n    plt.show();\n\n    # Plotting the confusion matrix with the results:-\n    fig, ax = plt.subplots(1,1, figsize = (3,3));\n    df = \\\n    OOF_Preds.iloc[:, 1:].groupby(level = 0).mean().\\\n    idxmax(axis=1).\\\n    apply(lambda x: x[-1]).astype(np.int8);\n    \n    sns.heatmap(confusion_matrix(ytrain.iloc[0: len(df)], df),\n                cbar = None, annot= True, fmt = \'.0f\',\n                annot_kws= {\'fontweight\': \'bold\',\'fontsize\': 8.5},\n                cmap = \'icefire\', linewidths = 3.5, ax = ax\n               );\n    ax.set_title(f"\\nConfusion matrix\\n", **CFG.title_specs);\n    plt.yticks(rotation = 0);\n    plt.show();\n    \ncollect();\nprint();\n')


# <a id="8"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:black; font-size:120%; text-align:left;padding:3.0px; background: #cceeff; border-bottom: 8px solid #004466" > SUBMISSION<br> <div> 

# In[31]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ML == "Y":\n    sub_fl = test[[\'hospital_number\', \'lesion_1\', \'lesion_2\']];\n    test.drop(columns = [CFG.target], inplace = True, errors = "ignore");\n\n    # Mapping the model predictions to the curated submission file:-\n    sub_fl[CFG.target] = \\\n    Mdl_Preds.idxmax(axis=1).apply(lambda x: x[-1]).astype(np.int8).values;\n    sub_fl[CFG.target] = \\\n    sub_fl[CFG.target].map({k: v for k, v in zip(CFG.tgt_mapper.values(), CFG.tgt_mapper.keys())}).values;\n\n    PrintColor(f"\\n---> Test set prediction counts from the model");\n    pprint(Counter(sub_fl[CFG.target]));\n\n    # Creating submission based post-processing:-\n    for label, idx in HNB_tgt.items():\n        sub_fl.loc[sub_fl[\'hospital_number\'].isin(idx.to_list()), f"{CFG.target}_pp"] = label;\n\n    sub_fl[f"{CFG.target}_pp"].fillna(sub_fl[f"{CFG.target}"], inplace = True);\n\n    PrintColor(f"\\nTest set prediction counts after processing");\n    pprint(Counter(sub_fl[f"{CFG.target}_pp"]));\n    \n    if CFG.pstprcs_test == "Y":\n        sub_fl[[f"{CFG.target}_pp"]].\\\n        reset_index().\\\n        rename({f"{CFG.target}_pp": CFG.target}, axis=1).\\\n        to_csv(f"Submission_V{CFG.version_nb}.csv", index = None);\n        \n        PrintColor(f"\\n---> Test set prediction file");\n        display(sub_fl[[f"{CFG.target}_pp"]].head(10));\n                \n    else:\n        sub_fl[[CFG.target]].\\\n        reset_index().\\\n        to_csv(f"Submission_V{CFG.version_nb}.csv", index = None);   \n        \n        PrintColor(f"\\n---> Test set prediction file");\n        display(sub_fl[[f"{CFG.target}"]].head(10));     \n    \ncollect();\nprint();\n')


# <a id="9"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0052cc; border-bottom: 8px solid #cc9966" > NEXT STEPS<br> <div> 

# <div style= "font-family: Cambria; letter-spacing: 0px; color:#000000; font-size:110%; text-align:left;padding:3.0px; background: #f2f2f2" >
# 1. Better feature engineering- encoding is key here along with feature reduction<br>
# 2. Try better CV oriented tuning<br>
# </div>
