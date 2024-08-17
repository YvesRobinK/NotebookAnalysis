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

# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# Installing select libraries:-\nfrom gc import collect;\nfrom warnings import filterwarnings;\nfilterwarnings('ignore');\nfrom IPython.display import clear_output;\n\n!pip install -q --upgrade scipy;\n!pip install -q category_encoders;\n!pip install -q pygwalker\n\nclear_output();\nprint();\ncollect();\n")


# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# General library imports:-\nfrom copy import deepcopy;\nimport pandas as pd;\nimport numpy as np;\nfrom scipy.stats import mode, kstest, normaltest, shapiro, anderson, jarque_bera;\nfrom collections import Counter;\nfrom itertools import product;\nfrom colorama import Fore, Style, init;\nfrom warnings import filterwarnings;\nfilterwarnings('ignore');\n\nfrom tqdm.notebook import tqdm;\nimport seaborn as sns;\nimport matplotlib.pyplot as plt;\n%matplotlib inline\nimport pygwalker as pyg;\nfrom pprint import pprint;\n\nprint();\ncollect();\nclear_output();\n")


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Importing model and pipeline specifics:-\nfrom category_encoders import OrdinalEncoder, OneHotEncoder;\n\n# Pipeline specifics:-\nfrom sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler;\nfrom sklearn.decomposition import PCA;\nfrom sklearn.model_selection import (RepeatedStratifiedKFold as RSKF, \n                                     StratifiedKFold as SKF,\n                                     KFold, \n                                     RepeatedKFold as RKF, \n                                     cross_val_score);\nfrom sklearn.inspection import permutation_importance, PartialDependenceDisplay as PDD;\nfrom sklearn.feature_selection import mutual_info_classif, RFE;\nfrom sklearn.pipeline import Pipeline, make_pipeline;\nfrom sklearn.base import BaseEstimator, TransformerMixin;\nfrom sklearn.compose import ColumnTransformer;\n\n# ML Model training:-\nfrom sklearn.calibration import CalibrationDisplay as Clb;\nfrom sklearn.metrics import roc_auc_score;\nfrom sklearn.svm import SVC;\nfrom xgboost import XGBClassifier, XGBRegressor;\nfrom lightgbm import LGBMClassifier, LGBMRegressor, log_evaluation;\nfrom catboost import CatBoostRegressor, CatBoostClassifier;\nfrom sklearn.ensemble import (RandomForestRegressor as RFR,\n                              ExtraTreesRegressor as ETR,\n                              GradientBoostingRegressor as GBR,\n                              HistGradientBoostingRegressor as HGBR,\n                              RandomForestClassifier as RFC,\n                              ExtraTreesClassifier as ETC,\n                              GradientBoostingClassifier as GBC,\n                              HistGradientBoostingClassifier as HGBC,\n                             );\nfrom sklearn.linear_model import LogisticRegression as LC;\n\n# Ensemble and tuning:-\nimport optuna;\nfrom optuna import Trial, trial, create_study;\nfrom optuna.samplers import TPESampler, CmaEsSampler;\noptuna.logging.set_verbosity = optuna.logging.ERROR;\n\nclear_output();\nprint();\ncollect();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Setting rc parameters in seaborn for plots and graphs- \n# Reference - https://matplotlib.org/stable/tutorials/introductory/customizing.html:-\n# To alter this, refer to matplotlib.rcParams.keys()\n\nsns.set({"axes.facecolor"       : "#ffffff",\n         "figure.facecolor"     : "#ffffff",\n         "axes.edgecolor"       : "#000000",\n         "grid.color"           : "#ffffff",\n         "font.family"          : [\'Cambria\'],\n         "axes.labelcolor"      : "#000000",\n         "xtick.color"          : "#000000",\n         "ytick.color"          : "#000000",\n         "grid.linewidth"       : 0.75,  \n         "grid.linestyle"       : "--",\n         "axes.titlecolor"      : \'#0099e6\',\n         \'axes.titlesize\'       : 8.5,\n         \'axes.labelweight\'     : "bold",\n         \'legend.fontsize\'      : 7.0,\n         \'legend.title_fontsize\': 7.0,\n         \'font.size\'            : 7.5,\n         \'xtick.labelsize\'      : 7.5,\n         \'ytick.labelsize\'      : 7.5,        \n        });\n\n# Color printing    \ndef PrintColor(text:str, color = Fore.BLUE, style = Style.BRIGHT):\n    "Prints color outputs using colorama using a text F-string";\n    print(style + color + text + Style.RESET_ALL); \n\n# Making sklearn pipeline outputs as dataframe:-\nfrom sklearn import set_config; \nset_config(transform_output = "pandas");\npd.set_option(\'display.max_columns\', 50);\npd.set_option(\'display.max_rows\', 50);\n\nprint();\ncollect();\n')


# <a id="2"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:black; font-size:120%; text-align:left;padding:3.0px; background: #cceeff; border-bottom: 8px solid #004466" > INTRODUCTION<br><div> 

# | Version<br>Number | Version Details | Best CV score| Single/ Ensemble|LB score|
# | :-: | --- | :-: | :-: |:-:|
# | **V1** |* EDA, plots and secondary features<br>* No scaling<br> * Used original data<br>* Tree based ML models and Optuna ensemble<br>* Introduced pygwalker|0.647882|Ensemble<br> Optuna |0.64466|
# | **V2** |* EDA, plots and secondary features<br>* No scaling<br> * Used original data<br>* Selected lesser features for the model<br>* Tree based ML models and Optuna ensemble||Ensemble<br> Optuna ||

# <a id="2.1"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > CONFIGURATION<br><div> 

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Configuration class:-\nclass CFG:\n    "Configuration class for parameters and CV strategy for tuning and training";\n    \n    # Data preparation:-   \n    version_nb         = 2;\n    test_req           = "N";\n    gpu_switch         = "OFF"; \n    state              = 42;\n    target             = [\'EC1\', \'EC2\'];\n    episode            = 18;\n    path               = f"/kaggle/input/playground-series-s3e{episode}/";\n    orig_path          = f"/kaggle/input/ec-mixed-class/";\n    \n    dtl_preproc_req    = "Y";\n    adv_cv_req         = "Y";\n    ftre_plots_req     = "Y";\n    ftre_imp_req       = "Y";\n    \n    # Data transforms and scaling:-    \n    conjoin_orig_data  = "Y";\n    sec_ftre_req       = "Y";\n    scale_req          = "N";\n    # NOTE---Keep a value here even if scale_req = N, this is used for linear models:-\n    scl_method         = "Z"; \n    enc_method         = \'Label\';\n    ncomp              = 3;\n    \n    # Model Training:- \n    baseline_req       = "N";\n    pstprcs_oof        = "Y";\n    pstprcs_train      = "Y";\n    ML                 = "Y";\n    use_orig_allfolds  = "N";\n    n_splits           = 5 ;\n    n_repeats          = 1 ;\n    nbrnd_erly_stp     = 200 ;\n    mdlcv_mthd         = \'RSKF\';\n    \n    # Ensemble:-    \n    ensemble_req       = "Y";\n    enscv_mthd         = "RSKF";\n    metric_obj         = \'maximize\';\n    ntrials            = 10 if test_req == "Y" else 250;\n    \n    # Global variables for plotting:-\n    grid_specs = {\'visible\': True, \'which\': \'both\', \'linestyle\': \'--\', \n                           \'color\': \'lightgrey\', \'linewidth\': 0.75};\n    title_specs = {\'fontsize\': 9, \'fontweight\': \'bold\', \'color\': \'tab:blue\'};\n\nprint();\nPrintColor(f"--> Configuration done!\\n");\ncollect();\n')


# In[ ]:


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
# |  dtl_preproc_req  | Proprocessing required                                  | Y/N                   |    
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
# **Reference** - <br>
# https://www.kaggle.com/competitions/playground-series-s3e18/discussion/419646 <br>
# https://www.kaggle.com/code/kimtaehun/multi-label-classification-with-complete-eda >br>
# 
# 
# - **BertzCT**: Bertz counter (Topological Charge Transfer) value associated with the structural complexity of the molecule.
# - **Chi1, Chi1n, Chi1v**: Kier's First-order Molecular Connectivity Index, representing the molecular surface area.
# - **Chi2n, Chi2v, Chi3v**: Kier's Second-order and Third-order Connectivity Index, representing the 2nd and 3rd degree connectivity of the molecule.
# - **Chi4n**: Kier's Fourth-order Connectivity Index, representing the 4th degree connectivity of the molecule.
# - **EState_VSA1, EState_VSA2**: EState-VSA (E-State Value Sum) 1 and 2, representing the electrotopological state contributions of the molecule.
# - **ExactMolWt**: Exact molecular weight of the molecule.
# - **FpDensityMorgan1, FpDensityMorgan2, FpDensityMorgan3**: Density values of the Morgan fingerprint.
# - **HallKierAlpha**: Hall-Kier Alpha value of the molecule, describing its core structure.
# - **HeavyAtomMolWt**: Molecular weight of atoms in the molecule, excluding hydrogen atoms.
# - **Kappa3**: Kappa-3 shape index of the molecule, describing its topology.
# - **MaxAbsEStateIndex**: Maximum absolute E-State index of the molecule, representing its maximum charge state.
# - **MinEStateIndex**: Minimum E-State index of the molecule, representing its minimum charge state.
# - **NumHeteroatoms**: Number of heteroatoms (non-carbon atoms) in the molecule.
# - **PEOE_VSA10, PEOE_VSA14, PEOE_VSA6, PEOE_VSA7, PEOE_VSA8**: PEOE (Partial Equalization of Orbital Electronegativity) - - VSA (Value Sum) 10, 14, 6, 7, 8, representing the partial charge surface area contributions of the molecule.
# - **SMR_VSA10, SMR_VSA5**: SMR (Simple Molecular Representation) VSA 10, 5, representing the similarity model radius surface area contributions of the molecule.
# - **SlogP_VSA3**: SlogP (Substructure-logP) VSA 3, representing the logarithm of the partition coefficient of the molecule.
# - **VSA_EState9**: E-State-VSA (E-State Value Sum) 9, representing the electrotopological state contributions of the molecule related to its surface area.
# - **fr_COO, fr_COO2**: Number of carboxyl groups in the molecule.
# 
# **Competition details and notebook objectives**<br>
# 1. This is a multi-label classification challenge to predict enzyme classes using the provided features. **GINI** is the metric for the challenge<br>
# 2. In this starter notebook, we start the assignment with a detailed EDA, feature plots, interaction effects, adversarial CV analysis and develop starter models to initiate the challenge. We will also incorporate other opinions and approaches as we move along the challenge.<br>
# 3. **Note:-** In this challenge, we could have a set of features being mapped to 2 classes, hence the name **multi-label classifier**. For multi-class, we need to exclusively classify features into 1 among many class options<br>

# <a id="3"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:black; font-size:120%; text-align:left;padding:3.0px; background: #cceeff; border-bottom: 8px solid #004466" > PREPROCESSING<br><div> 

# In[ ]:


get_ipython().run_line_magic('time', '')

class Preprocessor():
    """
    This class aims to do the below-
    1. Read the datasets
    2. In this case, process the original data
    3. Check information and description
    4. Check unique values
    5. Collate starting features 
    6. Conjoin train-original data if requested based on Adversarial CV results
    """;
    
    def __init__(self):
        self.train    = pd.read_csv(CFG.path + f"train.csv", index_col = 'id');
        self.test     = pd.read_csv(CFG.path + f"test.csv", index_col = 'id');
        self.targets  = self.train.columns[self.train.columns.str.startswith("EC")].tolist();
        self.test_req = CFG.test_req;
        self.dtl_preproc_req = CFG.dtl_preproc_req;
        self.conjoin_orig_data = CFG.conjoin_orig_data;
        
        self.sub_fl   = pd.read_csv(CFG.path + f"sample_submission.csv");
        
        PrintColor(f"Data shape - train-test = {self.train.shape} {self.test.shape}");
        
        PrintColor(f"\nTrain set head", color = Fore.GREEN);
        display(self.train.head(5).style.format(precision = 3));
        PrintColor(f"\nTest set head", color = Fore.GREEN);
        display(self.test.head(5).style.format(precision = 3));
             
    def _ProcessOrig(self):
        o1 = pd.read_csv(CFG.orig_path + f"mixed_desc.csv", index_col = 'CIDs');
        o2 = pd.read_csv(CFG.orig_path + f"mixed_ecfp.csv", index_col = 'CIDs');
        o3 = pd.read_csv(CFG.orig_path + f"mixed_fcfp.csv", index_col = 'CIDs');
        
        self.original = pd.concat([o1.iloc[:, 0:-1], o2.iloc[:, 0:-1], o3], axis=1);
        _ = self.original[o3.columns[-1]].str.split('_', expand = True).astype(np.int8);
        _.columns = ['EC1', 'EC2', 'EC3', 'EC4', 'EC5', 'EC6'];
        self.original = pd.concat([self.original.drop(o3.columns[-1], axis=1), _], 
                                  axis = 1)[self.train.columns];
        del o1, o2, o3, _;
        
        # Resetting original data index:-
        self.original.index = range(len(self.original));
        self.original.index+= max(self.test.index) + 1;
        self.original.index.name = 'id';
        PrintColor(f"\nOriginal data shape -- {self.original.shape}");
        return self;
    
    def _SampleData(self):
        if self.test_req == "Y":
            PrintColor(f"---> We are testing the code with 5% data sample", color = Fore.RED);
            self.train     = self.train.groupby(CFG.target).sample(frac = 0.05);
            self.original  = self.original.groupby(CFG.target).sample(frac = 0.05);
            self.test      = self.test.sample(frac = 0.05);
            self.sub_fl    = self.sub_fl.loc[self.sub_fl.id.isin(self.test.index)];
        
        return self;
    
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
    
    def _CollateUnqNorm(self):
        if self.dtl_preproc_req == "Y": 
            
            PrintColor(f"\n{'-'*20} Unique values and normality tests {'-'*20}\n", color = Fore.MAGENTA);

            # Dislaying the unique values across train-test-original:-
            PrintColor(f"\nUnique values\n");
            _ = pd.concat([self.train[self.strt_ftre].nunique(), 
                           self.test[self.strt_ftre].nunique(), 
                           self.original[self.strt_ftre].nunique()], 
                          axis=1);
            _.columns = ['Train', 'Test', 'Original'];

            display(_.T.style.background_gradient(cmap = 'Blues', axis=1).\
                    format(formatter = '{:,.0f}')
                   );

            # Normality check:-
            cols = list(self.strt_ftre[0:-1]);
            
            for lbl, test_lbl in {"Shapiro": shapiro, "NormalTest": normaltest}.items():
                PrintColor(f"\n{lbl} normality analysis\n");
                pprint({col: [np.round(test_lbl(self.train[col]).pvalue,decimals = 4), 
                              np.round(test_lbl(self.test[col]).pvalue,4) if col != CFG.target else np.NaN,
                              np.round(test_lbl(self.original[col]).pvalue,4)] for col in cols
                       }, indent = 5, width = 100, depth = 2, compact= True);   

        return self;
       
    def DoPreprocessing(self):
        self._ProcessOrig();
        self._SampleData();
        self._AddSourceCol();
        self._CollateInfoDesc();
        self._CollateUnqNorm();
        
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


# In[ ]:


get_ipython().run_cell_magic('time', '', '\npp = Preprocessor();\npp.DoPreprocessing();\n\nprint();\ncollect();\n')


# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > INFERENCES<br> <div>

# <div style= "font-family: Cambria; letter-spacing: 0px; color:#000000; font-size:110%; text-align:left;padding:3.0px; background: #f2f2f2" >
# 1. All the columns are numerical<br>
# 2. We do not have any nulls in the data<br>
# 3. All columns are non-normal<br>
# 4. The synthetic data is nearly 24.6 times the original data, creating a potential quasi-duplicate row issue. Duplicate handling could be a key challenge in this case<br>
# 5. This is a multi-label classifier with 6 targets, we need only 2 for the challenge, are the rest superfluous?<br>
# 6. fr_coo and fr_coo2 appear to be categorical columns<br>
# </div>

# <a id="4"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:black; font-size:120%; text-align:left;padding:3.0px; background: #cceeff; border-bottom: 8px solid #004466" > ADVERSARIAL CV<br><div>

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Performing adversarial CV between the 2 specified datasets:-\ndef Do_AdvCV(df1:pd.DataFrame, df2:pd.DataFrame, source1:str, source2:str):\n    "This function performs an adversarial CV between the 2 provided datasets if needed by the user";\n    \n    # Adversarial CV per column:-\n    ftre = pp.test.select_dtypes(include = np.number).\\\n    drop(columns = [\'id\', "Source"], errors = \'ignore\').columns;\n    adv_cv = {};\n\n    for col in ftre:\n        shuffle_state = np.random.randint(low = 10, high = 100, size= 1);\n\n        full_df = \\\n        pd.concat([df1[[col]].assign(Source = source1), df2[[col]].assign(Source = source2)], \n                  axis=0, ignore_index = True).\\\n        sample(frac = 1.00, random_state = shuffle_state);\n\n        full_df = full_df.assign(Source_Nb = full_df[\'Source\'].eq(source2).astype(np.int8));\n\n        # Checking for adversarial CV:-\n        model = LGBMClassifier(random_state = CFG.state, max_depth = 6, learning_rate = 0.05);\n        cv    = all_cv[\'SKF\'];\n        score = np.mean(cross_val_score(model, \n                                        full_df[[col]], \n                                        full_df.Source_Nb, \n                                        scoring= \'roc_auc\', \n                                        cv     = cv)\n                       );\n        adv_cv.update({col: round(score, 4)});\n        collect();\n    \n    del ftre;\n    collect();\n    \n    fig, ax = plt.subplots(1,1,figsize = (12, 5));\n    pd.Series(adv_cv).plot.bar(color = \'tab:blue\', ax = ax);\n    ax.axhline(y = 0.60, color = \'red\', linewidth = 2.75);\n    ax.grid(**CFG.grid_specs); \n    plt.yticks(np.arange(0.0, 0.81, 0.05));\n    plt.show();\n    \n# Implementing the adversarial CV:-\nif CFG.adv_cv_req == "Y":\n    PrintColor(f"\\n---------- Adversarial CV - Train vs Original ----------\\n", \n               color = Fore.MAGENTA);\n    Do_AdvCV(df1 = pp.train, df2 = pp.original, source1 = \'Train\', source2 = \'Original\');\n    \n    PrintColor(f"\\n---------- Adversarial CV - Train vs Test ----------\\n", \n               color = Fore.MAGENTA);\n    Do_AdvCV(df1 = pp.train, df2 = pp.test, source1 = \'Train\', source2 = \'Test\');\n    \n    PrintColor(f"\\n---------- Adversarial CV - Original vs Test ----------\\n", \n               color = Fore.MAGENTA);\n    Do_AdvCV(df1 = pp.original, df2 = pp.test, source1 = \'Original\', source2 = \'Test\');   \n    \nif CFG.adv_cv_req == "N":\n    PrintColor(f"\\nAdversarial CV is not needed\\n", color = Fore.RED);\n    \ncollect();\nprint();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nprint();\ntrain, test, strt_ftre = pp.ConjoinTrainOrig(), pp.test.copy(deep = True), deepcopy(pp.strt_ftre);\ncat_cols  = [\'fr_COO\', \'fr_COO2\'];\ncont_cols = [col for col in strt_ftre if col not in cat_cols + [\'Source\']];\n\nPrintColor(f"\\nCategory columns\\n");\ndisplay(cat_cols);\nPrintColor(f"\\nContinuous columns\\n");\ndisplay(np.array(cont_cols));\nPrintColor(f"\\nAll columns\\n");\ndisplay(strt_ftre);\n\nprint();\ncollect();\n')


# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > INFERENCES<br><div>

# <div style= "font-family: Cambria; letter-spacing: 0px; color:#000000; font-size:110%; text-align:left;padding:3.0px; background: #f2f2f2" >
# 1. Train-test belong to the same distribution, we can perhaps rely on the CV score<br>
# 2. We need to further check the train-original distribution further, adversarial validation results indicate that we can use the original dataset<br>
# </div>

# <a id="5"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:black; font-size:120%; text-align:left;padding:3.0px; background: #cceeff; border-bottom: 8px solid #004466" > VISUALS AND EDA <br><div> 
#  

# <a id="5.2"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > TARGET PLOT<br><div>

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ftre_plots_req == "Y":\n    \n    for tgt in CFG.target:\n        fig, axes = plt.subplots(1,2, figsize = (12, 5), sharey = True, gridspec_kw = {\'wspace\': 0.2});\n\n        for i, df in tqdm(enumerate([pp.train, pp.original]), "Target balance ---> "):\n            ax= axes[i];\n            a = df[tgt].value_counts(normalize = True);\n            _ = ax.pie(x = a , labels = a.index.values, \n                       explode      = [0.0, 0.3], \n                       startangle   = 40, \n                       shadow       = True, \n                       colors       = [\'#3377ff\', \'#66ffff\'], \n                       textprops    = {\'fontsize\': 7, \'fontweight\': \'bold\', \'color\': \'black\'},\n                       pctdistance  = 0.60, \n                       autopct = \'%1.1f%%\'\n                      );\n            df_name = \'Train\' if i == 0 else "Original";\n            _ = ax.set_title(f"\\n{df_name} data- {tgt}\\n", **CFG.title_specs);\n\n        plt.tight_layout();\n        plt.show();\n        \n        \n    \ncollect();\nprint();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Assessing target interactions:-\nif CFG.ftre_plots_req == "Y":\n    fig, axes = plt.subplots(1,2, figsize = (12, 4), gridspec_kw = {\'wspace\': 0.2});\n    \n    for i, (lbl, df) in enumerate({"Train": pp.train, "Original": pp.original}.items()):\n        ax = axes[i];\n        c = [\'#3377ff\', \'#6699cc\'];\n        df.groupby(CFG.target).size().plot.bar(ax = ax, color = c[i]);\n        ax.set_title(f"Target interaction - {lbl} set", **CFG.title_specs);\n        ax.set(xlabel = "");\n        \n    plt.tight_layout();\n    plt.show()\n        \n')


# <a id="5.4"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > CATEGORY COLUMN PLOTS<br><div>

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ftre_plots_req == "Y":\n    fig, axes = plt.subplots(len(cat_cols), 3, figsize = (20, len(cat_cols)* 4.5), \n                             gridspec_kw = {\'wspace\': 0.2, \'hspace\': 0.3});\n\n    for i, col in enumerate(cat_cols):\n        ax = axes[i, 0];\n        a = pp.train[col].value_counts(normalize = True);\n        a.sort_index().plot.barh(ax = ax, color = \'#007399\');\n        ax.set_title(f"{col}_Train", **CFG.title_specs);\n        ax.set_xticks(np.arange(0.0, 0.7, 0.03), \n                      labels = np.round(np.arange(0.0, 0.7, 0.03),2), \n                      rotation = 90);\n        del a;\n\n        ax = axes[i, 1];\n        a = pp.test[col].value_counts(normalize = True);\n        a.sort_index().plot.barh(ax = ax, color = \'#0088cc\');\n        ax.set_title(f"{col}_Test", **CFG.title_specs);\n        ax.set_xticks(np.arange(0.0, 0.7, 0.03), \n              labels = np.round(np.arange(0.0, 0.7, 0.03),2), \n              rotation = 90);\n        del a;\n        \n        ax = axes[i, 2];\n        a = pp.original[col].value_counts(normalize = True);\n        a.sort_index().plot.barh(ax = ax, color = \'#0047b3\');\n        ax.set_title(f"{col}_Original", **CFG.title_specs);\n        ax.set_xticks(np.arange(0.0, 0.7, 0.03), \n              labels = np.round(np.arange(0.0, 0.7, 0.03),2), \n              rotation = 90);\n        del a;       \n\n    plt.tight_layout();\n    plt.show();\n    \nprint();\ncollect();\n')


# <a id="5.5"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > CONTINUOUS COLUMN PLOTS<br><div>

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ftre_plots_req == "Y":\n    df = pd.concat([pp.train[cont_cols].assign(Source = \'Train\'), \n                    pp.test[cont_cols].assign(Source = \'Test\'),\n                    pp.original[cont_cols].assign(Source = "Original")\n                   ], \n                   axis=0, ignore_index = True\n                  );\n    \n    fig, axes = plt.subplots(len(cont_cols), 4 ,figsize = (16, len(cont_cols) * 4.2), \n                             gridspec_kw = {\'hspace\': 0.35, \'wspace\': 0.3, \'width_ratios\': [0.80, 0.20, 0.20, 0.20]});\n    \n    for i,col in enumerate(cont_cols):\n        ax = axes[i,0];\n        sns.kdeplot(data = df[[col, \'Source\']], x = col, hue = \'Source\', \n                    palette = [\'#0039e6\', \'#ff5500\', \'#00b300\'], \n                    ax = ax, linewidth = 2.1\n                   );\n        ax.set_title(f"\\n{col}", **CFG.title_specs);\n        ax.grid(**CFG.grid_specs);\n        ax.set(xlabel = \'\', ylabel = \'\');\n        \n        ax = axes[i,1];\n        sns.boxplot(data = df.loc[df.Source == \'Train\', [col]], y = col, width = 0.25,\n                    color = \'#33ccff\', saturation = 0.90, linewidth = 0.90, \n                    fliersize= 2.25,\n                    ax = ax);\n        ax.set(xlabel = \'\', ylabel = \'\');\n        ax.set_title(f"Train", **CFG.title_specs);\n        \n        ax = axes[i,2];\n        sns.boxplot(data = df.loc[df.Source == \'Test\', [col]], y = col, width = 0.25, fliersize= 2.25,\n                    color = \'#80ffff\', saturation = 0.6, linewidth = 0.90, \n                    ax = ax); \n        ax.set(xlabel = \'\', ylabel = \'\');\n        ax.set_title(f"Test", **CFG.title_specs);\n        \n        ax = axes[i,3];\n        sns.boxplot(data = df.loc[df.Source == \'Original\', [col]], y = col, width = 0.25, fliersize= 2.25,\n                    color = \'#99ddff\', saturation = 0.6, linewidth = 0.90, \n                    ax = ax); \n        ax.set(xlabel = \'\', ylabel = \'\');\n        ax.set_title(f"Original", **CFG.title_specs);\n              \n    plt.suptitle(f"\\nDistribution analysis- continuous columns\\n", **CFG.title_specs, \n                 y = 0.89, x = 0.57);\n    plt.tight_layout();\n    plt.show();\n    \nprint();\ncollect();\n')


# <a id="5.7"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > FEATURE INTERACTION AND UNIVARIATE RELATIONS<br><div>
#     
# We aim to do the below herewith<br>
# 1. Correlation<br>
# 2. Mutual information<br>

# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndef MakeCorrPlot(df: pd.DataFrame, data_label:str, figsize = (30, 9)):\n    """\n    This function develops the correlation plots for the given dataset\n    """;\n    \n    fig, axes = plt.subplots(1,2, figsize = figsize, gridspec_kw = {\'hspace\': 0.2, \'wspace\': 0.1},\n                             sharey = True\n                            );\n    \n    for i, method in enumerate([\'pearson\', \'spearman\']):\n        corr_ = df.drop(columns = [\'id\', \'Source\'], errors = \'ignore\').corr(method = method);\n        ax = axes[i];\n        sns.heatmap(data = corr_,  \n                    annot= True,\n                    fmt= \'.2f\', \n                    cmap = \'Blues\',\n                    annot_kws= {\'fontweight\': \'bold\',\'fontsize\': 6.75}, \n                    linewidths= 1.5, \n                    linecolor=\'white\', \n                    cbar= False, \n                    mask= np.triu(np.ones_like(corr_)),\n                    ax= ax\n                   );\n        ax.set_title(f"\\n{method.capitalize()} correlation- {data_label}\\n", **CFG.title_specs);\n        \n    collect();\n    print();\n\n# Implementing correlation analysis:-\nfor lbl, df in {"Train": pp.train, "Test": pp.test, "Original": pp.original}.items():\n    MakeCorrPlot(df = df, data_label = lbl, figsize = (38, 13));\n\nprint();\ncollect();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nfor tgt in CFG.target:\n    MutInfoSum = {};\n    for i, df in enumerate([pp.train, pp.original]):\n        MutInfoSum.update({\'Train\' if i == 0 else \'Original\': \n                           mutual_info_classif(df[strt_ftre[0:-1]], df[tgt], random_state = CFG.state)\n                          });\n\n    MutInfoSum = pd.DataFrame(MutInfoSum, index = strt_ftre[0:-1]);\n\n    fig, axes = plt.subplots(1,2, figsize = (28, 6), gridspec_kw = {\'wspace\': 0.2});\n    colors = [\'#4080bf\', \'#3377ff\'];\n    for i in range(2):\n        MutInfoSum.iloc[:, i].plot.bar(ax = axes[i], color = colors[i]);\n        axes[i].set_title(f"{MutInfoSum.columns[i]} - Mutual Information -- {tgt}", **CFG.title_specs);\n\n    plt.tight_layout();\n    plt.show();\nprint();\ncollect();\n')


# <a id="5.9"></a>
# ## <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0059b3; border-bottom: 8px solid #e6e6e6" > INFERENCES<br> <div>

# <div style= "font-family: Cambria; letter-spacing: 0px; color:#000000; font-size:110%; text-align:left;padding:3.0px; background: #f2f2f2" >
# 1. Feature selection is a very important part of the assignment. We have lots of features and feature selection will be a differentiator<br>
# 2. Almost all features have outliers. Outlier handling will be another differentiator in this challenge<br>
# 3. All features are non-normal, certain features like Fpdensity and Kappa3 need to be assessed in the next runs<br>
# 4. Should be dump the other EC columns (EC3-EC6)? I am sure we will extract valuable information from these columns<br>
# 5. Columns are highly correlated. Dimensionality reduction will surely help here<br>
# 6. Target interaction is common here, we have a sizable base with 1s in both EC1 and EC2<br>
# </div>

# <a id="6"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:black; font-size:120%; text-align:left;padding:3.0px; background: #cceeff; border-bottom: 8px solid #004466" > DATA TRANSFORMS <br><div> 
#     
# This section aims at creating secondary features, scaling and if necessary, conjoining the competition training and original data tables<br>
# Currently we will leave this and add them later<

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Data transforms:-\nclass Xformer(TransformerMixin, BaseEstimator):\n    """\n    This class is used to create secondary features from the existing data\n    """;\n    \n    def __init__(self): pass\n    \n    def fit(self, X, y= None, **params):\n        self.ip_cols = X.columns;\n        return self;\n    \n    def transform(self, X, y= None, **params):       \n        global strt_ftre;\n        df    = X.copy();      \n      \n        if CFG.sec_ftre_req == "Y":\n            pass\n        if CFG.sec_ftre_req != "Y": \n            PrintColor(f"Secondary features are not required", color = Fore.RED);    \n        \n        self.op_cols = df.columns;  \n        return df;\n    \n    def get_feature_names_in(self, X, y=None, **params): \n        return self.ip_cols;    \n    \n    def get_feature_names_out(self, X, y=None, **params): \n        return self.op_cols;\n     \n# Scaling:-\nclass Scaler(TransformerMixin, BaseEstimator):\n    """\n    This class aims to create scaling for the provided dataset\n    """;\n    \n    def __init__(self, scl_method: str, scale_req: str, scl_cols):\n        self.scl_method = scl_method;\n        self.scale_req  = scale_req;\n        self.scl_cols   = scl_cols;\n        \n    def fit(self,X, y=None, **params):\n        "This function calculates the train-set parameters for scaling";\n        \n        self.params          = X[self.scl_cols].describe(percentiles = [0.25, 0.50, 0.75]).drop([\'count\'], axis=0).T;\n        self.params[\'iqr\']   = self.params[\'75%\'] - self.params[\'25%\'];\n        self.params[\'range\'] = self.params[\'max\'] - self.params[\'min\'];\n        \n        return self;\n    \n    def transform(self,X, y=None, **params):  \n        "This function transform the relevant scaling columns";\n        \n        df = X.copy();\n        if self.scale_req == "Y":\n            if CFG.scl_method == "Z":\n                df[self.scl_cols] = (df[self.scl_cols].values - self.params[\'mean\'].values) / self.params[\'std\'].values;\n            elif CFG.scl_method == "Robust":\n                df[self.scl_cols] = (df[self.scl_cols].values - self.params[\'50%\'].values) / self.params[\'iqr\'].values;\n            elif CFG.scl_method == "MinMax":\n                df[self.scl_cols] = (df[self.scl_cols].values - self.params[\'min\'].values) / self.params[\'range\'].values;\n        else:\n            PrintColor(f"Scaling is not needed", color = Fore.RED);\n    \n        return df;\n    \ncollect();\nprint();\n')


# In[ ]:


strt_ftre


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nPrintColor(f"\\n{\'-\'* 20} Data transforms and encoding {\'-\'* 20}", color = Fore.MAGENTA);\n\n# Data transforms:--\nXtrain, Ytrain = train[strt_ftre], train[CFG.target];\n\npca = make_pipeline(*[Scaler(scl_method = CFG.scl_method, scale_req = "Y", scl_cols = cont_cols), \n                      PCA(random_state = CFG.state, n_components= CFG.ncomp)\n                     ]\n                   );\nxform = \\\nmake_pipeline(*[ColumnTransformer([(\'D\', pca, cont_cols), \n                                   (\'T\', Xformer(), strt_ftre)\n                                  ], verbose_feature_names_out= False, remainder= \'passthrough\')]\n             );\nprint();\ndisplay(xform);\n\nxform.fit(Xtrain, Ytrain[CFG.target[0]]);\nXtrain = xform.transform(Xtrain);\nXtest  = xform.transform(pp.test);\n\n# Adjusting the indices after transforms:-\nXtrain.index = range(len(Xtrain));\nYtrain.index = Xtrain.index;\n\nprint();\ncollect();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Displaying the transformed data descriptions for infinite/ null values:-\nPrintColor(f"\\n---- Transformed data description for distribution analysis ----\\n",\n          color = Fore.MAGENTA);\n\nPrintColor(f"\\nTrain data\\n");\ndisplay(Xtrain.describe(percentiles = [0.05, 0.25, 0.50, 0.75, 0.9, 0.95]).T.\\\n        drop(columns = [\'count\']).\\\n        style.format(formatter = \'{:,.2f}\').\\\n        background_gradient(cmap = \'Blues\')\n       );\n\nPrintColor(f"\\nTest data\\n");\ndisplay(Xtest.describe(percentiles = [0.05, 0.25, 0.50, 0.75, 0.9, 0.95]).T.\\\n        drop(columns = [\'count\']).\\\n        style.format(formatter = \'{:,.2f}\').\\\n        background_gradient(cmap = \'Blues\')\n       );\n')


# <a id="7"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:black; font-size:120%; text-align:left;padding:3.0px; background: #cceeff; border-bottom: 8px solid #004466" > MODEL TRAINING <br><div> 
#     
# We commence our model assignment with a simple ensemble of tree-based and linear models and then shall proceed with the next steps<br>
#   
# **Note**-<br>
# GINI metric is a rank based metric that does not necessitate the usage of classifiers only. Regressors could be used to good effect appropos to their contribution to the CV score. <br>
# **We will make 2 classifiers, one for each target and then execute the process**<br>

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Initializing model I-O:-\n\nMdl_Master = \\\n{\'CBR\': CatBoostRegressor(**{\'task_type\'           : "GPU" if CFG.gpu_switch == "ON" else "CPU",\n                             \'loss_function\'       : \'RMSE\',\n                             \'eval_metric\'         : \'RMSE\',\n                             \'bagging_temperature\' : 0.45,\n                             \'colsample_bylevel\'   : 0.75,\n                             \'iterations\'          : 4500,\n                             \'learning_rate\'       : 0.085,\n                             \'od_wait\'             : 40,\n                             \'max_depth\'           : 7,\n                             \'l2_leaf_reg\'         : 0.75,\n                             \'min_data_in_leaf\'    : 35,\n                             \'random_strength\'     : 0.2, \n                             \'max_bin\'             : 256,\n                             \'verbose\'             : 0,\n                           }\n                        ),\n \n \'CBC\': CatBoostClassifier(**{\'task_type\'           : "GPU" if CFG.gpu_switch == "ON" else "CPU",\n                              \'objective\'           : \'Logloss\',\n                              \'loss_function\'       : \'Logloss\',\n                              \'eval_metric\'         : \'AUC\',\n                              \'bagging_temperature\' : 0.425,\n                              \'colsample_bylevel\'   : 0.75,\n                              \'iterations\'          : 4_000,\n                              \'learning_rate\'       : 0.025,\n                              \'od_wait\'             : 32,\n                              \'max_depth\'           : 6,\n                              \'l2_leaf_reg\'         : 0.45,\n                              \'min_data_in_leaf\'    : 28,\n                              \'random_strength\'     : 0.15, \n                              \'max_bin\'             : 200,\n                              \'verbose\'             : 0,\n                           }\n                         ), \n\n \'LGBMR\': LGBMRegressor(**{\'device\'            : "gpu" if CFG.gpu_switch == "ON" else "cpu",\n                           \'objective\'         : \'regression\',\n                           \'metric\'            : \'rmse\',\n                           \'boosting_type\'     : \'gbdt\',\n                           \'random_state\'      : CFG.state,\n                           \'colsample_bytree\'  : 0.67,\n                           \'feature_fraction\'  : 0.70,\n                           \'learning_rate\'     : 0.06,\n                           \'max_depth\'         : 8,\n                           \'n_estimators\'      : 5000,\n                           \'num_leaves\'        : 120,                    \n                           \'reg_alpha\'         : 1.25,\n                           \'reg_lambda\'        : 3.5,\n                           \'verbose\'           : -1,\n                         }\n                      ),\n \n  \'LGBMC\': LGBMClassifier(**{\'device\'            : "gpu" if CFG.gpu_switch == "ON" else "cpu",\n                             \'objective\'         : \'binary\',\n                             \'metric\'            : \'auc\',\n                             \'boosting_type\'     : \'gbdt\',\n                             \'random_state\'      : CFG.state,\n                             \'colsample_bytree\'  : 0.675,\n                             \'subsample\'         : 0.925,\n                             \'learning_rate\'     : 0.025,\n                             \'max_depth\'         : 9,\n                             \'n_estimators\'      : 4000,\n                             \'num_leaves\'        : 90,                    \n                             \'reg_alpha\'         : 0.0001,\n                             \'reg_lambda\'        : 1.5,\n                             \'verbose\'           : -1,\n                         }\n                      ),\n\n \'XGBR\': XGBRegressor(**{\'objective\'          : \'reg:squarederror\',\n                         \'eval_metric\'        : \'rmse\',\n                         \'random_state\'       : CFG.state,\n                         \'tree_method\'        : "gpu_hist" if CFG.gpu_switch == "ON" else "hist",\n                         \'colsample_bytree\'   : 0.75,\n                         \'learning_rate\'      : 0.0125,\n                         \'max_depth\'          : 8,\n                         \'n_estimators\'       : 5000,                         \n                         \'reg_alpha\'          : 1.25,\n                         \'reg_lambda\'         : 1e-05,\n                         \'min_child_weight\'   : 40,\n                        }\n                     ),\n \n  \'XGBC\': XGBClassifier(**{\'tree_method\'        : "gpu_hist" if CFG.gpu_switch == "ON" else "hist",\n                           \'objective\'          : \'binary:logistic\',\n                           \'eval_metric\'        : \'auc\',\n                           \'random_state\'       : CFG.state,\n                           \'colsample_bytree\'   : 0.25,\n                           \'learning_rate\'      : 0.018,\n                           \'max_depth\'          : 8,\n                           \'n_estimators\'       : 4000,                         \n                           \'reg_alpha\'          : 0.0001,\n                           \'reg_lambda\'         : 2.25,\n                           \'min_child_weight\'   : 50,\n                        }\n                       ),\n \n  \'HGBC\': HGBC(learning_rate    = 0.045,\n               max_iter         = 2000,\n               max_depth        = 8,\n               min_samples_leaf = 32,\n               l2_regularization= 1.25,\n               max_bins         = 200,\n               n_iter_no_change = 50,\n               random_state     = CFG.state,\n              ),\n \n  \'RFC\' : RFC(n_estimators     = 250,\n              criterion        = \'gini\',\n              max_depth        = 6,\n              min_samples_leaf = 35,\n              max_features     = "log2",\n              bootstrap        = True,\n              oob_score        = True,\n              n_jobs           = -1,\n              random_state     = CFG.state,\n              verbose          = 0,\n             ),\n \n  \'ETC\' : ETC(n_estimators     = 350,\n              criterion        = \'gini\',\n              max_depth        = 7,\n              min_samples_leaf = 38,\n              max_features     = "log2",\n              bootstrap        = True,\n              oob_score        = True,\n              n_jobs           = -1,\n              random_state     = CFG.state,\n              verbose          = 0,\n             ),\n \n  \'LC\' : LC(penalty=\'l2\',\n            tol=0.0001,\n            C= 5.3,\n            random_state= CFG.state,\n           )\n};\n\nprint();\ncollect();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Selecting relevant columns for the train and test sets:-\nsel_cols = [\'pca0\', \'pca1\', \'pca2\', \n            \'BertzCT\', \'Chi1\', \'Chi1n\', \'Chi1v\', \'Chi2n\',\'Chi2v\', \'Chi3v\', \'Chi4n\',\n            \'EState_VSA1\', \'EState_VSA2\', \'ExactMolWt\',\n            \'FpDensityMorgan1\', \'FpDensityMorgan2\', \'FpDensityMorgan3\',\n            \'HallKierAlpha\', \'HeavyAtomMolWt\', \'Kappa3\', \'MaxAbsEStateIndex\',\n            \'MinEStateIndex\', \'NumHeteroatoms\', \n            \'PEOE_VSA10\', \'PEOE_VSA14\', \'PEOE_VSA6\', \'PEOE_VSA7\',\'PEOE_VSA8\', \n            \'SMR_VSA10\', \'SMR_VSA5\',\'SlogP_VSA3\', \'VSA_EState9\',\n            \'fr_COO\', \'fr_COO2\',\n            \'Source\'\n           ];\nprint(); \n\ntry: \n    Xtrain, Xtest = Xtrain[sel_cols], Xtest[sel_cols];\n    pprint(Xtest.columns, depth = 1, width = 10, indent = 5);\nexcept: \n    PrintColor(f"\\n---> Check the columns selected\\n---> Selected columns-", color = Fore.RED);\n    pprint(Xtest.columns, depth = 1, width = 10, indent = 5);\n        \n# Initializing output tables for the models:-\nmethods   = [col for col in Mdl_Master.keys() if col.endswith("C")];\nOOF_Preds = pd.DataFrame();\nMdl_Preds = pd.DataFrame(index = pp.sub_fl[\'id\']);\nFtreImp   = pd.DataFrame(index = Xtrain.drop(columns = [\'Source\'], errors = \'ignore\').columns);\nScores    = pd.DataFrame(columns = methods);\n\nPrintColor(f"\\n---> Selected model options- ");\npprint(methods, depth = 1, width = 100, indent = 5);\n\nprint();\ncollect();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ndef TrainMdl(method:str, ytrain):\n    \n    global Mdl_Master, Mdl_Preds, OOF_Preds, all_cv, FtreImp, Xtrain; \n    \n    model     = Mdl_Master.get(method); \n    cols_drop = [\'id\', \'Source\', \'Label\'];\n    scl_cols  = [col for col in Xtrain.columns if col not in cols_drop + [\'pca0\', \'pca1\', \'pca2\']];\n    cv        = all_cv.get(CFG.mdlcv_mthd);\n    Xt        = Xtest.copy(deep = True);\n    \n    if CFG.scale_req == "N" and method.upper() in [\'RIDGE\', \'LASSO\', \'SVR\', "SVC", "LC"]:\n        X, y        = Xtrain.copy(deep = True), ytrain.copy(deep = True);\n        scaler      = all_scalers[CFG.scl_method];\n        X[scl_cols] = scaler.fit_transform(X[scl_cols]);\n        Xt[scl_cols]= scaler.transform(Xt[scl_cols]);\n        PrintColor(f"--> Scaling the data for {method} model");\n\n    if CFG.use_orig_allfolds == "Y":\n        X    = Xtrain.query("Source == \'Competition\'");\n        y    = ytrain.loc[ytrain.index.isin(X.index)]; \n        Orig = pd.concat([Xtrain, ytrain], axis=1).query("Source == \'Original\'");\n        \n    elif CFG.use_orig_allfolds != "Y":\n        X,y = Xtrain.copy(deep = True), ytrain.copy(deep = True);\n                \n    # Initializing I-O for the given seed:-        \n    test_preds = 0;\n    oof_preds  = pd.DataFrame(); \n    scores     = [];\n    ftreimp    = 0;\n          \n    for fold_nb, (train_idx, dev_idx) in enumerate(cv.split(X, y)): \n        Xtr  = X.iloc[train_idx].drop(columns = cols_drop, errors = \'ignore\');   \n        Xdev = X.iloc[dev_idx].loc[X.Source == "Competition"].\\\n        drop(columns = cols_drop, errors = \'ignore\'); \n        ytr  = y.loc[y.index.isin(Xtr.index)];\n        ydev = y.loc[y.index.isin(Xdev.index)];\n\n        if CFG.use_orig_allfolds == "Y":\n            Xtr = pd.concat([Xtr, Orig.drop(columns = [CFG.target, \'Source\'], errors = \'ignore\')], \n                            axis = 0, ignore_index = True);\n            ytr = pd.concat([ytr, Orig[CFG.target]], axis = 0, ignore_index = True);\n            \n        # Fitting the model:- \n        if method in [\'CBR\', \'CBC\']:    \n            model.fit(Xtr, ytr, \n                      eval_set = [(Xdev, ydev)], \n                      verbose = 0,\n                      early_stopping_rounds = CFG.nbrnd_erly_stp,\n                      cat_features = cat_cols,\n                     ); \n            \n        elif method in [\'LGBMR\', \'LGBMC\']: \n            model.fit(Xtr, ytr, \n                      eval_set = [(Xdev, ydev)], \n                      verbose = 0,\n                      early_stopping_rounds = CFG.nbrnd_erly_stp,\n                      categorical_feature = cat_cols,\n                     );\n            \n        elif method in [\'XGBR\', \'XGBC\']:\n            model.fit(Xtr, ytr, \n                      eval_set = [(Xdev, ydev)], \n                      verbose = 0,\n                      early_stopping_rounds = CFG.nbrnd_erly_stp,\n                     );            \n           \n        else: \n            model.fit(Xtr, ytr); \n            \n        # Collecting predictions and scores and post-processing OOF based on model method:-\n        if method.upper().endswith(\'R\'):\n            dev_preds = PostProcessPred(model.predict(Xdev), \n                                        post_process= CFG.pstprcs_train);\n            test_preds = test_preds + \\\n            PostProcessPred(model.predict(Xt.drop(columns = cols_drop, errors = \'ignore\')),\n                            post_process= CFG.pstprcs_train); \n        else:\n            dev_preds = model.predict_proba(Xdev)[:,1];\n            test_preds = test_preds + \\\n            model.predict_proba(Xt.drop(columns = cols_drop, errors = \'ignore\'))[:,1];            \n        \n        score = ScoreMetric(ydev.values.flatten(), dev_preds);\n        scores.append(score); \n \n        Scores.loc[f"{tgt}_Fold{fold_nb}", method] = np.round(score, decimals= 6);\n        oof_preds = pd.concat([oof_preds,\n                               pd.DataFrame(index   = Xdev.index, \n                                            data    = dev_preds,\n                                            columns = [method])\n                              ],axis=0, ignore_index= False\n                             );  \n    \n        oof_preds = pd.DataFrame(oof_preds.groupby(level = 0)[method].mean());\n        oof_preds.columns = [method];\n        \n        try: ftreimp += model.feature_importances_;\n        except: ftreimp = 0;\n             \n    \n    OOF_Preds[f\'{method}_{tgt}\'] = PostProcessPred(oof_preds.values.flatten(), CFG.pstprcs_train);\n    \n    if CFG.mdlcv_mthd in [\'KF\', \'SKF\']:\n        Mdl_Preds[f\'{method}_{tgt}\'] = test_preds.flatten()/ CFG.n_splits; \n        FtreImp[f\'{method}_{tgt}\']   = ftreimp / CFG.n_splits;\n    else:\n        Mdl_Preds[f\'{method}_{tgt}\'] = test_preds.flatten()/ (CFG.n_splits * CFG.n_repeats); \n        FtreImp[f\'{method}_{tgt}\']   = ftreimp / (CFG.n_splits * CFG.n_repeats);\n    \n    collect(); \n    \ncollect();\nprint();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Implementing the ML models:-\nif CFG.ML == "Y": \n    for tgt in CFG.target:\n        PrintColor(f"{\'-\'* 40} {tgt} {\'-\'* 40}", color = Fore.MAGENTA);\n        for method in tqdm(methods, "ML models----"): \n            TrainMdl(method, ytrain = Ytrain[tgt]);\n            \n        PrintColor(f"\\n{\'-\' * 20} Mean CV scores till {tgt} {\'-\' * 20}\\n", color = Fore.MAGENTA);\n        display(pd.concat([Scores.mean(axis = 0), Scores.std(axis = 0)], axis=1).\\\n                rename(columns = {0: \'Mean\', 1: \'Std\'}).T.\\\n                style.format(precision = 6).\\\n                background_gradient(cmap = \'Pastel1\', axis=1)\n               );    \nelse:\n    PrintColor(f"\\nML models are not needed\\n", color = Fore.RED);\n    \nprint();\ncollect();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Analysing the model results and feature importances and calibration curves:-\nif CFG.ML == "Y":\n    for tgt in CFG.target:\n        ytrain = Ytrain[tgt];\n        \n        PrintColor(f"\\n{\'=\' * 150}\\n", color = Fore.MAGENTA);\n        \n        fig, axes = plt.subplots(len(methods), 2, figsize = (25, len(methods) * 7.5),\n                                 gridspec_kw = {\'hspace\': 0.2, \'wspace\': 0.2}, \n                                 width_ratios= [0.7, 0.3],\n                                );\n\n        for i, col in enumerate(methods):\n            ax = axes[i,0];\n            FtreImp[f"{col}_{tgt}"].plot.barh(ax = ax, color = \'#0073e6\');\n            ax.set_title(f"{col}_{tgt} Importances", **CFG.title_specs);\n            ax.set(xlabel = \'\', ylabel = \'\');\n\n            ax = axes[i,1];\n            Clb.from_predictions(ytrain[0:len(OOF_Preds)], OOF_Preds[f"{col}_{tgt}"], \n                                 n_bins= 20, ref_line = True,\n                                 **{\'color\': \'#0073e6\', \'linewidth\': 1.2, \n                                    \'markersize\': 3.75, \'marker\': \'o\', \'markerfacecolor\': \'#cc7a00\'},\n                                 ax = ax\n                                );\n            ax.set_title(f"{col} {tgt} Calibration", **CFG.title_specs);\n            ax.set(xlabel = \'\', ylabel = \'\',);\n            ax.set_yticks(np.arange(0,1.01, 0.05), labels = np.round(np.arange(0,1.01, 0.05), 2), fontsize = 7.0);\n            ax.set_xticks(np.arange(0,1.01, 0.05), \n                          labels = np.round(np.arange(0,1.01, 0.05), 2), \n                          fontsize = 7.0, \n                          rotation = 90\n                         );\n            ax.legend(\'\');\n\n        plt.tight_layout();\n        plt.show();\n    \ncollect();\nprint();\n')


# <a id="8"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:black; font-size:120%; text-align:left;padding:3.0px; background: #cceeff; border-bottom: 8px solid #004466" > ENSEMBLE AND SUBMISSION<br> <div> 
#    

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ensemble_req == "Y":\n    def Objective(trial):\n        "This function defines the objective for the optuna ensemble using variable models";\n\n        global OOF_Preds, all_cv, ytrain, methods, tgt, cols;\n\n        # Define the weights for the predictions from each model:-\n        weights  = [trial.suggest_float(f"M{n}", 0.0001, 0.9999, step = 0.001) \\\n                    for n in range(len(cols))\n                   ];\n\n        # Calculating the CV-score for the weighted predictions on the competition data only:-\n        scores = [];  \n        cv     = all_cv[CFG.enscv_mthd];\n        X,y    = OOF_Preds[cols], ytrain[0: len(OOF_Preds)];\n\n        for fold_nb, (train_idx, dev_idx) in enumerate(cv.split(X,y)):\n            Xtr, Xdev = X.iloc[train_idx], X.iloc[dev_idx];\n            ytr, ydev = y.loc[Xtr.index],  y.loc[Xdev.index];\n            scores.append(ScoreMetric(ydev, np.average(Xdev, axis=1, weights = weights)));\n\n        collect();\n        clear_output();\n        return np.mean(scores);\n    \nclear_output();\nprint();\ncollect();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ensemble_req == "Y":\n    sub_fl      = pp.sub_fl.copy(deep = True);\n    ens_weights = {};\n    ens_score   = {};\n    \n    for tgt in tqdm(CFG.target, f"Ensemble -"):\n        PrintColor(f"\\n{\'-\' * 20} Optuna Ensemble for {tgt} {\'-\' * 20}\\n", color = Fore.MAGENTA); \n        ytrain = Ytrain[tgt];\n        cols   = OOF_Preds.columns[OOF_Preds.columns.str.endswith(tgt)].tolist();\n\n        study = optuna.create_study(direction  = CFG.metric_obj, \n                                    study_name = "OptunaEnsemble", \n                                    sampler    = TPESampler(seed = CFG.state)\n                                   );\n        study.optimize(Objective, \n                       n_trials          = CFG.ntrials, \n                       gc_after_trial    = True,\n                       show_progress_bar = True\n                      );\n        \n        weights          = study.best_params;  \n        ens_weights[tgt] = weights;\n        ens_score[tgt]   = np.round(study.best_value, 6);\n\n        # Making weighted predictions on the test set:-\n        sub_fl[tgt] = np.average(Mdl_Preds[cols], \n                                 weights = list(weights.values()),\n                                 axis=1);\n        del weights, ytrain, cols; \n        clear_output();\n        \n    PrintColor(f"\\n--> Post ensemble weights");\n    pprint(ens_weights, indent = 5, width = 10, depth = 2);\n    PrintColor(f"\\n--> Post ensemble score");\n    pprint(ens_score, indent = 5, width = 10, depth = 2);\n    \n    sub_fl.to_csv(f"Submission_V{CFG.version_nb}.csv", index = None);\n         \ncollect();\nprint();\n')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ML == "Y":  \n    OOF_Preds.add_prefix(f"V{CFG.version_nb}_").to_csv(f"OOF_Preds_V{CFG.version_nb}.csv");\n    Mdl_Preds.add_prefix(f"V{CFG.version_nb}_").to_csv(f"Mdl_Preds_V{CFG.version_nb}.csv"); \n    if isinstance(Scores, pd.DataFrame) == True:\n        Scores.to_csv(f"Scores_V{CFG.version_nb}.csv");\n           \ncollect();\nprint();\n')


# <a id="9"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #0052cc; border-bottom: 8px solid #cc9966" > NEXT STEPS<br> <div> 

# <div style= "font-family: Cambria; letter-spacing: 0px; color:#000000; font-size:110%; text-align:left;padding:3.0px; background: #f2f2f2" >
# 1. Better feature engineering- this includes feature importance assessments, decision to include/ exclude features and new feature creation<br>
# 2. Better experiments with scaling, encoding with categorical columns. This seems to have some promise<br>
# 3. Better model tuning<br>
# 4. Model calibration- this may not be absolutely necessary with a rank-metric like GINI<br>
# 5. Adding more algorithms and new methods to the model suite<br>
# 6. Better ensemble strategy<br>
# 7. Do we exclude EC3- EC6??<br>
# 8. Do we treat this as a multi-label challenge only??
# </div>
