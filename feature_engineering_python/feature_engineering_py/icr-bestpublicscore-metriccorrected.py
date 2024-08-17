#!/usr/bin/env python
# coding: utf-8

# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #006bb3; border-bottom: 8px solid #a6a6a6" > TABLE OF CONTENTS<br><div>  
# 
# * [IMPORT](#1)
# * [INTRODUCTION](#2)
# * [PREPROCESSING](#3)
# * [MODEL DEFINITION](#4)
# * [FEATURE ENGINEERING](#5)
# * [MODEL TRAINING](#6)  
# * [SUBMISSION](#7)    

# <a id="1"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #006bb3; border-bottom: 8px solid #a6a6a6" > IMPORT<br><div> 

# In[1]:


get_ipython().run_cell_magic('time', '', '\nfrom IPython.display import clear_output;\n\n!pip install -q tabpfn --no-index --find-links=file:///kaggle/input/pip-packages-icr/pip-packages\n!mkdir -p /opt/conda/lib/python3.10/site-packages/tabpfn/models_diff\n!cp /kaggle/input/pip-packages-icr/pip-packages/prior_diff_real_checkpoint_n_0_epoch_100.cpkt /opt/conda/lib/python3.10/site-packages/tabpfn/models_diff/\n\nclear_output();\nprint();\n')


# In[2]:


get_ipython().run_cell_magic('time', '', "\nimport numpy as np;\nimport pandas as pd;\n\nfrom sklearn.preprocessing import LabelEncoder,normalize;\nfrom sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier;\nfrom sklearn.metrics import accuracy_score, log_loss;\nfrom sklearn.impute import SimpleImputer;\n\nimport imblearn;\nfrom imblearn.over_sampling import RandomOverSampler;\nfrom imblearn.under_sampling import RandomUnderSampler;\n\nfrom xgboost import XGBClassifier;\nfrom lightgbm import LGBMClassifier;\nimport inspect;\nfrom collections import defaultdict;\nfrom tabpfn import TabPFNClassifier;\n\nfrom tqdm.notebook import tqdm;\nfrom datetime import datetime;\nfrom sklearn.model_selection import KFold as KF, GridSearchCV;\nfrom colorama import Fore, Style, init;\nfrom pprint import pprint;\n\nimport warnings;\nwarnings.filterwarnings('ignore');\n\nfrom gc import collect;\n\nprint();\ncollect();\n")


# In[3]:


get_ipython().run_cell_magic('time', '', '\n# Color printing    \ndef PrintColor(text:str, color = Fore.BLUE, style = Style.BRIGHT):\n    "Prints color outputs using colorama using a text F-string";\n    print(style + color + text + Style.RESET_ALL); \n    \npd.set_option(\'display.max_columns\', 60);\npd.set_option(\'display.max_rows\', 50);\n\nfrom sklearn import set_config; \nset_config(transform_output = "pandas");\n\nclear_output();\nprint();\n')


# <a id="2"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #006bb3; border-bottom: 8px solid #a6a6a6" > INTRODUCTION<br><div> 

# **Notebook objective**<br>
# 
# This notebook is adapted from the best public scoring notebook for the competition- **ICR - Identifying Age-Related Conditions**. References are as below-<br>
# 1. https://www.kaggle.com/code/vadimkamaev/postprocessin-ensemble
# 2. https://www.kaggle.com/code/aikhmelnytskyy/public-krni-pdi-with-two-additional-models
# 3. https://www.kaggle.com/code/opamusora/changed-threshold
# 
# I also used the explanation from a recent discussion post correcting the metric implementation in the above references as below-
# https://www.kaggle.com/competitions/icr-identify-age-related-conditions/discussion/422442<br>
# 
# Many thanks to the contributors of these references and all the best to the participants too!<br>
# 
# **My contribution**<br>
# 1. Changed the implementation of the metric based on the above discussion post reference<br>
# 2. Added a couple of ML models in addition to the ones present in the reference files<br>
# 3. Made a small configuration class to enable the user to toggle and make simple experiments<br>
# 4. Added comments and organized the code efficiently for effective readibility<br>
# 
# 

# In[4]:


get_ipython().run_cell_magic('time', '', '\n# Defining a configuration class with key variables to toggle for simple experiments:-\n\nclass CFG:\n    """\n    This class defines several variables to toggle for experiments\n    """;\n    \n    state          = 42;\n    n_splits_outer = 10;\n    n_splits_inner = 5;\n    \n    # Defines post-processing cutoffs:-    \n    postprocess_req= "Y";\n    lw_cutoff      = 0.28;\n    up_cutoff      = 0.595;\n    \nprint();\n')


# <a id="3"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #006bb3; border-bottom: 8px solid #a6a6a6" > PREPROCESSING<br><div> 
#     
# **Key tasks**<br>
# 1. Data import<br>
# 2. Defining the correct competition metric function<br>
# 3. Under-sampling the datasets appropriately<br>

# In[5]:


get_ipython().run_cell_magic('time', '', '\n# Making the undersampled dataset:-\ndef MakeRndUndSmpl(df):\n    """\n    This function makes a dataset using the provided dataset with undersampling technique. \n    This is a slightly verbose implementation of the same.\n    \n    Input-->   df - pd.DataFrame\n    Returns--> Modified dataframe\n    """;\n    \n    # Calculate the number of samples for each label. \n    neg, pos = np.bincount(df[\'Class\']);\n\n    # Choose the samples with class label `1`.\n    one_df = df.loc[df[\'Class\'] == 1] ;\n    # Choose the samples with class label `0`.\n    zero_df = df.loc[df[\'Class\'] == 0];\n    # Select `pos` number of negative samples.\n    # This makes sure that we have equal number of samples for each label.\n    zero_df = zero_df.sample(n=pos);\n\n    # Join both label dataframes.\n    undersampled_df = pd.concat([zero_df, one_df]);\n\n    # Shuffle the data and return\n    return undersampled_df.sample(frac = 1);\n\nprint();\ncollect();\n')


# In[6]:


get_ipython().run_cell_magic('time', '', '\n# Defining the competition metric as per Chris Deotte\'s post:-\ndef ScoreMetric(ytrue, ypred):\n    """\n    This function provides the competition metric- balanced log loss correctly as per Chris Deotte post\n    \n    Inputs-->\n    ytrue, ypred- np.array - true and prediction arrays\n    Returns--> balanced log loss score\n    \n    Note:- The floor value of the returned balanced log loss is defined to prevent a divide-by-0-error\n    """;\n    \n    nc = np.bincount(ytrue);\n    return log_loss(ytrue, ypred, sample_weight = 1 / nc[ytrue], eps=1e-15);\n\nprint();\ncollect();\n')


# In[7]:


get_ipython().run_cell_magic('time', '', '\n# Importing the datasets:-\ntrain  = pd.read_csv(\'/kaggle/input/icr-identify-age-related-conditions/train.csv\')\ntest   = pd.read_csv(\'/kaggle/input/icr-identify-age-related-conditions/test.csv\')\nsample = pd.read_csv(\'/kaggle/input/icr-identify-age-related-conditions/sample_submission.csv\')\ngreeks = pd.read_csv(\'/kaggle/input/icr-identify-age-related-conditions/greeks.csv\')\n\n# Encoding the category column- EJ:-\nfirst_category = train.EJ.unique()[0];\ntrain.EJ       = train.EJ.eq(first_category).astype(\'int\');\ntest.EJ        = test.EJ.eq(first_category).astype(\'int\');\n\n# Implementing the under-sampling process:-\ntrain_good        = MakeRndUndSmpl(train);\npredictor_columns = [n for n in train.columns if n not in [\'Class\',\'Id\']];\nx                 = train[predictor_columns];\ny                 = train[\'Class\'];\ncv_outer          = KF(n_splits = CFG.n_splits_outer,  shuffle = True,   random_state = CFG.state);\ncv_inner          = KF(n_splits = CFG.n_splits_inner,  shuffle  = True,  random_state = CFG.state);\n\n\nPrintColor(f"\\nTrain and test feature predictors\\n");\ndisplay(np.array(predictor_columns));\n\nPrintColor(f"\\nOriginal and undersampled train data shape = {train.shape} {train_good.shape}\\n");\n\ncollect();\nprint();\n')


# <a id="4"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #006bb3; border-bottom: 8px solid #a6a6a6" > MODEL DEFINITION<br><div> 

# In[8]:


get_ipython().run_cell_magic('time', '', '\nclass Ensemble():\n    """\n    This class defines the below process-\n    1. Enlists the base models for the subsequent ensemble and imputes nulls using a SimpleImputer\n    2. Creates the fit method to fit them to the training data based on model choice\n    3. Curates the model probability predictions \n    """;\n    \n    def __init__(self):\n        """\n        This method initializes the imputation strategy and the classifier choices for the subsequent models\n        """;\n        \n        self.imputer = SimpleImputer(missing_values = np.nan, \n                                     strategy = \'median\');\n\n        self.classifiers = \\\n        {"XGBC": XGBClassifier(n_estimators     = 200,\n                               max_depth        = 3,\n                               learning_rate    = 0.15,\n                               subsample        = 0.9,\n                               colsample_bytree = 0.85,\n                               reg_alpha        = 0.0001,\n                               reg_lambda       = 0.85,\n                              ),\n         \n         "LGBMC":LGBMClassifier(**{\'device\'            : "cpu",\n                                   \'verbose\'           : -1,\n                                   \'boosting_type\'     : \'gbdt\',\n                                   \'random_state\'      : 42,\n                                   \'colsample_bytree\'  : 0.4,\n                                   \'learning_rate\'     : 0.10,\n                                   \'max_depth\'         : 3,\n                                   \'min_child_samples\' : 5,\n                                   \'n_estimators\'      : 150,\n                                   \'num_leaves\'        : 40,\n                                   \'reg_alpha\'         : 0.0001,\n                                   \'reg_lambda\'        : 0.65,\n                                   \'subsample\'         : 0.65, \n                                  }\n                               ),\n                           \n           "TPFN1C": TabPFNClassifier(N_ensemble_configurations = 24, seed = 42),\n                           \n           "TPFN2C": TabPFNClassifier(N_ensemble_configurations = 64, seed = 42),\n        };\n    \n    def fit(self,X,y): \n        "This method fits the classifier choices to the train dataset";\n        \n        y = y.values;\n        unique_classes, y = np.unique(y, return_inverse=True);\n        self.classes_     = unique_classes;\n        first_category    = X.EJ.unique()[0];\n        X.EJ              = X.EJ.eq(first_category).astype(\'int\');\n        X                 = self.imputer.fit_transform(X);\n\n        for method, classifier in tqdm(self.classifiers.items(), "--Model fit--"):\n            if method.upper().startswith("TPFN"):\n                classifier.fit(X,y,overwrite_warning = True);\n            else:\n                classifier.fit(X, y);\n     \n    def predict_proba(self, x):\n        "This method curates predictions from the individual fitted classifiers";\n        \n        x = self.imputer.transform(x);\n\n        probabilities          = np.stack([classifier.predict_proba(x) for classifier in self.classifiers.values()]);\n        averaged_probabilities = np.mean(probabilities, axis=0);\n        class_0_est_instances  = averaged_probabilities[:, 0].sum();\n        others_est_instances   = averaged_probabilities[:, 1:].sum();\n        \n        # Calculating weighted probabilities:-\n        new_probabilities = \\\n        averaged_probabilities * np.array([[1/(class_0_est_instances if i==0 else others_est_instances) \n                                            for i in range(averaged_probabilities.shape[1])\n                                           ]\n                                          ]\n                                         );\n        return new_probabilities / np.sum(new_probabilities, axis=1, keepdims=1);\n    \nprint();\ncollect();\n')


# In[9]:


get_ipython().run_cell_magic('time', '', '\n# Defining the training function:-\ndef TrainMdl(model, x, y, y_meta):\n    """\n    This function aims to do the below-\n    1. Trains models with a CV split using the inner-outer CV strategy\n    2. Curates model predictions for OOF score \n    3. Saves the best model available from the above strategy\n    \n    Inputs- \n    model - model object to be used for fit and prediction\n    x,y, y_meta - input data to be used to engender predictions and OOF score\n    \n    Returns- best model from CV strategy\n    """;\n    \n    outer_results = list();\n    best_loss     = np.inf;\n    split         = 0;\n    splits        = 5;\n    \n    for train_idx,val_idx in tqdm(cv_inner.split(x), total = splits):\n        split+=1;\n        \n        x_train, x_val = x.iloc[train_idx],x.iloc[val_idx];\n        y_train, y_val = y_meta.iloc[train_idx], y.iloc[val_idx];\n                \n        model.fit(x_train, y_train);\n        \n        y_pred        = model.predict_proba(x_val);\n        probabilities = np.concatenate((y_pred[:,:1], np.sum(y_pred[:,1:], 1, keepdims=True)), axis=1);\n        p0 = probabilities[:,:1];\n        p0[p0 > 0.86] = 1;\n        p0[p0 < 0.14] = 0;\n        \n        y_p = np.empty((y_pred.shape[0],));\n        \n        for i in range(y_pred.shape[0]):\n            if p0[i] >= 0.5: y_p[i] = False;\n            else: y_p[i] = True;\n                \n        y_p  = y_p.astype(int);\n        loss = ScoreMetric(y_val,y_p);\n\n        if loss < best_loss:\n            best_model = model;\n            best_loss  = loss;\n            PrintColor(f\'-----> Best model saved\', color = Fore.GREEN);\n            \n        outer_results.append(loss);\n        num_space = 5 if split <= 9 else 4;\n        PrintColor(f"--> Fold{split}. {\'-\' * num_space}> CV = {loss:.5f}");\n        del num_space;\n    \n    PrintColor(f"\\n ----> Mean CV score = {np.mean(outer_results):.5f} <----\\n", color= Fore.MAGENTA);\n  \n    return best_model;\n\nprint();\ncollect();  \n')


# <a id="5"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #006bb3; border-bottom: 8px solid #a6a6a6" > FEATURE ENGINEERING<br><div> 
#     
# 1. We append the greeks data to the train data to perhaps extract additional insights<br>
# 2. We over-sample the train data and then prepare the model Xtrain and ytrain data for the training and inference step

# In[10]:


get_ipython().run_cell_magic('time', '', "\n# Making use of the greeks dataset for additional information:-\ntimes = greeks.Epsilon.copy();\ntimes[greeks.Epsilon != 'Unknown'] = \\\ngreeks.Epsilon[greeks.Epsilon != 'Unknown'].\\\nmap(lambda x: datetime.strptime(x,'%m/%d/%Y').toordinal());\ntimes[greeks.Epsilon == 'Unknown'] = np.nan;\n\n# Appending the greeks data to the train data:-\ntrain_pred_and_time = pd.concat((train, times), axis=1);\ntest_predictors     = test[predictor_columns];\nfirst_category      = test_predictors.EJ.unique()[0];\ntest_predictors.EJ  = test_predictors.EJ.eq(first_category).astype('int');\ntest_pred_and_time  = \\\nnp.concatenate((test_predictors, np.zeros((len(test_predictors), 1)) + \n                train_pred_and_time.Epsilon.max() + 1), axis=1\n              );\n\nprint();\ncollect();\n")


# In[11]:


get_ipython().run_cell_magic('time', '', "\nros = RandomOverSampler(random_state = CFG.state);\n\ntrain_ros, y_ros = ros.fit_resample(train_pred_and_time, greeks.Alpha);\nPrintColor('\\nOriginal dataset shape\\n');\npprint(greeks.Alpha.value_counts());\nPrintColor('\\nResample dataset shape\\n');\npprint( y_ros.value_counts());\n\nx_ros = train_ros.drop(['Class', 'Id'],axis=1);\ny_    = train_ros.Class;\n\ncollect();\nprint();\n")


# <a id="6"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #006bb3; border-bottom: 8px solid #a6a6a6" > MODEL TRAINING<br><div> 

# In[12]:


get_ipython().run_cell_magic('time', '', '\nPrintColor(f"\\n {\'-\' * 30} Model training {\'-\' * 30} \\n", color = Fore.MAGENTA);\n\nyt = Ensemble();\nm  = TrainMdl(yt, x_ros, y_, y_ros);\ny_.value_counts() / y_.shape[0];\n\ny_pred        = m.predict_proba(test_pred_and_time);\nprobabilities = np.concatenate((y_pred[:,:1], np.sum(y_pred[:,1:], 1, keepdims=True)), axis=1);\np0            = probabilities[:,:1];\n\nprint();\n')


# <a id="7"></a>
# # <div style= "font-family: Cambria; font-weight:bold; letter-spacing: 0px; color:#ffffff; font-size:120%; text-align:left;padding:3.0px; background: #006bb3; border-bottom: 8px solid #a6a6a6" > SUBMISSION<br><div> 

# In[13]:


get_ipython().run_cell_magic('time', '', '\nif CFG.postprocess_req.upper() == "Y":\n    PrintColor(f"\\nPost-processing predictions with cutoffs = {CFG.lw_cutoff:.2f} {CFG.up_cutoff:.2f}\\n");\n    p0[p0 > CFG.up_cutoff] = 1;\n    p0[p0 < CFG.lw_cutoff] = 0; \n    \nelse:\n    PrintColor(f"Post-processing is not required", color = Fore.RED);\n    \nsubmission            = pd.DataFrame(test["Id"], columns = ["Id"]);\nsubmission["class_0"] = p0;\nsubmission["class_1"] = 1 - p0;\n\nsubmission.to_csv(\'submission.csv\', index = False);\nsubmission_df = pd.read_csv(\'submission.csv\')\ndisplay(submission_df);\n\ncollect();\nprint();\n')

