#!/usr/bin/env python
# coding: utf-8

# # Explained hoarders ensemble 💨
# 
# ## Introduction 🌟
# Welcome to this Jupyter notebook developed for the Optiver - Trading at the Close
# to  Predict US stocks closing movement 
# 
# ### Inspiration and Credits 🙌
# This notebook is inspired by the work of Wilmer E. Henao, available at [this Kaggle project](https://www.kaggle.com/code/verracodeguacas/hoarders-ensemble/notebook). I extend my gratitude to Wilmer E. Henao for sharing their insights and code.
# 
# 🌟 Explore my profile and other public projects, and don't forget to share your feedback! 
# 👉 [Visit my Profile](https://www.kaggle.com/zulqarnainali) 👈
# 
# 🙏 Thank you for taking the time to review my work, and please give it a thumbs-up if you found it valuable! 👍
# 
# ## Purpose 🎯
# The primary purpose of this notebook is to:
# - Load and preprocess the competition data 📁
# - Engineer relevant features for model training 🏋️‍♂️
# - Train predictive models to make target variable predictions 🧠
# - Submit predictions to the competition environment 📤
# 
# ## Notebook Structure 📚
# This notebook is structured as follows:
# 1. **Data Preparation**: In this section, we load and preprocess the competition data.
# 2. **Feature Engineering**: We generate and select relevant features for model training.
# 3. **Model Training**: We train machine learning models on the prepared data.
# 4. **Prediction and Submission**: We make predictions on the test data and submit them for evaluation.
# 
# 
# ## How to Use 🛠️
# To use this notebook effectively, please follow these steps:
# 1. Ensure you have the competition data and environment set up.
# 2. Execute each cell sequentially to perform data preparation, feature engineering, model training, and prediction submission.
# 3. Customize and adapt the code as needed to improve model performance or experiment with different approaches.
# 
# **Note**: Make sure to replace any placeholder paths or configurations with your specific information.
# 
# ## Acknowledgments 🙏
# We acknowledge the Optiver organizers for providing the dataset and the competition platform.
# 
# Let's get started! Feel free to reach out if you have any questions or need assistance along the way.
# 👉 [Visit my Profile](https://www.kaggle.com/zulqarnainali) 👈
# 

# ## General library imports

# In[1]:


get_ipython().run_cell_magic('time', '', '\n\nfrom IPython.display import display_html, clear_output, Markdown;\nfrom gc import collect;\nimport copy\nfrom copy import deepcopy;\nimport pandas as pd;\nimport numpy as np;\nimport joblib;\nfrom os import system, getpid, walk;\nfrom psutil import Process;\nimport ctypes;\nlibc = ctypes.CDLL("libc.so.6");\n\nfrom pprint import pprint;\nfrom colorama import Fore, Style, init;\nfrom warnings import filterwarnings;\nfilterwarnings(\'ignore\');\n\nfrom tqdm.notebook import tqdm;\n\nimport lightgbm as lgb\nimport gc\nfrom itertools import combinations\nimport warnings\nfrom sklearn.model_selection import KFold\nfrom sklearn.metrics import mean_absolute_error\nfrom warnings import simplefilter\nimport joblib\n\nwarnings.filterwarnings("ignore")\nsimplefilter(action="ignore", category=pd.errors.PerformanceWarning)\n\nprint();\ncollect();\n')


# ## Model development

# **Cell 2**
# 
# **Explaination**:
# 
# 
# 
# 1. `%%time`: This is a Jupyter Notebook cell magic command that measures the execution time of the code in the current cell. It starts a timer when the cell is executed and stops it when the code block completes. It provides information on how long the code took to run, helping you assess its efficiency.
# 
# 2. Import Statements: The code imports various libraries and modules for model development. Let's briefly explain each import statement:
# 
#    - `from sklearn.model_selection import ...`: This line imports several classes and functions related to cross-validation strategies and scoring metrics from scikit-learn, a popular machine learning library.
#    
#    - `from lightgbm import ...`: It imports classes and functions from the LightGBM library, which is a gradient boosting framework for tree-based learning algorithms. It includes `log_evaluation`, `early_stopping`, and the `LGBMRegressor` class for LightGBM models.
#    
#    - `from xgboost import ...`: Similar to LightGBM, this imports the `XGBRegressor` class from the XGBoost library, which is another popular gradient boosting library.
#    
#    - `from catboost import ...`: This line imports the `CatBoostRegressor` class from the CatBoost library, which is a gradient boosting library known for handling categorical features effectively.
#    
#    - `from sklearn.ensemble import ...`: It imports the `HistGradientBoostingRegressor` class from scikit-learn's ensemble module. This class is for a gradient boosting regressor that uses histogram-based gradient boosting.
# 
# 3. `from sklearn.metrics import ...`: This line imports the `mean_absolute_error` function from scikit-learn as `mae`. It is a popular metric used for regression problems, and it calculates the mean absolute error between predicted and actual values.
# 
# 4. `print()`: This line prints an empty line to the console. It's just for formatting and separating the output in the console.
# 
# 5. `collect()`: It appears to be a custom or user-defined function or command. The exact purpose of this function is not clear from the provided code snippet. It's possible that it's a part of your environment or a custom function defined elsewhere in your codebase.
# 
# The main purpose of this code block is to set up the environment for model development by importing necessary libraries and modules, and it also includes the timing measurement (`%%time`) to track how long the code execution takes. The code doesn't contain any actual model development or data processing logic; it's primarily for library and module imports.

# In[2]:


get_ipython().run_cell_magic('time', '', '\n\nfrom sklearn.model_selection import (RepeatedStratifiedKFold as RSKF, \n                                     StratifiedKFold as SKF,\n                                     KFold, \n                                     RepeatedKFold as RKF, \n                                     cross_val_score);\n\nfrom lightgbm import log_evaluation, early_stopping, LGBMRegressor as LGBMR;\nfrom xgboost import XGBRegressor as XGBR;\nfrom catboost import CatBoostRegressor as CBR;\nfrom sklearn.ensemble import HistGradientBoostingRegressor as HGBR;\nfrom sklearn.metrics import mean_absolute_error as mae, make_scorer;\n\nprint();\ncollect();\n')


# ## Defining global configurations and functions

# **Cell 3**
# 
# **Explaination**:
# 
# 
# 1. `%%time`: As in the previous code block, this is a Jupyter Notebook cell magic command used to measure the execution time of the code in the current cell.
# 
# 2. `PrintColor(text:str, color = Fore.BLUE, style = Style.BRIGHT)`: This is a custom Python function for printing text in color using the colorama library. It takes three arguments:
#    - `text`: The text to be printed in color.
#    - `color` (default is `Fore.BLUE`): The desired text color. `Fore.BLUE` comes from the colorama library and represents blue text. You can change this color when calling the function.
#    - `style` (default is `Style.BRIGHT`): The text style. `Style.BRIGHT` makes the text bold. You can also change this style when calling the function.
# 
# 3. `GetMemUsage()`: This function is defined to estimate the RAM memory usage of the kernel. It uses the `psutil` library to get memory information, and the result is formatted as a string indicating the RAM memory usage in gigabytes (GB).
# 
# 4. The code block includes a comment with a source link that appears to be a reference to the source of the `GetMemUsage` function.
# 
# 5. `from sklearn import set_config`: This imports the `set_config` function from the scikit-learn library, which allows you to configure various aspects of scikit-learn's behavior.
# 
# 6. `set_config(transform_output = "pandas")`: This configures scikit-learn to transform the outputs of transformations into pandas DataFrames. It sets the `transform_output` option to "pandas," which means that when you perform transformations using scikit-learn, the output will be in DataFrame format.
# 
# 7. The next two lines set some options for displaying DataFrames in the pandas library. It increases the maximum number of columns and rows displayed in a DataFrame. The `pd` object is assumed to be an alias for the pandas library.
# 
# 8. `print()`: This line prints an empty line to the console for formatting.
# 
# 9. `collect()`: Similar to the previous code block, this appears to be a custom or user-defined function or command. Its exact purpose is not clear from the provided code snippet. It's possible that it's a part of your environment or a custom function defined elsewhere in your codebase.
# 
# The main purpose of this code block is to set up custom print functions, memory usage tracking, and configuration options for scikit-learn to improve the display of output DataFrames as well as to measure the execution time using `%%time`. It doesn't contain any actual data processing or model development logic.
# 

# In[3]:


get_ipython().run_cell_magic('time', '', '\n# Color printing    \ndef PrintColor(text:str, color = Fore.BLUE, style = Style.BRIGHT):\n    "Prints color outputs using colorama using a text F-string";\n    print(style + color + text + Style.RESET_ALL); \n    \ndef GetMemUsage():\n    """\n    This function defines the memory usage across the kernel. \n    Source-\n    https://stackoverflow.com/questions/61366458/how-to-find-memory-usage-of-kaggle-notebook\n    """;\n    \n    pid = getpid();\n    py = Process(pid);\n    memory_use = py.memory_info()[0] / 2. ** 30;\n    return f"RAM memory GB usage = {memory_use :.4}";\n\n# Making sklearn pipeline outputs as dataframe:-\nfrom sklearn import set_config; \nset_config(transform_output = "pandas");\npd.set_option(\'display.max_columns\', 50);\npd.set_option(\'display.max_rows\', 50);\n\nprint();\ncollect();\n')


# ## Configuration class

# **Cell 4**
# 
# **Explaination**:
# 
# Configuration class `CFG` that contains various parameters and settings for data preparation, model training, ensemble methods, inference, and global variables for plotting. 
# 
# 1. `%%time`: As in the previous code blocks, this is a Jupyter Notebook cell magic command used to measure the execution time of the code in the current cell.
# 
# 2. `class CFG`: This defines a Python class named `CFG`, which is intended to hold configuration settings. The class is encapsulated within a docstring that provides some high-level information about its purpose.
# 
# 3. Inside the `CFG` class, various configuration parameters are set, such as:
# 
#    - Data preparation parameters: These include options related to data loading and preprocessing, such as specifying the data version, whether to use a test dataset, the fraction of test data, and more.
# 
#    - Model training parameters: These include settings for training machine learning models, such as the choice of models, number of splits for cross-validation, early stopping criteria, and more.
# 
#    - Ensemble parameters: These are related to ensemble learning, including whether to use ensemble methods, ensemble cross-validation, optimization metric, and ensemble weights.
# 
#    - Inference parameters: These specify whether inference is required.
# 
#    - Global variables for plotting: These variables define settings for grid lines and titles in plots.
# 
# 4. `print()`: This line prints an empty line for formatting.
# 
# 5. `PrintColor(f"--> Configuration done!\n")`: This line uses the custom `PrintColor` function to print a message indicating that the configuration is done. The message is printed in blue color.
# 
# 6. `collect()`: This appears to be a custom or user-defined function or command. Its exact purpose is not clear from the provided code snippet. It's possible that it's a part of your environment or a custom function defined elsewhere in your codebase.
# 
# 7. `PrintColor(f"\n" + GetMemUsage(), color = Fore.RED)`: This line uses the `GetMemUsage` function to print the memory usage information, and the result is printed in red color.
# 
# The main purpose of this code block is to set up a configuration class with various parameters and settings. It also prints messages about the configuration status and memory usage. This code block does not contain any actual data processing or model development logic. Instead, it prepares the configuration and environment for subsequent tasks.

# In[4]:


get_ipython().run_cell_magic('time', '', '\nclass CFG:\n    """\n    Configuration class for parameters and CV strategy for tuning and training\n    Please use caps lock capital letters while filling in parameters\n    """;\n    \n    # Data preparation:-   \n    version_nb         = 5;\n    test_req           = "N";\n    test_frac          = 0.01;\n    load_tr_data       = "N";\n    gpu_switch         = "OFF"; \n    state              = 42;\n    target             = \'target\';\n    \n    path               = f"/kaggle/input/optiver-memoryreduceddatasets/";\n    test_path          = f"/kaggle/input/optiver-trading-at-the-close/example_test_files/test.csv";\n    df_choice          = f"XTrIntCmpNewFtre.parquet";\n    mdl_path           = f\'/kaggle/working/BaselineML/\';\n    inf_path           = f\'/kaggle/input/optiverbaselinemodels/\';\n     \n    # Model Training:-\n    methods            = ["LGBMR", "CBR", "HGBR"];\n    ML                 = "N";\n    n_splits           = 5;\n    n_repeats          = 1;\n    nbrnd_erly_stp     = 100 ;\n    mdlcv_mthd         = \'KF\';\n    \n    # Ensemble:-    \n    ensemble_req       = "N";\n    enscv_mthd         = "KF";\n    metric_obj         = \'minimize\';\n    ntrials            = 10 if test_req == "Y" else 200;\n    ens_weights        = [0.54, 0.44, 0.02];\n    \n    # Inference:-\n    inference_req      = "Y";\n    \n    # Global variables for plotting:-\n    grid_specs = {\'visible\': True, \'which\': \'both\', \'linestyle\': \'--\', \n                  \'color\': \'lightgrey\', \'linewidth\': 0.75\n                 };\n    title_specs = {\'fontsize\': 9, \'fontweight\': \'bold\', \'color\': \'tab:blue\'};\n\nprint();\nPrintColor(f"--> Configuration done!\\n");\ncollect();\n\nPrintColor(f"\\n" + GetMemUsage(), color = Fore.RED);\n')


# ## Common cross-validation strategies

# **Cell 5**
# 
# **Explaination**: 
# 
# Common cross-validation strategies and a custom scoring metric for later usage in the project. 
# 
# 1. `%%time`: As in previous code blocks, this is a Jupyter Notebook cell magic command used to measure the execution time of the code in the current cell.
# 
# 2. `all_cv`: This is a dictionary that defines common cross-validation strategies. It includes the following strategies:
# 
#    - `'KF'`: K-Fold cross-validation with the number of splits specified by `CFG.n_splits`. The data is shuffled, and the random state is set to `CFG.state`.
#    - `'RKF'`: Repeated K-Fold cross-validation with the number of splits and repeats specified by `CFG.n_splits` and `CFG.n_repeats`, respectively. The random state is set to `CFG.state`.
#    - `'RSKF'`: Repeated Stratified K-Fold cross-validation with the number of splits and repeats specified by `CFG.n_splits` and `CFG.n_repeats`, respectively. The random state is set to `CFG.state`.
#    - `'SKF'`: Stratified K-Fold cross-validation with the number of splits specified by `CFG.n_splits`. The data is shuffled, and the random state is set to `CFG.state`.
# 
# 3. `ScoreMetric(ytrue, ypred)`: This is a custom Python function for calculating the competition metric. The function takes two arguments:
#    - `ytrue`: A ground truth array of values.
#    - `ypred`: Predicted values.
#    The function calculates and returns the metric value, which is typically a mean absolute error (MAE) in this context. The specific calculation is not provided in this code block.
# 
# 4. `myscorer = make_scorer(ScoreMetric, greater_is_better = False, needs_proba=False)`: This line defines a custom scoring metric using the `make_scorer` function from scikit-learn. The `greater_is_better` parameter is set to `False`, indicating that lower scores are better (typically for error metrics like MAE), and `needs_proba` is set to `False`, suggesting that this scoring function does not require predicted probabilities.
# 
# 5. `print()`: This line prints an empty line for formatting.
# 
# 6. `collect()`: This appears to be a custom or user-defined function or command. Its exact purpose is not clear from the provided code snippet. It's possible that it's a part of your environment or a custom function defined elsewhere in your codebase.
# 
# 7. `PrintColor(f"\n" + GetMemUsage(), color = Fore.RED)`: This line uses the `GetMemUsage` function to print the memory usage information, and the result is printed in red color.
# 
# The main purpose of this code block is to set up common cross-validation strategies and a custom scoring metric for later use in the project. It does not contain any actual data processing or model development logic; instead, it prepares the environment and tools for subsequent tasks.

# In[5]:


get_ipython().run_cell_magic('time', '', '\n# Commonly used CV strategies for later usage:-\nall_cv= {\'KF\'  : KFold(n_splits= CFG.n_splits, shuffle = True, random_state= CFG.state),\n         \'RKF\' : RKF(n_splits= CFG.n_splits, n_repeats = CFG.n_repeats, random_state= CFG.state),\n         \'RSKF\': RSKF(n_splits= CFG.n_splits, n_repeats = CFG.n_repeats, random_state= CFG.state),\n         \'SKF\' : SKF(n_splits= CFG.n_splits, shuffle = True, random_state= CFG.state)\n        };\n\n# Defining the competition metric:-\ndef ScoreMetric(ytrue, ypred)-> float:\n    """\n    This function calculates the metric for the competition. \n    ytrue- ground truth array\n    ypred- predictions\n    returns - metric value (float)\n    """;\n    \n    return mae(ytrue, ypred);\n\n# Designing a custom scorer to use in cross_val_predict and cross_val_score:-\nmyscorer = make_scorer(ScoreMetric, greater_is_better = False, needs_proba=False,);\n\nprint();\ncollect();\n\nPrintColor(f"\\n" + GetMemUsage(), color = Fore.RED);\n')


# ## conversion & Adjustment

# **Cell 6**
# 
# **Explaination**: 
# 
# Ttwo functions: `goto_conversion` and `zero_sum`. Let's break down each function:
# 
# 1. `goto_conversion(listOfOdds, total=1, eps=1e-6, isAmericanOdds=False)`: This function converts odds to probabilities. Here's a breakdown of its parameters and functionality:
# 
#    - `listOfOdds`: A list of odds, which can be in either American or decimal format.
#    - `total`: The total sum of probabilities (default is 1).
#    - `eps`: A small epsilon value to prevent division by zero or extreme probabilities (default is 1e-6).
#    - `isAmericanOdds`: A boolean flag indicating whether the input odds are in American format (default is False).
# 
#    The function first checks if the odds are in American format and converts them to decimal odds if needed. It then computes the probabilities based on the inverse of the odds. Afterward, it calculates the standard error (SE) for each probability. Finally, it adjusts the probabilities by stepping back based on the computed SE, ensuring that they sum to the specified `total`. The resulting adjusted probabilities are returned.
# 
# 2. `zero_sum(listOfPrices, listOfVolumes)`: This function performs an adjustment on a list of prices and volumes to achieve a zero-sum condition. Here's a breakdown of its parameters and functionality:
# 
#    - `listOfPrices`: A list of prices.
#    - `listOfVolumes`: A list of volumes corresponding to the prices.
# 
#    The function computes standard errors for each price based on the volumes. It then adjusts the prices by scaling them using the standard errors to achieve a zero-sum condition. The adjusted prices are returned.
# 
# Both functions are designed to perform necessary transformations for modeling and analysis, particularly in the context of financial data.
# 
# The provided source links reference the Kaggle platform, indicating that these functions might be part of a Kaggle project or competition. The `collect()` function is called at the end, which is assumed to be a custom function used to gather or organize certain resources. 
# 

# In[6]:


get_ipython().run_cell_magic('time', '', '\ndef goto_conversion(listOfOdds, total = 1, eps = 1e-6, isAmericanOdds = False):\n    "Source - https://www.kaggle.com/code/kaito510/goto-conversion-optiver-baseline-models";\n\n    #Convert American Odds to Decimal Odds\n    if isAmericanOdds:\n        for i in range(len(listOfOdds)):\n            currOdds = listOfOdds[i];\n            isNegativeAmericanOdds = currOdds < 0;\n            if isNegativeAmericanOdds:\n                currDecimalOdds = 1 + (100/(currOdds*-1));\n            else: \n                #Is non-negative American Odds\n                currDecimalOdds = 1 + (currOdds/100);\n            listOfOdds[i] = currDecimalOdds;\n\n    #Error Catchers\n    if len(listOfOdds) < 2:\n        raise ValueError(\'len(listOfOdds) must be >= 2\');\n    if any(x < 1 for x in listOfOdds):\n        raise ValueError(\'All odds must be >= 1, set isAmericanOdds parameter to True if using American Odds\');\n\n    #Computation:-\n    #initialize probabilities using inverse odds\n    listOfProbabilities = [1/x for x in listOfOdds];\n    \n    #compute the standard error (SE) for each probability\n    listOfSe = [pow((x-x**2)/x,0.5) for x in listOfProbabilities];\n    \n    #compute how many steps of SE the probabilities should step back by\n    step = (sum(listOfProbabilities) - total)/sum(listOfSe) ;\n    outputListOfProbabilities = [min(max(x - (y*step),eps),1) for x,y in zip(listOfProbabilities, listOfSe)];\n    return outputListOfProbabilities;\n\ndef zero_sum(listOfPrices, listOfVolumes):\n    """\n    Source - https://www.kaggle.com/code/kaito510/goto-conversion-optiver-baseline-models\n    """;\n    \n    #compute standard errors assuming standard deviation is same for all stocks\n    listOfSe = [x**0.5 for x in listOfVolumes];\n    step = sum(listOfPrices)/sum(listOfSe);\n    outputListOfPrices = [x - (y*step) for x,y in zip(listOfPrices, listOfSe)];\n    return outputListOfPrices;\n\ncollect();\n')


# ## Load and prepare training

# **Cell 7**
# 
# **Explaination**: 
# 
# Loads and prepares training data based on the configuration settings specified in the `CFG` class.
# 
# 1. `%%time`: As in the previous code blocks, this is a Jupyter Notebook cell magic command used to measure the execution time of the code in the current cell.
# 
# 2. The code begins with a conditional check that examines the configuration settings in `CFG` to determine whether to load and prepare training data. The conditions are as follows:
#    
#    - If `CFG.load_tr_data` is set to "Y" or `CFG.ML` is set to "Y" (indicating that loading training data or machine learning tasks are required) and `CFG.test_req` is set to "Y" (indicating the need for a test dataset), the following code block is executed:
# 
#      - If `CFG.test_frac` is a floating-point number (float), it reads a sample of the training data from a parquet file located at the path specified by `CFG.path` with the name specified by `CFG.df_choice`. The sample size is determined by `CFG.test_frac`.
#      - If `CFG.test_frac` is not a float, it reads a sample of the training data using a fixed sample size specified by `CFG.test_frac`.
#      - It also reads the corresponding target data (y) from a parquet file named "Ytrain.parquet" and indexes it to match the index of the sampled training data.
#      - The shapes of the sampled training data and the target data are printed in red color for code testing purposes.
#      - The index of `X` (training data) and `y` (target data) is set to integer ranges.
# 
#    - If `CFG.load_tr_data` is set to "Y" or `CFG.ML` is set to "Y" (indicating that loading training data or machine learning tasks are required), but `CFG.test_req` is not "Y," the following code block is executed:
# 
#      - It reads the full training data (no sampling) from a parquet file located at the path specified by `CFG.path` with the name specified by `CFG.df_choice`.
#      - It also reads the corresponding target data (y) from a parquet file named "Ytrain.parquet."
#      - The shapes of the training data and the target data are printed in red color.
# 
#    - If `CFG.load_tr_data` is not "Y" or `CFG.inference_req` is set to "Y" (indicating that training data is not required because inference is being performed), a message is printed to indicate that.
# 
# 3. `print()`: This line prints an empty line for formatting.
# 
# 4. `collect()`: This appears to be a custom or user-defined function or command. Its exact purpose is not clear from the provided code snippet. It's possible that it's a part of your environment or a custom function defined elsewhere in your codebase.
# 
# 5. `libc.malloc_trim(0)`: This appears to be a memory management operation that might release unused memory allocated by the C library's `malloc` function. It sets the trim threshold to 0.
# 
# 6. `PrintColor(f"\n" + GetMemUsage(), color = Fore.RED)`: This line uses the `GetMemUsage` function to print the memory usage information in red color.
# 
# The main purpose of this code block is to load and prepare training data based on the specified configuration settings. The code also performs some memory management operations to optimize memory usage. It provides feedback on the shapes of the loaded data and the memory usage.

# In[7]:


get_ipython().run_cell_magic('time', '', '\nif (CFG.load_tr_data == "Y" or CFG.ML == "Y") and CFG.test_req == "Y":\n    if isinstance(CFG.test_frac, float):\n        X = pd.read_parquet(CFG.path + CFG.df_choice).sample(frac = CFG.test_frac);\n    else:\n        X = pd.read_parquet(CFG.path + CFG.df_choice).sample(n = CFG.test_frac);\n        \n    y = pd.read_parquet(CFG.path + f"Ytrain.parquet").loc[X.index].squeeze();\n    PrintColor(f"---> Sampled train shapes for code testing = {X.shape} {y.shape}", \n               color = Fore.RED);\n    X.index, y.index = range(len(X)), range(len(y));\n    \n    PrintColor(f"\\n---> Train set columns for model development");\n    pprint(X.columns, width = 100, depth = 1, indent = 5);\n    print();\n\nelif CFG.load_tr_data == "Y" or CFG.ML == "Y":\n    X = pd.read_parquet(CFG.path + CFG.df_choice);\n    y = pd.read_parquet(CFG.path + f"Ytrain.parquet").squeeze();  \n    PrintColor(f"---> Train shapes for code testing = {X.shape} {y.shape}");\n\nelif CFG.load_tr_data != "Y" or CFG.inference_req == "Y":\n    PrintColor(f"---> Train data is not required as we are infering from the model");\n    \nprint();\ncollect();\nlibc.malloc_trim(0);\n\nPrintColor(f"\\n" + GetMemUsage(), color = Fore.RED);\n')


# ## Initializes model configurations

# **Cell 8**
# 
# **Explaination**: 
# 
# Initializes model configurations for various machine learning models if the `CFG.ML` flag is set to "Y" in the configuration settings. Here's an explanation of the code:
# 
# 1. `%%time`: As in previous code blocks, this is a Jupyter Notebook cell magic command used to measure the execution time of the code in the current cell.
# 
# 2. The code checks if the `CFG.ML` flag is set to "Y," indicating that machine learning models need to be configured. If this condition is met, it proceeds to initialize model configurations.
# 
# 3. The variable `Mdl_Master` is created as a dictionary that holds configurations for multiple machine learning models. The configurations include hyperparameters and settings for the following models:
# 
#    - `'CBR'`: CatBoostRegressor configuration using the CatBoost library.
#    - `'LGBMR'`: LightGBMRegressor configuration using the LightGBM library.
#    - `'XGBR'`: XGBRegressor configuration using the XGBoost library.
#    - `'HGBR'`: HistGradientBoostingRegressor configuration using scikit-learn's HistGradientBoosting.
# 
#    Each model's configuration is specified as a dictionary with specific hyperparameters and settings. These include model type, learning rate, maximum depth, number of estimators, regularization parameters, and more. The settings are based on the model's specific requirements and best practices.
# 
# 4. `print()`: This line prints an empty line for formatting.
# 
# 5. `collect()`: This appears to be a custom or user-defined function or command. Its exact purpose is not clear from the provided code snippet. It's possible that it's a part of your environment or a custom function defined elsewhere in your codebase.
# 
# 6. `PrintColor(f"\n" + GetMemUsage(), color = Fore.RED)`: This line uses the `GetMemUsage` function to print the memory usage information in red color.
# 
# The main purpose of this code block is to configure machine learning models for later use in model development or training. The code specifies hyperparameters and settings for each model and stores them in the `Mdl_Master` dictionary for easy access. It does not involve model training or data processing; instead, it prepares the environment for subsequent machine learning tasks.

# In[8]:


get_ipython().run_cell_magic('time', '', '\n# Initializing model I-O:-\n\nif CFG.ML == "Y":\n    Mdl_Master = \\\n    {\'CBR\': CBR(**{\'task_type\'           : "GPU" if CFG.gpu_switch == "ON" else "CPU",\n                   \'objective\'           : "MAE",\n                   \'eval_metric\'         : "MAE",\n                   \'bagging_temperature\' : 0.5,\n                   \'colsample_bylevel\'   : 0.7,\n                   \'iterations\'          : 500,\n                   \'learning_rate\'       : 0.065,\n                   \'od_wait\'             : 25,\n                   \'max_depth\'           : 7,\n                   \'l2_leaf_reg\'         : 1.5,\n                   \'min_data_in_leaf\'    : 1000,\n                   \'random_strength\'     : 0.65, \n                   \'verbose\'             : 0,\n                   \'use_best_model\'      : True,\n                  }\n               ), \n\n      \'LGBMR\': LGBMR(**{\'device\'            : "gpu" if CFG.gpu_switch == "ON" else "cpu",\n                        \'objective\'         : \'regression_l1\',\n                        \'boosting_type\'     : \'gbdt\',\n                        \'random_state\'      : CFG.state,\n                        \'colsample_bytree\'  : 0.7,\n                        \'subsample\'         : 0.65,\n                        \'learning_rate\'     : 0.065,\n                        \'max_depth\'         : 6,\n                        \'n_estimators\'      : 500,\n                        \'num_leaves\'        : 150,  \n                        \'reg_alpha\'         : 0.01,\n                        \'reg_lambda\'        : 3.25,\n                        \'verbose\'           : -1,\n                       }\n                    ),\n\n      \'XGBR\': XGBR(**{\'tree_method\'        : "gpu_hist" if CFG.gpu_switch == "ON" else "hist",\n                      \'objective\'          : \'reg:absoluteerror\',\n                      \'random_state\'       : CFG.state,\n                      \'colsample_bytree\'   : 0.7,\n                      \'learning_rate\'      : 0.07,\n                      \'max_depth\'          : 6,\n                      \'n_estimators\'       : 500,                         \n                      \'reg_alpha\'          : 0.025,\n                      \'reg_lambda\'         : 1.75,\n                      \'min_child_weight\'   : 1000,\n                      \'early_stopping_rounds\' : CFG.nbrnd_erly_stp,\n                     }\n                  ),\n\n      "HGBR" : HGBR(loss              = \'squared_error\',\n                    learning_rate     = 0.075,\n                    early_stopping    = True,\n                    max_iter          = 200,\n                    max_depth         = 6,\n                    min_samples_leaf  = 1500,\n                    l2_regularization = 1.75,\n                    scoring           = myscorer,\n                    random_state      = CFG.state,\n                   )\n    };\n\nprint();\ncollect();\n\nPrintColor(f"\\n" + GetMemUsage(), color = Fore.RED);\n')


# ## CFG.ML

# **Cell 9**
# 
# **Explaination**: 
# 
# I nitializes various components for machine learning model training if the `CFG.ML` flag is set to "Y" in the configuration settings. 
# 
# 1. `%%time`: As in previous code blocks, this is a Jupyter Notebook cell magic command used to measure the execution time of the code in the current cell.
# 
# 2. The code checks if the `CFG.ML` flag is set to "Y," indicating that machine learning model training is required. If this condition is met, it proceeds to initialize the necessary components.
# 
# 3. `methods = CFG.methods`: This line assigns the list of methods (machine learning models) from the configuration class `CFG` to the `methods` variable.
# 
# 4. `system('mkdir BaselineML')`: This command uses the `system` function to create a directory named "BaselineML" to store trained and fitted models. It is created in the current working directory.
# 
# 5. `model_path = CFG.mdl_path`: This line assigns the model path for storage from the configuration class `CFG` to the `model_path` variable.
# 
# 6. `cv = all_cv[CFG.mdlcv_mthd]`: This line assigns a cross-validation object to the `cv` variable based on the cross-validation method specified in the configuration class `CFG` (e.g., K-Fold or Stratified K-Fold).
# 
# 7. `Scores` and `FtreImp`: These variables are initialized as empty DataFrames. They will be used to store scores and feature importances during model training. The `Scores` DataFrame has a row for each cross-validation fold and a column for each machine learning model, while the `FtreImp` DataFrame has columns for each machine learning model and rows for each feature in the dataset. Both DataFrames are filled with zeros.
# 
# 8. `print()`: This line prints an empty line for formatting.
# 
# 9. `collect()`: This appears to be a custom or user-defined function or command. Its exact purpose is not clear from the provided code snippet. It's possible that it's a part of your environment or a custom function defined elsewhere in your codebase.
# 
# 10. `libc.malloc_trim(0)`: This appears to be a memory management operation that might release unused memory allocated by the C library's `malloc` function. It sets the trim threshold to 0.
# 
# 11. `PrintColor(f"\n" + GetMemUsage(), color = Fore.RED)`: This line uses the `GetMemUsage` function to print the memory usage information in red color.
# 
# The main purpose of this code block is to set up the environment and data structures for machine learning model training. It initializes cross-validation, storage paths, and data structures for tracking scores and feature importances. It does not involve actual model training or data processing at this stage.

# In[9]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ML == "Y":\n    # Initializing the models from configuration class:-\n    methods = CFG.methods;\n\n    # Initializing a folder to store the trained and fitted models:-\n    system(\'mkdir BaselineML\');\n\n    # Initializing the model path for storage:-\n    model_path = CFG.mdl_path;\n\n    # Initializing the cv object:-\n    cv = all_cv[CFG.mdlcv_mthd];\n        \n    # Initializing score dataframe:-\n    Scores = pd.DataFrame(index = range(CFG.n_splits * CFG.n_repeats),\n                          columns = methods).fillna(0).astype(np.float32);\n    \n    FtreImp = pd.DataFrame(index = X.columns, columns = [methods]).fillna(0);\n\nprint();\ncollect();\nlibc.malloc_trim(0);\n\nPrintColor(f"\\n" + GetMemUsage(), color = Fore.RED);\n')


# **Cell 10**
# 
# **Explaination**: 
# 
# 
# 1. `%%time`: As in previous code blocks, this is a Jupyter Notebook cell magic command used to measure the execution time of the code in the current cell.
# 
# 2. The code checks if the `CFG.ML` flag is set to "Y," indicating that machine learning model training is required. If this condition is met, it proceeds with model training.
# 
# 3. `PrintColor(f"\n{'=' * 25} ML Training {'=' * 25}\n")`: This line uses the `PrintColor` function to print a message indicating the start of the machine learning training section. The message is displayed in blue color.
# 
# 4. A loop is used to perform cross-validation training. The loop iterates over each fold generated by the cross-validation method. The following steps are performed for each fold:
# 
#    - Data is split into training (`Xtr`, `ytr`) and development (`Xdev`, `ydev`) sets based on the current fold.
#    - For each machine learning method in the list of methods (`methods`), the following steps are performed:
#      - The model is selected from `Mdl_Master`.
#      - Model-specific training is performed. The code includes separate logic for CatBoost (`CBR`), LightGBM (`LGBMR`), and XGBoost (`XGBR`) models with their respective training configurations.
#      - The trained model is saved to a file for later usage.
#      - The out-of-fold (OOF) score is calculated for the development set and recorded in the `Scores` DataFrame.
#      - Feature importances are collected if available.
#    - Memory management and cleaning are performed after each fold.
# 
# 5. `clear_output()`: This line clears the Jupyter Notebook cell output to keep the display clean.
# 
# 6. `PrintColor(f"\n---> OOF scores across methods <---\n")`: This line uses the `PrintColor` function to print a message indicating the OOF scores across different methods.
# 
# 7. The OOF scores are displayed in a table format, showing the scores for each fold and method. The table is color-coded with a background gradient.
# 
# 8. `PrintColor(f"\n---> Mean OOF scores across methods <---\n")`: This line prints a message indicating the mean OOF scores across different methods.
# 
# 9. The mean OOF scores for each method are displayed.
# 
# 10. The code attempts to save feature importances to a CSV file, if applicable.
# 
# 11. `collect()`: This appears to be a custom or user-defined function or command. Its exact purpose is not clear from the provided code snippet. It's possible that it's a part of your environment or a custom function defined elsewhere in your codebase.
# 
# 12. `libc.malloc_trim(0)`: This appears to be a memory management operation that might release unused memory allocated by the C library's `malloc` function. It sets the trim threshold to 0.
# 
# 13. `PrintColor(f"\n" + GetMemUsage(), color = Fore.GREEN)`: This line uses the `GetMemUsage` function to print the memory usage information in green color, indicating the end of the training process.
# 
# The main purpose of this code block is to train machine learning models using cross-validation and evaluate their performance. It records OOF scores, displays results, and saves feature importances for later analysis. This code performs the core machine learning training and evaluation tasks.

# In[10]:


get_ipython().run_cell_magic('time', '', '\nif CFG.ML == "Y":\n    PrintColor(f"\\n{\'=\' * 25} ML Training {\'=\' * 25}\\n");\n    \n    # Initializing CV splitting:-       \n    for fold_nb, (train_idx, dev_idx) in tqdm(enumerate(cv.split(X, y)), \n                                              f"{CFG.mdlcv_mthd} CV {CFG.n_splits}x{CFG.n_repeats}"\n                                             ): \n        # Creating the cv folds:-    \n        Xtr  = X.iloc[train_idx];   \n        Xdev = X.iloc[dev_idx];\n        ytr  = y.iloc[train_idx];\n        ydev = y.iloc[dev_idx];\n        \n        PrintColor(f"-------> Fold{fold_nb} <-------");\n        # Fitting the models:- \n        for method in methods:\n            model = Mdl_Master[method];\n            if method == "LGBMR":\n                model.fit(Xtr, ytr, \n                          eval_set = [(Xdev, ydev)], \n                          verbose = 0, \n                          eval_metric = "mae",\n                          callbacks = [log_evaluation(0,), \n                                       early_stopping(CFG.nbrnd_erly_stp, verbose = False)], \n                         );\n\n            elif method == "XGBR":\n                model.fit(Xtr, ytr, \n                          eval_set = [(Xdev, ydev)], \n                          verbose = 0, \n                          eval_metric = "mae",\n                         );  \n\n            elif method == "CBR":\n                model.fit(Xtr, ytr, \n                          eval_set = [(Xdev, ydev)], \n                          verbose = 0, \n                          early_stopping_rounds = CFG.nbrnd_erly_stp,\n                         ); \n\n            else:\n                model.fit(Xtr, ytr);\n\n            #  Saving the model for later usage:-\n            joblib.dump(model, CFG.mdl_path + f\'{method}V{CFG.version_nb}Fold{fold_nb}.model\');\n            \n            # Creating OOF scores:-\n            score = ScoreMetric(ydev, model.predict(Xdev));\n            Scores.at[fold_nb, method] = score;\n            num_space = 6- len(method);\n            PrintColor(f"---> {method} {\' \'* num_space} OOF = {score:.5f}", \n                       color = Fore.MAGENTA);  \n            del num_space, score;\n            \n            # Collecting feature importances:-\n            try:\n                FtreImp[method] = \\\n                FtreImp[method].values + (model.feature_importances_ / (CFG.n_splits * CFG.n_repeats));\n            except:\n                pass;\n            \n            collect();\n            \n        PrintColor(GetMemUsage());\n        print();\n        del Xtr, ytr, Xdev, ydev;\n        collect();\n    \n    clear_output();\n    PrintColor(f"\\n---> OOF scores across methods <---\\n");\n    Scores.index.name = "FoldNb";\n    Scores.index = Scores.index + 1;\n    display(Scores.style.format(precision = 5).\\\n            background_gradient(cmap = "Pastel1")\n           );\n    \n    PrintColor(f"\\n---> Mean OOF scores across methods <---\\n");\n    display(Scores.mean());\n    \n    try: FtreImp.to_csv(CFG.mdl_path + f"FtreImp_V{CFG.version_nb}.csv");\n    except: pass;\n        \ncollect();\nprint();\nlibc.malloc_trim(0);\n\nPrintColor(f"\\n" + GetMemUsage(), color = Fore.GREEN);\n')


# **Cell 11**
# 
# **Explaination**: 
# 
# A Python function called `MakeFtre` that is used to create new features in a pandas DataFrame based on price columns. 
# 
# 1. `%%time`: As in previous code blocks, this is a Jupyter Notebook cell magic command used to measure the execution time of the code in the current cell.
# 
# 2. `def MakeFtre(df: pd.DataFrame, prices: list) -> pd.DataFrame:`: This line defines a function named `MakeFtre`. The function takes two parameters:
#    - `df`: A pandas DataFrame that represents the input dataframe.
#    - `prices`: A list of price columns for transformation.
# 
# 3. A list of feature names called `features` is initialized. These feature names will be used to create new columns in the DataFrame.
# 
# 4. The function proceeds to create new features based on price columns. Specifically, it calculates several features related to price imbalances and price relationships.
# 
#    - It calculates features such as `imb_s1` and `imb_s2` based on bid and ask size columns.
#    - It calculates features related to the imbalance between different price columns in the `prices` list, creating new columns with names like `a_b_imb`.
#    - It calculates features related to the imbalance between three price columns, creating new columns with names like `a_b_c_imb2`.
# 
# 5. The function returns a modified DataFrame containing the newly created features.
# 
# 6. `print()`: This line prints an empty line for formatting.
# 
# 7. `collect()`: This appears to be a custom or user-defined function or command. Its exact purpose is not clear from the provided code snippet. It's possible that it's a part of your environment or a custom function defined elsewhere in your codebase.
# 
# 

# In[11]:


get_ipython().run_cell_magic('time', '', '\ndef MakeFtre(df : pd.DataFrame, prices: list) -> pd.DataFrame:\n    """\n    This function creates new features using the price columns. This was used in a baseline notebook as below-\n    https://www.kaggle.com/code/yuanzhezhou/baseline-lgb-xgb-and-catboost\n    \n    Inputs-\n    df:- pd.DataFrame -- input dataframe\n    cols:- price columns for transformation\n    \n    Returns-\n    df:- pd.DataFrame -- dataframe with extra columns\n    """;\n    \n    features = [\'overall_medvol\', "first5min_medvol", "last5min_medvol",\n                \'seconds_in_bucket\', \'imbalance_buy_sell_flag\',\n                \'imbalance_size\', \'matched_size\', \'bid_size\', \'ask_size\',\n                \'reference_price\',\'far_price\', \'near_price\', \'ask_price\', \'bid_price\', \'wap\',\n                \'imb_s1\', \'imb_s2\'\n               ];\n    \n    df[\'imb_s1\'] = df.eval(\'(bid_size-ask_size)/(bid_size+ask_size)\').astype(np.float32);\n    df[\'imb_s2\'] = df.eval(\'(imbalance_size-matched_size)/(matched_size+imbalance_size)\').astype(np.float32);\n       \n    for i,a in enumerate(prices):\n        for j,b in enumerate(prices):\n            if i>j:\n                df[f\'{a}_{b}_imb\'] = df.eval(f\'({a}-{b})/({a}+{b})\');\n                features.append(f\'{a}_{b}_imb\'); \n                    \n    for i,a in enumerate(prices):\n        for j,b in enumerate(prices):\n            for k,c in enumerate(prices):\n                if i>j and j>k:\n                    max_ = df[[a,b,c]].max(axis=1);\n                    min_ = df[[a,b,c]].min(axis=1);\n                    mid_ = df[[a,b,c]].sum(axis=1)-min_-max_;\n\n                    df[f\'{a}_{b}_{c}_imb2\'] = ((max_-mid_)/(mid_-min_)).astype(np.float32);\n                    features.append(f\'{a}_{b}_{c}_imb2\');\n    \n    return df[features];\n\nprint();\ncollect();\n')


# **Cell 12**
# 
# **Explaination**: 
# 
# 
# 1. `%%time`: As in previous code blocks, this is a Jupyter Notebook cell magic command used to measure the execution time of the code in the current cell.
# 
# 2. The code checks if inferencing is required, indicated by the `CFG.inference_req` flag set to "Y." If inferencing is not required, the code proceeds to set up the testing environment. If the environment is already set up, it attempts to clear the `X` and `y` variables if they exist.
# 
# 3. A list of price column names called `prices` is defined.
# 
# 4. The code attempts to create the testing environment using the `optiver2023` library. This environment is established for iteratively testing the model on new data.
# 
# 5. A list called `models` is initialized to store the machine learning models that will be used for inferencing.
# 
# 6. The code determines the model loading path based on whether the machine learning models were trained (`CFG.ML == "Y"`) or if they are to be loaded from input data (`CFG.ML != "Y"`).
# 
# 7. The code searches for model files in the specified model path and loads them using the `joblib.load` function. The loaded models are stored in the `models` list.
# 
# 8. The list `mdl_lbl` is created to store the labels or filenames of the loaded models.
# 
# 9. A dictionary called `model_dict` is created to map model labels to their corresponding loaded models.
# 
# 10. The code prints the list of trained models with their labels.
# 
# 11. `print()`: This line prints an empty line for formatting.
# 
# 12. `collect()`: This appears to be a custom or user-defined function or command. Its exact purpose is not clear from the provided code snippet. It's possible that it's a part of your environment or a custom function defined elsewhere in your codebase.
# 
# 13. `libc.malloc_trim(0)`: This appears to be a memory management operation that might release unused memory allocated by the C library's `malloc` function. It sets the trim threshold to 0.
# 
# 14. `PrintColor(f"\n" + GetMemUsage(), color = Fore.RED)`: This line uses the `GetMemUsage` function to print the memory usage information in red color.
# 
# The main purpose of this code block is to set up the environment for inferencing. It loads the trained models and prepares the testing environment for making predictions using these models.

# In[12]:


get_ipython().run_cell_magic('time', '', '\n# Creating the testing environment:-\nif CFG.inference_req == "Y":\n    try: \n        del X, y;\n    except: \n        pass;\n        \n    prices = [\'reference_price\', \'far_price\', \'near_price\', \'bid_price\', \'ask_price\', \'wap\'];\n    \n    # Making the test environment for inferencing:-\n    import optiver2023;\n    try: \n        env = optiver2023.make_env();\n        iter_test = env.iter_test();\n        PrintColor(f"\\n---> Curating the inference environment");\n    except: \n        pass;\n    \n    # Collating a list of models to be used for inferencing:-\n    models = [];\n\n    # Loading the models for inferencing:-\n    if CFG.ML != "Y": \n        model_path = CFG.inf_path;\n        PrintColor(f"---> Loading models from the input data for the kernel - V{CFG.version_nb}\\n", \n                  color = Fore.RED);\n    elif CFG.ML == "Y": \n        model_path = CFG.mdl_path;\n        PrintColor(f"---> Loading models from the working directory for the kernel\\n");\n    \n    # Loading the models from the models dataframe:-\n    mdl_lbl = [];\n    for _, _, filename in walk(model_path):\n        mdl_lbl.extend(filename);\n\n    models = [];\n    for filename in mdl_lbl:\n        models.append(joblib.load(model_path + f"{filename}"));\n        \n    mdl_lbl    = [m.replace(r".model", "") for m in mdl_lbl];\n    model_dict = {l:m for l,m in zip(mdl_lbl, models)};\n    PrintColor(f"\\n---> Trained models\\n");    \n    pprint(np.array(mdl_lbl), width = 100, indent = 10, depth = 1);  \n       \nprint();\ncollect();  \nlibc.malloc_trim(0);\nPrintColor(f"\\n" + GetMemUsage(), color = Fore.RED); \n')


# **Cell 13**
# 
# **Explaination**: 
# 
# 1. The variables `is_train`, `is_infer`, and `N_Folds` are defined. `is_train` is set to `False`, `is_infer` is set to `True`, and `N_Folds` is set to `4`. These variables likely control whether the code is intended for training or inference and the number of folds for cross-validation.
# 
# 2. Financial data from the "train.csv" file is read into a DataFrame called `train`.
# 
# 3. `median_sizes` and `std_sizes` are computed by grouping the data by the 'stock_id' column and calculating the median and standard deviation of the 'bid_size' and 'ask_size' columns.
# 
# 4. Rows with missing values in the 'target' column are removed from the DataFrame.
# 
# 5. A function called `feat_eng` is defined for feature engineering. It takes a DataFrame as input, performs various calculations and transformations on the data, and returns the modified DataFrame.
# 
#    - Several features are created, including 'imbalance_ratio,' 'bid_ask_volume_diff,' 'mid_price,' 'bid_plus_ask_sizes,' 'median_size,' 'std_size,' and 'high_volume.'
#    
#    - Combinations of price columns are used to create additional features, such as the difference between prices and imbalance ratios.
#    
#    - Combinations of three price columns are used to create more features.
#    
#    - Certain columns are dropped, and garbage collection is performed.
# 
# 6. The target values are extracted from the 'target' column and stored in a NumPy array called `y`.
# 
# 7. The feature engineering function `feat_eng` is applied to the training data, and the resulting DataFrame is stored in `X`.
# 
# 8. `y_min` and `y_max` are computed as the minimum and maximum values in the target variable.
# 
# 9. A dictionary called `params` is defined with various hyperparameters for a machine learning model.
# 
# 10. A K-Fold cross-validation strategy is defined with `N_Folds` folds. The variable `mae_scores` is initialized to store Mean Absolute Error (MAE) scores during cross-validation.
# 
# 11. A function called `zero_sum` is defined, which takes lists of prices and volumes and calculates the zero-sum adjusted prices.
# 
# 12. If the `is_infer` flag is set to `True`, a list called `predictions` is initialized. This list will be used to store predictions if inferencing is performed.
# 
# The code sets up the data, feature engineering, and cross-validation for a financial data prediction task. The actual training or inference steps are expected to follow in subsequent code blocks.

# In[13]:


is_train = False
is_infer = True
N_Folds = 4

train = pd.read_csv('../input/optiver-trading-at-the-close/train.csv')
median_sizes = train.groupby('stock_id')['bid_size'].median() + train.groupby('stock_id')['ask_size'].median()
std_sizes = train.groupby('stock_id')['bid_size'].std() + train.groupby('stock_id')['ask_size'].std()
train = train.dropna(subset=['target'])

def feat_eng(df):
    
    cols = [c for c in df.columns if c not in ['row_id', 'time_id']]
    df = df[cols]
    df['imbalance_ratio'] = df['imbalance_size'] / df['matched_size']
    df['bid_ask_volume_diff'] = df['ask_size'] - df['bid_size']
    df['mid_price'] = (df['ask_price'] + df['bid_price']) / 2
    df['bid_plus_ask_sizes'] = df['bid_size'] + df['ask_size']
    df['median_size'] = df['stock_id'].map(median_sizes.to_dict())
    df['std_size'] = df['stock_id'].map(std_sizes.to_dict())
    df['high_volume'] = np.where(df['bid_plus_ask_sizes'] > df['median_size'], 1, 0)
        
    prices = ['reference_price','far_price', 'near_price', 'ask_price', 'bid_price', 'wap']
    
    for c in combinations(prices, 2):
        df[f'{c[0]}_minus_{c[1]}'] = (df[f'{c[0]}'] - df[f'{c[1]}']).astype(np.float32)
        df[f'{c[0]}_{c[1]}_imb'] = df.eval(f'({c[0]}-{c[1]})/({c[0]}+{c[1]})')

    for c in combinations(prices, 3):
        
        max_ = df[list(c)].max(axis=1)
        min_ = df[list(c)].min(axis=1)
        mid_ = df[list(c)].sum(axis=1)-min_-max_

        df[f'{c[0]}_{c[1]}_{c[2]}_imb2'] = (max_-mid_)/(mid_-min_)
    
        
    df.drop(columns=[
        'date_id', 
    ], inplace=True)
        
    gc.collect()
    
    return df

y = train['target'].values
X = feat_eng(train.drop(columns='target'))

y_min = np.min(y)
y_max = np.max(y)

params = {
    'learning_rate': 0.018,
    'max_depth': 9,
    'n_estimators': 600,
    'num_leaves': 440,
    'objective': 'mae',
    'random_state': 42,
    'reg_alpha': 0.01,
    'reg_lambda': 0.01
}


kf = KFold(n_splits=N_Folds, shuffle=True, random_state=42)
mae_scores = []


def zero_sum(prices, volumes):
    std_error = np.sqrt(volumes)
    step = np.sum(prices)/np.sum(std_error)
    out = prices-std_error*step
    
    return out


if is_infer:
    predictions = []


# ## Target Variable Calculation
# 

# **Cell 14**
# 
# **Explaination**: 
# 
# A function `feature_eng(df)` for feature engineering. The function takes a DataFrame `df` as input and performs various feature engineering operations. Here's an explanation of the feature engineering steps:
# 
# 1. Columns not relevant for feature engineering, including 'row_id,' 'date_id,' and 'time_id,' are dropped from the DataFrame.
# 
# 2. The following features are created and added to the DataFrame:
#    - 'imbalance_ratio': The ratio of 'imbalance_size' to 'matched_size.'
#    - 'bid_ask_volume_diff': The difference between 'ask_size' and 'bid_size.'
#    - 'bid_plus_ask_sizes': The sum of 'bid_size' and 'ask_size.'
#    - 'mid_price': The average price between 'ask_price' and 'bid_price.'
#    - 'median_size': The median size of 'bid_size' and 'ask_size' for each 'stock_id.'
#    - 'std_size': The standard deviation of 'bid_size' and 'ask_size' for each 'stock_id.'
#    - 'max_size': The maximum size of 'bid_size' and 'ask_size' for each 'stock_id.'
#    - 'min_size': The minimum size of 'bid_size' and 'ask_size' for each 'stock_id.'
#    - 'mean_size': The mean size of 'bid_size' and 'ask_size' for each 'stock_id.'
#    - 'first_size': The first size of 'bid_size' and 'ask_size' for each 'stock_id.'
#    - 'last_size': The last size of 'bid_size' and 'ask_size' for each 'stock_id.'
#    - 'high_volume': A binary flag indicating whether 'bid_plus_ask_sizes' is greater than 'median_size.'
# 
# 3. Price combinations are used to create additional features. For each combination of two prices, the following features are created:
#    - The price difference: (e.g., 'reference_price_minus_far_price')
#    - The price imbalance ratio: (e.g., 'reference_price_far_price_imb')
# 
# 4. For combinations of three prices, the feature 'imb2' is calculated as follows:
#    - The maximum price in the combination
#    - The minimum price in the combination
#    - The sum of the prices minus the minimum and maximum
#    - The feature is computed as (max - mid) / (mid - min)
# 
# 5. Encoding of the 'imbalance_buy_sell_flag' column into dummy variables is performed. The resulting dummy variables are concatenated with the DataFrame.
# 
# 6. Garbage collection is called to release memory.
# 
# The function applies these feature engineering transformations to the input DataFrame and returns the modified DataFrame with additional features.
# 
# Note: The code contains some additional lines that are currently commented out, such as features related to 'seconds_in_bucket.' You can uncomment and customize these lines as needed for your analysis.

# In[14]:


median_sizes = train.groupby('stock_id')['bid_size'].median() + train.groupby('stock_id')['ask_size'].median()
std_sizes = train.groupby('stock_id')['bid_size'].std() + train.groupby('stock_id')['ask_size'].std()
max_sizes = train.groupby('stock_id')['bid_size'].max() + train.groupby('stock_id')['ask_size'].max()
min_sizes = train.groupby('stock_id')['bid_size'].min() + train.groupby('stock_id')['ask_size'].min()
mean_sizes = train.groupby('stock_id')['bid_size'].mean() + train.groupby('stock_id')['ask_size'].mean()
first_sizes = train.groupby('stock_id')['bid_size'].first() + train.groupby('stock_id')['ask_size'].first()
last_sizes = train.groupby('stock_id')['bid_size'].last() + train.groupby('stock_id')['ask_size'].last()

def feature_eng(df):
    cols = [c for c in df.columns if c not in ['row_id', 'date_id','time_id']]
    df = df[cols]
    
    
    df['imbalance_ratio'] = df['imbalance_size'] / df['matched_size']
    
    df['bid_ask_volume_diff'] = df['ask_size'] - df['bid_size']
    
    df['bid_plus_ask_sizes'] = df['bid_size'] + df['ask_size']
    
    
    df['mid_price'] = (df['ask_price'] + df['bid_price']) / 2
    
    
    df['median_size'] = df['stock_id'].map(median_sizes.to_dict())
    df['std_size'] = df['stock_id'].map(std_sizes.to_dict())
    df['max_size'] = df['stock_id'].map(max_sizes.to_dict())
    df['min_size'] = df['stock_id'].map(min_sizes.to_dict())
    df['mean_size'] = df['stock_id'].map(mean_sizes.to_dict())
    df['first_size'] = df['stock_id'].map(first_sizes.to_dict())    
    df['last_size'] = df['stock_id'].map(last_sizes.to_dict())       
    
    
    df['high_volume'] = np.where(df['bid_plus_ask_sizes'] > df['median_size'], 1, 0)
    
    prices = ['reference_price', 'far_price', 'near_price', 'ask_price', 'bid_price', 'wap']
    
    
    for c in combinations(prices, 2):
        df[f'{c[0]}_minus_{c[1]}'] = (df[f'{c[0]}'] - df[f'{c[1]}']).astype(np.float32)
        df[f'{c[0]}_{c[1]}_imb'] = df.eval(f'({c[0]} - {c[1]})/({c[0]} + {c[1]})')
        
    for c in combinations(prices, 3):
        max_ = df[list(c)].max(axis=1)
        min_ = df[list(c)].min(axis=1)
        mid_ = df[list(c)].sum(axis=1) - min_ - max_
        
        df[f'{c[0]}_{c[1]}_{c[2]}_imb2'] = (max_-mid_)/(mid_-min_)
    
    #另外的特征
    #df['less_5min'] = df['seconds_in_bucket'].apply(lambda x: 1 if x < 300 else 0)
    #df['5min_8min'] = df['seconds_in_bucket'].apply(lambda x: 1 if 300 <= x else 0)
    #df['more_8min'] = df['seconds_in_bucket'].apply(lambda x: 1 if 480 <= x else 0)
        
    df_encoded = pd.get_dummies(df['imbalance_buy_sell_flag'])
    df_encoded = df_encoded.rename(columns={-1: 'sell-side imbalance', 0: 'no imbalance', 1: 'buy-side imbalance'})

    df = pd.concat([df, df_encoded], axis=1)
    
    
    gc.collect()
    
    return df


# ## Inference Loop for Test Data
# 

# **Cell 15**
# 
# **Explaination**: 
# 
# 1. `y = train['target'].values`: You're extracting the 'target' column from the `train` DataFrame and converting it into a NumPy array. This will be the target variable for your machine learning model.
# 
# 2. `y_mean = train['target'].mean()`: You're calculating the mean (average) of the 'target' column, which represents the mean of the target values in your training dataset. This can be useful for understanding the central tendency of the target variable.
# 
# 3. `y_min = np.min(y)`: You're using NumPy to calculate the minimum value of the target variable 'y.' This provides the minimum value of the target variable in your training dataset.
# 
# 4. `y_max = np.max(y)`: Similarly, you're using NumPy to calculate the maximum value of the target variable 'y.' This provides the maximum value of the target variable in your training dataset.
# 
# These statistics (mean, minimum, and maximum) can be helpful in understanding the distribution and range of the target variable, which can be important for model evaluation and interpretation.

# In[15]:


y = train['target'].values
y_mean = train['target'].mean()
y_min = np.min(y)
y_max = np.max(y)


# ## Real-time Inference and Submission
# 

# **Cell 16**
# 
# **Explaination**: 
# 
# 
# 1. You start a loop to iterate through the test data provided by the environment, which simulates real-time inference.
# 
# 2. For each iteration of the loop, you perform the following steps:
# 
#    a. Merge the `test` data with `median_vol` data for feature engineering.
# 
#    b. Make predictions using models trained during the previous phase. The predictions are stored in the `sample_prediction` DataFrame.
# 
#    c. Apply a formula to clip and blend the model predictions and create a final prediction for the current iteration.
# 
#    d. If you have a PCA-based model, perform PCA computations and make predictions using the PCA model. Then, blend these predictions with the ones from other models.
# 
#    e. Handle missing stock_ids in the test data and get the filled pivot table.
# 
#    f. Merge PCA_WAP columns with the test dataset for PCA-based predictions.
# 
#    g. Perform feature engineering and make predictions using the PCA model.
# 
#    h. Update the final prediction by blending PCA-based predictions with the previous blended predictions.
# 
#    i. Finally, submit the predictions for this iteration.
# 
# 3. You repeat the loop for all iterations provided by the environment.
# 
# 4. After completing the inference loop, you display the submission file, which contains your model's predictions for the test data.
# 
# This code performs real-time inference for the test data, blends predictions from different models, and submits the final predictions to the environment. It's also important to note that PCA-based features and predictions are included as part of the inference process.

# In[16]:


get_ipython().run_cell_magic('time', '', '\n# Load PCA Loadings\npca_loadings = pd.read_csv(\'/kaggle/input/sectors-industries-fiesta-pca-magic/principal_components.csv\')\npca_loadings.fillna(0, inplace=True)\n\n# Extract all possible stock_ids from the PCA loadings\nall_stock_ids = pca_loadings[\'stock_id\'].unique()\n\n# Function to fill price pivot\ndef get_filled_price_pivot(test, all_stock_ids):\n    price_pivot = test.pivot_table(index=[\'date_id\', \'seconds_in_bucket\'], \n                                   columns=\'stock_id\', \n                                   values=\'wap\', \n                                   fill_value=0.0)\n    \n    missing_stock_ids = set(all_stock_ids) - set(price_pivot.columns)\n    for stock_id in missing_stock_ids:\n        price_pivot[stock_id] = 0.0\n    price_pivot = price_pivot[sorted(price_pivot.columns)]\n    \n    return price_pivot, missing_stock_ids\n\nif CFG.inference_req == "Y":\n    print();\n    counter = 0;\n    \n    try:\n        median_vol = pd.read_csv(CFG.path + f"MedianVolV2.csv", index_col = [\'Unnamed: 0\']);\n    except:\n        median_vol = pd.read_csv(CFG.path + f"MedianVolV2.csv"); \n    median_vol.index.name = "stock_id";\n    median_vol = median_vol[[\'overall_medvol\', "first5min_medvol", "last5min_medvol"]];\n    \n    for test, revealed_targets, sample_prediction in iter_test:\n        if counter >= 99: num_space = 1;\n        elif counter >= 9: num_space = 2;\n        else: num_space = 3;\n        \n        PrintColor(f"{counter + 1}. {\' \' * num_space} Inference", color = Fore.MAGENTA);\n        testforsecond = copy.deepcopy(test)\n        test  = test.merge(median_vol, how = "left", left_on = "stock_id", right_index = True);\n        Xtest = MakeFtre(test, prices = prices);\n        del num_space;\n        \n        # Curating model predictions across methods and folds:-        \n        preds = pd.DataFrame(columns = CFG.methods, index = Xtest.index).fillna(0);\n        for method in CFG.methods:\n            for mdl_lbl, mdl in model_dict.items():\n                if mdl_lbl.startswith(f"{method}V{CFG.version_nb}"):\n                    if CFG.test_req == "Y":\n                        print(mdl_lbl);\n                    else:\n                        pass;\n                    preds[method] = preds[method] + mdl.predict(Xtest)/ (CFG.n_splits * CFG.n_repeats);\n        \n        # Curating the weighted average model predictions:-    \n        \n        sample_prediction[\'target\'] = \\\n        np.average(preds.values, weights= CFG.ens_weights, axis=1);\n        \n        # Source - https://www.kaggle.com/code/kaito510/goto-conversion-optiver-baseline-models     \n        sample_prediction[\'target\'] = \\\n        zero_sum(sample_prediction[\'target\'], test.loc[:,\'bid_size\'] + test.loc[:,\'ask_size\'])\n        \n        #35\n        \n        feat = feat_eng(testforsecond)\n        fold_prediction = 0\n        for fold in range(0, N_Folds):\n            model_filename = f"/kaggle/input/lgb-models-optv2/model_fold_{fold+1}.pkl"\n            m = joblib.load(model_filename)\n            fold_prediction += m.predict(feat)  \n        \n        for fold in range(0, N_Folds):\n            model_filename = f"/kaggle/input/lgb-kf-with-optuna/model_fold_{fold+1}.pkl"\n            m = joblib.load(model_filename)\n            fold_prediction += m.predict(feat)   \n        \n        fold_prediction /= (2 * N_Folds)\n        fold_prediction = zero_sum(fold_prediction, test.loc[:,\'bid_size\'] + test.loc[:,\'ask_size\'])\n        clipped_predictions = np.clip(fold_prediction, y_min, y_max)\n        \n        sample_prediction[\'target\'] = 0.7 * clipped_predictions + 0.3 * sample_prediction[\'target\']\n        print(\'here5\')\n#         feat = feature_eng(testforsecond)\n#         fold_prediction = 0\n#         for fold in range(0, N_Folds):\n#             model_filename = f"/kaggle/input/lgb-baseline-train/lgb-models-optv2/model_fold_{fold+1}.pkl"\n#             m = joblib.load(model_filename)\n#             fold_prediction += m.predict(feat)   \n        \n#         fold_prediction /= N_Folds\n#         fold_prediction = zero_sum(fold_prediction, test.loc[:,\'bid_size\'] + test.loc[:,\'ask_size\'])\n#         clipped_predictions = np.clip(fold_prediction, y_min, y_max)\n        \n#         sample_prediction[\'target\'] = 0.8 * clipped_predictions + 0.2 * sample_prediction[\'target\']\n        \n        ####################### PCA submission\n        # Handle missing stock_ids and get the filled pivot table\n        test = copy.deepcopy(testforsecond)\n        price_pivot, missing_stocks = get_filled_price_pivot(test, all_stock_ids)\n        price_pivot.fillna(0, inplace=True)\n        # PCA computations\n        pca_wap_df = pd.DataFrame(index=price_pivot.index)\n        for i in range(1, 5):\n            pca_wap_df[f\'PCA_WAP_{i}\'] = (price_pivot.values * pca_loadings.set_index(\'stock_id\')[f\'PC{i}\'].values).sum(axis=1)\n        pca_wap_df = pca_wap_df.reset_index()\n        # Merge PCA_WAP columns with the test dataset\n        test = test.merge(pca_wap_df, on=[\'date_id\', \'seconds_in_bucket\'], how=\'left\')\n        # Feature engineering and model predictions\n        feat = feature_eng(test)\n        fold_prediction = 0\n        print(\'here6\')\n        for fold in range(0, N_Folds):\n            model_filename = f"/kaggle/input/bridging-gaps-linking-sectors-through-pca/lgb-modelos-con-sectores-y-pca/model_fold_{fold+1}.pkl"\n            m = joblib.load(model_filename)\n            fold_prediction += m.predict(feat)\n        fold_prediction /= N_Folds\n        # Remove predictions for missing stocks\n        if missing_stocks:\n            test = test[~test[\'stock_id\'].isin(missing_stocks)]\n        fold_prediction = zero_sum(fold_prediction, test.loc[:, \'bid_size\'] + test.loc[:, \'ask_size\'])\n        # Replace problematic values (NaNs, Nones, etc.) in fold_prediction with y_mean\n        fold_prediction = np.where(np.isnan(fold_prediction), sample_prediction[\'target\'].values, fold_prediction)\n        clipped_predictions = np.clip(fold_prediction, y_min, y_max)\n        sample_prediction[\'target\'] = 0.85 * clipped_predictions + 0.15 * sample_prediction[\'target\']\n        \n        try: \n            env.predict(sample_prediction);\n        except: \n            PrintColor(f"---> Submission did not happen as we have the file already");\n            pass;\n        \n        counter = counter+1;\n        collect();\n    \n    PrintColor(f"\\n---> Submission file\\n");\n    display(sample_prediction.head(10));\n            \nprint();\ncollect();  \nlibc.malloc_trim(0);\nPrintColor(f"\\n" + GetMemUsage(), color = Fore.RED); \n')


# ## Explore More! 👀
# Thank you for exploring this notebook! If you found this notebook insightful or if it helped you in any way, I invite you to explore more of my work on my profile.
# 
# 👉 [Visit my Profile](https://www.kaggle.com/zulqarnainali) 👈
# 
# ## Feedback and Gratitude 🙏
# We value your feedback! Your insights and suggestions are essential for our continuous improvement. If you have any comments, questions, or ideas to share, please don't hesitate to reach out.
# 
# 📬 Contact me via email: [zulqar445ali@gmail.com](mailto:zulqar445ali@gmail.com)
# 
# I would like to express our heartfelt gratitude for your time and engagement. Your support motivates us to create more valuable content.
# 
# Happy coding and best of luck in your data science endeavors! 🚀
# 
