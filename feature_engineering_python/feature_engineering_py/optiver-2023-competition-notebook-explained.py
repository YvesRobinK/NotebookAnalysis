#!/usr/bin/env python
# coding: utf-8

# # Optiver 2023 Trading Competition Notebook 📈📊
# 
# ## Introduction 🌟
# Welcome to this Jupyter notebook developed for the Optiver 2023 Trading Competition! This notebook is designed to help you participate in the competition and make predictions on the provided financial dataset.
# 
# ### Inspiration and Credits 🙌
# This notebook is inspired by the work of Yuanzhe Zhou, available at [this Kaggle project](https://www.kaggle.com/code/yuanzhezhou/baseline-lgb-xgb-and-catboost/). We extend our gratitude to Yuanzhe Zhou for sharing their insights and code.
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
# 5. **Conclusion**: We summarize the key findings and results.
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
# We acknowledge the Optiver 2023 Trading Competition organizers for providing the dataset and the competition platform.
# 
# Let's get started! Feel free to reach out if you have any questions or need assistance along the way.
# 👉 [Visit my Profile](https://www.kaggle.com/zulqarnainali) 👈

# ## 📚 Import the necessary libraries
# 

# In[1]:


# 📦 Import necessary libraries
import pandas as pd         # 🐼 Pandas for data manipulation
import lightgbm as lgb      # 🚥 LightGBM for gradient boosting
import xgboost as xgb       # 🌲 XGBoost for gradient boosting
import catboost as cbt      # 🐱 CatBoost for gradient boosting
import numpy as np          # 🔢 NumPy for numerical operations
import joblib               # 📦 Joblib for model serialization
import os                   # 📂 OS module for file operations
import optiver2023          # custom module


# ## 📋 Function to Calculate Imbalance Features
# 

# 
# 1. `def calculate_imbalance_features(df):`:
#    - This line defines a Python function called `calculate_imbalance_features` that takes a DataFrame `df` as an input.
# 
# 2. `df['imb_s1'] = df.eval('(bid_size - ask_size) / (bid_size + ask_size)')`:
#    - This line creates a new column in the DataFrame `df` called `'imb_s1'`.
#    - The values in this column are calculated using the `eval` method, which evaluates the mathematical expression `(bid_size - ask_size) / (bid_size + ask_size)` for each row in the DataFrame.
#    - This expression calculates the bid-ask imbalance, a measure of the difference between the sizes of buy (bid) and sell (ask) orders in a financial dataset.
# 
# 
# Moving on to the next line:
# 
# ```python
#     # Calculate and add imbalance feature 2 (imb_s2)
#     df['imb_s2'] = df.eval('(imbalance_size - matched_size) / (matched_size + imbalance_size)')
# ```
# 
# 3. `df['imb_s2'] = df.eval('(imbalance_size - matched_size) / (matched_size + imbalance_size)')`:
#    - Similar to the previous line, this line creates a new column in the DataFrame `df` called `'imb_s2'`.
#    - It calculates the values for this column using the `eval` method, which evaluates the expression `(imbalance_size - matched_size) / (matched_size + imbalance_size)` for each row in the DataFrame.
#    - This expression calculates another measure related to the size of imbalanced orders in a financial dataset.
# 
# 
# The `calculate_imbalance_features` function takes a DataFrame as input and calculates two financial metrics, 'imb_s1' and 'imb_s2', which represent bid-ask imbalance and another measure related to order size imbalances, respectively.

# In[2]:


def calculate_imbalance_features(df):
    # 📈 Calculate and add imbalance feature 1 (imb_s1)
    df['imb_s1'] = df.eval('(bid_size - ask_size) / (bid_size + ask_size)')  

    # 🔃 Calculate and add imbalance feature 2 (imb_s2)
    df['imb_s2'] = df.eval('(imbalance_size - matched_size) / (matched_size + imbalance_size)') 

    return df


# ## Function to Calculate Price-Based Features

# Explanation:
# 
# 1. `def calculate_price_features(df, features):`:
#    - This line defines a Python function called `calculate_price_features` that takes two parameters: a DataFrame `df` and a list `features`.
# 
# 2. `prices = ['reference_price', 'far_price', 'near_price', 'ask_price', 'bid_price', 'wap']`:
#    - This line creates a list called `prices` containing the names of various price-related columns.
# 
# 3. `for i, a in enumerate(prices):`:
#    - This line starts a loop that iterates over the elements in the `prices` list, and it uses `i` to keep track of the index and `a` to store the name of the first price.
# 
# 4. `for j, b in enumerate(prices):`:
#    - Inside the previous loop, this line starts another loop that iterates over the same `prices` list. It uses `j` to keep track of the index and `b` to store the name of the second price.
# 
# 5. `if i > j:`:
#    - This line checks if the index of the first price (a) is greater than the index of the second price (b). This condition ensures that we only calculate features where the first price comes after the second price in the list.
# 
# 6. `df[f'{a}_{b}_imb'] = df.eval(f'({a} - {b}) / ({a} + {b})')`:
#    - If the condition in the previous line is met, this line calculates a new feature by subtracting the second price (b) from the first price (a) and then dividing the result by the sum of the two prices.
#    - The calculated feature is added as a new column in the DataFrame with a name in the format `{a}_{b}_imb`, where `a` and `b` are the names of the prices being compared.
# 
# 7. `features.append(f'{a}_{b}_imb')`:
#    - This line adds the name of the newly created feature to the `features` list, keeping track of all the features that have been calculated.
# 
# 8. `return df, features`:
#    - Finally, the function returns the modified DataFrame (`df`) with the added features and the updated list of features (`features`).
# 
# This code is a function that calculates and adds price-related features to a DataFrame based on the differences and ratios between various price columns. It iterates through pairs of price columns and calculates features for pairs where the first price comes after the second price in the list. The function then returns the modified DataFrame and the list of calculated features.

# In[3]:


def calculate_price_features(df, features):
    # Define a list of price-related columns
    prices = ['reference_price', 'far_price', 'near_price', 'ask_price', 'bid_price', 'wap']

    # Loop through the price columns to create new features
    for i, a in enumerate(prices):
        for j, b in enumerate(prices):
            # Check if the first price (a) comes after the second price (b) in the list
            if i > j:
                # Calculate and add a new feature to the DataFrame
                df[f'{a}_{b}_imb'] = df.eval(f'({a} - {b}) / ({a} + {b})')
                # Add the new feature name to the list of features
                features.append(f'{a}_{b}_imb')

    # Return the modified DataFrame and the updated list of features
    return df, features


# ## Function to Calculate Additional Price-Based Features

# Explanation:
# 
# 1. `def calculate_additional_price_features(df, features):`:
#    - This line defines a Python function called `calculate_additional_price_features` that takes two parameters: a DataFrame `df` and a list `features`.
# 
# 2. `prices = ['reference_price', 'far_price', 'near_price', 'ask_price', 'bid_price', 'wap']`:
#    - This line creates a list called `prices` containing the names of various price-related columns.
# 
# 3. `for i, a in enumerate(prices):`:
#    - This line starts a loop that iterates over the elements in the `prices` list. It uses `i` to keep track of the index and `a` to store the name of the first price.
# 
# 4. `for j, b in enumerate(prices):`:
#    - Inside the previous loop, this line starts another loop that iterates over the same `prices` list. It uses `j` to keep track of the index and `b` to store the name of the second price.
# 
# 5. `for k, c in enumerate(prices):`:
#    - Inside the second loop, this line starts yet another loop that iterates over the `prices` list. It uses `k` to keep track of the index and `c` to store the name of the third price.
# 
# 6. `if i > j and j > k:`:
#    - This line checks if the indices of the prices a, b, and c are in descending order, ensuring that we consider combinations where a > b > c.
# 
# 7. `max_ = df[[a, b, c]].max(axis=1)`:
#    - This line calculates the maximum value among the prices a, b, and c for each row in the DataFrame `df`.
# 
# 8. `min_ = df[[a, b, c]].min(axis=1)`:
#    - This line calculates the minimum value among the prices a, b, and c for each row in the DataFrame `df`.
# 
# 9. `mid_ = df[[a, b, c]].sum(axis=1) - min_ - max_`:
#    - This line calculates the middle value among the prices a, b, and c for each row in the DataFrame `df` by subtracting the minimum and maximum values from the sum.
# 
# 10. `df[f'{a}_{b}_{c}_imb2'] = (max_ - mid_) / (mid_ - min_)`:
#     - If the condition in line 6 is met, this line calculates a new feature using the max, min, and mid values for a, b, and c.
#     - The calculated feature is added as a new column in the DataFrame with a name in the format `{a}_{b}_{c}_imb2`.
# 
# 11. `features.append(f'{a}_{b}_{c}_imb2')`:
#     - This line adds the name of the newly created feature to the `features` list, keeping track of all the features that have been calculated.
# 
# 12. `return df, features`:
#     - Finally, the function returns the modified DataFrame (`df`) with the added features and the updated list of features (`features`).
# 
# This code is a nested loop that calculates and adds price-related features to a DataFrame based on combinations of three price columns. It considers combinations where the prices are in descending order, and for each combination, it calculates a feature related to the max, min, and mid values among the selected prices. The function then returns the modified DataFrame and the list of calculated features.

# In[4]:


def calculate_additional_price_features(df, features):
    # Define a list of price-related columns
    prices = ['reference_price', 'far_price', 'near_price', 'ask_price', 'bid_price', 'wap']

    # Loop through the price columns to create new features
    for i, a in enumerate(prices):
        for j, b in enumerate(prices):
            for k, c in enumerate(prices):
                # Check if the order of prices a, b, and c is descending
                if i > j and j > k:
                    # Calculate the maximum, minimum, and mid values among a, b, and c
                    max_ = df[[a, b, c]].max(axis=1)
                    min_ = df[[a, b, c]].min(axis=1)
                    mid_ = df[[a, b, c]].sum(axis=1) - min_ - max_

                    # Calculate and add a new feature to the DataFrame
                    df[f'{a}_{b}_{c}_imb2'] = (max_ - mid_) / (mid_ - min_)
                    # Add the new feature name to the list of features
                    features.append(f'{a}_{b}_{c}_imb2')

    # Return the modified DataFrame and the updated list of features
    return df, features


# ## Function to Generate Features from DataFrame

# Explanation:
# 
# 1. `def generate_features(df):`:
#    - This line defines a Python function called `generate_features` that takes a DataFrame `df` as input.
# 
# 2. `features = ['seconds_in_bucket', 'imbalance_buy_sell_flag', ...]`:
#    - This line creates a list called `features` containing the names of selected feature columns that will be used to generate the final feature set.
# 
# 3. `df = calculate_imbalance_features(df)`:
#    - This line calls the `calculate_imbalance_features` function to calculate imbalance-related features based on the input DataFrame `df`. It updates the DataFrame `df` with these new features.
#    
# 4. `df, features = calculate_price_features(df, features)`:
#    - This line calls the `calculate_price_features` function to calculate price-related features based on price differences and adds them to the input DataFrame `df`. It also updates the `features` list to include the names of these newly calculated features.
#    
# 5. `df, features = calculate_additional_price_features(df, features)`:
#    - This line calls the `calculate_additional_price_features` function to calculate additional price-related features based on combinations of price differences. It adds these features to the input DataFrame `df` and updates the `features` list with their names.
# 
# 6. `return df[features]`:
#    - Finally, the function returns a DataFrame that includes only the selected features listed in the `features` list. This DataFrame represents the final set of features that will be used for further analysis or modeling.
# 
# This function takes an initial DataFrame as input, calculates various features related to imbalance and price differences, and returns a DataFrame containing the selected set of features for further analysis or modeling.

# In[5]:


def generate_features(df):
    # Define the list of feature column names
    features = ['seconds_in_bucket', 'imbalance_buy_sell_flag',
                'imbalance_size', 'matched_size', 'bid_size', 'ask_size',
                'reference_price', 'far_price', 'near_price', 'ask_price', 'bid_price', 'wap',
                'imb_s1', 'imb_s2'
               ]
    
    # Calculate imbalance features
    df = calculate_imbalance_features(df)  # 📊 Calculate imbalance features
    
    # Calculate features based on price differences
    df, features = calculate_price_features(df, features)  # 💰 Calculate price-related features
    
    # Calculate additional features based on price differences
    df, features = calculate_additional_price_features(df, features)  # 🔄 Calculate additional price features
    
    # Return the DataFrame with selected features
    return df[features]


# ## Training

# Explanation:
# 
# 1. `TRAINING = True`:
#    - This line defines a boolean variable `TRAINING` and sets it to `True`. It indicates that we are in a training mode, which suggests that the following code is meant for training purposes.
# 
# 2. `if TRAINING:`:
#    - This line starts an `if` statement that checks if the `TRAINING` variable is `True`. If it is `True`, it proceeds to the next block of code.
# 
# 3. `df_train = pd.read_csv('/kaggle/input/optiver-trading-at-the-close/train.csv')`:
#    - Inside the `if` block, this line reads the training data from a CSV file located at the specified path (`'/kaggle/input/optiver-trading-at-the-close/train.csv'`) using the Pandas library. It loads the data into a DataFrame called `df_train`.
# 
# 4. `df_ = generate_features(df_train)`:
#    - After reading the training data, this line calls the `generate_features` function to generate features based on the training data. It passes the `df_train` DataFrame as an argument.
#    - The resulting DataFrame with selected features is assigned to the variable `df_`.
# 
# This code block is meant for training a machine learning model or performing data analysis. It first checks if the `TRAINING` variable is `True`, indicating that it's in a training mode. If so, it reads the training data from a CSV file, generates features based on that data, and stores the resulting DataFrame in the variable `df_`.

# In[6]:


TRAINING = False

if TRAINING:
    # Read the training data from a CSV file
    df_train = pd.read_csv('/kaggle/input/optiver-trading-at-the-close/train.csv')  # 📁 Read training data
    
    # Generate features for the training data
    df_ = generate_features(df_train)  # 🏋️‍♂️ Generate features for training data


# ## pre-trained models loading

# Explanation:
# 
# 1. `os.system('mkdir models')`:
#    - This line creates a directory named 'models' using the `os.system` command. It's used for storing trained models.
# 
# 2. `model_path = '/kaggle/input/optiverbaselinezyz'`:
#    - This line sets the path to the directory where pre-trained models are located or where models will be saved during training.
# 
# 3. `N_fold = 5`:
#    - This line defines the number of folds for cross-validation.
# 
# 4. `if TRAINING:`:
#    - This block of code is executed only if the `TRAINING` flag is `True`. It's used for training models.
# 
# 5. Data preparation:
#    - This section prepares the input features `X` and the target variable `Y` for model training. It also filters out rows with non-finite target values and creates an index array for data splitting.
# 
# 6. `models = []`:
#    - This line initializes an empty list to store trained models.
# 
# 7. `def train(model_dict, modelname='lgb'):`:
#    - This defines a function `train` for training a model. It takes a model dictionary and a model name as parameters.
# 
# 8. Model training:
#    - This section fits the selected model on the training data and appends the trained model to the `models` list. It also saves the trained model to a file.
# 
# 9. Model dictionary:
#    - This section defines a dictionary `model_dict` that maps model names to their respective model objects.
# 
# 10. Cross-validation loop:
#     - This loop iterates `N_fold` times and trains models using different folds of the data. It calls the `train` function for each model and fold.
# 
# This code block is for training machine learning models using cross-validation. It also allows for the loading of pre-trained models if not in training mode.

# In[7]:


# Create a directory named 'models'
os.system('mkdir models')  # 📁 Create a directory for storing models

# Set the path for model loading
model_path = '/kaggle/input/optiverbaselinezyz'

# Define the number of folds for cross-validation
N_fold = 5

if TRAINING:
    # Prepare the input features and target variable
    X = df_.values
    Y = df_train['target'].values

    # Filter out rows with non-finite target values
    X = X[np.isfinite(Y)]
    Y = Y[np.isfinite(Y)]

    # Create an index array for data splitting
    index = np.arange(len(X))

# Initialize a list to store trained models
models = []

# Define a function for training a model
def train(model_dict, modelname='lgb'):
    if TRAINING:
        # Get the model from the model dictionary
        model = model_dict[modelname]
        
        # Fit the model on the training data
        model.fit(X[index % N_fold != i], Y[index % N_fold != i], 
                  eval_set=[(X[index % N_fold == i], Y[index % N_fold == i])], 
                  verbose=10, 
                  early_stopping_rounds=100
                 )
        
        # Append the trained model to the models list
        models.append(model)
        
        # Save the model to a file
        joblib.dump(model, './models/{modelname}_{i}.model')
    else:
        # Load a pre-trained model if not in training mode
        models.append(joblib.load(f'{model_path}/{modelname}_{i}.model'))
    return 

# Define a dictionary of models to train
model_dict = {
    'lgb': lgb.LGBMRegressor(objective='regression_l1', n_estimators=500),
    'xgb': xgb.XGBRegressor(tree_method='hist', objective='reg:absoluteerror', n_estimators=500),
    'cbt': cbt.CatBoostRegressor(objective='MAE', iterations=3000),
}

# Loop for training models using cross-validation
for i in range(N_fold):
    train(model_dict, 'lgb')  # Train LightGBM model
#     train(model_dict, 'xgb')  # Train XGBoost model (commented out)
    train(model_dict, 'cbt')  # Train CatBoost model


# ## Setting Up Optiver 2023 Environment and Initializing Test Data Iterator

# Explanation:
# 
# 1. `env = optiver2023.make_env()`:
#    - This line creates an environment for working with the Optiver 2023 competition. The `optiver2023` module is used to create this environment. It is a custom environment or module provided for the competition.
# 
# 2. `iter_test = env.iter_test()`:
#    - This line initializes an iterator for the testing data within the Optiver 2023 environment. The iterator allows you to loop through and access the testing data in an organized and efficient manner.
# 
# 

# In[8]:


# Create an Optiver 2023 environment
env = optiver2023.make_env()  # 🏭 Create an Optiver 2023 environment

# Initialize an iterator for testing data
iter_test = env.iter_test()  # 🔄 Initialize an iterator for testing data


# ## creating submission file 

# Explanation:
# 
# 1. `counter = 0`:
#    - This line initializes a counter variable to keep track of the number of iterations through the test data. The emoji 🔢 indicates that it's a numeric counter.
# 
# 2. `for (test, revealed_targets, sample_prediction) in iter_test:`:
#    - This line starts a `for` loop that iterates through the test data provided by the Optiver 2023 environment. The loop unpacks each tuple, containing `test`, `revealed_targets`, and `sample_prediction`, for processing.
# 
# 3. Generating Features:
#    - Inside the loop, this line calls the `generate_features` function to generate features for the test data. The emoji 🏋️‍♂️ suggests that it's a data transformation step.
# 
# 4. Making Predictions:
#    - This line calculates predictions for the target variable using the trained models (`models`) and computes the mean of those predictions. It assigns the calculated mean as the 'target' in the `sample_prediction` DataFrame.
# 
# 5. Submitting Predictions:
#    - The `env.predict(sample_prediction)` line submits the predictions to the Optiver 2023 environment for evaluation and scoring. The emoji 📤 indicates the action of submitting predictions.
# 
# 6. `counter += 1`:
#    - After processing each test data batch, this line increments the counter to keep track of the number of iterations through the test data.
# 
# This code block is used to iterate through the test data, generate features, make predictions using trained models, submit the predictions to the competition environment, and keep track of the iteration count.

# In[9]:


# Initialize a counter
counter = 0  # 🔢 Initialize a counter

# Iterate through the test data provided by the Optiver 2023 environment
for (test, revealed_targets, sample_prediction) in iter_test:  # 🔄 Iterate through the test data
    
    # Generate features for the test data
    feat = generate_features(test)  # 🏋️‍♂️ Generate features for the test data
    
    # Make predictions using the trained models and compute the mean
    sample_prediction['target'] = np.mean([model.predict(feat) for model in models], 0)
    
    # Submit the predictions to the Optiver 2023 environment
    env.predict(sample_prediction)  # 📤 Submit predictions to the environment
    
    # Increment the counter
    counter += 1  # 🔢 Increment the counter


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
