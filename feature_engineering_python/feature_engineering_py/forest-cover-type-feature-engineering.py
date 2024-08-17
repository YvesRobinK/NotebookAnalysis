#!/usr/bin/env python
# coding: utf-8

# # Notebook Overview
# 
# In this notebook, we test out several new features using an [ExtraTrees classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html) from the scikit-learn library. In particular, we will do the following:
# 
# * Define a function for cross-validation and generating test predictions
# * Get a baseline using the original data with no feature engineering
# * Consolidate prior feature engineering from other sources
# * Create new features from the soil type columns
# * Create a submission combining all new features
# 
# The main contribution of this notebook is in the testing of features based on the soil type. Using the [information given](https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.info) with the original dataset we can extract features from the soil type which improve the cross-validation accuracy. You can find all of my work using this dataset on [github](https://github.com/rsizem2/forest-cover-type-prediction).

# In[1]:


# Global variables for testing changes to this notebook quickly
RANDOM_SEED = 0
NUM_FOLDS = 12


# # Setup
# 
# ## 1. Imports
# 
# Import all relevant libraries, models, evaluation metrics, etc.

# In[2]:


import numpy as np
import pandas as pd
import math
import scipy
import time
import pyarrow
import gc

# Model & Evaluation
from functools import partial
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier


# ## 2. Load Data
# 
# Load training/test data and encode the target variable.

# In[3]:


# Load original data
train = pd.read_csv('../input/forest-cover-type-prediction/train.csv')
test = pd.read_csv('../input/forest-cover-type-prediction/test.csv')
submission = pd.read_csv('../input/forest-cover-type-prediction/sampleSubmission.csv')

# Label Encode
encoder = LabelEncoder()
train["Cover_Type"] = encoder.fit_transform(train["Cover_Type"])


# ## 3. Scoring Function
# 
# Function for performing k-fold cross-validation and averaging the test predictions across each fold.

# In[4]:


def train_model():
    
    print(f'{NUM_FOLDS}-fold Cross Validation\n')
    
    # Train/Test split
    X_temp = train[features]
    X_test = test[features]
    y_temp = train['Cover_Type']
    
    # Store the out-of-fold predictions
    test_preds = np.zeros((X_test.shape[0],7))
    oof_preds = np.zeros((X_temp.shape[0],))
    fi_scores = np.zeros((X_temp.shape[1],))
    scores, times = np.zeros(NUM_FOLDS), np.zeros(NUM_FOLDS)
    
    # Stratified k-fold cross-validation
    skf = StratifiedKFold(n_splits = NUM_FOLDS, shuffle = True, random_state = RANDOM_SEED)
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_temp,y_temp)):
       
        # Training and Validation Sets
        X_train, X_valid = X_temp.iloc[train_idx], X_temp.iloc[valid_idx]
        y_train, y_valid = y_temp.iloc[train_idx], y_temp.iloc[valid_idx]
        
        # Create model
        start = time.time()
        model = ExtraTreesClassifier(
            n_estimators = 118,
            min_samples_split = 2,
            max_features = 14,
            random_state = RANDOM_SEED,
            n_jobs = -1,
        )
        model.fit(X_train, y_train)

        # validation/holdout predictions
        valid_preds = np.ravel(model.predict(X_valid))
        oof_preds[valid_idx] = valid_preds
        test_preds += model.predict_proba(X_test)

        # Save scores and times
        scores[fold] = accuracy_score(y_valid, valid_preds)
        end = time.time()
        times[fold] = end-start
        print(f'Fold {fold}: {round(scores[fold], 5)} in {round(times[fold], 2)}s')
        time.sleep(0.5)
    
    test_preds = np.argmax(test_preds, axis = 1)
    print('\nModel: '+model.__class__.__name__)
    print("Train Accuracy:", round(scores.mean(), 5))
    print(f'Training Time: {round(times.sum(), 2)}s')
    
    return test_preds


# # Baseline Submission
# 
# We start by training a model on the original data without any feature engineering:

# In[5]:


# Baseline submission
features = [x for x in train.columns if x not in ['Id', 'Cover_Type']] 
submission['Cover_Type'] = encoder.inverse_transform(
    train_model()
)
submission.to_csv('baseline_submission.csv', index=False)


# # General Feature Engineering
# 
# In this section we consider several new features generated from the original data. Check out the following notebooks/discussions for more information and original sources for some of these features:
# 
# * Notebook: [Two models Random Forests](https://www.kaggle.com/shouldnotbehere/two-models-random-forests)
# * Notebook: [my_first_submission](https://www.kaggle.com/jianyu/my-first-submission)
# * Discussion: [TPS12 feature engineering](https://www.kaggle.com/c/tabular-playground-series-dec-2021/discussion/293612)
# 
# The features included below are those from the previous sources which resulted in improved CV accuracy.

# In[6]:


def misc_features(data):
    df = data.copy()
    
    # Use float64 for calculations
    for col, dtype in df.dtypes.iteritems():
        if dtype.name.startswith('float'):
            df[col] = df[col].astype('float64')
            
    # Interaction Terms
    df['Horizontal_Distance_To_Roadways_Log'] = [math.log(v+1) for v in df['Horizontal_Distance_To_Roadways']]
    df['Water Elevation'] = df['Elevation'] - df['Vertical_Distance_To_Hydrology']
    df['Hydro_Fire_1'] = df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Fire_Points']
    df['Hydro_Fire_2'] = abs(df['Horizontal_Distance_To_Hydrology'] - df['Horizontal_Distance_To_Fire_Points'])
    df['Hydro_Road_1'] = abs(df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Roadways'])
    df['Hydro_Road_2'] = abs(df['Horizontal_Distance_To_Hydrology'] - df['Horizontal_Distance_To_Roadways'])
    df['Fire_Road_1'] = abs(df['Horizontal_Distance_To_Fire_Points'] + df['Horizontal_Distance_To_Roadways'])
    df['Fire_Road_2'] = abs(df['Horizontal_Distance_To_Fire_Points'] - df['Horizontal_Distance_To_Roadways'])
    df['EHiElv'] = df['Horizontal_Distance_To_Roadways'] * df['Elevation']
    df['EVDtH'] = df.Elevation - df.Vertical_Distance_To_Hydrology
    df['EHDtH'] = df.Elevation - df.Horizontal_Distance_To_Hydrology * 0.2
    df['Elev_3Horiz'] = df['Elevation'] + df['Horizontal_Distance_To_Roadways']  + df['Horizontal_Distance_To_Fire_Points'] + df['Horizontal_Distance_To_Hydrology']
    df['Elev_Road_1'] = df['Elevation'] + df['Horizontal_Distance_To_Roadways']
    df['Elev_Road_2'] = df['Elevation'] - df['Horizontal_Distance_To_Roadways']
    df['Elev_Fire_1'] = df['Elevation'] + df['Horizontal_Distance_To_Fire_Points']
    df['Elev_Fire_2'] = df['Elevation'] - df['Horizontal_Distance_To_Fire_Points']

    # Fill NA
    df.fillna(0, inplace = True)
    
    # Downcast variables
    for col, dtype in df.dtypes.iteritems():
        if dtype.name.startswith('int'):
            df[col] = pd.to_numeric(df[col], downcast ='integer')
        elif dtype.name.startswith('float'):
            df[col] = pd.to_numeric(df[col], downcast ='float')
    
    return df


# In[7]:


# misc feature engineering
train = misc_features(train)
test = misc_features(test)


# # Soil Type Feature Engineering
# 
# In this section we consider several new features based on the soil-type variables:
# 
# 0. Categorical Encoding (40 columns -> 1 column)
# 1. Climatic Zone
# 2. Geologic Zone
# 3. Surface Cover
# 4. Rock Size
# 5. Interaction Terms
# 6. Drop Original (dimension reduction)
# 
# ## ELU Codes
# 
# From the original data description, the soil type number is based on the USFS Ecological Landtype Units (ELUs). The ELU code contains further information about the soils including the climatic zone and geologic zone. Furthermore, each ELU code comes with a brief description from which we can extract further information about the surface cover (by rocks/boulders) and rock size.

# In[8]:


# Mapping soil type to ELU code
ELU_CODE = {
    1:2702,2:2703,3:2704,4:2705,5:2706,6:2717,7:3501,8:3502,9:4201,
    10:4703,11:4704,12:4744,13:4758,14:5101,15:5151,16:6101,17:6102,
    18:6731,19:7101,20:7102,21:7103,22:7201,23:7202,24:7700,25:7701,
    26:7702,27:7709,28:7710,29:7745,30:7746,31:7755,32:7756,33:7757,
    34:7790,35:8703,36:8707,37:8708,38:8771,39:8772,40:8776
}


# ## 0. Categorical Encoding
# 
# **Note:** Soil type is a single variable which has been one-hot encoded presumably to behave nicely with a neural network. Generally, tree-based models work better without explicit one-hot encoding, so we will reverse engineer the soil type. We will eventually drop the original soil type columns which has the added effect of significantly reducing the total number of features.

# In[9]:


# Encode soil type ordinally
def categorical_encoding(input_df):
    data = input_df.copy()
    data['Soil_Type'] = 0
    for i in range(1,41):
        data['Soil_Type'] += i*data[f'Soil_Type{i}']
    return data


# In[10]:


# Encode soil type
train = categorical_encoding(train)
test = categorical_encoding(test)

# Original soil features
soil_features = [f'Soil_Type{i}' for i in range(1,41)]


# ## 1. Climatic Zone (Ordinal Variable)
# 
# We create a feature based on the climatic zone of the soil. This can be determined by the first digit of the ELU code. Note that the climatic zone has a natural ordering:
# 
# 1. lower montane dry
# 2. lower montane
# 3. montane dry
# 4. montane
# 5. montane dry and montane
# 6. montane and subalpine
# 7. subalpine
# 8. alpine

# In[11]:


def climatic_zone(input_df):
    df = input_df.copy()
    df['Climatic_Zone'] = input_df['Soil_Type'].apply(
        lambda x: int(str(ELU_CODE[x])[0])
    )
    return df


# In[12]:


# Climatic Zone
train = climatic_zone(train)
test = climatic_zone(test)


# ## 2. Geologic Zone (Nominal Variable)
# 
# This is another feature that comes directly from the ELU code. The geologic zone is determined by the second digit in the ELU code. Unlike the climatic zone, the geologic zone has no natural ordering:
# 
# 1. alluvium
# 2. glacial
# 3. shale
# 4. sandstone
# 5. mixed sedimentary
# 6. unspecified in the USFS ELU Survey
# 7. igneous and metamorphic
# 8. volcanic

# In[13]:


def geologic_zone(input_df):
    df = input_df.copy()
    df['Geologic_Zone'] = input_df['Soil_Type'].apply(
        lambda x: int(str(ELU_CODE[x])[1])
    )
    return df


# In[14]:


# Geologic Zone
train = geologic_zone(train)
test = geologic_zone(test)


# ## 3. Surface Cover (Ordinal Variable)
# 
# This feature is also based on the ELU code and the original data description for each soil type. Note that not all of the soil types have a description of their surface cover. According to the [USDA reference](https://www.nrcs.usda.gov/wps/portal/nrcs/detail/soils/ref/?cid=nrcs142p2_054253#surface_fragments) on soil profiling:
# 
# 1. (Stony/Bouldery) — Stones or boulders cover 0.01 to less than 0.1 percent of the surface. The smallest stones are at least 8 meters apart; the smallest boulders are at least 20 meters apart.
# 
# 2. (Very Stony/Very Bouldery) — Stones or boulders cover 0.1 to less than 3 percent of the surface. The smallest stones are not less than 1 meter apart; the smallest boulders are not less than 3 meters apart.
# 
# 3. (Extremely Stony/Extremely Bouldery) — Stones or boulders cover 3 to less than 15 percent of the surface. The smallest stones are as little as 0.5 meter apart; the smallest boulders are as little as 1 meter apart.
# 
# 4. (Rubbly) — Stones or boulders cover 15 to less than 50 percent of the surface. The smallest stones are as little as 0.3 meter apart; the smallest boulders are as little as 0.5 meter apart. In most places it is possible to step from stone to stone or jump from boulder to boulder without touching the soil.
# 
# 5. (Very Rubbly) — Stones or boulders appear to be nearly continuous and cover 50 percent or more of the surface. The smallest stones are less than 0.03 meter apart; the smallest boulders are less than 0.05 meter apart. Classifiable soil is among the rock fragments, and plant growth is possible.
# 
# If no description of the surface cover is given, we give it a value of 0.

# In[15]:


def surface_cover(input_df):
    # Group IDs
    no_desc = [7,8,14,15,16,17,19,20,21,23,35]
    stony = [6,12]
    very_stony = [2,9,18,26]
    extremely_stony = [1,22,24,25,27,28,29,30,31,32,33,34,36,37,38,39,40]
    rubbly = [3,4,5,10,11,13]

    # Create dictionary
    surface_cover = {i:0 for i in no_desc}
    surface_cover.update({i:1 for i in stony})
    surface_cover.update({i:2 for i in very_stony})
    surface_cover.update({i:3 for i in extremely_stony})
    surface_cover.update({i:4 for i in rubbly})
    
    # Create Feature
    df = input_df.copy()
    df['Surface_Cover'] = input_df['Soil_Type'].apply(
        lambda x: surface_cover[x]
    )
    return df


# In[16]:


# Surface Cover
train = surface_cover(train)
test = surface_cover(test)


# ## 4. Rock Size (Nominal)
# 
# The final soil type feature we consider is rock size. This is also determined from the ELU code, the original data description and the [USFS soil profiling reference](https://www.nrcs.usda.gov/wps/portal/nrcs/detail/soils/ref/?cid=nrcs142p2_054253#fragments):
# 
# 1. Stones
# 2. Boulders
# 3. Rubble
# 
# If the soil type description has no mention of rock size, we give it a default value of 0.

# In[17]:


def rock_size(input_df):
    
    # Group IDs
    no_desc = [7,8,14,15,16,17,19,20,21,23,35]
    stones = [1,2,6,9,12,18,24,25,26,27,28,29,30,31,32,33,34,36,37,38,39,40]
    boulders = [22]
    rubble = [3,4,5,10,11,13]

    # Create dictionary
    rock_size = {i:0 for i in no_desc}
    rock_size.update({i:1 for i in stones})
    rock_size.update({i:2 for i in boulders})
    rock_size.update({i:3 for i in rubble})
    
    df = input_df.copy()
    df['Rock_Size'] = input_df['Soil_Type'].apply(
        lambda x: rock_size[x]
    )
    return df


# In[18]:


# Surface Cover
train = rock_size(train)
test = rock_size(test)


# ## 5. Soil Type Interactions
# 
# In this section we form some interaction features using soil type and these new soil-type derived features. We include only those features which resulted in improved CV accuracy.

# In[19]:


def soiltype_interactions(data):
    df = data.copy()
            
    # Important Soil Types
    df['Soil_12_32'] = df['Soil_Type32'] + df['Soil_Type12']
    df['Soil_Type23_22_32_33'] = df['Soil_Type23'] + df['Soil_Type22'] + df['Soil_Type32'] + df['Soil_Type33']
    
    # Soil Type Interactions
    df['Soil29_Area1'] = df['Soil_Type29'] + df['Wilderness_Area1']
    df['Soil3_Area4'] = df['Wilderness_Area4'] + df['Soil_Type3']
    
    #  New Feature Interactions
    df['Climate_Area2'] = df['Wilderness_Area2']*df['Climatic_Zone'] 
    df['Climate_Area4'] = df['Wilderness_Area4']*df['Climatic_Zone'] 
    df['Rock_Area1'] = df['Wilderness_Area1']*df['Rock_Size']    
    df['Rock_Area3'] = df['Wilderness_Area3']*df['Rock_Size']  
    df['Surface_Area1'] = df['Wilderness_Area1']*df['Surface_Cover'] 
    df['Surface_Area2'] = df['Wilderness_Area2']*df['Surface_Cover']   
    df['Surface_Area4'] = df['Wilderness_Area4']*df['Surface_Cover'] 
    
    # Fill NA
    df.fillna(0, inplace = True)
    
    return df


# In[20]:


# Surface Cover
train = soiltype_interactions(train)
test = soiltype_interactions(test)


# ## 6. Drop Columns (Dimension Reduction)
# 
# Finally, we drop the original soil type columns.

# In[21]:


# Drop original soil features
train.drop(columns = soil_features, inplace = True)
test.drop(columns = soil_features, inplace = True)


# # Final Submission

# In[22]:


# Submission with feature engineering
features = [x for x in train.columns if x not in ['Id', 'Cover_Type']] 
submission['Cover_Type'] = encoder.inverse_transform(
    train_model()
)
submission.to_csv('features_submission.csv', index=False)

