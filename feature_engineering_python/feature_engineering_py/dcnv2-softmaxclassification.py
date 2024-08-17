#!/usr/bin/env python
# coding: utf-8

# # Welcome and have fun learning multiclass classification with plug and play Neural Network
# 
# #### You are dealing with a **heavly skewed** dataset when some classes(1, 2 and 3) are much more frequent than others. **Accuracy** is not the perferred performance measure for multiclass problem in this case. Softvoting and weighted average is the objective to score towards target class 1, 2 and 3.
# 
# Objective of this notebook used to be a ~simple~ and robust neural network multiclass classifier for future use.
# 
# <blockquote style="margin-right:auto; margin-left:auto; padding: 1em; margin:24px;">
#     <strong>Fork This Notebook!</strong><br>
# Create your own editable copy of this notebook by clicking on the <strong>Copy and Edit</strong> button in the top right corner.
# </blockquote>
# 
# **Notes:**
# Run time -
# 4 hours and 31 minutes 4000000 samples
# 3318.3s 395712 samples
# 4375.7s 1468136 samples
# 12905.1s 2262087 samples
# 5555.3s - GPU 2262087 samples
# 6941.8s - TPU v3-8 2262087 samples
# 
# Version 98: 300, 256, 128, 128 0.1DROP MCDrop
# Version 121: AlphaDropout dropped bad performance
# 
# ## Imports and Configuration ##

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
from scipy import stats
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, PercentFormatter
import seaborn as sns

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, RobustScaler, PowerTransformer, OneHotEncoder
le = LabelEncoder()
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.model_selection import KFold, StratifiedKFold

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

from datetime import datetime
from packaging import version

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This notebook requires TensorFlow 2.0 or above."
import tensorboard
tensorboard.__version__
# Clear any logs from previous runs
get_ipython().system('rm -rf ./logs/')

# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import gc
import os
import math
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ## Fine tuning
# Fine tune the system using the hyperparameters and configs below:
# * FOLD - 5, 10, 15, 20.
# * SAMPLE - Set it to True for full sample run. Max sample per class.
# * BEST_OR_FOLD - True: use Best model, False: use KFOLD softvote
# * TPU - Only works on save version.
# * selu love lecun_normal

# In[2]:


# -----------------------------------------------------------------
# Some parameters to config 
PRODUCTION = True # True: For submission run. False: Fast trial run

# Hyperparameters
FOLDS = 20 if PRODUCTION else 5   # Only 5 or 10.
EPOCHS = 63        # Does not matter with Early stopping. Deep network should not take too much epochs to learn
BATCH_SIZE = 2048   # large enough to fit RAM. If unstable, tuned downward. 4096 2048
ACTIVATION = 'swish' # swish mish relu selu ;swish overfit more cause of narrow global minimun
KERNEL_INIT = "glorot_normal" # Minimal impact, but give your init the right foot forward glorot_uniform lecun_normal
LEARNING_RATE = 0.000965713 # Not used. Optimal lr is about half the maximum lr 
LR_FACTOR = 0.5   # LEARNING_RATE * LR_FACTOR = New Learning rate on ReduceLROnPlateau. lower down when the LR oscillate
MIN_DELTA = 0.0000001 # Default 0.0001 0.0000001
RLRP_PATIENCE = 5 # Learning Rate reduction on ReduceLROnPlateau
ES_PATIENCE = 21  # Early stopping
DROPOUT = 0.05     # Act like L1 L2 regulator. lower your learning rate in order to overcome the "boost" that the dropout probability gives to the learning rate.
HIDDEN_LAYERS = [320, 288, 64, 32]

OPTIMIZER = 'adam' # adam adamax nadam
LOSS ='sparse_categorical_crossentropy' # sparse_categorical_crossentropy does not require onehot encoding on labels. categorical_crossentropy
METRICS ='accuracy'  # acc accuracy categorical_accuracy sparse_categorical_accuracy
ACC_VAL_METRICS = 'val_accuracy' # 'val_acc' val_accuracy val_sparse_categorical_accuracy
ACC_METRICS = 'accuracy' # acc accuracy 'sparse_categorical_accuracy'

# The dataset is too huge for trial. Sampling it for speed run!
SAMPLE = 2262087 if PRODUCTION else 11426   # True for FULL run. Max Sample size per category. For quick test: y counts [1468136, 2262087, 195712, 377, 1, 11426, 62261]  # 4000000 total rows
VALIDATION_SPLIT = 0.15 # Only used to min dataset for quick test
MAX_TRIAL = 3           # speed trial any% Not used here
MI_THRESHOLD = 0.001    # Mutual Information threshold value to drop.

RANDOM_STATE = 42
VERBOSE = 0

# Admin
ID = "Id"            # Id id x X index
INPUT = "../input/tabular-playground-series-dec-2021"
TPU = False           # True: use TPU.
BEST_OR_FOLD = False # True: use Best model, False: use KFOLD softvote
FEATURE_ENGINEERING = True
PSEUDO_LABEL = True
BLEND = True

assert BATCH_SIZE % 2 == 0, \
    "BATCH_SIZE must be even number."


# ## Data Preprocessing ##
# 
# Before we can do any feature engineering, we need to *preprocess* the data to get it in a form suitable for analysis. We'll need to:
# - **Load** the data from CSV files
# - **Clean** the data to fix any errors or inconsistencies
# - **Encode** the statistical data type (numeric, categorical)
# - **Impute** any missing values
# 
# We'll wrap all these steps up in a function, which will make easy for you to get a fresh dataframe whenever you need. After reading the CSV file, we'll apply three preprocessing steps, `clean`, `encode`, and `impute`, and then create the data splits: one (`df_train`) for training the model, and one (`df_test`) for making the predictions that you'll submit to the competition for scoring on the leaderboard.

# ### Handle Missing Values ###
# 
# Handling missing values now will make the feature engineering go more smoothly. We'll impute `0` for missing numeric values and `"None"` for missing categorical values. You might like to experiment with other imputation strategies. In particular, you could try creating "missing value" indicators: `1` whenever a value was imputed and `0` otherwise.

# In[3]:


def impute(df):
    for name in df.select_dtypes("number"):
        df[name] = df[name].fillna(0)
    for name in df.select_dtypes("category"):
        df[name] = df[name].fillna("None")
    return df


# ## Reduce Memory usage

# In[4]:


# for col in df.select_dtypes('int').columns:
#     df[col] = pd.to_numeric(df[col], downcast = 'integer')

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtypes

        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
 
    return df


# # Feature Engineering
# These features are borrowed from https://www.kaggle.com/gulshanmishra/tps-dec-21-tensorflow-nn-feature-engineering
# Do read dataset description from https://www.kaggle.com/c/forest-cover-type-prediction/data
# The log transform is a powerful tool for dealing with positive numbers with a heavy-
# tailed  distribution.  (A  heavy-tailed  distribution  places  more  probability  mass  in  the
# tail range than a Gaussian distribution.) It compresses the long tail in the high end of
# the  distribution  into  a  shorter  tail,  and  expands  the  low  end  into  a  longer  head.

# In[5]:


from category_encoders import MEstimateEncoder
# extra feature engineering


def feature_engineer(df):

    # Distance features
    # Euclidean distance to Hydrology
    df["ecldn_dist_hydrlgy"] = (
        df["Horizontal_Distance_To_Hydrology"]**2 + df["Vertical_Distance_To_Hydrology"]**2)**0.5
    df["fire_road"] = np.abs(df["Horizontal_Distance_To_Fire_Points"]) + \
        np.abs(df["Horizontal_Distance_To_Roadways"])

    # Elevation features
    df['highwater'] = (df.Vertical_Distance_To_Hydrology < 0).astype(int)

    # Aspect features, hardest FE
    df.loc[df["Aspect"] < 0, "Aspect"] += 360
    df.loc[df["Aspect"] > 359, "Aspect"] -= 360
    df['binned_aspect'] = [math.floor((v+60)/15.0) for v in df['Aspect']]
    df['binned_aspect2'] = [math.floor((v+180)/10.0) for v in df['Aspect']]

    # Soil and wilderness features
    soil_features = [x for x in df.columns if x.startswith("Soil_Type")]
    df["soil_type_count"] = df[soil_features].sum(axis=1)
    wilderness_features = [
        x for x in df.columns if x.startswith("Wilderness_Area")]
    df["wilderness_area_count"] = df[wilderness_features].sum(axis=1)
    df['soil_Type12_32'] = df['Soil_Type32'] + df['Soil_Type12']
    df['soil_Type23_22_32_33'] = df['Soil_Type23'] + \
        df['Soil_Type22'] + df['Soil_Type32'] + df['Soil_Type33']

    # Hillshade features
    features_Hillshade = ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']
    df.loc[df["Hillshade_9am"] < 0, "Hillshade_9am"] = 0
    df.loc[df["Hillshade_Noon"] < 0, "Hillshade_Noon"] = 0
    df.loc[df["Hillshade_3pm"] < 0, "Hillshade_3pm"] = 0
    df.loc[df["Hillshade_9am"] > 255, "Hillshade_9am"] = 255
    df.loc[df["Hillshade_Noon"] > 255, "Hillshade_Noon"] = 255
    df.loc[df["Hillshade_3pm"] > 255, "Hillshade_3pm"] = 255
    df['Hillshade_Noon_is_bright'] = (df.Hillshade_Noon == 255).astype(int)
    df['Hillshade_9am_is_zero'] = (df.Hillshade_9am == 0).astype(int)
    df['hillshade_3pm_is_zero'] = (df.Hillshade_3pm == 0).astype(int)

    df.drop(["Aspect", 'Horizontal_Distance_To_Hydrology'], axis=1, inplace=True)

    return df


# In[6]:


from pathlib import Path


def load_data():
    # Read data
    data_dir = Path(INPUT)
    df_train = pd.read_csv(data_dir / "train.csv", index_col=ID)
    df_test = pd.read_csv(data_dir / "test.csv", index_col=ID)
    column_y = df_train.columns.difference(
        df_test.columns)[0]  # column_y target_col label_col
    return df_train, df_test, column_y


# In[7]:


def process_data(df_train, df_test):
    # Preprocessing
    df_train = impute(df_train)
    df_test = impute(df_test)
    
    if FEATURE_ENGINEERING:
        df_train = feature_engineer(df_train)
        df_test = feature_engineer(df_test)
    
    df_train = reduce_mem_usage(df_train)
    df_test = reduce_mem_usage(df_test)

    return df_train, df_test


# # Load Data #
# 
# And now we can call the data loader and get the processed data splits:

# In[8]:


get_ipython().run_cell_magic('time', '', 'train_data, test_data, column_y = load_data()\n')


# ## Pseudolabeling

# In[9]:


if PSEUDO_LABEL:
    df_pseudolabels = pd.read_csv(
        "../input/dcnv2-softmaxclassification/pseudolabels_v0.csv", index_col=ID)
    df_pseudolabels.to_csv(
        "pseudolabels_v0.csv", index=True)
    # shuffle and dropout some pseudolabels
    df_sampling_pseudolabels = df_pseudolabels.sample(frac=0.5)
    
#     df_sampling_pseudolabels = df_pseudolabels.groupby(column_y).apply(lambda s: s.sample(frac=0.8))
#     df_sampling_pseudolabels.reset_index(drop=True, inplace=True)
#     idx = train_data[train_data[column_y] == 1].index
#     train_data.drop(idx, axis = 0, inplace = True)
#     idx = train_data[train_data[column_y] == 2].index
#     train_data.drop(idx, axis = 0, inplace = True)
    # join blending club for new baseline
    train_data = pd.concat([train_data, df_pseudolabels], axis=0)
    
    # Remove pseudolabel samples from test set
    pseudo_label_index = df_sampling_pseudolabels.index
    test_data = test_data.drop(pseudo_label_index, axis=0)
    # Save for submission
    test_data_index = test_data.index
    df_pseudo_label_preds = pd.DataFrame({ID: pseudo_label_index,
                                          column_y: df_sampling_pseudolabels[column_y]}).reset_index(drop=True)


# In[10]:


np.unique(df_sampling_pseudolabels[column_y], return_counts=True)


# In[11]:


get_ipython().run_cell_magic('time', '', 'train_data, test_data = process_data(train_data, test_data)\n')


# In[12]:


# customized XY TBR
# idx = train_data[train_data[column_y] == 4].index
# train_data.drop(idx, axis = 0, inplace = True)
idx = train_data[train_data[column_y] == 5].index
train_data.drop(idx, axis = 0, inplace = True)
# idx = train_data[train_data[column_y] == 6].index # Less then 0.5% significant different, dropped
# train_data.drop(idx, axis = 0, inplace = True)

cols = ["Soil_Type7", "Soil_Type15"]
train_data.drop(cols, axis = 1, inplace= True)
test_data.drop(cols, axis = 1, inplace = True)


# y to categorical

# In[13]:


X = train_data.drop(columns=column_y)
y = train_data[[column_y]].astype(int)

X_test = test_data.loc[:,X.columns]

gc.collect()


# In[14]:


# Check NA
missing_val = X.isnull().sum()
print(missing_val[missing_val > 0])


# # Undersampling
# For experiment measurements

# ## For quick test
# TomekLinks
# (array([1, 2, 3, 4, 5, 6, 7]),
#  array([1390684, 2169226,  155218,     218,       1,    7627,   38757]))

# In[15]:


# Uncomment to peek at samples across all targets.
small_sampling = train_data.groupby(column_y).apply(lambda s: s.sample(min(len(s), 3900)))


# In[16]:


from sklearn.feature_selection import mutual_info_regression
def make_mi_scores(X, y, random_state=0):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=random_state)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.rcParams["figure.figsize"] = (15,12)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")


# In[17]:


def drop_uninformative(df, mi_scores, threshold = 0.001):
    return df.loc[:, mi_scores > threshold]


# In[18]:


def plot_features(features, df):
    for feature in features:
        plt.rcParams["figure.figsize"] = (15,9)
        for ctype in list(df[column_y].unique()):
            values = df.loc[df[column_y] == ctype][feature].values
            sns.scatterplot(x=values, y=np.arange(values.size), label=f"{column_y} {ctype}", alpha=0.35, palette="deep")
        plt.title(feature)
        plt.legend()
        plt.show()


# # Basic EDA
# Scatter plots for features with continuous values.

# In[19]:


# Scatter plots for features with continuous values , 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'horizontal_Distance_To_Fire_Points_Log', 'hillshade_mean', 'hillshade_amp']
features_cols = ['Elevation',
                 'Horizontal_Distance_To_Fire_Points',
                ]

plot_features(features=features_cols,df=small_sampling)


# ## Feature Utility Scores
# 
# Use mutual information to compute a utility score for a feature, giving you an indication of how much potential the feature has. This hidden cell defines the two utility functions we used, make_mi_scores and plot_mi_scores:

# In[20]:


# # Choose a set of features to encode and a value for m
# encoder = MEstimateEncoder(
#     cols=["binned_aspect"],
#     m=1.0,
# )
# # Encode the training split
# small_sampling = encoder.fit_transform(small_sampling, small_sampling[column_y])


# ## Group by labels, how many predictors are obvious?

# In[21]:


train_data.groupby(column_y).apply(lambda s: s.sample(min(len(s), 5)))


# In[22]:


get_ipython().run_cell_magic('time', '', 'mi_scores = make_mi_scores(small_sampling, small_sampling[column_y], random_state=RANDOM_STATE)\n')


# In[23]:


plot_mi_scores(mi_scores)


# ### Top10 predictors

# In[24]:


mi_scores[:10]


# ### Worst 10 predictors

# In[25]:


mi_scores[-10:]


# In[26]:


def sampling_size_params(labels, sampling_max_size = SAMPLE):
    ''' Return sampling parameters {labels: sample_size}'''
    sampling_key, sampling_count = np.unique(labels, return_counts=True)
    sampling_count[sampling_count > sampling_max_size] = sampling_max_size
    zip_iterator = zip(sampling_key, sampling_count)
    return dict(zip_iterator)


# Undersample if SAMPLE parameter < actual count.

# In[27]:


# not minority
sampling_params = sampling_size_params(y, SAMPLE)
undersample = RandomUnderSampler(
    sampling_strategy=sampling_params, random_state=RANDOM_STATE)

X, y = undersample.fit_resample(X, y)


# Drop features with 0 MI

# In[28]:


# X = drop_uninformative(X, mi_scores, MI_THRESHOLD)
# X_test = drop_uninformative(X_test, mi_scores, MI_THRESHOLD)


# In[29]:


soil_features = [x for x in X.columns if x.startswith("Soil_Type")]
wilderness_features = [x for x in X.columns if x.startswith("Wilderness_Area")]
binary_features = soil_features + wilderness_features


# In[30]:


transform_cols = X.columns[~X.columns.isin(binary_features)] # Numeric features


# In[31]:


# Prepare for multiclass classification tf.keras.utils.to_categorical(le.fit_transform(y[column_y])) categorical_crossentropy
y_cat = le.fit_transform(y[column_y]) # y to categorical


# In[32]:


np.unique(y, return_counts=True)


# In[33]:


CSV_HEADER = list(train_data.columns[:])

TARGET_FEATURE_NAME = column_y

TARGET_FEATURE_LABELS = np.unique(y_cat)

NUMERIC_FEATURE_NAMES = list(X.columns[:])

CATEGORICAL_FEATURES_WITH_VOCABULARY = {}

CATEGORICAL_FEATURE_NAMES = list(CATEGORICAL_FEATURES_WITH_VOCABULARY.keys())

FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES

COLUMN_DEFAULTS = [
    [0] if feature_name in NUMERIC_FEATURE_NAMES + [TARGET_FEATURE_NAME] else ["NA"]
    for feature_name in CSV_HEADER
]

NUM_CLASSES = len(TARGET_FEATURE_LABELS)

INPUT_SHAPE = X.shape[-1]
OUTPUT_SHAPE = le.classes_.shape[-1]


# In[34]:


print(CSV_HEADER)
print(INPUT_SHAPE)
print(OUTPUT_SHAPE)


# In[35]:


del mi_scores
del small_sampling
del df_pseudolabels
del df_sampling_pseudolabels
del train_data
gc.collect()


# # Scaler transformer
# By using RobustScaler(), we can remove the outliers
# ![](https://github.com/furyhawk/kaggle_practice/blob/main/images/Scalers.png?raw=true)

# In[36]:


transformer_cat_cols = make_pipeline(
    SimpleImputer(),
)
transformer_num_cols = make_pipeline(
    RobustScaler(),
#     StandardScaler(),
#     MinMaxScaler(feature_range=(0, 1))
)

preprocessor = make_column_transformer(
    (transformer_num_cols, transform_cols), # X.columns[:] transform_cols remainder = 'passthrough'
    ('passthrough', binary_features)
)


# In[37]:


X["mean"] = X[transform_cols].mean(axis=1)
X["std"] = X[transform_cols].std(axis=1)
X["min"] = X[transform_cols].min(axis=1)
X["max"] = X[transform_cols].max(axis=1)
X["skew"] = X[transform_cols].skew(axis=1)

X_test["mean"] = X_test[transform_cols].mean(axis=1)
X_test["std"] = X_test[transform_cols].std(axis=1)
X_test["min"] = X_test[transform_cols].min(axis=1)
X_test["max"] = X_test[transform_cols].max(axis=1)
X_test["skew"] = X_test[transform_cols].skew(axis=1)


# In[38]:


X_train_transformed = preprocessor.fit_transform(X)
X_test_transformed = preprocessor.transform(X_test)


# # Train Model and Create Submissions #
# 
# Once you're satisfied with everything, it's time to create your final predictions! This cell will:
# - use the best trained model to make predictions from the test set
# - save the predictions to a CSV file
# 
# $Softmax: \sigma(z_i) = \frac{e^{z_{i}}}{\sum_{j=1}^K e^{z_{j}}} \ \ \ for\ i=1,2,\dots,K$
# 
# K - number of classes
# 
# $z_i$ - is a vector containing the scores of each class for the instance z.
# 
# $\sigma(z_i)$ - is the estimated probability that the instance z belongs to class K, given the scores of each class for that instance.
# 
# $Relu(z) = max(0, z)$
# 
# Binary Cross Entropy: $-{(y\log(p) + (1 - y)\log(1 - p))}$
# 
# For multiclass classification, we calculate a separate loss for each class label per observation and sum the result.
# 
# $-\sum_{c=1}^My_{o,c}\log(p_{o,c})$
# 
# 
#     M - number of classes
# 
#     log - the natural log
# 
#     y - binary indicator (0 or 1) if class label c is the correct classification for observation o
# 
#     p - predicted probability observation o is of class c
# 
# 

# ## Create Models

# In[39]:


modelCheckpoint = None

if TPU:
    save_locally = tf.saved_model.SaveOptions(experimental_io_device='/job:localhost')
    modelCheckpoint = ModelCheckpoint(  'best_model', options = save_locally,
                                        monitor = ACC_VAL_METRICS,
                                        mode = 'max',
                                        save_best_only = True,
                                        verbose = VERBOSE,
                                        )
else:
    modelCheckpoint = ModelCheckpoint(
                                        'best_model',
                                        monitor = ACC_VAL_METRICS,
                                        mode = 'max',
                                        save_best_only = True,
                                        verbose = VERBOSE,
                                        )

early_stopping = EarlyStopping(
        patience = ES_PATIENCE,
        min_delta = MIN_DELTA,
        monitor = ACC_VAL_METRICS,
        mode = 'max',
        restore_best_weights = True,       
        baseline = None,
        verbose = VERBOSE,
    )
plateau = ReduceLROnPlateau(
        patience = RLRP_PATIENCE,
        factor = LR_FACTOR,
        min_lr = 1e-7,
        monitor = 'val_loss', 
        mode = 'min',
        verbose = VERBOSE,
    )

def get_MLPmodel(**kwargs):
# -----------------------------------------------------------------
# Model , kernel_initializer="lecun_normal"
    model = keras.Sequential([
#     layers.BatchNormalization(input_shape = [X.shape[-1]], name = 'input'),
    layers.Dense(units = 300, input_shape = [INPUT_SHAPE], name = 'input', kernel_initializer = KERNEL_INIT, activation = ACTIVATION),
#     layers.Dropout(rate = DROPOUT),
    layers.BatchNormalization(),
    layers.Dense(units = 200, kernel_initializer = KERNEL_INIT, activation = ACTIVATION),
#     layers.Dropout(rate = DROPOUT),
    layers.BatchNormalization(),
    layers.Dense(units = 100, kernel_initializer = KERNEL_INIT, activation = ACTIVATION),
#     layers.Dropout(rate = DROPOUT),
    layers.BatchNormalization(),
    layers.Dense(units = 50, kernel_initializer = KERNEL_INIT, activation = ACTIVATION),
#     layers.Dropout(rate = DROPOUT),
    layers.BatchNormalization(),
    layers.Dense(units = OUTPUT_SHAPE, activation = 'softmax', name='output'), #y_cat.shape[-1]
    ])

    return model


# ## Basic neural network blocks

# In[40]:


class MCDropout(keras.layers.AlphaDropout):
    '''Boost the performance of any trained dropout model without having to retrain it or even modify it at all.
        Provide a much better measure of the models uncertainty'''
    def call(self, inputs):
        return super().call(inputs, training=True)

class Standardization(layers.Layer):
    def adapt(self, data_sample):
        self.means_ = np.mean(data_sample, axis = 0, keepdims = True)
        self.stds_ = np.std(data_sample, axis = 0, keepdims = True)
    def call(self, inputs):
        return (inputs - self.means_) / (self.stds_ + keras.backend.epsilon())
    
# create custom dense-block
class DenseBlock(layers.Layer):
    def __init__(self, units, activation = ACTIVATION, dropout_rate = 0, l2 = 0, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self.dense = layers.Dense(
            units = units, 
#             activation = activation,
            kernel_initializer = KERNEL_INIT, 
#             kernel_regularizer=keras.regularizers.l2(l2)
        )
        self.batchn = layers.BatchNormalization()
        self.activation = layers.Activation(activation)
        if dropout_rate > 0:
            self.dropout = layers.Dropout(rate = dropout_rate) #MCDropout layers.Dropout
        else:
            self.dropout = None
    
    def call(self, inputs):
        x = self.dense(inputs)
        x = self.activation(x)
        x = self.batchn(x)
        
        if self.dropout is not None:
            x = self.dropout(x)
            
        return x

# create fully-connected NN
class MLP(keras.Model):
    def __init__(self, hidden_layers = HIDDEN_LAYERS, activation = ACTIVATION, dropout_rate = DROPOUT, l2 = 0, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.hidden_layers = [DenseBlock(units = units,
                                         activation = activation,
                                         dropout_rate = dropout_rate,
                                         l2 = l2
                                        )
                              for units in hidden_layers
                             ]
        self.softmax = layers.Dense(units = OUTPUT_SHAPE, activation = 'softmax', name='output')
        
    def call(self, inputs):
        x = inputs
        for dense_layer in self.hidden_layers:
            x = dense_layer(x)
        x = self.softmax(x)
        return x


# ### DCNv2 Parrallel Deep & Cross Network
# CrossNet from https://www.kaggle.com/mlanhenke/tps-12-deep-cross-nn-keras

# In[41]:


# create dense & cross model
class CrossNet(keras.Model):
    def __init__(self, hidden_layers = HIDDEN_LAYERS, activation = ACTIVATION, dropout_rate = DROPOUT, l2 = 0, **kwargs):
        super(CrossNet, self).__init__(**kwargs)

        for i, units in enumerate(hidden_layers, start=1):
            if i == 1: # Dropout before last layer only len(hidden_layers)
                self.dense_layers = [DenseBlock(units = units, activation = activation, dropout_rate = dropout_rate, l2 = l2)]
            else:
                self.dense_layers.append(DenseBlock(units = units, activation = activation, dropout_rate = 0, l2 = l2))
        
        self.dense = layers.Dense(units = INPUT_SHAPE)
        self.concat = layers.Concatenate()
        self.batchn = layers.BatchNormalization()
        self.softmax = layers.Dense(units = OUTPUT_SHAPE, activation = 'softmax', name='output')
        
    def call(self, inputs):
        
        dense, cross = inputs, inputs
        
        for dense_layer in self.dense_layers:
            # Deep net TODO only dropout at last layer
            dense = dense_layer(dense)
            # Parrallel Cross net
            cross_current = self.dense(cross)
            cross = inputs * cross_current + cross
            
        cross = self.batchn(cross)
        
        merged = self.concat([dense, cross])
        return self.softmax(merged)


# ## Wide & Deep model

# In[42]:


def wide_deep_model(**kwargs):
    il = layers.Input(shape=(INPUT_SHAPE), name="input")
    x1 = layers.Dense(units=100, activation=ACTIVATION)(il)
    x = layers.Dropout(DROPOUT)(x1)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(units=200, activation=ACTIVATION)(x)
    x = layers.Dropout(DROPOUT)(x)
    x = layers.BatchNormalization()(x) #AlphaDropout
    x = layers.Dense(units=100, activation=ACTIVATION)(layers.Concatenate()([x, x1]))
    x = layers.Dropout(DROPOUT)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(units=50, activation=ACTIVATION)(x)
    x = layers.BatchNormalization()(x)
    output = layers.Dense(units=NUM_CLASSES, activation="softmax", name="output")(x)

    model = tf.keras.Model([il], output)
    return model


# In[43]:


# def encode_inputs(inputs, encoding_size):
#     encoded_features = []
#     for col in range(inputs.shape[1]):
#         encoded_feature = tf.expand_dims(inputs[:, col], -1)
#         encoded_feature = layers.Dense(units=encoding_size)(encoded_feature)
#         encoded_features.append(encoded_feature)
#     return encoded_features

# def create_model(encoding_size, dropout_rate=0.15):
#     inputs = layers.Input(len(X.columns))
#     feature_list = encode_inputs(inputs, encoding_size)
#     num_features = len(feature_list)

#     features = VariableSelection(num_features, encoding_size, dropout_rate)(
#         feature_list
#     )

#     outputs = layers.Dense(units=OUTPUT_SHAPE, activation="softmax")(features)
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
#     return model


# In[44]:


def create_model_inputs():
    inputs = {}
    for feature_name in FEATURE_NAMES:
        if feature_name in NUMERIC_FEATURE_NAMES:
            inputs[feature_name] = layers.Input(
                name=feature_name, shape=(), dtype=tf.float32
            )
        else:
            inputs[feature_name] = layers.Input(
                name=feature_name, shape=(), dtype=tf.string
            )
    return inputs


# In[45]:


def encode_inputs(inputs, use_embedding=False):
    encoded_features = []
    for col in range(inputs.shape[1]):
        encoded_feature = tf.expand_dims(inputs[:, col], -1)
        encoded_features.append(encoded_feature)

    all_features = layers.concatenate(encoded_features)
    return all_features


# In[46]:


def create_deep_and_cross_model(hidden_units = HIDDEN_LAYERS, dropout_rate = DROPOUT):
    '''DCNv2 Model'''
    inputs = layers.Input(len(X.columns)) #create_model_inputs()
    x0 = encode_inputs(inputs, use_embedding=True)

    cross = x0
    for _ in hidden_units:
        units = cross.shape[-1]
        x = layers.Dense(units)(cross)
        cross = x0 * x + cross
    cross = layers.BatchNormalization()(cross)

    deep = x0
    for units in hidden_units:
        deep = layers.Dense(units)(deep)
        deep = layers.BatchNormalization()(deep)
        deep = layers.ReLU()(deep)
        deep = layers.Dropout(dropout_rate)(deep)

    merged = layers.concatenate([cross, deep])
    outputs = layers.Dense(units=NUM_CLASSES, activation="softmax")(merged)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


# ## Classification with Gated Residual and Variable Selection Networks
# From [TPS-12] G-Res & Variable Selection NN (Keras) https://www.kaggle.com/mlanhenke/tps-12-g-res-variable-selection-nn-keras

# In[47]:


class GatedLinearUnit(layers.Layer):
    def __init__(self, units, **kwargs):
        super(GatedLinearUnit, self).__init__(**kwargs)
        self.linear = layers.Dense(units)
        self.sigmoid = layers.Dense(units, activation="sigmoid")

    def call(self, inputs):
        return self.linear(inputs) * self.sigmoid(inputs)

class GatedResidualNetwork(layers.Layer):
    def __init__(self, units, dropout_rate, **kwargs):
        super(GatedResidualNetwork, self).__init__(**kwargs)
        self.units = units
        self.elu_dense = layers.Dense(units, activation="elu")
        self.linear_dense = layers.Dense(units)
        self.dropout = layers.Dropout(dropout_rate)
        self.gated_linear_unit = GatedLinearUnit(units)
        self.layer_norm = layers.LayerNormalization()
        self.project = layers.Dense(units)

    def call(self, inputs):
        x = self.elu_dense(inputs)
        x = self.linear_dense(x)
        x = self.dropout(x)
        if inputs.shape[-1] != self.units:
            inputs = self.project(inputs)
        x = inputs + self.gated_linear_unit(x)
        x = self.layer_norm(x)
        return x

class VariableSelection(layers.Layer):
    def __init__(self, num_features, units, dropout_rate, **kwargs):
        super(VariableSelection, self).__init__(**kwargs)
        self.grns = list()
        for idx in range(num_features):
            grn = GatedResidualNetwork(units, dropout_rate)
            self.grns.append(grn)
        self.grn_concat = GatedResidualNetwork(units, dropout_rate)
        self.softmax = layers.Dense(units=num_features, activation="softmax")

    def call(self, inputs):
        v = layers.concatenate(inputs)
        v = self.grn_concat(v)
        v = tf.expand_dims(self.softmax(v), axis=-1)

        x = []
        for idx, input in enumerate(inputs):
            x.append(self.grns[idx](input))
        x = tf.stack(x, axis=1)

        outputs = tf.squeeze(tf.matmul(v, x, transpose_a=True), axis=1)
        return outputs


# In[48]:


def encode_vsn_inputs(inputs, encoding_size):
    encoded_features = []
    for col in range(inputs.shape[1]):
        encoded_feature = tf.expand_dims(inputs[:, col], -1)
        encoded_feature = layers.Dense(units=encoding_size)(encoded_feature)
        encoded_features.append(encoded_feature)
    return encoded_features

def create_grn_and_vsn_model(encoding_size, dropout_rate=DROPOUT):
    inputs = layers.Input(len(X.columns))
    feature_list = encode_vsn_inputs(inputs, encoding_size)
    num_features = len(feature_list)

    features = VariableSelection(num_features, encoding_size, dropout_rate)(
        feature_list
    )

    outputs = layers.Dense(units=NUM_CLASSES, activation="softmax")(features)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


# In[49]:


# keras.utils.plot_model(CrossNet(), show_shapes=True, rankdir="LR", to_file="model.png")


# # Performance Measures
# ## StratifiedKFold
# Perform stratified sampling to produce folds that contain a representative ratio of each class. Soft voting is used from the predicted y between each fold.
# * Hard Voting: In hard voting, the predicted output class is a class with the highest majority of votes.
# * Soft Voting: In soft voting, the output class is the prediction based on the average of probability given to that class

# In[50]:


kf = StratifiedKFold(
        n_splits=FOLDS, random_state=RANDOM_STATE, shuffle=True)


# In[51]:


get_ipython().run_cell_magic('time', '', '\n# Define the Keras TensorBoard callback.\n#logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")\n#tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)\n\nENCODING_SIZE = 96  # Encoding size for create_grn_and_vsn_model\nstrategy = None\nif TPU:\n    try:  # detect TPUs\n        tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()  # TPU detection\n        strategy = tf.distribute.experimental.TPUStrategy(tpu)\n        print("Running on TPU:", tpu.master())\n    except ValueError:  # detect GPUs\n        # default strategy that works on CPU and single GPU\n        strategy = tf.distribute.get_strategy()\n        print("Number of accelerators: ", strategy.num_replicas_in_sync)\n        # this is 8 on TPU v3-8, it is 1 on CPU and GPU\n        BATCH_SIZE = 16 * strategy.num_replicas_in_sync\n\n# if TPU:        # instantiating the model in the strategy scope creates the model on the TPU\n# resets\npreds_valid_f = {}\npreds_test = np.zeros((1, 1))  # [] init Soft voting\ntotal_acc = []\nf_scores = []\nmodels = []\n\n# best_model = load_model(\'../input/dcnv2-softmaxclassification/best_model\')\n# preds_test = preds_test + best_model.predict(X_test_transformed, batch_size = BATCH_SIZE)\n\nfor fold, (train_index, valid_index) in enumerate(kf.split(X=X, y=y[TARGET_FEATURE_NAME])):\n\n    X_train, X_valid = X_train_transformed[train_index], X_train_transformed[valid_index]\n    y_train, y_valid = y_cat[train_index], y_cat[valid_index]\n\n    #   --------------------------------------------------------\n    # Preprocessing\n    index_valid = valid_index.tolist()\n\n    #    --------------------------------------------------------\n    #\n    #  ----------------------------------------------------------\n    # Model# instantiating the model in the strategy scope creates the model on the TPU\n    if TPU:\n        with strategy.scope():\n            model = CrossNet(\n                hidden_layers=HIDDEN_LAYERS,\n                activation=ACTIVATION,\n                dropout_rate=DROPOUT\n            )\n            model.compile(\n                optimizer=OPTIMIZER, loss=LOSS, metrics=[METRICS],\n            )\n\n    else:  # CPU/GPU model = create_grn_and_vsn_model(encoding_size=ENCODING_SIZE) CrossNet wide_deep_model\n        model = wide_deep_model(\n            hidden_layers=HIDDEN_LAYERS,\n            activation=ACTIVATION,\n            dropout_rate=DROPOUT\n        )\n        model.compile(\n            optimizer=OPTIMIZER, loss=LOSS, metrics=[METRICS],\n        )\n\n    history = model.fit(X_train, y_train,\n                        validation_data=(X_valid, y_valid),\n                        batch_size=BATCH_SIZE,\n                        epochs=EPOCHS,\n                        callbacks=[early_stopping, plateau, modelCheckpoint,  # tensorboard_callback\n                                   ],\n                        shuffle=True,\n                        verbose=VERBOSE,\n                        )\n\n    #  ----------------------------------------------------------\n    #  oof\n    preds_valid = model.predict(X_valid, batch_size=BATCH_SIZE)\n\n    #  ----------------------------------------------------------\n    #********  test dataset predictions for submission *********#\n    if not BEST_OR_FOLD:\n        preds_test = preds_test + \\\n            model.predict(X_test_transformed, batch_size=BATCH_SIZE)\n\n    #  ----------------------------------------------------------\n    #  Saving scores to plot the end\n    scores = pd.DataFrame(history.history)\n    scores[\'folds\'] = fold\n    if fold == 0:\n        f_scores = scores\n        model.summary()\n#         keras.utils.plot_model(model, show_shapes=True, rankdir="LR")\n    else:\n        f_scores = pd.concat([f_scores, scores], axis=0)\n\n    #  ----------------------------------------------------------\n    #  concatenating valid preds\n    preds_valid_f.update(\n        dict(zip(index_valid, le.inverse_transform(np.argmax(preds_valid, axis=1)))))\n    # Getting score for a fold model\n    fold_acc = accuracy_score(y.iloc[valid_index].Cover_Type, le.inverse_transform(\n        np.argmax(preds_valid, axis=1)))\n    print(\n        f"Fold {fold} accuracy_score: {fold_acc} Train: {X_train.shape} Valid: {X_valid.shape}")\n    # Total acc\n    total_acc.append(fold_acc)\n\n    del model\n    gc.collect()\n    K.clear_session()\n\nprint(f"mean accuracy_score: {np.mean(total_acc)}, std: {np.std(total_acc)}")\n')


# In[52]:


# Load best_model
if BEST_OR_FOLD:
    if not TPU:
        # load the saved model
        best_model = load_model('best_model')
    else: # TPU
        with strategy.scope():
            load_locally = tf.saved_model.LoadOptions(experimental_io_device = '/job:localhost')
            best_model = tf.keras.models.load_model('best_model', options = load_locally) # loading in Tensorflow's "SavedModel" format
    # Using best model to predict
    # X_test = preprocessor.transform(X_test) Not using best loop fit TODO
    preds_test = preds_test + best_model.predict(X_test_transformed, batch_size = BATCH_SIZE)


# # Evaluation

# In[53]:


def plot_acc(f_scores):
    for fold in range(f_scores['folds'].nunique()):
        history_f = f_scores[f_scores['folds'] == fold]

        best_epoch = np.argmin(np.array(history_f['val_loss']))
        best_val_loss = history_f['val_loss'][best_epoch]

        fig, ax1 = plt.subplots(1, 2, tight_layout=True, figsize=(15,4))

        fig.suptitle('Fold : '+ str(fold+1) +
                     " Validation Loss: {:0.4f}".format(history_f['val_loss'].min()) +
                     " Validation Accuracy: {:0.4f}".format(history_f[ACC_VAL_METRICS].max()) +
                     " LR: {:0.8f}".format(history_f['lr'].min())
                     , fontsize=14)

        plt.subplot(1,2,1)
        plt.plot(history_f.loc[:, ['loss', 'val_loss']], label= ['loss', 'val_loss'])
                
        from_epoch = 0
        if best_epoch >= from_epoch:
            plt.scatter([best_epoch], [best_val_loss], c = 'r', label = f'Best val_loss = {best_val_loss:.5f}')
        if best_epoch > 0:
            almost_epoch = np.argmin(np.array(history_f['val_loss'])[:best_epoch])
            almost_val_loss = history_f['val_loss'][almost_epoch]
            if almost_epoch >= from_epoch:
                plt.scatter([almost_epoch], [almost_val_loss], c='orange', label = 'Second best val_loss')
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper left')   
        
        ax2 = plt.gca().twinx()
        ax2.plot(history_f.loc[:, ['lr']], 'y:', label='lr' ) # default color is same as first ax
        ax2.set_ylabel('Learning rate')
        ax2.legend(loc = 'upper right')
        ax2.grid()

        best_epoch = np.argmax(np.array(history_f[ACC_VAL_METRICS]))
        best_val_acc = history_f[ACC_VAL_METRICS][best_epoch]
        
        plt.subplot(1,2,2)
        plt.plot(history_f.loc[:, [ACC_METRICS, ACC_VAL_METRICS]],label= [ACC_METRICS, ACC_VAL_METRICS])
        if best_epoch >= from_epoch:
            plt.scatter([best_epoch], [best_val_acc], c = 'r', label = f'Best val_acc = {best_val_acc:.5f}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc = 'lower left')
        plt.legend(fontsize = 15)
        plt.grid(b = True, linestyle = '-')

plot_acc(f_scores)


# But instead of just looking at the mean accuracy across the 10 cross-validation folds, let's plot all 10 scores for each model, along with a box plot highlighting the lower and upper quartiles, and "whiskers" showing the extent of the scores. Note that the `boxplot()` function detects outliers (called "fliers") and does not include them within the whiskers. Specifically, if the lower quartile is $Q_1$ and the upper quartile is $Q_3$, then the interquartile range $IQR = Q_3 - Q_1$ (this is the box's height), and any score lower than $Q_1 - 1.5 \times IQR$ is a flier, and so is any score greater than $Q3 + 1.5 \times IQR$.

# In[54]:


plt.figure(figsize = (3, 5))
plt.plot([1]*FOLDS, total_acc, ".")
plt.boxplot([total_acc], labels = ("y"))
plt.ylabel("Accuracy", fontsize = 14)
plt.show()


# ## Confusion matrix
# 
# $Accuracy = \frac{TP+TN}{TP+TN+FP+FN}$
# 
# $Precision = \frac{TP}{TP+FP}$
# 
# $Recall = \frac{TP}{TP+FN}$
# 
# $F1 = \frac{2*Precision*Recall}{Precision+Recall} = \frac{2*TP}{2*TP+FP+FN}$
# 
# 

# In[55]:


def plot_cm(cm):
    metrics = {
        'accuracy': cm / cm.sum(),
        'recall' : cm / cm.sum(axis =1 ),
        'precision': cm / cm.sum(axis = 0)
    }
    
    fig, ax = plt.subplots(1,3, tight_layout = True, figsize = (15,5))
    ax = ax.flatten()

#     mask = (np.eye(cm.shape[0]) == 0) * 1

    for idx, (name, matrix) in enumerate(metrics.items()):

        ax[idx].set_title(name)

        sns.heatmap(
            data = matrix,
            cmap = sns.dark_palette("#69d", reverse=True, as_cmap=True),
            cbar = False,
#             mask=mask,
            lw = 0.25,
            annot = True,
            fmt = '.2f',
            ax = ax[idx]
        )
    sns.despine()


# In[56]:


oof_y_hat = []
for key, value in sorted(preds_valid_f.items()):
    oof_y_hat.append(value)


# In[57]:


# create confusion matrix, calculate accuracy,recall & precision
cm = pd.DataFrame(data = confusion_matrix(y, oof_y_hat, labels = le.classes_), index = le.classes_, columns = le.classes_)
plot_cm(cm)


# In[58]:


def plot_cm_error(cm):
    mask = (np.eye(cm.shape[0]) != 0) * 1
    fig, ax = plt.subplots(tight_layout=True, figsize=(15,5))
    sns.heatmap(
                data = pd.DataFrame(data=cm, index=le.classes_, columns = le.classes_),
    #             cmap=sns.dark_palette("#69d", reverse=True, as_cmap=True),
                cbar = False,
                lw = 0.25,
                mask = mask,
                annot = True,
                fmt = '.0f',
                ax = ax
            )
    sns.despine()
    
plot_cm_error(confusion_matrix(y, oof_y_hat, labels = le.classes_))


# In[59]:


preds_test.shape


# ## Bootstrap blending

# In[60]:


col_names = [f'{clss}' for clss in le.classes_]
if PRODUCTION:
    pd.DataFrame(preds_test, columns=col_names).to_parquet('matrix_resurrections.pq')


# In[61]:


# le.inverse_transform(np.argmax(preds_test, axis=1))
# p000 = pd.read_parquet('../input/dcnv2-softmaxclassification/matrix_resurrections.pq').to_numpy()
# p000.shape
# pred_hat = (preds_test + p000).argmax(axis=1)
# pred_hat0 = le.inverse_transform(pred_hat)


# In[62]:


pred_hat = preds_test.argmax(axis=1)
pred_hat0 = le.inverse_transform(pred_hat)


# Merging pseudolabels with prediction

# In[63]:


# p0 = pd.read_csv(INPUT + '/sample_submission.csv')
# p0[column_y] = pred_hat0

# 20% sampling for pred
new_test_preds_df = pd.DataFrame({ID: test_data_index,
                                      column_y: pred_hat0})
# Concatenate with pseudolabels
p00 = pd.concat([new_test_preds_df, df_pseudo_label_preds])
# Sort by id
p0 = p00.sort_values(by=ID, ascending=True)


# In[64]:


p0.reset_index(drop=True, inplace=True)


# # Super blender 👀👀👀
# Credits: All the submissions as linked in the Data tab.
# 
# - https://www.kaggle.com/kavehshahhosseini/tps-dec-2021-simple-ensemble-public-notebooks
# - https://www.kaggle.com/samuelcortinhas/tps-dec-feat-eng-pseudolab-clean-version
# - https://www.kaggle.com/kaaveland/tps202112-reasonable-xgboost-model
# - https://www.kaggle.com/mlanhenke/tps-12-g-res-variable-selection-nn-keras
# - https://www.kaggle.com/remekkinas/tps-12-super-fast-blending-tool
# - https://www.kaggle.com/remekkinas/tps-12-nn-tpu-pseudolabeling-0-95690
# - https://www.kaggle.com/gaolang/tps-dec-2021-simple-ensemble-public-notebooks
# - https://www.kaggle.com/ambrosm/tpsdec21-12-eliminate-cover-type-4
# - https://www.kaggle.com/pourchot/tps-12-simple-nn-with-skip-connection/notebook
# - https://www.kaggle.com/slythe/tps-dec-2021-lightgbm-top-200

# In[65]:


blenders = ["../input/k/kavehshahhosseini/tps-dec-2021-simple-ensemble-public-notebooks/submission.csv"]
# blenders.append("../input/gresvariableselection-softmaxclassification/submission.csv")
# blenders.append("../input/tps-12-super-fast-blending-tool/tps12-fast-blend.csv")
# blenders.append("../input/blend-of-blend-of-blend-of-blend-of-blend-of-ble/submission.csv")

blenders.append("../input/tps-12-g-res-variable-selection-nn-keras/baseline_nn.csv")
blenders.append("../input/tps-12-nn-tpu-pseudolabeling-0-95690/tps12-pseudeo-submission.csv")
blenders.append("../input/tps202112-reasonable-xgboost-model/submission.csv")
blenders.append("../input/k/kavehshahhosseini/tps-dec-2021-simple-ensemble-public-notebooks/submission.csv")
blenders.append("../input/k/kavehshahhosseini/tps-dec-2021-simple-ensemble-public-notebooks/submission.csv")
# blenders.append("../input/k/kavehshahhosseini/tps-dec-2021-simple-ensemble-public-notebooks/submission.csv")
# blenders.append("../input/k/kavehshahhosseini/tps-dec-2021-simple-ensemble-public-notebooks/submission.csv")
# blenders.append("../input/k/kavehshahhosseini/tps-dec-2021-simple-ensemble-public-notebooks/submission.csv")
blenders.append("../input/dcnv2-softmaxclassification/submission.csv")
blenders.append("../input/blender/95685.csv")
blenders.append("../input/tps-dec-feat-eng-pseudolab-clean-version/submission.csv")
blenders.append("../input/tpsdec21-12-eliminate-cover-type-4/submission_without4.csv")
blenders.append("../input/tps-12-simple-nn-with-skip-connection/sub_13folds_num_9.csv")
blenders.append("../input/tps-dec-2021-lightgbm-top-200/submission.csv")

def read_blenders(blenders):
    try:
        return [pd.read_csv(blender) for blender in blenders]
    except FileNotFoundError:
        return [pd.read_csv("../input/blender/95685.csv")]


# In[66]:


sub = pd.read_csv(INPUT + '/sample_submission.csv')
predictions = read_blenders(blenders)


# In[67]:


results = pd.DataFrame()
for i, ds in enumerate(predictions):
    results[f'p{i+1}'] = ds[column_y]

print(results.shape)
results.head(10)


# In[68]:


results['p0'] = p0[column_y]
results['p00'] = p0[column_y]
results['p000'] = p0[column_y]
# results['p0000'] = p0[column_y]
# results['p00000'] = p0[column_y]


# In[69]:


np.unique(p0[column_y], return_counts=True)


# In[70]:


get_ipython().run_cell_magic('time', '', 'results["ensemble"] = stats.mode(np.array(results), axis=1)[0]\nresults.head(10)\n')


# # Submission

# In[71]:


sub[column_y] = results["ensemble"]
sub.to_csv("submission.csv", index=False)
sub.to_csv("submission_00.csv", index=False)
sub.head(10)


# In[72]:


# if PSEUDO_LABEL:
#     # Save new predictions to df
#     new_test_preds_df = pd.DataFrame({ID: test_data_index,
#                                       column_y: le.inverse_transform(np.argmax(preds_test, axis=1))})
#     # Concatenate with pseudolabels
#     sub = pd.concat([new_test_preds_df, df_pseudo_label_preds])
#     # Sort by id
#     sub = sub.sort_values(by=ID, ascending=True)
#     # Check format
#     display(sub.head(10))
#     # Save to csv
#     sub.to_csv('submission.csv', index=False)
#     print('Submitted. Good Luck!!!')


# In[73]:


# if not PSEUDO_LABEL:
#     sub = pd.read_csv(INPUT + '/sample_submission.csv')
#     # (stats.mode(preds_test)[0][0]) preds_test[FOLDS-1] # argmax reverse of to_categorical sub[column_y] = (np.argmax(sum(preds_test), axis=1) + 1) # le.inverse_transform(np.argmax(preds_test, axis=1))
#     sub[column_y] = le.inverse_transform(
#         np.argmax(preds_test, axis=1)).astype(int)
#     sub.to_csv('submission.csv', index=False)
#     sub


# In[74]:


np.unique(sub[column_y], return_counts=True)


# In[75]:


np.unique(y_cat, return_counts=True)


# In[76]:


# Plot the distribution of the test predictions vs training set
plt.figure(figsize=(10,5))
plt.hist(y[column_y], bins = np.linspace(0.5, 7.5, 8), density = True, label = 'Training labels')
plt.hist(sub[column_y], bins = np.linspace(0.5, 7.5, 8), density = True, rwidth = 0.7, label = 'Test predictions')
plt.xlabel(column_y)
plt.ylabel('Frequency')
plt.gca().yaxis.set_major_formatter(PercentFormatter())
plt.legend()
plt.show()


# In[77]:


def plot_x_labels(ax):
    for rect in ax.patches:
        height = rect.get_height()
        ax.annotate(f'{int(height)}', xy=(rect.get_x()+rect.get_width()/2, height), 
                    xytext=(0, 5), textcoords='offset points', ha='center', va='bottom') 

# Plot the distribution of the test predictions
fig, ax = plt.subplots(1,2,figsize = (10,4))
sns.countplot(x = sub[column_y], ax = ax[0], orient = "h").set_title("Prediction")
plot_x_labels(ax[0])
# Plot the distribution of the training set
sns.countplot(x = y[column_y], ax = ax[1], orient = "h").set_title("Training labels")
plot_x_labels(ax[1])
fig.show()


# In[78]:


def nunique(a, axis):
    return (np.diff(np.sort(a,axis=axis),axis=axis)!=0).sum(axis=axis)+1
results["different"] = nunique(results.iloc[:,:len(predictions)].values,1) - 1


# In[79]:


fig, ax = plt.subplots(1,1,figsize = (10,4))
sns.countplot(x = results["different"], ax = ax, orient = "h").set_title("Prediction difference")
plot_x_labels(ax)


# ## Unknown Cover Type prediction

# In[80]:


X_results = X_test.copy()
X_results[column_y] = pred_hat0
features_cols = ['Elevation',
                ]

plot_features(features=features_cols,df=X_results)


# ## Prediction visualized

# In[81]:


data_dir = Path(INPUT)
df_test = pd.read_csv(data_dir / "test.csv", index_col=ID)
df_test.reset_index(drop=True, inplace=True)
df_test[column_y] = sub[column_y]

plot_features(features=features_cols,df=df_test)


# ## New pseudolabels

# In[82]:


new_index = np.arange(4000000, 4999999 + 1)
df_test[ID] = new_index
df_test.set_index(ID, inplace=True)
df_test.to_csv('pseudolabels_v5.csv', index=True)


# To submit these predictions to the competition, follow these steps:
# 
# 1. Begin by clicking on the blue **Save Version** button in the top right corner of the window.  This will generate a pop-up window.
# 2. Ensure that the **Save and Run All** option is selected, and then click on the blue **Save** button.
# 3. This generates a window in the bottom left corner of the notebook.  After it has finished running, click on the number to the right of the **Save Version** button.  This pulls up a list of versions on the right of the screen.  Click on the ellipsis **(...)** to the right of the most recent version, and select **Open in Viewer**.  This brings you into view mode of the same page. You will need to scroll down to get back to these instructions.
# 4. Click on the **Output** tab on the right of the screen.  Then, click on the file you would like to submit, and click on the blue **Submit** button to submit your results to the leaderboard.
# 
# You have now successfully submitted to the competition!
# 
# # Next Steps #
# 
# If you want to keep working to improve your performance, select the blue **Edit** button in the top right of the screen. Then you can change your code and repeat the process. There's a lot of room to improve, and you will climb up the leaderboard as you work.
# 
# Be sure to check out [other users' notebooks](https://www.kaggle.com/c/tabular-playground-series-dec-2021/code) in this competition. You'll find lots of great ideas for new features and as well as other ways to discover more things about the dataset or make better predictions. There's also the [discussion forum](https://www.kaggle.com/c/tabular-playground-series-dec-2021/discussion), where you can share ideas with other Kagglers.
# 
# Have fun Kaggling!

# ## Op-level graph
# 
# Start TensorBoard and wait a few seconds for the UI to load. Select the Graphs dashboard by tapping “Graphs” at the top. 

# In[83]:


# Load the TensorBoard notebook extension.
# %load_ext tensorboard
# %tensorboard --logdir logs


# In[84]:


# %reload_ext tensorboard


# In[85]:


# from xgboost import XGBClassifier
# #gpu_hist gpu_predictor cpu_predictor
# xgb_params = {
#     'objective': 'multi:softmax',
#     'tree_method': 'hist', 
#     'use_label_encoder': False,
#     'seed': RANDOM_STATE, 
#     'eval_metric': ['mlogloss', 'merror'],
#     'predictor': 'cpu_predictor'
# }


# In[86]:


# %%time
# y_train, y_valid = y_cat[train_index], y_cat[valid_index]
# xgb_model = XGBClassifier(**xgb_params)
# xgb_history = xgb_model.fit(X_train, y_train, eval_set = [(X_valid, y_valid)], verbose = VERBOSE)

# preds_train = xgb_model.predict(X_train)
# preds_valid = xgb_model.predict(X_valid)
# acc_train = accuracy_score(y_train, preds_train)
# acc = accuracy_score(y_valid, preds_valid)
# print(f"train: {acc_train:.6f}, valid: {acc:.6f}")

# preds_test = xgb_model.predict(X_test)


# In[87]:


# preds_valid.shape


# In[88]:


# create confusion matrix, calculate accuracy,recall & precision
# cm = pd.DataFrame(data=confusion_matrix(y_valid, preds_valid, labels=le.classes_), index=le.classes_, columns=le.classes_)
# plot_cm(cm)


# In[89]:


# plot_cm_error(cm)

