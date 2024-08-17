#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ===============================================================================
# Library
# ===============================================================================
import os
import gc
import re
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import random
import math
from tqdm.notebook import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import mixed_precision
from transformers import AutoTokenizer, TFAutoModel, AutoConfig
import Levenshtein
import difflib

# ===============================================================================
# Configurations
# ===============================================================================
class CFG:
    input_path = '../input/foursquare-location-matching/'
    target = 'point_of_interest'
    model = '../input/xlmroberta/xlm-roberta-base/'
    tokenizer = AutoTokenizer.from_pretrained(model)
    max_len = 150
    seed = 42
    batch_size = 32
    target_size = 1
    rounds = 6
    n_neighbors = 7
    best_thres = 0.45
    
# ===============================================================================
# Read data
# ===============================================================================
def read_data():
    test = pd.read_csv(CFG.input_path + 'test.csv')
    return test
    
# ===============================================================================
# Generate data for test
# ===============================================================================
def generate_test_data(df, rounds = 2, n_neighbors = 10, features = ['id', 'latitude', 'longitude']):
    # Scale data for KNN
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features[2:4]])
    # Fit KNN and predict indices
    knn_model = NearestNeighbors(
        n_neighbors = n_neighbors, 
        radius = 1.0, 
        algorithm = 'kd_tree', 
        leaf_size = 30, 
        metric = 'minkowski', 
        p = 2, 
        n_jobs = -1
    )
    knn_model.fit(scaled_data)
    indices = knn_model.kneighbors(scaled_data, return_distance = False)
    # Create a new dataframe to slice faster
    df_features = df[features]
    # Create a dataset to store final results
    dataset = []
    # Iterate through each round and get generated data
    for j in range(rounds):
        # Create temporal dataset to store round data
        tmp_dataset = []
        # Iterate through each row
        for k in tqdm(range(len(df))):
            neighbors = list(indices[k])
            # Remove self from neighbors if exist
            try:
                neighbors.remove(k)
            except:
                pass
            # Use iterator as first indices
            ind1 = k
            # Select from the neighbor list the second indices
            ind2 = neighbors[j]
            # Check if indices are the same, they should not be the same
            if ind1 == ind2:
                print('Indices are the same, error')
            # Slice features dataframe
            tmp1 = df_features.loc[ind1]
            tmp2 = df_features.loc[ind2]
            # Concatenate, don't add target, this is the test set
            tmp = np.concatenate([tmp1, tmp2], axis = 0)
            tmp_dataset.append(tmp)  
        # Transform tmp_dataset to a pd.DataFrame
        tmp_dataset = pd.DataFrame(tmp_dataset, columns = [i + '_1' for i in features] + [i + '_2' for i in features])
        # Append round
        dataset.append(tmp_dataset)
    # Concatenate rounds to get final dataset
    dataset = pd.concat(dataset, axis = 0)
    # Remove duplicates
    dataset.drop_duplicates(inplace = True)
    # Reset index
    dataset.reset_index(drop = True, inplace = True)
    col_64 = list(dataset.dtypes[dataset.dtypes == np.float64].index)
    for col in col_64:
        dataset[col] = dataset[col].astype(np.float32)
    return df, dataset

# ===============================================================================
# Get manhattan distance
# ===============================================================================
def manhattan(lat1, long1, lat2, long2):
    return np.abs(lat2 - lat1) + np.abs(long2 - long1)

# ===============================================================================
# Get haversine distance
# ===============================================================================
def vectorized_haversine(lats1, lats2, longs1, longs2):
    radius = 6371
    dlat=np.radians(lats2 - lats1)
    dlon=np.radians(longs2 - longs1)
    a = np.sin(dlat/2) * np.sin(dlat/2) + np.cos(np.radians(lats1)) \
        * np.cos(np.radians(lats2)) * np.sin(dlon/2) * np.sin(dlon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = radius * c
    return d

# ===============================================================================
# Compute distances + Euclidean
# ===============================================================================
def add_lat_lon_distance_features(df):
    lat1 = df['latitude_1']
    lat2 = df['latitude_2']
    lon1 = df['longitude_1']
    lon2 = df['longitude_2']
    df['latdiff'] = (lat1 - lat2)
    df['londiff'] = (lon1 - lon2)
    df['manhattan'] = manhattan(lat1, lon1, lat2, lon2)
    df['euclidean'] = (df['latdiff'] ** 2 + df['londiff'] ** 2) ** 0.5
    df['haversine'] = vectorized_haversine(lat1, lat2, lon1, lon2)
    col_64 = list(df.dtypes[df.dtypes == np.float64].index)
    for col in col_64:
        df[col] = df[col].astype(np.float32)
    return df

# ===============================================================================
# Compute distances for categorical features
# ===============================================================================
def get_distance_cat(df, column):
    geshs = []
    levens = []
    jaros = []
    for str1, str2 in df[[column + '_1', column + '_2']].values.astype(str):
        if str1==str1 and str2==str2:
            geshs.append(difflib.SequenceMatcher(None, str1, str2).ratio())
            levens.append(Levenshtein.distance(str1, str2))
            jaros.append(Levenshtein.jaro_winkler(str1, str2))
        else:
            geshs.append(-1)
            levens.append(-1)
            jaros.append(-1)
    df1 = pd.DataFrame({
        f"{column}_geshs": geshs,
        f"{column}_levens": levens,
        f"{column}_jaros": jaros,
        })
    if column not in ['country', 'phone', 'zip']:
        df1[f"{column}_len_1"] = df[column + '_1'].astype(str).map(len)
        df1[f"{column}_len_2"] = df[column + '_2'].astype(str).map(len)
        df1[f"{column}_nlevens"] = df1[f"{column}_levens"] / df1[[f"{column}_len_1", f"{column}_len_2"]].max(axis = 1)
    col_64 = list(df1.dtypes[df1.dtypes == np.float64].index)
    for col in col_64:
        df1[col] = df1[col].astype(np.float32)
    df = pd.concat([df, df1], axis = 1)
    return df

# ===============================================================================
# Add '[SEP]' token to all the categorical features we want to encode
# ===============================================================================
def add_sep_token(df):
    # Before concatenation, fill NAN with unknown
    df.fillna('unknown', inplace = True)
    df['text'] = df['name_1'] + '[SEP]' + df['address_1'] + '[SEP]' + df['city_1'] + '[SEP]' \
    + df['state_1'] + '[SEP]' + df['country_1'] + '[SEP]' + df['url_1'] + '[SEP]' + df['categories_1'] + '[SEP]' \
    + df['name_2'] + '[SEP]' + df['address_2'] + '[SEP]' + df['city_2'] + '[SEP]' \
    + df['state_2'] + '[SEP]' + df['country_2'] + '[SEP]' + df['url_2'] + '[SEP]' + df['categories_2']
    return df

# ===============================================================================
# Create model 
# ===============================================================================
def build_model(cfg):
    transformer = TFAutoModel.from_pretrained(cfg.model, from_pt = True)
    input_word_ids = tf.keras.layers.Input(shape = (cfg.max_len, ), dtype = tf.int32, name = 'input_word_ids')
    inp_num = tf.keras.layers.Input(shape = (cfg.num_shape, ), dtype = tf.float32, name = 'num_inputs')
    last_hidden_state = transformer(input_word_ids)['last_hidden_state']
    last_hidden_state_avg_pool = tf.keras.layers.GlobalAveragePooling1D()(last_hidden_state)
    last_hidden_state_avg_pool = tf.keras.layers.Dropout(0.40)(last_hidden_state_avg_pool)
    x = tf.keras.layers.Dense(1024, activation = 'relu')(inp_num)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.15)(x)
    x = tf.keras.layers.Dense(512, activation = 'relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.15)(x)
    x = tf.keras.layers.Concatenate()([last_hidden_state_avg_pool, x])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation = 'relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.20)(x)
    output = tf.keras.layers.Dense(cfg.target_size, activation = 'sigmoid')(x)
    model = tf.keras.models.Model(inputs = [input_word_ids, inp_num], outputs = [output])
    return model

# ====================================================
# Prepare input using tokenizer
# ====================================================
def prepare_input(data):
    inputs_ids = []
    for text in tqdm(data, total = len(data)):
        inputs = CFG.tokenizer(
            text,
            add_special_tokens = True,
            padding = 'max_length',
            truncation = True,
            return_offsets_mapping = False,
            max_length = CFG.max_len,
            return_token_type_ids = False,
            return_attention_mask = False,
        )
        inputs_ids.append(inputs['input_ids'])
    return np.array(inputs_ids)

# ===============================================================================
# Inference
# ===============================================================================
def inference(test_dataset, test):
    ds_len = len(test_dataset)
    # Get numeric features and text features
    ignore_cols = ['id_1', 'id_2', 'match', 'text']
    num_features = [col for col in test_dataset.columns if col not in ignore_cols]
    CFG.num_shape = len(num_features)
    # Build model
    model = build_model(CFG)
    # Load weights
    model.load_weights('../input/flm-models/baseline.h5')
    # Use a for loop to avoid memory problem (we could do this with a generator also)
    predictions = []
    for i in range(5):
        x_test_num = test_dataset[num_features].iloc[int(i * ds_len / 5) : int((i + 1) * ds_len / 5)]
        x_test_text = test_dataset['text'].iloc[int(i * ds_len / 5) : int((i + 1) * ds_len / 5)]
        # Scale numeric features
        scaler = StandardScaler()
        x_test_num = scaler.fit_transform(x_test_num)
        # Tokenize text
        x_test_text = prepare_input(x_test_text.tolist())
        # Create a list to predict
        x_test = [x_test_text, x_test_num]
        pred = model.predict(x_test, batch_size = CFG.batch_size).astype(np.float32).reshape(-1)
        print(pred.shape)
        predictions.append(pred)
    # Release memory
    del x_test_num, x_test_text, x_test, pred
    gc.collect()
    test_dataset['predictions'] = np.concatenate(predictions, axis = 0)
    # Slice val_dataset with only the required columns
    test_dataset = test_dataset[['id_1', 'id_2', 'predictions']]
    # Copy val dataset and swap ids so we have A, B -> B, A
    test_dataset_c = test_dataset.copy()
    id1 = test_dataset_c['id_1']
    id2 = test_dataset_c['id_2']
    test_dataset_c['id1'] = id2
    test_dataset_c['id2'] = id1
    test_dataset = pd.concat([test_dataset, test_dataset_c], axis = 0, ignore_index = True)
    del id1, id2, test_dataset_c
    gc.collect()
    test_dataset['match_prediction'] = np.where(test_dataset['predictions'] >= CFG.best_thres, 1, 0)
    # Filter all the matches and get ids
    predictions = test_dataset[test_dataset['match_prediction'] == 1].groupby(['id_1'])['id_2'].apply(lambda x: list(np.unique(x))).reset_index()
    predictions['id_2'] = predictions['id_2'].apply(lambda x: ' '.join(x))
    # Add self
    predictions['id_2'] = predictions['id_1'] + ' ' + predictions['id_2']
    predictions.columns = ['id', 'prediction']
    # Get all the ids that did not found a match
    not_in = test[~test['id'].isin(predictions['id'])]['id'].values
    # Create a dataframe with this ids
    only_one = pd.DataFrame({'id': not_in, 'prediction': not_in})
    # Concatenate
    predictions = pd.concat([predictions, only_one], axis = 0, ignore_index = True)
    # Change columns name to prediction format
    predictions.columns = ['id', 'matches']
    # Save submission to disk
    predictions.to_csv('submission.csv', index = False)

# Read data
test = read_data()
if len(test) == 5:
    CFG.rounds = 4
    CFG.n_neighbors = 5
    
# Get initial features
features = [col for col in test.columns if col not in [CFG.target]]
# Generate test data
test, test_dataset = generate_test_data(test, rounds = CFG.rounds, n_neighbors = CFG.n_neighbors, features = features)
# Numerical Feature Engineering
test_dataset = add_lat_lon_distance_features(test_dataset)
# Categorical Feature Engineering
cat_columns = ['name', 'address', 'city', 'state', 'zip', 'country', 'url', 'phone', 'categories']
pair_cat_columns = [col + '_1' for col in cat_columns] + [col + '_2' for col in cat_columns]
for col in cat_columns:
    test_dataset = get_distance_cat(test_dataset, col)
# Get text column
test_dataset = add_sep_token(test_dataset)
# Drop unwanted columns
test_dataset.drop(pair_cat_columns, axis = 1, inplace = True)
inference(test_dataset, test)

