#!/usr/bin/env python
# coding: utf-8

# # 1. Introduction
# 
# Hey, thanks for viewing my Kernel!
# 
# If you like my work, please, leave an upvote: it will be really appreciated and it will motivate me in offering more content to the Kaggle community ! 😊

# In[1]:


get_ipython().system('pip install textstat')


# In[2]:


get_ipython().system('pip install featdist')


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import warnings
import datetime as dt
import math
import time

from featdist import numerical_ttt_dist
import re
import spacy
import textstat

from sklearn.feature_extraction.text import TfidfVectorizer

np.random.seed(0)
warnings.simplefilter("ignore")


# In[4]:


train = pd.read_csv("../input/advanced-dataset/fb3/train_fe.csv")
test = pd.read_csv("../input/advanced-dataset/fb3/test_fe.csv")
sub = pd.read_csv("../input/feedback-prize-english-language-learning/sample_submission.csv")

train["LABEL"].fillna("NONE", inplace=True)

LABELS = ['conventions', 'grammar', 'syntax', 'vocabulary', 'phraseology', 'cohesion']
labels = LABELS
stat_features = ["flesch_reading_ease", "flesch_kincaid_grade", "smog_index", "coleman_liau_index", "automated_readability_index", 
                 "dale_chall_readability_score", "difficult_words", "linsear_write_formula", "gunning_fog", "text_standard", 
                 "fernandez_huerta", "szigriszt_pazos", "gutierrez_polini", "crawford", "gulpease_index", "osman"]

display(train.head(1))
display(sub.head(1))


# In[5]:


print("train shape:", train.shape)
print("test shape:", test.shape)
print("sub shape:", sub.shape)


# In[6]:


print("train nan value sum:", train.isna().sum().sum())
print("test nan value sum:", test.isna().sum().sum())


# In[7]:


print("train dublicated value sum:", train.duplicated().sum().sum())
print("test dublicated value sum:", test.duplicated().sum().sum())


# # 2. Exploratory Data Analysis

# ## Distributions

# In[8]:


fig, axes = plt.subplots(2, 3, figsize=(16,8))
for ax, label in zip(axes.ravel(), labels):
    sns.histplot(data=train, x=label, ax=ax, label=label);
    ax.legend()


# In[9]:


def plot_important_words(data=None, feature='', n_important=10):
    from sklearn.linear_model import LinearRegression
    
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(data[feature])
    
    fig, axes = plt.subplots(6, 2, figsize=(16,3*n_important))
    fig.suptitle(feature, fontsize=32, y=0.9)
    for index, label in enumerate(labels):
        model = LinearRegression().fit(X.toarray(), data[label])
        df = pd.DataFrame()
        df['word'] = vectorizer.get_feature_names_out()
        df['coef_'] = model.coef_
        df.sort_values('coef_', inplace=True, ascending=False)
        df.reset_index(inplace=True, drop=True)
        df_poz = df.loc[:n_important, :].copy()
        df_poz.sort_values('coef_', inplace=True, ascending=True)
        axes[index][0].barh(df_poz.loc[:, 'word'].values, df_poz.loc[:, 'coef_'].values)
        axes[index][0].set_title(label + ' pozitive')
        axes[index][1].barh(df.loc[df.shape[0]-n_important:, 'word'].values, df.loc[df.shape[0]-n_important:, 'coef_'].values)
        axes[index][1].set_title(label + ' negative')
    
    del X
    gc.collect()


# In[10]:


plot_important_words(data=train, feature='full_text', n_important=10)


# # 2.1. Feature Engineering

# In[11]:


def add_new_features(df):
    def get_pos(text, model=None):
        # Create doc object
        doc = model(text)
        # Generate list of POS tags
        pos = [token.pos_ for token in doc]
        return pos
    def get_lemma(text, model=None):
        # Create doc object
        doc = model(text)
        # Generate list of lemmas
        lemma = [token.lemma_ for token in doc]
        return lemma
    def get_label(text, model=None):
        # Create doc object
        doc = model(text)
        # Generate list of all named entities and their labels
        label = [ent.label_ for ent in doc.ents]
        return label
    
    nlp = spacy.load('en_core_web_sm')
    
    df["POS"] = ''
    df["LEMMA"] = ''
    df["LABEL"] = ''
    for index, text in df[['full_text']].iterrows():
        pos = get_pos(text.values[0], model=nlp)
        lemma = get_lemma(text.values[0], model=nlp)
        label = get_label(text.values[0], model=nlp)
        
        df.loc[index, 'POS'] = ' '.join(pos)
        df.loc[index, 'LEMMA'] = ' '.join(lemma)
        df.loc[index, 'LABEL'] = ' '.join(label)


# In[12]:


'''
add_new_features(train)
add_new_features(test)

end_mem = train.memory_usage().sum() / 1024**2
print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
''';


# In[13]:


plot_important_words(data=train, feature='POS', n_important=10)


# In[14]:


plot_important_words(data=train, feature='LEMMA', n_important=10)


# In[15]:


plot_important_words(data=train, feature='LABEL', n_important=10)


# In[16]:


def add_stat_features(df):
    df["flesch_reading_ease"] = train["full_text"].apply(textstat.flesch_reading_ease)
    df["flesch_kincaid_grade"] = train["full_text"].apply(textstat.flesch_kincaid_grade)
    df["smog_index"] = train["full_text"].apply(textstat.smog_index)
    df["coleman_liau_index"] = train["full_text"].apply(textstat.coleman_liau_index)
    df["automated_readability_index"] = train["full_text"].apply(textstat.automated_readability_index)
    df["dale_chall_readability_score"] = train["full_text"].apply(textstat.dale_chall_readability_score)
    df["difficult_words"] = train["full_text"].apply(textstat.difficult_words)
    df["linsear_write_formula"] = train["full_text"].apply(textstat.linsear_write_formula)
    df["gunning_fog"] = train["full_text"].apply(textstat.gunning_fog)
    df["text_standard"] = train["full_text"].apply(textstat.text_standard)
    df["text_standard"] = df["text_standard"].str.extract('(\d+)').astype(int)
    df["fernandez_huerta"] = train["full_text"].apply(textstat.fernandez_huerta)
    df["szigriszt_pazos"] = train["full_text"].apply(textstat.szigriszt_pazos)
    df["gutierrez_polini"] = train["full_text"].apply(textstat.gutierrez_polini)
    df["crawford"] = train["full_text"].apply(textstat.crawford)
    df["gulpease_index"] = train["full_text"].apply(textstat.gulpease_index)
    df["osman"] = train["full_text"].apply(textstat.osman)
    
    stat_features = ["flesch_reading_ease", "flesch_kincaid_grade", "smog_index", "coleman_liau_index", 
                    "automated_readability_index", "dale_chall_readability_score", "difficult_words", "linsear_write_formula", "gunning_fog", 
                    "text_standard", "fernandez_huerta", "szigriszt_pazos", "gutierrez_polini", "crawford", 
                    "gulpease_index", "osman"]
    return stat_features


# In[17]:


'''
stat_features  = add_stat_features(train)
_  = add_stat_features(test)

end_mem = train.memory_usage().sum() / 1024**2
print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
''';


# In[18]:


for label in labels:
    print('-'*40,label,'-'*40)
    df_stats = numerical_ttt_dist(train=train, test=test, features=stat_features, target=label, ncols=4, nbins=20)
    display(df_stats)


# In[19]:


for feature in stat_features:
    fig, ax = plt.subplots(figsize=(16, 2))
    ax.hist(train[feature], bins=50)
    ax.set_title(feature)


# # 3. Validation

# In[20]:


from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.multioutput import MultiOutputRegressor

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def mc_rmse(y_true, y_pred):
    rmse_score_all = 0
    for i in range(0, 6):
        rmse_score = mean_squared_error(y_true[:,0], y_pred[:,0], squared=False)
        rmse_score_all += rmse_score
    return rmse_score_all / 6

gc.collect()


# In[21]:


vectorizer_full_text_n1 = TfidfVectorizer(stop_words="english", ngram_range=(1, 1))
X_full_text_n1 = vectorizer_full_text_n1.fit_transform(train["full_text"]).toarray()
X_test_full_text_n1 = vectorizer_full_text_n1.transform(test["full_text"]).toarray()

vectorizer_pos_n1 = TfidfVectorizer(ngram_range=(1, 1))
X_pos_n1 = vectorizer_pos_n1.fit_transform(train["POS"]).toarray()
X_test_pos_n1 = vectorizer_pos_n1.transform(test["POS"]).toarray()

vectorizer_lemma_n1 = TfidfVectorizer(stop_words="english", ngram_range=(1, 1))
X_lemma_n1 = vectorizer_lemma_n1.fit_transform(train["LEMMA"]).toarray()
X_test_lemma_n1 = vectorizer_lemma_n1.transform(test["LEMMA"]).toarray()

vectorizer_label_n1 = TfidfVectorizer(ngram_range=(1, 1))
X_label_n1 = vectorizer_label_n1.fit_transform(train["LABEL"]).toarray()
X_test_label_n1 = vectorizer_label_n1.transform(test["LABEL"]).toarray()

X_stats = train[stat_features].to_numpy()
X_text_stats = test[stat_features].to_numpy()

X_all = np.hstack((X_full_text_n1, X_pos_n1, X_lemma_n1, X_stats, X_label_n1))
X_test_all = np.hstack((X_test_full_text_n1, X_test_pos_n1, X_test_lemma_n1, X_text_stats, X_test_label_n1))

y = train[labels].to_numpy()

train_dict = {
    "full_text_n1": X_full_text_n1,
    "pos_n1": X_pos_n1,
    "lemma_n1": X_lemma_n1,
    "label_n1": X_label_n1,
    "stats": X_stats,
    "all_n1": X_all,
}

test_dict = {
    "full_text_n1": X_test_full_text_n1,
    "pos_n1": X_test_pos_n1,
    "lemma_n1": X_test_lemma_n1,
    "label_n1": X_test_label_n1,
    "stats": X_text_stats,
    "all_n1": X_test_all,
}

print("X_full_text_n1.shape:", X_full_text_n1.shape)
print("X_pos_n1.shape:", X_pos_n1.shape)
print("X_lemma_n1.shape:", X_lemma_n1.shape)
print("X_label_n1.shape:", X_label_n1.shape)
print("X_stats.shape:", X_stats.shape)
print("X_all.shape:", X_all.shape)


# In[22]:


'''
NFOLD=5
kf = KFold(n_splits=NFOLD, shuffle=True, random_state=0)
selected_stat_features = ['difficult_words', 'coleman_liau_index', 'crawford', 'smog_index',
                           'gulpease_index', 'dale_chall_readability_score']
score_details_r2 = {key:0 for key in stat_features}
score_details_rmse = {key:0 for key in stat_features}
for fold, (train_index, val_index) in enumerate(kf.split(X_full_text_n1)):
    X_train, X_val = X_full_text_n1[train_index], X_full_text_n1[val_index]
    y_train, y_val = X_stats[train_index], X_stats[val_index]
    
    scaler = StandardScaler().fit(X_stats)
    y_train = scaler.transform(y_train)
    y_val = scaler.transform(y_val)
    
    model_lgbm = MultiOutputRegressor(LGBMRegressor(), n_jobs=-1).fit(X_train, y_train)
    val_preds_lgbm = model_lgbm.predict(X_val)
    ms_rmse_lgbm = mc_rmse(y_val, val_preds_lgbm)
    
    print(f"fold {fold}, score: {ms_rmse_lgbm}")
    for i in range(16):
        r_square = r2_score(y_val[:, i], val_preds_lgbm[:, i])
        rmse = mean_squared_error(y_val[:,i], val_preds_lgbm[:,i], squared=False)
        score_details_r2[stat_features[i]] += r_square/NFOLD
        score_details_rmse[stat_features[i]] += rmse/NFOLD
    
    del X_train, X_val, y_train, y_val
    gc.collect()
''';


# In[23]:


values = [['difficult_words', 0.5384438792921359, 0.7089487973147242],
       ['coleman_liau_index', 0.6895924750313855, 0.5226910999827121],
       ['crawford', 0.811208552901785, 0.3406058995044606],
       ['smog_index', 0.8661168439374631, 0.24789673496235784],
       ['gulpease_index', 0.9611410381066687, 0.07475228352665779],
       ['dale_chall_readability_score', 0.9840829546206842,
        0.0031449149760098835],
       ['linsear_write_formula', 1.032164898629809, -0.06838537800144114],
       ['osman', 1.0024250425735577, -0.07605820772382624],
       ['flesch_reading_ease', 1.0059407058399736, -0.07815090692870719],
       ['gutierrez_polini', 1.0071147228791597, -0.0831228369714935],
       ['gunning_fog', 1.0059338892132892, -0.08481962059538001],
       ['szigriszt_pazos', 1.006124665202933, -0.08677813136123906],
       ['fernandez_huerta', 1.009898070092174, -0.09286688100582485],
       ['automated_readability_index', 1.0074979352332347,
        -0.09476267029511928],
       ['flesch_kincaid_grade', 1.0108127947382344, -0.09686693846586399],
       ['text_standard', 1.0300002916286095, -0.11601953173037179]]


# In[24]:


df_scores = pd.DataFrame(values,columns=["feature", "rmse", "r2"])
print("Statistical Feature Predictability")
df_scores


# In[25]:


'''
NFOLD=5
selected_stat_features = ['difficult_words', 'coleman_liau_index', 'crawford', 'smog_index',
                           'gulpease_index', 'dale_chall_readability_score']

X_stats = train[selected_stat_features].to_numpy()
X_text_stats = test[selected_stat_features].to_numpy()

scaler = StandardScaler().fit(X_stats)
X_stats = scaler.transform(X_stats)
X_text_stats = scaler.transform(X_text_stats)

lgbm_params = {
    'objective':'regression',
    'metric':'rmse',
    'learning_rate': 0.1,
    'force_col_wise':True,
}

np_val = np.zeros_like(X_stats)

kf = KFold(n_splits=NFOLD, shuffle=True, random_state=0)
for f_index, feature in enumerate(selected_stat_features):
    for fold, (train_index, val_index) in enumerate(kf.split(X_full_text_n1)):
        X_train, X_val = X_full_text_n1[train_index], X_full_text_n1[val_index]
        y_train, y_val = X_stats[train_index, f_index], X_stats[val_index, f_index]
        
        if f_index in ["difficult_words", "gulpease_index", 'dale_chall_readability_score']:
            dtrain = lgb.Dataset(X_train, y_train)
            dval = lgb.Dataset(X_val, y_val)

            model = lgb.train(params=lgbm_params, train_set=dtrain, valid_sets=[dval], num_boost_round=2000, 
                              early_stopping_rounds=100, verbose_eval=100)
            preds = model.predict(X_val)

            model.save_model(f'lgbm_{feature}_fold{fold}.txt')
        else:
            model = Ridge(alpha=0.7).fit(X_train, y_train)
            preds = model.predict(X_val)
            
            filename = f'ridge_{feature}_fold{fold}.sav'
            pickle.dump(model, open(filename, 'wb'))
        
        np_val[val_index, f_index] = preds
            
df_meta = pd.DataFrame(np_val, columns=selected_stat_features)
df_meta.to_csv("train_meta_csv", index=False)
''';


# # 3.1. Sklearn Models

# In[26]:


def get_cross_val_score(train_dict, y, test_dict, nfolds=5):
    sub_copy = sub.copy()
    score_details = {
        "model_name": ['ridge', 'knn', 'dt', 'lgbm'],
        "full_text_n1": [],
        "pos_n1": [],
        "lemma_n1": [],
        "label_n1": [],
        "stats": [],
        "all_n1": [],
    }
    for key in train_dict.keys():
        X = train_dict[key]
        X_test = test_dict[key]
        kf = KFold(n_splits=nfolds, shuffle=True, random_state=0)
        test_ridge, test_knn, test_dt, test_lgbm = (np.zeros((X_test.shape[0], y.shape[1])), np.zeros((X_test.shape[0], y.shape[1])), 
                                                    np.zeros((X_test.shape[0], y.shape[1])), np.zeros((X_test.shape[0], y.shape[1])))
        val_ridge, val_knn, val_dt, val_lgbm = 0, 0, 0, 0
        for train_index, val_index in kf.split(X):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            
            model_ridge = Ridge().fit(X_train, y_train)
            model_knn = KNeighborsRegressor().fit(X_train, y_train)
            model_dt = DecisionTreeRegressor().fit(X_train, y_train)
            model_lgbm = MultiOutputRegressor(LGBMRegressor(), n_jobs=-1).fit(X_train, y_train)
            
            val_preds_ridge = model_ridge.predict(X_val)
            val_preds_knn = model_knn.predict(X_val)
            val_preds_dt = model_dt.predict(X_val)
            val_preds_lgbm = model_lgbm.predict(X_val)
            
            test_preds_ridge = model_ridge.predict(X_test)
            test_preds_knn = model_knn.predict(X_test)
            test_preds_dt = model_dt.predict(X_test)
            test_preds_lgbm = model_lgbm.predict(X_test)
            
            ms_rmse_ridge = mc_rmse(y_val, val_preds_ridge)
            ms_rmse_knn = mc_rmse(y_val, val_preds_knn)
            ms_rmse_dt = mc_rmse(y_val, val_preds_dt)
            ms_rmse_lgbm = mc_rmse(y_val, val_preds_lgbm)
            
            val_ridge += ms_rmse_ridge / nfolds
            val_knn += ms_rmse_knn / nfolds
            val_dt += ms_rmse_dt / nfolds
            val_lgbm += ms_rmse_lgbm / nfolds
            
            test_ridge += test_preds_ridge / nfolds
            test_knn += test_preds_knn / nfolds
            test_dt += test_preds_dt / nfolds
            test_lgbm += test_preds_lgbm / nfolds
            
            del X_train, X_val
            gc.collect()
        
        score_details[key] = [val_ridge, val_knn, val_dt, val_lgbm]
        sub_copy.iloc[:, 1:] = test_ridge
        sub_copy.to_csv(f"sub_{key}_ridge_f{nfolds}.csv", index=False)
        sub_copy.iloc[:, 1:] = test_knn
        sub_copy.to_csv(f"sub_{key}_knn_f{nfolds}.csv", index=False)
        sub_copy.iloc[:, 1:] = test_dt
        sub_copy.to_csv(f"sub_{key}_dt_f{nfolds}.csv", index=False)
        sub_copy.iloc[:, 1:] = test_lgbm
        sub_copy.to_csv(f"sub_{key}_lgbm_f{nfolds}.csv", index=False)
        
        del X, X_test
        gc.collect()
        
    return pd.DataFrame(score_details)


# In[27]:


'''
df_scores = get_cross_val_score(train_dict, y, test_dict, nfolds=5)
df_scores
''';


# In[28]:


details = {
        "model_name": ['ridge', 'knn', 'dt', 'lgbm'],
        "full_text_n1": [0.576139, 0.699949, 0.872198, 0.582541],
        "pos_n1": [0.617827, 0.686117, 0.866192, 0.613606],
        "lemma_n1": [0.584900, 0.704855, 0.864353, 0.591403],
        "label_n1": [0.667974, 0.729899, 0.817450, 0.687709],
        "stats": [0.618053, 0.677997, 0.866587, 0.631731],
        "all_n1": [0.566777, 0.677145, 0.836108, 0.557545],
}
df_scores = pd.DataFrame(details)
df_scores.sort_values("all_n1", inplace=True)
df_scores


# # 3.2. Tensorflow Models

# In[29]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer._decayed_lr(tf.float32) # I use ._decayed_lr method instead of .lr
    return lr

EPOCHS = 100
LEARNING_RATE = tf.keras.optimizers.schedules.ExponentialDecay(
  initial_learning_rate=.01, decay_steps=10, decay_rate=.9)
CALLBACK = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=100)
BATCH_SIZE = 512
VAL_SPLIT = 0.2
HIDDEN_DENSE_NUM = 2
HIDEEN_DENSE_SIZE = 64
TF_SEED = 0
#LEARNING_RATE = 0.0025118864315095803
#LEARNING_RATE = 0.003981071705534973

y_new = (y-1) / 4

print("full_text max len:", np.max(train["full_text"].str.len()))
print("full_text max len:", np.max(train["POS"].str.len()))
print("full_text max len:", np.max(train["LEMMA"].str.len()))
print("full_text max len:", np.max(train["LABEL"].str.len()))


# In[30]:


def get_best_lr(initial_model, X, y, epochs=100, batch_size=128, plot=True, verbose=0):
    # ref: https://towardsdatascience.com/how-to-optimize-learning-rate-with-tensorflow-its-easier-than-you-think-164f980a7c7b
    initial_history = initial_model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_split=0.2,
        callbacks=[
            tf.keras.callbacks.LearningRateScheduler(
                lambda epoch: 1e-4 * 10 ** (epoch / 30)
            )
        ]
    )
    learning_rates = 1e-4 * (10 ** (np.arange(epochs) / 30))
    if plot:
        plt.semilogx(learning_rates, initial_history.history['loss'], lw=3, color='#000')
        plt.title('Learning rate vs. loss', size=20)
        plt.xlabel('Learning rate', size=14)
        plt.ylabel('Loss', size=14);
    index = np.argmin(initial_history.history['loss'])
    return learning_rates[index]


# # 3.2.1. Tensorflow LSTM with N_Grams

# In[31]:


def get_prepared_idf_X(feature_name, n_grams=4):
    def get_n_grams(text_sentences, n_grams=n_grams):
        final_sentences = []
        for i in range(len(text_sentences)):
            sentences = []
            words = text_sentences[i]
            for j in range(n_grams, len(words)):
                sentences.append(words[j-n_grams:j])
            final_sentences.append(sentences)
        return np.array(final_sentences)
    series = train[feature_name]

    tokenizer = TfidfVectorizer(stop_words="english")
    sequences = tokenizer.fit_transform(series).toarray()
    
    return get_n_grams(sequences)


# In[32]:


def get_small_lstm(input_dim=(), lr_rate=0.001):
    features_inputs = tf.keras.Input(input_dim, dtype=tf.float32)
    
    x = tf.keras.layers.LSTM(64, return_sequences=False, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=TF_SEED))(features_inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    for i in range(1):
        x = tf.keras.layers.Dense(HIDEEN_DENSE_SIZE, activation='swish', kernel_initializer=tf.keras.initializers.glorot_uniform(seed=TF_SEED))(x)
        x = tf.keras.layers.BatchNormalization()(x)
    output = tf.keras.layers.Dense(6, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=TF_SEED))(x)
    
    OPT = tf.keras.optimizers.Adam(learning_rate=lr_rate)
    lr = get_lr_metric(OPT)
    model = tf.keras.Model(inputs=[features_inputs], outputs=[output], name="small")
    model.compile(optimizer=OPT, loss='binary_crossentropy', metrics=[lr])
    return model


# In[33]:


X_full_text_tf = get_prepared_idf_X("full_text", n_grams=4)
X_full_text_tf.shape


# In[34]:


'''
initial_model = get_small_lstm(input_dim=(X_full_text_tf.shape[1], X_full_text_tf.shape[2], ))
lr_rate = get_best_lr(initial_model, X_full_text_tf, y_new, epochs=int(EPOCHS/10), batch_size=128, plot=True, verbose=1)
print("Best lr_rate:", lr_rate)
''';


# In[35]:


lr_rate = 0.001
model = get_small_lstm(input_dim=(X_full_text_tf.shape[1], X_full_text_tf.shape[2], ), lr_rate=lr_rate)
model.summary()


# In[36]:


start_time = time.time()
history_lstm_ngrams = model.fit(X_full_text_tf, y_new, epochs=int(EPOCHS/10), batch_size=128, 
                                validation_split=VAL_SPLIT, callbacks=[CALLBACK])
lstm_ngrams_time = time.time() - start_time
print("--- %s seconds ---" % (lstm_ngrams_time))


# In[37]:


del model
gc.collect()

fig, ax = plt.subplots(figsize=(16,4))
ax.plot(history_lstm_ngrams.history["loss"], label='train')
ax.plot(history_lstm_ngrams.history["val_loss"], label='valid')
ax.legend()
ax.set_ylabel("loss");
ax.set_xlabel("epochs");
ax.set_title("History");


# # 3.2.2. Tensorflow Embedding with LSTM

# In[38]:


selected_news_length = round(np.percentile(([len(x.split()) for x in train['full_text']]), 95))
max_vocab_length = TfidfVectorizer(stop_words="english", ngram_range=(1, 1)).fit_transform(train['full_text']).toarray().shape[1]

textVectorizer = tf.keras.layers.TextVectorization(
    max_tokens=max_vocab_length,
    output_mode='int',
    output_sequence_length=selected_news_length
);
textVectorizer.adapt(train['full_text'])


# In[39]:


textVectorizer.get_config()


# In[40]:


print(textVectorizer.get_vocabulary()[:5])


# In[41]:


def get_model_lstm_embeddig(lr_rate=0.001):
    features_inputs = tf.keras.Input((1, ), dtype=tf.string)
    
    x = textVectorizer(features_inputs)
    x = tf.keras.layers.Embedding(input_dim=max_vocab_length, output_dim=200, input_length=selected_news_length, 
                                  embeddings_initializer=tf.keras.initializers.glorot_uniform(seed=TF_SEED))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=TF_SEED)))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    for i in range(HIDDEN_DENSE_NUM):
        x = tf.keras.layers.Dense(HIDEEN_DENSE_SIZE, activation='swish', kernel_initializer=tf.keras.initializers.glorot_uniform(seed=TF_SEED))(x)
        x = tf.keras.layers.BatchNormalization()(x)
    output = tf.keras.layers.Dense(6, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=TF_SEED))(x)
    
    OPT = tf.keras.optimizers.Adam(learning_rate=lr_rate)
    lr = get_lr_metric(OPT)
    model = tf.keras.Model(inputs=[features_inputs], outputs=[output], name="small")
    model.compile(optimizer=OPT, loss='binary_crossentropy', metrics=[lr])
    return model


# In[42]:


'''
initial_model = get_model_lstm_embeddig()
lr_rate = get_best_lr(initial_model, train['full_text'], y_new, epochs=EPOCHS, batch_size=BATCH_SIZE, plot=True, verbose=1)
print("Best lr_rate:", lr_rate)
''';


# In[43]:


lr_rate = 0.023263050671536264
model = get_model_lstm_embeddig(lr_rate=lr_rate)
model.summary()


# In[44]:


start_time = time.time()
history_lstm_embedding = model.fit(train['full_text'], y_new, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                                   validation_split=VAL_SPLIT, callbacks=[CALLBACK])
lstm_embedding_time = time.time() - start_time
print("--- %s seconds ---" % (lstm_embedding_time))


# In[45]:


del model
gc.collect()

fig, ax = plt.subplots(figsize=(16,4))
ax.plot(history_lstm_embedding.history["loss"], label='train')
ax.plot(history_lstm_embedding.history["val_loss"], label='valid')
ax.legend()
ax.set_ylabel("loss");
ax.set_xlabel("epochs");
ax.set_title("History");


# # 3.2.3. Tensorflow Embedding with NN

# In[46]:


def get_model_nn_embeddig(lr_rate=0.001):
    features_inputs = tf.keras.Input((1, ), dtype=tf.string)
    
    x = textVectorizer(features_inputs)
    x = tf.keras.layers.Embedding(input_dim=max_vocab_length, output_dim=200, input_length=selected_news_length, 
                                  embeddings_initializer=tf.keras.initializers.glorot_uniform(seed=TF_SEED))(x)
    #x = tf.keras.layers.Reshape((128, 788))(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    #x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='swish', kernel_initializer=tf.keras.initializers.glorot_uniform(seed=TF_SEED))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    for i in range(HIDDEN_DENSE_NUM):
        x = tf.keras.layers.Dense(HIDEEN_DENSE_SIZE, activation='swish', kernel_initializer=tf.keras.initializers.glorot_uniform(seed=TF_SEED))(x)
        x = tf.keras.layers.BatchNormalization()(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    output = tf.keras.layers.Dense(6, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=TF_SEED))(x)
    
    OPT = tf.keras.optimizers.Adam(learning_rate=lr_rate)
    lr = get_lr_metric(OPT)
    model = tf.keras.Model(inputs=[features_inputs], outputs=[output], name="small")
    model.compile(optimizer=OPT, loss='binary_crossentropy', metrics=[lr])
    return model


# In[47]:


'''
initial_model = get_model_nn_embeddig()
lr_rate = get_best_lr(initial_model, train['full_text'], y_new, epochs=EPOCHS, batch_size=BATCH_SIZE, plot=True, verbose=1)
print("Best lr_rate:", lr_rate)
''';


# In[48]:


lr_rate = 0.0009261187281287936
model = get_model_nn_embeddig(lr_rate=lr_rate)
model.summary()


# In[49]:


start_time = time.time()
history_nn_embedding = model.fit(train['full_text'], y_new, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                                 validation_split=VAL_SPLIT, callbacks=[CALLBACK])
nn_embedding_time = time.time() - start_time
print("--- %s seconds ---" % (nn_embedding_time))


# In[50]:


del model
gc.collect()

fig, ax = plt.subplots(figsize=(16,4))
ax.plot(history_nn_embedding.history["loss"], label='train')
ax.plot(history_nn_embedding.history["val_loss"], label='valid')
ax.legend()
ax.set_ylabel("loss");
ax.set_xlabel("epochs");
ax.set_title("History");


# # 3.2.4. Tensorflow Pre-trained Embedding with LSTM

# In[51]:


# ref: https://keras.io/examples/nlp/pretrained_word_embeddings/
path_to_glove_file = "../input/glove-global-vectors-for-word-representation/glove.6B.200d.txt"

embeddings_index = {}
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print("Found %s word vectors." % len(embeddings_index))


# In[52]:


embeddings_index.get('apple')


# In[53]:


# ref: https://keras.io/examples/nlp/pretrained_word_embeddings/
voc = textVectorizer.get_vocabulary()
word_index = dict(zip(voc, range(len(voc))))

num_tokens = len(voc)
embedding_dim = 200
hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # Words not found in embedding index will be all-zeros.
        # This includes the representation for "padding" and "OOV"
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))


# In[54]:


def get_prepared_tf_X(feature_name, n_grams=4):
    def get_n_grams(text_sentences, n_grams=n_grams):
        final_sentences = []
        for i in range(len(text_sentences)):
            sentences = []
            words = text_sentences[i]
            for j in range(n_grams, len(words)):
                sentences.append(words[j-n_grams:j])
            final_sentences.append(sentences)
        return np.array(final_sentences)
    series = train[feature_name].str.split()

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(series)
    sequences = tokenizer.texts_to_sequences(series)

    padded_text = pad_sequences(sequences, maxlen=6044, padding='post')
    return get_n_grams(padded_text)


# In[55]:


def get_model_lstm_pre_embeddig(lr_rate=0.001):
    features_inputs = tf.keras.Input((1, ), dtype=tf.string)
    
    x = textVectorizer(features_inputs)
    x = tf.keras.layers.Embedding(input_dim=num_tokens, output_dim=200, 
                                  embeddings_initializer=keras.initializers.Constant(embedding_matrix),
                                  trainable=False)(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=TF_SEED)))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    for i in range(HIDDEN_DENSE_NUM):
        x = tf.keras.layers.Dense(HIDEEN_DENSE_SIZE, activation='swish', kernel_initializer=tf.keras.initializers.glorot_uniform(seed=TF_SEED))(x)
        x = tf.keras.layers.BatchNormalization()(x)
    output = tf.keras.layers.Dense(6, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=TF_SEED))(x)
    
    OPT = tf.keras.optimizers.Adam(learning_rate=lr_rate)
    lr = get_lr_metric(OPT)
    model = tf.keras.Model(inputs=[features_inputs], outputs=[output], name="small")
    model.compile(optimizer=OPT, loss='binary_crossentropy', metrics=[lr])
    return model


# In[56]:


'''
initial_model = get_model_lstm_pre_embeddig()
lr_rate = get_best_lr(initial_model, train['full_text'], y_new, epochs=EPOCHS, batch_size=BATCH_SIZE, plot=True, verbose=1)
print("Best lr_rate:", lr_rate)
''';


# In[57]:


lr_rate = 0.11659144011798324
model = get_model_lstm_pre_embeddig(lr_rate=lr_rate)
model.summary()


# In[58]:


start_time = time.time()
history_lstm_pre_embedding = model.fit(train['full_text'], y_new, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                                       validation_split=VAL_SPLIT, callbacks=[CALLBACK])
lstm_pre_embedding_time = time.time() - start_time
print("--- %s seconds ---" % (lstm_pre_embedding_time))


# In[59]:


del model
gc.collect()

fig, ax = plt.subplots(figsize=(16,4))
ax.plot(history_lstm_pre_embedding.history["loss"], label='train')
ax.plot(history_lstm_pre_embedding.history["val_loss"], label='valid')
ax.legend()
ax.set_ylabel("loss");
ax.set_xlabel("epochs");
ax.set_title("History");


# # 3.2.5. Tensorflow Pre-trained Embedding with NN

# In[60]:


def get_model_nn_pre_embeddig(lr_rate=0.001):
    features_inputs = tf.keras.Input((1, ), dtype=tf.string)
    
    x = textVectorizer(features_inputs)
    x = tf.keras.layers.Embedding(input_dim=num_tokens, output_dim=200, 
                                  embeddings_initializer=keras.initializers.Constant(embedding_matrix),
                                  trainable=False)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(128, activation='swish', kernel_initializer=tf.keras.initializers.glorot_uniform(seed=TF_SEED))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    for i in range(HIDDEN_DENSE_NUM):
        x = tf.keras.layers.Dense(HIDEEN_DENSE_SIZE, activation='swish', kernel_initializer=tf.keras.initializers.glorot_uniform(seed=TF_SEED))(x)
        x = tf.keras.layers.BatchNormalization()(x)
    output = tf.keras.layers.Dense(6, kernel_initializer=tf.keras.initializers.glorot_uniform(seed=TF_SEED))(x)
    
    OPT = tf.keras.optimizers.Adam(learning_rate=lr_rate)
    lr = get_lr_metric(OPT)
    model = tf.keras.Model(inputs=[features_inputs], outputs=[output], name="small")
    model.compile(optimizer=OPT, loss='binary_crossentropy', metrics=[lr])
    return model


# In[61]:


initial_model = get_model_nn_pre_embeddig()
lr_rate = get_best_lr(initial_model, train['full_text'], y_new, epochs=EPOCHS, batch_size=BATCH_SIZE, plot=True, verbose=0)
print("Best lr_rate:", lr_rate)


# In[62]:


model = get_model_nn_pre_embeddig(lr_rate=lr_rate)
model.summary()


# In[63]:


start_time = time.time()
history_nn_pre_embedding = model.fit(train['full_text'], y_new, epochs=EPOCHS, batch_size=BATCH_SIZE, 
                                     validation_split=VAL_SPLIT, callbacks=[CALLBACK])
nn_pre_embedding_time = time.time() - start_time
print("--- %s seconds ---" % (nn_pre_embedding_time))


# In[64]:


del model
gc.collect()

fig, ax = plt.subplots(figsize=(16,4))
ax.plot(history_nn_pre_embedding.history["loss"], label='train')
ax.plot(history_nn_pre_embedding.history["val_loss"], label='valid')
ax.legend()
ax.set_ylabel("loss");
ax.set_xlabel("epochs");
ax.set_title("History");


# In[65]:


fig, ax = plt.subplots(figsize=(16,4))
ax.plot(history_lstm_ngrams.history["val_loss"], label='lstm_ngrams', linestyle="solid")
ax.plot(history_lstm_embedding.history["val_loss"], label='lstm_embedding', linestyle="dotted")
ax.plot(history_nn_embedding.history["val_loss"], label='nn_embedding', linestyle="dashed")
ax.plot(history_lstm_pre_embedding.history["val_loss"], label='lstm_pre_embedding', linestyle="dashdot")
ax.plot(history_nn_pre_embedding.history["val_loss"], label='nn_pre_embedding', linestyle="dotted")
ax.legend()
ax.set_ylabel("loss");
ax.set_xlabel("epochs");
ax.set_title("History");


# In[66]:


model_sorted = []
time_sorted = []

exe_times = [lstm_ngrams_time, lstm_embedding_time, nn_embedding_time, lstm_pre_embedding_time, nn_pre_embedding_time]
model_names = ["lstm_ngrams", "lstm_embedding", "nn_embedding", "lstm_pre_embedding", "nn_pre_embedding"]
sorted_index = sorted(range(len(exe_times)), key=lambda k: exe_times[k], reverse=True)

for i in sorted_index:
    model_sorted.append(model_names[i])
    time_sorted.append(exe_times[i])

fig, ax = plt.subplots(figsize=(16,4))
ax.barh(model_sorted, time_sorted)
ax.set_ylabel("model name");
ax.set_xlabel("exe time");
ax.set_title("Execution Time for Each Model");

for p in ax.patches:
    txt = str(p.get_width().round(2))
    txt_x = p.get_width() 
    txt_y = p.get_y() + p.get_height() / 2
    ax.text(txt_x,txt_y,txt)


# In[67]:


model_sorted = []
time_sorted = []

exe_times = [np.min(history_lstm_ngrams.history["val_loss"]), np.min(history_lstm_embedding.history["val_loss"]), 
             np.min(history_nn_embedding.history["val_loss"]), np.min(history_lstm_pre_embedding.history["val_loss"]), 
             np.min(history_nn_pre_embedding.history["val_loss"])]
model_names = ["lstm_ngrams", "lstm_embedding", "nn_embedding", "lstm_pre_embedding", "nn_pre_embedding"]
sorted_index = sorted(range(len(exe_times)), key=lambda k: exe_times[k], reverse=True)

for i in sorted_index:
    model_sorted.append(model_names[i])
    time_sorted.append(exe_times[i])

fig, ax = plt.subplots(figsize=(16,4))
ax.barh(model_sorted, time_sorted)
ax.set_ylabel("model name");
ax.set_xlabel("loss");
ax.set_title("Loss for Each Model");

for p in ax.patches:
    txt = str(p.get_width().round(5))
    txt_x = p.get_width() 
    txt_y = p.get_y() + p.get_height() / 2
    ax.text(txt_x,txt_y,txt)


# # 3.3. Transformers

# # 3.3.1. TF-Bert

# In[68]:


import transformers
from transformers import BertTokenizer, TFBertModel
from tqdm import tqdm

MAX_LENGTH = 512
BATCH = 16


# In[69]:


tokenizer = BertTokenizer.from_pretrained("bert-base-cased")


# In[70]:


token = tokenizer.encode_plus(
    train['full_text'].iloc[0],
    max_length=MAX_LENGTH,
    truncation=True,
    padding="max_length",
    add_special_tokens=True, # [CLS] [PAD] [SEP]
    return_tensors="tf"
)


# In[71]:


token["input_ids"]


# In[72]:


token["token_type_ids"]


# In[73]:


token["attention_mask"]


# In[74]:


X_input_ids = np.zeros((len(train), MAX_LENGTH))
X_attn_masks = np.zeros((len(train), MAX_LENGTH))

def generate_training_data(df, ids, masks, tokenizer):
    for i, text in tqdm(enumerate(df["full_text"])):
        tokenized_text = tokenizer.encode_plus(
            text,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            add_special_tokens=True, # [CLS] [PAD] [SEP]
            return_tensors="tf"
        )
        ids[i, :] = tokenized_text.input_ids
        masks[i, :] = tokenized_text.attention_mask
    return ids, masks

X_input_ids, X_attn_masks = generate_training_data(train, X_input_ids, X_attn_masks, tokenizer)
X_input_ids


# In[75]:


labels = train[LABELS].to_numpy()
labels


# In[76]:


X_train_tf = tf.data.Dataset.from_tensor_slices((X_input_ids, X_attn_masks, labels))
X_train_tf.take(1)


# In[77]:


def dataset_map(input_ids, attn_masks, labels):
    return {
        'input_ids': input_ids,
        'attention_mask': attn_masks,
    }, labels
X_train_tf = X_train_tf.map(dataset_map)
X_train_tf.take(1)


# In[78]:


X_train_tf = X_train_tf.shuffle(10000).batch(BATCH, drop_remainder=True)


# In[79]:


p=0.8
train_size = int((len(train)//BATCH)*p)
train_size


# In[80]:


train_dataset = X_train_tf.take(train_size)
val_dataset = X_train_tf.skip(train_size)


# In[81]:


bert_model = TFBertModel.from_pretrained("bert-base-cased", trainable=False)
for layer in bert_model.layers:
    layer.trainable=False
    for w in layer.weights: w._trainable=False


# In[82]:


def get_tf_bert():
    input_ids = tf.keras.layers.Input(shape=(MAX_LENGTH,), name="input_ids", dtype="int32")
    attention_masks = tf.keras.layers.Input(shape=(MAX_LENGTH,), name="attention_mask", dtype="int32")

    bert_embds = bert_model.bert(input_ids, attention_masks)[1]
    x = tf.keras.layers.Dense(512, activation="relu")(bert_embds)
    output_layer = tf.keras.layers.Dense(6, name="output_layer")(x)

    model = tf.keras.Model(inputs=[input_ids, attention_masks], outputs=output_layer)
    model.compile(optimizer="adam", loss='binary_crossentropy')
    return model


# In[83]:


model = get_tf_bert()
model.summary()


# In[84]:


start_time = time.time()
history_tf_bert = model.fit(train_dataset, validation_data=val_dataset, epochs=1)
tf_bert_time = time.time() - start_time
print("--- %s seconds ---" % (tf_bert_time))


# In[85]:


history_tf_bert.history["loss"], history_tf_bert.history["val_loss"]


# In[86]:


model.save("tf_bert_model")


# In[87]:


del model, bert_model
gc.collect()


# In[88]:


loaded_model = tf.keras.models.load_model("tf_bert_model")


# In[89]:


def prepare_testing_data(input_text, tokenizer):
    token = tokenizer.encode_plus(
        input_text,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
        add_special_tokens=True, # [CLS] [PAD] [SEP]
        return_tensors="tf"
    )
    return {
        'input_ids': tf.cast(token.input_ids,tf.float64),
        'attention_mask': tf.cast(token.attention_mask,tf.float64)
    }


# In[90]:


input_text = "I think that students would benefit from learning at home,because they wont have to change and get up early in the morning to shower and do there hair."
tokenized_input_text = prepare_testing_data(input_text,tokenizer)


# In[91]:


tokenized_input_text


# In[92]:


pred = loaded_model.predict(tokenized_input_text)
pred

