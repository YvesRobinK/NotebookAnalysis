#!/usr/bin/env python
# coding: utf-8

# # AI Generated Text Detection with DNN and TFIDF
# Forked and modified from: https://www.kaggle.com/code/lonnieqin/ai-generated-text-detection-with-dnn-and-tfidf
# - DNN and TFIDF fit only on train: 0.923
# - Add spell correction for test + Fit TFIDF on train + test: 0.913
# - Add spell correction for test + Fit TFIDF on test: 0.917
# 
# These results could vary due to the random elements of this notebook, but they may provide some insights.
# 

# ## Imports

# In[1]:


import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import TextVectorization
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import roc_auc_score


# ## Configuration

# In[2]:


class CFG:
    batch_size = 128
    is_training = True
    epochs = 30


# # Importing files and Feature Engineering

# In[3]:


external_df = pd.read_csv("/kaggle/input/daigt-external-dataset/daigt_external_dataset.csv", sep=',')
print(external_df.shape)
external_df = external_df.rename(columns={'generated': 'label'})
external_df = external_df[["source_text"]]
external_df.columns = ["text"]
external_df['text'] = external_df['text'].str.replace('\n', '')
external_df["label"] = 1

train = pd.read_csv("/kaggle/input/llm-7-prompt-training-dataset/train_essays_RDizzl3_seven_v1.csv")
train=pd.concat([train, external_df])
test = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/test_essays.csv')


# In[4]:


train.value_counts("label").plot(kind="bar")


# In[5]:


vectorizer = TextVectorization(max_tokens=50000, output_mode="tf-idf", ngrams=(3,7)) #3,7 from https://www.kaggle.com/code/yongsukprasertsuk/ai-generated-text-mod-weight-add-more-data-0-928?scriptVersionId=151460374
vectorizer.adapt(train["text"])


# ## Modeling

# In[6]:


def fbeta(y_true, y_pred, beta = 1.0):
    y_true_count = tf.reduce_sum(y_true)
    ctp = tf.reduce_sum(y_true * y_pred)
    cfp = tf.reduce_sum((1.0 - y_true) * y_pred)
    beta_squared = beta * beta
    c_precision = tf.where(ctp + cfp == 0.0, 0.0, ctp / (ctp + cfp))
    c_recall =  tf.where(y_true_count == 0.0, 0.0, ctp / y_true_count)
    return tf.where(
        c_precision + c_recall == 0, 
        0.0, 
        tf.divide((1.0 + beta_squared) * (c_precision * c_recall),  (beta_squared * c_precision + c_recall))
    )

def inference(model, X_val):
    if "keras" in str(type(model)):
        y_pred = model.predict(X_val, verbose=2).reshape(-1)
    else:
        y_pred = model.predict_proba(X_val)[:, 1].reshape(-1)
    return y_pred

def evaluate_model(model, X_val, y_val):
    y_pred = inference(model, X_val)
    auc = roc_auc_score(y_val, y_pred)
    print(f"AUC for {model}: {auc}")
    return {
        "model": model,
        "auc": auc
    }
def make_dataset(X, y, batch_size, mode):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if mode == "train":
        dataset = dataset.shuffle(batch_size * 4) 
    dataset = dataset.batch(batch_size)
    dataset = dataset.cache().prefetch(tf.data.AUTOTUNE)
    return dataset

def get_model():
    inputs = keras.Input(shape=(), dtype=tf.string)
    x = vectorizer(inputs)
    x = layers.Dense(32, activation="swish")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(16, activation="swish")(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1)(x)
    model = keras.Model(inputs, output, name="model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(4e-4), 
        loss=tf.keras.losses.BinaryCrossentropy(
            from_logits=True,
            label_smoothing=0.2,
        ), 
        metrics=[
            "accuracy", 
            keras.metrics.AUC(name="auc"),
            fbeta
        ]
    )
    return model

def train_models(X_train, y_train, X_val, y_val, fold):
    model_path = f"model_{fold}.tf"
    checkpoints = []
    if CFG.is_training:
        model = get_model()
        train_ds =  make_dataset(X_train, y_train, CFG.batch_size, "train")
        valid_ds =  make_dataset(X_val, y_val, CFG.batch_size, "valid")
        model.fit(
            train_ds, 
            epochs=CFG.epochs, 
            validation_data=valid_ds,
            callbacks=[
                keras.callbacks.ReduceLROnPlateau(patience=5, min_delta=1e-4, min_lr=1e-6),
                keras.callbacks.ModelCheckpoint(model_path, monitor="val_auc", mode="max", save_best_only=True)
            ]
        )
    else:
        model = keras.models.load_model(
            f"/kaggle/input/ai-generated-text-dnn-detector/model_{fold}.tf", 
            custom_objects={
                "fbeta": fbeta
            }
        )
    checkpoints.append(evaluate_model(model, X_val, y_val))
    return checkpoints


# In[7]:


model = get_model()
model.summary()
tf.keras.utils.plot_model(model, show_shapes=True)


# In[8]:


from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(5, shuffle=True, random_state=42)
models = []
for fold, (train_index, valid_index) in enumerate(kfold.split(train, train["label"])):
    X_train = train.iloc[train_index]["text"]
    y_train = train.iloc[train_index]["label"]
    X_val = train.iloc[valid_index]["text"]
    y_val = train.iloc[valid_index]["label"]
    models += train_models(X_train, y_train, X_val, y_val, fold)


# ## Create Submission

# In[9]:


test["generated"] = np.mean([inference(model["model"], test["text"]) for model in models], axis=0)
test[["id", "generated"]].to_csv('submission.csv', index=False)
test[["id", "generated"]].head()

