#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install -U -q wandb')


# In[2]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gc

import numpy as np
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

import tensorflow as tf
from tensorflow.keras import layers

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

import wandb
from wandb.keras import WandbCallback


# # Config TPU

# In[3]:


try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Device:', tpu.master())
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()
print('Number of replicas:', strategy.num_replicas_in_sync)


# # WandB Login
# 
# Thanks to [@usharengaraju](https://www.kaggle.com/usharengaraju) for her useful notebooks with W&B

# ![wandb](https://i.imgur.com/gb6B4ig.png)

# In[4]:


try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    secret_value_0 = user_secrets.get_secret("wandb_api_key")
    wandb.login(key=secret_value_0)
    anony=None
except:
    anony = "must"
    print('If you want to use your W&B account, go to Add-ons -> Secrets and provide your W&B access token. Use the Label name as wandb_api. \nGet your W&B access token from here: https://wandb.ai/authorize')


# # Load Data

# You can find **Pseudo Labels** Dataset [here](https://www.kaggle.com/remekkinas/tps12-pseudolabels) by [@remekkinas](https://www.kaggle.com/remekkinas)

# In[5]:


get_ipython().run_cell_magic('time', '', 'train = pd.read_csv(\'../input/tabular-playground-series-dec-2021/train.csv\').set_index("Id")\ntest = pd.read_csv(\'../input/tabular-playground-series-dec-2021/test.csv\').set_index("Id")\npseudo = pd.read_csv(\'../input/tps12-pseudolabels/tps12-pseudolabels_v2.csv\').set_index("Id")\n\ntrain = pd.concat([train, pseudo], axis=0)\n\nsample_submission = pd.read_csv("../input/tabular-playground-series-dec-2021/sample_submission.csv")\n\nfeature_cols = test.columns.tolist()\ncnt_cols = [col for col in feature_cols if (not col.startswith("Soil_Type")) and (not col.startswith("Wilderness_Area"))]\nbin_cols = [col for col in feature_cols if col not in cnt_cols]\n')


# In[6]:


plt.figure(figsize=(10,5))
axs = sns.countplot(x="Cover_Type", data=train)
plt.xlabel("Cover Type")
axs.bar_label(axs.containers[0])
plt.show()


# In[7]:


train["Cover_Type"] = train["Cover_Type"] - 1


# # Reduce Memory Usage

# In[8]:


for col in feature_cols:
    if col in cnt_cols:
        train[col] = train[col].astype("float32")
        test[col] = test[col].astype("float32")
    else:
        train[col] = train[col].astype("bool")
        test[col] = test[col].astype("bool")


# In[9]:


train.describe().T


# # Isolation Forest (Outlier Detection)
# The IsolationForest ‘isolates’ observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.

# In[10]:


get_ipython().run_cell_magic('time', '', 'isf = IsolationForest(random_state=42)\ntrain["outlier_isf"] = isf.fit_predict(train[feature_cols])\ntest["outlier_isf"] = isf.predict(test[feature_cols])\n\nprint(train["outlier_isf"].value_counts())\nprint(test["outlier_isf"].value_counts())\n')


# In[11]:


train["outlier_isf"] = train["outlier_isf"] == -1
test["outlier_isf"] = test["outlier_isf"] == -1

train["outlier_isf"] = train["outlier_isf"].astype("bool")
test["outlier_isf"] = test["outlier_isf"].astype("bool")


# In[12]:


plt.figure(figsize=(10,5))
axs = sns.countplot(x=train.loc[train.outlier_isf==True,"Cover_Type"])
axs.bar_label(axs.containers[0])
plt.title("Outliers Count Isolation Forest")
plt.show()


# In[13]:


del isf
_ = gc.collect()


# In[14]:


feature_cols.append("outlier_isf")
bin_cols.append("outlier_isf")


# # MiniBatch KMeans
# 
# The MiniBatchKMeans is a variant of the KMeans algorithm which uses mini-batches to reduce the computation time, while still attempting to optimise the same objective function. Mini-batches are subsets of the input data, randomly sampled in each training iteration. These mini-batches drastically reduce the amount of computation required to converge to a local solution. In contrast to other algorithms that reduce the convergence time of k-means, mini-batch k-means produces results that are generally only slightly worse than the standard algorithm.

# In[15]:


sc = StandardScaler()
x = train.copy()
t = test.copy()
x[cnt_cols] = sc.fit_transform(x[cnt_cols])
t[cnt_cols] = sc.transform(t[cnt_cols])


# In[16]:


get_ipython().run_cell_magic('time', '', 'n_clusters = 14\ncd_feature = False # cluster distance instead of cluster number  \n\nkmeans = MiniBatchKMeans(n_clusters=n_clusters, max_iter=300, batch_size=256*5, random_state=42)\n\nif cd_feature:\n    cluster_cols = [f"cluster{i+1}" for i in range(n_clusters)]\n    \n    X_cd = kmeans.fit_transform(x[feature_cols])\n    X_cd = pd.DataFrame(X_cd, columns=cluster_cols, index=x.index)\n    train = train.join(X_cd)\n    \n    X_cd = kmeans.transform(t[feature_cols])\n    X_cd = pd.DataFrame(X_cd, columns=cluster_cols, index=t.index)\n    test = test.join(X_cd)\n\nelse:\n    cluster_cols = ["cluster"]  \n    train["cluster"] = kmeans.fit_predict(x[feature_cols])\n    test["cluster"] = kmeans.predict(t[feature_cols])\n    \n\nfeature_cols += cluster_cols\n\ntrain.head()\n')


# In[17]:


plt.figure(figsize=(20,8))
ax = sns.countplot(x="cluster", data=train, hue="Cover_Type")
plt.xlabel("Clusters")
plt.show()


# # PCA

# In[18]:


x[cluster_cols] = train[cluster_cols].copy()
t[cluster_cols] = test[cluster_cols].copy()


# In[19]:


pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(x[feature_cols])
T_pca = pca.transform(t[feature_cols])

pca_cols = [f"PC{i+1}" for i in range(X_pca.shape[1])]

X_pca = pd.DataFrame(X_pca, columns=pca_cols, index=train.index)
T_pca = pd.DataFrame(T_pca, columns=pca_cols, index=test.index)

train = pd.concat([train, X_pca], axis=1)
test = pd.concat([test, T_pca], axis=1)
train.head()


# In[20]:


del x, t, X_pca, T_pca
_ = gc.collect()


# In[21]:


plt.figure(figsize=(15,8))
sns.scatterplot(data=train, x="PC1", y="PC2", hue="Cover_Type", alpha=0.8, palette="deep")
plt.show()


# In[22]:


feature_cols += ["PC1", "PC2"]


# In[23]:


train["likely_type3"] = train["PC2"] < -2.2
train["likely_type2"] = (train["PC2"] < 0) & (train["PC2"] > -2.2)
train["likely_type7"] = train["PC2"] > 3.9
train["likely_type1"] = (train["PC2"] > 1) & (train["PC2"] < 4)

test["likely_type3"] = test["PC2"] < -2.2
test["likely_type2"] = (test["PC2"] < 0) & (train["PC2"] > -2.2)
test["likely_type7"] = test["PC2"] > 3.9
test["likely_type1"] = (test["PC2"] > 1) & (train["PC2"] < 4)


# In[24]:


feature_cols += ["likely_type3", "likely_type2", "likely_type7", "likely_type1"]
bin_cols += ["likely_type3", "likely_type2", "likely_type7", "likely_type1"]


# # Add More Features
# 
# Thanks to [@lucamassaron](https://www.kaggle.com/lucamassaron) for [this discussion](https://www.kaggle.com/c/tabular-playground-series-dec-2021/discussion/291839).

# In[25]:


def r(x):
    if x+180>360:
        return x-180
    else:
        return x+180

train['Aspect2'] = train.Aspect.map(r)
test['Aspect2'] = test.Aspect.map(r)

train.loc[train["Aspect"] < 0, "Aspect"] += 360
test.loc[test["Aspect"] < 0, "Aspect"] += 360

train.loc[train["Aspect"] > 359, "Aspect"] -= 360
test.loc[test["Aspect"] > 359, "Aspect"] -= 360


# In[26]:


train['Highwater'] = train.Vertical_Distance_To_Hydrology < 0
test['Highwater'] = test.Vertical_Distance_To_Hydrology < 0

train['DistHydro'] = train.Horizontal_Distance_To_Hydrology < 0
test['DistHydro'] = test.Horizontal_Distance_To_Hydrology < 0

train['DistRoad'] = train.Horizontal_Distance_To_Roadways < 0
test['DistRoad'] = test.Horizontal_Distance_To_Roadways < 0

train['DistFire'] = train.Horizontal_Distance_To_Fire_Points < 0
test['DistFire'] = test.Horizontal_Distance_To_Fire_Points < 0

train['Hillshade_3pm_is_zero'] = train.Hillshade_3pm == 0
test['Hillshade_3pm_is_zero'] = test.Hillshade_3pm == 0


# In[27]:


train['EHiElv'] = train['Horizontal_Distance_To_Roadways'] * train['Elevation']
test['EHiElv'] = test['Horizontal_Distance_To_Roadways'] * test['Elevation']

train['EViElv'] = train['Vertical_Distance_To_Hydrology'] * train['Elevation']
test['EViElv'] = test['Vertical_Distance_To_Hydrology'] * test['Elevation']


# In[28]:


train['EVDtH'] = train.Elevation-train.Vertical_Distance_To_Hydrology
test['EVDtH'] = test.Elevation-test.Vertical_Distance_To_Hydrology

train['EHDtH'] = train.Elevation-train.Horizontal_Distance_To_Hydrology*0.2
test['EHDtH'] = test.Elevation-test.Horizontal_Distance_To_Hydrology*0.2


# In[29]:


train['Distanse_to_Hydrolody'] = (train['Horizontal_Distance_To_Hydrology']**2+train['Vertical_Distance_To_Hydrology']**2)**0.5
test['Distanse_to_Hydrolody'] = (test['Horizontal_Distance_To_Hydrology']**2+test['Vertical_Distance_To_Hydrology']**2)**0.5

train['Hydro_Fire_1'] = train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Fire_Points']
test['Hydro_Fire_1'] = test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Fire_Points']

train['Hydro_Fire_2'] = abs(train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Fire_Points'])
test['Hydro_Fire_2'] = abs(test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Fire_Points'])

train['Hydro_Road_1'] = abs(train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Roadways'])
test['Hydro_Road_1'] = abs(test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Roadways'])

train['Hydro_Road_2'] = abs(train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Roadways'])
test['Hydro_Road_2'] = abs(test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Roadways'])

train['Fire_Road_1'] = abs(train['Horizontal_Distance_To_Fire_Points']+train['Horizontal_Distance_To_Roadways'])
test['Fire_Road_1'] = abs(test['Horizontal_Distance_To_Fire_Points']+test['Horizontal_Distance_To_Roadways'])

train['Fire_Road_2'] = abs(train['Horizontal_Distance_To_Fire_Points']-train['Horizontal_Distance_To_Roadways'])
test['Fire_Road_2'] = abs(test['Horizontal_Distance_To_Fire_Points']-test['Horizontal_Distance_To_Roadways'])


# In[30]:


train["new_f1"] = train["Elevation"] + train["Horizontal_Distance_To_Roadways"] + train["Horizontal_Distance_To_Fire_Points"]
test["new_f1"] = test["Elevation"] + test["Horizontal_Distance_To_Roadways"] + test["Horizontal_Distance_To_Fire_Points"]

train["new_f2"] = (train["Hillshade_Noon"] + train["Hillshade_3pm"]) - train["Hillshade_9am"]
test["new_f2"] = (test["Hillshade_Noon"] + test["Hillshade_3pm"]) - test["Hillshade_9am"]


# In[31]:


train.loc[train["Hillshade_9am"] < 0, "Hillshade_9am"] = 0
test.loc[test["Hillshade_9am"] < 0, "Hillshade_9am"] = 0

train.loc[train["Hillshade_Noon"] < 0, "Hillshade_Noon"] = 0
test.loc[test["Hillshade_Noon"] < 0, "Hillshade_Noon"] = 0

train.loc[train["Hillshade_3pm"] < 0, "Hillshade_3pm"] = 0
test.loc[test["Hillshade_3pm"] < 0, "Hillshade_3pm"] = 0

train.loc[train["Hillshade_9am"] > 255, "Hillshade_9am"] = 255
test.loc[test["Hillshade_9am"] > 255, "Hillshade_9am"] = 255

train.loc[train["Hillshade_Noon"] > 255, "Hillshade_Noon"] = 255
test.loc[test["Hillshade_Noon"] > 255, "Hillshade_Noon"] = 255

train.loc[train["Hillshade_3pm"] > 255, "Hillshade_3pm"] = 255
test.loc[test["Hillshade_3pm"] > 255, "Hillshade_3pm"] = 255


# In[32]:


feature_cols += ["new_f1", "new_f2", "Aspect2", "Highwater", "EVDtH", "EHDtH",  'EHiElv', 'EViElv', 'Hillshade_3pm_is_zero',
                 "Distanse_to_Hydrolody", "Hydro_Fire_1", "Hydro_Fire_2", "Hydro_Road_1", "Hydro_Road_2", "Fire_Road_1", "Fire_Road_2"]
cnt_cols += ["new_f1", "new_f2", "Aspect2", "EVDtH", "EHDtH", 'EHiElv', 'EViElv', 
                 "Distanse_to_Hydrolody", "Hydro_Fire_1", "Hydro_Fire_2", "Hydro_Road_1", "Hydro_Road_2", "Fire_Road_1", "Fire_Road_2"]
bin_cols += ["Highwater", 'Hillshade_3pm_is_zero']


# # Mutual Information

# In[33]:


get_ipython().run_cell_magic('time', '', 'x = train.iloc[:5000,:][feature_cols].copy()\ny = train.iloc[:5000,:][\'Cover_Type\'].copy()\nmi_scores = mutual_info_regression(x, y)\nmi_scores = pd.Series(mi_scores, name="MI Scores", index=x.columns)\nmi_scores = mi_scores.sort_values(ascending=False)\n')


# In[34]:


top = 10
plt.figure(figsize=(20,7))
fig = sns.barplot(x=mi_scores.values[:top], y=mi_scores.index[:top], palette="summer")
plt.title(f"Top {top} Strong Relationships Between Feature Columns and Target Column")
plt.xlabel("Relationship with Target")
plt.ylabel("Feature Columns")
plt.savefig("mi_scores.png")
plt.show()


# # Scale Data

# In[35]:


sc = StandardScaler()
train[cnt_cols] = sc.fit_transform(train[cnt_cols]).astype(np.float32)
test[cnt_cols] = sc.transform(test[cnt_cols]).astype(np.float32)


# # Neural Network Model

# ## Prepare Data

# In[36]:


cnt_cols += cluster_cols
cnt_cols += ["PC1", "PC2"]


# In[37]:


x_cnt = train[cnt_cols].values.astype(np.float32)
x_bin = train[bin_cols].values.astype(np.float32)
y  = train['Cover_Type'].values


# ## Define Model

# In[38]:


def get_model():
    AF = "selu"
    KI = "lecun_normal"
    input_1 = layers.Input(shape=(x_cnt.shape[-1]), name="continuous")
    x_1 = layers.Dense(128, activation=AF, kernel_initializer=KI)(input_1)
    x_1 = layers.BatchNormalization()(x_1)
    x_1 = layers.Dense(128, activation=AF, kernel_initializer=KI)(x_1)
    x_1 = layers.BatchNormalization()(x_1)
    x_1 = layers.Dense(128, activation=AF, kernel_initializer=KI)(x_1)
    x_1 = layers.BatchNormalization()(x_1)
    
    input_2 = layers.Input(shape=x_bin.shape[-1], name="categories")
    x_2 = layers.Dense(128, activation=AF, kernel_initializer=KI)(input_2)
    x_2 = layers.BatchNormalization()(x_2)
    x_2 = layers.Dense(128, activation=AF, kernel_initializer=KI)(x_2)
    x_2 = layers.BatchNormalization()(x_2)
    x_2 = layers.Dense(128, activation=AF, kernel_initializer=KI)(x_2)
    x_2 = layers.BatchNormalization()(x_2)


    x = layers.Concatenate()([x_1,x_2])
    x = layers.Dense(128, activation=AF, kernel_initializer=KI)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation=AF, kernel_initializer=KI)(x)
    x = layers.BatchNormalization()(x)
    output = layers.Dense(7, activation="softmax", name="output")(x)

    model = tf.keras.Model([input_1,input_2], output)
    return model

with strategy.scope():
    model = get_model()
    model.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])
    
tf.keras.utils.plot_model(model, show_shapes=True)


# ## WandB Log

# In[39]:


CONFIG = dict(competition="TPS Dec",  
              Notebook="TPS Dec 2021 - TensorFlow NN (TPU) and W&B", 
              Desc="Added Unsupervised Features - Added Pseudo")
run = wandb.init(project="TPS_Dec", name="log_unsupervised_pseudo", entity="kaveh", anonymous=anony, config=CONFIG)

wandb.config = {
  "learning_rate": 0.001,
  "epochs": 40,
  "batch_size": 1024,
}

wandb.log({"MI scores of features": wandb.Image("./mi_scores.png")})
wandb.log({"Model Architecture": wandb.Image("./model.png")})


# ## Train Model

# In[40]:


cb_wb = WandbCallback()
cb_es = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, mode="max", restore_best_weights=True, verbose=1)
cb_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=2, mode="max", min_lr=0.00005, verbose=1)

history = model.fit((x_cnt, x_bin), 
                    y, 
                    epochs=40, 
                    validation_split=0.2, 
                    batch_size=1024, 
                    validation_batch_size=1024,
                    callbacks=[cb_es, cb_lr, cb_wb])


# ## Plot Metrics

# In[41]:


plt.figure(figsize=(25,7))
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1,len(acc)+1) 
ax1 = plt.subplot(1,2,1)
ax1.plot(epochs, acc, 'r')
ax1.plot(epochs, val_acc, 'b')
ax1.set_xticks([i for i in epochs])
ax1.set_title('Training and validation Accuracy')
ax1.legend(["Training", "Validation" ])
ax1.set_xlabel("epochs")
ax1.set_ylabel("Accuracy")

ax2 = plt.subplot(1,2,2)
ax2.plot(epochs, loss, 'r')
ax2.plot(epochs, val_loss, 'b')
ax2.set_xticks([i for i in epochs])
ax2.legend(["Training", "Validation" ])
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Loss")
ax2.set_title('Training and validation loss')

plt.show()


# # Predict

# In[42]:


preds = model.predict((test[cnt_cols].values.astype(np.float32), test[bin_cols].values.astype(np.float32)))
p = np.argmax(preds, axis=1) + 1


# In[43]:


plt.figure(figsize=(10,5))
ax = sns.countplot(x=p)
plt.title("Predictions")
plt.xlabel("Cover Type")
ax.bar_label(ax.containers[0])
plt.savefig("predictions.png")
plt.show()


# In[44]:


wandb.log({"Predictions Stats": wandb.Image("./predictions.png")})
wandb.finish()


# # Submission

# In[45]:


sample_submission['Cover_Type'] = p
sample_submission.to_csv("submission.csv", index=False)
sample_submission.head()


# In[ ]:




