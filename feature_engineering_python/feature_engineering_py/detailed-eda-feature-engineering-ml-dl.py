#!/usr/bin/env python
# coding: utf-8

# # 1. Load Data

# In[1]:


get_ipython().system(' pip install --upgarde seaborn')
get_ipython().system(' pip install lightgbm')
get_ipython().system(' pip install catboost')


# In[2]:


from functools import partial
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PowerTransformer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import tensorflow as tf
from tensorflow import keras

np.random.seed(42)
tf.random.set_seed(42)


# In[3]:


try:
    train_df = pd.read_csv('/kaggle/input/playground-series-s3e24/train.csv').drop(columns=["id"])
    test_df = pd.read_csv('/kaggle/input/playground-series-s3e24/test.csv').drop(columns=["id"])
except:
    train_df = pd.read_csv('train.csv').drop(columns=["id"])
    test_df = pd.read_csv('test.csv').drop(columns=["id"])

train_df["smoking"] = train_df["smoking"].map({0: "No", 1: "Yes"})
palette = {"No": "#0079FF", "Yes": "#FF0060"}


# In[4]:


display(train_df.info())
print("-" * 50)
test_df.info()


# Observations:
# <ul>
# <li>The data does not have any missing values.</li>
# <li>All Columns are numerical values.</li>
# </ul>

# # 2. EDA & Feature Engineering
# 
# Create a `DataFrame` That contain all data to visualize and check for data drift

# In[5]:


df = pd.concat([train_df.assign(Source="Train"), test_df.assign(Source="Test")]).reset_index(drop=True)
print("Number of unique values per feature:")
for col in test_df:
    print("\t" f"* {col:<20}:{df[col].nunique()}")


# Let us check to see if the train and test data represent each other.

# In[6]:


fig, axes = plt.subplots(4, 7, figsize=(21, 12))
for ax, col in zip(axes.flatten(), test_df.columns):
    if col in ["hearing(right)", "hearing(left)", "Urine protein", "dental caries"]:
        sns.countplot(data=df, x=col, hue="Source", ax=ax)
    else:
        sns.kdeplot(data=df, x=col, hue="Source", ax=ax)
    ax.set(title=col)
plt.tight_layout()
plt.show()


# Observation:
# <ul>
# <li>No data Drift between the train & test sets.</li>
# <li><code>age, height(cm) & weight(kg)</code> are probabaly rounded to certain values.</li>
# <li>Many Features have high left skew.</li>
# <li><code>hearing, Urine protein & dental caries</code> are categorical features.</li>
# </ul>

# In[7]:


fig, axes = plt.subplots(4, 7, figsize=(21, 12))
for ax, col in zip(axes.flatten(), test_df.columns):
    if col in [
        "eyesight(left)",
        "eyesight(right)",
        "fasting blood sugar",
        "LDL",
        "serum creatinine",
        "AST",
        "ALT",
        "Gtp",
    ]:
        feature = np.log(df[col])
    else:
        feature = df[col]
    if col in ["hearing(right)", "hearing(left)", "Urine protein", "dental caries"]:
        sns.countplot(x=feature, hue=df["smoking"], ax=ax, palette=palette)
    else:
        sns.kdeplot(x=feature, hue=df["smoking"], ax=ax, palette=palette)
    ax.set(title=col)
plt.tight_layout()
plt.show()


# Observations:
# <ul>
# <li>People in their20s and 30s are most likely to be smokers.</li>
# <li>most of the subjects were at their early 40s</li>
# <li>People older than 40 years are less likely to be smoker, espically older ages (health concerns)</li>
# <li>Tall people are most likely to be smoker based on the data, but height is a genetic feature, not related to smoking.</li>
# <li>Smokers are heavier and had larger waste than non smokers (Maybe because smoker find it hard to exercise).</li>
# <li>Smoking reduces the HDL and increases the hemoglobin level, Gtp & ALT.</li>
# <li>Hearing & Urine Protien has asingle dominant value, and need extra analysis.</li>
# </ul>

# # 2.1 BMI & Body Waist
# 
# Body Mass Index (BMI) is a person’s weight in kilograms divided by the square of height in meters. A high BMI can indicate high body fatness. BMI screens for weight categories that may lead to health problems, but it does not diagnose the body fatness or health of an individual (ref: <a href="https://www.cdc.gov/healthyweight/assessing/bmi/index.html">Centers for Disease Control and Prevention</a>).
# This feature gives a better describtion of the subject's <code>height & weight</code>.

# In[8]:


df["BMI"] = df["weight(kg)"] / (df["height(cm)"] / 100) ** 2
_, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, col in zip(axes, ["height(cm)", "weight(kg)", "BMI", "waist(cm)"]):
    sns.kdeplot(data=df, x=col, hue="smoking", ax=ax, palette=palette)
plt.suptitle("Anthropometric Features")
plt.tight_layout()
plt.show()
_, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, col in zip(axes, ["height(cm)", "weight(kg)", "BMI", "waist(cm)"]):
    x = (df[col] - df[col].mean()) / df[col].std()
    sns.kdeplot(data=df, x=x, hue="smoking", ax=ax, palette=palette)
plt.suptitle("Normalized Anthropometric Features")
plt.tight_layout()
plt.show()
sns.scatterplot(x=df["BMI"], y=df["waist(cm)"] / 100, hue=df["smoking"], palette=palette)
plt.title("BMI vs Waist (m)")
plt.show()


# Observation:
# <ul>
# <li>People with less than 160 cm mostly don't smoke.</li>
# <li>Subject's with BMI less than 25 Kg/m2 (underweight & Normal) are less likely to smoke.</li>
# <li>Normalized BMI is normally distributed for cases x < 0.</li>
# <li><code>BMI and waist</code> are lineary correlated. There are some outliers in the corretion because the BMI/waist can mis-judge some subjects (each one has a limitation)</li>
# </ul>

# # 2.2 Eyesight

# In[9]:


fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for i, col in enumerate(df.filter(like="eye").columns):
    sns.kdeplot(data=df, x=col, hue="smoking", ax=axes[i], palette=palette)
    axes[i].set_title(col)
    x = (df[col] - df[col].mean()) / df[col].std()
    sns.kdeplot(data=df, x=x, hue="smoking", ax=axes[i + 2], palette=palette)
    axes[i + 2].set_title(f"Normalized {col}")
sns.scatterplot(
    data=df,
    x=df["eyesight(left)"],
    y=df["eyesight(right)"],
    hue="smoking",
    ax=axes[-1],
    palette=palette,
)
axes[-1].set_title("L vs R eyesights")
plt.tight_layout()
plt.show()


# Observation:
# <ul>
# <li>Eyesight data are distributed between 0 ~ 2.</li>
# <li>Some subjects had an outlier value of 10 in one of the eyes, except for one subject with value of 10 in both eyes.</li>
# <li>Eyesight is more of a categorical feature.</li>
# </ul>

# In[10]:


def categorical_plot(df: pd.DataFrame, x: pd.Series):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.countplot(
        x=x,
        hue=df["smoking"],
        ax=axes[0],
        palette=palette,
    )
    sns.histplot(
        x=x,
        hue=df["smoking"],
        stat="probability",
        multiple="fill",
        ax=axes[1],
        kde=True,
        palette=palette,
    )
    axes[0].set_title("Count")
    axes[1].set_title("Probability")
    plt.suptitle(x.name, fontsize=16)
    plt.show()
    return


categorical_plot(df, df["eyesight(left)"])
categorical_plot(df, df["eyesight(right)"])


# Observation:
# <ul>
# <li>When the eyesight increased, the smoker probability increases, espically for the left eye.</li>
# <li>After 1.5, the counts are less. Therefore, I will limit the eyesight value to 1.5.</li>
# </ul>

# In[11]:


df['clipped eyesight(left)'] = np.where(df['eyesight(left)']>1.5, 1.5, df['eyesight(left)'])
df['clipped eyesight(right)'] = np.where(df['eyesight(right)']>1.5, 1.5, df['eyesight(right)'])
categorical_plot(df, df['clipped eyesight(left)'])
categorical_plot(df, df['clipped eyesight(right)'])


# # 2.3 Hearing

# In[12]:


df["Equal hearing"] = df["hearing(left)"] == df["hearing(right)"]
categorical_plot(df, df["hearing(left)"])
categorical_plot(df, df["hearing(left)"])
categorical_plot(df, df["Equal hearing"])


# Obsevation:
# 
# <ul>
# <li>Hearing data had one dominanat category.</li>
# <li>The probability of being a smoker is relatively similar for both categories.</li>
# <li>Both values are equal in most of the cases, and even this equality check does not give a good feature.</li>
# <li style="color: red">Hearing feature does not provide good information.</li>
# </ul>
# 

# # 2.4 Blood Pressure

# `Systolic` & `Relaxation` measures the blood pressure, they can also be combined in a single feature called `Mean Arterial Pressure`.

# In[13]:


fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for i, col in enumerate(["systolic", "relaxation"]):
    sns.kdeplot(data=df, x=col, hue="smoking", ax=axes[i], palette=palette)
    axes[i].set_title(col)
    x = (df[col] - df[col].mean()) / df[col].std()
    sns.kdeplot(data=df, x=x, hue="smoking", ax=axes[i + 2], palette=palette)
    axes[i + 2].set_title(f"Normalized {col}")
sns.scatterplot(
    data=df,
    x=df["systolic"],
    y=df["relaxation"],
    hue="smoking",
    ax=axes[-1],
    palette=palette,
)
axes[-1].set_title("Systolic vs Relaxation")
plt.title("Blood Pressure Test Results")
plt.tight_layout()
plt.show()


# In[14]:


fig, axes = plt.subplots(1, 2, figsize=(10, 4))
df["MAP"] = 1/3 * df['systolic'] + 2/3 * df['relaxation']
x = (df["MAP"] - df["MAP"].mean()) / df["MAP"].std()
sns.kdeplot(data=df, x='MAP', hue="smoking", ax=axes[0], palette=palette)
sns.kdeplot(data=df, x=x, hue="smoking", ax=axes[1], palette=palette)
axes[0].set_title("Mean Arterial Pressure ")
axes[1].set_title("Normalized Mean Arterial Pressure")
plt.tight_layout()
plt.show()


# Observation:
# <ul>
# <li><code>Systolic & Relaxation</code> data are not normaly distributed.</li>
# <li><code>Mean Arterial Pressure</code> had better distribution compared to the main features.</li>
# </ul>

# # 2.5 Blood Glucose Level

# In[15]:


fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for i, col in enumerate(["fasting blood sugar", "hemoglobin"]):
    sns.kdeplot(data=df, x=col, hue="smoking", ax=axes[i], palette=palette)
    axes[i].set_title(col)
    x = (df[col] - df[col].mean()) / df[col].std()
    sns.kdeplot(data=df, x=x, hue="smoking", ax=axes[i + 2], palette=palette)
    axes[i + 2].set_title(f"Normalized {col}")
sns.scatterplot(
    data=df,
    x=df["hemoglobin"],
    y=df["fasting blood sugar"],
    hue="smoking",
    ax=axes[-1],
    palette=palette,
)
axes[-1].set_title("Hemoglobin vs Fasting Blood Sugar")
plt.tight_layout()
plt.show()


# Observation:
# 
# <ul>
# <li><code>Hemoglobin</code> > 15 gives high smoking probability.</li>
# <li><code>Fasting blood sugar</code> is highly skewed.</li>
# </ul>

# # 2.6 Lipid

# Cholesterol, Triglyceride, HDL Cholesterol and LDL Cholesterol are important measurments to quantify the possability of getting a chornic heart disease. Since smoking and heart disease are linked, those measurments might Provide useful information for smoker.
# 
# `LDL` had large outliers values, therefore I will scale the `LDL` feature in the  `sns.pairplot`

# In[16]:


lipid_df = train_df[["Cholesterol", "triglyceride", "HDL", "LDL", "smoking"]]
sns.pairplot(lipid_df, hue="smoking", diag_kind="kde", palette=palette)
plt.show()


# Actions to investigate:
# 
# <ul>
# <li>Clip the <code>LDL</code> at 200 (only 1% of the subjects has <code>LDL</code> less greater than 200.)</li>
# <li>Combine <code>clipped LDL & HDL</code> in a single feature.</li>
# </ul>
# 

# In[17]:


df["Clipped LDL"] = np.where(df["LDL"].abs() > 250, 250, df["LDL"])
df["LDL HDL Total"] = df["HDL"] + df["Clipped LDL"]
df["LDL HDL diff"] = df["HDL"] - df["Clipped LDL"]
# ldl_hdl = np.where(ldl_hdl.abs() > 250, 250, ldl_hdl)
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
sns.kdeplot(data=df, x="HDL", hue="smoking", ax=axes[0], palette=palette)
sns.kdeplot(data=df, x="Clipped LDL", hue="smoking", ax=axes[1], palette=palette)
sns.kdeplot(data=df, x="LDL HDL Total", hue="smoking", ax=axes[2], palette=palette)
sns.kdeplot(data=df, x="LDL HDL diff", hue="smoking", ax=axes[3], palette=palette)
plt.tight_layout()
plt.show()


# Observation:
# 
# <ul>
# <li><code>Cholesterol</code> is lineary correlated with <code>LDL & HDL</code>.</li>
# <li><code>Triglyceride & HDL</code> are showing good features to identify the smookers.</li>
# <li><code>LDL</code> is highly skewed.</li>
# <li><code>HDL + LDL</code> feature had better data distribution than <code>HDL</code> while maintaing a seperator between the labels.</li>
# <li><code>HDL - LDL</code> had a nice normal distribution for both labels.</li>
# </ul>

# # 2.7 Kidney

# In[18]:


categorical_plot(df, df["Urine protein"])


# In[19]:


# Urine Protien has a dominant category and can be treated as a binary feature
df["clipped Urine protein"] = np.where(df["Urine protein"]==1, 1, 2)
categorical_plot(df, df["clipped Urine protein"])


# In[20]:


fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.kdeplot(x=df["serum creatinine"], hue=df["smoking"], ax=axes[0], palette=palette)
axes[0].set_title("serum creatinine")
sns.kdeplot(
    x=np.log(df["serum creatinine"]),
    hue=df["smoking"],
    ax=axes[1],
    palette=palette,
    fill=True,
)
axes[1].set_title("Log serum creatinine")
transformer = PowerTransformer("box-cox")
df["transformed serum creatinine"] = transformer.fit_transform(df["serum creatinine"].values.reshape(-1, 1))
sns.kdeplot(
    x=df["transformed serum creatinine"],
    hue=df["smoking"],
    ax=axes[2],
    palette=palette,
    fill=True,
)
axes[2].set_title("box-cox serum creatinine")
plt.tight_layout()
plt.show()


# Observation:
# 
# <ul>
# <li><code>Urine protein</code> is a categorical feature, with a single dominant category.</li>
# <li>When treating <code>Urine protein</code> as a binary feature, it does not add much additional information.</li>
# <li style="color:red"><code>Urine protein</code> will not be used.</li>
# <li><code>Serum Creatinine</code> had outliers, but when applying the log transform, the categories are seperatable better than the power transform.</li>
# </ul>

# # 2.9 Liver Tests

# <code>De Ritis ratio</code> is a complex parameter that depends on AST and ALT activity in serum and physiological and pathological factors that determine their level (ref: <a href="https://jlpm.amegroups.org/article/view/7533/html#:~:text=De%20Ritis%20ratio%20is%20a,ratio%20has%20a%20heritability%20component.">link</a>)
# 

# In[21]:


df["De Ritis ratio"] = df["ALT"] / df["AST"]
sns.pairplot(
    data=df[["AST", "ALT", "Gtp", "De Ritis ratio", "smoking"]],
    hue="smoking",
    diag_kind="kde",
    palette=palette,
)
plt.show()


# In[22]:


kideny_df = df[["AST", "ALT", "Gtp", "De Ritis ratio"]].copy().apply(lambda x: np.log(x))
kideny_df["smoking"] = df["smoking"]
sns.pairplot(
    data=kideny_df,
    hue="smoking",
    diag_kind="kde",
    palette=palette,
)
plt.show()


# Observation:
# 
# <ul>
# <li>All three features are skewed.</li>
# <li>Log Transformer fix the skewness.</li>
# <li><code>GTP</code> distinguish the smoker better than the other two features, followed by <code>ALT</code>.</li>
# <li>Log trnsformer of the <code>De Ritis ratio</code> is normally distributted with zero mean.</li>
# </ul>

# # 2.10 Dental Caries

# In[23]:


categorical_plot(train_df, train_df["dental caries"])


# Observation:
# 
# <ul>
# <li>One will thought at first that smoker will not go to the <code>dental caries</code>, but data suggest the reverse.</li>
# <li>Most of the subjects are not going to the dentist, but the propotion of the subjects going to the dentist is not so low, therefore, this can be a useful feature.</li>
# </ul>

# # 3. Dataset Prepration

# In[24]:


try:
    train_df = pd.read_csv('/kaggle/input/playground-series-s3e24/train.csv').drop(columns=["id"])
    test_df = pd.read_csv('/kaggle/input/playground-series-s3e24/test.csv').drop(columns=["id"])
except:
    train_df = pd.read_csv('train.csv').drop(columns=["id"])
    test_df = pd.read_csv('test.csv').drop(columns=["id"])
df = pd.concat([train_df.assign(Source="Train"), test_df.assign(Source="Test")]).reset_index(drop=True)


# # 3.1 Features Engineering

# In[25]:


df["BMI"] = df["weight(kg)"] / (df["height(cm)"] / 100) ** 2
df["clipped eyesight(left)"] = np.where(df["eyesight(left)"] > 1.5, 1.5, df["eyesight(left)"])
df["clipped eyesight(right)"] = np.where(df["eyesight(right)"] > 1.5, 1.5, df["eyesight(right)"])
df["MAP"] = 1 / 3 * df["systolic"] + 2 / 3 * df["relaxation"]
df["De Ritis ratio"] = df["ALT"] / df["AST"]
df["Clipped LDL"] = np.where(df["LDL"].abs() > 250, 250, df["LDL"])
df["LDL HDL Total"] = df["HDL"] + df["Clipped LDL"]
df["LDL HDL diff"] = df["HDL"] - df["Clipped LDL"]


# # 3.2 Data Transfomration

# In[26]:


stdCols = [
    "age",
    "waist(cm)",
    "BMI",
    "systolic",
    "relaxation",
    "MAP",
    "hemoglobin",
    "Cholesterol",
    "HDL",
    "Clipped LDL",
    "LDL HDL Total",
    "LDL HDL diff",
    "clipped eyesight(left)",
    "clipped eyesight(right)",
]

logCols = [
    "fasting blood sugar",
    "triglyceride",
    "serum creatinine",
    "AST",
    "ALT",
    "Gtp",
    "De Ritis ratio",
]

catCols = [
    "dental caries",
]


# In[27]:


df[logCols] = df[logCols].apply(lambda x: np.log(x))
df[stdCols + logCols] = (df[stdCols + logCols] - df[stdCols + logCols].mean()) / df[stdCols + logCols].std()
X_train, X_val, y_train, y_val = train_test_split(
    df.loc[df["Source"] == "Train", stdCols + logCols + catCols].values,
    df.loc[df["Source"] == "Train", "smoking"].values,
    test_size=0.2,
    shuffle=True
)
X_test = df.loc[df["Source"] == "Test", stdCols + logCols + catCols].values


# # 4. Models

# In[28]:


def fit_eval(model, verbose=None):
    if verbose is None:
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train, verbose=verbose)
    y_train_pred = model.predict_proba(X_train)[:, 1].reshape((-1, 1))
    y_val_pred = model.predict_proba(X_val)[:, 1].reshape((-1, 1))
    y_test_pred = model.predict_proba(X_test)[:, 1].reshape((-1, 1))
    print(f"train AUC: {roc_auc_score(y_train, y_train_pred)}")
    print(f"val   AUC: {roc_auc_score(y_val, y_val_pred)}")
    return y_train_pred, y_val_pred, y_test_pred


def createDataset(X, y, isTrain=True):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.cache()
    if isTrain:
        dataset = dataset.shuffle(buffer_size=100000)
        dataset = dataset.batch(batch_size=32, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = dataset.batch(batch_size=2000000)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def plot_learning_curve(history):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    auc = history.history["accuracy"]
    val_auc = history.history["val_accuracy"]
    # Create a figure with two subplots
    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    # Plot the training and validation loss
    sns.lineplot(loss, label="Training Loss", color="#687EFF", ax=axes[0])
    sns.lineplot(val_loss, label="Validation Loss", color="#FF8080", ax=axes[0])
    axes[0].set_title("Training and Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    sns.lineplot(auc, label="Training AUC", color="#687EFF", ax=axes[1])
    sns.lineplot(val_auc, label="Validation AUC", color="#FF8080", ax=axes[1])
    axes[1].set_title("Training and Validation AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC")
    axes[1].legend()
    plt.suptitle(f"{history.model.name} Learning Curve")
    plt.tight_layout()
    plt.show()
    return


def train_eval_tf(model: keras.Model, train_dataset, val_dataset, X_test):
    keras.backend.clear_session()
    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Nadam(learning_rate=0.03),
        metrics=["accuracy", tf.keras.metrics.AUC()],
    )
    earlyStop = keras.callbacks.EarlyStopping(
        monitor="val_auc",
        patience=20,
        restore_best_weights=True,
        mode="max",
    )
    reduceLR = keras.callbacks.ReduceLROnPlateau(
        monitor="val_auc", mode="max", factor=0.5, patience=8, min_lr=1e-7
    )
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=1000,
        callbacks=[earlyStop, reduceLR],
        verbose=0,
    )
    print(f"train AUC: {model.evaluate(train_dataset, verbose=0)[-1]}")
    print(f"val   AUC: {model.evaluate(val_dataset, verbose=0)[-1]}")
    plot_learning_curve(history)
    tf_predict = partial(model.predict, verbose=0)
    y_train_pred = tf_predict(X_train)
    y_val_pred = tf_predict(X_val)
    y_test_pred = tf_predict(X_test)
    return y_train_pred, y_val_pred, y_test_pred


# # 4.1 XGBOOST

# In[29]:


get_ipython().run_cell_magic('time', '', 'xg_model = XGBClassifier(\n    n_estimators=10000,\n    learning_rate=0.01752354328845971,\n    booster="gbtree",\n    reg_lambda= 0.08159630121074074,\n    reg_alpha=0.07564858712175693,\n    subsample=0.5065979400270813,\n    colsample_bytree=0.6187340851873067,\n    max_depth=4,\n    min_child_weight=5,\n    eta=0.2603059902806757,\n    gamma=0.6567360773618207,\n    tree_method="hist",\n)\ny_train_xg, y_val_xg, y_test_xg = fit_eval(xg_model)\n')


# # 4.2 LGBMClassifier

# In[30]:


get_ipython().run_cell_magic('time', '', 'lgbm = LGBMClassifier(\n    objective="binary",\n    metric="auc",\n    boosting_type="dart",\n    n_estimators=1000,\n    max_depth=7,\n    learning_rate=0.03,\n    num_leaves=50,\n    reg_alpha=3,\n    reg_lambda=3,\n    subsample=0.7,\n    colsample_bytree=0.7,\n    device_type="gpu",\n    verbosity=0,\n)\ny_train_lgbm, y_val_lgbm, y_test_lgbm = fit_eval(lgbm)\n')


# # 4.3 CatBoost

# In[31]:


get_ipython().run_cell_magic('time', '', 'cat_boost = CatBoostClassifier(\n    iterations=10000,\n    eval_metric="AUC",\n    loss_function="Logloss",\n    auto_class_weights="Balanced",\n    devices=\'gpu\',\n    \n)\ny_train_cat_boost, y_val_cat_boost, y_test_cat_boost = fit_eval(cat_boost, verbose=0)\n')


# # 4.4 Shallow model

# In[32]:


train_dataset = createDataset(X_train, y_train)
val_dataset = createDataset(X_val, y_val, False)
denseLayer = partial(
    keras.layers.Dense,
    units=32,
    activation="selu",
    kernel_initializer=keras.initializers.LecunNormal(seed=42)
)
hidden_layers = 10
keras.backend.clear_session()

input_ = keras.Input(shape=(X_train.shape[-1],))
X = input_
for _ in range(hidden_layers):
    X = denseLayer()(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.Dropout(0.3)(X)
X = denseLayer(units=1, activation="sigmoid", name="output")(X)
nn_model = keras.Model(inputs=input_, outputs=X, name="NN")
nn_model.summary()


# In[33]:


get_ipython().run_cell_magic('time', '', 'y_train_nn, y_val_nn, y_test_nn = train_eval_tf(nn_model, train_dataset, val_dataset, X_test)\n')


# # 4.7 Blending

# In[34]:


X_blend_train = np.column_stack(
    (
        y_train_xg,
        y_train_lgbm,
        y_train_cat_boost,
        y_train_nn
    )
)
X_blend_val = np.column_stack(
    (
        y_val_xg,
        y_val_lgbm,
        y_val_cat_boost,
        y_val_nn
    )
)
blend_model = LogisticRegression()
blend_model.fit(X_blend_val, y_val)
y_train_pred = blend_model.predict_proba(X_blend_train)[:, 1].reshape((-1, 1))
y_val_pred = blend_model.predict_proba(X_blend_val)[:, 1].reshape((-1, 1))
print(f"train AUC: {roc_auc_score(y_train, y_train_pred)}")
print(f"val   AUC: {roc_auc_score(y_val, y_val_pred)}")


# # 5. Submission

# In[35]:


X_blend_test = np.column_stack(
    (
        y_test_xg,
        y_test_lgbm,
        y_test_cat_boost,
        y_test_nn
    )
)
y_test = blend_model.predict_proba(X_blend_test)[:, 1].reshape((-1, 1))
try:
    submission = pd.read_csv("/kaggle/input/playground-series-s3e24/sample_submission.csv")
    submission["smoking"] = y_test.mean(axis=1)
    submission.to_csv("/kaggle/working/submission.csv", index=False)
except:
    submission = pd.read_csv("sample_submission.csv")
    submission["smoking"] = y_test.mean(axis=1)
    submission.to_csv("submission.csv", index=False)

