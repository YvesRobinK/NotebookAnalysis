#!/usr/bin/env python
# coding: utf-8

# ![petfinder_baner](http://www.mf-data-science.fr/images/projects/petfinder_baner.jpg)

# In this [competition](https://www.kaggle.com/c/petfinder-pawpularity-score), we’ll analyze raw images and metadata to predict the **“Pawpularity” of pet photos**. We'll train and test your model on PetFinder.my's thousands of pet profiles.
# 
# In this first Notebook, we will perform a quick exploratory analysis of the data and try to define new variables by feature engineering.
# 
# <span style="color:red">**Of course, if the Notebook helps you, don't hesitate to upvote!**</span>

# # <span style="color: #186fb4; font-variant:small-caps;" id="sommaire">Summary</span>
# 
# 1. [Exploratory Data Analysis](#section_1)     
# 2. [Feature Engineering](#section_2)
# 2. [Are these features really important ?](#section_3)      
# 3. [Modeling on Dataset images by Transfert Learning](#section_4)      
# 4. [Competition submission on Test set](#section_5)

# In[1]:


# Load Python libraries
import numpy as np
import pandas as pd
import os
from tqdm.notebook import tqdm
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import kstest
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from skimage.transform import resize
import colorsys
from sklearn.cluster import KMeans
from collections import Counter
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense

import warnings
warnings.filterwarnings("ignore")


# # <span style="color: #186fb4" id="section_1">Exploratory Data Analysis</span>
# 
# The data architecture is made up of 2 files: test and train each containing images of pets. 3 CSV files are also available. We will first look at the structure of these files.
# 
# ## <span style="color: #e7273e" id="section_1_1">CSV files</span>

# In[2]:


PATH = "../input/petfinder-pawpularity-score/"
train_df = pd.read_csv("".join([PATH,"train.csv"]))
test_df = pd.read_csv("".join([PATH,"test.csv"]))
submission_df = pd.read_csv("".join([PATH,"sample_submission.csv"]))


# ### Train file

# In[3]:


train_df.head()


# In[4]:


print(f"Number of images in Train set : {train_df.shape[0]}")


# In[5]:


train_df.info()


# In the train.csv file, there is an image identification variable: **Id**.
# Then, **12 binary variables** give us a quick description of the image *(Action, Blur, Human ...)*.
# Finally, the Pawpularity variable is our **target variable *(y)*** which gives the popularity of the image on a score between 0 and 100. We also note that **there is no missing data** in this dataset
# 
# ### Test file

# In[6]:


test_df.head()


# In[7]:


print(f"Number of images in Test set : {test_df.shape[0]}")


# The test file has the same fields, excluding the predictable variable `Pawpularity`. The **test set only has 8 images** against 9,912 images for training.
# 
# ### Submission file

# In[8]:


submission_df.head()


# For the submission file, we will need to provide the ID of the tested image as well as the popularity prediction between 0 and 100.
# 
# ## <span style="color: #e7273e" id="section_1_2">Pawpularity distribution</span>
# 
# Let us now look at the distribution of the variable to be predicted in the train set

# In[9]:


sns.set(rc={'figure.figsize':(14,9)})


# In[10]:


fig = plt.figure()
sns.histplot(data=train_df, x='Pawpularity', kde=True)
plt.axvline(train_df['Pawpularity'].mean(), c='orange', ls='-', lw=3, label="Mean Pawpularity")
plt.title('Pawpularity score Histogram', fontsize=20, fontweight='bold')
plt.legend()
plt.show()


# Note that the distribution of the variable $y$ is centered on the scores between 20 and 30. We will **check the normality** of the distribution with a quantile - quantile diagram.

# In[11]:


fig = plt.figure()
qqplot(train_df['Pawpularity'], line='s')
plt.title('Quantile-Quantile plot of Pawpularity distribution', 
          fontsize=20, fontweight='bold')
plt.show()


# We notice the deviation at this QQPlot which seems to indicate a non-Gaussian distribution. We will check with the **Kolmogorov-Smirnov test** *(Shapiro-Wilks is not suitable for a dataset greater than 5000 items)*.

# In[12]:


# Kolmogorov-Smirnov test with Scipy
stat, p = kstest(train_df['Pawpularity'],'norm')
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
    print(f'Sample looks Gaussian (fail to reject H0 at {int(alpha*100)}% test level)')
else:
    print(f'Sample does not look Gaussian (reject H0 at {int(alpha*100)}% test level)')


# The test clearly indicates that the distribution does not follow a Gaussian law. **It will therefore be important to normalize the data according to the modeling chosen**.
# 
# 
# ## <span style="color: #e7273e" id="section_1_3">Distribution of the predictor variables</span>
# We will now take a quick look at the distribution of the predictor variables.

# In[13]:


predictor = train_df.columns[1:-1]

fig = plt.figure(figsize=(25,20))
for i, x in enumerate(predictor):
    ax = plt.subplot(3,4,i+1)
    sns.countplot(data=train_df, x=x, ax=ax)
    ax.set_xlabel(None)
    ax.set_title(x, fontweight='bold', color="#e7273e")

plt.suptitle("Predictor distribution", y=0.93,
             fontsize=20, fontweight='bold')
plt.show()  


# The distribution of the predictor variables shows clear differences that are difficult to interpret for the moment.
# 
# ## <span style="color: #e7273e" id="section_1_4">Correlations between predictor variables</span>
# 
# We will see if there are marked correlations between our different predictor variables by calculating the **correlation matrix**.

# In[14]:


corr_matrix = train_df[predictor].corr()
fig = plt.figure()
sns.set_theme(style="white")
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr_matrix, annot=True, fmt='.1g', cmap=cmap, 
            mask=mask, square=True)
plt.title('Correlation Matrix', fontsize=20, fontweight='bold')
plt.show()


# 2 main correlations stand out *(> 0.5)*:
# - The first between **Occlusion and Human** *(Humans can hide part of the animal)*
# - The second between **Face and Eyes** which this time may seem logical.
# 
# We also need to check if there is too much **multicollinearity** that could degrade the performance of our models. For this we will use **VIF** : 
# 
# $$ \large VIF = \frac{1}{1- R^2}$$
# 
# Where, $R^2$ is the coefficient of determination in linear regression. Its value lies between 0 and 1.

# In[15]:


# VIF dataframe
vif_data = pd.DataFrame()
X = train_df[predictor]
vif_data["feature"] = X.columns
  
# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                          for i in range(len(X.columns))]  
vif_data = vif_data.sort_values("VIF", ascending=False)
vif_data


# As we can see, Face and Eyes have very high values of VIF, indicating that these two variables are highly correlated. Hence, considering these two features together leads to a model with high multicollinearity. **We will therefore use only one of these 2 variables for the modelizations**. 
# 
# We remove the variable that has the highest VIF.

# In[16]:


X.drop("Face", axis=1, inplace=True)
X.columns


# ## <span style="color: #e7273e" id="section_1_5">Correlations between predictor variables and Pawpularity</span>
# 
# We will now check whether there are strong linear correlations *(Pearson)* between the predictor variables and the variable to be predicted *(Pawpularity)*.

# In[17]:


print("-"*80)
print("Pearson correlation with Pawpularity (y)")
print("-"*80)
for x in X.columns:
    corr_y = round(np.corrcoef(train_df[x], train_df["Pawpularity"])[0,1],4)
    print(f"Pawpularity / {x}: {corr_y}")
print("-"*80)


# It can therefore be seen that **there is no linear correlation between the variable to be predicted and the predictive variables**.
# 
# ## <span style="color: #e7273e" id="section_1_6">Viewing sample training images</span>
# 
# Let's start by viewing a few random images

# In[18]:


fig, ax = plt.subplots(2,3,figsize=(15,9))
fig.patch.set_facecolor('#343434')

for i, a in zip(train_df[['Id', 'Pawpularity']].sample(6).iterrows(), ax.ravel()):
    a.set(xticks=[], yticks=[])
    img = plt.imread(PATH + "train/" + i[1][0] + ".jpg")
    a.imshow(img)
    a.set_title(f'Id: {i[0]}, Pawpularity Score: {i[1][1]}', color="white")

fig.suptitle('Pawpularity Images', fontsize=20, fontweight='bold', color="#e7273e")
fig.tight_layout()
fig.show()


# At the moment, it seems difficult to understand the difference between images that have a high popularity and those that do not win many votes. We also see that the orientations and sizes of images are different. 
# 
# We will now look at the **differences between the predictor variables at 0 or 1**.

# In[19]:


fig, ax = plt.subplots(11, 2, figsize=(14,50))
fig.patch.set_facecolor('#343434')

for a in ax.ravel():
    a.set(xticks=[], yticks=[])

for r in range(11):
    label = X.columns[r]
    for i in [0, 1]:
        img_id = train_df[train_df[label] == i].sample()['Id'].values[0]
        img = plt.imread(PATH + f"/train/{img_id}.jpg")
        ax[r, i].imshow(img)
        ax[r, i].set_title(f'{label}={i}', color="white")

fig.tight_layout()
fig.show()


# Finally, we will project the **top 3 most popular and least popular images** to see if the difference is marked and humanly understandable.

# In[20]:


top = train_df[train_df['Pawpularity'] == 100]['Id']

fig, ax = plt.subplots(1,3)
fig.patch.set_facecolor('#343434')

for i, ax in zip(top.sample(3), ax.ravel()):
    ax.set(xticks=[], yticks=[])
    img = plt.imread(PATH + f"/train/{i}.jpg")
    ax.imshow(img)
    
fig.suptitle('Most Pawpular Images', fontsize=20, fontweight='bold', color='#e7273e', y=0.95)
fig.tight_layout()
fig.show()


# In[21]:


bottom = train_df[train_df['Pawpularity'] == 1]['Id']

fig, ax = plt.subplots(1,3)
fig.patch.set_facecolor('#343434')

for i, ax in zip(bottom.sample(3), ax.ravel()):
    ax.set(xticks=[], yticks=[])
    img = plt.imread(PATH + f"/train/{i}.jpg")
    ax.imshow(img)
    
fig.suptitle('Least Pawpular Images', fontsize=20, fontweight='bold', color='#e7273e', y=0.95)
fig.tight_layout()
fig.show()


# It's hard to explain when you see them what really differentiates the most popular images from the less popular ones ...
# 
# # <span style="color: #186fb4" id="section_2">Feature Engineering</span>
# 
# Now we are going to create new features.
# 
# ## <span style="color: #e7273e" id="section_2_1">Extract dominant color of each image with KMeans</span>
# 
# Some photography experts agree that **the dominant color of an image can unconsciously affect its popularity** *(as well as overall exposure for that matter)*. We are therefore going to create a variable that will store the dominant color of each image. 
# 
# To do this, we will use **clustering methods on the RGB layers** of our jpg files to extract the dominant color in HLS *(Hue Lightness Saturation)* format. This format will allow us to recover in a single formula the information on the **hue, saturation and luminance of the dominant color of each image**.

# In[22]:


def get_dominant_color(image_path, k=4, image_processing_size = None):
    """
    takes an image as input
    returns the dominant color of the image as a list
    
    dominant color is found by running k means on the 
    pixels & returning the centroid of the largest cluster

    processing time is speed up by working with a smaller image; 
    this resizing can be done with the image_processing_size param 
    which takes a tuple of image dims as input
    """
    
    image = plt.imread(image_path)
    #resize image if new dims provided
    if image_processing_size is not None:
        image = cv2.resize(image, image_processing_size, 
                            interpolation = cv2.INTER_AREA)
    
    #reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    #cluster and assign labels to the pixels 
    clt = KMeans(n_clusters = k)
    labels = clt.fit_predict(image)

    #count labels to find most popular
    label_counts = Counter(labels)

    #subset out most popular centroid
    dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]
    dominant_color = list(dominant_color)
    r = int(dominant_color[0])
    g = int(dominant_color[1])
    b = int(dominant_color[2])
    
    #Convert to HLS color space
    dominant_hls = colorsys.rgb_to_hls(r, g, b)

    return list(dominant_hls)


# Let's look at the effect of this function on a test image :

# In[23]:


TRAIN_PATH = "../input/petfinder-pawpularity-score/train/"
TEST_PATH = "../input/petfinder-pawpularity-score/test/"


# In[24]:


sample_img = TRAIN_PATH+"0095f81bab3b68a4f70e99f0fcec7b06.jpg"
sample_hls = get_dominant_color(sample_img, k=3, image_processing_size = (50, 50))
sample_dom_color = colorsys.hls_to_rgb(sample_hls[0],
                                       sample_hls[1],
                                       sample_hls[2])
sample_dom_color = "#{:02x}{:02x}{:02x}".format(int(sample_dom_color[0]),
                                                int(sample_dom_color[1]),
                                                int(sample_dom_color[2]))
print("Dominant HLS : ", sample_hls)
print("Dominant Color Hex : ", sample_dom_color)

fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(121)
ax = plt.imshow(plt.imread(sample_img))
ax2 = fig.add_subplot(122)
rect1 = matplotlib.patches.Rectangle((0,0), 10, 5,color=sample_dom_color)
ax2.add_patch(rect1)
plt.axis('off')
plt.suptitle('Dominant color of sample image', fontsize=20, fontweight='bold', y=0.98)
fig.tight_layout()
plt.show()


# We are now going to **apply this function to all the datasets** :

# In[25]:


tqdm.pandas()
train_df["Dominant_color_hls"] = train_df["Id"].progress_apply(
    lambda x : get_dominant_color(
        TRAIN_PATH+x+".jpg", 
        k=3, 
        image_processing_size = (50, 50)))


# In[26]:


HLS_train_df = train_df["Dominant_color_hls"].apply(pd.Series)
HLS_train_df = HLS_train_df.rename(columns={0:"H",1:"L",2:"S"})
train_df = pd.concat([train_df, HLS_train_df], axis=1)
train_df.drop("Dominant_color_hls", axis=1, inplace=True)
train_df.head()


# In[27]:


fig = plt.figure(figsize=(20,6))
ax1 = fig.add_subplot(131)
sns.histplot(train_df["H"], ax=ax1)
ax1.set_title("Hue", fontsize=17, color="#186fb4")
ax2 = fig.add_subplot(132)
sns.histplot(train_df["L"], ax=ax2)
ax2.set_title("Luminance", fontsize=17, color="#186fb4")
ax3 = fig.add_subplot(133)
sns.histplot(train_df["S"], ax=ax3)
ax3.set_title("Saturation", fontsize=17, color="#186fb4")
plt.suptitle('Dominant HLS color of train images', 
             fontsize=20, fontweight='bold', y=0.98)
fig.tight_layout()
plt.show()


# In[28]:


test_df["Dominant_color_hls"] = test_df["Id"].progress_apply(
    lambda x : get_dominant_color(
        TEST_PATH+x+".jpg", 
        k=3, 
        image_processing_size = (50, 50)))


# In[29]:


HLS_test_df = test_df["Dominant_color_hls"].apply(pd.Series)
HLS_test_df = HLS_test_df.rename(columns={0:"H",1:"L",2:"S"})
test_df = pd.concat([test_df, HLS_test_df], axis=1)
test_df.drop("Dominant_color_hls", axis=1, inplace=True)
test_df.head()


# ## <span style="color: #e7273e" id="section_2_2">Original image resolution</span>
# For our image processing algorithms, we will have to perform resize to obtain *input_shape* conforming to what the models expect. We are therefore going to **save the initial size of the image in a variable** *(which could have an impact on the popularity of the photo)*.

# In[30]:


def get_img_size(path):
    width = []
    height = []
    landscape = []
    for image_path in tqdm(os.listdir(path)):
        image = plt.imread(path+image_path)
        width.append(image.shape[1])
        height.append(image.shape[0])
        if(image.shape[1] > image.shape[0]):
            landscape_img = 1
        else:
            landscape_img = 0
        landscape.append(landscape_img)
    return width, height, landscape


# In[31]:


train_width, train_height, train_landscape = get_img_size(TRAIN_PATH)


# In[32]:


train_df["width"] = train_width
train_df["height"] = train_height
train_df["landscape"] = train_landscape
train_df.head()


# In[33]:


test_width, test_height, test_landscape = get_img_size(TEST_PATH)


# In[34]:


test_df["width"] = test_width
test_df["height"] = test_height
test_df["landscape"] = test_landscape
test_df.head()


# We have now stored all the variables that we will need for our models. We will be able to create the final datasets. It will also be necessary to **standardize the data which are now on different scales**. We will use standardization thanks to StandardScaler from ScikitLearn.

# ## <span style="color: #e7273e" id="section_1_7">Define final Dataset for training</span>
# 
# For the variable to be predicted *(y = Pawpularity)*, **we are going to reduce its value between 0 and 1** so that they are more understandable for the models.

# In[35]:


ids = train_df[["Id"]].values
y = np.ravel(train_df[["Pawpularity"]]/100)
X = train_df.drop(["Id", "Pawpularity"], axis=1)
X_test = test_df.drop("Id", axis=1)


# In[36]:


# Normalization
encoder = MinMaxScaler()
encoder.fit(X)
X_scaled = encoder.transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

X_test_scaled = encoder.transform(X_test)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)


# # <span style="color: #186fb4" id="section_3">Are these features really important ?</span>
# 
# To check if the features at our disposal are really important, we will first model our data with **RandomForest and look at the importance of the features**.
# 
# ## <span style="color: #e7273e" id="section_2_1">RandomForest on binary features</span>
# We will also apply a **GridSearchCV to find the best hyperparameters**.

# In[37]:


X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.3, random_state=42)

print("-"*80)
print("Train and test split sizes")
print("-"*80)
print(f"X_train : {X_train.shape}")
print(f"X_test : {X_valid.shape}")
print(f"y_train : {y_train.shape[0]}")
print(f"y_test : {y_valid.shape[0]}")
print("-"*80)


# In[38]:


rfr = RandomForestRegressor(random_state=8)
param_grid = {
            "n_estimators" : [10,50,100],
            "max_features" : ["log2", "sqrt"],
            "max_depth"    : [5,15,25],
            "bootstrap"    : [True, False]
        }

grid_rfr = GridSearchCV(
    rfr,
    param_grid,
    cv = 5,
    verbose=1,
    n_jobs=-1)

best_rfr = grid_rfr.fit(X_train, y_train)


# After training the GrisSearchCV, we can extract the **best parameters** from the model:

# In[39]:


print("-"*80)
print("Best parameters for Random Forest model")
print("-"*80)
print(best_rfr.best_params_)
print("-"*80)


# Then, we will plot the **importance of features** in the modeling:

# In[40]:


importances = best_rfr.best_estimator_.feature_importances_

feature_names = X_train.columns
forest_importances = pd.DataFrame(importances, columns=["FI"], index=feature_names)
forest_importances = forest_importances.sort_values("FI", ascending=False)

fig, ax = plt.subplots()
sns.barplot(data=forest_importances, x = "FI", 
            y=forest_importances.index, ax=ax, 
            palette="Blues_d")
ax.set_title("Feature importances of RandomForestRegressor", 
             fontsize=20, fontweight='bold')
ax.set_xlabel("Mean decrease in impurity")
ax.set_ylabel("Features")
fig.tight_layout()


# We notice that the **Accessory, Near, Group and Info** variables have a greater importance in the decisions, without being in a very different order of magnitude.
# 
# Now we will **perform the predictions with this model on the validation set** to check the performance and distribution of the predicted values compared to the actual values.

# In[41]:


rfr_pred = best_rfr.predict(X_valid)


# In[42]:


fig = plt.figure(figsize=(12,8))
plt.scatter(x=rfr_pred, y=y_valid)
plt.ylabel("Pawpularity real values (y_valid)")
plt.xlabel("Predicted values (rfr_pred)")
plt.title("Predicted Pawpularity VS True values with RandomForest", 
          fontsize=20, fontweight='bold')
plt.show()


# We can see perfectly here that **the only modelization with the variables of the CSV file does not make it possible to obtain satisfactory performances**. RandomForest is very uncertain in its predictions which are concentrated on the 0.35 / 0.41 area. We will have to try another approach.
# 
# # <span style="color: #186fb4" id="section_4">Modeling on Dataset images by Transfert Learning</span>
# 
# Here we will use only the image data and perform transfer learning modeling with an **Xception model trained on ImageNet**.

# In[43]:


# Laod Keras application Xception
xcept_model = tf.keras.applications.Xception(
    include_top=False,
    weights=None,
    input_shape=(299,299,3),
    pooling="avg"
)

# Load ImageNet weights pre-saved
xcept_model.load_weights(
    '../input/resnet-imagenet-weights/xception_imagenet_weights.h5')

# Non trainable
xcept_model.trainable = False


# For better use in Tensorflow / Keras, we will **create generators by slightly modifying our DataFrame Pandas**. We will indeed add the name *(and extension)* of the image files to our y DataSets.

# In[44]:


k_df = train_df[["Id","Pawpularity"]]
k_df["Image"] = k_df["Id"].apply(lambda x: x+".jpg")
k_df["Pawpularity"] = k_df["Pawpularity"]/100
k_df.head()


# In[45]:


k_X_train, k_X_valid, k_y_train, k_y_valid = train_test_split(
    k_df["Image"], k_df["Pawpularity"], test_size=0.3, random_state=42)

print("-"*80)
print("Train and test split sizes")
print("-"*80)
print(f"X_train : {k_X_train.shape}")
print(f"X_test : {k_X_valid.shape}")
print(f"y_train : {k_y_train.shape[0]}")
print(f"y_test : {k_y_valid.shape[0]}")
print("-"*80)


# In[46]:


k_train_df = pd.DataFrame(k_X_train, columns=["Image"])
k_train_df["Pawpularity"] = k_y_train
k_valid_df = pd.DataFrame(k_X_valid, columns=["Image"])
k_valid_df["Pawpularity"] = k_y_valid


# In[47]:


datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.xception.preprocess_input,
    validation_split=0.2)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.xception.preprocess_input)


# In[48]:


train_generator = datagen.flow_from_dataframe(
    dataframe=k_train_df,
    directory=PATH+"train/",
    x_col="Image",
    y_col="Pawpularity",
    subset="training",
    target_size=(299,299),
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="raw")

valid_generator = datagen.flow_from_dataframe(
    dataframe=k_train_df,
    directory=PATH+"train/",
    x_col="Image",
    y_col="Pawpularity",
    subset="validation",
    target_size=(299,299),
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="raw")

test_generator = datagen.flow_from_dataframe(
    dataframe=k_valid_df,
    directory=PATH+"train/",
    x_col="Image",
    y_col="Pawpularity",
    target_size=(299,299),
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="raw")


# In[49]:


# Add new fully-connected layers
base_output = xcept_model.output
base_output = Dense(128, activation='relu')(base_output)
base_output = Dropout(0.2)(base_output)
# Output : new classifier
predictions = Dense(1, activation='linear')(base_output)

# Define new model
my_xcept_model = Model(inputs=xcept_model.input,
                       outputs=predictions)
my_xcept_model.compile(optimizer="adam",
                       loss=tf.keras.metrics.mean_squared_error)


# In[50]:


STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size

# Early Stopping to prevent overfitting
early_stopper = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", 
    patience=15, 
    verbose=2, 
    restore_best_weights=True)

history_xcept = my_xcept_model.fit(
    train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=valid_generator,
    validation_steps=STEP_SIZE_VALID,
    epochs=50,
    verbose=2,
    callbacks=[early_stopper])


# In[51]:


fig = plt.figure(figsize=(12, 7))
plt.plot(history_xcept.history["loss"],
         color="#186fb4", linestyle="-.",
         label="Train")
plt.plot(history_xcept.history["val_loss"],
         color="#186fb4",
         label="Validation")
plt.legend()
plt.title("RMSE metric of Xception model for Pawpularity", 
          fontsize=20, fontweight='bold')
plt.show()


# We see on the results plot that the RMSE metric follows a beautiful desendent curve in training but **struggles to drop in validation**. We will, as for RandomForest, carry out the predictions on the validation set.

# In[52]:


xcept_pred = my_xcept_model.predict(test_generator)
xcept_pred.shape


# In[53]:


fig = plt.figure(figsize=(12,8))
plt.scatter(x=xcept_pred, y=k_y_valid)
plt.ylabel("Pawpularity real values (k_y_valid)")
plt.xlabel("Predicted values (xcept_pred)")
plt.title("Predicted Pawpularity VS True values with Xception", 
          fontsize=20, fontweight='bold')
plt.show()


# This time again, **the predictions are very disparate and do not reflect the real values**. The processing of the images only part is therefore not a good solution for this competition *(Xception generally obtaining good results on image processing)*. 

# # <span style="color: #186fb4" id="section_5">Transfert Learning optimization</span>
# 
# Now, we are going to do a preprocessing of our images to try to improve the algorithm.

# In[54]:


datagen_v2 = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10, # rotation
    width_shift_range=0.2, # horizontal shift
    height_shift_range=0.2, # vertical shift
    zoom_range=0.2, # zoom
    horizontal_flip=True, # horizontal flip
    preprocessing_function=tf.keras.applications.xception.preprocess_input,
    validation_split=0.2)


# In[55]:


train_generator_v2 = datagen_v2.flow_from_dataframe(
    dataframe=k_train_df,
    directory=PATH+"train/",
    x_col="Image",
    y_col="Pawpularity",
    subset="training",
    target_size=(299,299),
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="raw")

valid_generator_v2 = datagen_v2.flow_from_dataframe(
    dataframe=k_train_df,
    directory=PATH+"train/",
    x_col="Image",
    y_col="Pawpularity",
    subset="validation",
    target_size=(299,299),
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="raw")


# In[56]:


STEP_SIZE_TRAIN = train_generator_v2.n//train_generator_v2.batch_size
STEP_SIZE_VALID = valid_generator_v2.n//valid_generator_v2.batch_size


# In[57]:


tf.keras.backend.clear_session()
history_xcept_v2 = my_xcept_model.fit(
    train_generator_v2,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=valid_generator_v2,
    validation_steps=STEP_SIZE_VALID,
    epochs=50,
    verbose=2,
    callbacks=[early_stopper])


# In[58]:


fig = plt.figure(figsize=(12, 7))
plt.plot(history_xcept_v2.history["loss"],
         color="#186fb4", linestyle="-.",
         label="Train")
plt.plot(history_xcept_v2.history["val_loss"],
         color="#186fb4",
         label="Validation")
plt.legend()
plt.title("RMSE metric of Xception augmented model for Pawpularity", 
          fontsize=20, fontweight='bold')
plt.show()


# In[59]:


xcept_pred_v2 = history_xcept_v2.model.predict(test_generator)
xcept_pred_v2.shape


# In[60]:


fig = plt.figure(figsize=(12,8))
plt.scatter(x=xcept_pred_v2, y=k_y_valid)
plt.ylabel("Pawpularity real values (k_y_valid)")
plt.xlabel("Predicted values (xcept_pred)")
plt.title("Predicted Pawpularity VS True values with Xception", 
          fontsize=20, fontweight='bold')
plt.show()


# The augmented model looks a bit better but still fails to predict popularity scores reliably enough.
# 
# # <span style="color: #186fb4" id="section_6">Hybrid approach with feature detection and RandomForest</span>
# 
# We are therefore going to **test a last hybrid approach** consisting in carrying out the feature detection with Xception, then in coupling the results with the database of image characteristics to finally predict $y$ with a RandomForestRegressor.

# In[61]:


def feature_detect_img(folder, img_size=299):
    listVectors = []
    for img in tqdm(os.listdir(PATH+folder+"/")):
        image = plt.imread(PATH+folder+"/"+img)
        #resize image if new dims provided
        image = cv2.resize(image, (img_size,img_size),
                           interpolation = cv2.INTER_AREA)
        image = np.expand_dims(image, axis=0)
        image = tf.keras.applications.xception.preprocess_input(image)
        
        img_vector = xcept_model.predict(image)
        listVectors.append(np.array(img_vector))
    
    return listVectors


# In[62]:


train_vectors_fd = feature_detect_img("train", img_size=299)


# In[63]:


train_vectors_fd = np.array(train_vectors_fd)
train_vectors_fd = np.squeeze(train_vectors_fd)
train_vectors_fd.shape
train_vectors_fd = pd.DataFrame(train_vectors_fd)


# In[64]:


hy_train_df = pd.concat([train_df,train_vectors_fd], axis=1)
hy_train_df.head(3)


# In[65]:


h_labels = hy_train_df["Id"]
h_y = hy_train_df["Pawpularity"]
h_X = hy_train_df.drop(["Id","Pawpularity"], axis=1)

# Normalization
encoder = MinMaxScaler()
encoder.fit(h_X)
h_X_scaled = encoder.transform(h_X)
h_X_scaled = pd.DataFrame(h_X_scaled, columns=h_X.columns)

h_X_train, h_X_valid, h_y_train, h_y_valid = train_test_split(
    h_X_scaled, h_y, test_size=0.3, random_state=42)

print("-"*80)
print("Train and test split sizes")
print("-"*80)
print(f"X_train : {h_X_train.shape}")
print(f"X_test : {h_X_valid.shape}")
print(f"y_train : {h_y_train.shape[0]}")
print(f"y_test : {h_y_valid.shape[0]}")
print("-"*80)


# In[66]:


h_rfr = RandomForestRegressor(random_state=8)
param_grid = {
            "n_estimators" : [10,50,100],
            "max_features" : ["log2", "sqrt"],
            "max_depth"    : [5,15,25],
            "bootstrap"    : [True, False]
        }

h_grid_rfr = GridSearchCV(
    h_rfr,
    param_grid,
    cv = 5,
    verbose=1,
    n_jobs=-1)

h_best_rfr = h_grid_rfr.fit(h_X_train, h_y_train)


# In[67]:


print("-"*80)
print("Best parameters for Random Forest model")
print("-"*80)
print(h_best_rfr.best_params_)
print("-"*80)


# In[68]:


h_rfr_pred = h_best_rfr.predict(h_X_valid)


# In[69]:


fig = plt.figure(figsize=(12,8))
plt.scatter(x=h_rfr_pred, y=h_y_valid)
plt.ylabel("Pawpularity real values (y_valid)")
plt.xlabel("Predicted values (rfr_pred)")
plt.title("Predicted Pawpularity VS True values with RandomForest", 
          fontsize=20, fontweight='bold')
plt.show()


# We will nevertheless **make a first submission to obtain a baseline**.
# 
# # <span style="color: #186fb4" id="section_5">Competition submission on Test set</span>

# ```Python
# submission_df = pd.read_csv("".join([PATH,"test.csv"]))
# submission_df = submission_df[["Id"]]
# submission_df["Image"] =  submission_df["Id"].apply(lambda x: x+".jpg")
# 
# submission_generator = test_datagen.flow_from_dataframe(
#     dataframe=submission_df,
#     directory=PATH+"test/",
#     x_col="Image",
#     y_col=None,
#     target_size=(299,299),
#     batch_size=32,
#     seed=42,
#     shuffle=False,
#     class_mode=None)
# ```

# ```Python
# submission_pred = my_xcept_model.predict(submission_generator)
# submission_pred.shape
# ```

# In[70]:


test_vectors_fd = feature_detect_img("test", img_size=299)


# In[71]:


test_vectors_fd = np.array(test_vectors_fd)
test_vectors_fd = np.squeeze(test_vectors_fd)
test_vectors_fd.shape
test_vectors_fd = pd.DataFrame(test_vectors_fd)


# In[72]:


hy_test_df = pd.concat([test_df,test_vectors_fd], axis=1)
hy_test_df.head(3)


# In[73]:


h_test_labels = hy_test_df["Id"]
h_X_test = hy_test_df.drop("Id", axis=1)
h_X_test_scaled = encoder.transform(h_X_test)
h_X_test_scaled = pd.DataFrame(h_X_test_scaled, columns=h_X_test.columns)


# In[74]:


submission_pred = h_best_rfr.predict(h_X_test_scaled)


# In[75]:


fig = plt.figure(figsize=(10,7))
plt.hist((submission_pred))
plt.xlabel("Pawpularity Score")
plt.ylabel("number of individuals")
plt.title("Distribution of predicted submission results", 
          fontsize=20, fontweight='bold')
plt.show()


# In[76]:


submission_df["Pawpularity"] = (submission_pred)
submission_df = submission_df[["Id","Pawpularity"]]
submission_df.head()


# In[77]:


submission_df.to_csv("submission.csv", sep=",", index=False)


# **The submission RMSE score on this Baseline model *(based on images only)* is approximately 25.96**.
# We will explore a multimodal approach in a second Notebook: https://www.kaggle.com/michaelfumery/pawpularity-multimodal-cnn
