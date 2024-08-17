#!/usr/bin/env python
# coding: utf-8

# # Introduction
# My first thought in this competition was that **cuteness** would be a big deal in predicting animal popularity  
# This notebook explores using CLIP's multi-modal representations (and understanding of abstract concepts) for feature engineering

# In[1]:


get_ipython().run_cell_magic('capture', '', "\n# https://www.kaggle.com/bguberfain/openai-clip-with-train/notebook\n\nimport sys\n!cp -r ../input/openai-clip/CLIP/CLIP-main /tmp/\n\n# Kaggle likes to unpack .gz files in datasets... so we have to pack it back\n!gzip -c /tmp/CLIP-main/clip/bpe_simple_vocab_16e6.txt > /tmp/CLIP-main/clip/bpe_simple_vocab_16e6.txt.gz\nsys.path.append('/tmp/CLIP-main')\n\n!pip install ../input/openai-clip/ftfy-5.9/ftfy-5.9\n!pip install ../input/openai-clip/torch-1.7.1+cu110-cp37-cp37m-linux_x86_64.whl \\\n             ../input/openai-clip/torchvision-0.8.2+cu110-cp37-cp37m-linux_x86_64.whl \\\n             ../input/faiss-163/faiss_gpu-1.6.3-cp37-cp37m-manylinux2010_x86_64.whl\n")


# In[2]:


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from os.path import join
import numpy as np
import pandas as pd
import clip, os, skimage
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm

clip.available_models()


# In[3]:


model, preprocess = clip.load("../input/openai-clip/ViT-B-32.pt", jit=False)
model = model.cuda().eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Input resolution:", input_resolution)
print("Context length:", context_length)
print("Vocab size:", vocab_size)


# # Visualising the Data

# In[4]:


train_image_path = Path("../input/petfinder-pawpularity-score/train")
file_names = [f.name for f in train_image_path.iterdir() if f.suffix == ".jpg"]


# In[5]:


original_images = []
images = []
plt.figure(figsize=(15, 12))

for filename in file_names[:9]:
    image = Image.open(join(train_image_path, filename))
  
    plt.subplot(3, 3, len(images) + 1)
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])

    original_images.append(image)
    images.append(preprocess(image))


# # Prompt Feature Engineering
# [Multimodal Neurons in Artificial Neural Networks](https://openai.com/blog/multimodal-neurons/) shows that CLIP can respond to abstract concepts, such as emotions or geographical regions. With some priors around what makes animals popular, this could create some interesting features.  
# 
# It's possible information around concepts could be extracted from an image, with well-defined prompts  
# We use the cosine similarity between language and image embeddings to extract features for modelling here

# In[6]:


texts = ['Cute',
         'Funny',
         'Derp', # let's see if this works
         'Small',
         'Happy',
         'Sad',
         'Aggressive',
         'Friendly',
         'Old',
         'Young',
         'Love']


# In[7]:


image_input = torch.tensor(np.stack(images)).cuda()
text_tokens = clip.tokenize(texts).cuda()

# text_tokens = clip.tokenize([f"A {w} photo of a" + w for w in texts]).cuda()


# In[8]:


with torch.no_grad():
    image_features = model.encode_image(image_input).float()
    text_features = model.encode_text(text_tokens).float()

image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

similarity_matrix = torch.inner(text_features, image_features).cpu()


# In[9]:


count = len(texts)

plt.figure(figsize=(20, 16))
plt.imshow(similarity_matrix, vmin=0.1, vmax=0.3, cmap = 'RdBu')

plt.yticks(range(count), texts, fontsize=18)
plt.xticks([])

for i, image in enumerate(original_images):
    plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
for x in range(similarity_matrix.shape[1]):
    for y in range(similarity_matrix.shape[0]):
        plt.text(x, y, f"{similarity_matrix[y, x]:.2f}", ha="center", va="center", size=12)

for side in ["left", "top", "right", "bottom"]:
    plt.gca().spines[side].set_visible(False)

plt.xlim([-0.5, count - 0.5])
plt.ylim([count + 0.5, -2])

plt.title("Cosine similarity matrix between text and image features", size=20, loc='left')
plt.show()


# # Labelling Training Set

# In[10]:


class PetDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.files = [f for f in path.iterdir() if f.suffix == ".jpg"]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        _im_path = self.files[idx]
        _img = Image.open(_im_path)
        _img = preprocess(_img)
        return _img, _im_path.name.split('.')[0]


# In[11]:


def create_similarity_features(dl):
    features = []
    names = []
    with torch.no_grad():
        for xb, name in dl:
            xb = xb.cuda()
            xb = model.encode_image(xb)
            xb /= xb.norm(dim=-1, keepdim=True)
            sim_matrix = torch.inner(text_features, xb.float()).cpu().numpy()
            features.append(sim_matrix)
            names.append(name)
    return features, names


# In[12]:


ds = PetDataset(train_image_path)
dl = DataLoader(ds, batch_size = 400, shuffle=False)
train_features, train_names = create_similarity_features(dl)


# In[13]:


train_features_df = pd.DataFrame(np.hstack(train_features).T,
                           index = np.hstack(train_names).T,
                           columns = texts)
df_corr = train_features_df.corr()
plt.figure(figsize=(13,8))

plt.title("Correlation matrix between engineered features", size=20, loc='left') 
sns.heatmap(df_corr, cmap='RdBu', annot=True, linewidths=2)


# In[14]:


# saving training features for later use
train_features_df.to_csv('clip_features.csv')


# ---

# # K-Fold Training and Search
# XGB training and hyperparameter search code come from Abhishek Thakur's notebook:  
# https://www.kaggle.com/abhishek/optuna-xgboost-meta-features-only

# In[15]:


from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import optuna
from functools import partial


# In[16]:


def run(trial, fold, df, useful_features):
    learning_rate = trial.suggest_float("learning_rate", 1e-2, 0.25, log=True)
    reg_lambda = trial.suggest_loguniform("reg_lambda", 1e-8, 100.0)
    reg_alpha = trial.suggest_loguniform("reg_alpha", 1e-8, 100.0)
    subsample = trial.suggest_float("subsample", 0.1, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.1, 1.0)
    max_depth = trial.suggest_int("max_depth", 1, 7)
    
    xtrain = df[df.kfold != fold].reset_index(drop=True)
    xvalid = df[df.kfold == fold].reset_index(drop=True)

    ytrain = xtrain.Pawpularity
    yvalid = xvalid.Pawpularity

    xtrain = xtrain[useful_features]
    xvalid = xvalid[useful_features]

    model = XGBRegressor(
        random_state=42,
        tree_method="gpu_hist",
        gpu_id=1,
        n_estimators=10000,
        predictor="gpu_predictor",
        learning_rate=learning_rate,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        max_depth=max_depth,
    )
    model.fit(xtrain, ytrain, early_stopping_rounds=300, eval_set=[(xvalid, yvalid)], verbose=1000)
    preds_valid = model.predict(xvalid)
    rmse = mean_squared_error(yvalid, preds_valid, squared=False)
    return rmse


# In[17]:


train_df = pd.read_csv("../input/same-old-creating-folds/train_10folds.csv")
test_df = pd.read_csv("../input/petfinder-pawpularity-score/test.csv")
sample_submission = pd.read_csv("../input/petfinder-pawpularity-score/sample_submission.csv")
test_image_path = train_image_path = Path("../input/petfinder-pawpularity-score/test")

useful_features = [
    'Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
    'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur',
    'Cute', 'Funny', 'Derp', 'Small', 'Happy', 'Sad', 'Aggressive',
    'Friendly', 'Old', 'Young', 'Love']


# In[18]:


ds = PetDataset(test_image_path)
dl = DataLoader(ds, batch_size = 400, shuffle=False)
test_features, test_names = create_similarity_features(dl)

test_features_df = pd.DataFrame(np.hstack(test_features).T,
                                index = np.hstack(test_names).T,
                                columns = texts)


# In[19]:


train_df = train_df.join(train_features_df, on = 'Id')
test_df = test_df.join(test_features_df, on = 'Id')


# In[20]:


opt_fun = partial(
    run,
    fold=0,
    df=train_df,
    useful_features=useful_features,
)

study = optuna.create_study(direction="minimize")
study.optimize(opt_fun, n_trials=200)
print(study.best_params)


# In[21]:


study.best_value, study.best_params


# In[22]:


def generate_predictions(params, fold, df, df_test, useful_features):    
    xtrain = df[df.kfold != fold].reset_index(drop=True)
    xvalid = df[df.kfold == fold].reset_index(drop=True)
    xtest = df_test.copy()

    ytrain = xtrain.Pawpularity
    yvalid = xvalid.Pawpularity

    xtrain = xtrain[useful_features]
    xvalid = xvalid[useful_features]
    xtest = xtest[useful_features]

    model = XGBRegressor(
        random_state=42,
        tree_method="gpu_hist",
        gpu_id=1,
        n_estimators=10000,
        predictor="gpu_predictor",
        **params,
    )
    model.fit(xtrain, ytrain, early_stopping_rounds=300, eval_set=[(xvalid, yvalid)], verbose=1000)
    preds_valid = model.predict(xvalid)
    test_preds = model.predict(xtest)
    rmse = mean_squared_error(yvalid, preds_valid, squared=False)
    print(rmse)
    return test_preds


# In[23]:


final_predictions = []
for fold_ in range(10):
    final_predictions.append(
        generate_predictions(
            study.best_params,
            fold=fold_,
            df=train_df,
            df_test=test_df,
            useful_features=useful_features,
        )
    )


# In[24]:


final_predictions = np.mean(np.column_stack(final_predictions), axis=1)
sample_submission.Pawpularity = final_predictions
sample_submission.to_csv("submission.csv", index=False)

