#!/usr/bin/env python
# coding: utf-8

# <img src="https://i.imgur.com/VJotrA9.png">
# 
# <center><h1> - Data Understanding & Language Analysis - </h1></center>
# 
# > 📜 **Goal**: Predict the correct ordering of the **cells** within a Jupyter Notebook.
# 
# **What is a 🦠 cell** - notice here that a **cell** can be both:
# * a `coding` cell - where you write code
# * a `markdown` cell - where you can write text, add images etc.
# 
# **❗ Important** - within the `.json` filesthe `code` cells are in their *correct* order - only the `markdown` cells have been *shuffled*.
# 
# Oh boy, this competition sounds like FUN. I owe MOST of what I've learned as Data Scientist to notebooks. I never was an "only code" person, so discovering that you can explain, comment, add images and schemas to your code in order to make it more *readable* was a life changer for me.
# 
# Also, notebooks can have a "learning-teaching" experience that a raw `.py` file just ... doesn't cut it for me.
# 
# So let's get started!
# 
# ### ⬇ Libraries

# In[1]:


get_ipython().system('pip install spacy-language-detection')


# In[2]:


# Libraries
import os
import gc
import wandb
from time import time
import random
import math
import glob
import json
from bisect import bisect
from scipy.sparse import vstack
from tqdm import tqdm
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
from matplotlib import cm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Rectangle
from IPython.display import display_html
plt.rcParams.update({'font.size': 16})

# Spacy Language Detector
import spacy
from spacy.language import Language
from spacy_language_detection import LanguageDetector

# Environment check
warnings.filterwarnings("ignore")
os.environ["WANDB_SILENT"] = "true"
CONFIG = {'competition': 'AI4Code', '_wandb_kernel': 'aot'}

# Custom colors
class clr:
    S = '\033[1m' + '\033[93m'
    E = '\033[0m'
    
my_colors = ["#CDFC74", "#F3EA56", "#EBB43D", 
             "#DF7D27", "#D14417", "#B80A0A", "#9C0042"]
my_pastels = ["#A5EC9B", "#B4E185", "#C3D973", 
             "#CDCD61", "#CCB049", "#CB812D", "#B93221"]
my_darks = ["#FCF238", "#F19321", "#E54F14", 
             "#C22318", "#B01028", "#9D0642", "#85006C"]

gradient1 = ["#a5ec9b", "#abe890", "#b0e485", "#b7e07b", "#bddb71", 
             "#c3d667", "#cad15e", "#d1cc55", "#d7c64e", "#dec147", "#e5ba41", "#ebb43d"]
CMAP1 = ListedColormap(my_colors)
CMAP2 = ListedColormap(my_colors[-1])

print(clr.S+"Notebook Color Schemes:"+clr.E)
sns.palplot(sns.color_palette(my_pastels))
sns.palplot(sns.color_palette(my_colors))
sns.palplot(sns.color_palette(my_darks))
plt.show()


# ### 🐝 W&B Fork & Run
# 
# In order to run this notebook you will need to input your own **secret API key** within the `! wandb login $secret_value_0` line. 
# 
# 🐝**How do you get your own API key?**
# 
# Super simple! Go to **https://wandb.ai/site** -> Login -> Click on your profile in the top right corner -> Settings -> Scroll down to API keys -> copy your very own key (for more info check [this amazing notebook for ML Experiment Tracking on Kaggle](https://www.kaggle.com/ayuraj/experiment-tracking-with-weights-and-biases)).
# 
# <center><img src="https://i.imgur.com/fFccmoS.png" width=500></center>

# In[3]:


# 🐝 Secrets
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("wandb")

get_ipython().system(' wandb login $secret_value_0')


# ### ⬇ Helper Functions

# In[4]:


def count_inversions(a):
    '''src: https://www.kaggle.com/code/ryanholbrook/getting-started-with-ai4code'''
    inversions = 0
    sorted_so_far = []
    for i, u in enumerate(a):
        j = bisect(sorted_so_far, u)
        inversions += i - j
        sorted_so_far.insert(j, u)
    return inversions


def kendall_tau(ground_truth, predictions):
    '''src: https://www.kaggle.com/code/ryanholbrook/getting-started-with-ai4code'''
    total_inversions = 0
    total_2max = 0  # twice the maximum possible inversions across all instances
    for gt, pred in zip(ground_truth, predictions):
        ranks = [gt.index(x) for x in pred]  # rank predicted order in terms of ground truth
        total_inversions += count_inversions(ranks)
        n = len(gt)
        total_2max += n * (n - 1)
    return 1 - 4 * total_inversions / total_2max


def show_values_on_bars(axs, h_v="v", space=0.4):
    '''Plots the value at the end of the a seaborn barplot.
    axs: the ax of the plot
    h_v: weather or not the barplot is vertical/ horizontal'''
    
    def _show_on_single_plot(ax):
        if h_v == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height()
                value = int(p.get_height())
                ax.text(_x, _y, format(value, ','), ha="center") 
        elif h_v == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height()
                value = int(p.get_width())
                ax.text(_x, _y, format(value, ','), ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _show_on_single_plot(ax)
    else:
        _show_on_single_plot(axs)
        
        
# === 🐝 W&B ===
def save_dataset_artifact(run_name, artifact_name, path):
    '''Saves dataset to W&B Artifactory.
    run_name: name of the experiment
    artifact_name: under what name should the dataset be stored
    path: path to the dataset'''
    
    run = wandb.init(project='AI4Code', 
                     name=run_name, 
                     config=CONFIG)
    artifact = wandb.Artifact(name=artifact_name, 
                              type='dataset')
    artifact.add_file(path)

    wandb.log_artifact(artifact)
    wandb.finish()
    print("Artifact has been saved successfully.")
    
    
def create_wandb_plot(x_data=None, y_data=None, x_name=None, y_name=None, title=None, log=None, plot="line"):
    '''Create and save lineplot/barplot in W&B Environment.
    x_data & y_data: Pandas Series containing x & y data
    x_name & y_name: strings containing axis names
    title: title of the graph
    log: string containing name of log'''
    
    data = [[label, val] for (label, val) in zip(x_data, y_data)]
    table = wandb.Table(data=data, columns = [x_name, y_name])
    
    if plot == "line":
        wandb.log({log : wandb.plot.line(table, x_name, y_name, title=title)})
    elif plot == "bar":
        wandb.log({log : wandb.plot.bar(table, x_name, y_name, title=title)})
    elif plot == "scatter":
        wandb.log({log : wandb.plot.scatter(table, x_name, y_name, title=title)})
        
        
def create_wandb_hist(x_data=None, x_name=None, title=None, log=None):
    '''Create and save histogram in W&B Environment.
    x_data: Pandas Series containing x values
    x_name: strings containing axis name
    title: title of the graph
    log: string containing name of log'''
    
    data = [[x] for x in x_data]
    table = wandb.Table(data=data, columns=[x_name])
    wandb.log({log : wandb.plot.histogram(table, x_name, title=title)})
    
    
# 🐝 Log Cover Photo
run = wandb.init(project='AI4Code', name='CoverPhoto', config=CONFIG)
cover = plt.imread("../input/ai4code-processed-data/AI4Code Cover.png")
wandb.log({"example": wandb.Image(cover)})
wandb.finish()


# # 1. The Data
# 
# ## 1.1 The .csv files
# * `train_orders.csv` - TRAIN DATA
#     * `id` - unique ID of the notebook
#     * `cell_order` - gives the correct order of *each cell* within this notebook
# 
# 
# * `train_ancestors`
#     * `id` - unique ID of the notebook (same as for `train_orders.csv`)
#     * `ancestor_id` - notebook that has a common origin or *ancestor* (good as grouping factor when constructing validation splits)
#     * `parent_id` - the *original* notebook, which may be present in the train data or not

# In[5]:


# 🐝 W&B Experiment
run = wandb.init(project='AI4Code', name='metadata-explore', config=CONFIG)

# Read in original data
orders = pd.read_csv("../input/AI4Code/train_orders.csv")
ancestors = pd.read_csv("../input/AI4Code/train_ancestors.csv")


# In[6]:


wandb.log({"unique_notebooks" : orders.id.nunique()})

print(clr.S+"~~~~ TRAIN ~~~~"+clr.E)
print(clr.S+"Orders:"+clr.E)
print(f"Shape: {orders.shape} with {orders.id.nunique()} unique IDs.", "\n")
orders.head()


# In[7]:


wandb.log({"unique_ancestors" : ancestors.ancestor_id.nunique()})

print(clr.S+"Ancestors:"+clr.E)
print(f"Shape: {ancestors.shape} also with {ancestors.id.nunique()} unique IDs.")
print(f"There are also {ancestors.ancestor_id.nunique()} unique ancestor_id and {ancestors.parent_id.nunique()} unique parent_id.")
ancestors.head()


# ### 🕸 Network Analysis on Ancestors

# In[8]:


# Get frequency per ancestor_id
data = ancestors.groupby("ancestor_id")["id"].count().reset_index().\
                sort_values("id", ascending=False).reset_index(drop=True)
data.columns = ["ancestor_id", "count"]


# Basic metrics
total_singles = data[data["count"]==1].shape[0]
total_double_plus = data[data["count"]>1].shape[0]

print(clr.S+"Total number of ids with only 1 ancestor:"+clr.E, total_singles, "\n"+ "\t"*4+
      clr.S+"percent:"+clr.E, round(total_singles/len(data), 3), "\n")
print(clr.S+"Total number of ids with 2+ ancestors:"+clr.E, total_double_plus, "\n"+ "\t"*4+
      clr.S+"percent:"+clr.E, round(total_double_plus/len(data), 3))


# In[9]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
fig.suptitle('Analysis on the 6,843 ids that have more than 1 ancestor', 
             weight="bold", size=25)

# Violinplot
sns.violinplot(data=data[data["count"]>1], y="count", ax=ax1, color=my_colors[3])
ax1.set_title("Frequency of ancestors for an id", weight="bold", size=19)
ax1.set_ylabel("No. of Ancestors", size = 18, weight="bold")
ax1.axhline(y=10, linestyle = '--', color=my_pastels[0], lw=4)
ax1.text(x=-0.43, y=8, s="most values are < 10", color=my_darks[3], size=15, weight="bold")
ax1.arrow(x=0.05, y=12, dx=0, dy=45, color=my_pastels[0], lw=4, 
          head_width=0.01, head_length=1, linestyle = '--')
ax1.text(x=0.055, y=59, s="There are 168 outliers \n with values > 10",
         color=my_darks[3], size=15, weight="bold")

# Barplot
sns.barplot(data=data[data["count"]>1].head(12), x="count", y="ancestor_id", ax=ax2, 
            palette=gradient1)
show_values_on_bars(axs=ax2, h_v="h", space=0.4)
ax2.set_title("Top 12 Ancestor IDs with most connections", weight="bold", size=19)
ax2.set_ylabel("Ancestor ID", size = 18, weight="bold")
ax2.set_xlabel("")
ax2.set_xticks([])

# Arrow
style = "Simple, tail_width=5, head_width=16, head_length=23"
kw = dict(arrowstyle=style, color=my_darks[3])
arrow = patches.FancyArrowPatch((40, 11.1), (64, 0.7),
                             connectionstyle="arc3,rad=-.10", **kw)
plt.gca().add_patch(arrow)

sns.despine(right=True, top=True);


# In[10]:


# 🐝 Log plots
wandb_data = data[data["count"]>1].head(12)
create_wandb_plot(x_data=wandb_data["ancestor_id"],
                  y_data=wandb_data["count"],
                  x_name="Ancestor ID", 
                  y_name="Frequency", 
                  title="Top 12 Ancestor IDs with most connections",
                  log="top_12", plot="bar")

wandb_data = data[data["count"]>1]
create_wandb_hist(x_data=wandb_data["count"],
                  x_name="No. of Ancestors", 
                  title="Frequency of ancestors for an id",
                  log="ancestor_freq")


# In[11]:


random.seed(25)

# Add a fictive y
# this "y" doesn't mean anything, it's just for
# showcasing purposes
data["y"] = [random.randint(0, 100) for i in range(len(data))]
perc = round(data[data["count"]<=10].shape[0]/len(data), 3)*100

plt.figure(figsize=(24, 10))
sns.scatterplot(data=data, x="count", y="y", size="count", alpha=0.65, sizes=(100, 7000),
               hue="count", palette=CMAP1)

plt.title("Distribution of Ancestor IDs per frequency", weight="bold", size=25)
plt.xlabel("No. of Ancestors", size = 18, weight="bold")
plt.ylabel("")
plt.yticks([])

plt.axvline(x=10, linestyle = '--', color="#757F62", lw=4)
plt.text(x=11, y=85, s=f"{perc}% of the data is here", color="#757F62", size=17, weight="bold")
plt.arrow(x=25, y=83, dx=-17, dy=0, color="#757F62", lw=4, 
          head_width=1.2, head_length=0.5)

plt.text(x=54, y=50, s=f"Biggest outlier", color=my_colors[-1], size=17, weight="bold")
plt.arrow(x=54, y=48, dx=8, dy=0, color=my_colors[-1], lw=4, 
          head_width=1.2, head_length=0.5)

plt.legend('',frameon=False)

sns.despine(right=True, top=True, left=True);


# In[12]:


# 🐝 Finish this experiment
wandb.finish()


# ## 1.2 The .json files
# 
# ❗ **Important**: the `.json` files have a `dict` structure and contain first the `code cells` (that are in correct order) and then the `markdown cells` that have been shuffled.
# 
# > The `.json` files have the following structure (e.g.):
# ```
# {
# 'cell_type': 
# 
#     {'1862f0a6': 'code',
#       '2a9e43d6': 'code',
#       '038b763d': 'code',
#       ...
#       '21616367': 'markdown',
#       'fcb6792d': 'markdown',
#       '63c26fa2': 'markdown',
#       ...
#      },
#  'source': 
#      {'1862f0a6': '# This Python 3 environment comes with many helpful analytics libraries ....',
#       '2a9e43d6': 'import numpy as np\nimport pandas as pd\nimport random\n\nfrom sklearn.model_selection ...',
#       '038b763d': "import warnings\nwarnings.filterwarnings('ignore')",
#       '2eefe0ef': "matplotlib.rcParams.update({'font.size': 14})",
#       ...
#       'aaad8355': '*Тип данных обучающего сета*',
#       '503926eb': 'Инициализация класса Data',
#       '3e5f860d': 'Признаки Rooms, KitchenSquare, HouseFloor имеют в некоторых наблюдениях нулевые значения'
#       }
# }
# ```
# 
# <center><img src="https://i.imgur.com/QrfbveZ.png"></center>

# In[13]:


def get_json_data(ID):
    '''
    Returns a df containing the .json information.
    ID: name of file
    return :: a df comtaining cols "cell_id", "cell_type", "source"
    '''

    # Read in the .json file
    file = json.load(open(f"../input/AI4Code/train/{ID}.json"))

    # Create an empty dataframe of size n
    # where n = numver of cell ids in the notebook
    n = len(file["cell_type"].keys())
    df = pd.DataFrame(index=range(n),columns=["cell_id", "cell_type", "source"])

    # Get all sources in order
    all_sources = list(file["source"].values())

    # Add cell id and type to dataframe
    for k, (cell_id, cell_type) in enumerate(file["cell_type"].items()):
        df.loc[k, "cell_id"] = cell_id
        df.loc[k, "cell_type"] = cell_type
        # as cell_id is in the same order for both "cell_type" and "source"
        df.loc[k, "source"] = all_sources[k]
        
    return df


# In[14]:


# Show an example
ID = "00001756c60be8"

example_df = get_json_data(ID)
example_df.head()


# # 2. Language Detection
# 
# > 📜 **Note**: The first example we get is a notebook that has all the `markdown` cells in what seems to be Russian! This means that throughout the notebooks there are **other languages that we could encounter besides English**.
# 
# Let's see what is the proportion on languages.
# 
# ## 2.1 Set up the Language Detector
# 
# I will be using the `spacy` library for this part.

# In[15]:


# Language Detector Function
def get_lang_detector(nlp, name):
    return LanguageDetector(seed=42)

# Load spacy model
nlp_model = spacy.load("en_core_web_sm")

# Create instance for language detection
Language.factory("language_detector", func=get_lang_detector)
nlp_model.add_pipe('language_detector', last=True)


# ## 2.2 Function to extract the language
# 
# The function `get_document_language()` simply accesses the information from a `.json` file and returns a dictionary of the form `{"language":"en", "score":0.97655}`, where:
# * `language` - specifies the preponderent language of the notebook
# * `score` - specifies the probability
# 
# *TODO: Add multiple language detection e.g.: 70% english + 25% french + 5% spanish*

# In[16]:


def get_document_language(ID):
    '''
    Returns the language of the document.
    ID: name of file
    return :: dictionary containing the language and score (probability)
    '''
    # Retrieve .json df
    df = get_json_data(ID)

    # Get a string of all doc text
    # Keep only first 200 chars to not overload memory
    all_doc_text = " ".join(df[df["cell_type"]=="markdown"]["source"].tolist())[:200]

    # Get document language
    doc = nlp_model(all_doc_text)
    language = doc._.language
    
    return language


# In[17]:


# An example
ID = "00001756c60be8"
language = get_document_language(ID)

print(clr.S+f"--- Notebook: {ID} ---"+clr.E)
print(clr.S+"The language of this document is:"+clr.E, language["language"])
print(clr.S+"With a probability of:"+clr.E, language["score"])


# ## 2.3 Retrieve language for all 140k files
# 
# > 📜 **Note**: Because the cell below takes ~ 1hr and 15 mins to run, I have commented it and saved the result to a separate [dataset](https://www.kaggle.com/datasets/andradaolteanu/ai4code-processed-data) and into my [W&B Dashboard](https://wandb.ai/andrada/AI4Code?workspace=user-andrada) for easy access.

# In[18]:


# # === Uncomment this cell to run it ===

# # Retrieve all languages for all notebooks
# all_languages = []

# # This takes ~ 1hr 15 mins
# for k, ID in tqdm(enumerate(orders["id"])):
#     all_languages.append(get_document_language(ID))
    
# # Convert to dataframe
# all_lang_df = pd.DataFrame(all_languages)
# all_lang_df["id"] = orders["id"]

# # Save file
# # .parquet is smaller than .csv
# all_lang_df.to_parquet("all_languages.parquet", index=False)


# In[19]:


# Read in the languages
all_lang_df = pd.read_parquet("../input/ai4code-processed-data/all_languages.parquet")

# 🐝 Save artifact to W&B
save_dataset_artifact(run_name="languages-data", 
                      artifact_name="language", 
                      path="../input/ai4code-processed-data/all_languages.parquet")


# ## 2.4 Add full-name mapping
# 
# As we have seen, the `spacy` library mapps using the convention ISO Code 2 - where the mapping is made using **2 letters** instead of the full name of the language.
# 
# Hence I have imported [this dataset from wikipedia](https://www.kaggle.com/datasets/andradaolteanu/iso-language-codes) to map the codes to the full language.

# In[20]:


# Import external mapping of the languages
iso_codes = pd.read_csv("../input/iso-language-codes/ISO_languages_codes.csv")

# Add full name
all_lang_df = all_lang_df.merge(right=iso_codes, 
                                left_on="language", right_on="2_letter_code", 
                                how="left").iloc[:, :4]

all_lang_df.head()


# In[21]:


# Save file
all_lang_df.to_parquet("all_languages_mapped.parquet", index=False)

# 🐝 Save artifact to W&B
save_dataset_artifact(run_name="languages-data-mapped", 
                      artifact_name="language_mapped", 
                      path="../input/ai4code-processed-data/all_languages_mapped.parquet")


# # 3. Language Analysis on the Notebooks
# 
# **📜 Main takeaways:**
# * *90% of the notebooks are in English* and only 10% of the rest are written in other languages
# * from the 10%, the most encountered languages are:
#     * Portuguese - by far the most frequent
#     * Russian
#     * Turkish
#     * Japanese
#     * Italian
#     * Korean
#     * Spanish
# 
# **❗ What should we do with this information?**
# * *should we delete* from training the notebooks that are non-english or should we try to *incorporate* them?
# * is this *90-10 proportion the same* for the `test` data too?

# In[22]:


# 🐝 W&B Experiment
run = wandb.init(project='AI4Code', name='language-explore', config=CONFIG)


# In[23]:


all_lang_df["is_en"] = all_lang_df["language"].apply(lambda x: "English" if x =="en" else "Other")

# 🐝 Log into W&B
wandb.log({"distinct_languages" : all_lang_df["iso_language_name"].nunique(),
           "perc_en_notebooks" : all_lang_df["iso_language_name"].value_counts()[0]/len(all_lang_df["iso_language_name"])})

print(clr.S+"Total number of unique languages present within the notebooks:"+clr.E,
      all_lang_df["iso_language_name"].nunique())
print(clr.S+"Percentage of notebooks in English:"+clr.E,
      all_lang_df["iso_language_name"].value_counts()[0]/len(all_lang_df["iso_language_name"]), "\n")
print(clr.S+"Other languages:"+clr.E,
      all_lang_df["iso_language_name"].value_counts().index[1:].tolist())


# In[24]:


# Barchart data
data = all_lang_df["iso_language_name"].value_counts().reset_index()
data.columns = ["language", "count"]
data = data[data["language"]!="English"]

# Piechart data
labels = all_lang_df["is_en"].value_counts().index.tolist()
sizes = all_lang_df["is_en"].value_counts().values.tolist()
explode = (0, 0.2)

# Plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 18))
fig.suptitle('Language Analysis',weight="bold", size=25)

# Pie
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90, colors=[my_colors[2], my_colors[4]],
        labeldistance=1.06)
ax1.set_title("Percentage of English Notebooks vs Other Languages", weight="bold", size=19)
ax1.axis('equal')

sns.barplot(data=data, x="count", y="language", ax=ax2, 
            palette="autumn")
show_values_on_bars(axs=ax2, h_v="h", space=0.4)
ax2.set_title("Frequency of Other Languages", weight="bold", size=19)
ax2.set_ylabel("Language", size = 18, weight="bold")
ax2.set_xlabel("")
ax2.set_xticks([])

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.8, hspace=None)
sns.despine(right=True, top=True, bottom=True);


# In[25]:


# 🐝 Finish this experiment
wandb.finish()


# <img src="https://i.imgur.com/3gcqR20.png">
# 
# <center><h1> - Baseline Model & Hyperparameter Tuning - </h1></center>
# 
# ### ⬇ Libraries

# In[26]:


from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBRanker
from scipy import sparse


# # 1. Create Rank & Analyse
# 
# We first need to create out `target` feature, which will be the `rank`, or the **order** in witch the cells are organised.

# In[27]:


# ~~~ For now I will REMOVE all notebooks that are not in english ~~~

# Filter out other languages
only_english = all_lang_df[all_lang_df["iso_language_name"]=="English"].reset_index(drop=True)

# Merge
data = pd.merge(left=orders, right=only_english, on="id").iloc[:, :2]

del iso_codes, all_lang_df
gc.collect()


# In[28]:


# ~~~ Create Rank (or Order) for each cell ~~~

# Explode cell_order into multiple rows
data["cell_order"] = data["cell_order"].apply(lambda x: x.split())
data = data.explode("cell_order").reset_index(drop=True)

# Create rank
data['rank'] = 1
data['rank'] = data.groupby(['id'])['rank'].cumsum()

data.head()


# ### How many cells are in a notebook?
# 
# With this new `rank` feature we can now look at a distribution for **the number of cells within notebooks**.
# 
# > 📜 **Note:** I would keep in mind here that the *number of cells* are *NOT necessarily correlated* with how much `code` there is in a notebook. A notebook *can have 2-3 cells with many lines of code or text* in it.

# In[29]:


no_cells = data.groupby("id")["rank"].max().values

# Plot
print(clr.S+"=== Metrics ==="+clr.E)
print(clr.S+"Min no. of cells:"+clr.E, no_cells.min(), "\n" +
      clr.S+"Mean no. of cells:"+clr.E, no_cells.mean(), "\n" +
      clr.S+"Max no. of cells:"+clr.E, no_cells.max())

plt.figure(figsize=(24, 10))
sns.distplot(no_cells, rug=True,
             rug_kws={"color": my_pastels[0]},
             kde_kws={"color": my_darks[-1], "lw": 5, "alpha": 0.7},
             hist_kws={"histtype": "step", "linewidth": 3, "alpha": 1, "color": my_pastels[0]})

plt.title("Distribution of the Number of Cells in a Notebook", weight="bold", size=25)
plt.xlabel("No. of Cells", size = 18, weight="bold")
plt.ylabel("Frequency")

plt.axvline(x=50, linestyle = '--', color=my_darks[-1], lw=2)
plt.text(x=60, y=0.015, s=f"Most notebooks have ~ 50 total cells", 
         color=my_darks[-1], size=17, weight="bold")

sns.despine(right=True, top=True, left=True);


# In[30]:


del no_cells
gc.collect()


# # 2. Create training dataset
# 
# The second step consists of **retrieving the `source` information** (meaning the `code` and `markdown` info) from the `.json` files.
# 
# ## 2.1 Retrieving the `source` for each cell
# 
# *❗ The cell bellow takes a while, so I have commented it and saved the `train` file to [my dataset](https://www.kaggle.com/datasets/andradaolteanu/ai4code-processed-data).*

# In[31]:


# # === Uncomment this cell to run it ===

# # Get all data from the .json files
# all_id_data = []

# for ID in tqdm(data["id"].unique()):
#     id_data = get_json_data(ID)
#     id_data["id"] = [ID] * len(id_data)
#     all_id_data.append(id_data)
    
# # Concatenate all dataframes together
# train = pd.DataFrame(columns=["cell_id", "cell_type", "source", "id"])
# train = pd.concat(all_id_data)

# # Merge Rank info
# train = pd.merge(left=train, right=data, 
#                  left_on=["id", "cell_id"], right_on=["id", "cell_order"])
# train.drop(columns="cell_order", inplace=True)

# # Compute the percentage rank
# # Divide each rank to the total number of cells within a notebook
# train["percent_rank"] = train["rank"] / train.groupby("id")["cell_id"].transform("count")

# # Add ancestor data
# train = pd.merge(left=train, right=ancestors, on=["id"])

# train.to_parquet("train.parquet", index=False)


# In[32]:


# Load in the saved file
train = pd.read_parquet("../input/ai4code-processed-data/train.parquet")

# 🐝 Save artifact to W&B
save_dataset_artifact(run_name="train_data", 
                      artifact_name="train_data", 
                      path="../input/ai4code-processed-data/train.parquet")


# ### - Cells Analysis -
# 
# 📜 **Notes**:
# * On average in the notebooks there are twice more `code` cells than `markdown` cells.
# * Usually the distribution between `code` and `markdown` matches - let's see the correlation.

# In[33]:


# Data
cell_analysis = train.groupby(["id", "cell_type"])["cell_id"].count().reset_index()
cell_avg = cell_analysis.groupby("cell_type")["cell_id"].mean().reset_index()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
fig.suptitle('Cell Type Analysis', 
             weight="bold", size=25)

# Barplot
sns.barplot(data=cell_avg, x="cell_type", y="cell_id", ax=ax1,
            palette=[my_colors[4], my_darks[-2]])
show_values_on_bars(axs=ax1, h_v="v", space=0.4)
ax1.set_title("Average count per Cell Type", weight="bold", size=19)
ax1.set_ylabel("Cell Type", size = 18, weight="bold")
ax1.set_xlabel("")
ax1.arrow(x=1.5, y=22, dx=-1, dy=0, color=my_colors[4], lw=3, 
          head_width=0.2, head_length=0.05, linestyle = '-')
ax1.text(x=0.43, y=23, s="Twice more code cells than markdown cells",
         color=my_colors[4], size=15, weight="bold")

# Hist
sns.histplot(data=cell_analysis, x="cell_id", hue="cell_type", ax=ax2,
             palette=[my_colors[4], my_darks[-2]])
ax2.set_title("Cell Type distribution per Notebook", weight="bold", size=19)
ax2.set_ylabel("Frequency", size = 18, weight="bold")

sns.despine(right=True, top=True);


# 📜 **Notes**:
# * The coorelation shows that in the notebooks with small number of `markdown cells` there is also a small number of `code cells`
# * The relationship works vice versa too - as the correlation is **direct and positive** between these 2 cell types.

# In[34]:


# Data
scatter_data = pd.pivot(data=cell_analysis, index="id", columns="cell_type", values="cell_id")
scatter_data["size"] = 30

plt.figure(figsize=(24, 10))
sns.scatterplot(data=scatter_data, x="code", y="markdown", size="size",
                hue="size", alpha=0.65, palette=CMAP2, sizes=(300, 6000))

plt.title("Correlation between Markdown & Code", weight="bold", size=25)
plt.xlabel("Code", size = 18, weight="bold")
plt.ylabel("Markdown", size = 18, weight="bold")

plt.legend('')

sns.despine(right=True, top=True, left=True);


# In[35]:


del cell_analysis, cell_avg, scatter_data
gc.collect()


# ## 2.2 Sample Down the data
# 
# > 📜 **Note**: as the amount of data to work with is huge, **I will be sampling down** considerably in this notebook, in order to make a runable pipeline and perform the hyperparameter tuning.

# In[36]:


# ~~~ Choose a % of the data ~~~
PERC_DATA = 0.3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# In[37]:


random.seed(24)

# Get all unique ids
unique_ids = train["id"].unique().tolist()
print(clr.S+"Total unique ids:"+clr.E, len(unique_ids))

# Sample down
unique_ids = random.sample(unique_ids, k=int(len(unique_ids)*PERC_DATA))
print(clr.S+"Sampled unique ids:"+clr.E, len(unique_ids))

train = train[train["id"].isin(unique_ids)].reset_index(drop=True)
print(clr.S+"Sampled train Shape:"+clr.E, train.shape)


# In[38]:


del unique_ids
gc.collect()


# # 3. Data Split
# 
# **🙏 Code below is from [Getting Started with AI4Code](https://www.kaggle.com/code/ryanholbrook/getting-started-with-ai4code)**.

# In[39]:


# Size of validation set
NVALID = 0.1

# Create selection object
splitter = GroupShuffleSplit(n_splits=1, test_size=NVALID, random_state=0)


# Set column "id" as index
train = train.set_index("id")

# Split, keeping notebooks with a common origin (ancestor_id) together
ids = train.index.unique('id')
ancestor_data = ancestors.set_index('id').loc[ids, 'ancestor_id']
ids_train, ids_valid = next(splitter.split(ids, groups=ancestor_data))
ids_train, ids_valid = ids[ids_train], ids[ids_valid]

# Create train and validation sets
df_train = train.loc[ids_train, :]
df_valid = train.loc[ids_valid, :]

print(clr.S+"Original Shape:"+clr.E, train.shape)
print(clr.S+"Train Shape:"+clr.E, df_train.shape)
print(clr.S+"Valid Shape:"+clr.E, df_valid.shape)


# In[40]:


del train, ids, ancestor_data, ids_train, ids_valid
gc.collect()


# # 4. Feature Engineering
# 
# ◾ **min_df**: when building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold.

# In[41]:


# Glimpse of how the train dataset looks now
df_train.head(3)


# In[42]:


# Remove tokens that have a frequency lower than 15%
tfidf = TfidfVectorizer(min_df=0.15)

# Create the features from the text
# within each cell
X_train = tfidf.fit_transform(df_train['source'].astype(str))

# Create the target variable
# which is the rank (order) of the cells
y_train = df_train["rank"].to_numpy()

# Number of cells in each notebook
groups = df_train.groupby("id")["rank"].max().values


# In[43]:


# The same process for data validation
X_valid = tfidf.transform(df_valid['source'].astype(str))
y_valid = df_valid["rank"].to_numpy()


# ### ❗ Creating data leakage on purpose
# 
# A question I've asked myself is: **how do you train the `rank` target while telling the model that it has to rank ONLY for `markdown` cells**?
# 
# As we know, within the `.json` files the code cells are in the correct order and only the `markdown` cells are shuffled. So, we don't really have to do anything for the `cell` codes, but only for markdown.
# 
# 📜 In the [Getting Started with AI4Code](https://www.kaggle.com/code/ryanholbrook/getting-started-with-ai4code) notebook, the authors do something that I found SUPER cool. They *leak* the information for the code cells, meaning that they create a **new feature column** that:
# * has the *correct* rank for all `code` cells
# * has a dummy value `0` for all the `markdown` cells
# 
# Genius, never seen smth like this before. Let us do the same:

# In[44]:


X_train = sparse.hstack((X_train, 
                         np.where(df_train['cell_type']=='code', 
                                  df_train['rank'], 0).reshape(-1, 1)))


# In[45]:


X_valid = sparse.hstack((X_valid, 
                         np.where(df_valid['cell_type']=='code', 
                                  df_valid['rank'], 0).reshape(-1, 1)))


# # 5. XGBRanker
# 
# As the model we'll be using the `XGBRanker()` from `xgboost`.
# 
# > 📜 **Note**: This algorithm **doesn't try to predict the `rank` per se**, but rather **what is the order importance** of each `cell` within the notebook. 
# 
# Hence, as the end instead of this:
# 
# `actual_rank: [1, 2, 3, 4, 5, 6, 7]` & `predicted_rank: [1, 2, 3, 4, 5, 6, 7]`
# 
# we'll have this:
# 
# `actual_rank: [1, 2, 3, 4, 5, 6, 7]` & `predicted_rank: [-0.7, 0, 0.33, 0.4, 0.45, 0.8, 0.88]`
# 
# ## 5.1 Training Function
# 
# Let us first create the training function. I'll also initiate a new `wandb` experiment and I will log into the dashboard the `kendall_tau` correlation as the final metric.

# In[46]:


def train_XGBRanker():
    
    config_defaults = {"booster":'gbtree',
                   "objective":'rank:pairwise',
                   "random_state":24, 
                   "learning_rate":0.1,
                   "n_estimators":110}
    
    # 🐝 W&B Experiment
    config_defaults.update(CONFIG)
    run = wandb.init(project='AI4Code', name='xgbRanker', config=config_defaults)
    config = wandb.config
    
    # Initiate the model
    model = XGBRanker(booster=config.booster,
                      objective=config.objective,
                      random_state=config.random_state, 
                      learning_rate=config.learning_rate,
                      n_estimators=config.n_estimators)

    # Train the model
    model.fit(X_train, y_train, group=groups, verbose=True)

    # Create df containing the cell_id and the prediction
    predict = pd.DataFrame({"cell_id" : df_valid["cell_id"],
                            "pred" : model.predict(X_valid)}, index = df_valid.index)

    # Sort (using the predicted rank) and then group
    predict = predict.sort_values(by = ['id', 'pred'], ascending = [False, True])\
                        .groupby('id')['cell_id'].apply(list)

    # Create the same but for actual data
    actual = df_valid.sort_values(by = ['id', 'rank'], ascending = [False, True])\
                            .groupby('id')['cell_id'].apply(list)

    # Kendall Metric
    metric = kendall_tau(actual, predict)
    print(clr.S+"Kendall Tau"+clr.E, metric)
    wandb.log({"kendall_tau": np.float(metric)})


# ## 5.2 First Run - Baseline
# 
# > 📜 We get a score of **0.5849**.

# In[47]:


train_XGBRanker()


# ## 5.3 Hyperparameter Tuning [🐝 W&B Sweeps]
# 
# For the hyperparameter tuning part I will be using the W&B integrated [Sweeps for XGBoost](https://docs.wandb.ai/guides/integrations/xgboost) method to log all my experiments.
# 
# *🙏 The tutorial I am following is [Using_W&B_Sweeps_with_XGBoost](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/boosting/Using_W%26B_Sweeps_with_XGBoost.ipynb#scrollTo=VCRlDRL6_5aA).*
# 
# <center><img src="https://i.imgur.com/iIFr3w4.png"></center>
# 
# > ❗ **Note:** for the good functioning of Sweeps it is very important that the training function aka `train_XGBRanker()` does NOT have any **arguments** passed. Hence, the format of the `wandb.agent()` should always be `wandb.agent(sweep_id, train_XGBRanker, count=20)` and **NOT** `wandb.agent(sweep_id, train_XGBRanker(data, model, config), count=20)`.

# In[48]:


# Sweep Config
sweep_config = {
    "method": "random", # grid for all
    "metric": {
      "name": "kendall_tau",
      "goal": "maximize"   
    },
    "parameters": {
        "learning_rate": {
            "values": [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
        },
        "n_estimators": {
            "values": [130, 140, 150, 160, 170, 180, 190, 200]
        },
        "random_state": {
            "values": [21, 22, 23, 24, 25, 26, 27, 28]
        }
    }
}

# Sweep ID
sweep_id = wandb.sweep(sweep_config, project="AI4Code")


# In[49]:


# 🐝 RUN SWEEPS
start = time()

# count = the number of trials/experiments to run
wandb.agent(sweep_id, train_XGBRanker, count=20)
print("Sweeping took:", round((time()-start)/60, 1), "mins")


# The [Sweeps Dashboard](https://wandb.ai/andrada/AI4Code/sweeps/w957m2lh?workspace=user-andrada) shows the following:
# * All the runs on time vs the performance of the Kendall metric + an importance panel containing the most important features during training.
# <center><img src="https://i.imgur.com/jUy6J2J.png" width=700></center>
# 
# * A visualization with every experiment and its performance.
# <center><img src="https://i.imgur.com/L8eXSD0.png" width=700></center>
# 
# <div class="alert alert-block alert-info">
#   <p>📜<b> Best Score So Far:</b> Kendall Tau 0.55 | learning_rate: 0.09 | n_estimators: 130 | learning_rate: 24 | DATA_PERC: 0.04</p>
# </div>
# 
# <div class="alert alert-block alert-info">
#   <p>📜<b> UPDATE Best Score So Far:</b> Kendall Tau 0.5938 | learning_rate: 0.08 | n_estimators: 190 | learning_rate: 24 | DATA_PERC: 0.1</p>
# </div>

# In[50]:


# 🐝 Finish the Experiment
wandb.finish()


# # 6. Prediction
# 
# The final step is to save our best model and create the **Submission Pipeline**.
# 
# > 📜 **Note**: Because we are training the final model with ALL the data, I will also merge the `train` and `valid` datasets together, as well as recompute the `groups` instance.

# In[51]:


# Create the group
# As now we'll train with ALL the data (train + valid)
final_groups = pd.concat((df_train, df_valid)).groupby("id")["rank"].max().values

best_configs = {"booster":'gbtree',
                "objective":'rank:pairwise',
                "random_state":22, 
                "learning_rate":0.09,
                "n_estimators":200}
    
# Initiate the model
final_model = XGBRanker(booster=best_configs["booster"],
                  objective=best_configs["objective"],
                  random_state=best_configs["random_state"], 
                  learning_rate=best_configs["learning_rate"],
                  n_estimators=best_configs["n_estimators"])

# Train the final model
final_model.fit(vstack((X_train, X_valid)), np.concatenate((y_train, y_valid)),
                group=final_groups, verbose=True)

# Save it
final_model.save_model("XGBRanker_best.json")


# ### Load Final Model

# In[52]:


final_model = XGBRanker()
final_model.load_model("../input/ai4code-processed-data/XGBRanker_best.json")


# In[ ]:





# In[ ]:





# In[53]:


# <center><video src="mp4" width=800 controls></center>


# <center><img src="https://i.imgur.com/0cx4xXI.png"></center>
# 
# ### 🐝 W&B Dashboard
# 
# > My [W&B Dashboard](https://wandb.ai/andrada/AI4Code?workspace=user-andrada).
# 
# <center><img src="https://i.imgur.com/ZRwRJcw.png"></center>
# 
# <center><img src="https://i.imgur.com/knxTRkO.png"></center>
# 
# ### My Specs
# 
# * 🖥 Z8 G4 Workstation
# * 💾 2 CPUs & 96GB Memory
# * 🎮 2x NVIDIA A6000
# * 💻 Zbook Studio G7 on the go
