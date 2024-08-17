#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy import stats
import pandas as pd
import numpy as np
import warnings
import gc
import os

import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
from IPython.core.display import display, HTML


warnings.simplefilter("ignore")
gc.enable()


# In[2]:


# notebook's styles
def css_styling():
    styles = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500&display=swap');    
    
    /* Variables */
    
    :root {
        --primary-text-color: #ef4444;
        --text-color: #fff;
        --text-font-size: 15px !important;
        --primary-title-border-color: linear-gradient(to right, var(--primary-text-color) 0%, var(--primary-text-color) 50%, var(--background-color)) 0 1 100% 1;
        --background-color: #202020;
        --shadow-color: #6b7280;
        --note-background-color: rgba(191,219,254, 0.1);
        --note-text-color: #93c5fd;
        --notebook-font: "Inter", sans-serif;
        --quote-background-color: rgba(148,163,184,0.25);
        --quote-text-color: #f3f4f6;
    }
    
    /* Notebook's defaults */
    
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    .jp-NotebookPanel-notebook, #notebook-container, .text_cell, .text_cell_render  {
        background-color: var(--background-color) !important;
        padding-bottom: 50px;
    }
    
    #kaggle-portal-root-global + div {
        display: none !important;
    }
    
    .jp-OutputArea-output, .jp-OutputArea-output pre, .jp-OutputArea-output thead th {
        color: var(--text-color) !important;
    }
    
    /* Notebook's styles */
    
    /* Title's styles */
    
    .title_container {
        background-color: var(--background-color); 
        position: relative;
        border-width: 5px;
        border-style: solid;
        border-image: var(--primary-title-border-color);
        margin-bottom: 10px;
        font-family: var(--notebook-font);
    }
    
    .border-0 {
         border-width: 0px;
         margin-bottom: 0px;
    }
    
    .title span {
        color: var(--text-color); 
        font-weight: 600;
    }
    
    .paragraph_text {
        color: var(--primary-text-color) !important;
        font-family: var(--notebook-font);
    }
    
    .paragraph {
        color: var(--primary-text-color) !important;
        font-family: var(--notebook-font);
    }
    
    h1 .paragraph {
        font-size: 35px;
    }

    h2 .paragraph {
        font-size: 28px;
    }
    
    /* Text's styles */
    
    .text {
        color: var(--text-color) !important;
        font-size: var(--text-font-size);
        font-family: var(--notebook-font);
    }
    
    .highlight_text {
        background-color: var(--primary-text-color);
        color: var(--text-color);
        font-weight: bold;
        padding: 1px 5px 1px 5px; 
        border-radius: 3px;
        font-size: var(--text-font-size);
    }
    
    .text_link {
        color: #60a5fa !important;
        font-size: var(--text-font-size);
    }
    
    /* List's styles */
    
    .list ul li  {
        color:  var(--text-color) !important;
        font-size: var(--text-font-size);
    }
    
    .list ul li a:not(.text_link) {
        color: var(--text-color) !important;
        font-size: var(--text-font-size);
    }
    
    .list ul li {
        margin-top: 0px !important;
    }
    
    .list ul li::marker {
        color: var(--primary-text-color);
    }
    
    /* Image's styles */
    
    .image_container {
        display: flex;
        flex-direction: column;
        align-items: center;
        font-family: var(--notebook-font);
    }
     
    .image_container img {
        max-width: 99%;
        box-shadow: 0px 0px 15px var(--shadow-color) !important;
    }
    
    .image_container .text {
        text-align: center;
        margin-top: 10px;
        color: #d1d5db !important;
        font-size: var(--text-font-size);
    }    
    
    /* Note's styles */
    
    .note {
        background-color: var(--note-background-color);
        padding: 10px;
        border-radius: 5px;
        font-family: var(--notebook-font);
    }
    
    .note_text {
        color: var(--note-text-color);
        font-size: var(--text-font-size);
    }
    
    /* Quote's styles */
    
    .quote {
        background-color: var(--quote-background-color);
        padding: 10px;
        border-radius: 5px;
        font-family: var(--notebook-font);
    }
    
    .quote_text {
        color: var(--quote-text-color);
        font-size: var(--text-font-size);
    }
    
    
    """
    return HTML("<style>"+styles+"</style>")

css_styling()


# In[3]:


# colors
BACKGROUND_COLOR = "#202020"
PRIMARY_COLOR = "#ef4444"
TEXT_COLOR = "#fff"
MALE_COLOR = "#60a5fa"
FEMALE_COLOR = "#d946ef"

colors = ["#ef4444",  "#f59e0b",  "#eab308", "#22c55e", "#60a5fa", "#4f46e5", "#9333ea", "#6b7280"]
colors = sns.color_palette(colors)

palette_colors = ["#ef4444", "#7f1d1d", "#f5f5f1", "#ef4444", "#7f1d1d"]
palette = mpl.colors.LinearSegmentedColormap.from_list("", palette_colors)

# visualization styles
visualizaiton_parameters = {
    # figure styles
    "figure.facecolor": BACKGROUND_COLOR,
    
    # axes styles
    "axes.facecolor": BACKGROUND_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "axes.labelpad": 7.0,
    "xtick.color": TEXT_COLOR,
    "xtick.minor.size": 0.0,
    "xtick.major.size": 0.0,
    "xtick.major.pad": 7.0,
    "xtick.minor.pad": 7.0,
    "ytick.color": TEXT_COLOR,
    "ytick.minor.size": 0.0,
    "ytick.major.size": 0.0,
    "ytick.major.pad": 7.0,
    "ytick.minor.pad": 7.0,
    
    # text styles
    "text.color": TEXT_COLOR,
    "font.family": "serif",
    
    # spines styles
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.bottom": True,
    "axes.spines.left": False,
    
    # grid styles
    "grid.alpha": 0.5,
    "grid.color": TEXT_COLOR,
    "grid.linewidth": 1.0,
    "grid.linestyle": "-",
    
}

mpl.rcParams.update(visualizaiton_parameters)

# visualization utilties
def hide_spines(ax, spines=["top", "right", "left", "bottom"]):
    for spine in spines:
        ax.spines[spine].set_visible(False)
        
        
def get_missing_values(data_frame, stat="count"):
    missing_values = data_frame.isnull().sum()
    
    if stat == "percentage":
        num_samples = len(data_frame)
        missing_values = (missing_values / num_samples) * 100
    
    columns = missing_values.index.tolist()
    values = missing_values.values
    
    missing_values_data_frame = pd.DataFrame({
        "column": columns,
        "missing_values": values,
    })
    
    return missing_values_data_frame


# In[4]:


# utilities

# the below code is copied from https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
def compute_cramers_v_function(x, y): 
    confusion_matrix = pd.crosstab(x,y)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))


def get_cramers_v_correlation(data_frame, categorical_columns=None):
    if categorical_columns is None:
        categorical_columns = data_frame.columns[data_frame.dtypes == "object"]
    
    cat_data_frame = data_frame[categorical_columns]
    
    rows = []
    for x in cat_data_frame:
        col = []
        for y in cat_data_frame :
            cramers = compute_cramers_v_function(cat_data_frame[x], cat_data_frame[y]) 
            col.append(round(cramers,2))
        rows.append(col)
        
    cramers_results = np.array(rows)
    cramers_v_correlation = pd.DataFrame(cramers_results, columns=cat_data_frame.columns, index=cat_data_frame.columns)
    
    return cramers_v_correlation


# <div class="list">
#     <div class="title_container">
#         <h1 class="title">
#             <span>Table of contents</span>
#         </h1>
#     </div>
#     <ul>
#         <li><a href="#introduction">Introduction</a></li>
#         <li><a href="#data">Data</a></li>
#         <li>
#             <a href="#eda">Exploratory Data Analysis</a>
#             <ul>
#                 <li>First view</li>
#                 <li>Target Variable Distribution</li>
#                 <li>Numerical Features Distributions</li>
#                 <li>Numerical Features Distributions vs Target Variable</li>
#                 <li>Categorical Features Distributions</li>
#                 <li>Categorical Features Distributions vs Target Variable</li>
#                 <li>Pearson's Correlation Coefficient</li>
#                 <li>Cramer's V Correlation Coefficient</li>
#                 <li>Missing values</li>
#                 <li>Duplicates</li>
#             </ul>
#         </li>
#         <li>
#             <a href="#baseline">Baseline</a>
#             <ul>
#                 <li>Metric</li>
#                 <li>Configuration</li>
#                 <li>Reproducibility</li>
#                 <li>Utilities</li>
#                 <li>Validation split</li>
#                 <li>Training</li>
#             </ul>
#         </li>
#         <li>
#             <a href="#baseline">Submission</a>
#             <ul>
#                 <li>Format</li>
#                 <li>Submission</li>
#                 <li>Leaderboard</li>
#             </ul>
#         </li>
#         <li><a href="#references">References</a></li>
#         <li><a href="#other">Other</a></li>
#     </ul>
#  </div>

# <br>
# <div class="image_container">
#     <img src="https://i.ibb.co/QX3YrTS/Screenshot-7.png">
# </div>
# 
# <div class="title_container">
#     <h1 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>1.</span>
#         <span>Introduction</span>
#     </h1>
# </div>
# <div>
# <p class="text">
#     The task of the competition is to predict whether a patient is likely to get stroke based on the input parameters like gender, age, various diseases, and smoking status. Each row in the data provides relavant information about the patient.<br><br>
#     A stroke is a medical condition in which poor blood flow to the brain causes cell death. There are two main types of stroke: ischemic, due to lack of blood flow, and hemorrhagic, due to bleeding. Both cause parts of the brain to stop functioning properly.<br> 
# </p>
# <div class="image_container">
#     <img src="https://kobrincrb.by/wp-content/uploads/2022/01/2630186.jpg">
# </div>
# <br>
# <p class="text">Signs and symptoms of a stroke may include an inability to move or feel on one side of the body, problems understanding or speaking, dizziness, or loss of vision to one side. Signs and symptoms often appear soon after the stroke has occurred. If symptoms last less than one or two hours, the stroke is a transient ischemic attack (TIA), also called a mini-stroke. A hemorrhagic stroke may also be associated with a severe headache. The symptoms of a stroke can be permanent. Long-term complications may include pneumonia and loss of bladder control.<br><br><span class="highlight_text">The main risk factor for stroke is high blood pressure. Other risk factors include high blood cholesterol, tobacco smoking, obesity, diabetes mellitus, a previous TIA, end-stage kidney disease, and atrial fibrillation.</span> An ischemic stroke is typically caused by blockage of a blood vessel, though there are also less common causes. A hemorrhagic stroke is caused by either bleeding directly into the brain or into the space between the brain's membranes. Bleeding may occur due to a ruptured brain aneurysm. Diagnosis is typically based on a physical exam and supported by medical imaging such as a CT scan or MRI scan. A CT scan can rule out bleeding, but may not necessarily rule out ischemia, which early on typically does not show up on a CT scan. Other tests such as an electrocardiogram (ECG) and blood tests are done to determine risk factors and rule out other possible causes. Low blood sugar may cause similar symptoms.<br><br>Prevention includes decreasing risk factors, surgery to open up the arteries to the brain in those with problematic carotid narrowing, and warfarin in people with atrial fibrillation. Aspirin or statins may be recommended by physicians for prevention. A stroke or TIA often requires emergency care. An ischemic stroke, if detected within three to four and half hours, may be treatable with a medication that can break down the clot. Some hemorrhagic strokes benefit from surgery. Treatment to attempt recovery of lost function is called stroke rehabilitation, and ideally takes place in a stroke unit; however, these are not available in much of the world.<br><br>In 2013, approximately 6.9 million people had an ischemic stroke and 3.4 million people had a hemorrhagic stroke. In 2015, there were about 42.4 million people who had previously had a stroke and were still alive. Between 1990 and 2010 the number of strokes which occurred each year decreased by approximately 10% in the developed world and increased by 10% in the developing world. In 2015, stroke was the second most frequent cause of death after coronary artery disease, accounting for 6.3 million deaths (11% of the total). About 3.0 million deaths resulted from ischemic stroke while 3.3 million deaths resulted from hemorrhagic stroke. About half of people who have had a stroke live less than one year. Overall, two thirds of strokes occurred in those over 65 years old.<br><br>Source: <a href="https://en.wikipedia.org/wiki/Stroke" class="text_link">Stroke</a> </p>

# <div class="title_container">
#     <h1 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>2.</span>
#         <span>Data</span>
#     </h1>
# </div>
# <div>
# <p class="text">
#     The dataset for this competition (both train and test) <span class="highlight_text">was generated from a deep learning model</span> trained on the <a href="https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset" class="text_link">Stroke Prediction Dataset</a>. Feature distributions are close to, but not exactly the same, as the original. <span class="highlight_text">Feel free to use the original dataset as part of this competition</span>, both to explore differences as well as to see whether incorporating the original in training improves model performance.<br><br>
#     <b>train.csv</b> - the training dataset; stroke is the binary target.<br>
#     <b>test.csv</b> - the test dataset; your objective is to predict the probability of positive stroke.<br>
#     <b>sample_submission.csv</b> - a sample submission file in the correct format<br><br>
#     Source: <a href="https://www.kaggle.com/competitions/playground-series-s3e2/data" class="text_link">Dataset Description</a>
# </p>
# <p class="text">
#     <b>id</b> - unique identifier<br>
#     <b>gender </b>- "Male", "Female" or "Other"<br>
#     <b>age</b> - age of the patient<br>
#     <b>hypertension</b> - 0 if the patient doesn't have hypertension, 1 if the patient has hypertension<br>
#     <b>heart_disease</b> - 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease<br>
#     <b>ever_married</b> - "No" or "Yes"<br>
#     <b>work_type</b> - "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"<br>
#     <b>Residence_type</b> - "Rural" or "Urban"<br>
#     <b>avg_glucose_level</b> - average glucose level in blood<br>
#     <b>bmi</b> - body mass index<br>
#     <b>smoking_status</b> - "formerly smoked", "never smoked", "smokes" or "Unknown"*<br>
#     <b>stroke</b> - 1 if the patient had a stroke or 0 if not
# </p>
# <div class="note">
#     <span class="note_text">Note: "Unknown" in smoking_status means that the information is unavailable for this patient.</span>
# </div>
# <br>
# <p class="text">
#     Source: <a href="https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset" class="text_link">About Dataset</a>
# </p>

# In[5]:


directory = "/kaggle/input/playground-series-s3e2/"
train_path = os.path.join(directory, "train.csv")
test_path = os.path.join(directory, "test.csv")
sample_submission_path = os.path.join(directory, "sample_submission.csv")


# In[6]:


train = pd.read_csv(train_path)
train_ids = train.pop("id")

test = pd.read_csv(test_path)
test_ids = test.pop("id")

sample_submission = pd.read_csv(sample_submission_path)


# In[7]:


# data types preprocessing
binary_features = ["hypertension", "heart_disease"]
train[binary_features] = train[binary_features].astype(str)
test[binary_features] = test[binary_features].astype(str)


# <div class="title_container">
#     <h1 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>3.</span>
#         <span>Exploratory Data Analysis</span>
#     </h1>
# </div>
# <p class="text">
#  Before doing the model development, we always should do Exploratory Data Analysis first. Exploratory Data Analysis is one of the most important steps during the whole project development because we can find important and useful insights, relations, etc from the data itself, for example, we can make the decision to do additional data preprocessing steps or Feature Engineering and it can even make a better model generalization and finally improve our metrics significantly.
# </p>

# <div class="title_container border-0">
#     <h2 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>3.1.</span>
#         <span>First view</span>
#     </h2>
# </div>

# In[8]:


train


# <div class="title_container border-0">
#     <h2 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>3.2.</span>
#         <span>Target Variable Distribution</span>
#     </h2>
# </div>

# In[9]:


target_column = "stroke"


# In[10]:


figure_colors = [colors[3], colors[0]]
figure = plt.figure(figsize=(10, 7))
axis = figure.add_subplot()
axis.grid(axis="y", zorder=0)
sns.countplot(x=target_column,  palette=figure_colors, data=train, edgecolor="#fff", linewidth=1.25, ax=axis, zorder=2)
axis.xaxis.set_tick_params(size=0, labelsize=14)
axis.set_xlabel(target_column, fontsize=17)
axis.yaxis.set_tick_params(size=0, labelsize=12)
axis.set_ylabel("count", fontsize=14)
axis.tick_params(axis="both", which="both", length=0)
axis.set_ylim(1)
axis.spines["bottom"].set(linewidth=1, color=TEXT_COLOR)
figure.show()


# <div class="title_container border-0">
#     <h2 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>3.3.</span>
#         <span>Numerical Features Distributions</span>
#     </h2>
# </div>

# In[11]:


features = np.array([column for column in train.columns if column != target_column])
dtypes = np.array([dtype for column, dtype in zip(train.columns, train.dtypes) if column != target_column])
numeric_columns = features[dtypes != "object"]
num_numeric_columns = len(numeric_columns)

categorical_columns = features[dtypes == "object"]
num_categorical_columns = len(categorical_columns)


# In[12]:


columns = 3
rows = (num_numeric_columns // columns)  + 1

figure = plt.figure(figsize=(20, 12))
for index, column in enumerate(numeric_columns):
    data = train[column].values
    
    axis = figure.add_subplot(rows, columns, index+1)
    axis.grid(axis="both", zorder=0)
    sns.kdeplot(x=data, fill=True, alpha=1.0, color=colors[0],  edgecolor="#fff",  linewidth=1.25, zorder=2, ax=axis)
    axis.xaxis.set_tick_params(labelsize=12)
    axis.set_xlabel(column, fontsize=13, labelpad=7)
    ylabel_text = "density"  if (index % columns) == 0 else ""
    axis.set_ylabel(ylabel_text, fontsize=13)
    axis.yaxis.set_tick_params(labelsize=10)
    axis.spines["bottom"].set(linewidth=1, color=TEXT_COLOR)
    axis.set_ylim(1e-6)
    
    # statistics
    mean = np.mean(data)
    median = np.median(data)
    std = np.std(data)
    axis.plot([], [], " ", label=f"Mean: {mean:.2f}\nMedian: {median:.2f}\nSTD: {std:.2f}")
    axis.legend()
    
figure.show()


# <div class="title_container border-0">
#     <h2 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>3.4.</span>
#         <span>Numerical Features Distributions vs Target Variable</span>
#     </h2>
# </div>

# In[13]:


columns = 3
num_numeric_columns = len(numeric_columns)
rows = (num_numeric_columns // columns)  + 1

figure = plt.figure(figsize=(20, 12))
axis_colors = [colors[3], colors[0]]
for index, column in enumerate(numeric_columns):
    axis = figure.add_subplot(rows, columns, index+1)
    axis.grid(axis="both", zorder=0)
    sns.kdeplot(x=column, data=train[train[target_column] == 0], fill=True, alpha=0.9, color=colors[3],  edgecolor="#fff",  linewidth=1.25, zorder=2, label="No stroke", ax=axis)
    sns.kdeplot(x=column, data=train[train[target_column] == 1], fill=True, alpha=0.9, color=colors[0],  edgecolor="#fff",  linewidth=1.25, zorder=3, label="Stroke", ax=axis)
    axis.xaxis.set_tick_params(labelsize=12)
    axis.set_xlabel(column, fontsize=13)
    ylabel_text = "density"  if (index % columns) == 0 else ""
    axis.set_ylabel(ylabel_text, fontsize=13)
    axis.yaxis.set_tick_params(labelsize=10)
    axis.spines["bottom"].set(linewidth=1, color=TEXT_COLOR)
    axis.legend()
    axis.set_ylim(1e-6)
    
figure.show()


# <div class="list">
#     <div class="title_container border-0">
#         <h3 class="title">
#             <span>Insights</span>
#         </h3>
#     </div>
#     <ul>
#         <li>Old people more suffers from stroke than young people while teenagers don't suffer at all.</li>
#         <li>Average glucose level's distribution for stroke people has "second distribution" in the right tail.</li>
#         <li>Body Mass Index's distribution has a little bias. People, who have lower BMI less suffer from stroke.</li>
#     </ul>
#  </div>

# <div class="title_container border-0">
#     <h2 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>3.5.</span>
#         <span>Categorical Features Distributions</span>
#     </h2>
# </div>

# In[14]:


columns = 2
rows = (num_categorical_columns // columns)  + 1

figure = plt.figure(figsize=(15, 12))
figure_colors = [[MALE_COLOR, FEMALE_COLOR, PRIMARY_COLOR]]

for index, column in enumerate(categorical_columns):
    axis_colors = colors
    if index < len(figure_colors):
        axis_colors = figure_colors[index]
    
    axis = figure.add_subplot(rows, columns, index+1)
    axis.grid(axis="y", zorder=0)
    sns.countplot(x=column, data=train, fill=True, palette=axis_colors, alpha=1.0, edgecolor="#fff", linewidth=1.25, zorder=2, ax=axis)
    axis.xaxis.set_tick_params(labelsize=12)
    axis.set_xlabel(column, fontsize=13)
    ylabel_text = "count"  if (index % columns) == 0 else ""
    axis.set_ylabel(ylabel_text, fontsize=13)
    axis.yaxis.set_tick_params(labelsize=10)
    axis.spines["bottom"].set(linewidth=1, color=TEXT_COLOR)
    axis.set_ylim(1e-6)
    
figure.tight_layout(h_pad=1)
figure.show()


# <div class="title_container border-0">
#     <h2 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>3.6.</span>
#         <span>Categorical Features Distributions vs Target Variable</span>
#     </h2>
# </div>

# In[15]:


columns = 2
rows = (num_categorical_columns // columns)  + 1

axis_colors = [colors[3], colors[0]]

figure = plt.figure(figsize=(15, 12))
for index, column in enumerate(categorical_columns):
    axis = figure.add_subplot(rows, columns, index+1)
    axis.grid(axis="y", zorder=0)
    sns.countplot(x=column, hue=target_column, data=train, fill=True, palette=axis_colors, alpha=1.0,  edgecolor="#fff",  linewidth=1.25, zorder=2, ax=axis)
    axis.xaxis.set_tick_params(labelsize=12)
    axis.set_xlabel(column, fontsize=13)
    ylabel_text = "count"  if (index % columns) == 0 else ""
    axis.set_ylabel(ylabel_text, fontsize=13)
    axis.yaxis.set_tick_params(labelsize=10)
    axis.spines["bottom"].set(linewidth=1, color=TEXT_COLOR)
    axis.set_ylim(1e-6)
    axis.legend(["No stroke", "Stroke"])
    
figure.tight_layout(h_pad=1)
figure.show()


# <div class="list">
#     <div class="title_container border-0">
#         <h3 class="title">
#             <span>Insights</span>
#         </h3>
#     </div>
#     <ul>
#         <li>We observe a huge imbalance for <b>stroke</b> patients.</li>
#         <li>There are small number of samples for some categories in the features.</li>
#     </ul>
#  </div>

# <div class="title_container border-0">
#     <h2 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>3.6</span>
#         <span>Pearson's Correlation Coefficient</span>
#     </h2>
# </div>
# <p class="text">In statistics, the Pearson correlation coefficient (PCC) ― also known as Pearson's r, the Pearson product-moment correlation coefficient (PPMCC), the bivariate correlation, or colloquially simply as the correlation coefficient ― is a measure of linear correlation between two sets of data. It is the ratio between the covariance of two variables and the product of their standard deviations; thus, it is essentially a normalized measurement of the covariance, such that the result always has a value between −1 and 1. As with covariance itself, the measure can only reflect a linear correlation of variables, and ignores many other types of relationships or correlations. As a simple example, one would expect the age and height of a sample of teenagers from a high school to have a Pearson correlation coefficient significantly greater than 0, but less than 1 (as 1 would represent an unrealistically perfect correlation).<br><br>Source: <a href="https://en.wikipedia.org/wiki/Pearson_correlation_coefficient" class="text_link">Pearson correlation coefficient
# </a></p>

# In[16]:


pearson_correlation = train[numeric_columns].corr(method="pearson")

figure = plt.figure(figsize=(num_numeric_columns*2, num_numeric_columns))
axis = figure.add_subplot()
sns.heatmap(
    pearson_correlation, 
    vmin=-1, 
    vmax=1,
    center=0, 
    annot=True, 
    linecolor=BACKGROUND_COLOR, 
    linewidth=1.5, 
    fmt=".2f", 
    cbar=False, 
    ax=axis,
    cmap=palette,
)
axis.xaxis.set_tick_params(labelsize=12)
axis.yaxis.set_tick_params(labelsize=12)
hide_spines(axis)
figure.show()


# <div class="title_container border-0">
#     <h2 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>3.7</span>
#         <span>Crammer's V Correlation Coefficient</span>
#     </h2>
# </div>
# <p class="text">In statistics, Cramér's V (sometimes referred to as Cramér's phi) is a measure of association between two nominal variables, giving a value between 0 and +1 (inclusive). It is based on Pearson's chi-squared statistic and was published by Harald Cramér in 1946.<br><br>Source: <a href="https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V" class="text_link">Cramér's V</a></p>

# In[17]:


cramers_v_correlation_data_frame = get_cramers_v_correlation(train, categorical_columns=categorical_columns)

figure = plt.figure(figsize=(num_categorical_columns*2, num_categorical_columns))
axis = figure.add_subplot()
sns.heatmap(
    cramers_v_correlation_data_frame,
    vmin=-1,
    vmax=1,
    center=0,
    annot=True,
    linecolor=BACKGROUND_COLOR,
    linewidth=1.5,
    fmt=".2f",
    cbar=False,
    cmap=palette,
    ax=axis,
)
axis.xaxis.set_tick_params(labelsize=12)
axis.yaxis.set_tick_params(labelsize=12)
hide_spines(axis)
figure.show()


# <div class="title_container border-0">
#     <h2 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>3.8.</span>
#         <span>Missing values</span>
#     </h2>
# </div>
# <p class="text">In statistics, missing data, or missing values, occur when no data value is stored for the variable in an observation. Missing data are a common occurrence and can have a significant effect on the conclusions that can be drawn from the data.<br><br>Source: <a href="https://en.wikipedia.org/wiki/Missing_data#:~:text=In%20statistics%2C%20missing%20data%2C%20or,be%20drawn%20from%20the%20data." class="text_link">Missing data</a></p>

# In[18]:


missing_values = get_missing_values(train, stat="percentage")

figure = plt.figure(figsize=(15, 7))
axis = figure.add_subplot()
axis.grid(axis="y", zorder=0)
sns.barplot(x="column", y="missing_values", data=missing_values, color=colors[0], edgecolor="#fff", alpha=1.0, linewidth=1.25, ax=axis, zorder=2)
axis.xaxis.set_tick_params(labelsize=14, rotation=40)
axis.set_xlabel("column", fontsize=17)
axis.yaxis.set_tick_params(labelsize=12)
axis.set_ylabel("missing values (%)", fontsize=14)
axis.spines["bottom"].set(linewidth=1, color=TEXT_COLOR)
axis.set_ylim(0, 100)
figure.show()


# <div class="title_container border-0">
#     <h2 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>3.9.</span>
#         <span>Duplicates</span>
#     </h2>
# </div>

# In[19]:


duplicates = train[train.duplicated()]
duplicates


# <div class="title_container">
#     <h1 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>4.</span>
#         <span>Baseline</span>
#     </h1>
# </div>
# <div>
# <p class="text"></p>

# <div class="title_container border-0">
#     <h2 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>4.1.</span>
#         <span>Metric</span>
#     </h2>
# </div>
# <p class="text">
#  Before building and training our model, we should decide what metric we should optimize. The organizators of the competition offer <a href="https://en.wikipedia.org/wiki/Receiver_operating_characteristic" class="text_link">ROC-AUC (Receiver Operating Characteristic Area Under Curve)</a> for evaluating our models (please refer to <a href="https://www.kaggle.com/competitions/playground-series-s3e2/overview/evaluation" class="text_link">Evaluation page</a>).  I will use <a href="https://scikit-learn.org/stable/modules/model_evaluation.html#receiver-operating-characteristic-roc" class="text_link">scikit-learn's implementation of ROC-AUC metric</a>.
#  </p>

# In[20]:


from sklearn.metrics import roc_auc_score


# <div class="title_container border-0">
#     <h2 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>4.2.</span>
#         <span>Configuration</span>
#     </h2>
# </div>

# In[21]:


NUM_FOLDS = 5
FOLDS = [1, 2, 3, 4, 5]
RANDOM_SEED = 42
OUTPUT_DIRECTORY = "./"
DECIMALS = 5


# <div class="title_container border-0">
#     <h2 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>4.3.</span>
#         <span>Reproducibility</span>
#     </h2>
# </div>

# In[22]:


import numpy as np
import random


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    
    return seed


RANDOM_SEED = seed_everything(RANDOM_SEED)

os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)

print(f"Random seed: {RANDOM_SEED}")


# <div class="title_container border-0">
#     <h2 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>4.2.</span>
#         <span>Utilities</span>
#     </h2>
# </div>

# In[23]:


from sklearn.model_selection import StratifiedKFold
import json
import os


def create_folds(dataset, labels, num_folds=5, folds=None, random_seed=42, **kwargs):
    strategy = StratifiedKFold(n_splits=num_folds, random_state=random_seed, **kwargs)
    splits = strategy.split(dataset, labels)
    
    if folds is None:
        folds = list(range(1, num_folds+1))
        
    folds = np.array(folds)
    folds_indexes = folds - 1
    splits = np.array(list(splits))
    splits = splits[folds_indexes]
    splits = np.array(list(zip(folds, splits)))
    
    return splits


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as file:
        data = json.dumps(data)
        file.write(data)
        
        
def remove_files_from_directory(directory, verbose=False):
    """
    Removes all files and folders from directory.
    """
        
    filenames = os.listdir(directory)
    pathes = [os.path.join(directory, filename) for filename in filenames]
        
    for path in pathes:
        if os.path.isfile(path) or os.path.islink(path):
            os.unlink(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)

        if verbose:
            print(f"Removed '{path}' from '{directory}'.")
        
def make_directory(directory, overwriting=False):
    """
    Makes directory
    """
    
    if not os.path.exists(directory):
        os.mkdir(directory)
    else:
        if overwriting:
            remove_files_from_directory(directory=directory)
    
    return directory


# <div class="title_container border-0">
#     <h2 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>4.4.</span>
#         <span>Validation split</span>
#     </h2>
# </div>

# In[24]:


folds_labels = train[target_column].values
folds = create_folds(
    dataset=train, 
    labels=folds_labels, 
    num_folds=NUM_FOLDS, 
    folds=FOLDS, 
    random_seed=RANDOM_SEED, 
    shuffle=True,
)

for fold, (train_indexes, validation_indexes) in folds:
    print(f"Fold {fold}: {len(validation_indexes)}")


# <div class="title_container border-0">
#     <h2 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>4.5.</span>
#         <span>Training</span>
#     </h2>
# </div>

# In[25]:


from catboost import CatBoostClassifier


oof = pd.DataFrame()
oof_scores = []
test_predictions = []
prediction_column = f"{target_column}_prediction"

for fold, (train_indexes, validation_indexes) in folds:
    # initializing fold
    print(f"Fold {fold}")
    fold_directory = os.path.join(OUTPUT_DIRECTORY, f"fold {fold}")
    make_directory(fold_directory, overwriting=True)
    
    # datasets
    train_fold = train.iloc[train_indexes]
    train_dataset = train_fold.drop(target_column, axis=1)
    train_labels = train_fold[target_column].values
    
    print(f"Train samples: {len(train_dataset)}")
    
    validation_fold = train.iloc[validation_indexes]
    validation_dataset =  validation_fold.drop(target_column, axis=1)
    validation_labels = validation_fold[target_column].values
    
    print(f"Validation samples: {len(validation_dataset)}")
    
    # model
    model_parameters = {
        "iterations":  1000,
        "learning_rate": 1e-3,
        "max_depth": 5,
        "early_stopping_rounds": 500,
        "verbose": 1000,
        "use_best_model": True,
        "eval_metric": "AUC",
        "task_type": "CPU",
        "cat_features": categorical_columns,
    }
    
    model = CatBoostClassifier(**model_parameters)
    
    # training
    plot_file = os.path.join(fold_directory, "training.html")
    model.fit(
        X=train_dataset, y=train_labels, 
        eval_set=(validation_dataset, validation_labels), 
        plot_file=plot_file,
    )
    
    model_path = os.path.join(fold_directory, "model")
    model.save_model(model_path)
    
    # validation
    validation_predictions = model.predict_proba(validation_fold)[:, 1]
    validation_fold[prediction_column] = validation_predictions
    validation_fold["fold"] = fold
    oof = pd.concat([oof, validation_fold], axis=0)
    
    fold_score = roc_auc_score(validation_labels, validation_predictions)
    oof_scores.append(fold_score)
    
    # inference
    test_fold_predictions = model.predict_proba(test)[:, 1]
    test_predictions.append(test_fold_predictions)
    
    # removing cache
    del model, train_fold, validation_fold
    del train_dataset, validation_dataset, train_labels, validation_labels
    gc.collect()
    
    print("\n"*3)
    
# oof scores
oof_scores = np.array(oof_scores)
oof_scores = np.round(oof_scores, DECIMALS)
oof_mean = round(oof_scores.mean(), DECIMALS)
oof_std = round(oof_scores.std(), DECIMALS)

print(f"OOF scores: {oof_scores}")
print(f"OOF mean: {oof_mean}")
print(f"OOF std: {oof_std}")
print()

# test predictions
test_predictions = np.array(test_predictions)
test_predictions = np.mean(test_predictions, axis=0)

# saving files
oof_path = os.path.join(OUTPUT_DIRECTORY, "oof.npy")
np.save(file=oof_path, arr=oof_scores)
print(f"OOF scores path: '{oof_path}'")


# <div class="title_container">
#     <h1 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>5.</span>
#         <span>Submission</span>
#     </h1>
# </div>
# <div>
# <p class="text">
# To submit our test predictions for the test dataset and see the test metric, the participants should submit a CSV file, which must have a certain format (names of columns and the input values). 
#  </p>
#  <div class="note">
#     <span class="note_text"> Note: Some competitions require submitting the notebook, which must generate the submission file with a certain name. For detailed information please refer to <a href="https://www.kaggle.com/docs/competitions" class="text_link">How to use Kaggle</a>.</span>
# </div>

# <div class="title_container border-0">
#     <h2 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>5.1.</span>
#         <span>Format</span>
#     </h2>
# </div>

# In[26]:


def create_submission(ids, predictions, path=None):
    submission = pd.DataFrame({
        "id": ids,
        "stroke": predictions,
    })
    
    if path is not None:
        submission.to_csv(path, index=False)
    
    return submission


# <div class="title_container border-0">
#     <h2 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>5.2.</span>
#         <span>Submission</span>
#     </h2>
# </div>

# In[27]:


submission_path = "submission.csv"
submission = create_submission(ids=test_ids, predictions=test_predictions, path=submission_path)
submission


# In[28]:


figure = plt.figure(figsize=(12, 7))
axis = figure.add_subplot()
axis.grid(axis="both", zorder=0)
sns.kdeplot(x=target_column, data=submission, fill=True, alpha=1.0, color=colors[0],  edgecolor="#fff",  linewidth=1.25, zorder=2, ax=axis)
axis.xaxis.set_tick_params(labelsize=12)
axis.set_xlabel("prediction", fontsize=13)
axis.set_ylabel("density", fontsize=13)
axis.yaxis.set_tick_params(labelsize=10)
hide_spines(axis)
axis.set_ylim(1)
figure.show()


# <div class="title_container border-0">
#     <h2 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>5.3.</span>
#         <span>Leaderboard</span>
#     </h2>
# </div>
# <p class="text">After submitting test predictions and getting the test metric, we also can check the leaderboard place and compare our results with other participants' results.</p>

# <div class="image_container">
#     <img src="https://i.ibb.co/j4wFL7f/Screenshot-8.png" alt="Screenshot-8" border="0">
# </div>

#  <div class="quote">
#     <span class="quote_text">"Trust CV!" - Kaggle participants' motto.</span>
# </div>

# <a id="references"></a>
# <div class="list">
#     <div class="title_container">
#         <h1 class="title">
#             <span class="paragraph_text"><span class="paragraph">§</span>6.</span>
#             <span>References</span>
#         </h1>
#     </div>
#     <p class="text">The following references were used by the author during the writing of this notebook.</p>
#     <ul>
#         <li><a href="https://www.kaggle.com/code/jcaliz/ps-s03e02-eda-baseline">PS S03E02: EDA + Baseline ⭐️⭐️⭐️</a></li>
#         <li><a href="https://www.kaggle.com/code/desalegngeb/heart-disease-predictions">Heart Disease Predictions</a></li>
#     </ul>
#  </div>

#  <div class="list">
#     <div class="title_container">
#         <h1 class="title">
#             <span>Author's other works</span>
#         </h1>
#     </div>
#     <p class="text">Readers are offered to take a look at some other author's works. </p>
#     <ul>
#         <li><a href="https://www.kaggle.com/code/vad13irt/plant-disease-classification">Plant Disease Classification</a></li>
#         <li><a href="https://www.kaggle.com/code/vad13irt/flowers-classification-resnet50">Flowers Classification | ResNet50</a></li>
#         <li><a href="https://www.kaggle.com/code/vad13irt/fish-semantic-segmentation-resnet34-unet">Fish Semantic Segmentation | ResNet34 + UNet</a></li>
#     </ul>
#  </div>

#  <div class="list">
#     <div class="title_container">
#         <h1 class="title">
#             <span>Credits</span>
#         </h1>
#     </div>
#     <p class="text">The author very much appreciates the help of the following people in writing this notebook.</p>
#     <ul>
#         <li>Vitaliy Irtlach (<a href="https://github.com/vitaliyirtlach" class="text_link">GitHub</a>, <a href="https://twitter.com/vitaliyirtlach" class="text_link">Twitter</a>) - for helping with designing and CSS styles.</li>
#     </ul>
#  </div>

# <div class="list">
#     <div class="title_container">
#         <h1 class="title">
#             <span>Releases</span>
#         </h1>
#     </div>
#     <ul>
#         <li>10.01.2023 - Initial version.</li>
#         <li>12.01.2023 - Minor fixing.</li>
#         <li>16.01.2023 - Added some plots, extended visualization defaults styles, remove excess lines of code.</li>
#     </ul>
#  </div>
