#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy import stats
import pandas as pd
import numpy as np
import datetime
import warnings
import copy
import gc
import os

import seaborn as sns
from cycler import cycler
import matplotlib as mpl
import matplotlib.dates as mdates
from matplotlib import pyplot as plt, animation
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from IPython.core.display import display, HTML


pd.set_option("display.max_rows", 10)
pd.set_option("display.max_columns", None)

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 
gc.enable()


# In[2]:


# notebook's styles
def css_styling():
    styles = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500&display=swap');    
    
    /* Variables */
    
    :root {
        --primary-color: #ef4444;
        --text-color: #fff;
        --text-font-size: 15px !important;
        --primary-title-border-color: linear-gradient(to right, var(--primary-color) 0%, var(--primary-color) 50%, var(--background-color)) 0 1 100% 1;
        --background-color: #202020;
        --shadow-color: #6b7280;
        --note-background-color: rgba(191,219,254, 0.1);
        --note-text-color: #93c5fd;
        --notebook-font: "Inter", sans-serif;
        --quote-background-color: rgba(148,163,184,0.25);
        --quote-text-color: #f3f4f6;
        --warning-background-color: rgba(250, 204, 21, 0.1);
        --warning-text-color: #fde047;
        --formula-font-size: 25px;
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
        color: var(--primary-color) !important;
        font-family: var(--notebook-font);
    }
    
    .paragraph {
        color: var(--primary-color) !important;
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
        background-color: var(--primary-color);
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
        color: var(--primary-color);
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

    /* Attention blocks */

    .attention {
        padding: 10px;
        border-radius: 5px;
        font-family: var(--notebook-font);
    }
    
    .note {
        background-color: var(--note-background-color);
    }
    
    .note_text {
        color: var(--note-text-color);
        font-size: var(--text-font-size);
    }
    
    .warning {
        background-color: var(--warning-background-color);
    }
    
    .warning_text {
        color: var(--warning-text-color);
        font-size: var(--text-font-size);
    }
    
    .quote {
        background-color: var(--quote-background-color);
    }
    
    .quote_text {
        color: var(--quote-text-color);
        font-size: var(--text-font-size);
    }
        
    .formula {
        font-size: var(--formula-font-size) !important;
    }
        
    """
    return HTML("<style>"+styles+"</style>")

css_styling()


# In[3]:


# colors
BACKGROUND_COLOR = "#202020"
PRIMARY_COLOR = "#ef4444"
TEXT_COLOR = "#fff"
MALE_COLOR = "#3b82f6"
FEMALE_COLOR = "#d946ef"

colors = ["#ef4444",  "#f59e0b",  "#eab308", "#22c55e", "#60a5fa", "#4f46e5", "#9333ea", "#6b7280"]
colors = sns.color_palette(colors)

palette_colors = ["#ef4444", "#7f1d1d", "#f5f5f1", "#ef4444", "#7f1d1d"]
palette = mpl.colors.LinearSegmentedColormap.from_list("", palette_colors)

# visualization styles

# matplotlib
visualization_parameters = {
    # figure styles
    "figure.facecolor": BACKGROUND_COLOR,
    
    # axes styles
    "axes.facecolor": BACKGROUND_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "axes.prop_cycle": cycler(color=colors),
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
    
    "axes.titlecolor": TEXT_COLOR,
    "axes.titlelocation": "left",
    "axes.titlepad": 7.0,
    "axes.titleweight": "bold",
    
    # spines styles
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.bottom": False,
    "axes.spines.left": False,
    
    # grid styles
    "grid.alpha": 0.5,
    "grid.color": TEXT_COLOR,
    "grid.linewidth": 1.0,
    "grid.linestyle": "-",
    
    # box plot styles
    "boxplot.boxprops.color": PRIMARY_COLOR,
    "boxplot.capprops.color": PRIMARY_COLOR,
    "boxplot.flierprops.color": PRIMARY_COLOR,
    "boxplot.flierprops.markeredgecolor": PRIMARY_COLOR,
    "boxplot.flierprops.markerfacecolor": PRIMARY_COLOR,
    "boxplot.meanprops.color": TEXT_COLOR,
    "boxplot.meanprops.markeredgecolor": TEXT_COLOR,
    "boxplot.meanprops.markerfacecolor": TEXT_COLOR,
    "boxplot.medianprops.color": TEXT_COLOR,
    "boxplot.whiskerprops.color": TEXT_COLOR,
}

boxplot_styles = {
    "capprops": {"color": TEXT_COLOR, "zorder": 2},
    "boxprops": {"edgecolor": TEXT_COLOR, "zorder": 2},
    "whiskerprops": {"color": TEXT_COLOR, "zorder": 2},
    "flierprops": {"color": TEXT_COLOR, "marker": "o", "markerfacecolor": PRIMARY_COLOR, "markeredgecolor": BACKGROUND_COLOR, "zorder": 2},
    "medianprops": {"color": TEXT_COLOR, "zorder": 2},
    "meanprops": {"color": TEXT_COLOR, "zorder": 2},
}

mpl.rcParams.update(visualization_parameters)
mpl.rc("animation", html="jshtml")

numerical_distribution_legend_template = "Mean: {mean:.2f}\nMedian: {median:.2f}\nSTD: {std:.2f}\nSkew: {skew:.2f}\nKurtosis: {kurtosis:.2f}"

# plotly
layout_styles = {
    "paper_bgcolor": BACKGROUND_COLOR,
    "mapbox_style": "carto-darkmatter",
    "margin": {"r": 0, "t": 0, "l": 0, "b": 0},
}

remove_trace_text = "<extra></extra>"

EPS = 1e-9


# In[4]:


# utilities

# visualization utilities
def hide_spines(ax, spines=["top", "right", "left", "bottom"]):
    for spine in spines:
        ax.spines[spine].set_visible(False)
        
def pair_plot(columns, data, figsize=None, triangular_form=False, size_coef=3, **plot_args):
    num_columns = len(columns)
    if figsize is None:
        width, height = num_columns * size_coef, num_columns * size_coef
        figsize = (width, height)
        
    figure, axises = plt.subplots(num_columns, num_columns, figsize=figsize)
    for i, x_column in enumerate(columns):
        for j, y_column in enumerate(columns):
            x_data = data[x_column].values
            y_data = data[y_column].values
            
            axis = axises[i, j]
            
            if j > i and triangular_form:
                axis.set_xticks([])
                axis.set_yticks([])
            else:
                axis.grid(axis="both", zorder=0)
                sns.scatterplot(x=y_data, y=x_data, zorder=2, ax=axis, **plot_args)
            
                if i == (num_columns - 1):
                    axis.set_xlabel(y_column, fontsize=13)
                    
                if j == 0:
                    axis.set_ylabel(x_column, fontsize=13)
                
            hide_spines(axis)
    
    figure.tight_layout(w_pad=1, h_pad=1)
    figure.show()

    
def categorical_vs_categorical_plot(x, y, data, **plot_args):
    data = data.groupby(x)[y].value_counts()
    cat_vs_cat = np.array([list(index) for index in data.index])
    counts = np.expand_dims(data.values, axis=-1)
    data = np.concatenate([cat_vs_cat, counts], axis=-1)
    data = pd.DataFrame(data, columns=[x, y, "count"])
    data["count"] = data["count"].astype(int)
    
    max_count = data["count"].max()
    min_count = data["count"].min()
    
    data = data.pivot(y, x, "count")
    
    plot_args["vmin"] = plot_args.get("vmin", min_count)
    plot_args["vmax"] = plot_args.get("vmax", max_count)
    plot_args["fmt"] = plot_args.get("fmt", ".0f")
    
    sns.heatmap(data, **plot_args)
    
def binary_plot(columns, data, **plot_args):
    values = data[columns].mean(axis=0).values
    x, y = columns, values
    
    orient = plot_args.get("orient", "v")
    if orient == "h":
        x, y = values, columns
        
    sns.barplot(x=x, y=y, **plot_args)
    
def seasonality_plot(column, datetime_column, data_frame, seasonality="month", **plot_args):
    data_frame = data_frame[[column, datetime_column]]
    
    if seasonality not in data_frame.columns:
        data_frame[seasonality] = getattr(data_frame[datetime_column].dt, seasonality)
        
    sns.lineplot(x=seasonality, y=column, data=data_frame, **plot_args)
    
def run_animation(images):    
    """
    Runs matplotlib animation.
    :param plt_ims: list of numpy.ndarray
    :return: animation.FuncAnimation
    
    Source:
        https://www.kaggle.com/code/sergiosaharovskiy/tps-oct-2022-viz-players-positions-animated
    """
    
    fig = plt.figure(figsize=(12,8))
    
    img = plt.imshow(images[0]) 
    plt.axis("off")
    plt.close()
    
    def frame(i):
        img.set_array(images[i])
        return [img]

    return animation.FuncAnimation(fig, frame, frames=len(images), interval=70)
    
# data utilities
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

def set_data_types(data_frame, numerical_columns=[], binary_columns=[], categorical_columns=[], datetime_columns=[], return_copy=False):
    if return_copy:
        data_frame = copy.deepcopy(data_frame)
    
    data_frame[numerical_columns] = data_frame[numerical_columns].astype(float)
    data_frame[binary_columns] = data_frame[binary_columns].astype(bool)
    data_frame[categorical_columns] = data_frame[categorical_columns].astype(str)
    
    if len(datetime_columns) > 0:
        data_frame[datetime_columns] = pd.to_datetime(data_frame[datetime_columns])
    
    return data_frame


UNITS = ("KB", "MB", "GB", "TB")

def get_memory_usage(data_frame, unit="MB"):
    memory_usage = data_frame.memory_usage().sum()
    
    if unit not in UNITS:
        raise ValueError(f"{unit} is not supported value for `unit`. Please, choose one of {UNITS}.")
        
    memory_usage /= (1024 ** (UNITS.index(unit) + 1))
    memory_usage = round(memory_usage, 2)
    
    return memory_usage

def reduce_memory_usage(data_frame, return_copy=True):
    """
    References:
        https://github.com/a-milenkin/ML_tricks_and_hacks/blob/main/dataframe_optimize.py
    """
    
    if return_copy:
        data_frame = copy.deepcopy(data_frame)
        
    for column, dtype in zip(data_frame.columns, data_frame.dtypes):
        dtype = str(dtype)
        column_data = data_frame[column]
        if any([_ in dtype for _ in ("int", "float")]):
            column_min, column_max = column_data.min(), column_data.max()
            if "int" in dtype:
                if (column_min > np.iinfo(np.int8).min) and (column_max < np.iinfo(np.int8).max):
                    column_data = column_data.astype(np.int8)
                elif (column_min > np.iinfo(np.int16).min) and (column_max < np.iinfo(np.int16).max):
                    column_data = column_data.astype(np.int16)
                elif (column_min > np.iinfo(np.int32).min) and (column_max < np.iinfo(np.int32).max):
                    column_data = column_data.astype(np.int32)
                else:
                    column_data = column_data.astype(np.int64)
            else:
                if (column_min > np.finfo(np.float16).min) and (column_max < np.finfo(np.float16).max):
                    column_data = column_data.astype(np.float16)
                elif (column_min > np.finfo(np.float32).min) and (column_max < np.finfo(np.float32).max):
                    column_data = column_data.astype(np.float32)
                else:
                    column_data = column_data.astype(np.float64)
        elif dtype == "object":
            column_data = column_data.astype("category")
        elif "datetime" in dtype:
            column_data = pd.to_datetime(column_data)
            
        data_frame[column] = column_data
        
    return data_frame

def get_dtypes(data_frame):
    dtypes = data_frame.dtypes
    dtypes = pd.DataFrame(dtypes).reset_index(drop=False)
    dtypes = dtypes.rename(columns={"index": "column", 0: "dtype"})
    
    return dtypes

def get_cardinality(data_frame):
    num_unique_values = data_frame.nunique()
    num_unique_values = pd.DataFrame(num_unique_values).reset_index(drop=False)
    num_unique_values = num_unique_values.rename(columns={"index": "column", 0: "num_unique_values"})
    
    return num_unique_values

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


def get_cramers_v_correlation(data_frame, categorical_columns):
    if categorical_columns is None:
        categorical_columns_mask = [dtype in ("object", "category") for dtype in data_frame.dtypes]
        categorical_columns = data_frame.columns[categorical_columns_mask]
    
    cat_data_frame = data_frame[categorical_columns]
    
    rows = []
    for x in cat_data_frame:
        col = []
        for y in cat_data_frame :
            try:
                cramers = compute_cramers_v_function(cat_data_frame[x], cat_data_frame[y]) 
            except:
                cramers = 0.0
                
            col.append(round(cramers,2))
        rows.append(col)
        
    cramers_results = np.array(rows)
    cramers_v_correlation = pd.DataFrame(cramers_results, columns=cat_data_frame.columns, index=cat_data_frame.columns)
    
    return cramers_v_correlation


# <div class="title_container">
#     <h1 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>1.</span>
#         <span>Competition</span>
#     </h1>
# </div>
# <p class="text">
# <i>Learning is meant to be fun, which is where game-based learning comes in. This educational approach allows students to engage with educational content inside a game framework, making it enjoyable and dynamic. Although game-based learning is being used in a growing number of educational settings, there are still a limited number of open datasets available to apply data science and learning analytic principles to improve game-based learning.<br><br>
# Most game-based learning platforms do not sufficiently make use of knowledge tracing to support individual students. Knowledge tracing methods have been developed and studied in the context of online learning environments and intelligent tutoring systems. But there has been less focus on knowledge tracing in educational games.<br><br>
# Competition host Field Day Lab is a publicly-funded research lab at the Wisconsin Center for Educational Research. They design games for many subjects and age groups that bring contemporary research to the public, making use of the game data to understand how people learn. Field Day Lab's commitment to accessibility ensures all of its games are free and available to anyone. The lab also partners with ​​​nonprofits like The Learning Agency Lab, which is focused on developing science of learning-based tools and programs for the social good.<br><br>
# If successful, you'll enable game developers to improve educational games and further support the educators who use these games with dashboards and analytic tools. In turn, we might see broader support for game-based learning platforms.</i> - <a href="https://www.kaggle.com/competitions/predict-student-performance-from-game-play/overview/description" class="text_link">Context</a>
# </p><br>
# <div class="image_container">
#     <img src="https://content.pbswisconsineducation.org/wp-content/uploads/2021/06/25221325/jw-facebook-playthegame.png">
#     <span class="text"></span>
# </div>

# <div class="title_container border-0">
#     <h2 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>1.1.</span>
#         <span>Task</span>
#     </h2>
# </div>
# <p class="text">
# <i>The goal of this competition is to predict student performance during game-based learning in real-time. You'll develop a model trained on one of the largest open datasets of game logs.<br><br>
# Your work will help advance research into knowledge-tracing methods for game-based learning. You'll be supporting developers of educational games to create more effective learning experiences for students.
# </i> - <a href="https://www.kaggle.com/competitions/predict-student-performance-from-game-play/overview" class="text_link">Goal of the Competition</a>
# </p>

# <div class="title_container border-0">
#     <h2 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>1.2.</span>
#         <span>Metric</span>
#     </h2>
# </div>
# <p class="text">
#     <i>Submissions will be evaluated based on their F1 score.</i> - <a href="https://www.kaggle.com/competitions/predict-student-performance-from-game-play/overview/evaluation" class="text_link">Evaluation</a><br>
# </p>
# <p class="text">
# <i>
#     F-score or F-measure is a measure of a test's accuracy. It is calculated from the precision and recall of the test, where the precision is the number of true positive results divided by the number of all positive results, including those not identified correctly, and the recall is the number of true positive results divided by the number of all samples that should have been identified as positive. Precision is also known as positive predictive value, and recall is also known as sensitivity in diagnostic binary classification.<br><br>
# The F1 score is the harmonic mean of the precision and recall. The more generic $ F_{\beta } $ score applies additional weights, valuing one of precision or recall more than the other.<br><br>
# The highest possible value of an F-score is 1.0, indicating perfect precision and recall, and the lowest possible value is 0, if either precision or recall are zero.
# </i> - <a href="https://en.wikipedia.org/wiki/F-score" class="text_link">Wikipedia</a><br>
# </p>
# <p class="text formula">
# $ F_1 = \frac{2}{recall^{-1} + precision^{-1}} = 2\frac{precision \cdot recall}{precision + recall} = \frac{2tp}{2tp + fp + fn} $
# </p>

# <div class="title_container border-0">
#     <h2 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>1.3.</span>
#         <span>Efficiency Prize</span>
#     </h2>
# </div>
# <p class="text">
# <i>We are hosting a second track that focuses on model efficiency, because highly accurate models are often computationally heavy. Such models have a stronger carbon footprint and frequently prove difficult to utilize in real-world educational contexts. We hope to use these models to help educational organizations, which have limited computational capabilities.</i> - <a href="https://www.kaggle.com/competitions/predict-student-performance-from-game-play/overview/efficiency-prize-evaluation" class="text_link">Efficiency Prize</a>
# </p>
# <p class="text">
#     <i>We compute a submission's efficiency score by:</i>
# </p>
# <p class="text formula">
# $ \text{Efficiency} = \frac{1}{ \text{Benchmark} - \max\text{F1} }\text{F1} + \frac{1}{32400}\text{RuntimeSeconds} $
# </p>
# <p class="text">
#     <i>where $ F1 $ is the submission's score on the main competition metric, $ Benchmark $ is the score of the benchmark sample_submission.csv, $ maxF1 $ is the maximum of all submissions on the Private Leaderboard, and $ RuntimeSeconds $ is the number of seconds it takes for the submission to be evaluated. The objective is to minimize the efficiency score.</i> - <a href="https://www.kaggle.com/competitions/predict-student-performance-from-game-play/overview/efficiency-prize-evaluation" class="text_link">Evaluation Metric</a>
# </p>
# <div class="attention note">
#     <span class="note_text"><i>Note: this competition is aimed at producing models that are small and lightweight. We have introduced compute constraints to match - your VMs will have only 2 CPUs, 8GB of RAM, and no GPU available. You will still have a maximum of 9 hours to complete the task, but between the constraints and the efficiency prize there will be some interesting sub-problems to solve.</i></span>
# </div>

# <div class="title_container">
#     <h1 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>3.</span>
#         <span>Data</span>
#     </h1>
# </div>
# <p class="text">
# <i>This competition uses the Kaggle's time series API. Test data will be delivered in groupings that do not allow access to future data. The objective of this competition is to use time series data generated by an online educational game to determine whether players will answer questions correctly. There are three question checkpoints (level 4, level 12, and level 22), each with a number of questions. At each checkpoint, you will have access to all previous test data for that section.</i> - <a href="https://www.kaggle.com/competitions/predict-student-performance-from-game-play/data" class="text_link">Dataset Description</a><br><br>
# <i>You have access to the training data and labels. There are 18 questions for each sessions - you are not given the answers, but are simply told whether the user for a particular session answered each question correctly.<br><br>
# When you are ready to predict, use the sample notebook to iterate over the test data, which is split as described above and served up as Pandas dataframes. Make your predictions for each group of questions - at the end of this process a submission.csv file will have been created for you. Simply submit your notebook.</i> - <a href="https://www.kaggle.com/competitions/predict-student-performance-from-game-play/data" class="text_link">What files do I need?
# </a><br><br>
# <i>The training columns are as listed below. The label rows are identified with `session_id_question #`. Each session will have 18 rows, representing 18 questions.</i> - <a href="https://www.kaggle.com/competitions/predict-student-performance-from-game-play/data" class="text_link">What should I expect the data format to be?</a>
# </p>

# <div class="title_container border-0">
#     <h2 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>3.1.</span>
#         <span>Columns</span>
#     </h2>
# </div>

# <div class="list">
#     <ul>
#         <li><b>session_id</b> - the ID of the session the event took place in</li>
#         <li><b>index</b> - the index of the event for the session</li>
#         <li><b>elapsed_time</b> - how much time has passed (in milliseconds) between the start of the session and when the event was recorded</li>
#         <li><b>event_name</b> - the name of the event type</li>
#         <li><b>name</b> - the event name (e.g. identifies whether a notebook_click is is opening or closing the notebook)</li>
#         <li><b>level</b> - what level of the game the event occurred in (0 to 22)</li>
#         <li><b>page</b> - the page number of the event (only for notebook-related events)</li>
#         <li><b>room_coor_x</b> - the coordinates of the click in reference to the in-game room (only for click events)</li>
#         <li><b>room_coor_y</b> - the coordinates of the click in reference to the in-game room (only for click events)</li>
#         <li><b>screen_coor_x</b> - the coordinates of the click in reference to the player’s screen (only for click events)</li>
#         <li><b>screen_coor_y</b> - the coordinates of the click in reference to the player’s screen (only for click events)</li>
#         <li><b>hover_duration</b> - how long (in milliseconds) the hover happened for (only for hover events)</li>
#         <li><b>text</b> - the text the player sees during this event</li>
#         <li><b>fqid</b> - the fully qualified ID of the event</li>
#         <li><b>room_fqid</b> - the fully qualified ID of the room the event took place in</li>
#         <li><b>text_fqid</b> - the fully qualified ID of the</li>
#         <li><b>fullscreen</b> - whether the player is in fullscreen mode</li>
#         <li><b>hq</b> - whether the game is in high-quality</li>
#         <li><b>music</b> - whether the game music is on or off</li>
#         <li><b>level_group</b> - which group of levels - and group of questions - this row belongs to (0-4, 5-12, 13-22)</li>
#     </ul>
# </div>

# <div class="title_container border-0">
#     <h2 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>3.2.</span>
#         <span>Files</span>
#     </h2>
# </div>

# In[5]:


directory = "/kaggle/input/predict-student-performance-from-game-play"
train_path = "/kaggle/input/predict-student-performance-from-game-play/train.csv"
train_labels_path = "/kaggle/input/predict-student-performance-from-game-play/train_labels.csv"
test_path = "/kaggle/input/predict-student-performance-from-game-play/test.csv"
sample_submission_path = "/kaggle/input/predict-student-performance-from-game-play/sample_submission.csv"


# In[6]:


train = pd.read_csv(train_path)
train_labels = pd.read_csv(train_labels_path)
test = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_submission_path)


# <p class="text">
#     <b>train.csv</b> - the training set
# </p>

# In[7]:


train


# <p class="text">
#     <b>train_labels.csv</b> - correct value for all 18 questions for each session in the training set
# </p>

# In[8]:


train_labels


# <p class="text">
#     <b>test.csv</b> - the test set
# </p>

# In[9]:


test


# <p class="text">
#     <b>sample_submission.csv</b> - a sample submission file in the correct format
# </p>

# In[10]:


sample_submission


# <div class="title_container border-0">
#     <h2 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>3.3.</span>
#         <span>Data types</span>
#     </h2>
# </div>

# In[11]:


dtypes = get_dtypes(train)

with pd.option_context("display.max_rows", None):
    display(dtypes)


# <div class="title_container border-0">
#     <h2 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>3.4.</span>
#         <span>Cardinality</span>
#     </h2>
# </div>

# In[12]:


cardinality = get_cardinality(train)

with pd.option_context("display.max_rows", None):
    display(cardinality)


# <div class="title_container border-0">
#     <h2 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>3.5.</span>
#         <span>Duplicates</span>
#     </h2>
# </div>
# <p class="text">
# Sometimes some data are duplicated and because of duplicates, the model becomes slightly overfitted, because, roughly speaking, it looks at the same samples during training, i.e. don't observe other "cases".
# </p>

# In[13]:


exclude_columns = ("index")
columns = [column for column in train.columns if column not in exclude_columns]
duplicates = train[train.duplicated(subset=columns)]

num_samples = len(train)
num_duplicates = len(duplicates)
print(f"Number of duplicates: {num_duplicates}")

duplicates_percentage = (num_duplicates / num_samples) * 100
duplicates_percentage = round(duplicates_percentage, 2)
print(f"Percentage of duplicates: {duplicates_percentage}%")
print()

duplicates


# <div class="title_container border-0">
#     <h2 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>3.6.</span>
#         <span>Missing values</span>
#     </h2>
# </div>
# <p class="text">
#     <i>Missing data, or missing values, occur when no data value is stored for the variable in an observation. Missing data are a common occurrence and can have a significant effect on the conclusions that can be drawn from the data.</i> - <a href="https://en.wikipedia.org/wiki/Missing_data#:~:text=In%20statistics%2C%20missing%20data%2C%20or,be%20drawn%20from%20the%20data" class="text_link">Wikipedia</a>
# </p>
# <p class="text">
# There are many techniques to produce the imputation/filling of missing values. The basic technique is to use mean or median for missed numerical values and mode (the most frequent value) for missed categorical values respectively. More powerful and accurate approaches are k Nearest Neighbors (KNN) or even training additional model(s), where the target(s) is (are) the missed column(s).
# </p>
# <p class="text">
# The following papers provide more information about missing values and imputation techniques:
# </p>
# <div class="list">
#     <ul>
#         <li><a href="https://scikit-learn.org/stable/modules/impute.html">Imputation of missing values</a></li>
#         <li><a href="https://towardsdatascience.com/6-different-ways-to-compensate-for-missing-values-data-imputation-with-examples-6022d9ca0779">6 Different Ways to Compensate for Missing Values In a Dataset (Data Imputation with examples)</a></li>
#     </ul>
# </div>
# <div class="attention note">
#     <span class="note_text">Tree-based models can handle missing values.</span>
# </div>

# In[14]:


missing_values = get_missing_values(train, stat="percentage")

figure = plt.figure(figsize=(20, 7))
axis = figure.add_subplot()
axis.grid(axis="y", zorder=0)
sns.barplot(x="column", y="missing_values", data=missing_values, color=colors[0], edgecolor=TEXT_COLOR, label="Train dataset", alpha=1.0, linewidth=2.0, ax=axis, zorder=2)
axis.xaxis.set_tick_params(labelsize=14, rotation=40)
axis.set_xlabel("column", fontsize=17)
axis.yaxis.set_tick_params(labelsize=12)
axis.set_ylabel("missing values (%)", fontsize=14)
axis.set_yticks(range(0, 110, 10))
figure.show()


# <p class="text">
# Some columns don't have any information, i.e the columns are fully missed, so let's drop them from the future analysis. 
# </p>

# In[15]:


missed_columns = ["fullscreen", "hq", "music", "page", "hover_duration"]
train = train.drop(missed_columns, axis=1)
test = test.drop(missed_columns, axis=1)


# <div class="title_container">
#     <h1 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>4.</span>
#         <span>Optimization</span>
#     </h1>
# </div>
# <p class="text">
#   Optimization as well as other steps of project development is the necessary intermediate step. Due to proper optimization productivity rate is increased, hence allows making experiments much faster, especially in Data Science competitions.<br><br>
# There are a lot of techniques to optimize data pre-processing (e.g Parallelling and reducing data types) and training models (e.g. Intel® Extension). Also, operations can be applied on GPU machines with help of the RAPIDS library, which is made by NVIDIA. 
# </p>

# In[16]:


unit = "MB"

memory_usage = get_memory_usage(train, unit=unit)
print(f"Memory usage of train dataset is {memory_usage} {unit}.")

train = reduce_memory_usage(train, return_copy=False)
optimized_memory_usage = get_memory_usage(train, unit=unit)
print(f"Memory usage of optimized train dataset is {optimized_memory_usage} {unit}.")

memory_usage_difference = (memory_usage - optimized_memory_usage)
memory_usage_percentage_difference = (memory_usage_difference / memory_usage) * 100
memory_usage_percentage_difference = round(memory_usage_percentage_difference, 2)
print(f"Memory usage of train dataset was decreased by {memory_usage_percentage_difference}%!")

print()

memory_usage = get_memory_usage(test, unit=unit)
print(f"Memory usage of test dataset is {memory_usage} {unit}.")

test = reduce_memory_usage(test, return_copy=False)
optimized_memory_usage = get_memory_usage(test, unit=unit)
print(f"Memory usage of optimized test dataset is {optimized_memory_usage} {unit}.")

memory_usage_difference = (memory_usage - optimized_memory_usage)
memory_usage_percentage_difference = (memory_usage_difference / memory_usage) * 100
memory_usage_percentage_difference = round(memory_usage_percentage_difference, 2)
print(f"Memory usage of test dataset was decreased by {memory_usage_percentage_difference}%!")

gc.collect()


# <div class="title_container">
#     <h1 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>5.</span>
#         <span>Exploratory Data Analysis</span>
#     </h1>
# </div>
# <p class="text">
# Exploratory Data Analysis (EDA) is an integral and the most important stage during developing any Data Science projects. Exploratory Data Analysis can give important observations, relations and even ideas for further defining hypotheses and testing, which can be very beneficial for model development. From extracted observations, we can decide what metric, pre-processing, feature engineering, validation strategy, post-processing, model, etc, we should use, which potentially can improve model generalization, hence, improving of predictive ability. 
# </p>

# <div class="title_container border-0">
#     <h2 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>5.1.</span>
#         <span>Target Variable Distribution</span>
#     </h2>
# </div>

# In[17]:


column = "correct"

figure = plt.figure(figsize=(10, 7))
axis = figure.add_subplot()
axis.grid(axis="y", zorder=0)
sns.countplot(x=column, data=train_labels, palette=colors,  edgecolor=TEXT_COLOR, linewidth=2.0, ax=axis, zorder=2)
axis.xaxis.set_tick_params(size=0, labelsize=14)
axis.set_xlabel(column, fontsize=17)
axis.yaxis.set_tick_params(size=0, labelsize=12)
axis.set_ylabel("count", fontsize=14)
axis.tick_params(axis="both", which="both", length=0)
axis.set_ylim(1)
figure.show()


# <p class="text">There is slight class imbalance. Because of imbalance model will be biased to more balanced classes, i.e will be overfitted and won’t generalize well for imbalance classes. There are tecniques to slightly smooth that problem: class-wise weights, oversampling or under-sampling, extracting external or generating synthetic data, and apply augmentations during training.</p>

# <div class="title_container border-0">
#     <h2 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>5.2.</span>
#         <span>Session-wise statistics</span>
#     </h2>
# </div>

# In[18]:


grouped_session = train.groupby("session_id")
session_wise_statistics = pd.DataFrame({
    "num_events": grouped_session.size(),
    "elapsed_time": grouped_session["elapsed_time"].mean(),
})


# In[19]:


session_wise_columns = session_wise_statistics.columns
num_columns = len(session_wise_columns)

columns = 2
rows = (num_columns // columns)  + 1

figure = plt.figure(figsize=(17, 10))
for index, column in enumerate(session_wise_columns):
    data = session_wise_statistics[column].values
    
    axis = figure.add_subplot(rows, columns, index+1)
    axis.grid(axis="both", zorder=0)
    sns.kdeplot(x=data, fill=True, alpha=1.0, color=colors[0],  edgecolor=TEXT_COLOR,  linewidth=2, zorder=2, ax=axis)
    axis.xaxis.set_tick_params(labelsize=12)
    axis.set_xlabel(column, fontsize=13, labelpad=7)
    ylabel_text = "density"  if (index % columns) == 0 else ""
    axis.set_ylabel(ylabel_text, fontsize=13)
    axis.yaxis.set_tick_params(labelsize=10)

figure.tight_layout()
figure.show()


# In[20]:


non_outliers_mask = (session_wise_statistics["num_events"] < 3000) & (session_wise_statistics["elapsed_time"] < 0.02*1e8)
non_outliers_session_wise_statistics = session_wise_statistics[non_outliers_mask]
session_wise_columns = non_outliers_session_wise_statistics.columns
num_columns = len(session_wise_columns)

columns = 2
rows = (num_columns // columns)  + 1

figure = plt.figure(figsize=(17, 10))
for index, column in enumerate(session_wise_columns):
    data = non_outliers_session_wise_statistics[column].values
    
    axis = figure.add_subplot(rows, columns, index+1)
    axis.grid(axis="both", zorder=0)
    sns.kdeplot(x=data, fill=True, alpha=1.0, color=colors[0],  edgecolor=TEXT_COLOR,  linewidth=2, zorder=2, ax=axis)
    axis.xaxis.set_tick_params(labelsize=12)
    axis.set_xlabel(column, fontsize=13, labelpad=7)
    ylabel_text = "density"  if (index % columns) == 0 else ""
    axis.set_ylabel(ylabel_text, fontsize=13)
    axis.yaxis.set_tick_params(labelsize=10)

figure.suptitle("Session-wise statistics without outliers", x=0.25, fontsize=25, fontweight="bold")
figure.tight_layout()
figure.show()


# <div class="title_container border-0">
#     <h2 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>5.3.</span>
#         <span>Numerical columns distributions</span>
#     </h2>
# </div>

# In[21]:


common_numerical_columns = ["room_coor_x", "room_coor_y", "screen_coor_x", "screen_coor_y"]
num_common_numerical_columns_columns = len(common_numerical_columns)

columns = 2
rows = (num_common_numerical_columns_columns // columns)  + 1

figure = plt.figure(figsize=(17, 10))
for index, column in enumerate(common_numerical_columns):
    data = train[column].values
    
    axis = figure.add_subplot(rows, columns, index+1)
    axis.grid(axis="both", zorder=0)
    sns.kdeplot(x=data, fill=True, alpha=1.0, color=colors[0],  edgecolor=TEXT_COLOR,  linewidth=2.0, zorder=2, ax=axis)
    axis.xaxis.set_tick_params(labelsize=12)
    axis.set_xlabel(column, fontsize=13, labelpad=7)
    ylabel_text = "density"  if (index % columns) == 0 else ""
    axis.set_ylabel(ylabel_text, fontsize=13)
    axis.yaxis.set_tick_params(labelsize=10)

figure.tight_layout(w_pad=2, h_pad=2)
figure.show()


# <div class="title_container border-0">
#     <h2 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>5.4.</span>
#         <span>Categorical columns distributions</span>
#     </h2>
# </div>

# In[22]:


common_categorical_columns = ["event_name", "name", "level", "level_group", "room_fqid"]
num_common_categorical_columns = len(common_categorical_columns)

columns = 2
rows = (num_common_categorical_columns // columns)  + 1
rotate_ticks_threshold = 5

figure = plt.figure(figsize=(15, 17))
figure_colors = []
for index, column in enumerate(common_categorical_columns):
    data = train[column].values
    
    axis_colors = colors
    if index < len(figure_colors):
        axis_colors = figure_colors[index]
    
    axis = figure.add_subplot(rows, columns, index+1)
    axis.grid(axis="y", zorder=0)
    sns.countplot(x=data, fill=True, palette=axis_colors, alpha=1.0, edgecolor=TEXT_COLOR, linewidth=2.0, zorder=2, ax=axis)
    axis.xaxis.set_tick_params(labelsize=12)
    axis.set_xlabel(column, fontsize=13)
    ylabel_text = "count"  if (index % columns) == 0 else ""
    axis.set_ylabel(ylabel_text, fontsize=13)
    axis.yaxis.set_tick_params(labelsize=10)
    
    num_categories = len(set(data))
    if num_categories > rotate_ticks_threshold:
        axis.tick_params(axis="x", labelrotation=90)
    
    axis.set_ylim(1)
    
figure.tight_layout(h_pad=1, w_pad=2)
figure.show()


# <div class="title_container border-0">
#     <h2 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>5.5.</span>
#         <span>Level vs elapsed time</span>
#     </h2>
# </div>

# In[23]:


level_vs_elapsed_time = pd.DataFrame(train.groupby("level").agg({"elapsed_time": "mean"}))
level_vs_elapsed_time = level_vs_elapsed_time.reset_index(drop=False)
level_vs_elapsed_time.columns = ["level", "elapsed_time"]
level_vs_elapsed_time = level_vs_elapsed_time.sort_values(by="level", ascending=True)


# In[24]:


x, y = level_vs_elapsed_time["level"].values, level_vs_elapsed_time["elapsed_time"]
min_x, max_x = np.min(x), np.max(x)

figure = plt.figure(figsize=(20, 7))
axis = figure.add_subplot()
axis.grid(axis="both", zorder=0)
sns.lineplot(x=x, y=y, color=TEXT_COLOR, linewidth=2, alpha=1.0, ax=axis, zorder=2)
axis.yaxis.set_tick_params(labelsize=14)
axis.set_ylabel("elapsed_time", fontsize=15)
axis.set_xlabel("level", fontsize=15)
axis.xaxis.set_tick_params(labelsize=12)
axis.yaxis.set_tick_params(labelsize=12)
axis.fill_between(x, y, color=PRIMARY_COLOR, edgecolor=TEXT_COLOR, alpha=1.0, zorder=2)
axis.set_xticks(range(min_x, max_x+1, 1))
axis.set_xlim(min_x, max_x)
axis.set_ylim(0.0)
figure.show()


# <p class="text">
# We are observing trend/correlation: more level => more time required to complete. 
# </p>

# <div class="title_container border-0">
#     <h2 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>5.6.</span>
#         <span>Text length (number of words) distribution</span>
#     </h2>
# </div>

# In[25]:


texts = train[~train["text"].isna()]["text"].values
num_words = np.array([len(str(text).split()) for text in texts])


# In[26]:


figure = plt.figure(figsize=(15, 5))
axis = figure.add_subplot()
axis.grid(axis="both", zorder=0)
sns.countplot(x=num_words, fill=True, alpha=1.0, color=colors[0],  edgecolor=TEXT_COLOR,  linewidth=2.0, zorder=2, ax=axis)
axis.xaxis.set_tick_params(labelsize=12)
axis.set_xlabel("num_words", fontsize=13, labelpad=7)
axis.set_ylabel("count", fontsize=13)
axis.yaxis.set_tick_params(labelsize=10)
axis.set_ylim(EPS)
    
figure.tight_layout()
figure.show()


# <div class="title_container border-0">
#     <h2 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>5.7.</span>
#         <span>Level intersections</span>
#     </h2>
# </div>

# In[27]:


level_intersect_columns = ["room_fqid", "event_name", "name"]
num_level_intersect_columns = len(level_intersect_columns)

columns = 1
rows = (num_level_intersect_columns // columns) + 1
figure = plt.figure(figsize=(17, 25))
for index, column in enumerate(level_intersect_columns):
    axis = figure.add_subplot(rows, columns, index+1)
    categorical_vs_categorical_plot(x="level", y=column, data=train, annot=True, linecolor=TEXT_COLOR, linewidth=2.0, cbar=False, cmap=palette, ax=axis)
    axis.xaxis.set_tick_params(labelsize=12)
    xlabel_text = "level" if index >= (num_level_intersect_columns - columns) else "" 
    axis.set_xlabel(xlabel_text, fontsize=13)
    axis.yaxis.set_tick_params(labelsize=10)
    axis.set_ylabel(column, fontsize=13)
    
figure.tight_layout(w_pad=2, h_pad=2)
figure.show()


# <div class="title_container border-0">
#     <h2 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>5.8.</span>
#         <span>Session</span>
#     </h2>
# </div>
# <p class="text">
# Let's take a view on the one game session. 
# </p>

# In[28]:


def visualize_session(session, plot_directory=None, plot_filename_format="{session_id}_{index}.png"):
    if plot_directory is None:
        plot_directory = str(session["session_id"].values[0])
        if not os.path.exists(plot_directory):
            os.mkdir(plot_directory)
    
    # screen
    screen_x_min, screen_y_min = session["screen_coor_x"].min(), session["screen_coor_y"].min()
    screen_x_max, screen_y_max = session["screen_coor_x"].max(), session["screen_coor_y"].max()
    screen_width, screen_height = screen_x_max - screen_x_min, screen_y_max - screen_y_min
    screen_padding = 25
    
    # room
    room_x_min, room_y_min = session["room_coor_x"].min(), session["room_coor_y"].min()
    room_x_max, room_y_max = session["room_coor_x"].max(), session["room_coor_y"].max()
    room_width, room_height = room_x_max - room_x_min, room_y_max - room_y_min
    
    plot_pathes = []
    for index, sample in session.iterrows():
        figure = plt.figure(figsize=(12, 10))
        axis = figure.add_subplot()
        
        # environment
        screen_rectangle = mpl.patches.Rectangle(xy=(screen_x_min, screen_y_min), width=screen_width, height=screen_height, edgecolor="#fff", label="Screen", color=colors[-1])
        axis.add_patch(screen_rectangle)
        
        room_rectangle = mpl.patches.Rectangle(xy=(room_x_min, room_y_min), width=room_width, height=room_height, edgecolor="#fff", label="Room", color=colors[2])
        axis.add_patch(room_rectangle)
        
        axis.set_xlim(screen_x_min - screen_padding, screen_x_max + screen_padding)
        axis.set_ylim(screen_y_min - screen_padding, screen_y_max + screen_padding)
        axis.set(xticks=[], yticks=[])
        axis.legend()
        
        # events
        room_x, room_y = sample["room_coor_x"], sample["room_coor_y"]
        screeen_x, screeen_y = sample["screen_coor_x"], sample["screen_coor_y"]
        axis.plot(room_x, room_y, color=colors[0], marker="o", markersize=5)
        axis.plot(screeen_x, screeen_y, color=colors[3], marker="o", markersize=5)
        axis.set_title("FQID: {fqid}, Level: {level}, Index: {index}, Event name: {event_name}".format(**sample.to_dict()), fontsize=20)
        
        # saving
        plot_filename = plot_filename_format.format(**sample.to_dict())
        plot_path = os.path.join(plot_directory, plot_filename)
        figure.savefig(plot_path)
        plt.close(figure) 
        
        plot_pathes.append(plot_path)
    
    plt_ims = [plt.imread(plot_path) for plot_path in plot_pathes]
    return run_animation(plt_ims)


# In[29]:


session_id = 20110112402567744
session = train[train["session_id"] == session_id].iloc[:50]

visualize_session(session)


# <div class="title_container">
#     <h1 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>6.</span>
#         <span>Feature Engineering</span>
#     </h1>
# </div>

# <div class="title_container border-0">
#     <h2 class="title">
#         <span class="paragraph_text"><span class="paragraph">§</span>6.1.</span>
#         <span>session_id Reverse Engineering</span>
#     </h2>
# </div>

# In[30]:


def extract_date_from_session_id(data_frame, return_copy=False):
    """
    https://www.kaggle.com/code/pdnartreb/session-id-reverse-engineering
    """
    
    if return_copy:
        data_frame = copy.deepcopy(data_frame)
    
    data_frame["year"] = data_frame["session_id"].apply(lambda x: int(str(x)[:2]))
    data_frame["month"] = data_frame["session_id"].apply(lambda x: int(str(x)[2:4]) + 1)
    data_frame["weekday"] = data_frame["session_id"].apply(lambda x: int(str(x)[4:6]))
    data_frame["hour"] = data_frame["session_id"].apply(lambda x: int(str(x)[6:8]))
    data_frame["minute"] = data_frame["session_id"].apply(lambda x: int(str(x)[8:10]))
    data_frame["second"] = data_frame["session_id"].apply(lambda x: int(str(x)[10:12]))
    data_frame["ms"] = data_frame["session_id"].apply(lambda x: int(str(x)[12:15]))
    data_frame["?"] = data_frame["session_id"].apply(lambda x: int(str(x)[15:17]))
    
    return data_frame


# In[31]:


train = extract_date_from_session_id(train)


# In[32]:


datetime_columns = ["year", "month", "weekday", "hour"]
num_datetime_columns = len(datetime_columns)

columns = 2
rows = int(num_datetime_columns // columns) + 1

figure = plt.figure(figsize=(15, 12))
for index, column in enumerate(datetime_columns):
    column_grouped = train.groupby(column).agg({"elapsed_time": "mean"})
    column_grouped = column_grouped.reset_index(drop=False)
    column_grouped.columns = [column, "elapsed_time"]
    
    x, y = column_grouped[column].values, column_grouped["elapsed_time"]
    min_x, max_x = np.min(x), np.max(x)

    axis = figure.add_subplot(rows, columns, index+1)
    axis.grid(axis="both", zorder=0)
    sns.lineplot(x=x, y=y, color=TEXT_COLOR, linewidth=2.0, alpha=1.0, ax=axis, zorder=2)
    axis.yaxis.set_tick_params(labelsize=14)
    axis.set_ylabel("elapsed_time", fontsize=15)
    axis.set_xlabel(column, fontsize=15)
    axis.xaxis.set_tick_params(labelsize=12)
    axis.yaxis.set_tick_params(labelsize=12)
    axis.fill_between(x, y, color=PRIMARY_COLOR, edgecolor=TEXT_COLOR, alpha=1.0, zorder=2)
    axis.set_xticks(range(min_x, max_x+1, 1))
    axis.set_xlim(min_x, max_x)
    axis.set_ylim(0.0)

figure.tight_layout()
figure.show()


# <p class="text">
#   The game is mostly played at the beginning of the school year (August, September, October, November, and December months) and during the school day (from 8 a.m. to 15p.m., then increasingly are appeared in the evening (maybe children do homework). Each year, the mean gaming time is increasing linearly.
# </p>
