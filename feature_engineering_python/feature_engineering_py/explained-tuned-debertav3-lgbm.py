#!/usr/bin/env python
# coding: utf-8

# # Explained Tuned Debertav3+LGBM 📈
# 
# ## Introduction 🌟
# Welcome to this Jupyter notebook developed for the CommonLit - Evaluate Student Summaries
# ! This notebook is designed to help you participate in the competition and make Automatically assess summaries written by students in grades 3-12.
# 
# ### Inspiration and Credits 🙌
# This notebook is inspired by the work of dangnguyen1997, available at [this Kaggle project](https://www.kaggle.com/code/dangnguyen97/tuned-debertav3-lgbm). i extend our gratitude to dangnguyen1997 for sharing their insights and code.
# 
# 🌟 Explore my profile and other public projects, and don't forget to share your feedback! 
# 👉 [Visit my Profile](https://www.kaggle.com/zulqarnainali) 👈
# 
# 🙏 Thank you for taking the time to review my work, and please give it a thumbs-up if you found it valuable! 👍
# 
# ## Purpose 🎯
# The primary purpose of this notebook is to:
# - Load and preprocess the competition data 📁
# - Engineer relevant features for model training 🏋️‍♂️
# - Train predictive models to make target variable predictions 🧠
# - Submit predictions to the competition environment 📤
# 
# ## Notebook Structure 📚
# This notebook is structured as follows:
# 1. **Data Preparation**: In this section, we load and preprocess the competition data.
# 2. **Feature Engineering**: We generate and select relevant features for model training.
# 3. **Model Training**: We train machine learning models on the prepared data.
# 4. **Prediction and Submission**: We make predictions on the test data and submit them for evaluation.
# 5. **Conclusion**: We summarize the key findings and results.
# 
# ## How to Use 🛠️
# To use this notebook effectively, please follow these steps:
# 1. Ensure you have the competition data and environment set up.
# 2. Execute each cell sequentially to perform data preparation, feature engineering, model training, and prediction submission.
# 3. Customize and adapt the code as needed to improve model performance or experiment with different approaches.
# 
# **Note**: Make sure to replace any placeholder paths or configurations with your specific information.
# 
# ## Acknowledgments 🙏
# I acknowledge the OThe Learning Agency Lab organizers for providing the dataset and the competition platform.
# 
# Let's get started! Feel free to reach out if you have any questions or need assistance along the way.
# 👉 [Visit my Profile](https://www.kaggle.com/zulqarnainali) 👈

# ## 📚 Importing Libraries

# **Explaination**:
# 
# 
# . `!pip install "/kaggle/input/autocorrect/autocorrect-2.6.1.tar"`:
#    - This line uses the `!` symbol to execute a shell command within a Jupyter Notebook or similar environment.
#    - It invokes the `pip install` command, which is a package manager for Python, used to install Python packages.
#    - The package it is trying to install is specified with the path "/kaggle/input/autocorrect/autocorrect-2.6.1.tar".
#    - The path points to a `.tar` file named "autocorrect-2.6.1.tar" located within the "/kaggle/input/autocorrect/" directory.
# 

# In[1]:


get_ipython().system('pip install "/kaggle/input/autocorrect/autocorrect-2.6.1.tar"')
get_ipython().system('pip install "/kaggle/input/pyspellchecker/pyspellchecker-0.7.2-py3-none-any.whl"')


# ## 📚 Importing other Libraries and Setting Environment 🔧

# In[2]:


from typing import List  # 👉 Importing the List type from the typing module
import numpy as np  # 👉 Importing the NumPy library and aliasing it as np
import pandas as pd  # 👉 Importing the Pandas library and aliasing it as pd
import warnings  # 👉 Importing the warnings module
import logging  # 👉 Importing the logging module
import os  # 👉 Importing the os module
import shutil  # 👉 Importing the shutil module
import json  # 👉 Importing the json module
import transformers  # 👉 Importing the transformers library
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification  # 👉 Importing specific classes from transformers
from transformers import DataCollatorWithPadding  # 👉 Importing the DataCollatorWithPadding class from transformers
from datasets import Dataset, load_dataset, load_from_disk  # 👉 Importing specific functions/classes from the datasets module
from transformers import TrainingArguments, Trainer  # 👉 Importing specific classes from transformers
from datasets import load_metric, disable_progress_bar  # 👉 Importing specific functions/classes from datasets
from sklearn.metrics import mean_squared_error  # 👉 Importing the mean_squared_error function from scikit-learn
import torch  # 👉 Importing the torch library
from sklearn.model_selection import KFold, GroupKFold  # 👉 Importing KFold and GroupKFold classes from scikit-learn
from tqdm import tqdm  # 👉 Importing the tqdm function from the tqdm module

import nltk  # 👉 Importing the nltk library
from nltk.corpus import stopwords  # 👉 Importing the stopwords corpus from nltk
from nltk.tokenize import word_tokenize  # 👉 Importing the word_tokenize function from nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer  # 👉 Importing the TreebankWordDetokenizer class from nltk
from collections import Counter  # 👉 Importing the Counter class from the collections module
import spacy  # 👉 Importing the spacy library
import re  # 👉 Importing the re module
from autocorrect import Speller  # 👉 Importing the Speller class from autocorrect
from spellchecker import SpellChecker  # 👉 Importing the SpellChecker class from spellchecker
import lightgbm as lgb  # 👉 Importing the lightgbm library and aliasing it as lgb

warnings.simplefilter("ignore")  # 👉 Ignore warnings in the code
logging.disable(logging.ERROR)  # 👉 Disable logging to ERROR level
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 👉 Set an environment variable for tokenizers
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 👉 Set the minimum Tensorflow log level to 3
disable_progress_bar()  # 👉 Disable progress bars
tqdm.pandas()  # 👉 Enable tqdm progress bars


# ## Seed Initialization for Reproducibility 🌱

# **Explaination**:
# 
# 
# 1. `def seed_everything(seed: int):`
#    - Defines a function named `seed_everything` that takes an integer argument `seed`. This function will initialize seeds for various random number generators.
# 
# 2. `import random, os`
#    - Imports the `random` and `os` modules, which are used for random number generation and system-related operations, respectively.
# 
# 3. `import numpy as np`
#    - Imports the `numpy` library and aliases it as `np`. NumPy is commonly used for numerical computations.
# 
# 4. `import torch`
#    - Imports the `torch` library, which is the core library for PyTorch, a popular deep learning framework.
# 
# 5. `random.seed(seed)`
#    - Sets the seed for Python's random module to the provided `seed` value. This ensures reproducibility of random numbers generated using Python's `random` functions.
# 
# 6. `os.environ['PYTHONHASHSEED'] = str(seed)`
#    - Sets the hash seed for Python to the provided `seed` value. This helps ensure consistent hash-based operations across different runs of the program.
# 
# 7. `np.random.seed(seed)`
#    - Sets the seed for NumPy's random number generator to the provided `seed`, making NumPy's random operations reproducible.
# 
# 8. `torch.manual_seed(seed)`
#    - Sets the seed for PyTorch's random number generator for CPU operations to the provided `seed`.
# 
# 9. `torch.cuda.manual_seed(seed)`
#    - Sets the seed for PyTorch's random number generator for GPU operations to the provided `seed`.
# 
# 10. `torch.backends.cudnn.deterministic = True`
#     - Enforces deterministic behavior for CuDNN (CUDA Deep Neural Network library) to ensure consistent results when using GPU acceleration.
# 
# 11. `torch.backends.cudnn.benchmark = True`
#     - Enables CuDNN benchmark mode, which can optimize GPU performance during training.
# 
# 12. `seed_everything(seed=42)`
#     - Calls the `seed_everything` function with a specific seed value, in this case, `42`, to initialize all the random number generators with this seed value, ensuring reproducibility.
# 

# In[3]:


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(seed=42)


# ## Configuration Settings 🛠️

# **Explaination**:
# 
# 
# 1. `class CFG:`: 
#    - Defines a Python class named `CFG` for configuration settings. This class will hold various configuration parameters for your project.
# 
# 2. `model_name = "debertav3base"`:
#    - Defines a variable `model_name` with the value "debertav3base" as a reference to the model being used.
# 
# 3. `learning_rate = 0.000016`:
#    - Sets the learning rate (a hyperparameter for gradient descent) to `0.000016`.
# 
# 4. `weight_decay = 0.03`:
#    - Sets the weight decay, a regularization term (L2 regularization), to `0.03`. It helps prevent overfitting during training.
# 
# 5. `hidden_dropout_prob = 0.007`:
#    - Defines the dropout probability for hidden layers in the model, set to `0.007`. Dropout helps prevent overfitting by randomly dropping some neurons during training.
# 
# 6. `attention_probs_dropout_prob = 0.007`:
#    - Sets the dropout probability for attention layers in the model to `0.007`. This dropout is specific to the attention mechanism.
# 
# 7. `num_train_epochs = 5`:
#    - Specifies the number of training epochs (iterations over the entire training dataset) as `5`.
# 
# 8. `n_splits = 4`:
#    - Defines the number of splits for cross-validation, typically used to assess model performance. Set to `4` in this case.
# 
# 9. `batch_size = 12`:
#    - Sets the batch size for training data to `12`. This determines how many samples are processed together in each training step.
# 
# 10. `random_seed = 42`:
#     - Sets the random seed to `42` for reproducibility. This ensures that random operations produce the same results across different runs.
# 
# 11. `save_steps = 100`:
#     - Specifies the number of steps (or iterations) before saving model checkpoints during training. In this case, checkpoints are saved every `100` steps.
# 
# 12. `max_length = 512`:
#     - Defines the maximum sequence length for input data as `512`. This can be important when working with text data, as it limits the length of input sequences.
# 

# In[4]:


# Define a class for configuration settings
class CFG:
    model_name = "debertav3base"  # Model name for reference
    learning_rate = 0.000016 # Learning rate 📚
    weight_decay = 0.03  # Weight decay (L2 regularization) 🏋️‍♂️
    hidden_dropout_prob = 0.007  # Dropout probability for hidden layers 🙈
    attention_probs_dropout_prob = 0.007  # Dropout probability for attention layers 🙉
    num_train_epochs = 5  # Number of training epochs 🚂
    n_splits = 4  # Number of splits for cross-validation 🔄
    batch_size = 12  # Batch size for training data 📦
    random_seed = 42  # Random seed for reproducibility 🌱
    save_steps = 100  # Number of steps before saving model checkpoints 📥
    max_length = 512  # Maximum sequence length for input data 📏


# ##  Data Loading 📁

# **Explaination**:
# 
# 1. `DATA_DIR = "/kaggle/input/commonlit-evaluate-student-summaries/"`:
#    - Defines a variable `DATA_DIR` to store the directory path where the data files are located.
# 
# 2. `prompts_train = pd.read_csv(DATA_DIR + "prompts_train.csv")`:
#    - Loads the training prompts data from the CSV file named "prompts_train.csv" located in `DATA_DIR` into a Pandas DataFrame named `prompts_train`.
# 
# 3. `prompts_test = pd.read_csv(DATA_DIR + "prompts_test.csv")`:
#    - Loads the testing prompts data from the CSV file named "prompts_test.csv" located in `DATA_DIR` into a Pandas DataFrame named `prompts_test`.
# 
# 4. `summaries_train = pd.read_csv(DATA_DIR + "summaries_train.csv")`:
#    - Loads the training summaries data from the CSV file named "summaries_train.csv" located in `DATA_DIR` into a Pandas DataFrame named `summaries_train`.
# 
# 5. `summaries_test = pd.read_csv(DATA_DIR + "summaries_test.csv")`:
#    - Loads the testing summaries data from the CSV file named "summaries_test.csv" located in `DATA_DIR` into a Pandas DataFrame named `summaries_test`.
# 
# 6. `sample_submission = pd.read_csv(DATA_DIR + "sample_submission.csv")`:
#    - Loads the sample submission data from the CSV file named "sample_submission.csv" located in `DATA_DIR` into a Pandas DataFrame named `sample_submission`.
# 

# In[5]:


# Define the directory where the data files are located
DATA_DIR = "/kaggle/input/commonlit-evaluate-student-summaries/"

# Load the training prompts data from a CSV file
prompts_train = pd.read_csv(DATA_DIR + "prompts_train.csv")

# Load the test prompts data from a CSV file
prompts_test = pd.read_csv(DATA_DIR + "prompts_test.csv")

# Load the training summaries data from a CSV file
summaries_train = pd.read_csv(DATA_DIR + "summaries_train.csv")

# Load the test summaries data from a CSV file
summaries_test = pd.read_csv(DATA_DIR + "summaries_test.csv")

# Load the sample submission data from a CSV file
sample_submission = pd.read_csv(DATA_DIR + "sample_submission.csv")


# ## 📝 Text Preprocessing and Feature Extraction 🔍

# **Explaination**:
# 
# 
# 1. `class Preprocessor:`:
#    - Defines a Python class named `Preprocessor` for text preprocessing and feature extraction.
# 
# 2. `def __init__(self, model_name: str) -> None:`:
#    - Defines the class constructor method, which initializes the class instance.
#    - Takes `model_name` as a parameter, which specifies the name of the model to be used for tokenization.
# 
# 3. `self.tokenizer = AutoTokenizer.from_pretrained(f"/kaggle/input/{model_name
# 
# }")`:
#    - Initializes the tokenizer using the Hugging Face Transformers library based on the specified model name.
# 
# 4. `self.twd = TreebankWordDetokenizer()`:
#    - Initializes the `TreebankWordDetokenizer` object, which is used for detokenization.
# 
# 5. `self.STOP_WORDS = set(stopwords.words('english'))`:
#    - Defines a set of English stopwords using NLTK's `stopwords` corpus.
# 
# 6. `self.spacy_ner_model = spacy.load('en_core_web_sm',)`:
#    - Loads the spaCy Named Entity Recognition (NER) model for English.
# 
# 7. `self.speller = Speller(lang='en')`:
#    - Initializes the `Speller` object from the `pyspellchecker` library for spelling correction.
# 
# 8. `self.spellchecker = SpellChecker()`:
#    - Initializes the `SpellChecker` object from the `spellchecker` library for spelling checking.
# 
# The rest of the code defines various methods within the `Preprocessor` class for performing text preprocessing and feature extraction tasks, including word overlap count, n-grams, NER overlap count, quotes count, spelling correction, and more. These methods are used to process the input data and extract relevant features.
# 
# At the end of the cell, an instance of the `Preprocessor` class named `preprocessor` is created, and it is configured with the specified `model_name` from the configuration settings.

# In[6]:


# Define a class for text preprocessing and feature extraction
class Preprocessor:
    def __init__(self, model_name: str) -> None:
        # Initialize the tokenizer for the specified model
        self.tokenizer = AutoTokenizer.from_pretrained(f"/kaggle/input/{model_name}")
        
        # Initialize the TreebankWordDetokenizer for detokenization
        self.twd = TreebankWordDetokenizer()
        
        # Define a set of English stopwords
        self.STOP_WORDS = set(stopwords.words('english'))
        
        # Load the spaCy NER model for English
        self.spacy_ner_model = spacy.load('en_core_web_sm',)
        
        # Initialize the pyspellchecker Speller for spelling correction
        self.speller = Speller(lang='en')
        
        # Initialize the pyspellchecker SpellChecker for spelling checking
        self.spellchecker = SpellChecker() 
        
    def word_overlap_count(self, row):
        """ Count the overlapping words between prompt and summary """        
        def check_is_stop_word(word):
            return word in self.STOP_WORDS
        
        prompt_words = row['prompt_tokens']
        summary_words = row['summary_tokens']
        
        # Filter out stopwords if they are defined
        if self.STOP_WORDS:
            prompt_words = list(filter(check_is_stop_word, prompt_words))
            summary_words = list(filter(check_is_stop_word, summary_words))
        
        # Calculate the count of overlapping words
        return len(set(prompt_words).intersection(set(summary_words)))
            
    def ngrams(self, token, n):
        # Use the zip function to generate n-grams
        ngrams = zip(*[token[i:] for i in range(n)])
        return [" ".join(ngram) for ngram in ngrams]

    def ngram_co_occurrence(self, row, n: int) -> int:
        # Tokenize the original text and summary into words
        original_tokens = row['prompt_tokens']
        summary_tokens = row['summary_tokens']

        # Generate n-grams for the original text and summary
        original_ngrams = set(self.ngrams(original_tokens, n))
        summary_ngrams = set(self.ngrams(summary_tokens, n))

        # Calculate the number of common n-grams
        common_ngrams = original_ngrams.intersection(summary_ngrams)
        return len(common_ngrams)
    
    def ner_overlap_count(self, row, mode:str):
        model = self.spacy_ner_model
        def clean_ners(ner_list):
            return set([(ner[0].lower(), ner[1]) for ner in ner_list])
        prompt = model(row['prompt_text'])
        summary = model(row['text'])

        if "spacy" in str(model):
            prompt_ner = set([(token.text, token.label_) for token in prompt.ents])
            summary_ner = set([(token.text, token.label_) for token in summary.ents])
        elif "stanza" in str(model):
            prompt_ner = set([(token.text, token.type) for token in prompt.ents])
            summary_ner = set([(token.text, token.type) for token in summary.ents])
        else:
            raise Exception("Model not supported")

        prompt_ner = clean_ners(prompt_ner)
        summary_ner = clean_ners(summary_ner)

        intersecting_ners = prompt_ner.intersection(summary_ner)
        
        ner_dict = dict(Counter([ner[1] for ner in intersecting_ners]))
        
        if mode == "train":
            return ner_dict
        elif mode == "test":
            return {key: ner_dict.get(key) for key in self.ner_keys}

    
    def quotes_count(self, row):
        summary = row['text']
        text = row['prompt_text']
        quotes_from_summary = re.findall(r'"([^"]*)"', summary)
        if len(quotes_from_summary) > 0:
            return [quote in text for quote in quotes_from_summary].count(True)
        else:
            return 0

    def spelling(self, text):
        wordlist = text.split()
        amount_miss = len(list(self.spellchecker.unknown(wordlist)))
        return amount_miss
    
    def add_spelling_dictionary(self, tokens: List[str]) -> List[str]:
        """Dictionary update for pyspell checker and autocorrect"""
        self.spellchecker.word_frequency.load_words(tokens)
        self.speller.nlp_data.update({token: 1000 for token in tokens})
    
    def run(self, prompts: pd.DataFrame, summaries: pd.DataFrame, mode: str) -> pd.DataFrame:
        # Before merge preprocessing
        prompts["prompt_length"] = prompts["prompt_text"].apply(lambda x: len(word_tokenize(x)))
        prompts["prompt_tokens"] = prompts["prompt_text"].apply(lambda x: word_tokenize(x))

        summaries["summary_length"] = summaries["text"].apply(lambda x: len(word_tokenize(x)))
        summaries["summary_tokens"] = summaries["text"].apply(lambda x: word_tokenize(x))
        
        # Add prompt tokens into spelling checker dictionary
        prompts["prompt_tokens"].apply(lambda x: self.add_spelling_dictionary(x))
        
        # Fix misspelling in summaries
        summaries["fixed_summary_text"] = summaries["text"].progress_apply(lambda x: self.speller(x))
        
        # Count misspellings in summaries
        summaries["splling_err_num"] = summaries["text"].progress_apply(self.spelling)
        
        # Merge prompts and summaries
        input_df = summaries.merge(prompts, how="left", on="prompt_id")

        # After merge preprocessing
        input_df['word_overlap_count'] = input_df.progress_apply(self.word_overlap_count, axis=1)
        input_df['bigram_overlap_count'] = input_df.progress_apply(self.ngram_co_occurrence, args=(2,), axis=1)
        input_df['bigram_overlap_ratio'] = input_df['bigram_overlap_count'] / (input_df['summary_length'] - 1)
        input_df['trigram_overlap_count'] = input_df.progress_apply(self.ngram_co_occurrence, args=(3,), axis=1)
        input_df['trigram_overlap_ratio'] = input_df['trigram_overlap_count'] / (input_df['summary_length'] - 2)
        input_df['quotes_count'] = input_df.progress_apply(self.quotes_count, axis=1)
        
        return input_df.drop(columns=["summary_tokens", "prompt_tokens"])

# Create an instance of the Preprocessor class
preprocessor = Preprocessor(model_name=CFG.model_name)


# ## 📝Text Preprocessing and Feature Extraction  🔍

# **Explaination**:
# 
# 1. `train = preprocessor.run(prompts_train, summaries_train, mode="train")`:
#    - Calls the `run` method of the `preprocessor` instance to preprocess and extract features from the training data.
#    - The `mode` parameter is set to "train" to indicate that this is the training data.
#    - The resulting DataFrame is assigned to the variable `train`.
# 
# 2. `test = preprocessor.run(prompts_test, summaries_test, mode="test")`:
#    - Calls the `run` method of the `preprocessor` instance to preprocess and extract features from the testing data.
#    - The `mode` parameter is set to "test" to indicate that this is the testing data.
#    - The resulting DataFrame is assigned to the variable `test`.
# 
# 3. `train.head()`:
#    - Displays the first few rows of the `train` DataFrame to provide an overview of the preprocessed training data.
# 
# These lines of code preprocess and extract features from both the training and testing data, making them ready for use in machine learning or deep learning models. The `train` DataFrame contains the preprocessed training data, and the `test` DataFrame contains the preprocessed testing data.

# In[7]:


# Perform text preprocessing and feature extraction on training data
train = preprocessor.run(prompts_train, summaries_train, mode="train")

# Perform text preprocessing and feature extraction on testing data
test = preprocessor.run(prompts_test, summaries_test, mode="test")

# Display the first few rows of the training data
train.head()


# ## Group K-Fold Cross-Validation Splitting 🔄

# **Explaination**:
# 
# 1. `gkf = GroupKFold(n_splits=CFG.n_splits)`:
#    - Creates a `GroupKFold` object with the specified number of splits, which is determined by the `CFG.n_splits` configuration parameter.
# 
# 2. `for i, (_, val_index) in enumerate(gkf.split(train, groups=train["prompt_id"])):`:
#    - Iterates through the cross-validation splits generated by `GroupKFold`.
#    - `i` is used as the fold index, and `_` is used to ignore the training indices.
#    - `val_index` contains the indices of the validation data for the current fold.
# 
# 3. `train.loc[val_index, "fold"] = i`:
#    - Assigns the fold index `i` to the rows in the `train` DataFrame that correspond to the validation data for the current fold.
#    - This allows for grouping and tracking data within each fold for cross-validation.
# 
# 4. `train.head()`:
#    - Displays the first few rows of the training data with the newly assigned fold numbers.
#    - Each row in the `train` DataFrame now has a "fold" column indicating which fold it belongs to for cross-validation.
# 
# This code cell performs group-based K-Fold cross-validation splitting and assigns fold numbers to the training data. It's a common technique for evaluating machine learning models on multiple subsets of the data while ensuring that related data points stay within the same fold.

# In[8]:


# Create a GroupKFold object with the specified number of splits
gkf = GroupKFold(n_splits=CFG.n_splits)

# Iterate through the splits and assign fold numbers to validation data
for i, (_, val_index) in enumerate(gkf.split(train, groups=train["prompt_id"])):
    train.loc[val_index, "fold"] = i

# Display the first few rows of the training data with fold assignments
train.head()


# ## Evaluation Metrics 📊📈

# **Explaination**:
# 
# 
# 1. `compute_metrics(eval_pred)`:
#    - This function takes in an argument `eval_pred`, which is expected to be a tuple containing predictions and true labels.
#    - It calculates the Root Mean Squared Error (RMSE) between the predictions and the labels.
#    - RMSE measures the average magnitude of the errors between predicted values and actual values.
#    - The calculated RMSE is returned as a dictionary with the key `"rmse"`.
# 
# 2. `compute_mcrmse(eval_pred)`:
#    - This function also takes in an argument `eval_pred`, which is expected to be a tuple containing predictions and true labels.
#    - It calculates the Mean Columnwise Root Mean Squared Error (MCRMSE) based on the predictions and labels.
#    - MCRMSE is a metric used in the competition and calculates the RMSE for each column (content and wording) and then takes the mean of these RMSE values.
#    - The calculated MCRMSE is returned as a dictionary with three keys:
#      - `"content_rmse"`: RMSE for the content column.
#      - `"wording_rmse"`: RMSE for the wording column.
#      - `"mcrmse"`: Mean of the column-wise RMSE values.
# 
# 3. `compt_score(content_true, content_pred, wording_true, wording_pred)`:
#    - This function takes in four arguments: `content_true`, `content_pred`, `wording_true`, and `wording_pred`.
#    - It calculates the competition score based on the RMSE between true and predicted values for both content and wording.
#    - The competition score is computed as the average of the RMSE for content and wording.
#    - The function returns the competition score, which provides an overall evaluation of model performance.
# 
# These functions are designed to evaluate the performance of a model on the competition's specific metrics, including RMSE and MCRMSE, which  used in regression task.

# In[9]:


# Define a function to compute root mean squared error (RMSE)
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # Calculate RMSE
    rmse = mean_squared_error(labels, predictions, squared=False)
    
    # Return RMSE as a dictionary
    return {"rmse": rmse}

# Define a function to compute mean columnwise root mean squared error (MCRMSE)
def compute_mcrmse(eval_pred):
    """
    Calculates mean columnwise root mean squared error
    Reference: https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/overview/evaluation
    """
    preds, labels = eval_pred

    # Calculate column-wise RMSE and mean RMSE
    col_rmse = np.sqrt(np.mean((preds - labels) ** 2, axis=0))
    mcrmse = np.mean(col_rmse)

    # Return MCRMSE along with individual column RMSE
    return {
        "content_rmse": col_rmse[0],
        "wording_rmse": col_rmse[1],
        "mcrmse": mcrmse,
    }

# Define a function to compute the competition score
def compt_score(content_true, content_pred, wording_true, wording_pred):
    # Calculate RMSE for content and wording
    content_score = mean_squared_error(content_true, content_pred)**(1/2)
    wording_score = mean_squared_error(wording_true, wording_pred)**(1/2)
    
    # Calculate the competition score as the average of content and wording scores
    return (content_score + wording_score) / 2


# ## 🔍 Content Score Regressor 📈

# **Explaination**:
# 
# 
# This code defines a Python class named `ContentScoreRegressor`, which  used for fine-tuning a pre-trained model and making predictions for content scores.
# 
# 1. `__init__` method:
#    - This is the class constructor that initializes the `ContentScoreRegressor` object.
#    - It takes several parameters including `model_name`, `model_dir`, `target`, `hidden_dropout_prob`, `attention_probs_dropout_prob`, and `max_length`.
#    - It sets various attributes including `inputs`, `input_col`, `text_cols`, `target_cols`, and model-related attributes like `tokenizer`, `model_config`, and initializes the random seed.
# 
# 2. `tokenize_function` method:
#    - This method takes a DataFrame `examples` as input, which is expected to contain text data and target labels.
#    - It tokenizes the text data using the pre-trained tokenizer with specified max length and returns a dictionary with tokenized inputs and labels.
# 
# 3. `tokenize_function_test` method:
#    - Similar to `tokenize_function`, but used for tokenizing test data without labels.
# 
# 4. `train` method:
#    - This method is used for fine-tuning a pre-trained model on the training data.
#    - It takes various training-related parameters like `fold`, `train_df`, `valid_df`, `batch_size`, `learning_rate`, `weight_decay`, `num_train_epochs`, and `save_steps`.
#    - It concatenates the text columns (e.g., prompt title, prompt question, and fixed summary text) in both training and validation DataFrames to create input text.
#    - It loads the pre-trained model and configures training arguments.
#    - It fine-tunes the model using the Hugging Face `Trainer` class and saves the best model to the specified directory.
# 
# 5. `predict` method:
#    - This method is used for making predictions on test data.
#    - It takes `test_df` and `fold` as input parameters.
#    - It concatenates the text columns in the test DataFrame to create input text.
#    - It loads the pre-trained model for inference and makes predictions using the Hugging Face `Trainer`.
#    - The predicted content scores are returned as an array.
# 
# The class is designed to encapsulate the fine-tuning and prediction process for content scores using a pre-trained model. It appears to be used in a machine learning pipeline to train and evaluate models .

# In[10]:


# Define a class for the Content Score Regressor
class ContentScoreRegressor:
    def __init__(self, 
                model_name: str,
                model_dir: str,
                target: str,
                hidden_dropout_prob: float,
                attention_probs_dropout_prob: float,
                max_length: int,
                ):
        # Define input columns and target column
        self.inputs = ["prompt_text", "prompt_title", "prompt_question", "fixed_summary_text"]
        self.input_col = "input"
        self.text_cols = [self.input_col] 
        self.target = target
        self.target_cols = [target]

        # Initialize model-related attributes
        self.model_name = model_name
        self.model_dir = model_dir
        self.max_length = max_length
        
        # Initialize tokenizer and model configuration
        self.tokenizer = AutoTokenizer.from_pretrained(f"/kaggle/input/{model_name}")
        self.model_config = AutoConfig.from_pretrained(f"/kaggle/input/{model_name}")
        
        # Update model configuration with additional parameters
        self.model_config.update({
            "hidden_dropout_prob": hidden_dropout_prob,
            "attention_probs_dropout_prob": attention_probs_dropout_prob,
            "num_labels": 1,
            "problem_type": "regression",
        })
        
        # Set a fixed random seed for reproducibility
        seed_everything(seed=42)

        # Initialize data collator for padding
        self.data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer
        )

    def tokenize_function(self, examples: pd.DataFrame):
        labels = [examples[self.target]]
        tokenized = self.tokenizer(examples[self.input_col],
                         padding=False,
                         truncation=True,
                         max_length=self.max_length)
        return {
            **tokenized,
            "labels": labels,
        }
    
    def tokenize_function_test(self, examples: pd.DataFrame):
        tokenized = self.tokenizer(examples[self.input_col],
                         padding=False,
                         truncation=True,
                         max_length=self.max_length)
        return tokenized
        
    def train(self, 
            fold: int,
            train_df: pd.DataFrame,
            valid_df: pd.DataFrame,
            batch_size: int,
            learning_rate: float,
            weight_decay: float,
            num_train_epochs: float,
            save_steps: int,
        ) -> None:
        """Fine-tuning the model"""
        
        sep = self.tokenizer.sep_token
        
        # Create input text by concatenating title, question, and summary
        train_df[self.input_col] = (
                    train_df["prompt_title"] + sep 
                    + train_df["prompt_question"] + sep 
                    + train_df["fixed_summary_text"]
                  )

        valid_df[self.input_col] = (
                    valid_df["prompt_title"] + sep 
                    + valid_df["prompt_question"] + sep 
                    + valid_df["fixed_summary_text"]
                  )
        
        # Select relevant columns for training and validation
        train_df = train_df[[self.input_col] + self.target_cols]
        valid_df = valid_df[[self.input_col] + self.target_cols]
        
        # Load the pre-trained model for content score prediction
        model_content = AutoModelForSequenceClassification.from_pretrained(
            f"/kaggle/input/{self.model_name}", 
            config=self.model_config
        )

        # Create datasets from DataFrames
        train_dataset = Dataset.from_pandas(train_df, preserve_index=False) 
        val_dataset = Dataset.from_pandas(valid_df, preserve_index=False) 
    
        # Tokenize and preprocess the datasets
        train_tokenized_datasets = train_dataset.map(self.tokenize_function, batched=False)
        val_tokenized_datasets = val_dataset.map(self.tokenize_function, batched=False)

        # Define model training arguments
        model_fold_dir = os.path.join(self.model_dir, str(fold)) 
        training_args = TrainingArguments(
            output_dir=model_fold_dir,
            load_best_model_at_end=True,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=8,
            num_train_epochs=num_train_epochs,
            weight_decay=weight_decay,
            report_to='none',
            greater_is_better=False,
            save_strategy="steps",
            evaluation_strategy="steps",
            eval_steps=save_steps,
            save_steps=save_steps,
            metric_for_best_model="rmse",
            save_total_limit=1
        )

        # Create a trainer for model training
        trainer = Trainer(
            model=model_content,
            args=training_args,
            train_dataset=train_tokenized_datasets,
            eval_dataset=val_tokenized_datasets,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
            data_collator=self.data_collator
        )

        # Train the model
        trainer.train()
        
        # Save the trained model and tokenizer
        model_content.save_pretrained(self.model_dir)
        self.tokenizer.save_pretrained(self.model_dir)

        
    def predict(self, 
                test_df: pd.DataFrame,
                fold: int,
               ):
        """Predict content score for test data"""
        
        sep = self.tokenizer.sep_token
        
        # Create input text for test data
        in_text = (
                    test_df["prompt_title"] + sep 
                    + test_df["prompt_question"] + sep 
                    + test_df["fixed_summary_text"]
                  )
        test_df[self.input_col] = in_text

        # Select the relevant columns
        test_ = test_df[[self.input_col]]
    
        # Create a dataset from the test data
        test_dataset = Dataset.from_pandas(test_, preserve_index=False) 
        test_tokenized_dataset = test_dataset.map(self.tokenize_function_test, batched=False)

        # Load the trained content score prediction model
        model_content = AutoModelForSequenceClassification.from_pretrained(f"{self.model_dir}")
        model_content.eval()
        
        # Define model prediction arguments
        model_fold_dir = os.path.join(self.model_dir, str(fold)) 
        test_args = TrainingArguments(
            output_dir=model_fold_dir,
            do_train = False,
            do_predict = True,
            per_device_eval_batch_size = 4,
            dataloader_drop_last = False,
        )

        # Initialize a trainer for inference
        infer_content = Trainer(
                      model = model_content, 
                      tokenizer=self.tokenizer,
                      data_collator=self.data_collator,
                      args = test_args)

        # Perform predictions
        preds = infer_content.predict(test_tokenized_dataset)[0]

        return preds


# ## Training, Validation, and Prediction Functions 🚀📊

# **Explaination**:
# 
# This code defines several functions for training and evaluating a model using K-Fold cross-validation and making predictions. 
# 
# 1. `train_by_fold` function:
#    - This function is used for training a model by fold using K-Fold cross-validation.
#    - It takes various parameters including `train_df` (the training data), `model_name`, `target` (the target variable), and several hyperparameters for training.
#    - It first checks if a directory with the `model_name` exists and deletes it if it does. Then, it creates a new directory.
#    - It iterates through each fold and splits the training data into training and validation sets.
#    - For each fold, it creates an instance of the `ContentScoreRegressor` class and trains the model using the specified hyperparameters.
#    - If `save_each_model` is set to `True`, it saves each model in a separate directory based on the fold.
# 
# 2. `validate` function:
#    - This function is used for validating the model and making out-of-fold (oof) predictions.
#    - It takes parameters similar to `train_by_fold` and iterates through each fold.
#    - For each fold, it creates an instance of the `ContentScoreRegressor` class and makes predictions on the validation data.
#    - The predictions are then assigned to the corresponding rows in the training DataFrame.
#    - This function returns the updated training DataFrame with predictions.
# 
# 3. `predict` function:
#    - This function is used for making predictions on the test data.
#    - It also takes parameters similar to `train_by_fold` and iterates through each fold.
#    - For each fold, it creates an instance of the `ContentScoreRegressor` class and makes predictions on the test data.
#    - The fold-specific predictions are stored in columns named `"target_pred_fold_{fold}"`.
#    - Finally, it calculates the mean of fold predictions for each data point and assigns it to the `"target"` column in the test DataFrame.
# 
# These functions are designed to automate the process of training, validating, and making predictions using K-Fold cross-validation for a regression task.

# In[11]:


# Define a function for training by fold
def train_by_fold(
        train_df: pd.DataFrame,
        model_name: str,
        target: str,
        save_each_model: bool,
        n_splits: int,
        batch_size: int,
        learning_rate: int,
        hidden_dropout_prob: float,
        attention_probs_dropout_prob: float,
        weight_decay: float,
        num_train_epochs: int,
        save_steps: int,
        max_length: int
    ):

    # Delete old model files
    if os.path.exists(model_name):
        shutil.rmtree(model_name)
    
    os.mkdir(model_name)
        
    for fold in range(n_splits):
        print(f"fold {fold}:")
        
        train_data = train_df[train_df["fold"] != fold]
        valid_data = train_df[train_df["fold"] == fold]
        
        if save_each_model == True:
            model_dir =  f"{target}/{model_name}/fold_{fold}"
        else: 
            model_dir =  f"{model_name}/fold_{fold}"

        csr = ContentScoreRegressor(
            model_name=model_name,
            target=target,
            model_dir=model_dir, 
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_length=max_length,
        )
        
        csr.train(
            fold=fold,
            train_df=train_data,
            valid_df=valid_data, 
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_train_epochs=num_train_epochs,
            save_steps=save_steps,
        )

# Define a function for validation (predicting oof data)
def validate(
    train_df: pd.DataFrame,
    target: str,
    save_each_model: bool,
    model_name: str,
    hidden_dropout_prob: float,
    attention_probs_dropout_prob: float,
    max_length: int
    ) -> pd.DataFrame:
    """Predict out-of-fold (oof) data"""
    for fold in range(CFG.n_splits):
        print(f"fold {fold}:")
        
        valid_data = train_df[train_df["fold"] == fold]
        
        if save_each_model == True:
            model_dir =  f"{target}/{model_name}/fold_{fold}"
        else: 
            model_dir =  f"{model_name}/fold_{fold}"
        
        csr = ContentScoreRegressor(
            model_name=model_name,
            target=target,
            model_dir=model_dir,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_length=max_length,
        )
        
        pred = csr.predict(
            test_df=valid_data, 
            fold=fold
        )
        
        train_df.loc[valid_data.index, f"{target}_pred"] = pred

    return train_df
    
# Define a function for prediction (using mean folds)
def predict(
    test_df: pd.DataFrame,
    target: str,
    save_each_model: bool,
    model_name: str,
    hidden_dropout_prob: float,
    attention_probs_dropout_prob: float,
    max_length: int
    ):
    """Predict using mean of folds"""

    for fold in range(CFG.n_splits):
        print(f"fold {fold}:")
        
        if save_each_model == True:
            model_dir =  f"{target}/{model_name}/fold_{fold}"
        else: 
            model_dir =  f"{model_name}/fold_{fold}"

        csr = ContentScoreRegressor(
            model_name=model_name,
            target=target,
            model_dir=model_dir, 
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_length=max_length,
        )
        
        pred = csr.predict(
            test_df=test_df, 
            fold=fold
        )
        
        test_df[f"{target}_pred_{fold}"] = pred
    
    test_df[f"{target}"] = test_df[[f"{target}_pred_{fold}" for fold in range(CFG.n_splits)]].mean(axis=1)

    return test_df


# ## Training📊, Validation🚀, and Prediction Pipeline 🔄

# **Explaination**:
# 
# 
# This code snippet performs the following tasks for two target variables, `"content"` and `"wording"`:
# 
# 1. Training and Validation:
#    - For each target variable, it calls the `train_by_fold` function to train a model using K-Fold cross-validation.
#    - The training data `train` is used, along with various hyperparameters and settings from the `CFG` configuration.
#    - After training, it calls the `validate` function to make out-of-fold predictions on the training data.
#    - The predictions are stored in columns named `{target}_pred`, where `target` is either `"content"` or `"wording"`.
# 
# 2. RMSE Calculation:
#    - After making predictions, it calculates the Root Mean Squared Error (RMSE) between the true target values and the predicted values.
#    - The calculated RMSE is printed to the console to assess the model's performance on the training data.
# 
# 3. Test Data Prediction:
#    - Finally, it calls the `predict` function to make predictions on the test data.
#    - The test data `test` is used, and predictions are stored in columns named `{target}_pred_fold_{fold}` for each fold.
#    - Then, it calculates the mean of fold predictions for each data point and assigns it to the `{target}` column in the test DataFrame.
# 
# This code segment essentially trains, validates, and makes predictions for both `"content"` and `"wording"` target variables using K-Fold cross-validation. It allows for evaluating the model's performance on both the training data and the test data.

# In[12]:


for target in ["content", "wording"]:
    train_by_fold(
        train,
        model_name=CFG.model_name,
        save_each_model=False,
        target=target,
        learning_rate=CFG.learning_rate,
        hidden_dropout_prob=CFG.hidden_dropout_prob,
        attention_probs_dropout_prob=CFG.attention_probs_dropout_prob,
        weight_decay=CFG.weight_decay,
        num_train_epochs=CFG.num_train_epochs,
        n_splits=CFG.n_splits,
        batch_size=CFG.batch_size,
        save_steps=CFG.save_steps,
        max_length=CFG.max_length
    )
    
    
    train = validate(
        train,
        target=target,
        save_each_model=False,
        model_name=CFG.model_name,
        hidden_dropout_prob=CFG.hidden_dropout_prob,
        attention_probs_dropout_prob=CFG.attention_probs_dropout_prob,
        max_length=CFG.max_length
    )

    rmse = mean_squared_error(train[target], train[f"{target}_pred"], squared=False)
    print(f"cv {target} rmse: {rmse}")

    test = predict(
        test,
        target=target,
        save_each_model=False,
        model_name=CFG.model_name,
        hidden_dropout_prob=CFG.hidden_dropout_prob,
        attention_probs_dropout_prob=CFG.attention_probs_dropout_prob,
        max_length=CFG.max_length
    )


# In[13]:


train.head()


# ## Define Targets and Columns to Drop 📊🔍

# In[14]:


# Define the target variables
targets = ["content", "wording"]

# Define columns to drop from the dataset
drop_columns = [
    "fold", "student_id", "prompt_id", "text", "fixed_summary_text",
    "prompt_question", "prompt_title", "prompt_text"
] + targets


# ## LightGBM Model Training 🌟🔍🚀

# **Explaination**:
# 
# This code  is responsible for training LightGBM (Gradient Boosting) models for the target variables `"content"` and `"wording"` using K-Fold cross-validation.
# 
# 1. `model_dict = {}`:
#    - Initializes an empty dictionary called `model_dict` that will be used to store the trained models for each target.
# 
# 2. `for target in targets:`:
#    - Iterates over the target variables. It seems like `targets` is expected to contain the target variable names, which are `"content"` and `"wording"` in this case.
# 
# 3. `models = []`:
#    - Initializes an empty list called `models` to store the LightGBM models for the current target variable.
# 
# 4. `for fold in range(CFG.n_splits):`:
#    - Initiates a loop that iterates through the K-Fold cross-validation folds (specified by `CFG.n_splits`).
# 
# 5. `X_train_cv = train[train["fold"] != fold].drop(columns=drop_columns)`:
#    - Filters the training data `train` to exclude the current fold. This creates a training dataset `X_train_cv` that does not include the fold being evaluated.
#    - The `drop_columns` are columns that are not included in the training data.
# 
# 6. `y_train_cv = train[train["fold"] != fold][target]`:
#    - Extracts the target variable (`target`) for the training dataset `X_train_cv`.
# 
# 7. `X_eval_cv = train[train["fold"] == fold].drop(columns=drop_columns)`:
#    - Filters the training data `train` to include only the current fold. This creates an evaluation dataset `X_eval_cv` that consists of data from the fold being evaluated.
# 
# 8. `y_eval_cv = train[train["fold"] == fold][target]`:
#    - Extracts the target variable (`target`) for the evaluation dataset `X_eval_cv`.
# 
# 9. `dtrain = lgb.Dataset(X_train_cv, label=y_train_cv)`:
#    - Creates a LightGBM dataset `dtrain` using the training data `X_train_cv` and associated labels `y_train_cv`.
# 
# 10. `dval = lgb.Dataset(X_eval_cv, label=y_eval_cv)`:
#     - Creates a LightGBM dataset `dval` using the evaluation data `X_eval_cv` and associated labels `y_eval_cv`.
# 
# 11. `params = {...}`:
#     - Defines a dictionary `params` that contains LightGBM hyperparameters for the model. These hyperparameters include boosting type, random state, objective (regression), metric (root mean squared error), learning rate, maximum depth, and regularization terms.
# 
# 12. `evaluation_results = {}`:
#     - Initializes an empty dictionary `evaluation_results` to store evaluation results for the model.
# 
# 13. `model = lgb.train(...)`:
#     - Trains a LightGBM model using the specified hyperparameters and data.
#     - The `num_boost_round` parameter specifies the maximum number of boosting rounds.
#     - Early stopping is implemented using `lgb.early_stopping`, which stops training if the validation metric (RMSE) does not improve for a certain number of rounds.
#     - Evaluation results are recorded using `lgb.log_evaluation` and `lgb.callback.record_evaluation`.
# 
# 14. `models.append(model)`:
#     - Appends the trained LightGBM model to the `models` list for the current fold.
# 
# 15. `model_dict[target] = models`:
#     - Stores the list of trained LightGBM models for the current target variable (`target`) in the `model_dict` dictionary.
# 
# This code performs K-Fold cross-validation to train multiple LightGBM models for each target variable, with each model corresponding to a different fold. The trained models are stored in `model_dict` for later use.
# 

# In[15]:


# Initialize an empty dictionary to store models for each target
model_dict = {}

# Iterate over the target variables "content" and "wording"
for target in targets:
    models = []
    
    for fold in range(CFG.n_splits):

        # Create training and evaluation datasets for LightGBM
        X_train_cv = train[train["fold"] != fold].drop(columns=drop_columns)
        y_train_cv = train[train["fold"] != fold][target]

        X_eval_cv = train[train["fold"] == fold].drop(columns=drop_columns)
        y_eval_cv = train[train["fold"] == fold][target]

        dtrain = lgb.Dataset(X_train_cv, label=y_train_cv)
        dval = lgb.Dataset(X_eval_cv, label=y_eval_cv)

        # Define LightGBM hyperparameters
        params = {
            'boosting_type': 'gbdt',
            'random_state': 42,
            'objective': 'regression',
            'metric': 'rmse',
            'learning_rate': 0.040,
            'max_depth': 4,  # 3
            'lambda_l1': 0.0,
            'lambda_l2': 0.011
        }

        evaluation_results = {}
        # Train the LightGBM model
        model = lgb.train(params,
                          num_boost_round=20000,
                            #categorical_feature = categorical_features,
                          valid_names=['train', 'valid'],
                          train_set=dtrain,
                          valid_sets=dval,
                          callbacks=[
                              lgb.early_stopping(stopping_rounds=30, verbose=True),
                              lgb.log_evaluation(100),
                              lgb.callback.record_evaluation(evaluation_results)
                            ],
                          )
        models.append(model)
    
    model_dict[target] = models


# ## Cross-Validation and Evaluation 🔄📈

# **Explaination**:
# 
# 
# This code calculates evaluation metrics for the trained LightGBM models using K-Fold cross-validation and prints the RMSE (Root Mean Squared Error) for each target variable. 
# 
# 1. `rmses = []`:
#    - Initializes an empty list called `rmses` to store the RMSE values for each target variable.
# 
# 2. `for target in targets:`:
#    - Iterates over the target variables. It seems like `targets` is expected to contain the target variable names, which are `"content"` and `"wording"` in this case.
# 
# 3. `models = model_dict[target]`:
#    - Retrieves the list of trained LightGBM models corresponding to the current target variable (`target`) from the `model_dict` dictionary.
# 
# 4. `preds = []` and `trues = []`:
#    - Initialize empty lists to store the predicted values (`preds`) and true values (`trues`) for the current target variable.
# 
# 5. `for fold, model in enumerate(models):`:
#    - Iterates over the folds and corresponding models for the current target variable.
# 
# 6. `X_eval_cv = train[train["fold"] == fold].drop(columns=drop_columns)` and `y_eval_cv = train[train["fold"] == fold][target]`:
#    - Extracts the evaluation data (`X_eval_cv`) and the corresponding true target values (`y_eval_cv`) for the current fold.
# 
# 7. `pred = model.predict(X_eval_cv)`:
#    - Uses the trained LightGBM model to make predictions on the evaluation data.
# 
# 8. `trues.extend(y_eval_cv)` and `preds.extend(pred)`:
#    - Extends the `trues` and `preds` lists with the true and predicted values, respectively, for the current fold.
# 
# 9. `rmse = np.sqrt(mean_squared_error(trues, preds))`:
#    - Calculates the RMSE (Root Mean Squared Error) by comparing the true values (`trues`) and predicted values (`preds`) for the current target variable.
# 
# 10. `print(f"{target}_rmse : {rmse}")`:
#     - Prints the RMSE value for the current target variable to the console.
# 
# 11. `rmses = rmses + [rmse]`:
#     - Appends the calculated RMSE value to the `rmses` list.
# 
# 12. After the loop for each target variable, the code calculates and prints the mean RMSE (`mcrmse`) across all target variables by averaging the RMSE values stored in the `rmses` list.
# 
# This code calculates and reports the RMSE for each target variable and provides an average RMSE across all targets to evaluate the performance of the LightGBM models in a K-Fold cross-validation setting.

# In[16]:


# Cell Title: Cross-Validation and Evaluation 🔄📊📈

# Initialize a list to store RMSE values for each target
rmses = []

# Iterate over the target variables "content" and "wording"
for target in targets:
    models = model_dict[target]

    preds = []
    trues = []
    
    # Iterate over folds and evaluate models
    for fold, model in enumerate(models):
        X_eval_cv = train[train["fold"] == fold].drop(columns=drop_columns)
        y_eval_cv = train[train["fold"] == fold][target]

        pred = model.predict(X_eval_cv)

        trues.extend(y_eval_cv)
        preds.extend(pred)
        
    # Calculate RMSE for the target
    rmse = np.sqrt(mean_squared_error(trues, preds))
    print(f"{target}_rmse : {rmse}")
    rmses = rmses + [rmse]

# Calculate and print the mean columnwise RMSE (mcrmse)
print(f"mcrmse : {sum(rmses) / len(rmses)}")


# ## Define Columns to Drop for Test Data 📊🔍

# In[17]:


# Define columns to drop from the test dataset
drop_columns = [
    #"fold",
    "student_id", "prompt_id", "text", "fixed_summary_text",
    "prompt_question", "prompt_title", "prompt_text",
    "input"
] + [
    f"content_pred_{i}" for i in range(CFG.n_splits)
] + [
    f"wording_pred_{i}" for i in range(CFG.n_splits)
]


# ## Generating Predictions for Test Data 🚀📊

# In[18]:


# Initialize a dictionary to store predictions for each target
pred_dict = {}

# Iterate over the target variables "content" and "wording"
for target in targets:
    models = model_dict[target]
    preds = []

    # Iterate over folds and generate predictions for the test data
    for fold, model in enumerate(models):
        X_eval_cv = test.drop(columns=drop_columns)

        pred = model.predict(X_eval_cv)
        preds.append(pred)
    
    # Store the predictions for the target in the dictionary
    pred_dict[target] = preds


# ## Combining and Averaging Test Predictions 🚀📊📈

# In[19]:


# Iterate over the target variables "content" and "wording"
for target in targets:
    preds = pred_dict[target]
    
    # Iterate over the predictions for each fold
    for i, pred in enumerate(preds):
        test[f"{target}_pred_{i}"] = pred

    # Calculate the mean prediction for the target across folds and store it in the test dataset
    test[target] = test[[f"{target}_pred_{fold}" for fold in range(CFG.n_splits)]].mean(axis=1)


# In[20]:


test


# ##  Submission.csv 📝

# In[21]:


sample_submission


# In[22]:


# Save the final predictions for "student_id", "content", and "wording" to a CSV submission file
test[["student_id", "content", "wording"]].to_csv("submission.csv", index=False)


# ## Explore More! 👀
# Thank you for exploring this notebook! If you found this notebook insightful or if it helped you in any way, I invite you to explore more of my work on my profile.
# 
# 👉 [Visit my Profile](https://www.kaggle.com/zulqarnainali) 👈
# 
# ## Feedback and Gratitude 🙏
# We value your feedback! Your insights and suggestions are essential for our continuous improvement. If you have any comments, questions, or ideas to share, please don't hesitate to reach out.
# 
# 📬 Contact me via email: [zulqar445ali@gmail.com](mailto:zulqar445ali@gmail.com)
# 
# I would like to express our heartfelt gratitude for your time and engagement. Your support motivates us to create more valuable content.
# 
# Happy coding and best of luck in your data science endeavors! 🚀
# 
