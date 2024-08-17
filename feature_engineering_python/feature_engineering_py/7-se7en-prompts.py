#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import Image
from IPython.core.display import HTML 

Image(url="https://www.nme.com/wp-content/uploads/2020/09/Brad_Pitt_Se7en.jpg")


# # Comprehensive Analysis and Modeling Notebook
# 
# Welcome to this comprehensive notebook where we dive deep into the world of data analysis and machine learning. This document is meticulously crafted to guide you through various stages of data processing, modeling, and prediction. Here's what to expect:
# 
# ## What This Notebook Offers:
# 1. **Data Preprocessing**: Initial steps to clean and prepare the data for analysis.
# 2. **Exploratory Data Analysis (EDA)**: Insights and patterns unraveled through visual and statistical methods.
# 3. **Feature Engineering**: Enhancing the dataset with new, informative features.
# 4. **Model Development**: Implementation of various machine learning models, including both traditional and advanced techniques.
# 5. **Evaluation and Optimization**: Assessing model performance and tuning them for better accuracy.
# 6. **Ensemble Techniques**: Leveraging the power of multiple models to improve predictions.
# 7. **Final Predictions and Submission**: Preparing the final predictions for submission, demonstrating the practical application of our analysis.
# 
# ## Intended Audience:
# This notebook is designed for both beginners and experienced practitioners in the field of data science. Whether you're looking to learn new skills, seeking to understand specific methodologies, or aiming to apply advanced techniques in machine learning, this notebook has something to offer.
# 
# ## Feedback and Collaboration:
# Your feedback is highly appreciated! If you have any suggestions, questions, or ideas for improvement, please feel free to share. Collaboration is the key to success in the ever-evolving field of data science, and your input is invaluable.
# 
# ---
# 
# Let's embark on this data science journey together and uncover the stories hidden within the data!
# 

# In[2]:


import sys
import gc

import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.metrics import roc_auc_score
import numpy as np
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
    SentencePieceBPETokenizer
)

from datasets import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier


# This cell checks if we are rerunning a Kaggle competition kernel:
# - If `KAGGLE_IS_COMPETITION_RERUN` is set in the environment, it indicates a rerun, and we execute the corresponding logic
# - Otherwise, we read the sample submission file using pandas, and then write our submission file. This step is crucial for making submissions in Kaggle competitions.
# - `sys.exit()` is called to gracefully exit the script when not in a rerun mode. This is important to prevent executing further cells unnecessarily after creating the submission file. It will save you time and even though the notebooks takes hours during submission, it will only use a few seconds of your GPU
# 

# In[3]:


import os
import pandas as pd

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    pass
else:
    sub = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/sample_submission.csv')
    sub.to_csv('submission.csv', index=False)
    sys.exit()


# ### Sentence Transformers
# 
# This cell is dedicated to initializing the tokenizer and model from the Hugging Face `transformers` library for natural language processing tasks:
# - We import `AutoTokenizer` and `AutoModel` from the `transformers` package, and `torch` from PyTorch.
# - The `AutoTokenizer.from_pretrained` method is used to load a pre-trained tokenizer for the MiniLM model (`all-MiniLM-L6-v2`). MiniLM is known for its compact size and efficiency while maintaining strong performance, making it ideal for various NLP tasks.
# - Similarly, `AutoModel.from_pretrained` loads the corresponding MiniLM model. This model is designed for high performance in a wide range of NLP applications.
# - We check the type of the loaded model using `type(model)` to confirm it's correctly loaded and then display the model's architecture and configuration by simply calling `model`. This provides insights into the model's structure, such as its layers and parameters, which is useful for understanding its capabilities and potential customizations.
# 

# In[ ]:


from transformers import AutoTokenizer, AutoModel
import torch
tokenizer = AutoTokenizer.from_pretrained('/kaggle/input/sentence-transformers/minilm-l6-v2/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('/kaggle/input/sentence-transformers/minilm-l6-v2/all-MiniLM-L6-v2')
type(model)
model


# Now read several files and put them together. Starting with the famous daigt-v2 dataset: https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset

# In[ ]:


test = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/test_essays.csv')
sub = pd.read_csv('/kaggle/input/llm-detect-ai-generated-text/sample_submission.csv')
train_1 = pd.read_csv("/kaggle/input/daigt-v2-train-dataset/train_v2_drcat_02.csv", sep=',')[['text', 'label']]
train_1.shape


# Adding the 7 prompt dataset: https://www.kaggle.com/datasets/carlmcbrideellis/llm-7-prompt-training-dataset

# In[ ]:


train = pd.read_csv("/kaggle/input/llm-7-prompt-training-dataset/train_essays_RDizzl3_seven_v1.csv")
print(train.shape)
train = pd.concat([train, train_1])
print(train.shape)


# In[ ]:


train.head()


# In[ ]:


train = train.drop_duplicates(subset=['text'])
train.reset_index(drop=True, inplace=True)


# This cell contains the setup and definition of a function to generate embeddings from text using a pre-trained language model:
# 
# - First, we set the `device` to either 'cuda' or 'cpu' depending on whether CUDA (GPU support) is available. Using 'cuda' enables faster computation on GPU if it's available.
# 
# - We then move our pre-trained model to the selected device (GPU or CPU). This step ensures that the model computations are optimized for the available hardware.
# 
# - The `get_embeddings` function is defined to convert input text into embeddings:
#   - The text is tokenized using the pre-trained tokenizer. Special tokens are added, and the text is truncated as needed for the model.
#   - The tokenized text is then passed through the model to get the last hidden state. This step is done within a `torch.inference_mode()` context, which reduces memory usage and increases computation speed by disabling gradient calculations.
#   - We compute the mean of the last hidden state across all tokens to get a single embedding vector for the input text. This embedding represents the entire input text in a high-dimensional space and captures its semantic features.
#   - The embedding is moved back to the CPU and converted to a NumPy array before being returned. This is done for compatibility with Python's standard data processing tools and for potential use in non-GPU environments.
# 

# In[ ]:


from tqdm import tqdm
device = 'cuda' if torch.cuda.is_available() else 'cpu'  ## for faster computation

model.to(device)

from tqdm import tqdm
device = 'cuda' if torch.cuda.is_available() else 'cpu'  ## for faster computation

model.to(device)
def get_embeddings(texts, batch_size=16):
    # Ensure texts is a list
    if not isinstance(texts, list):
        raise ValueError("Input must be a list of texts")

    # Initialize the progress bar
    pbar = tqdm(total=len(texts), desc='Processing', unit='text')

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]

        # Tokenize all texts in the batch
        inputs = tokenizer.batch_encode_plus(batch_texts,
                                             add_special_tokens=True,
                                             truncation=True,
                                             padding=True,  # Pad to the longest in the batch
                                             return_tensors='pt')
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        # Process in batch
        with torch.inference_mode():
            outputs = model(input_ids, attention_mask=attention_mask)
            last_hidden_states = outputs.last_hidden_state

        # Calculate embeddings for each text
        embeddings = torch.mean(last_hidden_states, dim=1)
        all_embeddings.extend(embeddings.cpu().numpy())  # Move the embeddings back to CPU

        # Update the progress bar
        pbar.update(len(batch_texts))

    pbar.close()
    return all_embeddings


# In this cell, we use the previously defined `get_embeddings` function to generate embeddings for our training dataset:
# 
# - We start by converting the 'text' column of the `train` DataFrame into a list. This conversion is necessary because our `get_embeddings` function is designed to take a list of texts as input.
# - The `get_embeddings` function is then called with this list of texts. It processes each text through the pre-trained language model to generate embeddings.
# - These embeddings capture the semantic essence of each text in a high-dimensional space, making them suitable for various machine learning tasks, including classification, clustering, and similarity comparison.
# - The resulting `train_embeddings` variable holds the embeddings for the entire training dataset, which can be used for further machine learning or data analysis tasks.
# 
# This step is crucial in converting raw text data into a numerical format that machine learning algorithms can efficiently process.
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', "train_embeddings = get_embeddings(train['text'].tolist())\n")


# In[ ]:


get_ipython().run_cell_magic('time', '', "test_embeddings = get_embeddings(test['text'].tolist())\n")


# This cell is focused on computing the cosine similarity between a set of specific topics and the texts in our training and test datasets:
# 
# - We start by defining a list of seven topics from the PERSUADE dataset. These range from political systems and space exploration to technological advancements and unique human experiences and are written as a general description of what these texts contain.
# 
# - The `get_embeddings` function, previously defined, is used to generate embeddings for these topics. This transforms the thematic textual descriptions into numerical vectors in a high-dimensional semantic space.
# 
# - The embeddings for both the training texts (`train_embeddings`) and the topics (`topic_embeddings`) are normalized using `normalize` from `sklearn.preprocessing`. Normalization is probably not necessary for meaningful cosine similarity calculations, but I just want it ensures that the length of each vector is 1, allowing the cosine similarity to effectively measure the angle between the vectors.
# 
# - `cosine_similarity` from `sklearn.metrics.pairwise` is then used to calculate the cosine similarity between the normalized embeddings of the training texts and the topics. This results in a matrix (`train_cosine_similarities`) where each element represents the similarity between a topic and a training text.
# 
# - The same process is applied to the test dataset embeddings (`test_embeddings`) to create `test_cosine_similarities`.
# 
# - We add 1 to the cosine similarity values to shift the range from [-1, 1] to [0, 2]. This adjustment can be useful for subsequent processing steps, especially those that might assume non-negative input.
# 
# - The resulting `train_cosine_similarities` matrix is displayed for inspection.
# 
# - Finally, to manage memory usage, embeddings and normalized data are deleted, and garbage collection is invoked.
# 
# This step is crucial for understanding the relationship between the predefined topics and the corpus of texts, potentially aiding in tasks such as topic modeling, recommendation systems, or thematic text clustering.
# 

# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import numpy as np

# Assuming train_embeddings are already calculated for your earlier texts
# and train['text'] is your list of earlier texts

# Topics
topics = [
    "the advantages of limiting car usage",
    "Arguments in favor of the electoral college system or in favor of changing the electoral college system to a popular vote for the president of the United States",
    "The challenge of Exploring Venus: It is a worthy pursuit despite of the risks it represents",
    "Unmasking the face on Mars: Convincing someone that The Face on Mars is a natural landform and not created by aliens",
    "Making Mona Lisa smile: Facial action coding system as a technology that enables computers to identify human emotions. How valuable it to Open the doors to evaluate human expressions of students in the classroom",
    "A Cowboy Who Rode the Waves: writing an article from Luke's point of view to convince others to participate in the seagoing cowboys program, which allowed Luke to experience adventures and visit many unique places",
    "Driverless cars are coming: arguments for and agains the developments of these cars taking into account the advantages and disadvantages of doing so."
]

# Generate embeddings for the topics
topic_embeddings = get_embeddings(topics)

# Normalize the embeddings
train_embeddings_normalized = normalize(train_embeddings)
test_embeddings_normalized = normalize(test_embeddings)
topic_embeddings_normalized = normalize(topic_embeddings)

# Calculate cosine similarity
# This will create a matrix with the similarity of each topic to each text
train_cosine_similarities = 1 + cosine_similarity(train_embeddings_normalized, topic_embeddings_normalized)
test_cosine_similarities = 1 + cosine_similarity(test_embeddings_normalized, topic_embeddings_normalized)

train_cosine_similarities 

del train_embeddings, test_embeddings, topic_embeddings, train_embeddings_normalized, topic_embeddings_normalized, test_embeddings_normalized
_ = gc.collect()


# This cell defines a function `keep_max_similarity_above_threshold_sparse` and applies it to our cosine similarity matrices. The purpose of this function is twofold:
# 
# 1. **Sparse Matrix Creation**: For each row in the input matrix (representing cosine similarities between texts and topics), the function identifies the maximum similarity value. If this value exceeds a specified threshold (1.4 in this case), the function retains this value; otherwise, it's ignored. This selective process is used to create a sparse matrix (CSR format), which is memory-efficient and retains only the most significant similarity scores.
# 
# 2. **Generating Optimization Weights**: Alongside creating the sparse matrix, the function generates an array (`max_values_per_row`) containing the highest similarity score for each row. If the maximum value in a row doesn't exceed the threshold, a default value of 0.25 is assigned. This array can serve as "optimization weights" in subsequent processes, such as weighted machine learning models, where these weights can help emphasize texts with higher thematic relevance based on the defined threshold.
# 
# - The function is applied to both the training (`train_cosine_similarities`) and test (`test_cosine_similarities`) cosine similarity matrices. As a result, we obtain `train_cosine_similarity_sparse` and `test_cosine_similarity_sparse` for compact representation, and `optimization_weights` as potential input for weighted model training.
# 
# - This approach is particularly useful for focusing on highly relevant thematic connections and reducing noise from less significant ones, thus optimizing the dataset for more focused and efficient processing in downstream tasks.
# 

# In[ ]:


import numpy as np
from scipy.sparse import csr_matrix

def keep_max_similarity_above_threshold_sparse(matrix, threshold = 1.4):
    # Find the indices of the maximum values in each row
    max_indices = np.argmax(matrix, axis=1)

    # Extract the maximum values using these indices
    max_values = matrix[np.arange(matrix.shape[0]), max_indices]

    # Mask to identify rows where the maximum value exceeds the threshold
    mask = max_values > threshold

    # Create row indices for the sparse matrix
    row_indices = np.arange(matrix.shape[0])[mask]

    # Filter the column indices and values by the mask
    col_indices = max_indices[mask]
    data = max_values[mask]

    # Create a CSR sparse matrix
    sparse_matrix = csr_matrix((data.astype(np.float16), (row_indices, col_indices)), shape = matrix.shape)
    
    # Create a list of max values for each row, including zeros
    max_values_per_row = np.where(mask, max_values, 1.25)

    return sparse_matrix, max_values_per_row

# Apply this to your cosine similarity matrices
train_cosine_similarity_sparse, optimization_weights = keep_max_similarity_above_threshold_sparse(train_cosine_similarities)
test_cosine_similarity_sparse, _ = keep_max_similarity_above_threshold_sparse(test_cosine_similarities)


# In[ ]:


LOWERCASE = False
VOCAB_SIZE = 42000


# # Creating Byte-Pair Encoding Tokenizer
# 
# This cell initializes a Byte-Pair Encoding (BPE) tokenizer, a method effective for subword tokenization in NLP tasks. The tokenizer is configured with special tokens like `[UNK]`, `[PAD]`, `[CLS]`, `[SEP]`, and `[MASK]`. We use normalization and pre-tokenization strategies suitable for BPE. The tokenizer is trained on a subset of the dataset iteratively and wrapped in `PreTrainedTokenizerFast` for efficient tokenization. Finally, it's applied to both the test and training text data.
# 

# In[ ]:


# Creating Byte-Pair Encoding tokenizer
raw_tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
raw_tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()] + [normalizers.Lowercase()] if LOWERCASE else [])
raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=special_tokens)
dataset = Dataset.from_pandas(test[['text']])
def train_corp_iter(): 
    for i in range(0, len(dataset), 25):
        yield dataset[i : i + 25]["text"]
raw_tokenizer.train_from_iterator(train_corp_iter(), trainer=trainer)
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=raw_tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)


# In[ ]:


tokenized_texts_test = []

for text in tqdm(test['text'].tolist()):
    tokenized_texts_test.append(tokenizer.tokenize(text))

tokenized_texts_train = []

for text in tqdm(train['text'].tolist()):
    tokenized_texts_train.append(tokenizer.tokenize(text))


# # TF-IDF Vectorization with Custom Tokenization
# 
# This cell implements the TF-IDF (Term Frequency-Inverse Document Frequency) vectorization process, customized for specific tokenization needs. The `TfidfVectorizer` is set up with a 3-5 n-gram range and various parameters including sublinear term frequency scaling and unicode accent stripping. Custom functions are used for both tokenization and preprocessing to maintain control over these processes. After fitting the vectorizer to the tokenized test data, we extract the vocabulary. This vocabulary is then used to initialize a new `TfidfVectorizer` which transforms both the training and test datasets. Post-processing, the vectorizer is deleted to free up memory.
# 

# In[ ]:


def dummy(text):
    return text

vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, analyzer = 'word',
    tokenizer = dummy,
    preprocessor = dummy,
    token_pattern = None, strip_accents='unicode')

vectorizer.fit(tokenized_texts_test)

# Getting vocab
vocab = vectorizer.vocabulary_

print(vocab)

vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, vocabulary=vocab,
                            analyzer = 'word',
                            tokenizer = dummy,
                            preprocessor = dummy,
                            token_pattern = None, strip_accents='unicode'
                            )

tf_train = vectorizer.fit_transform(tokenized_texts_train)
tf_test = vectorizer.transform(tokenized_texts_test)

del vectorizer
gc.collect()


# In[ ]:


from scipy.sparse import hstack, csr_matrix
import numpy as np


#tf_train = hstack([tf_train, train_cosine_similarity_sparse])
#tf_test = hstack([tf_test, test_cosine_similarity_sparse])

del train_cosine_similarity_sparse, test_cosine_similarity_sparse
_ = gc.collect()


# In[ ]:


y_train = train['label'].values


# # Ensemble Learning with Multiple Classifiers
# 
# This cell sets up an ensemble learning model using various classifiers, each with its specific configurations:
# 
# 1. **Multinomial Naive Bayes**: 
#    A `MultinomialNB` classifier with a smoothing parameter `alpha` set to 0.02.
# 
# 2. **SGD Classifier**: 
#    An `SGDClassifier` for linear models with a modified Huber loss function, a maximum of 8000 iterations, and a tolerance of 1e-4 for stopping criteria.
# 
# 3. **LightGBM Classifier**: 
#    An `LGBMClassifier` configured with custom parameters such as learning rate, lambda values, max depth, and more, specified in the `p6` dictionary.
# 
# 4. **CatBoost Classifier**: 
#    A `CatBoostClassifier` with 1000 iterations, silent mode (no verbose output), specific learning rate, L2 regularization, and a subsampling rate of 0.4.
# 
# 5. **Ensemble Model - Voting Classifier**: 
#    A `VotingClassifier` that combines the above models (`MultinomialNB`, `SGDClassifier`, `LGBMClassifier`, `CatBoostClassifier`) using soft voting. The weights for each classifier in the ensemble are specified, with a focus on the three non-Naive Bayes models.
# 
# The ensemble model is then trained on the transformed training data (`tf_train`) and labels (`y_train`). Finally, the ensemble model is used to predict probabilities on the test dataset (`tf_test`), and garbage collection is run to manage memory.
# ``
# 

# In[ ]:


# clf = MultinomialNB(alpha=0.02)
# sgd_model = SGDClassifier(max_iter=8000, tol=1e-4, loss="modified_huber") 
# p6={'n_iter': 2500,'verbose': -1,'objective': 'cross_entropy','metric': 'auc',
#     'learning_rate': 0.00581909898961407, 'colsample_bytree': 0.78,
#     'colsample_bynode': 0.8, 'lambda_l1': 4.562963348932286, 
#     'lambda_l2': 2.97485, 'min_data_in_leaf': 115, 'max_depth': 23, 'max_bin': 898}
# lgb=LGBMClassifier(**p6)
# cat=CatBoostClassifier(iterations=2500,
#                        verbose=0,
#                        l2_leaf_reg=6.6591278779517808,
#                        learning_rate=0.005599066836106983,
#                        subsample = 0.4,
#                        allow_const_label=True,loss_function = 'CrossEntropy')
# weights = [0.07,0.31,0.31,0.31]

# ensemble = VotingClassifier(estimators=[('mnb',clf),
#                                         ('sgd', sgd_model),
#                                         ('lgb',lgb), 
#                                         #('cat', cat)
#                                        ],
#                             weights=weights, voting='soft', n_jobs=-1)
# ensemble.fit(tf_train, y_train)
# gc.collect()
# final_preds_bpe = ensemble.predict_proba(tf_test)[:,1]

# del tokenized_texts_test, tokenized_texts_train, dataset, raw_tokenizer, tokenizer
# _ = gc.collect()


# In[ ]:


def calculate_voting(tf_train, tf_test, y_train, optimization_weights):
    # Initialize classifiers
    clf = MultinomialNB(alpha = 0.02)
    sgd_model = SGDClassifier(max_iter=8000, tol=1e-4, loss="modified_huber") 
    p7={'n_iter': 1500,'verbose': -1,'objective': 'cross_entropy','metric': 'auc',
    'learning_rate': 0.0058, 'colsample_bytree': 0.78,
    'colsample_bynode': 0.8, 'lambda_l1': 4.563, 
    'lambda_l2': 2.97, 'min_data_in_leaf': 112, 'max_depth': 22, 'max_bin': 900}
    lgb = LGBMClassifier(**p7)
    cat=CatBoostClassifier(iterations=1000,
                       verbose=0,
                       l2_leaf_reg=6.659,
                       learning_rate=0.0056,
                       subsample = 0.4,
                       allow_const_label=True,loss_function = 'CrossEntropy')
    # Fit classifiers and make predictions
    clf.fit(tf_train, y_train)
    predictions_mnb = clf.predict_proba(tf_test)[:, 1]
    del clf

    sgd_model.fit(tf_train, y_train)
    predictions_sgd = sgd_model.predict_proba(tf_test)[:, 1]
    del sgd_model
    
    lgb.fit(tf_train, y_train, sample_weight = optimization_weights)
    predictions_lgb = lgb.predict_proba(tf_test)[:, 1]
    print('done with lightgbm')
    del lgb
    _ = gc.collect()
    
    cat.fit(tf_train, y_train, sample_weight = optimization_weights)
    predictions_cat = cat.predict_proba(tf_test)[:, 1]
    print('done with catboost')
    del cat
    
    # Define weights
    weights = [0.07,0.31,0.31,0.31]

    # Calculate weighted average of predictions
    final_preds = (weights[0] * predictions_mnb + weights[1] * predictions_sgd + weights[2] * predictions_lgb + weights[3] * predictions_cat) / sum(weights)

    # Garbage collection
    gc.collect()

    return final_preds
final_preds_bpe = calculate_voting(tf_train, tf_test, y_train, optimization_weights)
_ = gc.collect()


# # Integrating Sentencepiece Encoding with Machine Learning Models
# 
# ## Tokenizer Setup
# 1. **Sentencepiece Tokenizer Initialization**:
#    A `SentencePieceBPETokenizer` is initialized for subword tokenization.
# 
# 2. **Normalization and Pre-Tokenization**:
#    The tokenizer is configured with NFC normalization and optional lowercase conversion based on the `LOWERCASE` flag. Byte level pre-tokenization is also applied.
# 
# 3. **Special Tokens and Training**:
#    Special tokens like `[UNK]`, `[PAD]`, `[CLS]`, `[SEP]`, `[MASK]` are added. The tokenizer is trained on the test dataset using a custom generator function that iterates over the dataset in chunks.
# 
# 4. **Tokenization**:
#    The tokenizer processes both test and training datasets to create tokenized text data.
# 
# ## TF-IDF Vectorization
# 1. **Vectorization Setup**:
#    A `TfidfVectorizer` with a 3-5 n-gram range and custom tokenizer and preprocessor (`dummy` function) is used. The vectorizer is first fitted to the tokenized test data.
# 
# 2. **Vocabulary Extraction and Transformation**:
#    The vocabulary from the test data vectorization is extracted and used to initialize a new `TfidfVectorizer`. This vectorizer then transforms both training and test tokenized texts.
# 
# 3. **Memory Management**:
#    The vectorizer is deleted, and garbage collection is run to manage memory.
# 
# ## Model Training and Prediction
# 1. **Model Initialization**:
#    Multiple models including `MultinomialNB`, `SGDClassifier`, `LGBMClassifier`, and `CatBoostClassifier` are initialized with specific parameters.
# 
# 2. **Ensemble Model Creation**:
#    A `VotingClassifier` ensemble, using soft voting and specified weights, combines the aforementioned models.
# 
# 3. **Training and Prediction**:
#    The ensemble model is trained on the TF-IDF transformed training data and labels. It then predicts probabilities on the transformed test dataset.
# 
# 4. **Final Predictions and Cleanup**:
#    Predicted probabilities (`final_preds_spe`) are stored, and memory cleanup is performed with garbage collection.
# 

# In[ ]:


# # Creating Sentencepiece Encoding tokenizer
# raw_tokenizer = SentencePieceBPETokenizer()

# # Adding normalization and pre_tokenizer
# raw_tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()] + [normalizers.Lowercase()] if LOWERCASE else [])
# raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
# # Adding special tokens and creating trainer instance
# special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]

# # Creating huggingface dataset object
# dataset = Dataset.from_pandas(test[['text']])

# def train_corp_iter():
#     """
#     A generator function for iterating over a dataset in chunks.
#     """    
#     for i in range(0, len(dataset), 300):
#         yield dataset[i : i + 300]["text"]

# # Training from iterator REMEMBER it's training on test set...
# raw_tokenizer.train_from_iterator(train_corp_iter())

# tokenizer = PreTrainedTokenizerFast(
#     tokenizer_object = raw_tokenizer,
#     unk_token="[UNK]",
#     pad_token="[PAD]",
#     cls_token="[CLS]",
#     sep_token="[SEP]",
#     mask_token="[MASK]",
# )

# tokenized_texts_test = []

# # Tokenize test set with new tokenizer
# for text in tqdm(test['text'].tolist()):
#     tokenized_texts_test.append(tokenizer.tokenize(text))

# # Tokenize train set
# tokenized_texts_train = []

# for text in tqdm(train['text'].tolist()):
#     tokenized_texts_train.append(tokenizer.tokenize(text))
    
# vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, analyzer = 'word',
#     tokenizer = dummy,
#     preprocessor = dummy,
#     token_pattern = None, strip_accents='unicode')

# vectorizer.fit(tokenized_texts_test)

# # Getting vocab
# vocab = vectorizer.vocabulary_

# print(vocab)

# vectorizer = TfidfVectorizer(ngram_range=(3, 5), lowercase=False, sublinear_tf=True, vocabulary=vocab,
#                             analyzer = 'word',
#                             tokenizer = dummy,
#                             preprocessor = dummy,
#                             token_pattern = None, strip_accents='unicode'
#                             )

# tf_train = vectorizer.fit_transform(tokenized_texts_train)
# tf_test = vectorizer.transform(tokenized_texts_test)

# del vectorizer
# gc.collect()

# y_train = train['label'].values

# def calculate_voting(tf_train, tf_test, y_train):
#     # Initialize classifiers
#     clf = MultinomialNB(alpha = 0.02)
#     sgd_model = SGDClassifier(max_iter=8000, tol=1e-4, loss="modified_huber") 
#     p7={'n_iter': 1500,'verbose': -1,'objective': 'cross_entropy','metric': 'auc',
#     'learning_rate': 0.0058, 'colsample_bytree': 0.78,
#     'colsample_bynode': 0.8, 'lambda_l1': 4.563, 
#     'lambda_l2': 2.97, 'min_data_in_leaf': 112, 'max_depth': 22, 'max_bin': 900}
#     lgb = LGBMClassifier(**p7)
#     cat=CatBoostClassifier(iterations=1000,
#                        verbose=0,
#                        l2_leaf_reg=6.659,
#                        learning_rate=0.0056,
#                        subsample = 0.4,
#                        allow_const_label=True,loss_function = 'CrossEntropy')
#     # Fit classifiers and make predictions
#     clf.fit(tf_train, y_train)
#     predictions_mnb = clf.predict_proba(tf_test)[:, 1]
#     del clf

#     sgd_model.fit(tf_train, y_train)
#     predictions_sgd = sgd_model.predict_proba(tf_test)[:, 1]
#     del sgd_model
    
#     lgb.fit(tf_train, y_train)
#     predictions_lgb = lgb.predict_proba(tf_test)[:, 1]
#     print('done with lightgbm')
#     del lgb
#     _ = gc.collect()
    
#     cat.fit(tf_train, y_train)
#     predictions_cat = cat.predict_proba(tf_test)[:, 1]
#     print('done with catboost')
#     del cat
    
#     # Define weights
#     weights = [0.07,0.31,0.31,0.31]

#     # Calculate weighted average of predictions
#     final_preds = (weights[0] * predictions_mnb + weights[1] * predictions_sgd + weights[2] * predictions_lgb + weights[3] * predictions_cat) / sum(weights)

#     # Garbage collection
#     gc.collect()

#     return final_preds
# final_preds_spe = calculate_voting(tf_train, tf_test, y_train)
# _ = gc.collect()
# ###############################


# Adjustments for type 1, 2 errors

# In[ ]:


# # Create a boolean mask for values greater than 0.5
# mask = final_preds_bpe > 0.5

# # Add 0.05 only to the values greater than 0.5
# final_preds_bpe[mask] += 0.05

# # Clip values to be between 0 and 1
# final_preds_bpe = np.clip(final_preds_bpe, 0, 1)


# In[ ]:


# # Create a boolean mask for values less than 0.5
# mask = final_preds_bpe < 0.5

# # Subtract 0.05 only from the values less than 0.5
# final_preds_bpe[mask] -= 0.05

# # Clip values to be between 0 and 1
# final_preds_bpe = np.clip(final_preds_bpe, 0, 1)


# This section is for those who believe that their model has no idea when the probabilities are 0.45, 0.55, so you might as well score it as 50/50

# In[ ]:


# Create a boolean mask for values between 0.45 and 0.55
mask = (final_preds_bpe >= 0.45) & (final_preds_bpe <= 0.55)

# Set values in this range to 0.5
final_preds_bpe[mask] = 0.5


# # Final Submission and Closing Remarks
# 
# ## Submission Preparation
# In this final cell, we prepare our submission:
# 
# 1. **Ensemble Prediction Averaging**:
#    We combine the predictions from both the Byte-Pair Encoding (BPE) and Sentencepiece Encoding (SPE) models by averaging them. This approach helps in harnessing the strengths of both models.
# 
#    ```python
#    sub['generated'] = (final_preds_bpe + final_preds_spe) / 2
# 

# In[ ]:


# Assign to the final solution
sub['generated'] = final_preds_bpe
sub.to_csv('submission.csv', index=False)
sub


# In[ ]:




