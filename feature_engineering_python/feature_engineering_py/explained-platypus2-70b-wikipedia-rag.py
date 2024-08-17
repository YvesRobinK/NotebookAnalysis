#!/usr/bin/env python
# coding: utf-8

# # Explained Platypus2-70B + Wikipedia RAG
# 
# ## Introduction 🌟
# Welcome to this Jupyter notebook developed for Kaggle - LLM Science Exam This notebook is designed to help you participate in the competition and to Use LLMs to answer difficult science questions
# 
# 
# ### Inspiration and Credits 🙌
# This notebook is inspired by the work of simjeg
# , available at [this Kaggle project](https://www.kaggle.com/code/simjeg/platypus2-70b-with-wikipedia-rag/notebook). I extend my gratitude to simjeg
#  for sharing their insights and code.
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
# 
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
# We acknowledge the Kaggleorganizers for providing the dataset and the competition platform.
# 
# Let's get started! Feel free to reach out if you have any questions or need assistance along the way.
# 👉 [Visit my Profile](https://www.kaggle.com/zulqarnainali) 👈
# 
# 
# 

# ## 📦 Install offline dependencies
# 
# 
# 

# **Explanation:**
# - In this cell, you are installing two specific Python packages from local files.
# - The `!pip install` command is used to install Python packages.
# - The `-U` flag is used to upgrade the packages if they are already installed.
# - `--no-deps` flag is used to skip installing dependencies since you are installing from local files.
# - The paths to the `.whl` files are provided after `--no-deps` to specify the location of the packages.
# 

# In[1]:


get_ipython().system('pip install -U --no-deps /kaggle/input/faiss-gpu-173-python310/faiss_gpu-1.7.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl')
get_ipython().system('pip install -U --no-deps /kaggle/input/datasets-214/datasets-2.14.5-py3-none-any.whl')


# ## Importing Libraries and Setting Constants
# 

# **Explanation:**
# - This cell imports various libraries and sets constants that will be used throughout the notebook.
# - Libraries include standard Python libraries (e.g., `gc`, `logging`, `time`) and popular data science libraries (e.g., `numpy`, `pandas`, `torch`).
# - Constants like `NUM_TITLES`, `MAX_SEQ_LEN`, and `MODEL_PATH` are defined here for easy adjustment and reference throughout the notebook.
# 

# In[2]:


import gc
import logging
from time import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import ctypes
from functools import partial

import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# For RAG
import faiss
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_from_disk, Dataset

NUM_TITLES = 5
MAX_SEQ_LEN = 512
MODEL_PATH = "/kaggle/input/bge-small-faiss/"

# For LLM
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModel
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from safetensors.torch import load_file

N_BATCHES = 5 
MAX_CONTEXT = 2750
MAX_LENGTH = 4096


#  ## 📂Function to Clean Memory and Data Loading

# **Explanation:**
# - This cell defines a function `clean_memory()` to clear RAM and vRAM (video RAM) using garbage collection, memory trimming, and CUDA memory clearing.
# - Data is loaded from a CSV file into a Pandas DataFrame using `pd.read_csv()`. The file path and index column are specified.
# - The `IS_TEST_SET` variable controls whether the notebook runs on a full dataset or a smaller subset (for testing purposes).
# - There's an option to uncomment a block of code to load the train set and adjust `IS_TEST_SET` and `N_BATCHES` accordingly.
# 

# In[3]:


# Function to clean RAM & vRAM
def clean_memory():
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()

# Load data
df = pd.read_csv("/kaggle/input/kaggle-llm-science-exam/test.csv", index_col="id")

# Variable used to avoid running the notebook for 3 hours when submitting. Credit : CPMP
IS_TEST_SET = len(df) != 200

# Uncomment this to see results on the train set
# df = pd.read_csv("/kaggle/input/kaggle-llm-science-exam/train.csv", index_col="id")
# IS_TEST_SET = True
# N_BATCHES = 1


# ## 📦 SentenceTransformer Class

# **Explanation:**
# - This cell defines a custom `SentenceTransformer` class used for encoding sentences into embeddings using a pre-trained model.
# - The class has an `__init__` method for initializing the model, a `transform` method for preprocessing sentences, a `get_dataloader` method for creating a DataLoader for sentences, and an `encode` method for encoding sentences into embeddings.
# - In the `__init__` method, the pre-trained model and tokenizer are loaded based on the specified checkpoint.
# - The `transform` method tokenizes and preprocesses sentences using the tokenizer.
# - The `get_dataloader` method prepares a DataLoader for a list of sentences.
# - The `encode` method encodes sentences into embeddings using the loaded model.
# 
# 
# 

# In[4]:


class SentenceTransformer:
    def __init__(self, checkpoint, device="cuda:0"):
        self.device = device
        self.checkpoint = checkpoint
        self.model = AutoModel.from_pretrained(checkpoint).to(self.device).half()
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def transform(self, batch):
        tokens = self.tokenizer(batch["text"], truncation=True, padding=True, return_tensors="pt", max_length=MAX_SEQ_LEN)
        return tokens.to(self.device)  

    def get_dataloader(self, sentences, batch_size=32):
        sentences = ["Represent this sentence for searching relevant passages: " + x for x in sentences]
        dataset = Dataset.from_dict({"text": sentences})
        dataset.set_transform(self.transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return dataloader

    def encode(self, sentences, show_progress_bar=False, batch_size=32):
        dataloader = self.get_dataloader(sentences, batch_size=batch_size)
        pbar = tqdm(dataloader) if show_progress_bar else dataloader

        embeddings = []
        for batch in pbar:
            with torch.no_grad():
                e = self.model(**batch).pooler_output
                e = F.normalize(e, p=2, dim=1)
                embeddings.append(e.detach().cpu().numpy())
        embeddings = np.concatenate(embeddings, axis=0)
        return embeddings


# ## Processing and Extracting Context for Test Set**
# 
# **Explanation:**
# - This cell processes and extracts context for the test set if `IS_TEST_SET` is `True`.
# - It starts by loading an embedding model (`SentenceTransformer`) and initializing a timer (`start`) to measure the elapsed time.
# - The prompts and answer choices are combined into a single string using a lambda function and applied to the DataFrame, resulting in `inputs` for embedding.
# - A Faiss index is loaded to perform efficient nearest neighbor search for text matching.
# - Text search is performed using Faiss by searching for the closest sentences to the prompt embeddings.
# - Context is extracted from a pre-loaded dataset, and each entry in the `df` DataFrame is updated with the extracted context.
# - Memory is freed, including resetting the Faiss index, deleting variables, and running the `clean_memory()` function.
# - The elapsed time for the entire process is printed.
# 
# **Note:**
# - The Faiss library is used for efficient similarity search, which can significantly speed up the search for relevant passages in a large dataset.
# 

# In[5]:


if IS_TEST_SET:
    # Load embedding model
    start = time()
    print(f"Starting prompt embedding, t={time() - start :.1f}s")
    model = SentenceTransformer(MODEL_PATH, device="cuda:0")

    # Get embeddings of prompts
    f = lambda row : " ".join([row["prompt"], row["A"], row["B"], row["C"], row["D"], row["E"]])
    inputs = df.apply(f, axis=1).values # better results than prompt only
    prompt_embeddings = model.encode(inputs, show_progress_bar=False)

    # Search closest sentences in the wikipedia index 
    print(f"Loading faiss index, t={time() - start :.1f}s")
    faiss_index = faiss.read_index(MODEL_PATH + '/faiss.index')
    # faiss_index = faiss.index_cpu_to_all_gpus(faiss_index) # causes OOM, and not that long on CPU

    print(f"Starting text search, t={time() - start :.1f}s")
    search_index = faiss_index.search(np.float32(prompt_embeddings), NUM_TITLES)[1]

    print(f"Starting context extraction, t={time() - start :.1f}s")
    dataset = load_from_disk("/kaggle/input/all-paraphs-parsed-expanded")
    for i in range(len(df)):
        df.loc[i, "context"] = "-" + "\n-".join([dataset[int(j)]["text"] for j in search_index[i]])

    # Free memory
    faiss_index.reset()
    del faiss_index, prompt_embeddings, model, dataset
    clean_memory()
    print(f"Context added, t={time() - start :.1f}s")


# ##  Creating Symlinks from Kaggle Datasets to Cached Model
# 
# 
# **Explanation:**
# - This cell creates symbolic links (symlinks) from dataset files in Kaggle to a cached model checkpoint directory.
# - It starts by defining the `checkpoint_path`, which is the directory where the symlinks will be created.
# - The script then loops through two parts (part 1 and part 2).
# - For each part, it defines the `source_dir`, which is the directory containing dataset files.
# - It then iterates through files in the `source_dir`.
# - For each file, it attempts to create a symlink in the `checkpoint_path` directory with the same name, pointing to the original file in the `source_dir`.
# - Any exceptions that occur during symlink creation are caught and ignored.
# 
# **Note:**
# - This code is useful for setting up symlinks between Kaggle dataset files and a cached model directory, potentially reducing the need to download data repeatedly.
# 

# In[6]:


# Create symlinks from kaggle datasets to fake cached model

checkpoint_path = Path("/root/.cache/")
checkpoint_path.mkdir(exist_ok=True, parents=True)

for part in [1, 2]:
    source_dir = Path(f"/kaggle/input/platypus2-70b-instruct-part{part}")
    for path in source_dir.glob("*"):
        try:
            (checkpoint_path / path.name).symlink_to(path)
        except:
            pass


#  ## 🦙ShardedLlama Class for Language Model

# **Explaination**:
# 
# 1. `class ShardedLlama:`: Defines a Python class named `ShardedLlama`.
# 
# 2. `def __init__(self, checkpoint_path, device="cuda:0", dtype=torch.float16):`: Defines the constructor for the `ShardedLlama` class, which initializes an instance of the class.
# 
#   
# 
#  - `checkpoint_path`: The path to the checkpoint.
#    - `device` (optional): The device to use (default is "cuda:0" for GPU).
#    - `dtype` (optional): The data type for tensors (default is torch.float16).
# 
# 3. Inside the constructor, various attributes are initialized:
#    - `self.checkpoint_path`: Stores the provided checkpoint path as a `Path` object.
#    - `self.device`: Stores the specified device for computation.
#    - `self.dtype`: Stores the specified data type for tensors.
#    - `self.config`: Loads the configuration from the specified checkpoint path using `AutoConfig`.
#    - `self.tokenizer`: Initializes a tokenizer using `AutoTokenizer` from the checkpoint.
#    - Sets the tokenizer's `pad_token` to the tokenizer's `eos_token` and sets `padding_side` to "right."
#    - Calls the `init_model` method to initialize the model.
# 
# 4. `def init_model(self):`: Defines a method for initializing the model.
# 
#    - Loads the meta-model (without using memory) using the `init_empty_weights()` context manager.
#    - Initializes the model using `AutoModelForCausalLM.from_config`.
#    - Ties the model's weights.
#    - Stores the layers of the model in `self.layers`.
#    - Moves model buffers to the specified device.
# 
# 5. `def load_layer(self, layer_name):`: Defines a method for loading a specific layer of the model.
# 
#    - Loads the state dictionary from a file based on the provided `layer_name` and device.
#    - Iterates through the parameters in the state dictionary and moves them to the specified device.
# 
# 6. `def __call__(self, inputs, output_token):`: Defines the `__call__` method to make instances of `ShardedLlama` callable.
# 
#    - `inputs` is a list of tuples, where each tuple contains a prefix and suffix.
#    - `output_token` is an index used to select an output token.
# 
# 7. Inside the `__call__` method:
#    - The model is rebooted to ensure that buffers are loaded, and memory is clean.
#    - The input batch is sent to the specified device.
#    - `n_suffixes` is the number of suffixes in the batch, and `suffix_eos` calculates the position of the end-of-sequence token in each suffix.
# 
# 8. Attention mask and position IDs are created for model inputs.
# 
# 9. The code uses a `ThreadPoolExecutor` to parallelize the loading of model layers.
# 
# 10. For each layer:
#     - It loads the current layer and waits for the previous layer to be loaded.
#     - For each input in the batch:
#       - If it's the "model.embed_tokens" layer, the prefix and suffix are passed through this layer.
#       - If it's the "model.norm" layer, only the last token in each suffix is kept.
#       - If it's the "lm_head" layer, predictions are made for the specified output token.
#       - For other layers, both prefix and suffix are processed, and intermediate results are stored.
# 
# 11. After processing each layer, the previous layer is removed from memory.
# 
# 12. The final batch of results is returned.
# 
# This `ShardedLlama` class is designed for language modeling, where the model is split into layers and each layer is loaded sequentially to optimize memory usage. The `__call__` method takes a batch of inputs and processes them through the model, providing the results for further analysis.

# In[7]:


# Class for sharded llama

class ShardedLlama:
    def __init__(self, checkpoint_path, device="cuda:0", dtype=torch.float16):
        """
        Sharded version of LlamaForCausalLM : the model is splitted into layer shards to reduce GPU memory usage.
        During the forward pass, the inputs are processed layer by layer, and the GPU memory is freed after each layer.
        To avoid loading the layers multiple times, we could save all the intermediate activations in RAM, but
        as Kaggle accelerators have more GPU memory than CPU, we simply batch the inputs and keep them on the GPU.

        Parameters
        ----------
        checkpoint_path : str or Path
            path to the checkpoint
        device : str, optional
            device, by default "cuda:0"
        dtype : torch.dtype, optional
            dtype, by default torch.float16
        """
        
        # Save parameters
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device 
        self.dtype = dtype

        # Create model
        self.config = AutoConfig.from_pretrained(self.checkpoint_path)
        # For flash attention when Turing architecture will be supported : https://github.com/Dao-AILab/flash-attention/issues/542
        # self.config.auto_map = {"AutoModelForCausalLM" : "togethercomputer/LLaMA-2-7B-32K--modeling_flash_llama.LlamaForCausalLM"} 
        
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.init_model()
        self.layer_names = ["model.embed_tokens"] + [f"model.layers.{i}" for i in range(len(self.model.model.layers))] + ["model.norm", "lm_head"]

    def init_model(self):
    
        # Load meta model (no memory used)
        with init_empty_weights():
            self.model = AutoModelForCausalLM.from_config(self.config, trust_remote_code=True)
            self.model.tie_weights()
            
        self.layers = [self.model.model.embed_tokens] + list(self.model.model.layers) + [self.model.model.norm, self.model.lm_head]
            
        # Move buffers to device (not that much GPU memory used)
        for buffer_name, buffer in self.model.named_buffers():
            set_module_tensor_to_device(self.model, buffer_name, self.device, value=buffer, dtype=self.dtype)

    def load_layer(self, layer_name):
        state_dict = load_file(self.checkpoint_path / (layer_name + ".safetensors"), device=self.device)
        for param_name, param in state_dict.items():
            assert param.dtype != torch.int8, "int8 not supported (need to add fp16_statistics)"
            set_module_tensor_to_device(self.model, param_name, self.device, value=param, dtype=self.dtype)

    def __call__(self, inputs, output_token):
        # inputs = [(prefix, suffix), ...] with prefix.shape[0] = 1 and suffix.shape[0] = 5
        
        # Reboot the model to make sure buffers are loaded and memory is clean
        del self.model
        clean_memory()
        self.init_model()
        
       # Send batch to device
        batch = [(prefix.to(self.device), suffix.to(self.device)) for prefix, suffix in inputs]
        n_suffixes = len(batch[0][1])
        suffix_eos = [(suffix != self.tokenizer.pad_token_id).sum(1) - 1 for _, suffix in inputs]

        # Create attention mask for the largest input, and position ids to use KV cache
        attention_mask = torch.finfo(self.dtype).min * torch.ones(MAX_LENGTH, MAX_LENGTH)
        attention_mask = attention_mask.triu(diagonal=1)[None, None, ...]
        attention_mask = attention_mask.to(self.device)
        position_ids = torch.arange(MAX_LENGTH, dtype=torch.long, device=self.device)[None, :]

        with ThreadPoolExecutor() as executor, torch.inference_mode():

            # Load first layer
            #future = executor.submit(self.load_layer, "model.embed_tokens")
            self.load_layer("model.embed_tokens")

            for i, (layer_name, layer) in tqdm(enumerate(zip(self.layer_names, self.layers)), desc=self.device, total=len(self.layers)):

                # Wait for previous layer to be loaded and load next layer
                #future.result()
                if (i + 1) < len(self.layer_names):
                    #future = executor.submit(self.load_layer, self.layer_names[i + 1])
                    self.load_layer(self.layer_names[i + 1])

                # Run layer
                for j, (prefix, suffix) in enumerate(batch):
                    if layer_name == "model.embed_tokens":
                        batch[j] = (layer(prefix), layer(suffix))
                    elif layer_name == "model.norm":
                        # Only keep the last token at this point
                        batch[j] = (None, layer(suffix[torch.arange(n_suffixes), suffix_eos[j]][:, None]))
                    elif layer_name == "lm_head":
                        batch[j] = layer(suffix)[:, 0, output_token].detach().cpu().numpy()
                    else:
                        # Run prefix
                        len_p, len_s = prefix.shape[1], suffix.shape[1]
                        new_prefix, (k_cache, v_cache) = layer(prefix, use_cache=True, attention_mask=attention_mask[:, :, -len_p:, -len_p:])
                        
                        # Run suffix
                        pos = position_ids[:, len_p:len_p + len_s].repeat(n_suffixes, 1)
                        attn = attention_mask[:, :, -len_s:, -len_p - len_s:].repeat(n_suffixes, 1, 1, 1)
                        kv_cache = (k_cache.repeat(n_suffixes, 1, 1, 1), v_cache.repeat(n_suffixes, 1, 1, 1))
                        new_suffix = layer(suffix, past_key_value=kv_cache, position_ids=pos, attention_mask=attn)[0]
                        batch[j] = (new_prefix, new_suffix)

                # Remove previous layer from memory (including buffers)
                layer.to("meta")
                clean_memory() # proposed by CPMP

        # Get scores
        return batch


# ## Running the Model on Multiple GPUs

# 
# ```python
# # Define a function to get tokens for the model input
# def get_tokens(row, tokenizer):
#     system_prefix = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_prefix}"
#     instruction = "Your task is to analyze the question and answer below. If the answer is correct, respond yes, if it is not correct respond no. As a potential aid to your answer, background context from Wikipedia articles is at your disposal, even if they might not always be relevant."
#     input_prefix = f"Context: {row['context'][:MAX_CONTEXT]}\nQuestion: {row['prompt']}\nProposed answer: "
#     prompt_prefix = system_prefix.format(instruction=instruction, input_prefix=input_prefix)
#     prefix = tokenizer(prompt_prefix, return_tensors="pt", return_attention_mask=False, truncation=True, max_length=MAX_LENGTH)["input_ids"]
#     prompt_suffix = [f"{row[letter]}\n\n### Response:\n" for letter in "ABCDE"]
#     suffix = tokenizer(prompt_suffix, return_tensors="pt", return_attention_mask=False, truncation=True, max_length=MAX_LENGTH, padding=True)["input_ids"][:, 1:]
#     return prefix, suffix
# ```
# 
# - This code defines a function named `get_tokens` that takes two arguments: `row` and `tokenizer`.
# - `system_prefix` contains a template for a system prefix, which is a text block describing the task and input context.
# - `instruction` is a string with instructions for the task.
# - `input_prefix` is generated using information from the input data `row`.
# - `prompt_prefix` combines `instruction` and `input_prefix` by formatting the `system_prefix` template with the provided values.
# - `prefix` is generated by tokenizing `prompt_prefix` using the `tokenizer`, converting it to PyTorch tensors, and extracting the input IDs.
# - `prompt_suffix` is a list of strings generated for each answer choice A, B, C, D, and E.
# - `suffix` is generated similarly to `prefix` but for `prompt_suffix`, and the padding token is removed from the start.
# 
# ```python
# # Define a function to run the model on a device
# def run_model(device, df):
#     model = ShardedLlama(checkpoint_path, device=f"cuda:{device}")
#     f = partial(get_tokens, tokenizer=model.tokenizer)
#     inputs = df.apply(f, axis=1).values
#     batches = np.array_split(inputs, N_BATCHES)
#     outputs = []
#     for i, batch in enumerate(batches):
#         # Token #4874 is yes.
#         outputs += model(batch, output_token=4874)
#     return outputs
# ```
# 
# - This code defines a function named `run_model` that takes two arguments: `device` and `df` (DataFrame).
# - Inside the function, a `model` is initialized using the `ShardedLlama` class, specifying the GPU device.
# - A partial function `f` is created using `get_tokens` and the model's tokenizer. This partial function will be used to process inputs.
# - Inputs for the model are generated by applying the `f` function to each row of the DataFrame `df`.
# - The inputs are split into batches using `np.array_split`.
# - The function initializes an empty list called `outputs` to collect model outputs.
# - It then iterates through the batches, running the model on each batch and appending the results to the `outputs` list.
# 
# ```python
# # Run model
# if IS_TEST_SET: 
#     with ThreadPoolExecutor() as executor:
#         outputs = list(executor.map(run_model, [0, 1], np.array_split(df, 2)))
#         outputs = sum(outputs, [])
#         
#     # Save results
#     n = len(df)
#     for i, scores in enumerate(outputs):
#         top3 = np.argsort(scores)[::-1]
#         df.loc[i, "prediction"] = " ".join(["ABCDE"[j] for j in top3])
#     
#     # Display performances if train set is used (in this case use IS_TEST_SET=True !)
#     if "answer" in df.columns:
#         for i in range(n):
#             df.loc[i, "top_1"] = df.loc[i, "prediction"][0]
#             df.loc[i, "top_2"] = df.loc[i, "prediction"][2]
#             df.loc[i, "top_3"] = df.loc[i, "prediction"][4]
# 
#         top_i = [(df[f"top_{i}"] == df["answer"]).sum() for i in [1, 2, 3]]
#         print(f"top1 : {top_i[0]}/{n}, top2 : {top_i[1]}/{n}, top3 : {top_i[2]}/{n} (total={sum(top_i)} / {n})")
#         print(f"Accuracy: {100*top_i[0]/n:.1f}%, map3: {100*(top_i[0] + top_i[1]*1/2 + top_i[2]*1/3).sum()/n:.1f}%")
# else:
#     df["prediction"] = "A B C"
# 
# df[["prediction"]].to_csv("submission.csv")
# ```
# 
# - This part of the code is conditional on the value of `IS_TEST_SET`. If it's `True`, the model is run
# 
#  on the test set.
# - A `ThreadPoolExecutor` is used to run the `run_model` function on two GPUs concurrently.
# - The outputs from the model are collected and combined into a single list.
# - The results are saved back to the DataFrame `df`, and for each row, the top 3 predictions are determined and stored.
# - If the DataFrame contains an "answer" column, performance metrics such as accuracy and mean average precision (map3) are calculated and printed.
# - If `IS_TEST_SET` is `False`, a default prediction of "A B C" is assigned to each row in the DataFrame.
# - Finally, the DataFrame's "prediction" column is saved to a CSV file named "submission.csv."
# 
# This code effectively runs the language model on the given data and produces predictions or evaluation metrics depending on the context.

# In[8]:


# Run model on the 2 GPUs

def get_tokens(row, tokenizer):
        system_prefix = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_prefix}"
        instruction = "Your task is to analyze the question and answer below. If the answer is correct, respond yes, if it is not correct respond no. As a potential aid to your answer, background context from Wikipedia articles is at your disposal, even if they might not always be relevant."
        input_prefix = f"Context: {row['context'][:MAX_CONTEXT]}\nQuestion: {row['prompt']}\nProposed answer: "
        prompt_prefix = system_prefix.format(instruction=instruction, input_prefix=input_prefix)
        prefix = tokenizer(prompt_prefix, return_tensors="pt", return_attention_mask=False, truncation=True, max_length=MAX_LENGTH)["input_ids"]
        prompt_suffix = [f"{row[letter]}\n\n### Response:\n" for letter in "ABCDE"]
        suffix = tokenizer(prompt_suffix, return_tensors="pt", return_attention_mask=False, truncation=True, max_length=MAX_LENGTH, padding=True)["input_ids"][:, 1:]
        return prefix, suffix

def run_model(device, df):
    model = ShardedLlama(checkpoint_path, device=f"cuda:{device}")
    f = partial(get_tokens, tokenizer=model.tokenizer)
    inputs = df.apply(f, axis=1).values
    batches = np.array_split(inputs, N_BATCHES)
    outputs = []
    for i, batch in enumerate(batches):
        # Token #4874 is yes.
        outputs += model(batch, output_token=4874)
    return outputs

# Run model
if IS_TEST_SET: 
    with ThreadPoolExecutor() as executor:
        outputs = list(executor.map(run_model, [0, 1], np.array_split(df, 2)))
        outputs = sum(outputs, [])
        
    # Save results
    n = len(df)
    for i, scores in enumerate(outputs):
        top3 = np.argsort(scores)[::-1]
        df.loc[i, "prediction"] = " ".join(["ABCDE"[j] for j in top3])
    
    # Display performances if train set is used (in this case use IS_TEST_SET=True !)
    if "answer" in df.columns:
        for i in range(n):
            df.loc[i, "top_1"] = df.loc[i, "prediction"][0]
            df.loc[i, "top_2"] = df.loc[i, "prediction"][2]
            df.loc[i, "top_3"] = df.loc[i, "prediction"][4]

        top_i = [(df[f"top_{i}"] == df["answer"]).sum() for i in [1, 2, 3]]
        print(f"top1 : {top_i[0]}/{n}, top2 : {top_i[1]}/{n}, top3 : {top_i[2]}/{n} (total={sum(top_i)} / {n})")
        print(f"Accuracy: {100*top_i[0]/n:.1f}%, map3: {100*(top_i[0] + top_i[1]*1/2 + top_i[2]*1/3).sum()/n:.1f}%")
else:
    df["prediction"] = "A B C"

df[["prediction"]].to_csv("submission.csv")


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
