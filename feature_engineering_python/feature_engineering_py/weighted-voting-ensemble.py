#!/usr/bin/env python
# coding: utf-8

# # Notes
# 
# Be sure to check out the original models linked here:
# 
# Oringinal Ensemble: https://www.kaggle.com/code/judith007/ensemble-score-boost-lb-0-763
# Original Models: https://www.kaggle.com/code/nlztrk/openbook-debertav3-large-baseline-single-model
# https://www.kaggle.com/code/hycloud/2023kagglellm-deberta-v3-large-model1-inference
# 
# And of course no notebook would be complete for this competition at this moment without giving a shoutout to @radek1
# 
# A number of improvements could still be made here really anywhere along the pipeline. A number of good discussions have come out regaurding prompt engineering, which would undoubtably increase data quality and thus the model scores. There could be other models added into the mix, I have seen a lot of deberta-v3-large and would expect to see some other big ones in there too. I would also suggest that voting ensemble is just 1 of many ways to make a submission and I would imagine that there are many other ways to make an even more successful version! Perhaps we will save that for later!

# In[1]:


# installing offline dependencies
get_ipython().system('pip install -U /kaggle/input/faiss-gpu-173-python310/faiss_gpu-1.7.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl')
get_ipython().system('cp -rf /kaggle/input/sentence-transformers-222/sentence-transformers /kaggle/working/sentence-transformers')
get_ipython().system('pip install -U /kaggle/working/sentence-transformers')
get_ipython().system('pip install -U /kaggle/input/blingfire-018/blingfire-0.1.8-py3-none-any.whl')

get_ipython().system('pip install --no-index --no-deps /kaggle/input/llm-whls/transformers-4.31.0-py3-none-any.whl')
get_ipython().system('pip install --no-index --no-deps /kaggle/input/llm-whls/peft-0.4.0-py3-none-any.whl')
get_ipython().system('pip install --no-index --no-deps /kaggle/input/llm-whls/datasets-2.14.3-py3-none-any.whl')
get_ipython().system('pip install --no-index --no-deps /kaggle/input/llm-whls/trl-0.5.0-py3-none-any.whl')


# In[2]:


import os
import gc
import pandas as pd
import numpy as np
import re
from tqdm.auto import tqdm
import blingfire as bf
from __future__ import annotations

from collections.abc import Iterable

import faiss
from faiss import write_index, read_index

from sentence_transformers import SentenceTransformer

import torch
import ctypes
libc = ctypes.CDLL("libc.so.6")


# In[3]:


def process_documents(documents: Iterable[str],
                      document_ids: Iterable,
                      split_sentences: bool = True,
                      filter_len: int = 3,
                      disable_progress_bar: bool = False) -> pd.DataFrame:
    """
    Main helper function to process documents from the EMR.

    :param documents: Iterable containing documents which are strings
    :param document_ids: Iterable containing document unique identifiers
    :param document_type: String denoting the document type to be processed
    :param document_sections: List of sections for a given document type to process
    :param split_sentences: Flag to determine whether to further split sections into sentences
    :param filter_len: Minimum character length of a sentence (otherwise filter out)
    :param disable_progress_bar: Flag to disable tqdm progress bar
    :return: Pandas DataFrame containing the columns `document_id`, `text`, `section`, `offset`
    """
    
    df = sectionize_documents(documents, document_ids, disable_progress_bar)

    if split_sentences:
        df = sentencize(df.text.values, 
                        df.document_id.values,
                        df.offset.values, 
                        filter_len, 
                        disable_progress_bar)
    return df


def sectionize_documents(documents: Iterable[str],
                         document_ids: Iterable,
                         disable_progress_bar: bool = False) -> pd.DataFrame:
    """
    Obtains the sections of the imaging reports and returns only the 
    selected sections (defaults to FINDINGS, IMPRESSION, and ADDENDUM).

    :param documents: Iterable containing documents which are strings
    :param document_ids: Iterable containing document unique identifiers
    :param disable_progress_bar: Flag to disable tqdm progress bar
    :return: Pandas DataFrame containing the columns `document_id`, `text`, `offset`
    """
    processed_documents = []
    for document_id, document in tqdm(zip(document_ids, documents), total=len(documents), disable=disable_progress_bar):
        row = {}
        text, start, end = (document, 0, len(document))
        row['document_id'] = document_id
        row['text'] = text
        row['offset'] = (start, end)

        processed_documents.append(row)

    _df = pd.DataFrame(processed_documents)
    if _df.shape[0] > 0:
        return _df.sort_values(['document_id', 'offset']).reset_index(drop=True)
    else:
        return _df


def sentencize(documents: Iterable[str],
               document_ids: Iterable,
               offsets: Iterable[tuple[int, int]],
               filter_len: int = 3,
               disable_progress_bar: bool = False) -> pd.DataFrame:
    """
    Split a document into sentences. Can be used with `sectionize_documents`
    to further split documents into more manageable pieces. Takes in offsets
    to ensure that after splitting, the sentences can be matched to the
    location in the original documents.

    :param documents: Iterable containing documents which are strings
    :param document_ids: Iterable containing document unique identifiers
    :param offsets: Iterable tuple of the start and end indices
    :param filter_len: Minimum character length of a sentence (otherwise filter out)
    :return: Pandas DataFrame containing the columns `document_id`, `text`, `section`, `offset`
    """

    document_sentences = []
    for document, document_id, offset in tqdm(zip(documents, document_ids, offsets), total=len(documents), disable=disable_progress_bar):
        try:
            _, sentence_offsets = bf.text_to_sentences_and_offsets(document)
            for o in sentence_offsets:
                if o[1]-o[0] > filter_len:
                    sentence = document[o[0]:o[1]]
                    abs_offsets = (o[0]+offset[0], o[1]+offset[0])
                    row = {}
                    row['document_id'] = document_id
                    row['text'] = sentence
                    row['offset'] = abs_offsets
                    document_sentences.append(row)
        except:
            continue
    return pd.DataFrame(document_sentences)


# In[4]:


SIM_MODEL = '/kaggle/input/sentencetransformers-allminilml6v2/sentence-transformers_all-MiniLM-L6-v2'
DEVICE = 0 #torch.device('cuda')
MAX_LENGTH = 384
BATCH_SIZE = 16

WIKI_PATH = "/kaggle/input/wikipedia-20230701"
wiki_files = os.listdir(WIKI_PATH)


# # Relevant Title Retrieval

# In[5]:


trn = pd.read_csv("/kaggle/input/kaggle-llm-science-exam/test.csv").drop("id", 1)
trn.head()


# In[6]:


model2 = SentenceTransformer(SIM_MODEL, device=torch.device('cuda'))
model2.max_seq_length = MAX_LENGTH
model2 = model2.half()


# In[7]:


sentence_index = read_index("/kaggle/input/wikipedia-2023-07-faiss-index/wikipedia_202307.index")


# In[8]:


prompt_embeddings = model2.encode(trn.prompt.values, batch_size=BATCH_SIZE, device=DEVICE, show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True)
prompt_embeddings = prompt_embeddings.detach().cpu().numpy()
_ = gc.collect()


# In[9]:


## Get the top 3 pages that are likely to contain the topic of interest
search_score, search_index = sentence_index.search(prompt_embeddings, 3)


# In[10]:


## Save memory - delete sentence_index since it is no longer necessary
del sentence_index
del prompt_embeddings
_ = gc.collect()
libc.malloc_trim(0)


# # Getting Sentences from the Relevant Titles

# In[11]:


df = pd.read_parquet("/kaggle/input/wikipedia-20230701/wiki_2023_index.parquet",
                     columns=['id', 'file'])


# In[12]:


## Get the article and associated file location using the index
wikipedia_file_data = []

for i, (scr, idx) in tqdm(enumerate(zip(search_score, search_index)), total=len(search_score)):
    scr_idx = idx
    _df = df.loc[scr_idx].copy()
    _df['prompt_id'] = i
    wikipedia_file_data.append(_df)
wikipedia_file_data = pd.concat(wikipedia_file_data).reset_index(drop=True)
wikipedia_file_data = wikipedia_file_data[['id', 'prompt_id', 'file']].drop_duplicates().sort_values(['file', 'id']).reset_index(drop=True)

## Save memory - delete df since it is no longer necessary
del df
_ = gc.collect()
libc.malloc_trim(0)


# In[13]:


## Get the full text data
wiki_text_data = []

for file in tqdm(wikipedia_file_data.file.unique(), total=len(wikipedia_file_data.file.unique())):
    _id = [str(i) for i in wikipedia_file_data[wikipedia_file_data['file']==file]['id'].tolist()]
    _df = pd.read_parquet(f"{WIKI_PATH}/{file}", columns=['id', 'text'])

    _df_temp = _df[_df['id'].isin(_id)].copy()
    del _df
    _ = gc.collect()
    libc.malloc_trim(0)
    wiki_text_data.append(_df_temp)
wiki_text_data = pd.concat(wiki_text_data).drop_duplicates().reset_index(drop=True)
_ = gc.collect()


# In[14]:


## Parse documents into sentences
processed_wiki_text_data = process_documents(wiki_text_data.text.values, wiki_text_data.id.values)


# In[15]:


## Get embeddings of the wiki text data
wiki_data_embeddings = model2.encode(processed_wiki_text_data.text,
                                    batch_size=BATCH_SIZE,
                                    device=DEVICE,
                                    show_progress_bar=True,
                                    convert_to_tensor=True,
                                    normalize_embeddings=True)#.half()
wiki_data_embeddings = wiki_data_embeddings.detach().cpu().numpy()


# In[16]:


_ = gc.collect()


# In[17]:


## Combine all answers
trn['answer_all'] = trn.apply(lambda x: " ".join([x['A'], x['B'], x['C'], x['D'], x['E']]), axis=1)


## Search using the prompt and answers to guide the search
trn['prompt_answer_stem'] = trn['prompt'] + " " + trn['answer_all']


# In[18]:


trn


# In[19]:


question_embeddings = model2.encode(trn.prompt_answer_stem.values, batch_size=BATCH_SIZE, device=DEVICE, show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True)
question_embeddings = question_embeddings.detach().cpu().numpy()


# # Extracting Matching Prompt-Sentence Pairs

# In[20]:


## Parameter to determine how many relevant sentences to include
NUM_SENTENCES_INCLUDE = 5

## List containing just Context
contexts = []

for r in tqdm(trn.itertuples(), total=len(trn)):

    prompt_id = r.Index

    prompt_indices = processed_wiki_text_data[processed_wiki_text_data['document_id'].isin(wikipedia_file_data[wikipedia_file_data['prompt_id']==prompt_id]['id'].values)].index.values

    if prompt_indices.shape[0] > 0:
        prompt_index = faiss.index_factory(wiki_data_embeddings.shape[1], "Flat")
        prompt_index.add(wiki_data_embeddings[prompt_indices])

        context = ""
        
        ## Get the top matches
        ss, ii = prompt_index.search(question_embeddings, NUM_SENTENCES_INCLUDE)
        for _s, _i in zip(ss[prompt_id], ii[prompt_id]):
            context += processed_wiki_text_data.loc[prompt_indices]['text'].iloc[_i] + " "
        
    contexts.append(context)


# In[21]:


trn['context'] = contexts


# In[22]:


trn[["prompt", "context", "A", "B", "C", "D", "E"]].to_csv("./test_context.csv", index=False)


# In[23]:


trn


# # Inference

# In[24]:


model_dir = "/kaggle/input/llm-se-debertav3-large"


# In[25]:


from dataclasses import dataclass
from typing import Optional, Union

import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy


# In[26]:


test_df = pd.read_csv("test_context.csv")
test_df.index = list(range(len(test_df)))
test_df.id = list(range(len(test_df)))
test_df["prompt"] = test_df["context"] + " #### " +  test_df["prompt"]


# In[27]:


test_df['answer'] = 'A'
test_ds = Dataset.from_pandas(test_df)


# In[28]:


tokenizer = AutoTokenizer.from_pretrained(model_dir)
model2 = AutoModelForMultipleChoice.from_pretrained(model_dir)
model2.eval()


# In[29]:


import numpy as np
def predictions_to_map_output(predictions):
    sorted_answer_indices = np.argsort(-predictions)
    top_answer_indices = sorted_answer_indices[:,:3] # Get the first three answers in each row
    top_answers = np.vectorize(index_to_option.get)(top_answer_indices)
    return np.apply_along_axis(lambda row: ' '.join(row), 1, top_answers)


# In[30]:


# We'll create a dictionary to convert option names (A, B, C, D, E) into indices and back again
options = 'ABCDE'
indices = list(range(5))

option_to_index = {option: index for option, index in zip(options, indices)}
index_to_option = {index: option for option, index in zip(options, indices)}

def preprocess(example):
    # The AutoModelForMultipleChoice class expects a set of question/answer pairs
    # so we'll copy our question 5 times before tokenizing
    first_sentence = [example['prompt']] * 5
    second_sentence = []
    for option in options:
        second_sentence.append(example[option])
    # Our tokenizer will turn our text into token IDs BERT can understand
    tokenized_example = tokenizer(first_sentence, second_sentence, truncation=True)
    tokenized_example['label'] = option_to_index[example['answer']]
    return tokenized_example


# In[31]:


@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    
    def __call__(self, features):
        label_name = "label" if 'label' in features[0].keys() else 'labels'
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]['input_ids'])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch['labels'] = torch.tensor(labels, dtype=torch.int64)
        return batch


# In[32]:


trainer = Trainer(
    model=model2,
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer)
)


# In[33]:


tokenized_test_ds = test_ds.map(preprocess, batched=False, remove_columns=['prompt', 'A', 'B', 'C', 'D', 'E', 'answer'])


# In[34]:


test_predictions = trainer.predict(tokenized_test_ds)


# In[35]:


print(type(test_predictions.predictions))
test_predictions.predictions


# In[36]:


from typing import Optional, Union
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from dataclasses import dataclass
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer, AutoModel
from torch.utils.data import DataLoader

deberta_v3_large = '/kaggle/input/deberta-v3-large-hf-weights'


# In[37]:


best_preds = torch.tensor(test_predictions.predictions)


# In[38]:


option_to_index = {option: idx for idx, option in enumerate('ABCDE')}
index_to_option = {v: k for k,v in option_to_index.items()}

def preprocess(example):
    first_sentence = [example['prompt']] * 5
    second_sentences = [example[option] for option in 'ABCDE']
    tokenized_example = tokenizer(first_sentence, second_sentences, truncation=False)
    tokenized_example['label'] = option_to_index[example['answer']]
    
    return tokenized_example

@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    
    def __call__(self, features):
        label_name = 'label' if 'label' in features[0].keys() else 'labels'
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]['input_ids'])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch['labels'] = torch.tensor(labels, dtype=torch.int64)
        return batch 


# In[39]:


tokenizer = AutoTokenizer.from_pretrained(deberta_v3_large)

test_df = pd.read_csv('/kaggle/input/kaggle-llm-science-exam/test.csv')
test_df['answer'] = 'A' # dummy answer that allows us to preprocess the test datataset using functionality that works for the train set

tokenized_test_dataset = Dataset.from_pandas(test_df.drop(columns=['id'])).map(preprocess, remove_columns=['prompt', 'A', 'B', 'C', 'D', 'E', 'answer'])
data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)
test_dataloader = DataLoader(tokenized_test_dataset, 10, shuffle=False, collate_fn=data_collator)


# In[40]:


get_ipython().run_cell_magic('time', '', "\nall_preds_my_runs = []\nfor i in range(3):\n    model = AutoModelForMultipleChoice.from_pretrained(f'/kaggle/input/science-exam-trained-model-weights/run_{i}').cuda()\n    model.eval()\n    preds = []\n    for batch in test_dataloader:\n        for k in batch.keys():\n            batch[k] = batch[k].cuda()\n        with torch.no_grad():\n            outputs = model(**batch)\n        preds.append(outputs.logits.cpu().detach())\n\n    preds = torch.cat(preds)\n    all_preds_my_runs.append(preds)\n\nall_preds_my_runs = torch.stack(all_preds_my_runs)\n")


# In[41]:


model = AutoModelForMultipleChoice.from_pretrained(f'/kaggle/input/2023kagglellm-deberta-v3-large-model1').cuda()
model.eval()
preds = []
for batch in test_dataloader:
    for k in batch.keys():
        batch[k] = batch[k].cuda()
    with torch.no_grad():
        outputs = model(**batch)
    preds.append(outputs.logits.cpu().detach())

hyc_preds = torch.cat(preds)


# In[42]:


model = AutoModelForMultipleChoice.from_pretrained(f'/kaggle/input/my-1-epoch').cuda()
model.eval()
preds = []
for batch in test_dataloader:
    for k in batch.keys():
        batch[k] = batch[k].cuda()
    with torch.no_grad():
        outputs = model(**batch)
    preds.append(outputs.logits.cpu().detach())

kb_preds = torch.cat(preds)


# In[43]:


all_preds_my_runs.shape, hyc_preds.shape, kb_preds.shape, best_preds.shape


# In[44]:


from collections import defaultdict

voting_ensemble = defaultdict(list)

for i_preds in range(all_preds_my_runs.shape[0]):
    for row in range(all_preds_my_runs.shape[1]):
        preds = all_preds_my_runs[i_preds][row]
        voting_ensemble[row].append(preds.argsort(descending=True)[:3])


# In[45]:


for row in range(hyc_preds.shape[0]):
    preds = hyc_preds[row]
    voting_ensemble[row].append(preds.argsort(descending=True)[:3])


# In[46]:


for row in range(kb_preds.shape[0]):
    preds = kb_preds[row]
    voting_ensemble[row].append(preds.argsort(descending=True)[:3])


# In[47]:


for row in range(best_preds.shape[0]):
    preds = best_preds[row]
    voting_ensemble[row].append(preds.argsort(descending=True)[:3])


# In[48]:


predictions = []
for i_preds in range(all_preds_my_runs.shape[1]):
        # 3 model from 
    votes = defaultdict(lambda: 0)
    for preds in voting_ensemble[i_preds][:3]:
        votes[preds[0].item()] += 3
        votes[preds[1].item()] += 2
        votes[preds[2].item()] += 1
        
    hyc_preds = voting_ensemble[i_preds][3]
    votes[hyc_preds[0].item()] += 3 * 3.1
    votes[hyc_preds[1].item()] += 2 * 2.85 
    votes[hyc_preds[2].item()] += 1 * 2.85 
    
    kb_preds = voting_ensemble[i_preds][4]
    votes[kb_preds[0].item()] += 3 * 3.0
    votes[kb_preds[1].item()] += 2 * 3.0 
    votes[kb_preds[2].item()] += 1 * 3.0 
    
    best_preds = voting_ensemble[i_preds][5]
    votes[best_preds[0].item()] += 3 * 3.3
    votes[best_preds[1].item()] += 2 * 3.0 
    votes[best_preds[2].item()] += 1 * 3.0 
    
    predictions.append([t[0] for t in sorted(votes.items(), key=lambda x:x[1], reverse=True)][:3])


# In[49]:


predictions[:5]


# In[50]:


predictions_as_answer_letters = np.array(list('ABCDE'))[predictions]
predictions_as_answer_letters[:3]


# In[51]:


predictions_as_string = test_df['prediction'] = [
    ' '.join(row) for row in predictions_as_answer_letters[:, :3]
]
predictions_as_string[:3]


# In[52]:


submission = test_df[['id', 'prediction']]
submission.to_csv('submission.csv', index=False)

pd.read_csv('submission.csv').head()

