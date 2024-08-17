#!/usr/bin/env python
# coding: utf-8

# ![undraw_Dashboard_re_3b76.png](attachment:94fffed0-c390-4f7b-af92-e246e58e1fe0.png)
# 
# ## Main Concept
# 
# [以前のnotebook](https://www.kaggle.com/code/tsunotsuno/updated-debertav3-lgbm-with-feature-engineering)で、DebertaとLGBMを組み合わせたモデルを試した。
# 
# しかし、このnotebookではautocorrectを使用しているが、このライブラリのライセンスはLGPLであり、コンペのルールを満たさない可能性が高い。([discussion](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/433208))
# 
# また、[@cody11null](https://www.kaggle.com/cody11null)氏の[notebook](https://www.kaggle.com/code/cody11null/tuned-debertav3-lgbm-autocorrect)によると、さらにパラメータチューニングによって性能を向上をさせることができるようだ。
# 
# 今回は、下記2点でDebertaのモデルを修正し、性能を向上させる。
# 
# - autocorrectを使用しない
# - パラメータチューニング
# 
# In [my previous NOTEBOOK](https://www.kaggle.com/code/tsunotsuno/updated-debertav3-lgbm-with-feature-engineering), we tried a model that combines Deberta and LGBM.
# 
# However, this notebook uses autocorrect which license is LGPL, it seems that the notebook using this library may not meet this competition. ([discussion](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/433208))
# 
# By the way, [@cody11null](https://www.kaggle.com/cody11null)'s [notebook](https://www.kaggle.com/code/cody11null/tuned-debertav3-lgbm- According to [autocorrect](), it seems that performance can be further improved by tuning parameters.
# 
# In this notebook, I will modify Deberta's model in the following two ways to improve performance.
# 
# - not autocorrect
# - parameter tuning
# 
# ### Feature Engineering
# 
# LGBMに使用する特徴量は以前と同じものを使用している。
# 
# I use the same features for LGBM as before.
# 
# [Using features]
# 
# - Text Length
# - Length Ratio
# - Word Overlap
# - N-grams Co-occurrence
#   - count
#   - ratio
# - Quotes Overlap
# - Grammar Check
#   - spelling: pyspellchecker
# 
# ### Model Architecture
# 
# 図のような構成のモデルを作成する。
# 今回はwordingとcontentを同時に学習する。
# 
# I use a model shown in the figure.
# In this notebook, wording and content are learned together.
# 
# ![image.png](attachment:2ff98339-9b9d-4e51-b984-a72d641abb8b.png)
# 
# ### Tuning
# 
# 下記の変更を行う。
# 
# Make the following changes for tuning.
# 
# - content, wordingを同時に学習・推論
# - fp16の使用
# - epoch数を増加（5->10）
# - hidden_dropout_prob(0.005 -> 0.007)
# - attention_probs_dropout_prob(0.005 -> 0.007)
# - auto_find_batch_sizeを使用
# 
# ### References
# 
# - https://www.kaggle.com/code/cody11null/tuned-debertav3-lgbm-autocorrect
# - https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/discussion/433208
# 
# ### My previous notebooks
# 
# - https://www.kaggle.com/code/tsunotsuno/debertav3-baseline-content-and-wording-models
# - https://www.kaggle.com/code/tsunotsuno/debertav3-w-prompt-title-question-fields
# - https://www.kaggle.com/code/tsunotsuno/debertav3-with-llama2-example
# - https://www.kaggle.com/code/tsunotsuno/debertav3-lgbm-with-feature-engineering
# - https://www.kaggle.com/code/tsunotsuno/updated-debertav3-lgbm-with-spell-autocorrect
# - https://www.kaggle.com/code/tsunotsuno/lgbm-debertav3-another-impl
# 

# In[ ]:


get_ipython().system('pip install "/kaggle/input/pyspellchecker/pyspellchecker-0.7.2-py3-none-any.whl"')


# In[ ]:


from typing import List
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
import logging
import os
import gc
import shutil
import json
import transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding
from datasets import Dataset,load_dataset, load_from_disk
from transformers import TrainingArguments, Trainer
from datasets import load_metric, disable_progress_bar
from sklearn.metrics import mean_squared_error
import torch
from sklearn.model_selection import KFold, GroupKFold
from tqdm import tqdm

import nltk
from nltk.corpus import stopwords
from collections import Counter
import spacy
import re
from spellchecker import SpellChecker
import lightgbm as lgb

# logging setting 

warnings.simplefilter("ignore")
logging.disable(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
disable_progress_bar()
tqdm.pandas()


# In[ ]:


# set random seed
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


# In[ ]:


class CFG:
    model_name="debertav3base"
    learning_rate=1.5e-5
    weight_decay=0.02
    hidden_dropout_prob=0.007
    attention_probs_dropout_prob=0.007
    num_train_epochs=10
    n_splits=4
    batch_size=12
    random_seed=42
    save_steps=100
    max_length=512


# ## Dataload

# In[ ]:


DATA_DIR = "/kaggle/input/commonlit-evaluate-student-summaries/"

prompts_train = pd.read_csv(DATA_DIR + "prompts_train.csv")
prompts_test = pd.read_csv(DATA_DIR + "prompts_test.csv")
summaries_train = pd.read_csv(DATA_DIR + "summaries_train.csv")
summaries_test = pd.read_csv(DATA_DIR + "summaries_test.csv")
sample_submission = pd.read_csv(DATA_DIR + "sample_submission.csv")


# summaries_train = summaries_train.head(10) # for dev mode


# ## Preprocess
# 
# 
# [Using features]
# 
# - Text Length
# - Length Ratio
# - Word Overlap
# - N-grams Co-occurrence
# - Quotes Overlap
# - Grammar Check
#   - spelling: pyspellchecker
# 
# NOTE: I don't know why, but I can't use nltk.ngrams function. (my submission fails "Submission CSV Not Found") 
# So, I write and use original ngram function.

# In[ ]:


class Preprocessor:
    def __init__(self, 
                model_name: str,
                ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(f"/kaggle/input/{model_name}")
        self.STOP_WORDS = set(stopwords.words('english'))
        
        self.spacy_ner_model = spacy.load('en_core_web_sm',)
        self.speller = SpellChecker() #Speller(lang='en')
        
    def count_text_length(self, df: pd.DataFrame, col:str) -> pd.Series:
        """ text length """
        tokenizer=self.tokenizer
        return df[col].progress_apply(lambda x: len(tokenizer.encode(x)))

    def word_overlap_count(self, row):
        """ intersection(prompt_text, text) """        
        def check_is_stop_word(word):
            return word in self.STOP_WORDS
        
        prompt_words = row['prompt_tokens']
        summary_words = row['summary_tokens']
        if self.STOP_WORDS:
            prompt_words = list(filter(check_is_stop_word, prompt_words))
            summary_words = list(filter(check_is_stop_word, summary_words))
        return len(set(prompt_words).intersection(set(summary_words)))
            
    def ngrams(self, token, n):
        # Use the zip function to help us generate n-grams
        # Concatentate the tokens into ngrams and return
        ngrams = zip(*[token[i:] for i in range(n)])
        return [" ".join(ngram) for ngram in ngrams]

    def ngram_co_occurrence(self, row, n: int):
        # Tokenize the original text and summary into words
        original_tokens = row['prompt_tokens']
        summary_tokens = row['summary_tokens']

        # Generate n-grams for the original text and summary
        original_ngrams = set(self.ngrams(original_tokens, n))
        summary_ngrams = set(self.ngrams(summary_tokens, n))

        # Calculate the number of common n-grams
        common_ngrams = original_ngrams.intersection(summary_ngrams)

        # # Optionally, you can get the frequency of common n-grams for a more nuanced analysis
        # original_ngram_freq = Counter(ngrams(original_words, n))
        # summary_ngram_freq = Counter(ngrams(summary_words, n))
        # common_ngram_freq = {ngram: min(original_ngram_freq[ngram], summary_ngram_freq[ngram]) for ngram in common_ngrams}

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
        if len(quotes_from_summary)>0:
            return [quote in text for quote in quotes_from_summary].count(True)
        else:
            return 0

    def spelling(self, text):
        
        wordlist=text.split()
        amount_miss = len(list(self.speller.unknown(wordlist)))

        return amount_miss
    
    def run(self, 
            prompts: pd.DataFrame,
            summaries:pd.DataFrame,
            mode:str
        ) -> pd.DataFrame:
        
        # before merge preprocess
        prompts["prompt_length"] = prompts["prompt_text"].apply(
            lambda x: len(self.tokenizer.encode(x))
        )
        prompts["prompt_tokens"] = prompts["prompt_text"].apply(
            lambda x: self.tokenizer.convert_ids_to_tokens(
                self.tokenizer.encode(x), 
                skip_special_tokens=True
            )
        )

        summaries["summary_length"] = summaries["text"].apply(
            lambda x: len(self.tokenizer.encode(x))
        )
        summaries["summary_tokens"] = summaries["text"].apply(
            lambda x: self.tokenizer.convert_ids_to_tokens(
                self.tokenizer.encode(x), 
                skip_special_tokens=True
            )

        )
        summaries["splling_err_num"] = summaries["text"].progress_apply(self.spelling)

        # merge prompts and summaries
        input_df = summaries.merge(prompts, how="left", on="prompt_id")

        # after merge preprocess
        input_df['length_ratio'] = input_df['summary_length'] / input_df['prompt_length']
        
        input_df['word_overlap_count'] = input_df.progress_apply(self.word_overlap_count, axis=1)
        input_df['bigram_overlap_count'] = input_df.progress_apply(
            self.ngram_co_occurrence,args=(2,), axis=1 
        )
        input_df['trigram_overlap_count'] = input_df.progress_apply(
            self.ngram_co_occurrence, args=(3,), axis=1
        )
        
        # Crate dataframe with count of each category NERs overlap for all the summaries
        # Because it spends too much time for this feature, I don't use this time.
#         ners_count_df  = input_df.progress_apply(
#             lambda row: pd.Series(self.ner_overlap_count(row, mode=mode), dtype='float64'), axis=1
#         ).fillna(0)
#         self.ner_keys = ners_count_df.columns
#         ners_count_df['sum'] = ners_count_df.sum(axis=1)
#         ners_count_df.columns = ['NER_' + col for col in ners_count_df.columns]
#         # join ner count dataframe with train dataframe
#         input_df = pd.concat([input_df, ners_count_df], axis=1)
        
        input_df['quotes_count'] = input_df.progress_apply(self.quotes_count, axis=1)
        
        return input_df.drop(columns=["summary_tokens", "prompt_tokens"])
    
preprocessor = Preprocessor(model_name=CFG.model_name)


# In[ ]:


train = preprocessor.run(prompts_train, summaries_train, mode="train")
test = preprocessor.run(prompts_test, summaries_test, mode="test")

test.head()


# In[ ]:


gkf = GroupKFold(n_splits=CFG.n_splits)

for i, (_, val_index) in enumerate(gkf.split(train, groups=train["prompt_id"])):
    train.loc[val_index, "fold"] = i

train.head()


# ## Model Function Definition

# In[ ]:


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    rmse = mean_squared_error(labels, predictions, squared=False)
    return {"rmse": rmse}

def compute_mcrmse(eval_pred):
    """
    Calculates mean columnwise root mean squared error
    https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries/overview/evaluation
    """
    preds, labels = eval_pred

    col_rmse = np.sqrt(np.mean((preds - labels) ** 2, axis=0))
    mcrmse = np.mean(col_rmse)

    return {
        "content_rmse": col_rmse[0],
        "wording_rmse": col_rmse[1],
        "mcrmse": mcrmse,
    }

def compt_score(content_true, content_pred, wording_true, wording_pred):
    content_score = mean_squared_error(content_true, content_pred)**(1/2)
    wording_score = mean_squared_error(wording_true, wording_pred)**(1/2)
    
    return (content_score + wording_score)/2


# ## Deberta Regressor

# In[ ]:


class ScoreRegressor:
    def __init__(self, 
                model_name: str,
                model_dir: str,
                inputs: List[str],
                target_cols: List[str],
                hidden_dropout_prob: float,
                attention_probs_dropout_prob: float,
                max_length: int,
                ):
        
        self.input_col = "input" # col name of model input after text concat sep token
        
        self.input_text_cols = inputs
        self.target_cols = target_cols
        self.model_name = model_name
        self.model_dir = model_dir
        self.max_length = max_length
        
        self.tokenizer = AutoTokenizer.from_pretrained(f"/kaggle/input/{model_name}")
        self.model_config = AutoConfig.from_pretrained(f"/kaggle/input/{model_name}")
        
        self.model_config.update({
            "hidden_dropout_prob": hidden_dropout_prob,
            "attention_probs_dropout_prob": attention_probs_dropout_prob,
            "num_labels": 2,
            "problem_type": "regression",
        })

        self.data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer
        )

    def concatenate_with_sep_token(self, row):
        sep = " " + self.tokenizer.sep_token + " "        
        return sep.join(row[self.input_text_cols])

    def tokenize_function(self, examples: pd.DataFrame):
        labels = [examples["content"], examples["wording"]]
        tokenized = self.tokenizer(examples[self.input_col],
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_length)
        return {
            **tokenized,
            "labels": labels,
        }
    
    def tokenize_function_test(self, examples: pd.DataFrame):
        tokenized = self.tokenizer(examples[self.input_col],
                        padding="max_length",
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
        """fine-tuning"""
                
        train_df[self.input_col] = train_df.apply(self.concatenate_with_sep_token, axis=1)
        valid_df[self.input_col] = valid_df.apply(self.concatenate_with_sep_token, axis=1)        

        train_df = train_df[[self.input_col] + self.target_cols]
        valid_df = valid_df[[self.input_col] + self.target_cols]
        
        model = AutoModelForSequenceClassification.from_pretrained(
            f"/kaggle/input/{self.model_name}", 
            config=self.model_config
        )

        train_dataset = Dataset.from_pandas(train_df, preserve_index=False) 
        val_dataset = Dataset.from_pandas(valid_df, preserve_index=False) 
    
        train_tokenized_datasets = train_dataset.map(self.tokenize_function, batched=False)
        val_tokenized_datasets = val_dataset.map(self.tokenize_function, batched=False)

        # eg. "bert/fold_0/"
        model_fold_dir = os.path.join(self.model_dir, str(fold)) 
        
        training_args = TrainingArguments(
            output_dir=model_fold_dir,
            load_best_model_at_end=True, # select best model
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size, 
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_train_epochs,
            weight_decay=weight_decay,
            report_to='none',
            greater_is_better=False,
            save_strategy="steps",
            evaluation_strategy="steps",
            eval_steps=save_steps,
            save_steps=save_steps,
            metric_for_best_model="mcrmse",
            save_total_limit=1,
            fp16=True,
            auto_find_batch_size=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized_datasets,
            eval_dataset=val_tokenized_datasets,
            tokenizer=self.tokenizer,
            compute_metrics=compute_mcrmse,
            data_collator=self.data_collator
        )

        trainer.train()
        
        model.save_pretrained(self.model_dir)
        self.tokenizer.save_pretrained(self.model_dir)

        model.cpu()
        del model
        gc.collect()
        torch.cuda.empty_cache()

        
    def predict(self, 
                test_df: pd.DataFrame,
                batch_size: int,
                fold: int,
               ):
        """predict content score"""
        
        test_df[self.input_col] = test_df.apply(self.concatenate_with_sep_token, axis=1)

        test_dataset = Dataset.from_pandas(test_df[[self.input_col]], preserve_index=False) 
        test_tokenized_dataset = test_dataset.map(self.tokenize_function_test, batched=False)

        model = AutoModelForSequenceClassification.from_pretrained(f"{self.model_dir}")
        model.eval()
        
        # e.g. "bert/fold_0/"
        model_fold_dir = os.path.join(self.model_dir, str(fold)) 

        test_args = TrainingArguments(
            output_dir=model_fold_dir,
            do_train=False,
            do_predict=True,
            per_device_eval_batch_size=batch_size,
            dataloader_drop_last=False,
            fp16=True,
            auto_find_batch_size=True,
        )

        # init trainer
        infer_content = Trainer(
                      model = model, 
                      tokenizer=self.tokenizer,
                      data_collator=self.data_collator,
                      args = test_args)

        preds = infer_content.predict(test_tokenized_dataset)[0]
        pred_df = pd.DataFrame(
            preds, 
            columns=[
                f"content_pred", 
                f"wording_pred"
           ]
        )
        
        model.cpu()
        del model
        gc.collect()
        torch.cuda.empty_cache()

        return pred_df


# In[ ]:


def train_by_fold(
        train_df: pd.DataFrame,
        model_name: str,
        targets: List[str],
        inputs: List[str],
        save_each_model: bool,
        n_splits: int,
        batch_size: int,
        learning_rate: int,
        hidden_dropout_prob: float,
        attention_probs_dropout_prob: float,
        weight_decay: float,
        num_train_epochs: int,
        save_steps: int,
        max_length:int
    ):

    # delete old model files
    if os.path.exists(model_name):
        shutil.rmtree(model_name)
    
    os.mkdir(model_name)
        
    for fold in range(n_splits):
        print(f"fold {fold}:")
        train_data = train_df[train_df["fold"] != fold]
        valid_data = train_df[train_df["fold"] == fold]
        
        model_dir =  f"{model_name}/fold_{fold}"

        csr = ScoreRegressor(
            model_name=model_name,
            target_cols=targets,
            inputs= inputs,
            model_dir = model_dir, 
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

def validate(
    train_df: pd.DataFrame,
    mode: str,
    targets: List[str],
    inputs: List[str],
    save_each_model: bool,
    n_splits: int,
    batch_size: int,
    model_name: str,
    hidden_dropout_prob: float,
    attention_probs_dropout_prob: float,
    max_length : int
    ) -> pd.DataFrame:
    """predict oof data"""
    
    columns = list(train_df.columns.values)
    
    for fold in range(n_splits):
        print(f"fold {fold}:")
        
        valid_data = train_df[train_df["fold"] == fold]
        
        model_dir =  f"{model_name}/fold_{fold}"
        
        csr = ScoreRegressor(
            model_name=model_name,
            target_cols=targets,
            inputs= inputs,
            model_dir = model_dir,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_length=max_length,
           )
        
        pred_df = csr.predict(
            test_df=valid_data, 
            batch_size=batch_size,
            fold=fold
        )

        train_df.loc[valid_data.index, f"content_{mode}_pred"] = pred_df[f"content_pred"].values
        train_df.loc[valid_data.index, f"wording_{mode}_pred"] = pred_df[f"wording_pred"].values
                
    return train_df[columns + [f"content_{mode}_pred", f"wording_{mode}_pred"]]
    
def predict(
    test_df: pd.DataFrame,
    mode: str,
    targets:List[str],
    inputs: List[str],
    save_each_model: bool,
    n_splits: int,
    batch_size: int,
    model_name: str,
    hidden_dropout_prob: float,
    attention_probs_dropout_prob: float,
    max_length : int
    ):
    """predict using mean folds"""
    
    columns = list(test_df.columns.values)

    for fold in range(n_splits):
        print(f"fold {fold}:")
        
        model_dir =  f"{model_name}/fold_{fold}"

        csr = ScoreRegressor(
            model_name=model_name,
            target_cols=targets,
            inputs= inputs,
            model_dir = model_dir, 
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_length=max_length,
           )
        
        pred_df = csr.predict(
            test_df=test_df, 
            batch_size=batch_size,
            fold=fold
        )

        test_df[f"content_{mode}_pred_{fold}"] = pred_df[f"content_pred"].values
        test_df[f"wording_{mode}_pred_{fold}"] = pred_df[f"wording_pred"].values

    test_df[f"content_{mode}_pred"] = test_df[[f"content_{mode}_pred_{fold}" for fold in range(n_splits)]].mean(axis=1)
    test_df[f"wording_{mode}_pred"] = test_df[[f"wording_{mode}_pred_{fold}" for fold in range(n_splits)]].mean(axis=1)
    
    return test_df[columns + [f"content_{mode}_pred", f"wording_{mode}_pred"]]


# In[ ]:


targets = ["wording", "content"]
mode = "multi"
input_cols = ["prompt_title", "prompt_question", "text"]
model_cfg = CFG

train_by_fold(
    train,
    model_name=model_cfg.model_name,
    save_each_model=False,
    targets=targets,
    inputs=input_cols,
    learning_rate=model_cfg.learning_rate,
    hidden_dropout_prob=model_cfg.hidden_dropout_prob,
    attention_probs_dropout_prob=model_cfg.attention_probs_dropout_prob,
    weight_decay=model_cfg.weight_decay,
    num_train_epochs=model_cfg.num_train_epochs,
    n_splits=CFG.n_splits,
    batch_size=model_cfg.batch_size,
    save_steps=model_cfg.save_steps,
    max_length=model_cfg.max_length
)


train = validate(
    train,
    mode=mode,
    targets=targets,
    inputs=input_cols,
    save_each_model=False,
    n_splits=CFG.n_splits,
    batch_size=model_cfg.batch_size,
    model_name=model_cfg.model_name,
    hidden_dropout_prob=model_cfg.hidden_dropout_prob,
    attention_probs_dropout_prob=model_cfg.attention_probs_dropout_prob,
    max_length=model_cfg.max_length
)

# set validate result
for target in ["content", "wording"]:
    rmse = mean_squared_error(train[target], train[f"{target}_{mode}_pred"], squared=False)
    print(f"cv {target} rmse: {rmse}")

test = predict(
    test,
    mode=mode,
    targets=targets,
    inputs=input_cols,
    save_each_model=False,
    batch_size=model_cfg.batch_size,
    n_splits=CFG.n_splits,
    model_name=model_cfg.model_name,
    hidden_dropout_prob=model_cfg.hidden_dropout_prob,
    attention_probs_dropout_prob=model_cfg.attention_probs_dropout_prob,
    max_length=model_cfg.max_length
)


# In[ ]:


train.head()


# ## LGBM model

# In[ ]:


targets = ["content", "wording"]

drop_columns = ["fold", "student_id", "prompt_id", "text", 
                "prompt_question", "prompt_title", 
                "prompt_text"
               ] + targets


# In[ ]:


model_dict = {}

for target in targets:
    models = []
    
    for fold in range(CFG.n_splits):

        X_train_cv = train[train["fold"] != fold].drop(columns=drop_columns)
        y_train_cv = train[train["fold"] != fold][target]

        X_eval_cv = train[train["fold"] == fold].drop(columns=drop_columns)
        y_eval_cv = train[train["fold"] == fold][target]

        dtrain = lgb.Dataset(X_train_cv, label=y_train_cv)
        dval = lgb.Dataset(X_eval_cv, label=y_eval_cv)

        params = {
                  'boosting_type': 'gbdt',
                  'random_state': 42,
                  'objective': 'regression',
                  'metric': 'rmse',
                  'learning_rate': 0.05,
                  }

        evaluation_results = {}
        model = lgb.train(params,
                          num_boost_round=10000,
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


# ## CV Score

# In[ ]:


# cv
rmses = []

for target in targets:
    models = model_dict[target]

    preds = []
    trues = []
    
    for fold, model in enumerate(models):
        # ilocで取り出す行を指定
        X_eval_cv = train[train["fold"] == fold].drop(columns=drop_columns)
        y_eval_cv = train[train["fold"] == fold][target]

        pred = model.predict(X_eval_cv)

        trues.extend(y_eval_cv)
        preds.extend(pred)
        
    rmse = np.sqrt(mean_squared_error(trues, preds))
    print(f"{target}_rmse : {rmse}")
    rmses = rmses + [rmse]

print(f"mcrmse : {sum(rmses) / len(rmses)}")


# # Predict 

# In[ ]:


drop_columns = [
                #"fold", 
                "student_id", "prompt_id", "text", 
                "prompt_question", "prompt_title", 
                "prompt_text"
               ]


# In[ ]:


pred_dict = {}
for target in targets:
    models = model_dict[target]
    preds = []

    for fold, model in enumerate(models):
        # ilocで取り出す行を指定
        X_eval_cv = test.drop(columns=drop_columns)

        pred = model.predict(X_eval_cv)
        preds.append(pred)
    
    pred_dict[target] = preds


# In[ ]:


for target in targets:
    preds = pred_dict[target]
    for i, pred in enumerate(preds):
        test[f"{target}_pred_{i}"] = pred

    test[target] = test[[f"{target}_pred_{fold}" for fold in range(CFG.n_splits)]].mean(axis=1)


# In[ ]:


test


# ## Create Submission file

# In[ ]:


sample_submission


# In[ ]:


test[["student_id", "content", "wording"]].to_csv("submission.csv", index=False)


# ## Summary
# 
# CV result is like this.
# 
# | | content rmse |wording rmse | mcrmse | LB| |
# | -- | -- | -- | -- | -- | -- |
# |baseline| 0.494 | 0.630 | 0.562 | 0.509 | [link](https://www.kaggle.com/code/tsunotsuno/debertav3-baseline-content-and-wording-models)|
# | use title and question field | 0.476| 0.619 | 0.548 | 0.508 | [link](https://www.kaggle.com/code/tsunotsuno/debertav3-w-prompt-title-question-fields) |
# | Debertav3 + LGBM | 0.451 | 0.591 | 0.521 | 0.461 | [link](https://www.kaggle.com/code/tsunotsuno/debertav3-lgbm-with-feature-engineering) |
# | Debertav3 + LGBM with spell autocorrect | 0.448 | 0.581 | 0.514 | 0.459 | [link](https://www.kaggle.com/code/tsunotsuno/updated-debertav3-lgbm-with-spell-autocorrect) |
# | LGBM + Debertav3 another impl | 0.453 | 0.621 | 0.537 | 0.481 | [link](https://www.kaggle.com/code/tsunotsuno/lgbm-debertav3-another-impl) |
# | Debertav3+LGBM (no autocorrect) | 0.454 | 0.611 | 0.532 | 0.451 | this notebook |
# 
# 
# 多少LB scoreは改善した。
# 
# autocorrectを使っていなくても0.451までは出せることを確認できた。
# ちなみに、学習と推論を分離して同様のパラメータチューニングをした結果、LBのスコアは悪化した。(LB 0.487)
# contentとwordingの分離の有無に関わらず性能に大差ないのであれば、同時に学習したほうがなにかと便利そうではある。
# 
# 懸念点としては、現状ですでに学習+推論で8h以上かかっており、これ以上学習を増やしたり複雑なことは難しいと考えられる。
# そのため、今後は学習と推論の分離が必要になると思われる。
# 
# LB score improved a little.
# 
# I confirmed that we can get up to 0.451 even without using autocorrect.
# 
# Incidentally, similar parameter tuning with learning and inference separated resulted in worse LB scores. (LB 0.487)
# I think there is no significant difference in performance regardless of whether content and wording are separated or not, and it may be more convenient to learn together.
# 
# One concern is that the current training and inference process already takes more than 8 hours, and it would be difficult to increase the learning and complexity any further.
# Therefore, it will be necessary to separate learning and inference in the future.
# 
# At any rate, thank you for reading!
# 

# In[ ]:




