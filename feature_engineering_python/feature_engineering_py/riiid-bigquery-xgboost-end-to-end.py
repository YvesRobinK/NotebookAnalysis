#!/usr/bin/env python
# coding: utf-8

# In[1]:


# <hide-input>
def show_version_history():
    from IPython.display import HTML
    style_header = 'mui--align-bottom mui--bg-primary mui--text-light mui--text-center'
    style_cell = 'mui--align-top mui--text-center'

    # print(('\n').join(list(map(make_li, sorted(dtypes.keys())))))

    html_str = f"""
    <link href="//cdn.muicss.com/mui-0.10.3/css/mui.min.css" rel="stylesheet" type="text/css" />
    <div class="mui-container-fluid">
        <h2>Version History</h2>
        <div style="max-width:1016px" class="mui-row">
            <div class="mui-col-8">
                <table class="mui-table mui-table--bordered">
                    <tr>
                        <th width="12%" class="{style_header}">Version</th>
                        <th width="12%" class="{style_header}">Date</th>
                        <th width="12%" class="{style_header}">Local CV</th>
                        <th width="12%" class="{style_header}">Public<br>Leaderboard</th>
                        <th class="{style_header} mui--align-bottom mui--bg-primary
                            mui--text-dark mui--text-left">Notes</th>                    
                    </tr>
                    <tr>
                        <td class="{style_cell}">60</td>
                        <td class="{style_cell}">2020-11-15</td>
                        <td class="{style_cell}">0.756</td>
                        <td class="{style_cell}">0.762</td>
                        <td><ul>
                                <li>Completed submission pipeline with minimal feature set:
                                    <ul>
                                        <li><code>answered_correctly_content_id_cumsum</code></li>
                                        <li><code>answered_correctly_cumsum</code></li>
                                        <li><code>answered_correctly_cumsum_pct</code></li>
                                        <li><code>answered_incorrectly_content_id_cumsum</code></li>
                                        <li><code>answered_incorrectly_cumsum</code></li>
                                        <li><code>part</code></li>
                                        <li><code>part_correct_pct</code></li>
                                        <li><code>question_id_correct_pct</code></li>
                                        <li><code>tag__0</code></li>
                                        <li><code>tag__0_correct_pct</code></li>
                                        <li><code>task_container_id</code></li>
                                        <li><code>timestamp</code></li>
                                    </ul>
                                </li>
                                <li>Changed logic on roll sum to be over trailing
                                    rows preceding the current <code>task_container_id</code> instead
                                    of over trailing task containers
                                    (expensive)
                                </li>
                            </ul>
                        </td>
                    </tr>
                    <tr>
                        <td class="{style_cell}">53</td>
                        <td class="{style_cell}">2020-11-07</td>
                        <td class="{style_cell}">0.761</td>
                        <td class="{style_cell}">--</td>
                        <td><ul><li>Housekeeping:
                                    <ul>
                                        <li>Consolidated notebook and modules in single repo</li>
                                        <li>Streamlined Colab repo workflow using Drive</>
                                        <li>Included modules in notebook when pushed to Kaggle</li>
                                        <li>Eliminated CONFIG requirement when run in Kaggle</li>
                                    </ul>
                                </li>
                            </ul>
                        </td>
                    </tr>
                    <tr>
                        <td class="{style_cell}">40</td>
                        <td class="{style_cell}">2020-11-05</td>
                        <td class="{style_cell}">0.761</td>
                        <td class="{style_cell}">--</td>
                        <td>
                            <ul>
                                <li>Features added:
                                    <ul>
                                        <li><code>answered_correctly_content_id_cumsum</code></li>
                                        <li><code>answered_correctly_content_id_cumsum_pct</code></li>
                                        <li><code>answered_correctly_cumsum10</code></li>
                                        <li><code>answered_correctly_cumsum_pct</code></li>
                                        <li><code>answered_correctly_rollsum_pct</code></li>
                                        <li><code>answered_incorrectly_content_id_cumsum</code></li>
                                        <li><code>lectures_cumcount</code></li>
                                        <li><code>prior_question_elapsed_time_rollavg</code></li>
                                    </ul>
                                </li>
                                <li>Single model, single fold</li>
                                <li>No public leaderboard - efficient inference in progress</li>
                                <li>Refactored code to move queries and helper functions into
                                    separate modules</li>
                                <li>Completed set up to commit code to Github from Colab and</li>
                                <li>Completed set up to push kernels to Kaggle from Colab</li>
                            </ul>
                        </td>
                    </tr>
                    <tr>
                        <td class="{style_cell}">37</td>
                        <td class="{style_cell}">2020-11-04</td>
                        <td class="{style_cell}">0.751</td>
                        <td class="{style_cell}">0.748</td>
                        <td>
                            <ul>
                                <li>Features added:
                                    <ul>
                                        <li><code>answered_correctly_cumsum</code></li>
                                        <li><code>answered_correctly_rollsum</code></li>
                                        <li><code>answered_incorrectly_cumsum</code></li>
                                        <li><code>answered_incorrectly_rollsum</code></li>
                                        <li><code>part</code></li>
                                        <li><code>part_correct_pct</code></li>
                                        <li><code>question_id_correct_pct</code></li>
                                        <li><code>tag__0</code></li>
                                        <li><code>tag__0_correct_pct</code></li>
                                    </ul>
                                </li>
                                <li>Single model, single fold</li>
                                <li>Model for public leaderboard didn't include
                                    rolling features - still working out how to
                                    efficiently calculate for inference</li>
                            </ul>
                        </td>
                    </tr>
                </table>
            </div>
        </div>
        <p>
            <a href="https://colab.research.google.com/github/CalebEverett/riiid_2020/blob/master/riiid-2020.ipynb" target="_blank" rel="nofollow">
            <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
        </p>
    </div>
    """

    html = HTML(html_str)
    display(html)
show_version_history()


# ## Intro

# This kernel is an end to end pipeline that uses BigQuery to store data and perform feature engineering, and trains a model using XGBoost. I was resorting to breaking up tables and still waiting a long time to see the results of my analysis and to process my engineered features, so I decided to learn about BigQuery. This kernel is the current state of my setup, which is working very well. It is much faster than my previous local setup, even with having to download files. It also is making it easier to keep the structure of the data and and code clean, which in turn makes it easier to stay focused on thinking about and executing ideas without getting bogged down waiting for things to finish or wading through extraneous processing code.
# 
# I've attempted to put  this book together in such a way that somebody else can fork it, update a few environment variables, run it and then be in the game engineering features and improving the model. The only requirements are a GCP project and storage bucket. Other than that, it is turn key, starting with creating a BigQuery dataset and ending with a saved model and two feature tables that get uploaded to a Kaggle dataset where they are used in a separate kernel to make predictions and submit to the competition api.
# 
# A couple of cool features:
# * Uses the gcs version of the competition datset to create a dataset and upload to BigQuery in around a minute
# * Transformations get run on the entire train table at once and run in under 10 minutes
# * Feature engineering gets done on a sample of the train table, taking advantage of BigQuery' graphical query editing interface that includes tab completion, syntax checking and the ability to run queries and inspect results
# * Stores queries as methods on a dedicated class, where they can be easily reused
# * Dtypes for local dataframes, schema for BigQuery tables and all tranformations are maintained locally so that the transformed tables can be recreated from the original competition dataset files automatically at any time (see description of workflow below to continue with this practice)
# * Exports to gcs using temporary tables created by BigQuery avoiding unnecessary storage and wasted time rerunning and exporting duplicate queries
# * Separate [submission kernel](https://www.kaggle.com/calebeverett/riiid-submit) uses sqlite3 to achieve sub two hour submission times while maintaining state for questions, users and user-content (80+ million rows)
# 
# I've engineered a few features as a starting point to demonstrate how additional features can be efficiently developed and processed, including:
# * Cumulative and rolling sums of questions answered correctly and incorrectly by user
# * Percent of questions answered correctly by question id, part and the first question tag
# 
# The model is also just a starting point, with a first pass at a train/validation split and no hyperparameter tuning. I have included some basic diagnostics on both the train/validtion split and model performance as a starting place for further development.
# 
# I have the table creation and transformation functions set to not run, but you can set them to run, by changing the flags to `True` for:
# * Loading tables - one flag for the questions table and another for the train and lectures tables
# * Updating the schemas in BigQuery
# * Performing the transformations

# ## Resources
# * [BigQuery Console](https://console.cloud.google.com/bigquery)
# * [Python Client for Google BigQuery](https://googleapis.dev/python/bigquery/latest/index.html)
# * [Analytic function concepts in Standard SQL](https://cloud.google.com/bigquery/docs/reference/standard-sql/analytic-function-concepts)
# * [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/index.html)
# * [Storge Client](https://googleapis.dev/python/storage/latest/client.html)
# * [pandas documentation](https://pandas.pydata.org/docs/)
# * [Plotly Python Open Source Graphing Library](https://plotly.com/python/)
# * [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)

# ## Imports

# In[2]:


# <hide-input>
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from datetime import datetime
import gc
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import ipywidgets as widgets
from google.cloud import storage, bigquery
from google.cloud.bigquery import SchemaField
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

BUCKET = 'riiid-caleb'
DATASET = 'data'
LOCATION = 'us-west4'
KAGGLE_SUBMIT_DATASET = 'riiid-submission'
PROJECT = 'riiid-caleb'
REPO = 'riiid_2020'
NOT_KAGGLE = os.getenv('KAGGLE_URL_BASE') is None

if NOT_KAGGLE:
    from google.colab import drive
    DRIVE = Path('/content/drive/My Drive')
    if not DRIVE.exists():
        drive.mount(str(DRIVE.parent))    
    sys.path.append(str(DRIVE))
    g_creds_path = 'credentials/riiid-caleb-faddd0c9d900.json'
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(DRIVE/g_creds_path)

bucket = storage.Client(project=PROJECT).get_bucket(BUCKET)
dataset = bigquery.Dataset(f'{PROJECT}.{DATASET}')
bq_client = bigquery.Client(project=PROJECT, location=LOCATION)

if NOT_KAGGLE:
    CONFIG = json.loads(bucket.get_blob('config.json').download_as_string())
    os.environ = {**os.environ, **CONFIG}
    from riiid_2020.utilities import check_packages, Git
    from riiid_2020.bqhelpers import BQHelper
    from riiid_2020.queries import Queries
    
    git = Git(REPO, CONFIG.get('GIT_USERNAME'), CONFIG.get('GIT_PASSWORD'),
              CONFIG.get('EMAIL'), DRIVE)

    packages = {
        'comet-ml': '3.2.5',
        'gcsfs': '0.7.1',
        'kaggle': '1.5.9',
        'plotly': '4.12.0',
        'xgboost': '1.2.0',
    }
    check_packages(packages)

    from comet_ml import Experiment
    from kaggle.api.kaggle_api_extended import KaggleApi
    kaggle_api = KaggleApi()
    kaggle_api.authenticate()

import plotly
import plotly.express as px
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
import xgboost as xgb
pd.options.plotting.backend = 'plotly'


# ## Modules
# Included in notebook for convenience when in a Kaggle kernel. Github repo [here](https://github.com/CalebEverett/riiid_2020).

# In[3]:


from pathlib import Path
import pytz
import sys

from google.cloud.bigquery import ExtractJobConfig, LoadJobConfig, \
    SchemaField, SourceFormat
import pandas as pd
from tqdm.notebook import tqdm


class BQHelper:
    def __init__(self, bucket, DATASET, bq_client):
        self.bucket = bucket
        self.BUCKET = self.bucket.name
        self.DATASET = DATASET
        self.bq_client = bq_client
   
    # LOAD FUNCTTIONS 
    # ================
    def load_job_cb(self, future):
        """Prints update upon completion to output of last run cell."""
        
        seconds = (future.ended - future.created).total_seconds()
        print(f'Loaded {future.output_rows:,d} rows to table {future.job_id.split("_")[0]} in '
            f'{seconds:>4,.1f} sec, {int(future.output_rows / seconds):,d} per sec.')
        
    def load_csv_uri(self, table_id, schemas_orig):
        full_table_id = f'{self.DATASET}.{table_id}'

        job_config = LoadJobConfig(
            schema=schemas_orig[table_id],
            source_format=SourceFormat.CSV,
            skip_leading_rows=1
            )

        uri = f'gs://{self.BUCKET}/{table_id}.csv'
        load_job = self.bq_client.load_table_from_uri(uri, full_table_id,
                                                job_config=job_config,
                                                job_id_prefix=f'{table_id}_')
        print(f'job {load_job.job_id} started')
        load_job.add_done_callback(self.load_job_cb)
        
        return load_job
        
    def load_json_file(self, table_id, schemas_orig):
        full_table_id = f'{self.DATASET}.{table_id}'

        job_config = LoadJobConfig(
            schema=schemas_orig[table_id],
            source_format=SourceFormat.NEWLINE_DELIMITED_JSON)

        file_path = f'{table_id}.json'
        with open(file_path, "rb") as source_file:
            load_job = self.bq_client.load_table_from_file(source_file,
                                                    full_table_id,
                                                    job_config=job_config,
                                                    job_id_prefix=f'{table_id}_')
        print(f'job {load_job.job_id} started')
        load_job.add_done_callback(self.load_job_cb)
        
        return load_job

    def get_table(self, table_id):
        return self.bq_client.get_table(f'{self.DATASET}.{table_id}')

    def del_table(self, table_id):
        return self.bq_client.delete_table(f'{self.DATASET}.{table_id}',
                                    not_found_ok=True)

    def get_df_jobs(self, max_results=10):
        jobs = self.bq_client.list_jobs(max_results=max_results, all_users=True)
        jobs_list = []

        if jobs.num_results:
            for job in jobs:
                ended = job.ended if job.ended else datetime.now(pytz.UTC)
                exception = job.exception() if job.ended else None
                jobs_list.append({'job_id': job.job_id, 'job_type': job.job_type,
                            'started': job.started, 'ended': ended,
                            'running': job.running(),
                            'exception': exception,
                            })
            df_jobs = pd.DataFrame(jobs_list)
            df_jobs['seconds'] = (df_jobs.ended - df_jobs.started).dt.seconds
            df_jobs.started = df_jobs.started.astype(str).str[:16]
            del df_jobs['ended']
            return df_jobs
        else:
            return None

    def get_df_table_list(self):
        tables = []
        for t in self.bq_client.list_tables(self.DATASET):
            table = self.bq_client.get_table(t)
            tables.append({'table_id': table.table_id, 'cols': len(table.schema),
                        'rows': table.num_rows, 'kb': int(table.num_bytes/1e3)})
        df_tables = pd.DataFrame(tables)
        
        return df_tables

    # QUERY FUNCTIONS
    # ================
    def done_cb(self, future):
        seconds = (future.ended - future.started).total_seconds()
        print(f'Job {future.job_id} finished in {seconds} seconds.')

    def run_query(self, query, job_id_prefix=None, wait=False):
        query_job = self.bq_client.query(query, job_id_prefix=job_id_prefix)
        print(f'Job {query_job.job_id} started.')
        query_job.add_done_callback(self.done_cb)
        if wait:
            query_job.result()
        
        return query_job

    def get_df_query(self, query, dtypes=None):
        query_job = self.run_query(*query)

        df_query = query_job.to_dataframe(dtypes=dtypes, 
                                progress_bar_type='tqdm_notebook')
        return df_query

    def get_df_table(self, table_id, max_results=10000, dtypes=None):
        table = self.get_table(table_id)
        df_table = (self.bq_client.list_rows(table, max_results=max_results)
                    .to_dataframe(dtypes=dtypes,
                                progress_bar_type='tqdm_notebook'))
        return df_table

    # EXPORT FUNCTIONS
    # ================
    def export_query_gcs(self, query, file_format='csv', wait=True):
        """ Uses BigQuery temporary table reference as gcs prefix.
        Runs query and exports to gcs if it doesn't already exist in gcs.
        Exported in multiple files if over 1GB. Returns gcs prefix.
        """
        qj = self.run_query(*query, wait=wait)
        
        prefix = ('/').join(qj.destination.path.split('/')[-2:])
        blobs_list = list(self.bucket.list_blobs(prefix=prefix))
        
        if not blobs_list:
            
            job_prefix_id = sys._getframe().f_code.co_name + '_'
            
            formats={'csv': 'CSV',
                    'json': 'NEWLINE_DELIMITED_JSON'}
            
            job_config = ExtractJobConfig(destination_format=formats[file_format])
            
            ex_job = self.bq_client.extract_table(
                source=qj.destination,
                destination_uris=f'gs://{self.BUCKET}/{prefix}/*.{file_format}',
                job_id_prefix=job_prefix_id,
                job_config=job_config)
        
            ex_job.add_done_callback(self.done_cb)
            
            print(f'Job {ex_job.job_id} started.')
        
            if wait:
                ex_job.result()
                blobs_list = list(self.bucket.list_blobs(prefix=prefix))
                n_files = len(blobs_list) 
                print(f'{n_files} file{"s" if n_files > 1 else ""} '
                    f'exported to gcs with prefix {prefix}.')
        
        else:
            n_files = len(blobs_list) 
            print(f'{n_files} file{"s" if n_files > 1 else ""} '
                  f'already exist{"s" if n_files == 1 else ""} in gcs with '
                  f'prefix {prefix}.')
        
        return prefix

    def get_table_gcs(self, prefix):
        """Downloads all files at prefix if they don't exist locally.
        Returns list of file paths.
        """
        
        file_paths = list(Path().glob(prefix))
        if not file_paths:
            blobs_list = list(self.bucket.list_blobs(prefix=prefix))
            n_files = len(blobs_list)
            print(f'Downloading {n_files} file{"s" if n_files > 1 else ""} '
                  f'from gcs for table {prefix}...')

            for b in tqdm(blobs_list, desc='Files Downloaded: '):
                print('Downloading', b.name, b.size)
                Path(b.name).parent.mkdir(parents=True, exist_ok=True)
                b.download_to_filename(b.name)
        
        else:
            n_files = len(list(file_paths[0].iterdir()))
            print(f'{n_files} file{"s" if n_files > 1 else ""} already '
                  f'exist{"s" if n_files == 1 else ""} locally for '
                  f'table{prefix}.')
            
        return list(list(Path().glob(prefix))[0].iterdir())

    def get_df_files(self, file_paths, dtypes):
        """ Creates data frame from list of local file paths.
        Returns dataframe.
        """
        
        prefix = str(file_paths[0].parent)
        suffix = file_paths[0].suffix
        
        n_files = len(file_paths)
        print(f'Creating dataframe from {n_files} file{"s" if n_files > 1 else ""} for table {prefix}...')
        
        dfs = []
        if suffix == '.csv':
            for f in tqdm(file_paths, desc='Files Read: '):
                dfs.append(pd.read_csv(f, dtype=dtypes))
        else:
            for f in tqdm(file_paths, desc='Files Read: '):
                dfs.append(pd.read_json(f, dtype=dtypes, lines=True))
        
        df_train = pd.concat(dfs)
        
        print(f'Dataframe finished for train table at {prefix} with'
            f' {len(df_train.columns):,d} columns and '
            f'{len(df_train):,d} rows.')
        
        return df_train

    def get_df_query_gcs(self, query, dtypes, file_format='csv', wait=True):
        prefix = self.export_query_gcs(query, file_format, wait)
        file_paths = self.get_table_gcs(prefix)
        df = self.get_df_files(file_paths, dtypes)
        
        return df# <include-bqhelpers.py><hide-input>


# In[4]:


import sys

class Queries:
    def __init__(self, DATASET):
        self.DATASET = DATASET
    
    def select_rows(self, table_id='train', limit=100):
        return f"""
            SELECT *
            FROM {self.DATASET}.{table_id}
            LIMIT {limit}
        """, sys._getframe().f_code.co_name + '_'

    def update_missing_values(self, table_id='train', column_id=None, value=None):
        return f"""
            UPDATE {self.DATASET}.{table_id}
            SET {column_id} = {value}
            WHERE {column_id} is NULL;
        """, sys._getframe().f_code.co_name + '_'

    def update_task_container_id(self, table_id='train', column_id_orig='task_container_id_orig'):
        return f"""
            UPDATE {self.DATASET}.{table_id}
            SET {column_id_orig} = task_container_id
            WHERE true;

            UPDATE {self.DATASET}.{table_id} t
            SET task_container_id = target.calc
            FROM (
              SELECT row_id, DENSE_RANK()
                OVER (
                  PARTITION BY user_id
                  ORDER BY timestamp
                ) - 1 calc
              FROM {self.DATASET}.{table_id}
            ) target
            WHERE target.row_id = t.row_id
        """, sys._getframe().f_code.co_name + '_'

    def create_train_sample(self, suffix='sample', user_id_max=50000):
        return f"""
            CREATE OR REPLACE TABLE {self.DATASET}.train_{suffix} AS
            SELECT *
            FROM {self.DATASET}.train
            WHERE user_id <= {user_id_max}
            ORDER BY user_id, task_container_id, row_id
        """, sys._getframe().f_code.co_name + '_'

    def select_train(self, columns=['*'], user_id_max=50000,
                     excl_lectures=False, table_id='train'):
        
        where_condition = f'user_id <= {user_id_max}' if user_id_max else 'true'
        where_condition = (where_condition + ' AND content_type_id = 0'
                           if excl_lectures else where_condition)
        
        return f"""
            SELECT {(', ').join(columns)}
            FROM {self.DATASET}.{table_id} t
            LEFT JOIN {self.DATASET}.questions q
            ON t.content_id = q.question_id
            WHERE {where_condition}
            ORDER BY user_id, task_container_id, row_id
        """, sys._getframe().f_code.co_name + '_'
    
    def update_answered_incorrectly(self, table_id='train'):
        """Sets annswered_incorrectly to inverse of answered_correctly for questions.
        Sets answered_correctly to 0 for lectures so window totals for correct and
        incorrect are caculated correctly, including lectures.
        """
    
        return f"""
            UPDATE {self.DATASET}.{table_id}
            SET answered_incorrectly = 0
            WHERE true;

            UPDATE {self.DATASET}.{table_id}
            SET answered_incorrectly = 1 - answered_correctly
            WHERE content_type_id = 0;

            UPDATE {self.DATASET}.{table_id}
            SET answered_correctly = 0
            WHERE content_type_id = 1;
        """, sys._getframe().f_code.co_name + '_'


    def update_questions_tag__0(self):
        return f"""
            UPDATE data.questions
            SET tag__0 = tags[OFFSET(0)]
            WHERE true;
        """, sys._getframe().f_code.co_name + '_'    
    
    def update_train_window_containers(self, table_id='train'):
        return f"""            
        UPDATE {self.DATASET}.{table_id} t
        SET answered_correctly_cumsum = IFNULL(calc.answered_correctly_cumsum, 0),
            answered_incorrectly_cumsum = IFNULL(calc.answered_incorrectly_cumsum, 0),
            lectures_cumcount = IFNULL(calc.lectures_cumcount, 0),
            prior_question_elapsed_time_rollavg = IFNULL(calc.prior_question_elapsed_time_rollavg, 0),
            answered_correctly_content_id_cumsum = IFNULL(calc.answered_correctly_content_id_cumsum, 0),
            answered_incorrectly_content_id_cumsum = IFNULL(calc.answered_incorrectly_content_id_cumsum, 0)
        FROM (
        SELECT row_id,
            SUM(answered_correctly) OVER (b) answered_correctly_cumsum,
            SUM(answered_incorrectly) OVER (b) answered_incorrectly_cumsum,
            SUM(content_type_id) OVER (b) lectures_cumcount,
            AVG(prior_question_elapsed_time) OVER (c) prior_question_elapsed_time_rollavg,
            SUM(answered_correctly) OVER (e) answered_correctly_content_id_cumsum,
            SUM(answered_incorrectly) OVER (e) answered_incorrectly_content_id_cumsum
        FROM {self.DATASET}.{table_id}
        WINDOW
            a AS (PARTITION BY user_id ORDER BY task_container_id),
            b AS (a RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
            c AS (a RANGE BETWEEN 3 PRECEDING AND 0 PRECEDING),
            d AS (PARTITION BY user_id, content_id ORDER BY task_container_id),
            e AS (d RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
        ORDER BY user_id, task_container_id, row_id
        ) calc
        WHERE calc.row_id = t.row_id
        """, sys._getframe().f_code.co_name + '_'

    def update_train_window_rows(self, table_id='train', window=10):
        """Calculates aggregate over window number of rows with task_container_id
        less than task_container_id of current row.
        """

        return f"""            
        UPDATE {self.DATASET}.{table_id} u
        SET answered_correctly_rollsum = IFNULL(calc.answered_correctly_rollsum, 0),
            answered_incorrectly_rollsum = IFNULL(calc.answered_incorrectly_rollsum, 0)
        FROM (
        SELECT t.row_id,
            COUNT(j2.row_id) row_id_rollcount,
            SUM(j2.answered_correctly) answered_correctly_rollsum,
            SUM(j2.answered_incorrectly) answered_incorrectly_rollsum,
        FROM {self.DATASET}.{table_id} t
        JOIN (
            SELECT user_id, task_container_id, MIN(row_id) min_row
            FROM {self.DATASET}.{table_id}
            GROUP BY user_id, task_container_id
        ) j ON (j.user_id = t.user_id AND j.task_container_id = t.task_container_id)
        LEFT JOIN {self.DATASET}.{table_id} j2 ON (
            j2.user_id = t.user_id
            AND j2.task_container_id < t.task_container_id
            AND j2.row_id >= (j.min_row - {window + 1})
        )
        GROUP BY t.user_id, t.task_container_id, t.row_id
        ) calc
        WHERE
        calc.row_id = u.row_id
        """, sys._getframe().f_code.co_name + '_'


    def update_answered_correctly_cumsum_upto(self, table_id='train'):        
        return f"""            
        UPDATE {self.DATASET}.{table_id} t
        SET answered_correctly_cumsum_upto = IF(row_number < 11, r.answered_correctly_cumsum, m.ac_max)
        FROM (
        SELECT user_id, row_id, answered_correctly_cumsum,
            ROW_NUMBER() OVER(W) row_number,
        FROM {self.DATASET}.{table_id}
        WHERE content_type_id = 0
        WINDOW
            w AS (PARTITION BY user_id ORDER BY row_id)
        ) r
        JOIN (
        SELECT user_id, MAX(answered_correctly_cumsum) ac_max
        FROM (
            SELECT user_id, row_id, answered_correctly_cumsum,
            ROW_NUMBER() OVER(W) row_number,
            FROM {self.DATASET}.{table_id}
            WINDOW
                w AS (PARTITION BY user_id ORDER BY row_id)
        )
        WHERE row_number < 11
        GROUP BY user_id
        ) m
        ON (m.user_id = r.user_id)
        WHERE r.row_id = t.row_id
        """, sys._getframe().f_code.co_name + '_'

    def update_correct_cumsum_pct(self, column_id_correct=None,
                                  column_id_incorrect=None,
                                  update_column_id=None, table_id='train'):
        return f"""
            CREATE TEMP FUNCTION calcCorrectPct(c INT64, ic INT64) AS (
              CAST(SAFE_DIVIDE(c, (c + ic)) * 100 AS INT64)
            );

            UPDATE {self.DATASET}.{table_id}
            SET {update_column_id} =
                calcCorrectPct({column_id_correct}, {column_id_incorrect})
            WHERE true;
            
            UPDATE {self.DATASET}.{table_id}
            SET {update_column_id} = 0
            WHERE {update_column_id} IS NULL;
        """, sys._getframe().f_code.co_name + '_'

    def update_question_correct_pct(self, column_id):
        return f"""  
            CREATE TEMP FUNCTION calcCorrectPct(c INT64, ic INT64) AS (
              CAST(SAFE_DIVIDE(c, (c + ic)) * 100 AS INT64)
            );

            UPDATE {self.DATASET}.questions q
            SET q.{column_id}_correct_pct = calcCorrectPct(c.c, c.ic)
            FROM (
                SELECT cq.{column_id}, SUM(answered_correctly) c, SUM(answered_incorrectly) ic
                FROM {self.DATASET}.train t
                JOIN {self.DATASET}.questions cq
                ON t.content_id = cq.question_id
                WHERE t.content_type_id = 0
                GROUP BY cq.{column_id}
            ) c
            WHERE q.{column_id} = c.{column_id}
        """, sys._getframe().f_code.co_name + '_'

    def select_user_id_rows(self, table_id='train', rows=30000):
        return f"""            
            SELECT user_id
            FROM {self.DATASET}.{table_id}
            WHERE row_id = {rows}
        """, sys._getframe().f_code.co_name + '_'
    
    def select_user_final_state(self, table_id='train'):
        return f"""            
        CREATE TEMP FUNCTION calcCorrectPct(c INT64, ic INT64) AS (
        IFNULL(CAST(SAFE_DIVIDE(c, (c + ic)) * 100 AS INT64), 0)
        );
        
        SELECT *, calcCorrectPct(answered_correctly_cumsum, answered_incorrectly_cumsum) answered_correctly_cumsum_pct,
        calcCorrectPct(answered_correctly_rollsum, answered_incorrectly_rollsum) answered_correctly_rollsum_pct
        FROM (
        SELECT row_id, user_id, answered_correctly_cumsum_upto, content_type_id,
            SUM(answered_correctly) OVER (b) answered_correctly_cumsum,
            SUM(answered_incorrectly) OVER (b) answered_incorrectly_cumsum,
            SUM(answered_correctly) OVER (d) answered_correctly_rollsum,
            SUM(answered_incorrectly) OVER (d) answered_incorrectly_rollsum,
            SUM(content_type_id) OVER (b) lectures_cumcount,
            AVG(prior_question_elapsed_time) OVER (c) prior_question_elapsed_time_rollavg,
            ROW_NUMBER() OVER(y) row_no_desc,
            SUM(answered_correctly + answered_incorrectly) OVER (d) answer_row_id_rollcount,
            SUM(answered_correctly + answered_incorrectly) OVER (c) time_row_id_rollcount,
            SUM(answered_correctly + answered_incorrectly) OVER (a) question_row_id_rollcount,
        FROM {self.DATASET}.{table_id}
        WINDOW
            x AS (PARTITION BY user_id),
            y AS (x ORDER BY task_container_id DESC, row_id DESC),
            a AS (x ORDER BY task_container_id),
            b AS (a ROWS BETWEEN UNBOUNDED PRECEDING AND 0 PRECEDING),
            c AS (a RANGE BETWEEN 3 PRECEDING AND 0 PRECEDING),
            d AS (a ROWS BETWEEN 9 PRECEDING AND 0 PRECEDING)
        )
        WHERE row_no_desc = 1 AND content_type_id = 0
        ORDER BY user_id
        """, sys._getframe().f_code.co_name + '_'

    def select_user_content_final_state(self, table_id='train'):
        return f"""            
        SELECT user_id, content_id, SUM(answered_correctly) answered_correctly,
            SUM(answered_incorrectly) answered_incorrectly,
        FROM {self.DATASET}.{table_id}
        WHERE content_type_id = 0
        GROUP BY user_id, content_id
        ORDER BY user_id, content_id
        """, sys._getframe().f_code.co_name + '_'
    # <include-queries.py><hide-input>


# In[5]:


import os
import subprocess

def porc(c):
    print_output(run_command(c))

def print_output(output):
    """Prints output from string."""
    for l in output.split('\n'):
        print(l)

def run_command(command):
    """Runs command line command as a subprocess returning output as string."""
    STDOUT = subprocess.PIPE
    process = subprocess.run(command, shell=True, check=False,
                             stdout=STDOUT, stderr=STDOUT, universal_newlines=True)
    
    output = process.stdout if process.stdout else process.stderr
    
    return output

def get_v_tuple(v):
    return tuple([int(s) for s in v.split('.')])

def check_package(p, v):
    output = run_command(f'pip freeze | grep {p}==')
    if output == '':
        porc(f'pip install -q {p}')
    elif get_v_tuple(output[output.find('==')+2:]) < get_v_tuple(v):
        porc(f'pip install -q -U {p}')
    else:
        print_output(output)

def check_packages(packages):
    for p, v in packages.items():
        check_package(p, v)

class Git:
    def __init__(self, repo, username, password, email, base_path):
        self.repo = repo
        self.username = username
        self.password = password
        self.email = email
        self.repo_path = base_path/repo
        self.cred_repo = (
            f'https://{self.username}:{self.password}'
            f'@github.com/{self.username}/{self.repo}.git'
        )
        self.config()
    
    def config(self):
        commands = []
        commands.append(f'git config --global user.email {self.email}')
        commands.append(f'git config --global user.name {self.username}')
        for cmd in commands:
            porc(cmd)
        print('Git global user.name and user.email set.')
    
    def clone(self, latest=False):
        cwd = os.getcwd()
        os.chdir(self.base_path)

        if latest:
            cred_repo = f'--depth 1 {cred_repo}'

        commands = []
        commands.append(f'git clone {cred_repo}')
        for cmd in commands:
            porc(cmd)

        os.chdir(cwd)

    def commit(self, message='made some changes'):
        cwd = os.getcwd()
        os.chdir(self.repo_path)
        porc('git add -A')
        porc(f'git commit -m "{message}"')
        os.chdir(cwd)

    def command(self, command):
        cwd = os.getcwd()
        os.chdir(self.repo_path)
        porc(f'git {command}')
        os.chdir(cwd)

    def status(self):
        self.command('status')

    def push(self, branch='master'):
        self.command(f'push origin {branch}')

    def set_remote(self):
        self.command(f'remote set-url origin {self.cred_repo}')# <include-utilities.py><hide-input>


# In[6]:


{"COMET_API_KEY": "", "COMET_PROJECT_NAME": "", "COMET_WORKSPACE": "", "EMAIL": "", "GIT_PASSWORD": "", "GIT_USERNAME": "", "KAGGLE_USERNAME": "", "KAGGLE_KEY": ""}# <include-config.json><hide-input><hide-output>


# In[7]:


Q = Queries(DATASET)
bqh = BQHelper(bucket, DATASET, bq_client)


# ## Create BigQuery Dataset

# In[8]:


if False:
    delete_contents=False
    bq_client.delete_dataset(DATASET, delete_contents=delete_contents)
    print(f'Dataset {dataset.dataset_id} deleted from project {dataset.project}.')


# In[9]:


try:
    dataset = bq_client.get_dataset(dataset.dataset_id)
    print(f'Dataset {dataset.dataset_id} already exists '
          f'in location {dataset.location} in project {dataset.project}.')
except:
    dataset = bq_client.create_dataset(dataset)
    print(f'Dataset {dataset.dataset_id} created '
          f'in location {dataset.location} in project {dataset.project}.')


# ## Load Tables

# ### Dataframe dtypes

# In[10]:


# <hide-input>
dtypes_orig = {
    'lectures': {
        'lecture_id': 'uint16',
        'tag': 'uint8',
        'part': 'uint8',
        'type_of': 'str',
    },
    'questions': {
        'question_id': 'uint16',
        'bundle_id': 'uint16',
        'correct_answer': 'uint8',
        'part': 'uint8',
        'tags': 'str',
        
    },
    'train': {
        'row_id': 'int64',
        'timestamp': 'int64',
        'user_id': 'int32',
        'content_id': 'int16',
        'content_type_id': 'int8',
        'task_container_id': 'int16',
        'user_answer': 'int8',
        'answered_correctly': 'int8',
        'prior_question_elapsed_time': 'float32', 
        'prior_question_had_explanation': 'bool'
    }
    
}

dtypes_new = {
    'lectures': {},
    'questions': {
        'tag__0': 'uint8',
        'part_correct_pct': 'uint8',
        'tag__0_correct_pct': 'uint8',
        'question_id_correct_pct': 'uint8'
    },
    'train': {
        'task_container_id_orig': 'int16',
        'answered_correctly_cumsum': 'int16',
        'answered_correctly_rollsum': 'int8',
        'answered_incorrectly': 'int8',
        'answered_incorrectly_cumsum': 'int16',
        'answered_incorrectly_rollsum': 'int8',
        'answered_correctly_cumsum_pct': 'int8',
        'answered_correctly_rollsum_pct': 'int8',
        'answered_correctly_content_id_cumsum': 'int16',
        'answered_incorrectly_content_id_cumsum': 'int16',
        'answered_correctly_content_id_cumsum_pct': 'int16',
        'answered_correctly_cumsum_upto': 'int8',
        'prior_question_elapsed_time_rollavg': 'float32',
        'lectures_cumcount': 'int16',
    }
}

one_hot_tags = False
if one_hot_tags:
    for tag in range(189):
        for table_id in ['questions']:
            dtypes_new[table_id][f'tag_{tag:03d}'] = 'uint8'

dtypes = {}
for table_id in dtypes_orig:
    dtypes[table_id] = {**dtypes_orig[table_id], **dtypes_new[table_id]}

dtypes = {
    **dtypes['lectures'],
    **dtypes['questions'],
    **dtypes['train']
}


# ### BigQuery Table Schemas

# In[11]:


# <hide-input>
type_map = {
    'int64': 'INTEGER',
    'int32': 'INTEGER',
    'int16': 'INTEGER',
    'int8': 'INTEGER',
    'uint8': 'INTEGER',
    'uint16': 'INTEGER',
    'str': 'STRING',
    'bool': 'BOOL',
    'float32': 'FLOAT'
}

schemas_orig = {table: [SchemaField(f, type_map[t]) for f, t in
                   fields.items()] for table, fields in dtypes_orig.items()}
schemas_orig['questions'][-1] = SchemaField('tags', 'INTEGER', 'REPEATED')

schemas = {}
for table_id, fields in dtypes_new.items():
    new_fields = [SchemaField(f, type_map[t]) for
                  f, t in fields.items()]
    schemas[table_id] = schemas_orig[table_id] + new_fields


# ### Load Tables

# In[12]:


# <hide-input>
# Load questions from local json file - can't load tags as array from csv.

if False:
    bqh.del_table('questions')
    
    df_questions = pd.read_csv(f'gs://{BUCKET}/questions.csv')
    df_questions.tags = df_questions.tags.fillna('188')
    df_questions.tags = df_questions.tags.str.split()
    
    if one_hot_tags:
        mlb = MultiLabelBinarizer()
        one_hots = (mlb.fit_transform(df_questions.tags
                    .apply(lambda l: [f'tag_{int(t):03d}' for t in l])))
        df_one_hots = pd.DataFrame(one_hots, columns = mlb.classes_)
        df_questions = pd.concat([df_questions, df_one_hots], axis=1)
    
    df_questions.to_json('questions.json', orient="records", lines=True)
    lj = bqh.load_json_file('questions', schemas).result()


# In[13]:


# <hide-input>
if False:
    for table_id in ['lectures', 'train']:
        bqh.del_table(table_id)
        lj = bqh.load_csv_uri(table_id, schemas_orig).result()


# In[14]:


# <hide-input>
df_jobs = bqh.get_df_jobs()
df_jobs


# In[15]:


# <hide-input>
df_table_list = bqh.get_df_table_list()
df_table_list


# ### Update Table Schemas

# In[16]:


# <hide-input>
if False:
    for table_id, schema in schemas.items():
        table = bqh.get_table(table_id)
        table.schema = schema
        table = bq_client.update_table(table, ['schema'])


# ## Engineer Features

# A good workflow here is:
# * Create a sample of the train table.
# * Use the BigQuery query editor user interface to get the SQL for a new feature worked out as a selection from the `train_sample` table. The user interface there has tab completion, syntax checking and displays results, which makes creating and debugging queries a snap.
#     * [BigQuery Console](https://console.cloud.google.com/bigquery?project=riiid-caleb) (Update project query string for your project.)
#     * [BigQuery Query syntax in Standard SQL](https://cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax) is your friend.
# * Optional: create a local dataframe, using the export functions below, to confirm that it is working the right way.
# * Add a column to the appropriate table by adding a value to `dtypes_new`
# * Update the schema for the table in BigQuery by running the Update Table Schemas cell above
# * Recreate the `train_sample` table by running the cell below.
# * Use the BigQuery query editor user interface add the logic to update the new column.
# * Optional: create a local dataframe, using the export functions below, to confirm that the update is working the right way.
# * Copy the SQL to a new method in the `Queries` class above
# * Add the query to the appropriate `run_transformations` function above
# * Run transformations on `train_sample` table
# * Inspect `train_sample` table in BigQuery to confirm everything is working correctly
# * Optional: load load local dataframe using `get_df_query` function for further inspection
# * Run transformations on `train` table
# * Inspect `train` table in BigQuery to confirm everything is working correctly
# * Optional: load local dataframe using `get_df_query` function for further inspection

# ### Perform Transformations

# #### Train Table
# * Add question columns
#     * Adding question part and the first associated tag. (There wasn't any official information regarding the order of the tags as recorded for each question, but they did not appear to be sorted so it seems possible the order in which they are recorded is significant.)
# * Update task_container_id to increase monotonically with timestamp
#     * There were some `task_conatiner_id`s that were out of order with respect to timestamp. They needed to be be ordered correctly so that cumulative and rolling sums partitioned by `task_container_id` would be include only interactions with earlier `timestamps`. Even though all interactions with the same `task_container_id` have the same `timestamp`, partioning by `timestamp` is much slower (because the range of values is so much wider?).
# * Calc answered_incorrectly
#     * `answered_correctly` for lectures was recorded as -1 and needed to be set to 0 to calculate cumulative and rolling sums correctly including lectures. As a consequence, `answered_incorrectly` could be calculated as the inverse of `answered_correctly`.
# * Calc cumsum for `answered_correctly` and `answered_incorrectly` by `user_id` and by `user_id` and `content_id` and rolling avg for `prior_question_elapsed_time` by user 
#     * This is done so that the totals are as of the preceding `task_container_id`
# * Calculate rolling sum for `answered_correctly` and `answered_incorrectly` by `user_id`
#     * Includes the 10 rows preceding the current `task_container_id`
#     * I couldn't figure out how to get this done with the standard window functionality since I wanted a set number of rows preceding the current task container (as opposed to just the current row), so it joins on `user_id` with a `task_container_id` less than the current one, which takes a while to complete.
# * Calculate answered correctly percentages for `answered_correctly_cumsum`, `answered_correctly_rollsum` and `answered_correctly_content_id_cumsum_pct`
# 
# #### Questions Table
# * Calculate percent answered correctly for `question_id`, `part` and `tag__0` 

# In[17]:


# <hide-input>
cumsum_pct_specs = [
    dict(column_id_correct='answered_correctly_cumsum',
         column_id_incorrect='answered_incorrectly_cumsum',
         update_column_id='answered_correctly_cumsum_pct'),
    
    dict(column_id_correct='answered_correctly_rollsum',
         column_id_incorrect='answered_incorrectly_rollsum',
         update_column_id='answered_correctly_rollsum_pct'),
    
    dict(column_id_correct='answered_correctly_content_id_cumsum',
         column_id_incorrect='answered_incorrectly_content_id_cumsum',
         update_column_id='answered_correctly_content_id_cumsum_pct'),                   
]

def run_update_correct_cumsum_pct(spec):
    query, job_id_prefix = Q.update_correct_cumsum_pct(**spec)
    job_id_prefix = f'{job_id_prefix}{spec["update_column_id"]}_'
    bqh.run_query(query=query, job_id_prefix=job_id_prefix, wait=True)


# In[18]:


# <hide-input>
def run_train_transforms(table_id=None):
    # Run serially to avoid update conflicts
    
    train_queries = [
        Q.update_task_container_id(table_id=table_id),
        Q.update_answered_incorrectly(table_id=table_id),
        Q.update_missing_values(table_id=table_id,
                                column_id='prior_question_had_explanation',
                                value='false'),
        Q.update_missing_values(table_id=table_id,
                                column_id='prior_question_elapsed_time',
                                value='0'),
        Q.update_train_window_containers(table_id=table_id),
        Q.update_train_window_rows(table_id=table_id, window=10),
        Q.update_answered_correctly_cumsum_upto(table_id=table_id)
    ]
    
    _ = [bqh.run_query(*q, wait=True) for q in train_queries]

    _ = [spec.update(table_id=table_id) for spec in cumsum_pct_specs]
    _ = list(map(run_update_correct_cumsum_pct, cumsum_pct_specs))


# In[19]:


# <hide-input>
def run_questions_transforms():
    """These have to be run after the transforms are run on the full
    train table.
    """
    
    questions_queries = [Q.update_questions_tag__0()]
    for column_id in ['question_id', 'part', 'tag__0']:
        questions_queries.append(Q.update_question_correct_pct(column_id))
    
    _ = [bqh.run_query(*q, wait=True).result() for q in questions_queries]


# In[20]:


# <hide-input>
if False:
    run_train_transforms('train')
    run_questions_transforms()


# ### Check Output

# In[21]:


query = Q.select_train(table_id='train', excl_lectures=True)
df_query = bqh.get_df_query(query, dtypes=dtypes)


# In[22]:


# <hide-input>
cols = [
        'row_id',
        'task_container_id_orig',
        'timestamp',
        'content_type_id',
        'user_id',
        'task_container_id',
        'part',
        'tag__0',
        'answered_correctly',
        'answered_incorrectly',
        'answered_correctly_cumsum',
        'answered_incorrectly_cumsum',
        'answered_correctly_content_id_cumsum',
        'answered_correctly_rollsum',
        'answered_incorrectly_rollsum',
        'answered_incorrectly_content_id_cumsum',
        'part_correct_pct',
        'tag__0_correct_pct',
        'question_id_correct_pct',
        'prior_question_elapsed_time',
        'prior_question_elapsed_time_rollavg',
        'prior_question_had_explanation',
        'lectures_cumcount',
        'answered_correctly_cumsum_upto'
]

df_user = df_query[cols].copy()
df_user.timestamp = df_user.timestamp / (1000*60*60)

df_user.loc[df_user.user_id == 44331].head(20)


# ### Visually Inspect Features

# The charts below can also be used to visually inspect whether the transformations have been performed correctly.

# In[23]:


# <hide-input>
groups = {
    'cum': {
        'columns': {
            'task_container_id': 0,
            'answered_correctly_cumsum': 2,
            'answered_incorrectly_cumsum': 1
        },
        'xaxis': 'elapsed_hours'
    },
    'roll': {
        'columns': {
            'answered_correctly_rollsum': 2,
            'answered_correctly': 7,
            'answered_incorrectly_rollsum': 1,
            'answered_incorrectly': 8,
            'part': 9
        },
        'xaxis': 'row_id'
    },  
    'correct_pct': {
        'columns': {
            'question_id_correct_pct': 0,
            'part_correct_pct': 5,
            'tag__0_correct_pct': 6
        },
        'xaxis': 'row_id'
    },  
    'prior_question_elapsed_time': {
        'columns': {
            'prior_question_elapsed_time': 0,
        },
        'xaxis': 'row_id'
    },  
    'prior_question_had_explanation': {
        'columns': {
            'prior_question_had_explanation': 0,
        },
        'xaxis': 'row_id'
    }
}

def plot_user_learning(user_id=None, group=None, suffix=None):
    theme = px.colors.qualitative.Plotly
    columns = list(group['columns'].keys())
    colors = [theme[c] for c in group['columns'].values()]

    df_query['elapsed_hours'] = df_query.timestamp / (1000*60*60)

    df = (df_query.loc[(df_user.user_id == user_id) &
                       (df_user.content_type_id == 0)])

    # labels = {'value': 'answer count'}

    fig = df.plot(x=group['xaxis'], y=columns, color_discrete_sequence=colors,
                  title=f'Learning Progress - user_id = {user_id} - {suffix}')
    fig.data

    return fig

user_id_random = np.random.choice(df_query.user_id.unique(), (1,))[0]
use_random = False
user_id =  user_id_random if use_random else 5382

for k, v in groups.items():
    fig = plot_user_learning(user_id, v, k)
    fig.show()


# ### Create Sample of Train Table for R&D

# In[24]:


# <hide-input>
if False:
    bqh.run_query(*Q.create_train_sample(), wait=True)
    q = Q.select_train(excl_lectures=True, table_id='train_sample')
    df_sample = bqh.get_df_query(q)


# ## Create Local Training Dataframe

# With feature engineering being performed in BigQuery, data has to be exported to train models locally. The [Python Client for Google BigQuery](https://googleapis.dev/python/bigquery/latest/index.html) [to_dataframe()](https://googleapis.dev/python/bigquery/latest/generated/google.cloud.bigquery.job.QueryJob.html?highlight=to_dataframe#google.cloud.bigquery.job.QueryJob.to_dataframe) makes it possible to create dataframes directly, but is prohibitively slow for large datasets. While it is not possible to export table directly to the local file system, it is possible to export to cloud storage and then download locally from there. This is reasonably efficient, taking a couple of minutes to run a query, export to cloud storage, download to the local file system and then read the files into a dataframe. The is another api, the [BigQuery Storage API](https://cloud.google.com/bigquery/docs/reference/storage), that a client can be created with that is really fast and works with the `to_dataframe` method, but unforunatley it isn't working with the current Kaggle kernel environment.
# 
# The functions below take advantage of the fact BigQuery stores queries in temporary tables so that preveiously requested queries can be retrieved without having to run them again. Similarly, the functions below name the exported files with the reference to the BigQuery temporary table, so that if a function is run to create a dataframe from a query for which the files already exist in cloud storage or locally, they won't be exported or downloaded again. 

# ### Create DataFrame

# In[25]:


# <hide-input>
features = {
    'answered_correctly':                       True,
    'answered_correctly_content_id_cumsum':     True,
    'answered_correctly_content_id_cumsum_pct': True,
    'answered_correctly_cumsum':                True,
    'answered_correctly_cumsum_upto':           True,
    'answered_correctly_cumsum_pct':            True,
    'answered_correctly_rollsum':               True,
    'answered_correctly_rollsum_pct':           True,
    'answered_incorrectly':                     True,
    'answered_incorrectly_content_id_cumsum':   True,
    'answered_incorrectly_cumsum':              True,
    'answered_incorrectly_rollsum':             True,
    'bundle_id':                                False,
    'content_id':                               True,
    'content_type_id':                          True,
    'correct_answer':                           False,
    'lecture_id':                               False,
    'lectures_cumcount':                        True,
    'part':                                     True,
    'part_correct_pct':                         True,
    'prior_question_elapsed_time':              True,
    'prior_question_elapsed_time_rollavg':      True,
    'prior_question_had_explanation':           True,
    'question_id':                              False,
    'question_id_correct_pct':                  True,
    'row_id':                                   True,
    'tag':                                      False,
    'tag__0':                                   True,
    'tag__0_correct_pct':                       True,
    'tags':                                     False,
    'task_container_id':                        True,
    'task_container_id_orig':                   False,
    'timestamp':                                True,
    'type_of':                                  False,
    'user_answer':                              False,
    'user_id':                                  True
}

tag_cols = [f'tag_{tag:03d}' for tag in range(189)] if one_hot_tags else []

columns_export = [f for f, v in features.items() if v]
if one_hot_tags:
    columns_export = columns_export +  tag_cols


# In[26]:


# <hide-input>
def get_features_widget(features_dict, columns_list):

    names = []
    widget_list = []
    for key, v in features_dict.items():
        widget_list.append(widgets.ToggleButton(value=v,
                                                description=key,
                                                layout={'width': '290px'},
                                                button_style='primary'))
        names.append(key)

    arg_dict = {names[i]: widget for i, widget in enumerate(widget_list)}

    layout = widgets.Layout(grid_template_columns="repeat(3, 300px)")
    ui = widgets.GridBox(widget_list, layout=layout)

    def select_data(**kwargs):
        columns_list.clear()

        for key in kwargs:
            features_dict[key] = False
            if kwargs[key]:
                columns_list.append(key)
                features_dict[key] = True

        print(f'{len(columns_list)} columns selected')

    output = widgets.interactive_output(select_data, arg_dict)
    return ui, output


# In[27]:


# <hide-input>
columns_export = []
display(*get_features_widget(features, columns_export))


# In[28]:


# <hide-input>
# get user_ids for rows in thousands. will be approximate, excludes lectures
# and selects all records for user_ids less than specified.

if False:    
    r = Q.run_query(*Q.select_user_id_rows(rows=int(2e6))).result()
    user_id_max = list(r)[0].user_id
    print(user_id_max)
    
user_ids = {
    10: 91216,
    100: 2078569,
    1000: 20949024,
    2000: 42207371,
    10000: 216747867,
    30000: 643006676
}


# In[29]:


# <hide-output>
if True:
    query = Q.select_train(columns=columns_export, user_id_max=user_ids[10000],
                           excl_lectures=True)
    df_train = bqh.get_df_query_gcs(query, dtypes=dtypes, file_format='json')


# ## Train Model

# ### Create Train and Validation Splits

# This is a first pass at a validation split to be able to have something to get the mechanics of evaluating the model up and running. It simply takes the last 20 `task_container_id`s for each user. The result is that all of the records in the validation set have `task_container_ids` greater than those in the training set for each user. There are also users in the validation set that are not present in the training set. However, a significant problem with this methodology is that the number of records per user in the validation set is much lower than it is in the training set.

# In[30]:


# <hide-input>
# get unique user_id-task_container_id combinations
df_user_task = df_train.groupby(['user_id',
                                 'task_container_id'])[['user_id',
                                                        'task_container_id',
                                                        'row_id']].head(1)

# get index of trailing number of unique user_id-task_container_id combinations
index_valid = (df_user_task.groupby('user_id').tail(20)
               .set_index(['user_id', 'task_container_id']).index)

# use index to get ids of all rows in the chosen set of user_id-task_container
# combinations
row_valid = (df_train.set_index(['user_id', 'task_container_id'])['row_id']
             .loc[index_valid].values)

df_train['valid'] = df_train.row_id.isin(row_valid)


# In[31]:


# <hide-input>
title = 'Train and Validation Splits - Record Counts'
df_train.valid.value_counts().plot(kind='bar', title=title)


# In[32]:


# <hide-input>
(df_train.groupby(['valid','user_id'])[['valid','user_id']].head(1)
 .reset_index().groupby('valid').count().user_id
 .plot(kind='bar', title='Count of Users by Split'))


# In[33]:


# <hide-input>
g_user_ct = (df_train[['valid', 'row_id', 'user_id']]
             .groupby(['valid', 'user_id']).count())

bins = [0,10,20,50,100,250,500,1000,2500,5000,20000]
g_user_ct['bin'] = pd.cut(g_user_ct.row_id, bins=bins, duplicates='drop')
g_counts = (g_user_ct.reset_index()
            .groupby(['valid', 'bin'])['row_id'].count().reset_index())

px.bar(x=g_counts.bin.apply(str), y=g_counts.row_id,
       facet_col=g_counts.valid.map({True: 'Validation', False: 'Train'}),
       title='Count of Users by Count of Interactions by Split',
       labels={'x': 'Count of Interactions',
               'y': 'Count of Users',
               'facet_col': 'Validation Split'})


# ### Select Columns for Training

# In[34]:


# <hide-input>
features_train = {
    'answered_correctly':                       False,
    'answered_correctly_content_id_cumsum':     True,
    'answered_correctly_content_id_cumsum_pct': False,
    'answered_correctly_cumsum':                True,
    'answered_correctly_cumsum_upto':           False,
    'answered_correctly_cumsum_pct':            True,
    'answered_correctly_rollsum':               False,
    'answered_correctly_rollsum_pct':           False,
    'answered_incorrectly':                     False,
    'answered_incorrectly_content_id_cumsum':   True,
    'answered_incorrectly_cumsum':              True,
    'answered_incorrectly_rollsum':             False,
    'bundle_id':                                False,
    'content_id':                               False,
    'content_type_id':                          False,
    'correct_answer':                           False,
    'lecture_id':                               False,
    'lectures_cumcount':                        False,
    'part':                                     True,
    'part_correct_pct':                         True,
    'prior_question_elapsed_time':              False,
    'prior_question_elapsed_time_rollavg':      False,
    'prior_question_had_explanation':           False,
    'question_id':                              False,
    'question_id_correct_pct':                  True,
    'row_id':                                   False,
    'tag':                                      False,
    'tag__0':                                   True,
    'tag__0_correct_pct':                       True,
    'tags':                                     False,
    'task_container_id':                        True,
    'task_container_id_orig':                   False,
    'timestamp':                                True,
    'type_of':                                  False,
    'user_answer':                              False,
    'user_id':                                  False
    }

columns_train = [f for f, v in features_train.items() if v] + tag_cols


# In[35]:


# <hide-input>
columns_train = []
display(*get_features_widget(features_train, columns_train))


# In[36]:


# <hide-input>
def show_features():
    df_features = pd.DataFrame([features, features_train]).T.reset_index()
    df_features.columns = ['feature', 'export', 'train']
    df_features

    def highlight_true(s):
        return ['background-color: lightskyblue' if v else '' for v in s]
    return df_features.style.apply(highlight_true, subset=['export', 'train'])
show_features()


# In[37]:


# <hide-input>
y_train_col = ['answered_correctly']

x_train_cols = columns_train

train_matrix = xgb.DMatrix(data=df_train.loc[~df_train.valid][x_train_cols],
                           label=df_train.loc[~df_train.valid][y_train_col])

valid_matrix = xgb.DMatrix(data=df_train.loc[df_train.valid][x_train_cols],
                           label=df_train.loc[df_train.valid][y_train_col])


# ### Train Model

# In[38]:


# <hide-output>
params = {
    'eta': 0.2,
    'max_depth': 6,
    'max_bin': 256,
    'tree_method': 'gpu_hist',
    'grow_policy': 'lossguide',
    'sampling_method': 'gradient_based',
    'objective': 'binary:logistic',
    'eval_metric': ['error', 'logloss', 'auc']
}

if NOT_KAGGLE:
    experiment = Experiment()

evals_result = {}
model = xgb.train(params=params, dtrain=train_matrix, num_boost_round=300,
                  evals=[(train_matrix, 'train'), (valid_matrix, 'valid')],
                  evals_result=evals_result, early_stopping_rounds=10)

if NOT_KAGGLE:
    experiment.end()


# ## Evaluate Model

# In[39]:


# <hide-input>
def get_evals_df(evals_result):
    evals_list = []
    for k,v in evals_result.items():
        for j,u in v.items():
            evals_list.extend([{'epoch': i,
                                'split': k,
                                'metric': j,
                                'result': r} for i,r in enumerate(u)])
    
    df_evals = (pd.DataFrame(evals_list).set_index(['split', 'metric', 'epoch'])
                .unstack('metric'))
    df_evals.columns = df_evals.columns.get_level_values(1)
    df_evals.columns.name = None
    
    return df_evals.reset_index()

df_evals = get_evals_df(evals_result)


# In[40]:


# <hide-input>
df_evals.plot(x='epoch', y=['auc', 'logloss'],
              facet_col='split', title='Learning Curves')


# In[41]:


# <hide-input>
imps = model.get_score(importance_type='gain').items()
df_imp = pd.DataFrame(imps, columns=['feature', 'importance'])
df_imp = df_imp.set_index('feature').sort_values('importance', ascending=False)
df_imp.plot(kind='bar', y='importance', title='Feature Importances - Gain')


# ## Prepare Prediction Data

# ### Download Final Users State

# In[42]:


# <hide-input><hide-output>
# not using this currently, creating dataframe from users-content
# table in submission notebook
if False:
    query = Q.select_user_final_state(table_id='train')
    prefix = bqh.export_query_gcs(query, wait=True)
    file_paths = bqh.get_table_gcs(prefix)
    df_users = (bqh.get_df_files(file_paths, dtypes=dtypes)
                .reset_index(drop=True).set_index('user_id'))


# ### Download Final User-Content State

# In[43]:


# <hide-input><hide-output>
if False:
    query = Q.select_user_content_final_state(table_id='train')
    prefix = bqh.export_query_gcs(query, wait=True)
    file_paths = bqh.get_table_gcs(prefix)
    df_users_content = (bqh.get_df_files(file_paths, dtypes=dtypes)
                        .sort_values(['user_id', 'content_id']))


# ### Download Questions Table

# In[44]:


# <hide-input><hide-output>
if False:
    # only 13k rows, so it downloaded directly from BigQuery
    df_questions = bqh.get_df_table('questions',
                                    max_results=None,
                                    dtypes=dtypes).sort_values('question_id')


# ### Update Submission Dataset

# In[45]:


# <hide-input>
if False:
    Path(KAGGLE_SUBMIT_DATASET).mkdir(exist_ok=True)

    model.save_model(f'{KAGGLE_SUBMIT_DATASET}/model.xgb')

    with open(f'{KAGGLE_SUBMIT_DATASET}/columns.json', 'w') as cj:
            json.dump(columns_train, cj)
    
    df_files = {
        # 'df_users.pkl': df_users,
        'df_users_content.pkl': df_users_content,
        'df_questions.pkl': df_questions,
    }

    for file_path, df in df_files.items():
        df.to_pickle(f'{KAGGLE_SUBMIT_DATASET}/{file_path}')
            
    kaggle_id = f"{os.getenv('KAGGLE_USERNAME')}/{KAGGLE_SUBMIT_DATASET}"
    
    metadata = {
        "licenses": [{"name": "CC0-1.0"}],
        "id": kaggle_id,
        "title": KAGGLE_SUBMIT_DATASET
           }

    with open(f'{KAGGLE_SUBMIT_DATASET}/dataset-metadata.json', 'w') as f:
        json.dump(metadata, f)
            
    if kaggle_api.dataset_status(kaggle_id):
        kaggle_api.dataset_create_version(KAGGLE_SUBMIT_DATASET,
                                          version_notes='update dataset',
                                          delete_old_versions=True,
                                          dir_mode='tar',
                                          quiet=True
                                         )
    else:
        kaggle_api.dataset_create_new(KAGGLE_SUBMIT_DATASET,
                                      dir_mode='tar', quiet=True)


# ## Submit From Kernel

# * Go to [RIIID Submit](https://www.kaggle.com/calebeverett/riiid-submit), fork and update to reference your dataset.

# ## Push Kernel to Kaggle

# In[46]:


# <hide-input>
if NOT_KAGGLE:
    if True:
        
        code_file = 'riiid-2020.ipynb'
        with open(DRIVE/REPO/code_file, 'r') as nb:
            nb_json = json.load(nb)       
        
        for i, cell in enumerate(nb_json['cells']):
            if cell['cell_type'] == 'code':
                
                # update show/hide code cells
                for h in ['input', 'output']:
                    if cell['source'][0].find(f'<hide-{h}') > 1:
                        nb_json['cells'][i]['metadata'].update({f'_kg_hide-{h}': True})
                    else:
                        nb_json['cells'][i]['metadata'].pop(f'_kg_hide-{h}', None)

                # add modules as cells
                if len(cell['source']) == 1:
                    groups = re.search(r'(?<=\<include-)(.*?)(?=\>)', cell['source'][0])
                    
                    if groups:
                        with open(DRIVE/REPO/groups.group(0), 'r') as m:
                            nb_json['cells'][i]['source'] = m.readlines() + nb_json['cells'][i]['source']    


        if Path(code_file).exists():
            Path(code_file).unlink()
        
        with open(f'{code_file}', 'w') as f:
            json.dump(nb_json, f)

        data = {'id': 'calebeverett/riiid-bigquery-xgboost-end-to-end',
                        'title': 'RIIID: BigQuery-XGBoost End-to-End',
                        'code_file': code_file,
                        'language': 'python',
                        'kernel_type': 'notebook',
                        'is_private': 'false',
                        'enable_gpu': 'true',
                        'enable_internet': 'true',
                        'dataset_sources': [],
                        'competition_sources': ['riiid-test-answer-prediction'],
                        'kernel_sources': []}
        
        with open('kernel-metadata.json', 'w') as f:
            json.dump(data, f)

        kaggle_api.kernels_push('.')

