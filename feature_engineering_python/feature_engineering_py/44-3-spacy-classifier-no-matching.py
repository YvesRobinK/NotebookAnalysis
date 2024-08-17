#!/usr/bin/env python
# coding: utf-8

# This notebook extracts words before Abbreviations in prentheses and extracts captilaized word sequences.  These extractions are then processed by a spacy classifier to predict if they are a dataset or not.

# In[1]:


import os
import glob
import numpy as np
import pandas as pd
import re
import simplejson
from joblib import Parallel, delayed
from typing import *
import time

import spacy
from spacy import displacy
from spacy.util import minibatch, compounding
import random

print('packages loaded')


# In[2]:


training_titles_pos = pd.read_csv('../input/training-dataset-titles/dataset226_no_entities.csv')
training_titles_pos['target']=1

training_titles_neg = pd.read_csv('../input/training-dataset-titles/not_dataset.csv')
training_titles_neg["title"] = training_titles_neg["title"].str.lower()


# In[3]:


def clean_text(text: str) -> str:               return re.sub('[^A-Za-z0-9]+', ' ', str(text).lower()).strip()
def clean_texts(texts: List[str]) -> List[str]: return [ clean_text(text) for text in texts ] 

def read_json(index: str, test_train) -> Dict:
    filename = f"../input/coleridgeinitiative-show-us-the-data/{test_train}/{index}.json"
    with open(filename) as f:
        json = simplejson.load(f)
    return json
        
def json2text(index: str, test_train) -> str:
    json  = read_json(index, test_train)
    texts = [
        row["section_title"] + " " + row["text"] 
        for row in json
    ]
    # texts = clean_texts(texts)
    text  = " ".join(texts)
    return text

def filename_to_index(filename):
    return re.sub("^.*/|\.[^.]+$", '', filename)

def glob_to_indices(globpath):
    return list(map(filename_to_index, glob.glob(globpath)))

# Inspired by: https://www.kaggle.com/hamditarek/merge-multiple-json-files-to-a-dataframe
def dataset_df(test_train="test"):
    indices = glob_to_indices(f"../input/coleridgeinitiative-show-us-the-data/{test_train}/*.json")    
    texts   = Parallel(-1)( 
        delayed(json2text)(index, test_train)
        for index in indices  
    )
    df = pd.DataFrame([
        { "id": index, "text": text}
        for index, text in zip(indices, texts)
    ])
    df.to_csv(f"{test_train}.json.csv", index=False)
    return df

train_data = dataset_df("train")
#train_data["text"] = train_data["text"].str.lower()
test_data  = dataset_df("test")
#test_data["text"] = train_data["text"].str.lower()


# In[4]:


# random sample of training data
train_sample=train_data.sample(n = 50, replace=False)

train_sample


# In[5]:


list1=[]

for index, row in training_titles_neg.iterrows():
    #("higher education research and development survey", {'cats': {'POSITIVE': 1}} )
    a='("'
    b='",'
    z="{'cats':{'POSITIVE': 0}}),"
    data=a+row['title']+b+z
    #print (data)
    list1.append(data)
    #print (list1)


# In[6]:


spacy_train_data=[("adni",{'cats':{'POSITIVE': 1}}),
("cccsl",{'cats':{'POSITIVE': 1}}),
("ibtracs",{'cats':{'POSITIVE': 1}}),
("noaa c cap",{'cats':{'POSITIVE': 1}}),
("noaa c-cap",{'cats':{'POSITIVE': 1}}),
("slosh model",{'cats':{'POSITIVE': 1}}),
("noaa tide gauge",{'cats':{'POSITIVE': 1}}),
("noaa tide station",{'cats':{'POSITIVE': 1}}),
("jh-crown registry",{'cats':{'POSITIVE': 1}}),
("jh crown registry",{'cats':{'POSITIVE': 1}}),
("our world in data",{'cats':{'POSITIVE': 1}}),
("noaa tidal station",{'cats':{'POSITIVE': 1}}),
("covid 19 death data",{'cats':{'POSITIVE': 1}}),
("covid-19 death data",{'cats':{'POSITIVE': 1}}),
("common core of data",{'cats':{'POSITIVE': 1}}),
("world ocean database",{'cats':{'POSITIVE': 1}}),
("covid-19 deaths data",{'cats':{'POSITIVE': 1}}),
("census of agriculture",{'cats':{'POSITIVE': 1}}),
("covid 19 genome sequence",{'cats':{'POSITIVE': 1}}),
("noaa water level station",{'cats':{'POSITIVE': 1}}),
("covid-19 genome sequence",{'cats':{'POSITIVE': 1}}),
("nces common core of data",{'cats':{'POSITIVE': 1}}),
("baccalaureate and beyond",{'cats':{'POSITIVE': 1}}),
("noaa world ocean database",{'cats':{'POSITIVE': 1}}),
("aging integrated database",{'cats':{'POSITIVE': 1}}),
("2019-ncov genome sequence",{'cats':{'POSITIVE': 1}}),
("covid 19 genome sequences",{'cats':{'POSITIVE': 1}}),
("covid-19 genome sequences",{'cats':{'POSITIVE': 1}}),
("2019 ncov genome sequence",{'cats':{'POSITIVE': 1}}),
("our world in data covid-19",{'cats':{'POSITIVE': 1}}),
("sars-cov-2 genome sequence",{'cats':{'POSITIVE': 1}}),
("nass census of agriculture",{'cats':{'POSITIVE': 1}}),
("2019-ncov genome sequences",{'cats':{'POSITIVE': 1}}),
("anss comprehensive catalog",{'cats':{'POSITIVE': 1}}),
("our world in data covid 19",{'cats':{'POSITIVE': 1}}),
("2019 ncov genome sequences",{'cats':{'POSITIVE': 1}}),
("sars cov 2 genome sequence",{'cats':{'POSITIVE': 1}}),
("usda census of agriculture",{'cats':{'POSITIVE': 1}}),
("covid open research dataset",{'cats':{'POSITIVE': 1}}),
("genome sequence of covid-19",{'cats':{'POSITIVE': 1}}),
("covid 19 open research data",{'cats':{'POSITIVE': 1}}),
("genome sequence of covid 19",{'cats':{'POSITIVE': 1}}),
("sars-cov-2 genome sequences",{'cats':{'POSITIVE': 1}}),
("rural urban continuum codes",{'cats':{'POSITIVE': 1}}),
("covid-19 open research data",{'cats':{'POSITIVE': 1}}),
("sars cov 2 genome sequences",{'cats':{'POSITIVE': 1}}),
("noaa storm surge inundation",{'cats':{'POSITIVE': 1}}),
("survey of earned doctorates",{'cats':{'POSITIVE': 1}}),
("rural-urban continuum codes",{'cats':{'POSITIVE': 1}}),
("genome sequence of 2019-ncov",{'cats':{'POSITIVE': 1}}),
("education longitudinal study",{'cats':{'POSITIVE': 1}}),
("genome sequence of 2019 ncov",{'cats':{'POSITIVE': 1}}),
("genome sequences of covid 19",{'cats':{'POSITIVE': 1}}),
("genome sequences of covid-19",{'cats':{'POSITIVE': 1}}),
("genome sequences of 2019 ncov",{'cats':{'POSITIVE': 1}}),
("genome sequence of sars-cov-2",{'cats':{'POSITIVE': 1}}),
("genome sequences of 2019-ncov",{'cats':{'POSITIVE': 1}}),
("genome sequence of sars cov 2",{'cats':{'POSITIVE': 1}}),
("genome sequences of sars-cov-2",{'cats':{'POSITIVE': 1}}),
("covid-19 open research dataset",{'cats':{'POSITIVE': 1}}),
("high school longitudinal study",{'cats':{'POSITIVE': 1}}),
("genome sequences of sars cov 2",{'cats':{'POSITIVE': 1}}),
("survey of doctorate recipients",{'cats':{'POSITIVE': 1}}),
("covid 19 image data collection",{'cats':{'POSITIVE': 1}}),
("covid-19 image data collection",{'cats':{'POSITIVE': 1}}),
("covid 19 open research dataset",{'cats':{'POSITIVE': 1}}),
("coastal change analysis program",{'cats':{'POSITIVE': 1}}),
("sars cov 2 full genome sequence",{'cats':{'POSITIVE': 1}}),
("beginning postsecondary student",{'cats':{'POSITIVE': 1}}),
("sars-cov-2 full genome sequence",{'cats':{'POSITIVE': 1}}),
("aging integrated database agid ",{'cats':{'POSITIVE': 1}}),
("nsf survey of earned doctorates",{'cats':{'POSITIVE': 1}}),
("sars-cov-2 full genome sequences",{'cats':{'POSITIVE': 1}}),
("aging integrated database (agid)",{'cats':{'POSITIVE': 1}}),
("beginning postsecondary students",{'cats':{'POSITIVE': 1}}),
("sars cov 2 full genome sequences",{'cats':{'POSITIVE': 1}}),
("school survey on crime and safety",{'cats':{'POSITIVE': 1}}),
("our world in data covid 19 dataset",{'cats':{'POSITIVE': 1}}),
("early childhood longitudinal study",{'cats':{'POSITIVE': 1}}),
("our world in data covid-19 dataset",{'cats':{'POSITIVE': 1}}),
("sars cov 2 complete genome sequence",{'cats':{'POSITIVE': 1}}),
("sars-cov-2 complete genome sequence",{'cats':{'POSITIVE': 1}}),
("2019-ncov complete genome sequences",{'cats':{'POSITIVE': 1}}),
("2019 ncov complete genome sequences",{'cats':{'POSITIVE': 1}}),
("north american breeding bird survey",{'cats':{'POSITIVE': 1}}),
("sars-cov-2 complete genome sequences",{'cats':{'POSITIVE': 1}}),
("ncses survey of doctorate recipients",{'cats':{'POSITIVE': 1}}),
("sars cov 2 complete genome sequences",{'cats':{'POSITIVE': 1}}),
("baltimore longitudinal study of aging",{'cats':{'POSITIVE': 1}}),
("ffrdc research and development survey",{'cats':{'POSITIVE': 1}}),
("national education longitudinal study",{'cats':{'POSITIVE': 1}}),
("national teacher and principal survey",{'cats':{'POSITIVE': 1}}),
("anss comprehensive earthquake catalog",{'cats':{'POSITIVE': 1}}),
("covid 19 open research dataset cord 19 ",{'cats':{'POSITIVE': 1}}),
("agricultural resource management survey",{'cats':{'POSITIVE': 1}}),
("agricultural resources management survey",{'cats':{'POSITIVE': 1}}),
("covid-19 open research dataset (cord-19)",{'cats':{'POSITIVE': 1}}),
("usgs north american breeding bird survey",{'cats':{'POSITIVE': 1}}),
("national water level observation network",{'cats':{'POSITIVE': 1}}),
("north american breeding bird survey bbs ",{'cats':{'POSITIVE': 1}}),
("north american breeding bird survey (bbs)",{'cats':{'POSITIVE': 1}}),
("nsf ffrdc research and development survey",{'cats':{'POSITIVE': 1}}),
("national assessment of education progress",{'cats':{'POSITIVE': 1}}),
("alzheimers disease neuroimaging initiative",{'cats':{'POSITIVE': 1}}),
("coastal change analysis program land cover",{'cats':{'POSITIVE': 1}}),
("baccalaureate and beyond longitudinal study",{'cats':{'POSITIVE': 1}}),
("baltimore longitudinal study of aging blsa ",{'cats':{'POSITIVE': 1}}),
("sea lake and overland surges from hurricanes",{'cats':{'POSITIVE': 1}}),
("baltimore longitudinal study of aging (blsa)",{'cats':{'POSITIVE': 1}}),
("noaa national water level observation network",{'cats':{'POSITIVE': 1}}),
("optimum interpolation sea surface temperature",{'cats':{'POSITIVE': 1}}),
("sea surface temperature optimum interpolation",{'cats':{'POSITIVE': 1}}),
("survey of industrial research and development",{'cats':{'POSITIVE': 1}}),
("nws storm surge risk",{'cats':{'POSITIVE': 1}}),
("storm surge risk",{'cats':{'POSITIVE': 1}}),
("cas covid 19 antiviral candidate compounds data",{'cats':{'POSITIVE': 1}}),
("sea surface temperature - optimum interpolation",{'cats':{'POSITIVE': 1}}),
("cas covid-19 antiviral candidate compounds data",{'cats':{'POSITIVE': 1}}),
("higher education research and development survey",{'cats':{'POSITIVE': 1}}),
("rsna international covid open radiology database",{'cats':{'POSITIVE': 1}}),
("alzheimer s disease neuroimaging initiative adni ",{'cats':{'POSITIVE': 1}}),
("noaa sea lake and overland surges from hurricanes",{'cats':{'POSITIVE': 1}}),
("noaa sea lake and overland surges from hurricanes",{'cats':{'POSITIVE': 1}}),
("nsf survey of industrial research and development",{'cats':{'POSITIVE': 1}}),
("arms farm financial and crop production practices",{'cats':{'POSITIVE': 1}}),
("noaa optimum interpolation sea surface temperature",{'cats':{'POSITIVE': 1}}),
("cas covid 19 antiviral candidate compounds dataset",{'cats':{'POSITIVE': 1}}),
("alzheimer's disease neuroimaging initiative (adni)",{'cats':{'POSITIVE': 1}}),
("cas covid-19 antiviral candidate compounds dataset",{'cats':{'POSITIVE': 1}}),
("cas covid 19 antiviral candidate compounds data set",{'cats':{'POSITIVE': 1}}),
("cas covid-19 antiviral candidate compounds data set",{'cats':{'POSITIVE': 1}}),
("survey of state government research and development",{'cats':{'POSITIVE': 1}}),
("rsna international covid-19 open radiology database",{'cats':{'POSITIVE': 1}}),
("beginning postsecondary students longitudinal study",{'cats':{'POSITIVE': 1}}),
("rsna international covid 19 open radiology database",{'cats':{'POSITIVE': 1}}),
("nsf higher education research and development survey",{'cats':{'POSITIVE': 1}}),
("trends in international mathematics and science study",{'cats':{'POSITIVE': 1}}),
("noaa c-cap",{'cats':{'POSITIVE': 1}}),
("noaa c cap",{'cats':{'POSITIVE': 1}}),
("survey of science and engineering research facilities",{'cats':{'POSITIVE': 1}}),
("advanced national seismic system comprehensive catalog",{'cats':{'POSITIVE': 1}}),
("complexity science hub covid-19 control strategies list",{'cats':{'POSITIVE': 1}}),
("survey of earned doctorates",{'cats':{'POSITIVE': 1}}),
("complexity science hub covid 19 control strategies list",{'cats':{'POSITIVE': 1}}),
("international best track archive for climate stewardship",{'cats':{'POSITIVE': 1}}),
("nsf survey of science and engineering research facilities",{'cats':{'POSITIVE': 1}}),
("survey of doctorate recipients",{'cats':{'POSITIVE': 1}}),
("rsna international covid 19 open radiology database ricord ",{'cats':{'POSITIVE': 1}}),
("common core of data",{'cats':{'POSITIVE': 1}}),
("rsna international covid-19 open radiology database (ricord)",{'cats':{'POSITIVE': 1}}),
("noaa international best track archive for climate stewardship",{'cats':{'POSITIVE': 1}}),
("program for the international assessment of adult competencies",{'cats':{'POSITIVE': 1}}),
("complexity science hub covid 19 control strategies list cccsl ",{'cats':{'POSITIVE': 1}}),
("complexity science hub covid-19 control strategies list (cccsl)",{'cats':{'POSITIVE': 1}}),
("covid-19 precision medicine analytics platform registry (jh-crown)",{'cats':{'POSITIVE': 1}}),
("advanced national seismic system anss comprehensive catalog comcat ",{'cats':{'POSITIVE': 1}}),
("world ocean database",{'cats':{'POSITIVE': 1}}),
("advanced national seismic system (anss) comprehensive catalog (comcat)",{'cats':{'POSITIVE': 1}}),
("survey of industrial research and development",{'cats':{'POSITIVE': 1}}),
("survey of graduate students and postdoctorates in science and engineering",{'cats':{'POSITIVE': 1}}),
("higher education research and development survey",{'cats':{'POSITIVE': 1}}),
("nsf survey of graduate students and postdoctorates in science and engineering",{'cats':{'POSITIVE': 1}}),
("characterizing health associated risks and your baseline disease in sars cov 2",{'cats':{'POSITIVE': 1}}),
("characterizing health associated risks and your baseline disease in sars-cov-2",{'cats':{'POSITIVE': 1}}),
("ncses survey of graduate students and postdoctorates in science and engineering",{'cats':{'POSITIVE': 1}}),
("genetics of alzheimer s disease data storage site",{'cats':{'POSITIVE': 1}}),
("genetics of alzheimer's disease data storage site",{'cats':{'POSITIVE': 1}}),
("survey of science and engineering research facilities",{'cats':{'POSITIVE': 1}}),
("survey of earned doctorates",{'cats':{'POSITIVE': 1}}),
("survey of doctorate recipients",{'cats':{'POSITIVE': 1}}),
("characterizing health associated risks and your baseline disease in sars cov 2 charybdis ",{'cats':{'POSITIVE': 1}}),
("characterizing health associated risks and your baseline disease in sars-cov-2 (charybdis)",{'cats':{'POSITIVE': 1}}),
("alzheimer s disease data storage site niagads ",{'cats':{'POSITIVE': 1}}),
("alzheimer's disease data storage site (niagads)",{'cats':{'POSITIVE': 1}}),
("optimum interpolation sea surface temperature",{'cats':{'POSITIVE': 1}}),
("survey of industrial research and development",{'cats':{'POSITIVE': 1}}),
("survey of graduate students and postdoctorates in science and engineering",{'cats':{'POSITIVE': 1}}),
("higher education research and development survey",{'cats':{'POSITIVE': 1}}),
("survey of science and engineering research facilities",{'cats':{'POSITIVE': 1}}),
("survey of graduate students and postdoctorates in science and engineering",{'cats':{'POSITIVE': 1}}),
("local food economic impact assessment",{'cats':{'POSITIVE': 1}}),
("tvb model",{'cats':{'POSITIVE': 1}}),
("mexican american study project",{'cats':{'POSITIVE': 1}}),
("national longitudinal study of adolescent health",{'cats':{'POSITIVE': 1}}),
("national health interview survey",{'cats':{'POSITIVE': 1}}),
("project talent longitudinal study",{'cats':{'POSITIVE': 1}}),
("sea lake and overland surges from hurricanes slosh model",{'cats':{'POSITIVE': 1}}),
("schools and staffing survey",{'cats':{'POSITIVE': 1}}),
("private school universe survey",{'cats':{'POSITIVE': 1}}),
("fast response survey system",{'cats':{'POSITIVE': 1}}),
("schools and staffing survey",{'cats':{'POSITIVE': 1}}),
("national assessment of educational progress",{'cats':{'POSITIVE': 1}}),
("coinmon core of data",{'cats':{'POSITIVE': 1}}),
("national postsecondary student aid study",{'cats':{'POSITIVE': 1}}),
("postsecondary student aid study",{'cats':{'POSITIVE': 1}}),
("national study of postsecondary faculty",{'cats':{'POSITIVE': 1}}),
("survey on vocational programs in secondary schools",{'cats':{'POSITIVE': 1}}),
("district survey of alternative schools and programs",{'cats':{'POSITIVE': 1}}),
("state nonfiscal survey of public elementary secondary education",{'cats':{'POSITIVE': 1}}),
("national public education financial survey",{'cats':{'POSITIVE': 1}}),
("annual survey of government finances school systems f 33 survey",{'cats':{'POSITIVE': 1}}),
("integrated postsecondary education data system",{'cats':{'POSITIVE': 1}}),
("ipeds",{'cats':{'POSITIVE': 1}}),
("nsopf",{'cats':{'POSITIVE': 1}}),
("bps data",{'cats':{'POSITIVE': 1}}),
("postsecondary students longitudinal study",{'cats':{'POSITIVE': 1}}),
("public elementary secondary school universe survey",{'cats':{'POSITIVE': 1}}),
("ccd local education agency universe survey",{'cats':{'POSITIVE': 1}}),
("ccd national public education financial survey",{'cats':{'POSITIVE': 1}}),
("nces national education longitudinal study of 1988",{'cats':{'POSITIVE': 1}}),
("agricultural resource management survey",{'cats':{'POSITIVE': 1}}),
("aez model",{'cats':{'POSITIVE': 1}}),
("usda corn yield data",{'cats':{'POSITIVE': 1}}),
("argonne national laboratory s greet",{'cats':{'POSITIVE': 1}}),
("greet model",{'cats':{'POSITIVE': 1}}),
("argonne national laboratory s cclub model",{'cats':{'POSITIVE': 1}}),
("national education longitudinal survey",{'cats':{'POSITIVE': 1}}),
("education longitudinal study of 2002",{'cats':{'POSITIVE': 1}}),
("progress in international reading literacy study",{'cats':{'POSITIVE': 1}}),
("usgs national water quality assessment",{'cats':{'POSITIVE': 1}}),
("ipeds fall enrollment dataset",{'cats':{'POSITIVE': 1}}),
("fall enrollment dataset",{'cats':{'POSITIVE': 1}}),
("delta cost project",{'cats':{'POSITIVE': 1}}),
("annual survey of colleges standard research compilation",{'cats':{'POSITIVE': 1}}),
("chesapeake bay watershed land cover data series",{'cats':{'POSITIVE': 1}}),
("nccpi corn and soybeans sub model",{'cats':{'POSITIVE': 1}}),
("national longitudinal survey of youth",{'cats':{'POSITIVE': 1}}),
("national survey of teachers",{'cats':{'POSITIVE': 1}}),
("high school transcript study",{'cats':{'POSITIVE': 1}}),
("clinical dementia rating",{'cats':{'POSITIVE': 0}}),
("services web feature services",{'cats':{'POSITIVE': 0}}),
("university south carolina",{'cats':{'POSITIVE': 0}}),
("latent dirichlet allocation",{'cats':{'POSITIVE': 0}}),
("montreal neurological institute",{'cats':{'POSITIVE': 0}}),
("national board professional teaching standards",{'cats':{'POSITIVE': 0}}),
("multiple endmember spectral mixture analysis",{'cats':{'POSITIVE': 0}}),
("bay katmai national park",{'cats':{'POSITIVE': 0}}),
("united nations development programme",{'cats':{'POSITIVE': 0}}),
("compact high resolution imaging spectrometer",{'cats':{'POSITIVE': 0}}),
("office management budget",{'cats':{'POSITIVE': 0}}),
("florida department natural resources",{'cats':{'POSITIVE': 0}}),
("university neuroinformatics research group",{'cats':{'POSITIVE': 0}}),
("national canadian pacific rail service",{'cats':{'POSITIVE': 0}}),
("portobello marine laboratory",{'cats':{'POSITIVE': 0}}),
("hilbert schmidt independence criterion",{'cats':{'POSITIVE': 0}}),
("genomic distance-based regression",{'cats':{'POSITIVE': 0}}),
("national weather service",{'cats':{'POSITIVE': 0}}),
("deep boltzmann machine",{'cats':{'POSITIVE': 0}}),
("multiple indicators multiple cause",{'cats':{'POSITIVE': 0}}),
("quasi maximum likelihood",{'cats':{'POSITIVE': 0}}),
("the eastern equatorial pacific",{'cats':{'POSITIVE': 0}}),
("autoregressive distributed lag",{'cats':{'POSITIVE': 0}}),
("city chesapeake mosquito control commission",{'cats':{'POSITIVE': 0}}),
("the california verbal learning test",{'cats':{'POSITIVE': 0}}),
("long short term memory",{'cats':{'POSITIVE': 0}}),
("detection computer-aided detection",{'cats':{'POSITIVE': 0}}),
("stacked sparse autoencoder",{'cats':{'POSITIVE': 0}}),
("brain tumor segmentation",{'cats':{'POSITIVE': 0}}),
("geriatric depression scale",{'cats':{'POSITIVE': 0}}),
("australian institute teaching school leadership",{'cats':{'POSITIVE': 0}}),
("glasgow coma scale",{'cats':{'POSITIVE': 0}}),
("shared environmental information system",{'cats':{'POSITIVE': 0}}),
("international conferences agricultural statistics",{'cats':{'POSITIVE': 0}}),
("least absolute shrinkage selection operator",{'cats':{'POSITIVE': 0}}),
("color trails test",{'cats':{'POSITIVE': 0}}),
("international consortium brain mapping",{'cats':{'POSITIVE': 0}}),
("hardyweinberg disequilibrium",{'cats':{'POSITIVE': 0}}),
("advanced very high resolution radiometer",{'cats':{'POSITIVE': 0}}),
("spatial empirical bayesian",{'cats':{'POSITIVE': 0}}),
("rectified linear unit",{'cats':{'POSITIVE': 0}}),
("networks generative adversarial networks",{'cats':{'POSITIVE': 0}}),
("digital database screening mammography",{'cats':{'POSITIVE': 0}}),
("rhode island emergency management agency",{'cats':{'POSITIVE': 0}}),
("national marine fisheries service",{'cats':{'POSITIVE': 0}}),
("office marine aviation operations",{'cats':{'POSITIVE': 0}}),
("national oceanic atmospheric administration",{'cats':{'POSITIVE': 0}}),
("national hurricane center",{'cats':{'POSITIVE': 0}}),
("federal emergency management administration",{'cats':{'POSITIVE': 0}}),
("the social science genetic association consortium",{'cats':{'POSITIVE': 0}}),
("usda economic research service",{'cats':{'POSITIVE': 0}}),
("world health organization",{'cats':{'POSITIVE': 0}}),
("global positioning systems",{'cats':{'POSITIVE': 0}}),
("national center education statistics",{'cats':{'POSITIVE': 0}}),
("amyotrophic lateral sclerosis",{'cats':{'POSITIVE': 0}}),
("the economic research service",{'cats':{'POSITIVE': 0}}),
("clinical advisory board",{'cats':{'POSITIVE': 0}}),
("institute education sciences",{'cats':{'POSITIVE': 0}}),
("national centers environmental prediction",{'cats':{'POSITIVE': 0}}),
("coastal environmental research committee",{'cats':{'POSITIVE': 0}}),
("non-governmental organizations",{'cats':{'POSITIVE': 0}}),
("high performance computing",{'cats':{'POSITIVE': 0}}),
("engineer research development center",{'cats':{'POSITIVE': 0}}),
("usda economic research service",{'cats':{'POSITIVE': 0}}),
("national oceanic atmospheric administration",{'cats':{'POSITIVE': 0}}),
("national hurricane center",{'cats':{'POSITIVE': 0}}),
("economic research service",{'cats':{'POSITIVE': 0}}),
("higher order statistics",{'cats':{'POSITIVE': 0}}),
("common data elements",{'cats':{'POSITIVE': 0}}),
("north east south east",{'cats':{'POSITIVE': 0}}),
("national climatic data center",{'cats':{'POSITIVE': 0}}),
("national institute aging",{'cats':{'POSITIVE': 0}}),
("genome wide",{'cats':{'POSITIVE': 0}}),
("national science foundation",{'cats':{'POSITIVE': 0}}),
("national center education statistics",{'cats':{'POSITIVE': 0}}),
("local binary pattern",{'cats':{'POSITIVE': 0}}),
("higher order statistics",{'cats':{'POSITIVE': 0}}),
("north east south east",{'cats':{'POSITIVE': 0}}),
("world health organization",{'cats':{'POSITIVE': 0}}),
("clinically diagnosed dementia alzheimer type",{'cats':{'POSITIVE': 0}}),
("u s geological survey",{'cats':{'POSITIVE': 0}}),
("c pittsburgh compound b",{'cats':{'POSITIVE': 0}}),
("principal component analysis",{'cats':{'POSITIVE': 0}}),
("u s department education",{'cats':{'POSITIVE': 0}}),
("u s trained",{'cats':{'POSITIVE': 0}}),
("coronaviruses",{'cats':{'POSITIVE': 0}}),
("ordinary least square",{'cats':{'POSITIVE': 0}}),
("materials and methods u s geological survey",{'cats':{'POSITIVE': 0}}),
("mean decrease accuracy",{'cats':{'POSITIVE': 0}}),
("undergraduate training program",{'cats':{'POSITIVE': 0}}),
("education for all",{'cats':{'POSITIVE': 0}}),
("foldcurv curvind gauscurv",{'cats':{'POSITIVE': 0}}),
("community atmosphere model",{'cats':{'POSITIVE': 0}}),
("annual percent change",{'cats':{'POSITIVE': 0}}),
("new york stock exchange",{'cats':{'POSITIVE': 0}}),
("average treatment effect",{'cats':{'POSITIVE': 0}}),
("demographic",{'cats':{'POSITIVE': 0}}),
("introduction coronaviruses",{'cats':{'POSITIVE': 0}}),
("autism brain imaging data exchange",{'cats':{'POSITIVE': 0}}),
("national cancer data base",{'cats':{'POSITIVE': 0}}),
("programa internacional avalia o alunos",{'cats':{'POSITIVE': 0}}),
("national science foundation s",{'cats':{'POSITIVE': 0}}),
("tes bakat skolastik",{'cats':{'POSITIVE': 0}}),
("oregon department environmental quality",{'cats':{'POSITIVE': 0}}),
("study variables facility oncology registry data standards",{'cats':{'POSITIVE': 0}}),
("state university entrance",{'cats':{'POSITIVE': 0}}),
("south china sea",{'cats':{'POSITIVE': 0}}),
("introduction human",{'cats':{'POSITIVE': 0}}),
("dulbecco s eagle s",{'cats':{'POSITIVE': 0}}),
("institute medical genetics cardiff",{'cats':{'POSITIVE': 0}}),
("sars cov 2 a",{'cats':{'POSITIVE': 0}}),
("disease coronaviruses",{'cats':{'POSITIVE': 0}}),
("office management budget s",{'cats':{'POSITIVE': 0}}),
("rna dependent rna",{'cats':{'POSITIVE': 0}}),
("t4 dna ligase",{'cats':{'POSITIVE': 0}}),
("u s department agriculture",{'cats':{'POSITIVE': 0}}),
("chicago parent program",{'cats':{'POSITIVE': 0}}),
("indian ocean",{'cats':{'POSITIVE': 0}}),
("turkish republic northern cyprus",{'cats':{'POSITIVE': 0}}),
("the south african schools act",{'cats':{'POSITIVE': 0}}),
("blood oxygen level dependent",{'cats':{'POSITIVE': 0}}),
("duke university s institutional review board",{'cats':{'POSITIVE': 0}}),
("international archive education data",{'cats':{'POSITIVE': 0}}),
("nonprofit technology network",{'cats':{'POSITIVE': 0}}),
("muller s opportunity tolearn",{'cats':{'POSITIVE': 0}}),
("global extreme sea level analysis",{'cats':{'POSITIVE': 0}}),
("the addenbrooke s cognitive examination",{'cats':{'POSITIVE': 0}}),
("information resource incorporated",{'cats':{'POSITIVE': 0}}),
("norwegian cognitive neurogenetics",{'cats':{'POSITIVE': 0}}),
("introduction cape hatteras national seashore",{'cats':{'POSITIVE': 0}}),
("heart study second generation cohort",{'cats':{'POSITIVE': 0}}),
("national hurricane center",{'cats':{'POSITIVE': 0}}),
("second follow up",{'cats':{'POSITIVE': 0}}),
("title iv",{'cats':{'POSITIVE': 0}}),
("jackknife ii",{'cats':{'POSITIVE': 0}}),
("statistics brief",{'cats':{'POSITIVE': 0}}),
("research cnn rnn based",{'cats':{'POSITIVE': 0}}),
("ct ct",{'cats':{'POSITIVE': 0}}),
("designed cnn",{'cats':{'POSITIVE': 0}}),
("pca and umap assisted k means",{'cats':{'POSITIVE': 0}}),
("in figure",{'cats':{'POSITIVE': 0}}),
("roundy frank",{'cats':{'POSITIVE': 0}}),
("abnormality recognition early ad",{'cats':{'POSITIVE': 0}}),
("lite ed2 5 top of atmosphere",{'cats':{'POSITIVE': 0}}),
("global forest resource assessment",{'cats':{'POSITIVE': 0}}),
("foodprint model",{'cats':{'POSITIVE': 0}}),
("local tci slr relied",{'cats':{'POSITIVE': 0}}),
("ad t1 weighted",{'cats':{'POSITIVE': 0}}),
("italy england sars cov 2",{'cats':{'POSITIVE': 0}}),
("wuhan italy",{'cats':{'POSITIVE': 0}}),
("the nsf",{'cats':{'POSITIVE': 0}}),
("comparison previous study the",{'cats':{'POSITIVE': 0}}),
("the netherlands",{'cats':{'POSITIVE': 0}}),
("international database",{'cats':{'POSITIVE': 0}}),
("assessment martin",{'cats':{'POSITIVE': 0}}),
("sas system windows",{'cats':{'POSITIVE': 0}}),
("section ii",{'cats':{'POSITIVE': 0}}),
("d cnn",{'cats':{'POSITIVE': 0}}),
("in table",{'cats':{'POSITIVE': 0}}),
("different adni 1database",{'cats':{'POSITIVE': 0}}),
("adni 1 adni 2",{'cats':{'POSITIVE': 0}}),
("results we",{'cats':{'POSITIVE': 0}}),
("whereas foland ross",{'cats':{'POSITIVE': 0}}),
("performance iq",{'cats':{'POSITIVE': 0}}),
("verbal iq",{'cats':{'POSITIVE': 0}}),
("nc flood risk information system",{'cats':{'POSITIVE': 0}}),
("nc sea level rise risk management study",{'cats':{'POSITIVE': 0}}),
("canada",{'cats':{'POSITIVE': 0}}),
("germany",{'cats':{'POSITIVE': 0}})
]
print ('training data loaded')


# In[7]:


#train_data = [("higher education research and development survey", {'cats': {'POSITIVE': 1}} ), ("the social science genetic association consortium", {'cats': {'POSITIVE': 0}})]

nlp = spacy.load('en_core_web_sm')

if 'textcat' not in nlp.pipe_names:
    textcat = nlp.create_pipe("textcat")
    nlp.add_pipe(textcat, last=True)
else:
    textcat = nlp.get_pipe("textcat")

textcat.add_label('POSITIVE')
#textcat.add_label('NEGATIVE')

other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']

n_iter = 15


with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.begin_training()
    print("Training model...")
    
    for i in range(n_iter):
        losses = {}
        batches = minibatch(spacy_train_data, size=compounding(4,32,1.001))
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(texts, annotations, sgd=optimizer,drop=0.2, losses=losses)
            
print('done')


# In[8]:


df = pd.read_csv('../input/training-dataset-titles/data_used_language.csv')
df.drop_duplicates(keep="first", inplace=True)

data_language=df['words'].astype(str).values.tolist()

#print (data_language)

long_string="In principle, it would have been possible to generate as well a multiplicity of alternatives same SCbi-MFM counterparts, leading to an even more varied augmented dataset, but we were limited by the computational resources needed to build a multiplicity of alternative SCbi-MFM 's for each of the subjects in our ADNI-derived dataset (nonlinear FC-to-SC completion is way harder computationally than SC-to-FC completion)"

words = [word for word in data_language if word in long_string]

print (len(words))


# In[9]:


from gensim.parsing.preprocessing import remove_stopwords
from nltk.tokenize import sent_tokenize

#### remove >.5 jaccard matches from predicitons
def jaccard_similarity(s1, s2):
    l1 = s1.split()
    l2 = s2.split()    
    intersection = len(list(set(l1).intersection(l2)))
    union = (len(l1) + len(l2)) - intersection
    return float(intersection) / union

#############################

start_time = time.time()
#test
column_names = ["Id", "PredictionString"]
#train
#column_names = ["Id", "PredictionString", "matched"]
submission = pd.DataFrame(columns = column_names)

no_delete = ['study', 'dataset', 'model','survey','data','adni','codes', 'genome', 'program','assessment','database','census','initiative','gauge','system','stewardship','surge']

#doc1 = nlp(u"Data used in preparation of this article were obtained from")
#doc1 = nlp("using data from")
#doc1 = nlp("used estimate studied data from")

for index, row in test_data.iterrows():
    #print ('############################')
    #print (row['id'])
    to_append=[row['id'],'']
    #to_append=[row['id'],'','']
    passage=row['text']
    passage=passage.replace("'s","s")
    passage=passage.replace("-"," ")
    passage=passage.replace(","," ")
    ##### isin
    #for index, row2 in training_titles_pos.iterrows():
        #query_string = str(row2['title'].lower())
        #if query_string in passage.lower(): #and "data" in sentence
            #print ('-->',query_string)
    #print ('____________________')
    
    ##################### sentences 
    #sentences=passage.split(".")
    #sentences=sent_tokenize(passage)
    #for sentence in sentences:
        #if len(sentence)>100:
            #words = [word for word in data_language if word in sentence.lower()]
            #if len(words)>=2:
    ######## ACRONYMS
    for match in re.finditer(r"(\(([A-Z]{2,})\))", passage):
    #for match in re.finditer(r"(\((.*?)\))", data):
        caps=[]
        start_index = match.start()
        abbr = match.group(1)
        size = len(abbr)
        words = passage[:start_index].split()[-size:]
        #print (abbr,words)
        for word in words:
            if word[0].isupper():
                caps.append(word)
        definition = " ".join(caps)
        #print (definition)
        if sum(1 for c in definition if c.isupper()) < 15:
            words = [word for word in no_delete if word in definition.lower()]
            doc=nlp(definition)
            score=doc.cats['POSITIVE']
            #print(definition,abbr, score)
            if len(words)>0 and  score > .99:
                #print(definition,abbr, score)
                if to_append[1]!='' and definition not in to_append[1]:
                    to_append[1] = to_append[1]+'|'+definition+'|'+abbr
                    to_append[1] = to_append[1]+'|'+abbr
                if to_append[1]=='':
                    to_append[1] = definition
                    to_append[1] = to_append[1]+'|'+abbr
                            
    ######
    #### cap word sequence
    if to_append[1]=='':        
        mylist=re.findall('([A-Z][\w-]*(?:\s+[A-Z][\w-]*)+)', remove_stopwords(passage))
        mylist = list(dict.fromkeys(mylist))
        for match in mylist:
            upper_score=sum(1 for c in match if c.isupper())
            #print (match, upper_score)
            if upper_score < 15:
                words = [word for word in no_delete if word in match.lower()]
                doc=nlp(match)
                score=doc.cats['POSITIVE']
                #print (match, score)
                if len(words)>0 and len(match.split())>=2 and score > .99:
                    #print (match, score)
                    #print ('__________________')
                    if to_append[1]!='' and match not in to_append[1]:
                        to_append[1] = to_append[1]+'|'+match
                    if to_append[1]=='':
                        to_append[1] = match
            
                            
    #if row['id']=='3f316b38-1a24-45a9-8d8c-4e05a42257c6':
        #print ("start next paper")
        #print (passage)
    
    #print (to_append[1])
    ###### remove similar jaccard
    got_label=to_append[1].split('|')
    filtered=[]
    filtered_labels = ''
    for label in sorted(got_label, key=len):
        label = clean_text(label)
        if len(filtered) == 0 or all(jaccard_similarity(label, got_label) < .5 for got_label in filtered):
            filtered.append(label)
            if filtered_labels!='':
                filtered_labels=filtered_labels+'|'+label
            if filtered_labels=='':
                filtered_labels=label
    
    to_append[1] = filtered_labels  
    
    
    #print ('################')
    #print (to_append)
    #print ('################')
    ###### remove similar jaccard
    df_length = len(submission)
    submission.loc[df_length] = to_append
print("--- %s seconds ---" % (time.time() - start_time))
submission.to_csv('submission.csv', index = False)
submission

