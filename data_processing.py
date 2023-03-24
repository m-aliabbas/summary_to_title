# ------------------------ Importig different Modules ------------------------
#
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)  
import os, json, gc, re, random,psutil 
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
import warnings
import logging
from datasets import Dataset, DatasetDict
import nltk
from nltk.tokenize import sent_tokenize
from datasets import load_from_disk
from utils import *
#------------------------ Different Configs ----------------------------------
#
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

#---------------------- File loading -----------------------------------------
#
papers = pd.read_csv('arxiv_data.csv')
# splitting the data
df_train,df_test = train_test_split(papers,test_size=0.2,random_state=1231)
dataset_train = Dataset.from_pandas(df_train)
dataset_validation = Dataset.from_pandas(df_test)
ds = DatasetDict()
ds['train'] = dataset_train
ds['validation'] = dataset_validation
# save as huggingface dataset 
ds.save_to_disk('arxix-paper-abstracts')