# ---------------------- importing libraries ---------------------------------
#
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)  
import os, json, gc, re, random,psutil 
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
from datasets import Dataset, DatasetDict
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
import wandb
from datasets import load_from_disk, load_metric
from utils import *
from const import *
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, \
                         DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, \
                         Seq2SeqTrainer

#---------------------- different inits / configs ----------------------------
#
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)
nltk.download('punkt')

#------------------- Tokenizer for Data --------------------------------------
#
tokenizer = AutoTokenizer.from_pretrained('google/t5-v1_1-base')
dataset = load_from_disk('arxix-paper-abstracts')

def preprocess_data(example):
    model_inputs = tokenizer(example['summaries'], max_length=MAX_SOURCE_LEN, padding=True, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(example['titles'], max_length=MAX_TARGET_LEN, padding=True, truncation=True)
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    ]
    model_inputs['labels'] = labels["input_ids"]
    return model_inputs

# data processing 
# Apply preprocess_data() to the whole dataset
processed_dataset = dataset.map(
    preprocess_data,
    batched=True,
    remove_columns=['summaries', 'titles'],
    desc="Running tokenizer on dataset",
)

# -------------------------------- training settings -------------------------
#
# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="model-t51-base1",
    evaluation_strategy="steps",
    eval_steps=eval_every,
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=weight_decay,
    save_steps=500,
    save_total_limit=3,
    num_train_epochs=num_epochs,
    predict_with_generate=True,
    logging_steps=log_every,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    resume_from_checkpoint=True,
)

#------------------------ model setup-----------------------------------------
#
model = AutoModelForSeq2SeqLM.from_pretrained('google/t5-v1_1-base')
# Define ROGUE metrics on evaluation data
metric = load_metric("rouge")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    # Compute ROUGE scores and get the median scores
    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model,
    training_args,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
#-----------------------------------begin training ----------------------------
#

trainer.train()

# saving model...
