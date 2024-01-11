import keras
import torch
import transformers
import pandas as pd
from datasets import Dataset
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

model_name = 'distilbert-base-uncased'
df = pd.read_csv('data/train_processed.csv')


def process_dataset(model_name, df):
    dataset = Dataset.from_pandas(df)
    label_cols = [col for col in dataset.column_names if col not in ['document_text', 'name', 'id']]

    def create_labels(example):
        example['labels'] = [example[col] for col in label_cols]
        return example

    dataset = dataset.map(create_labels, remove_columns=label_cols)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(example):
        # text = example['name'] + " " + tokenizer.sep_token + " " + example['document_text']
        # return tokenizer(text, truncation=True, add_special_tokens=True, max_length=512)
        return tokenizer(example['name'], example['document_text'], truncation=True, add_special_tokens=True, max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, remove_columns=['document_text', 'name', 'id'], batched=True)
    # tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2, seed=42, shuffle=True)
    tokenized_dataset.save_to_disk('data/' + model_name + '_tokenized_dataset')
    return tokenized_dataset



tokenized_dataset = process_dataset(model_name, df)
print(tokenized_dataset)
print(tokenized_dataset['train'][0])