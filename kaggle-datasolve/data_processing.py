import keras
import torch
import transformers
import pandas as pd


train = pd.read_csv('data/train.csv')
train.sort_values(by=['id', 'cat_name'], inplace=True)


def process_data(train):
    train.sort_values(by=['id'], inplace=True)
    encoded_labels = pd.get_dummies(train['cat_name'])
    encoded_labels['id'] = train['id']
    Y = encoded_labels.groupby('id').sum()
    train = train.drop_duplicates(subset=['id'])
    train.index = train['id'].copy()
    train=train.drop(columns=['id', 'cat_name'])
    combined = train.join(Y, on='id')
    return train, Y, combined
    

if __name__ == "__main__":
    train, Y, combined = process_data(train)
    print(train.shape)
    print(Y.shape)
    print(combined.shape)
    print(combined.head())
    combined.to_csv('data/train_processed_.csv')