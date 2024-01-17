import numpy as np
import pandas as pd
import glob
import os
import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoModel, AutoTokenizer

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint = "prajjwal1/bert-tiny"  # L=2, H=128
bert_model = AutoModel.from_pretrained(checkpoint)
bert_tokenizer = AutoTokenizer.from_pretrained(checkpoint)


class UserDataset(Dataset):
    def __init__(self, data_dir, tokenizer=bert_tokenizer, model=bert_model, max_len=512):
        self.tokenizer = tokenizer
        self.model = model
        self.max_len = max_len

        self.books = pd.read_csv(os.path.join(data_dir, 'processed', 'books.csv'))
        # book cols are ['Id', 'Name', 'Authors', 'Rating', 'PublishYear', 'Description']

        self.ratings = pd.read_csv(os.path.join(data_dir, 'processed', 'ratings.csv'))
        # rating cols are ['ID', 'Name', 'Rating']

        self.user_ids = self.ratings['ID'].unique()

    def __len__(self):
        return len(self.books)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        user_ratings = self.ratings[self.ratings['ID'] == user_id]
        # take one random book from the user's ratings
        user_rating = user_ratings.sample(1)
        book = self.books[self.books['Name'] == user_rating['Name'].values[0]]
        if len(book) == 0:
            print(user_ratings)  # TODO some books are not in the books dataset, why?
        book = book.sample(1)
        print(book['Name'].values)
        e = (book['Authors'].values + ' ' +
             str(book['Rating'].values) + ' ' +
             str(book['PublishYear'].values) + ' ' +
             book['Description'].values)
        # print(e)
        # e = data['Description']
        # e = e.replace("\n", " ")
        # encoded_input = bert_tokenizer.encode_plus(
        #     e,
        #     add_special_tokens=True,
        #     padding='max_length',
        #     truncation=True,
        #     max_length=self.max_len,
        #     return_attention_mask=True,
        #     return_tensors='pt'
        # )
        # # print(encoded_input['input_ids'].shape)
        # return {
        #     'input_ids': encoded_input['input_ids'].flatten(),  # B * 512 * 128 (L=hidden_size)
        #     'attention_mask': encoded_input['attention_mask'].flatten(),
        #     'target': torch.tensor(self.data.iloc[index]['Rating'], dtype=torch.float)
        # }

    def book_name2id(self, name):
        book_id = self.books[self.books['Name'] == name]['Id'].values
        return book_id[0] if len(book_id) > 0 else None

    def get_book(self, b_idx):
        # print(self.books['Id'].head(5))
        book = self.books.loc[self.books['Id'] == b_idx]
        e = (book['Authors'].values[0] + ' ' +
             book['Year'].values[0] + ' ' +
             book['Rating'].values[0] + ' ' +
             book['PublishYear'].values[0] + ' ' +
             book['Description'].values[0])
        e = e.replace("\n", " ")


        # encoded_input = bert_tokenizer.encode_plus(
        #     e,
        #     add_special_tokens=True,
        #     padding='max_length',
        #     truncation=True,
        #     max_length=self.max_len,
        #     return_attention_mask=True,
        #     return_tensors='pt'
        # )
        # print(encoded_input['input_ids'].shape)
        # return {
        #     'input_ids': encoded_input['input_ids'].flatten(),  # B * 512 * 128 (L=hidden_size)
        #     'attention_mask': encoded_input['attention_mask'].flatten(),
        #     'target': torch.from_numpy(book['Rating'].values.astype(float))
        # }

    def get_rating(self, u_idx, b_idx=None):
        data = self.ratings
        user = data[data['ID'] == u_idx]

        user_embed = torch.zeros(512)
        # print(user.columns)
        for idx, row in user.iterrows():
            # print(row['Name'])
            # break
            book_id = self.book_name2id(row['Name'])
            if book_id is None:
                continue
            if b_idx is not None and book_id == b_idx:
                continue
            book_embed = self.get_book(book_id)['input_ids']
            # print(f'{book_embed.shape=} {sm_user.shape=}')
            user_embed += book_embed

            print(f'{book_embed.sum()} {user_embed.sum()}')
            assert False
        user_embed /= user.shape[0]

        return user_embed

