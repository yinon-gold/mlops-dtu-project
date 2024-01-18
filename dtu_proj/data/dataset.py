import pandas as pd
import os

import torch
from torch.utils.data import Dataset, DataLoader

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class UserDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.books = pd.read_csv(os.path.join(data_dir, 'processed', 'books.csv'))
        # book cols are ['book_id', 'name', 'authors', 'avg_rating', 'publish_year', 'desc']

        self.ratings = pd.read_csv(os.path.join(data_dir, 'processed', 'ratings.csv'))
        # rating cols are ['user_id', 'name', 'rating']

        self.user_ids = self.ratings['user_id'].unique()

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        user_ratings = self.ratings[self.ratings['user_id'] == user_id]
        # take one random book from the user's ratings
        # user_rating = user_ratings.sample(1)
        # book = self.books[self.books['Name'] == user_rating['Name'].values[0]]
        user_books = user_ratings.merge(self.books, on='name')
        user_books['embed'] = (user_books['authors'] +
                               str(user_books['avg_rating'].values) +
                               str(user_books['publish_year'].values) +
                               user_books['desc'])
        user_books['embed'] = user_books['embed'].replace("\n", " ")
        user_books.drop(labels=['user_id', 'name', 'authors', 'avg_rating', 'publish_year', 'desc'],
                        axis=1, inplace=True)
        e = self.tokenizer(
            user_books['embed'].values.tolist(),
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'book_embed': e['input_ids'],
            'book_attention_mask': e['attention_mask'],
            'book_id': user_books['book_id'].values,
            'rating': torch.transpose(torch.from_numpy(user_books['rating'].values), 0, -1).float()
        }

    # def book_name2id(self, name):
    #     book_id = self.books[self.books['Name'] == name]['Id'].values
    #     return book_id[0] if len(book_id) > 0 else None

    # def get_book(self, b_idx):
    #     # print(self.books['Id'].head(5))
    #     book = self.books.loc[self.books['Id'] == b_idx]
    #     e = (book['Authors'].values[0] + ' ' +
    #          book['Year'].values[0] + ' ' +
    #          book['Rating'].values[0] + ' ' +
    #          book['PublishYear'].values[0] + ' ' +
    #          book['Description'].values[0])
    #     e = e.replace("\n", " ")

    # def get_rating(self, u_idx, b_idx=None):
    #     data = self.ratings
    #     user = data[data['ID'] == u_idx]
    #
    #     user_embed = torch.zeros(512)
    #     # print(user.columns)
    #     for idx, row in user.iterrows():
    #         # print(row['Name'])
    #         # break
    #         book_id = self.book_name2id(row['Name'])
    #         if book_id is None:
    #             continue
    #         if b_idx is not None and book_id == b_idx:
    #             continue
    #         book_embed = self.get_book(book_id)['input_ids']
    #         # print(f'{book_embed.shape=} {sm_user.shape=}')
    #         user_embed += book_embed
    #
    #         print(f'{book_embed.sum()} {user_embed.sum()}')
    #         assert False
    #     user_embed /= user.shape[0]
    #
    #     return user_embed

