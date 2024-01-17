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


class RatingDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len

        df_list = []
        for idx, file in enumerate(glob.glob(data_dir + 'book*.csv')):
            df = pd.read_csv(file)
            # print(df.shape)
            if 'Description' not in df.columns:
                continue
            df = df.dropna(subset=['Description'])
            df_list.append(df)
        self.data = pd.concat(df_list, ignore_index=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data.iloc[index]
        e = data['Description']
        e = e.replace("\n", " ")
        encoded_input = self.tokenizer.encode_plus(
            e,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_attention_mask=True,
            return_tensors='pt'
        )
        # print(encoded_input['input_ids'].shape)
        return {
            'input_ids': encoded_input['input_ids'].flatten(),  # B * 512 * 128 (L=hidden_size)
            'attention_mask': encoded_input['attention_mask'].flatten(),
            'target': torch.tensor(self.data.iloc[index]['Rating'], dtype=torch.float)
        }


dataset = RatingDataset(data_dir='../data/raw/', tokenizer=bert_tokenizer)

