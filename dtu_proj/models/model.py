import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class BERTClassifier(nn.Module):
    def __init__(self, device, num_classes=1, checkpoint="prajjwal1/bert-tiny", freeze_bert=False):
        super(BERTClassifier, self).__init__()
        # Instantiating BERT-based model object
        self.bert = AutoModel.from_pretrained(checkpoint)
        self.bert.config.problem_type = 'regression'
        # freeze bert layers
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        # Defining layers like dropout and linear
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, sample):
        input_ids = sample['book_embed']
        attention_mask = sample['book_attention_mask']
        input_ids = input_ids
        attention_mask = attention_mask

        output = torch.zeros((input_ids.shape[1], self.bert.config.hidden_size)).to(input_ids.device)
        # Feeding the input to BERT-based model to obtain contextualized representations
        for i in range(input_ids.shape[1]):
            outputs = self.bert(input_ids=input_ids[:, i, :], attention_mask=attention_mask[:, i, :])
            # Extracting the representations of [CLS] head
            last_hidden_state_cls = outputs.pooler_output  # shape = (B, ratings, 128)
            output[i] = last_hidden_state_cls
        # outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = self.dropout(output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        logits = self.sigmoid(x)
        logits = torch.transpose(logits, 0, 1)
        return 4 * logits + 1
