import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class BERTClassifier(nn.Module):
    def __init__(self, device, num_classes=1, checkpoint="prajjwal1/bert-tiny", freeze_bert=False):
        super(BERTClassifier, self).__init__()
        # Instantiating BERT-based model object
        self.bert = AutoModel.from_pretrained(checkpoint)#.to(device)
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


class RecommenderNet(torch.nn.Module):
    """ Basic neural network class. 
    
    Args:
        in_features: number of input features
        out_features: number of output features
    
    """

    def __init__(self, n_users, n_books, n_factors=50, n_hidden=40):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, n_factors)
        self.book_emb = nn.Embedding(n_books, n_factors)
        self.fc = nn.Linear(n_factors*2, n_hidden)
        self.hl1= nn.Linear(n_hidden, 5)
        self.hl2= nn.Linear(5, 1)
        self.drop = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: input tensor expected to be of shape [N,in_features]

        Returns:
            Output tensor with shape [N,out_features]

        """
        users = self.user_emb(x[:,0])
        books = self.book_emb(x[:,1])
        x = F.relu(torch.cat([users, books], dim=1))
        x = self.drop(x)
        x = F.relu(self.fc(x))
        x = F.relu(self.hl1(x))
        x = F.relu(self.hl2(x))
        #x = self.sigmoid(x) * 4 + 1
        return x
