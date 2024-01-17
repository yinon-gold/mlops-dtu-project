import torch
import torch.nn as nn
import torch.nn.functional as F

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
