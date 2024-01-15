import torch
import torch.nn as nn
import torch.nn.functional as F

class RecommenderNet(torch.nn.Module):
    """ Basic neural network class. 
    
    Args:
        in_features: number of input features
        out_features: number of output features
    
    """
    def __init__(self, n_users, n_books, n_names, n_factors=50, n_hidden=124):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, n_factors)
        self.book_emb = nn.Embedding(n_books, n_factors)
        self.name_emb = nn.Embedding(n_names, n_factors)
        self.hl1 = nn.Linear(n_factors*3, n_hidden)
        self.hl2 = nn.Linear(n_hidden, 1)
        self.drop = nn.Dropout(0.1)
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
        names = self.name_emb(x[:,2])

        x = F.relu(torch.cat([users, books, names], dim=1))
        x = self.drop(x)
        x = F.relu(self.hl1(x))
        x = self.hl2(x)
        # Below restricts the output to between 1 and 5
        x = self.sigmoid(x) * 4 + 1
        return x