import torch
import torch.nn as nn

class RecommenderNet(torch.nn.Module):
    """ Basic neural network class. 
    
    Args:
        in_features: number of input features
        out_features: number of output features
    
    """
    def __init__(self, n_users, n_books, n_factors):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, n_factors)
        self.book_emb = nn.Embedding(n_books, n_factors)
        self.drop = nn.Dropout(0.05)
        self.fc = nn.Linear(n_factors*2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: input tensor expected to be of shape [N,in_features]

        Returns:
            Output tensor with shape [N,out_features]

        """
        users = self.user_emb(x[:,0])
        books = self.book_emb(x[:,1])
        x = torch.cat([users, books], dim=1)
        x = self.drop(x)
        x = self.fc(x)
        return x