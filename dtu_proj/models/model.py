import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import zip_longest

class RecommenderNet(torch.nn.Module):
    """ Basic neural network class. 
    
    Args:
        in_features: number of input features
        out_features: number of output features
    
    """

    def __init__(self, n_users, n_books, n_factors=50, hidden=[100,200,300], dropouts=[0.25,0.5]):
        super().__init__()


        self.drop = nn.Dropout(0.02)
        self.sigmoid = nn.Sigmoid()
        hidden = get_list(hidden)
        dropouts = get_list(dropouts)
        n_last = hidden[-1]

        def gen_layers(n_in):
            nonlocal hidden, dropouts
            assert len(dropouts) <= len(hidden)

            
            for n_out, rate in zip_longest(hidden, dropouts):
                yield nn.Linear(n_in, n_out)
                yield nn.ReLU()
                if rate is not None and rate > 0.:
                    yield nn.Dropout(rate)
                n_in = n_out



        self.u = nn.Embedding(n_users, n_factors)
        self.m = nn.Embedding(n_books, n_factors)
        self.drop = nn.Dropout(0.2)
        self.hidden = nn.Sequential(*list(gen_layers(n_factors * 2)))
        self.fc = nn.Linear(n_last, 1)
        self._init()
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: input tensor expected to be of shape [N,in_features]

        Returns:
            Output tensor with shape [N,out_features]

        """
        users = self.u(x[:,0])
        books = self.m(x[:,1])
        x = torch.cat([users, books], dim=1)
        x = self.drop(x)
        x = self.hidden(x)
        x = self.sigmoid(self.fc(x)) * 4 + 1
        return x

    def _init(self):
        
        def init(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
                
        self.u.weight.data.uniform_(-0.05, 0.05)
        self.m.weight.data.uniform_(-0.05, 0.05)
        self.hidden.apply(init)
        init(self.fc)
    
def get_list(n):
    if isinstance(n, (int, float)):
        return [n]
    elif hasattr(n, '__iter__'):
        return list(n)
    raise TypeError('layers configuraiton should be a single number or a list of numbers')