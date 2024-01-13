if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import pandas as pd
    import glob

    from model import RecommenderNet
    
    user_rating = pd.read_csv('data/processed/clean.csv')

    # get amount of unique books and users
    n_books = user_rating['Book_Id'].unique().sum()
    n_users = user_rating['ID'].unique().sum()

    # Get the data and process it, use gpu if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Previous value n_factors=50
    model = RecommenderNet(n_users, n_books, n_factors=2).to(device)
    # Save the model as model.pt
    torch.save(model.state_dict(), "models/model.pt")