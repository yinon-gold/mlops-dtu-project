if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import pandas as pd
    import glob
    from sklearn.model_selection import train_test_split
    from model import RecommenderNet
    
    # load the data
    data = pd.read_csv('data/processed/clean.csv')
    # split the data
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False, random_state=42)

    # get amount of unique books and users
    n_books = train_data['Book_Id'].unique().sum()
    n_users = train_data['ID'].unique().sum()

    # Get the data and process it, use gpu if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Previous value n_factors=50
    model = RecommenderNet(n_users, n_books, n_factors=2).to(device)
    # Save the model as model.pt
    torch.save(model.state_dict(), "models/embeddings.pt")