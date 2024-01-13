if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import pandas as pd
    import glob

    from models.model import RecommenderNet



    ##Read in the data
    book_rating = pd.DataFrame()
    for file in glob.glob("../data/raw/book*.csv"):
        df = pd.read_csv(file)
    # discard empty
    if book_rating.empty:
        book_rating = df
    else:
        df = pd.concat([book_rating, df], ignore_index=True)

    user_rating_temp = pd.DataFrame()
    for file in glob.glob("../data/raw/user_rating*.csv"):
        df = pd.read_csv(file)
    if user_rating_temp.empty:
        user_rating_temp = df
    else:
        df = pd.concat([user_rating_temp, df], ignore_index=True)

    #Import and merge revelvant data
    book_map = user_rating_temp[['Name']]
    book_map.drop_duplicates(subset=['Name'],keep='first',inplace=True)
    book_map['Book_Id']=book_map.index.values
    user_rating_temp = pd.merge(user_rating_temp,book_map, on='Name', how='left')
    user_rating = user_rating_temp[user_rating_temp['Name']!='Rating'] ##Dropping users who have not rated any books
    user_rating.head()

    # Map reviews to numerical data
    rating_mapping = {'it was amazing': 5, 'really liked it': 4, 'liked it': 3, 'it was ok': 2, 'did not like it': 1}
    user_rating['Rating'] = user_rating['Rating'].map(rating_mapping)

    # get amount of unique books and users
    n_books = user_rating['Book_Id'].unique().sum()
    n_users = user_rating['ID'].unique().sum()

    # Get the data and process it, use gpu if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Previous value n_factors=50
    model = RecommenderNet(n_users, n_books, n_factors=50).to(device)
    # Save the model as model.pt
    torch.save(model.state_dict(), "model.pt")