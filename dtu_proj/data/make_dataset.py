if __name__ == '__main__':
    import torch.nn as nn
    import pandas as pd
    import glob
    from pathlib import Path  
    from keras.preprocessing.text import Tokenizer

    ##Read in the data
    book_rating = pd.DataFrame()
    for file in glob.glob("data/raw/book*.csv"):
        df = pd.read_csv(file)

    # discard empty
        if book_rating.empty:
            book_rating = df
        else:
            df = pd.concat([book_rating, df], ignore_index=True)

    user_rating_temp = pd.DataFrame()
    for file in glob.glob("data/raw/user_rating*.csv"):
        df = pd.read_csv(file)
        if user_rating_temp.empty:
            user_rating_temp = df
        else:
            user_rating_temp = pd.concat([user_rating_temp, df], ignore_index=True)


    #Import and merge revelvant data
    book_map = user_rating_temp[['Name']]
    book_map.drop_duplicates(subset=['Name'],keep='first',inplace=True)
    book_map['Book_Id']=book_map.index.values
    user_rating_temp = pd.merge(user_rating_temp,book_map, on='Name', how='left')
    ##Dropping users who have not rated any books
    user_rating = user_rating_temp[user_rating_temp['Name']!='Rating'] 

    # Map reviews to numerical data
    rating_mapping = {'it was amazing': 5, 'really liked it': 4, 'liked it': 3, 'it was ok': 2, 'did not like it': 1}
    user_rating['Rating'] = user_rating['Rating'].map(rating_mapping)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(user_rating['Name'])
    user_rating['Name'] = tokenizer.texts_to_sequences(user_rating['Name'])

    # drop the description
    user_rating = user_rating.drop('Name', axis=1)

    # Save processed file 
    filepath = Path('data/processed/clean.csv')  
    user_rating.to_csv(filepath)