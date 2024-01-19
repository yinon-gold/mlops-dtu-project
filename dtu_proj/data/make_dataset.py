import pandas as pd
import glob
import os


data_dir = 'data'  # TODO move to config
print(os.getcwd())
print(os.listdir(data_dir))
raw_data_dir = os.path.join(data_dir, 'raw')
processed_data_dir = os.path.join(data_dir, 'processed')

# Book dataset columns to keep
cols = ['Id', 'Name', 'Description', 'Authors', 'PublishYear', 'Rating']


def create_book_dataset():
    if os.path.exists(os.path.join(processed_data_dir, 'books.csv')):
        print('Book Dataset already exists')
        return pd.read_csv(os.path.join(processed_data_dir, 'books.csv'))
    book_df_list = []
    for idx, file in enumerate(glob.glob(os.path.join(raw_data_dir, 'book*.csv'))):
        book_df = pd.read_csv(file)
        if 'Description' not in book_df.columns:
            continue
        book_df = book_df.dropna(subset=cols)
        book_df_list.append(book_df)
    books = pd.concat(book_df_list, ignore_index=True)

    cols_to_drop = filter(lambda c: c not in cols, books.columns)
    books.drop(columns=cols_to_drop, inplace=True)

    books.rename(columns={
                    'Id': 'book_id',
                    'Name': 'name',
                    'Rating': 'avg_rating',
                    'Authors': 'authors',
                    'PublishYear': 'publish_year',
                    'Description': 'desc'},
                 inplace=True)

    latest_release = books.groupby('name')['publish_year'].transform("max") == books['publish_year']
    books = books[latest_release]

    highest_rating_latest = books.groupby('name')['avg_rating'].transform("max") == books['avg_rating']
    books = books[highest_rating_latest]

    books = books.groupby('name').first().reset_index()

    if not os.path.exists(processed_data_dir):
        os.mkdir(processed_data_dir)
    books.to_csv(os.path.join(processed_data_dir, 'books.csv'), index=False)

    print('Book Dataset Done')
    return books


def create_rating_dataset(books_df):
    if os.path.exists(os.path.join(processed_data_dir, 'ratings.csv')):
        print('Rating Dataset already exists')
        return
    rating_df_list = []
    for idx, file in enumerate(glob.glob(os.path.join(raw_data_dir, 'user_rating*.csv'))):
        rating_df = pd.read_csv(file)
        rating_df_list.append(rating_df)
    ratings = pd.concat(rating_df_list, ignore_index=True)
    rating_mapping = {'it was amazing': 5, 'really liked it': 4, 'liked it': 3, 'it was ok': 2,
                      'did not like it': 1, "This user doesn't have any rating": 0}
    ratings['Rating'] = ratings['Rating'].apply(lambda r: rating_mapping[r])
    ratings = ratings[ratings['Rating'] != 0]

    ratings = ratings[ratings['Name'].isin(books_df['name'])]

    ratings.rename(columns={
                    'ID': 'user_id',
                    'Name': 'name',
                    'Rating': 'rating'},
                    inplace=True)

    if not os.path.exists(processed_data_dir):
        os.mkdir(processed_data_dir)
    ratings.to_csv(os.path.join(processed_data_dir, 'ratings.csv'), index=False)
    print('Rating Dataset Done')


if __name__ == '__main__':
    books = create_book_dataset()
    create_rating_dataset(books)
    print('Done creating datasets')
