if __name__ == '__main__':
    import torch
    import pandas as pd
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import Adam
    from torch.nn import MSELoss
    from model import RecommenderNet
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm

    class RatingDataset(Dataset):
        def __init__(self, user_rating):
            self.user_rating = user_rating

        def __len__(self):
            return len(self.user_rating)

        def __getitem__(self, idx):
            user = torch.tensor(self.user_rating.iloc[idx]['ID'])
            book = torch.tensor(self.user_rating.iloc[idx]['Book_Id'])
            rating = torch.tensor(self.user_rating.iloc[idx]['Rating'])
            return user, book, rating
            
    # previous learning rate was 0.01
    def train(model, user_rating, epochs=5, lr=0.1, batch_size=32):
        # Create a DataLoader from the DataFrame
        dataset = RatingDataset(user_rating)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Use mean squared error loss
        criterion = MSELoss()

        # Use Adam optimizer
        optimizer = Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            progress_bar = tqdm(dataloader, desc='Epoch {:03d}'.format(epoch + 1), leave=False, disable=False)
            for user, book, rating in dataloader:
                
                # Forward pass, have to stack them like so to do forward pass
                outputs = model(torch.stack((user, book), dim=1))
                loss = criterion(outputs, rating.float().unsqueeze(1))

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item())})

            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')


# load the data
data = pd.read_csv('data/processed/clean.csv')
# split the data
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False, random_state=42)

# get amount of unique books and users
n_books = train_data['Book_Id'].unique().sum()
n_users = train_data['ID'].unique().sum()

print(n_books)
print(n_users)

# load previously calculated embeddings
model = RecommenderNet(n_users, n_books, n_factors=2)
moedel= model.load_state_dict(torch.load("models/embeddings.pt"))

# train the model
train(model, train_data)