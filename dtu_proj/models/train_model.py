if __name__ == '__main__':
    import torch
    import pandas as pd


    from torch.utils.data import Dataset, DataLoader
    from torch.optim import Adam
    from torch.nn import MSELoss
    from model import RecommenderNet
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
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
    def train(model, train_data, test_data, epochs=50, lr=0.01, batch_size=256, step_size=5, gamma=0.1):

        # Create a DataLoader from the DataFrame
        train_dataset = RatingDataset(train_data)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        test_dataset = RatingDataset(test_data)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Use mean squared error loss
        criterion = MSELoss()

        # Use Adam optimizer
        optimizer = Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        for epoch in range(epochs):
            progress_bar = tqdm(train_dataloader, desc='Epoch {:03d}'.format(epoch + 1), leave=False, disable=False)
            
            for user, book, rating in train_dataloader:
                
                outputs = model(torch.stack((user, book),dim=1))
                loss = criterion(outputs, rating.float().unsqueeze(1))

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                progress_bar.set_postfix({'training_loss': '{:.3f}'.format(loss.item())})
                progress_bar.update()
            # Validation loss
            model.eval()
            total_test_loss = 0
            with torch.no_grad():
                for user, book, rating in test_dataloader:
                    test_outputs = model(torch.stack((user, book), dim=1))
                    test_loss = criterion(test_outputs, rating.float().unsqueeze(1))
                    total_test_loss += test_loss.item()
            avg_test_loss = total_test_loss / len(test_dataloader)
            model.train()

            scheduler.step()
            print(f'Epoch {epoch+1}/{epochs}, Training Loss: {loss.item()}, Validation Loss: {avg_test_loss}, Learning Rate: {scheduler.get_last_lr()[0]}')
        
        return model

# load the data
data = pd.read_csv('data/processed/clean.csv')
# split the data
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False, random_state=42)

n_factor=128

# get amount of unique books and users
n_users = data['ID'].max()+100
n_books = data['Book_Id'].max()+100

# load previously calculated embeddings
model = RecommenderNet(n_users, n_books, n_factors=n_factor)
#model.load_state_dict(torch.load("models/embeddings.pt"))


print(model)

# train the model
trained_model = train(model, train_data, test_data)

torch.save(trained_model.state_dict(), "models/model.pt")
