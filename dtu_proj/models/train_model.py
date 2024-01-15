if __name__ == '__main__':
    import torch
    import pandas as pd
    from torch.utils.data import DataLoader
    from torch.optim import Adam
    from torch.nn import MSELoss
    from model import RecommenderNet
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm
    from util.ratingdataset import RatingDataset

    # previous learning rate was 0.01
    def train(model, user_rating, epochs=5, lr=0.01, batch_size=64):
        # Create a DataLoader from the DataFrame
        dataset = RatingDataset(user_rating)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Use mean squared error loss
        criterion = MSELoss()

        # Use Adam optimizer
        optimizer = torch.optim.Adam(model.parameters())

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
                progress_bar.update()

            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
        return model


    # load the data
    data = pd.read_csv('data/processed/clean.csv')
    # split the data
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)


    # get amount of unique books and users
    n_books = len(data['Book_Id'].unique()) * 50
    n_users = len(data['ID'].unique()) * 50
    

    # load previously calculated embeddings
    model = RecommenderNet(n_users, n_books, n_factors=50)

    # train the model
    trained_model = train(model, test_data)

    torch.save(trained_model.state_dict(), "models/model.pt")