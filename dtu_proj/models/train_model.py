if __name__ == '__main__':
    import torch
    import pandas as pd
    from torch.utils.data import DataLoader
    from torch.nn.utils.rnn import pad_sequence
    from torch.nn import MSELoss
    from model import RecommenderNet
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm
    from util.ratingdataset import RatingDataset
    import ast


    def collate_fn(batch):
        users, books, names, ratings = zip(*batch)

        users = torch.tensor(users)
        books = torch.tensor(books)
        names = pad_sequence([torch.tensor(name) for name in names], batch_first=True)
        ratings = torch.tensor(ratings)

    
        # flatten the newly padded names
        names = names.flatten()

        # pad the tensors together
        tensors = [users, books, names, ratings]
        padded_tensor = pad_sequence(tensors, batch_first=True)
        return padded_tensor
    
    # previous learning rate was 0.01
    def train(model, user_rating, epochs=5, lr=0.001, batch_size=64):

        # Create a DataLoader from the DataFrame
        dataset = RatingDataset(user_rating)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        # Use mean squared error loss
        criterion = MSELoss()

        # Use Adam optimizer
        optimizer = torch.optim.Adam(model.parameters())

        for epoch in range(epochs):
            progress_bar = tqdm(dataloader, desc='Epoch {:03d}'.format(epoch + 1), leave=False, disable=False)

            for user, book, name, rating in dataloader:
                # Forward pass, have to stack them like so to do forward pass

                #print(user.size())
                #print(book.size())
                #print(name.size())
                
                #print(data)
                #print(torch.max(name))
                #print(torch.max(book))
                #print(torch.max(user))

                outputs = model(torch.stack((user, book, name),dim=1))
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
    # Convert book description to integer value
    data['Name'] = data['Name'].apply(lambda x: [int(i) for i in ast.literal_eval(x)])
    #print(data['Name'])

    # split the data
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)


    n_factor = 100
    # get amount of unique books and users
    n_books = len(data['Book_Id'].unique()) * n_factor
    n_users = len(data['ID'].unique()) * n_factor
    flat_list = [item for sublist in data['Name'] for item in sublist]
    n_names = len(set(flat_list)) * n_factor
    #print(n_names)

    # load previously calculated embeddings
    model = RecommenderNet(n_users, n_books, n_names, n_factors=n_factor)

    #print('hi')
    # train the model
    trained_model = train(model, train_data)

    torch.save(trained_model.state_dict(), "models/model.pt")