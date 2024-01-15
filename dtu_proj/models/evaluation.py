    
if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    import torch 
    import pandas as pd
    from model import RecommenderNet
    from torch.utils.data import DataLoader
    from util.ratingdataset import RatingDataset

    # Load the data
    data = pd.read_csv('data/processed/clean.csv')
    # split the data
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

    
    trained_model = RecommenderNet(len(data['ID'].unique()) * 50, len(data['Book_Id'].unique()) * 50)
    # Load the model
    trained_model.load_state_dict(torch.load("models/model.pt"))


    dataset = RatingDataset(test_data)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
    # Make predictions on the test data
    for user, book, rating in dataloader:
        with torch.no_grad():
            trained_model.eval()
            #print(book)
            #print(torch.max(book))
            
            #print(torch.max(book))
            predictions = trained_model(torch.stack((user, book), dim=1))

        #  Calculate the loss
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(predictions.squeeze(1), rating)

    print(f'Test Loss: {loss.item()}')