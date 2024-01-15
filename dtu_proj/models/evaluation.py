    
if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    import torch 
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from model import RecommenderNet
    from torch.utils.data import DataLoader
    from util.ratingdataset import RatingDataset
    from sklearn.metrics import mean_absolute_error, mean_squared_error


    # Load the data
    data = pd.read_csv('data/processed/clean.csv')
    # split the data
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

    
    trained_model = RecommenderNet(len(data['ID'].unique()) * 50, len(data['Book_Id'].unique()) * 50)
    # Load the model
    trained_model.load_state_dict(torch.load("models/model.pt"))


    dataset = RatingDataset(test_data)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    actual, predicted = [], []
    # Make predictions on the test data
    for user, book, rating in dataloader:
        with torch.no_grad():
            trained_model.eval()
            predictions = trained_model(torch.stack((user, book), dim=1))
            actual.extend(rating.tolist())
            predicted.extend(predictions.squeeze(1).tolist())

        #  Calculate the loss
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(predictions.squeeze(1), rating)

    print(f'Test Loss: {loss.item()}')
    print(f'MAE: {mean_absolute_error(actual, predicted)}')
    print(f'RMSE: {np.sqrt(mean_squared_error(actual, predicted))}')

    # Plot actual vs predicted
    plt.figure(figsize=(8, 8))
    plt.scatter(actual, predicted, alpha=0.5)
    plt.xlabel('Actual Ratings')
    plt.ylabel('Predicted Ratings')
    plt.title('Actual vs Predicted Ratings')
    plt.show()