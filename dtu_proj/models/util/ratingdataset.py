from torch.utils.data import Dataset
import torch

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
            