from torch.utils.data import Dataset

class MyDataset(Dataset):

    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, idx):
        return self.data_tensor[idx], self.target_tensor[idx]

    def __len__(self):
        return len(self.data_tensor)