from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data_tensor, target_tensor, index):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.index = index

    def __getitem__(self, idx):
        return self.data_tensor[self.index[idx]], self.target_tensor[self.index[idx]]

    def __len__(self):
        return len(self.index)