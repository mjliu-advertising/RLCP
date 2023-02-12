import torch.utils.data.dataset as Dataset

class libsvm_dataset(Dataset.Dataset):
    def __init__(self, Data, label):
        super(libsvm_dataset, self).__init__()
        self.Data = Data
        self.label = label

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, item):
        data = self.Data[item]
        label = self.label[item]

        return data, label