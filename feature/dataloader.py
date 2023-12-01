import torch
from torch.utils.data import Dataset, DataLoader

class Custom_Dataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.data = list(dataset['data'])
        self.tokenizer = tokenizer
        self.label = list(dataset['label'])

    def __getitem__(self, ind):
        item = {
            key: torch.tensor(val[ind]) for key, val in self.tokenized_dataset.items()
        }
        item["label"] = torch.tensor(self.label[ind])
        return item

    def __len__(self):
        return len(self.label)
    
    def data_loader(self, max_len, batch_size, num_workers, shuffle=False):
        self.tokenizing(max_len)
        dataloader = DataLoader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
        )
        return dataloader
    
    def tokenizing(self, max_len):
        tokenized_dataset = self.tokenizer(
            self.data,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        self.tokenized_dataset = tokenized_dataset