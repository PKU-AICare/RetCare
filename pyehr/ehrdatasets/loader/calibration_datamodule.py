import os

import lightning as L
import pandas as pd
import torch
import torch.utils.data as data


class CalibDataset(data.Dataset):
    def __init__(self, data_path, mode='train'):
        super().__init__()
        self.data = pd.read_pickle(os.path.join(data_path, f'{mode}_x.pkl'))
        self.label = pd.read_pickle(os.path.join(data_path, f'{mode}_y.pkl'))
        self.pid = pd.read_pickle(os.path.join(data_path, f'{mode}_pid.pkl'))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.pid[index]


class CalibDataModule(L.LightningDataModule):
    def __init__(self, data_path, batch_size=32):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size

        self.train_dataset = CalibDataset(self.data_path, mode="train")
        self.test_dataset = CalibDataset(self.data_path, mode='test')

    # def setup(self, stage: str):
    #     if stage=="fit":
    #         self.train_dataset = EhrDataset(self.data_path, mode="train")
    #         self.val_dataset = EhrDataset(self.data_path, mode='val')
    #     if stage=="test":
    #         self.test_dataset = EhrDataset(self.data_path, mode='test')

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True , collate_fn=self.pad_collate, num_workers=8)

    def test_dataloader(self):
        return data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False , collate_fn=self.pad_collate, num_workers=8)

    def pad_collate(self, batch):
        x, y, pid = zip(*batch)
        x = torch.tensor(x).unsqueeze(1)
        y = torch.tensor(y)
        return x.float(), y.float(), pid
