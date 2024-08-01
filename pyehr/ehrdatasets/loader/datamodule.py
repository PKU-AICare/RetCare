import os

import lightning as L
import pandas as pd
import torch
import torch.utils.data as data


class EhrDataset(data.Dataset):
    def __init__(self, data_path, mode='train'):
        super().__init__()
        self.data = pd.read_pickle(os.path.join(data_path,f'{mode}_x.pkl'))
        self.label = pd.read_pickle(os.path.join(data_path,f'{mode}_y.pkl'))
        self.pid = pd.read_pickle(os.path.join(data_path,f'{mode}_pid.pkl'))

    def __len__(self):
        return len(self.label) # number of patients

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.pid[index]


class EhrDataModule(L.LightningDataModule):
    def __init__(self, data_path, batch_size=32):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size

        self.train_dataset = EhrDataset(self.data_path, mode="train")
        self.val_dataset = EhrDataset(self.data_path, mode='val')
        self.test_dataset = EhrDataset(self.data_path, mode='test')

    # def setup(self, stage: str):
    #     if stage=="fit":
    #         self.train_dataset = EhrDataset(self.data_path, mode="train")
    #         self.val_dataset = EhrDataset(self.data_path, mode='val')
    #     if stage=="test":
    #         self.test_dataset = EhrDataset(self.data_path, mode='test')

    def train_dataloader(self):
        return data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True , collate_fn=self.pad_collate, num_workers=8)

    def val_dataloader(self):
        return data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False , collate_fn=self.pad_collate, num_workers=8)

    def test_dataloader(self):
        return data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False , collate_fn=self.pad_collate, num_workers=8)

    def pad_collate(self, batch):
        xx, yy, pid = zip(*batch)
        lens = torch.as_tensor([len(x) for x in xx])
        # convert to tensor
        xx = [torch.tensor(x) for x in xx]
        yy = [torch.tensor(y) for y in yy]
        xx_pad = torch.nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=0)
        yy_pad = torch.nn.utils.rnn.pad_sequence(yy, batch_first=True, padding_value=0)
        return xx_pad.float(), yy_pad.float(), lens, pid
