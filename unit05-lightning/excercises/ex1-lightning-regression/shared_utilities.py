import lightning as L
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
import torchmetrics
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split
import os

class PyTorchMLPRegressor(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.all_layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_features, 50),
            torch.nn.ReLU(),
            # 2nd hidden layer
            torch.nn.Linear(50, 25),
            torch.nn.ReLU(),
            # outputs layer
            torch.nn.Linear(25, 1),
        )

    def forward(self, x):
        outputs = self.all_layers(x)
        outputs = outputs.flatten()
        return outputs


class LightningModel(L.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()

        self.learning_rate = learning_rate
        self.model = model

        self.train_mse = torchmetrics.MeanSquaredError(squared=True)
        self.val_mse = torchmetrics.MeanSquaredError(squared=True)
        self.test_mse = torchmetrics.MeanSquaredError(squared=True)

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        features, true_values = batch
        outputs = self(features)

        loss = F.mse_loss(outputs, true_values, reduction='mean')
        return loss, true_values, outputs

    def training_step(self, batch, batch_idx):
        loss, true_values, outputs = self._shared_step(batch)

        self.log("train_loss", loss)
        self.train_mse(outputs, true_values)
        self.log(
            "train_mse", self.train_mse, prog_bar=True, on_epoch=True, on_step=False
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, true_values, outputs = self._shared_step(batch)

        self.log("val_loss", loss, prog_bar=True)
        self.val_mse(outputs, true_values)
        self.log("val_mse", self.val_mse, prog_bar=True)

    def test_step(self, batch, batch_idx):
        _, true_values, outputs = self._shared_step(batch)
        self.test_mse(outputs, true_values)
        self.log("test_mse", self.test_mse)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        return optimizer

class AmesHousingDataset(Dataset):
    def __init__(self, csv_path):
        columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area',
                   'Central Air', 'Total Bsmt SF', 'SalePrice']

        df = pd.read_csv(csv_path,
                         sep='\t',
                         usecols=columns)

        #df['Central Air'] = df['Central Air'].map({'N': 0, 'Y': 1})
        df = df.dropna(axis=0)

        X = df[['Overall Qual',
                'Gr Liv Area',
                'Total Bsmt SF']].values
        y = df['SalePrice'].values

        sc_x = StandardScaler()
        sc_y = StandardScaler()
        X_std = sc_x.fit_transform(X)
        y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()

        self.x = torch.tensor(X_std, dtype=torch.float)
        self.y = torch.tensor(y_std, dtype=torch.float).flatten()

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.y.shape[0]


class AmesHousingDataModule(L.LightningDataModule):
    def __init__(self,
                 csv_path='http://jse.amstat.org/v19n3/decock/AmesHousing.txt',
                 batch_size=32):
        super().__init__()
        self.csv_path = csv_path
        self.batch_size = batch_size

    def prepare_data(self):
        if not os.path.exists('AmesHousing.txt'):
            df = pd.read_csv(self.csv_path)
            df.to_csv('AmesHousing.txt', index=False)

    def setup(self, stage: str):
        all_data = AmesHousingDataset(csv_path='AmesHousing.txt')
        temp, self.val = random_split(all_data, [2500, 429], 
                                      torch.Generator().manual_seed(1))
        self.train, self.test = random_split(temp, [2000, 500],
                                             torch.Generator().manual_seed(1))

    def train_dataloader(self):
        return DataLoader(
            self.train, batch_size=self.batch_size,
            shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False)
