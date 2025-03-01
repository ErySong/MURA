from typing import Any, Dict, Optional, Tuple

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

from .data_loader import (
    Dataset_ETT_hour,
    Dataset_ETT_minute,
    Dataset_Custom,
    Dataset_Pred,
    Dataset_Solar,
    Dataset_PEMS,
)

class DataModule(LightningDataModule):
    def __init__(
            self,
            data_params: Dict[str, Any],
            batch_size=64,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.data = data_params.loader
        self.embed = data_params.embed
        self.freq = data_params.freq
        self.root_path = data_params.root_path
        self.data_path = data_params.data_path
        self.features = data_params.features
        self.target = data_params.target
        self.seq_len = data_params.seq_len
        self.label_len = 0
        self.pred_len = data_params.pred_len

    def data_provider(self, flag):
        data_dict = {
            "ETTh1": Dataset_ETT_hour,
            "ETTh2": Dataset_ETT_hour,
            "ETTm1": Dataset_ETT_minute,
            "ETTm2": Dataset_ETT_minute,
            "custom": Dataset_Custom,
            "Solar": Dataset_Solar,
            "PEMS": Dataset_PEMS,
        }
        Data = data_dict[self.data]
        timeenc = 0 if self.embed != "timeF" else 1

        if flag == "test":
            shuffle_flag = False
            drop_last = False
            batch_size = self.batch_size
            freq = self.freq
        elif flag == "pred":
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            freq = self.freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = self.batch_size
            freq = self.freq

        if flag == "val":
            drop_last = False
        data_set = Data(
            root_path=self.root_path,
            data_path=self.data_path,
            flag=flag,
            size=[self.seq_len, self.label_len, self.pred_len],
            features=self.features,
            target=self.target,
            timeenc=timeenc,
            freq=freq,
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=8,
            drop_last=drop_last,
        )
        return data_loader

    def train_dataloader(self):
        return self.data_provider("train")

    def val_dataloader(self):
        return self.data_provider("val")

    def test_dataloader(self):
        return self.data_provider("test")
