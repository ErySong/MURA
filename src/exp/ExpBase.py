import torch
from lightning.pytorch import LightningModule
from torch import optim, Tensor
import yaml, os, math
from utils.metrics import metric
from utils.tools import test_params_flop, visual_channels, _get_profile
from . import RevIN
import torch.nn.functional as F
import numpy as np


class LitModule(LightningModule):
    def __init__(self, net_params, BackboneModel):
        super().__init__()
        self.save_hyperparameters()
        self.lr = net_params.lr  # 学习率
        # 时序固定参数
        self.net_params = net_params
        self.channels = net_params.channels
        self.seq_len = net_params.seq_len
        self.pred_len = net_params.pred_len
        self.features = net_params.features
        self.norm_type = net_params.norm_type
        self.visual_cs = net_params.visual_cs
        # 非固定参数
        self.Model = BackboneModel
        self.model_ = None
        self.build_model()
        self.criterion = F.mse_loss
        self.train_epoch_outputs = []
        self.val_epoch_outputs = []
        self.test_epoch_outputs = []

    def build_model(self):
        self.model_ = self.Model(self.net_params).float()

    def forward(self, batch_x):
        return self.model_(batch_x)

    def model_step(self, batch, mode="train"):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        # norm
        if self.norm_type == "seq":
            seq_mean = torch.mean(batch_x, dim=1, keepdim=True)
            seq_var = torch.var(batch_x, dim=1, keepdim=True) + 1e-5
            batch_x = (batch_x - seq_mean) / torch.sqrt(seq_var)
            output = self.forward(batch_x)
            output = output * torch.sqrt(seq_var) + seq_mean
        elif self.norm_type == "revin":
            revin_layer = RevIN.RevIN(self.channels).to(batch_x.device)
            x_in = revin_layer(batch_x, "norm")
            x_out = self.forward(x_in)
            output = revin_layer(x_out, "denorm")
        else:
            output = self.forward(batch_x)
        # 判断什么类型的预测
        f_dim = -1 if self.features == "MS" else 0
        # 整理输出
        output = output[:, -self.pred_len :, f_dim:]
        batch_y = batch_y[:, -self.pred_len :, f_dim:]

        loss = self.criterion(output, batch_y)

        return output, batch_y, loss

    def training_step(self, batch, batch_idx) -> dict[str, Tensor]:
        _, _, loss = self.model_step(batch, "train")
        output = {"loss": loss}
        self.train_epoch_outputs.append(output)
        self.log(
            "train_step_loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,
        )
        return output

    def epoch_end(self, outputs, mode="train") -> None:
        loss_sum = 0
        for o in outputs:
            loss_sum += o["loss"]
        self.log(
            f"{mode}_loss",
            loss_sum / (len(outputs) if len(outputs) != 0 else 1),
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True,
        )

    def on_train_epoch_end(self) -> None:
        outputs = self.train_epoch_outputs
        self.epoch_end(outputs, mode="train")
        self.train_epoch_outputs = []

    def validation_step(self, batch, batch_idx) -> dict[str, Tensor]:
        _, _, loss = self.model_step(batch, "val")
        output = {"loss": loss}
        self.val_epoch_outputs.append(output)
        return output

    def on_validation_epoch_end(self) -> None:
        outputs = self.val_epoch_outputs
        self.epoch_end(outputs=outputs, mode="val")
        self.val_epoch_outputs = []

    def test_step(self, batch, batch_idx):
        y_hat, y, loss = self.model_step(batch, "test")
        output = {
            "y": y,
            "y_hat": y_hat,
        }
        self.test_epoch_outputs.append(output)
        if self.visual_cs != 0 and batch_idx % 3 == 0:
            folder_path = f"{self.logger.log_dir}/visual"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            visual_channels(
                history=batch[0][0].cpu().numpy(),
                true=y[0].cpu().numpy(),
                preds=y_hat[0].cpu().numpy(),
                visual_cs=self.visual_cs,
                name=f"{folder_path}/batch-{batch_idx}.png",
            )

        return output

    def on_test_epoch_end(self) -> None:
        outputs = self.test_epoch_outputs
        out_len = len(outputs)
        # 初始化这些指标的累加器
        metrics = {
            "MAE": 0,
            "MSE": 0,
            # "RMSE": 0,
            "MAPE": 0,
            # "MSPE": 0,
            # "RSE": 0,
        }
        # 假设 metric 函数返回各个度量的值
        ys = []
        y_hats = []
        for output in outputs:
            y = output["y"].detach().cpu().numpy()
            y_hat = output["y_hat"].detach().cpu().numpy()
            ys.append(y)
            y_hats.append(y_hat)
        ys = np.concatenate(ys, axis=0)
        y_hats = np.concatenate(y_hats, axis=0)
        updates = metric(y_hats, ys)
        # print(updates['mse'])
        for k, v in updates.items():
            if k.upper() in metrics.keys():
                metrics[k.upper()] += v

        metrics = {k: round(v.item(), 5) for k, v in metrics.items()}
        log_metrics = {f"hp_metric/{k}": v for k, v in metrics.items()}
        self.log("hp_metric", metrics["MSE"], prog_bar=False, sync_dist=True)
        self.log_dict(log_metrics, on_epoch=True, prog_bar=True, sync_dist=True)
        metrics["MACs"], metrics["params"] = _get_profile(
            self.model_,
            (
                1,
                self.seq_len,
                self.channels,
            ),
            device=self.device,
        )
        with open(f"{self.logger.log_dir}/metrics.yaml", "w") as f:
            yaml.dump(metrics, f)  # save metrics to yaml file

        self.test_epoch_outputs = []

    def configure_optimizers(self):
        optimizer = optim.Adam(self.trainer.model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.3, patience=1
        )
        return [optimizer], [{"scheduler": scheduler, "monitor": "val_loss"}]
