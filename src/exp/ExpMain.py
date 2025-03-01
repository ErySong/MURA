from models import MURA
from .ExpBase import LitModule
import torch
import torch.nn as nn
from . import RevIN

model_dict = {"MURA": MURA}


class ExpModel(LitModule):
    def __init__(self, net_params):
        super().__init__(net_params, BackboneModel=model_dict[net_params.name].Model)

    def model_step(self, batch, mode="train"):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        if self.norm_type == "seq":
            seq_mean = torch.mean(batch_x, dim=1, keepdim=True)
            seq_var = torch.var(batch_x, dim=1, keepdim=True) + 1e-5
            batch_x = (batch_x - seq_mean) / torch.sqrt(seq_var)
            output = self.forward(batch_x)
            output = output * torch.sqrt(seq_var) + seq_mean
        elif self.norm_type == "revin":
            revin_layer = RevIN.RevIN(self.enc_in).to(batch_x.device)
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
        batch_y_freq = torch.fft.rfft(batch_y, dim=1)
        output_freq = torch.fft.rfft(output, dim=1)
        output = torch.fft.irfft(output_freq, dim=1)
        loss = (output_freq - batch_y_freq).abs().mean()
        if mode == "val":
            loss = self.criterion(output, batch_y)

        return output, batch_y, loss
