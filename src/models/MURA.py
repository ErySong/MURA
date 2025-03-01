import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.channels

        self.stride = configs.stride
        self.d_model = configs.d_model
        #
        self.seg_num_x = int(self.seq_len / self.stride)
        self.seg_num_y = int(self.pred_len / self.stride)

        #
        self.len_ratio = (self.seq_len + self.pred_len) / self.seq_len

        self.trend_freq = int((self.seq_len / 2 + 1) * configs.trend_freq)
        #
        self.trend_model = nn.Linear(
            self.trend_freq, int(self.trend_freq * self.len_ratio)
        ).to(torch.cfloat)
        #
        self.res_model = nn.Sequential(
            nn.Linear(self.seg_num_x, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.seg_num_y),
        )
        print(f"all_freq:{int(self.seq_len / 2 + 1)}_trend_freq:{self.trend_freq}")

    def VSP(self, x, mode=0):
        if mode == 0:
            # x : B,L,C
            x = x.permute(0, 2, 1)  # (B,C,L)
            x = x.reshape(-1, self.seg_num_x, self.stride).permute(0, 2, 1)  # (B*C,s,n)
        elif mode == 1:
            # x : B*c,s,n
            x = x.permute(0, 2, 1)  # (Bc,n,s)
            x = x.reshape(-1, self.channels, self.pred_len).permute(0, 2, 1)  # (B,H,C)
        return x

    def forward(self, x):
        x_freq = torch.fft.rfft(x, dim=1)
        x_freq_trend = x_freq[:, : self.trend_freq, :]
        # 2. TFM
        xy_freq_trend = self.trend_model(x_freq_trend.permute(0, 2, 1)).permute(0, 2, 1)
        # 3. fill  [B,(L+H)/2+1,C]
        xy_freq_len = int((self.seq_len + self.pred_len) / 2 + 1)
        xy_freq = torch.zeros(
            [
                x.shape[0],
                xy_freq_len,
                self.channels,
            ],
            dtype=xy_freq_trend.dtype,
        ).to(xy_freq_trend.device)
        # 4
        if xy_freq_trend.size(1) > xy_freq_len:
            xy_freq = xy_freq_trend[:, :xy_freq_len, :]
        else:
            xy_freq[:, : xy_freq_trend.size(1), :] = xy_freq_trend
        xy_trend = torch.fft.irfft(xy_freq, dim=1)
        # split by [L:H]
        x_trend, y_trend = (
            xy_trend[:, : self.seq_len, :],
            xy_trend[:, -self.pred_len :, :],
        )
        # 5. RFM
        x_patch = self.VSP(x - x_trend, mode=0)
        y_patch = self.res_model(x_patch).float()
        y_res = self.VSP(y_patch, mode=1)

        return y_trend + y_res
