import numpy as np
import torch
import matplotlib.pyplot as plt
import time, math
import matplotlib.gridspec as gridspec
from .metrics import *
from thop import profile

plt.switch_backend("agg")


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == "type1":
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == "type2":
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    elif args.lradj == "type3":
        lr_adjust = {
            epoch: (
                args.learning_rate
                if epoch < 3
                else args.learning_rate * (0.8 ** ((epoch - 3) // 1))
            )
        }
    elif args.lradj == "constant":
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == "3":
        lr_adjust = {
            epoch: args.learning_rate if epoch < 10 else args.learning_rate * 0.1
        }
    elif args.lradj == "4":
        lr_adjust = {
            epoch: args.learning_rate if epoch < 15 else args.learning_rate * 0.1
        }
    elif args.lradj == "5":
        lr_adjust = {
            epoch: args.learning_rate if epoch < 25 else args.learning_rate * 0.1
        }
    elif args.lradj == "6":
        lr_adjust = {
            epoch: args.learning_rate if epoch < 5 else args.learning_rate * 0.1
        }
    elif args.lradj == "TST":
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        if printout:
            print("Updating learning rate to {}".format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), path + "/" + "checkpoint.pth")
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual_channels(
    history, true, preds=None, visual_cs=[0, -1], name="./pic/test.png"
):
    """
    Visualize channels of a multivariate time series dataset
    preds: (L,K)
    """
    CL = history.shape[0]
    PL = preds.shape[0]
    time = np.arange(CL + PL)
    if time[-1] >= 1200:
        col = 1
        row = math.ceil(len(visual_cs) / col)
        hsapce = 0.8
        wspace = 0.2
    elif time[-1] >= 400:
        col = 2
        row = math.ceil(len(visual_cs) / col)
        hsapce = 0.6
        wspace = 0.1
    else:
        col = 3
        row = math.ceil(len(visual_cs) / col)
        hsapce = 0.4
        wspace = 0.1
    # fig, axes = plt.subplots(row, col, figsize=(row * 10, col * 10))
    fig = plt.figure(figsize=(30, 25))
    gs = gridspec.GridSpec(row, col, figure=fig)
    plt.subplots_adjust(
        hspace=hsapce, wspace=wspace
    )  # hspace 控制行间距，wspace 控制列间距
    # 设置每个子图的大小（可以通过调整 left, right, top, bottom 来控制）
    axes = []
    for i in range(row):
        for j in range(col):
            if i * col + j > len(visual_cs) - 1:
                break
            ax = fig.add_subplot(gs[i, j])
            axes.append(ax)
            channel = visual_cs[i * col + j]
            ax.plot(
                time,
                np.concatenate((history[:, channel], true[:, channel])),
                label="GroundTruth",
                linewidth=1.5,
                color="#1F77B4",
            )
            ax.plot(
                time[-PL:],
                preds[:, channel],
                label="Prediction",
                color="#FF7F0E",
                linewidth=1.5,
            )
            mse = MSE(preds[:, channel], true[:, channel])
            mae = MAE(preds[:, channel], true[:, channel])
            ax.axvline(x=CL, color="r", linestyle="--", linewidth=1)
            ax.set_title(f"Channel[{channel}]: MSE={mse:.3f}, MAE={mae:.3f}")
            ax.legend(
                loc="best",
                frameon=True,
                framealpha=1,
                edgecolor="black",
                facecolor="white",
            )
            ax.set_xlim(0, time[-1])  # 根据实际情况调整
            ax.grid(True)  # 为每个子图设置网格
        plt.legend()
    plt.savefig(name, bbox_inches="tight", dpi=300, format="png")


def visual(true, preds=None, name="./pic/test.pdf"):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label="GroundTruth", linewidth=1, color="blue")
    if preds is not None:
        plt.plot(preds, label="Prediction", linewidth=1, color="green")
    plt.axvline(x=2, color="r", linestyle="--", linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches="tight", dpi=300, format="png")


def test_params_flop(model, x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    # model_params = 0
    # for parameter in model.parameters():
    #     model_params += parameter.numel()
    #     print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info

    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(
            model.cuda(), x_shape, as_strings=True, print_per_layer_stat=False
        )
        # print("{:<30}  {:<8}".format("Computational complexity: ", macs))
        # print("{:<30}  {:<8}".format("Number of parameters: ", params))
    return macs, params


def _get_profile(model, shape, device):
    _input = torch.randn(shape).to(device)
    macs, params = profile(model, inputs=(_input,))
    macs = f"{macs / 10**6:.2f}M"
    params = f"{params / 10**3:.2f}k"
    print("FLOPs: ", macs)
    print("params: ", params)
    return macs, params
