import hydra, yaml
import lightning as L
import torch
from lightning.pytorch import Callback, LightningDataModule, LightningModule, Trainer
from omegaconf import DictConfig
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger, WandbLogger
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TQDMProgressBar,
    Timer,
)
import warnings
from utils.pylogger import RankedLogger

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")

log = RankedLogger(__name__, rank_zero_only=True)
from datetime import timedelta


def train(cfg: DictConfig):
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    print(
        f"Running {cfg.model.net_params.name} model on {cfg.data.data_params.name} dataset: {cfg.forecast.seq_len}->{cfg.forecast.pred_len}"
    )
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    logger = TensorBoardLogger(**cfg.logger)
    mcheck = ModelCheckpoint(monitor="val_loss")
    estop = EarlyStopping(monitor="val_loss", patience=5)
    timer = Timer(duration=timedelta(weeks=1))

    trainer: Trainer = hydra.utils.instantiate(
        # cfg.trainer, logger=logger, callbacks=[estop]
        cfg.trainer,
        logger=logger,
        callbacks=[estop, mcheck, timer],
    )

    # force training to stop after given time limit
    # trainer = Trainer(callbacks=[timer])

    # query training/validation/test time (in seconds)
    

    if cfg.get("train"):
        ckpt = None
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt)
        print("epoch_time:", timer.time_elapsed("train"))
        if trainer.should_stop:
            log.info("Early stopping!")
        test_output = trainer.test(datamodule=datamodule)
        setting = str(trainer.logger.log_dir)
        f = open(
            f"{trainer.logger.save_dir}_result.txt",
            "a",
        )
        f.write(setting + "  \n")
        f.write(
            f"MSE: {test_output[0]['hp_metric/MSE']:.5f}, MAE: {test_output[0]['hp_metric/MAE']:.5f}"
        )
        f.write("\n\n")


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    main()
