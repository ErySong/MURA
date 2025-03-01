import hydra, torch
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import CSVLogger
from omegaconf import DictConfig

torch.set_float32_matmul_precision("medium")


def evaluate(cfg: DictConfig):
    assert cfg.ckpt_path

    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    logger = CSVLogger(save_dir=cfg.logger.save_dir, name="test")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    test_output = evaluate(cfg)


if __name__ == "__main__":
    main()
