import logging
import os.path as osp

import torch
import torch.nn as nn

from configs.base import Config
from engine.train import TrainEngine


class EvaluateEngine(TrainEngine):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.logger = logging.getLogger("Evaluate")
        self.logger.setLevel(logging.root.level)
        self.loss_mse = nn.MSELoss()

    def load_checkpoint(self, model):
        """Save model checkpoint.

        Args:
            epoch (int): Current epoch number.
            keep_only_latest (bool): Whether to keep only the latest checkpoint. If True, save with the name 'latest.pth'.
        """
        ckpt_path = osp.join(self.cfg.ckpt_dir, self.cfg.current_time, "best_latest.path")
        if not osp.exists(ckpt_path):
            ckpt_path = osp.join(self.cfg.ckpt_dir, self.cfg.current_time, "latest.pth")
        if not osp.exists(ckpt_path):
            raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path))
        self.logger.info(f"Loaded checkpoint from {ckpt_path}")

    def run(self):
        """Run the training process."""
        datasets = self.load_train_dataset(
            self.cfg.batch_size,
            shuffle=self.cfg.shuffle,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )
        val_loader = datasets["val_loader"]

        model = self.build_model()
        self.load_checkpoint(model)

        self.eval_epoch(model, val_loader, False, mlflow_log=False)
