import os.path as osp
from typing import Dict

import mlflow
import torch
import torch.nn as nn
from torch.optim import SGD
from tqdm.auto import tqdm

import networks
from configs.base import Config
from data.basedataset import BaseDataset
from engine.base import BaseEngine
from utils.dataloader import get_dataloader
from utils.schedulers import StepLR


class TrainEngine(BaseEngine):
    def __init__(self, cfg: Config):
        super(TrainEngine, self).__init__(osp.join(cfg.ckpt_dir, cfg.current_time, "train.log"))
        self.cfg = cfg
        cfg.save(osp.join(cfg.ckpt_dir, cfg.current_time, "config.json"))
        self.setup_mlflow(run_name="train_run", experiment_name="train_experiment")
        with mlflow.start_run(run_name=self.mlflow_run_name, run_id=self.mlflow_id):
            # Log configuration parameters
            mlflow.log_params(vars(cfg))

        self.loss_mse = nn.MSELoss()
        self.best_val_loss = float("inf")

    def build_model(self):
        """Build the model for training."""
        self.logger.info("Building the model.")
        # Model building logic would go here
        model_class = getattr(networks, self.cfg.model_type)
        return model_class(self.cfg)

    def load_train_dataset(
        self,
        batch_size: int,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = True,
    ) -> Dict:
        """Build the training dataset.

        Args:
            batch_size (int): Batch size for the dataloaders.
            shuffle (bool): Whether to shuffle the training data.
            num_workers (int): Number of worker threads for data loading.
            pin_memory (bool): Whether to use pinned memory for data loading.
        Returns:
            Dict: A dictionary containing training and validation datasets and dataloaders.
        """
        self.logger.info("Building training dataset.")
        x_train = list(range(100))
        y_train = [i * 2 for i in x_train]
        train_dataset = BaseDataset(x_train, y_train)

        x_val = list(range(100, 120))
        y_val = [i * 2 for i in x_val]
        val_dataset = BaseDataset(x_val, y_val)
        return {
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
            "train_loader": get_dataloader(
                train_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
            ),
            "val_loader": get_dataloader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
            ),
        }

    def calculate_loss(self, predictions: Dict, targets: Dict) -> Dict:
        """Calculate loss given predictions and targets.

        Args:
            predictions (Dict): Model predictions.
            targets (Dict): Ground truth targets.
        Returns:
            Dict: Calculated loss values.
        """
        logits = predictions["logits"]
        targets = targets["y"]

        # Calculate Mean Squared Error loss
        loss = self.loss_mse(logits, targets)
        return {"total_loss": loss, "mse_loss": loss}

    def save_checkpoint(self, model, epoch: int, keep_only_latest: bool = True, prefix: str = "") -> None:
        """Save model checkpoint.

        Args:
            epoch (int): Current epoch number.
            keep_only_latest (bool): Whether to keep only the latest checkpoint. If True, save with the name 'latest.pth'.
        """
        ckpt_path = osp.join(self.cfg.ckpt_dir, self.cfg.current_time)
        if keep_only_latest:
            ckpt_file = osp.join(ckpt_path, prefix + "latest.pth")
        else:
            ckpt_file = osp.join(ckpt_path, prefix + f"epoch_{epoch}.pth")
        torch.save(model.state_dict(), ckpt_file)
        self.logger.info(f"Saving checkpoint to {ckpt_file}")

    def train_epoch(self, model, dataloader, optimizer, scheduler) -> None:
        loss_epoch = 0.0
        model.train()
        with mlflow.start_run(run_name=self.mlflow_run_name, run_id=self.mlflow_id):
            with tqdm(total=len(dataloader), ascii=True) as pbar:
                for batch in dataloader:
                    optimizer.zero_grad()
                    x, y = batch["x"].float().unsqueeze(1), batch["y"].float().unsqueeze(1)
                    # Forward pass
                    outputs = model(x)
                    # Compute loss
                    loss_all = self.calculate_loss(outputs, {"y": y})
                    # Backward pass and optimization

                    loss_all["total_loss"].backward()
                    optimizer.step()

                    loss_epoch += loss_all["total_loss"].item()

                    pbar.set_description(f"Loss: {loss_all['total_loss'].item():.4f}")
                    pbar.update()

            scheduler.step()
            mlflow.log_metric("learning_rate", scheduler.get_last_lr()[0], step=scheduler.last_epoch)
            mlflow.log_metric("train_epoch_loss", loss_epoch / len(dataloader), step=scheduler.last_epoch)
        self.logger.info(f"Epoch training loss: {loss_epoch / len(dataloader):.4f}")

    def eval_epoch(self, model, dataloader, save_best: bool = True, mlflow_log=True) -> None:
        loss_epoch = 0.0
        model.eval()

        with torch.no_grad():
            with tqdm(dataloader, desc="Evaluating", unit="batch") as tepoch:
                for batch in tepoch:
                    x, y = batch["x"].float().unsqueeze(1), batch["y"].float().unsqueeze(1)
                    # Forward pass
                    outputs = model(x)
                    # Compute loss
                    loss_all = self.calculate_loss(outputs, {"y": y})

                    loss_epoch += loss_all["total_loss"].item()

                    tepoch.set_postfix(loss=loss_all["total_loss"].item())
                    tepoch.update()

        avg_val_loss = loss_epoch / len(dataloader)

        if save_best and avg_val_loss < self.best_val_loss:
            self.save_checkpoint(model, epoch=0, keep_only_latest=self.cfg.keep_only_latest, prefix="best_")
            self.logger.info(
                f"Validation loss improved from {self.best_val_loss:.4f} to {avg_val_loss:.4f}. Saved best model."
            )
            self.best_val_loss = avg_val_loss
        if mlflow_log:
            with mlflow.start_run(run_name=self.mlflow_run_name, run_id=self.mlflow_id):
                mlflow.log_metric("val_epoch_loss", avg_val_loss, step=self.cfg.num_epochs)
        self.logger.info(f"Epoch validation loss: {avg_val_loss:.4f}")

    def run(self):
        """Run the training process."""
        datasets = self.load_train_dataset(
            self.cfg.batch_size,
            shuffle=self.cfg.shuffle,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )
        train_loader = datasets["train_loader"]
        val_loader = datasets["val_loader"]

        model = self.build_model()
        optimizer = SGD(model.parameters(), lr=self.cfg.learning_rate, momentum=self.cfg.sgd_momentum)
        scheduler = StepLR(optimizer, self.cfg)

        for epoch in range(self.cfg.num_epochs):
            self.logger.info(f"Starting epoch {epoch + 1}/{self.cfg.num_epochs}")
            self.train_epoch(model, train_loader, optimizer, scheduler)
            self.eval_epoch(model, val_loader, True)
            self.save_checkpoint(model, epoch + 1, keep_only_latest=True)
        self.logger.info("Training completed.")
