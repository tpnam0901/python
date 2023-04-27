import logging
import os
from abc import ABC, abstractmethod
from typing import Dict

from flax.training.train_state import TrainState

# from utils.torch.trainer import TorchTrainer


class Callback(ABC):
    @abstractmethod
    def __call__(
        self,
        trainer,  #: TorchTrainer,
        global_step: int,
        global_epoch: int,
        logs: Dict,
        isValPhase: bool = False,
    ):
        """Abstract method to be implemented by the user.

        Args:
            trainer (trainer.Trainer): trainer.TorchTrainer module
            global_step (int): The global step of the training.
            global_epoch (int): The global epoch of the training.
            logs (Dict): The logs of the training which contains the loss and the metrics. For example:
                                                            {
                                                                "loss": 0.1,
                                                                "accuracy": 0.9
                                                                "some_custom_metric": 0.5
                                                            }
            isValPhase (bool, optional): Whether the callback is called during the validation phase. Defaults to False.
        """
        pass


class CheckpointsCallback(Callback):
    def __init__(
        self,
        checkpoint_dir: str,
        save_freq: int = 1000,
        keep_one_only: bool = False,
    ):
        """Callback to save checkpoints during training.

        Args:
            checkpoint_dir (str): Path to the directory where checkpoints will be saved.
            save_freq (int, optional): The frequency at which checkpoints will be saved. Defaults to 1000.
            keep_one_only (bool, optional): Whether to keep only the last checkpoint. Defaults to True.
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.save_freq = save_freq
        self.keep_one_only = keep_one_only

    def __call__(
        self,
        trainer,  # trainer.TorchTrainer
        global_step: int,
        global_epoch: int,
        logs: Dict,
        isValPhase: bool = False,
    ):
        """Abstract method to be implemented by the user.

        Args:
            trainer (trainer.Trainer): trainer.TorchTrainer module
            global_step (int): The global step of the training.
            global_epoch (int): The global epoch of the training.
            logs (Dict): The logs of the training which contains the loss and the metrics. For example:
                                                            {
                                                                "loss": 0.1,
                                                                "accuracy": 0.9
                                                                "some_custom_metric": 0.5
                                                            }
            isValPhase (bool, optional): Whether the callback is called during the validation phase. Defaults to False.
        """
        if not isValPhase:
            if global_step % self.save_freq == 0:
                if self.keep_one_only:
                    global_step = ""
                trainer.save(self.checkpoint_dir, global_step)
