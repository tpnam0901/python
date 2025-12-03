import logging
import math
import unittest

import torch

from configs.base import Config
from data.basedataset import BaseDataset
from utils import schedulers
from utils.dataloader import get_dataloader

_logger = logging.getLogger(f"{__name__}")
_logger.setLevel(logging.root.level)


class TestDataloader(unittest.TestCase):
    """Test cases for lock and unlock functionality in Config."""

    def setUp(self):
        """Set up test fixtures."""
        x = [i for i in range(10)]
        y = [i % 2 for i in range(10)]
        self.dataset = BaseDataset(x, y)

    def test_dataloader_batch_size(self):
        """Test that dataloader returns correct batch size."""
        _logger.debug("Testing dataloader batch size.")
        dataloader = get_dataloader(self.dataset, batch_size=4)
        batch = next(iter(dataloader))
        self.assertEqual(len(batch["x"]), 4)
        self.assertEqual(len(batch["y"]), 4)

    def test_dataloader_iteration(self):
        """Test that dataloader can iterate through the dataset."""
        _logger.debug("Testing dataloader iteration.")
        dataloader = get_dataloader(self.dataset, batch_size=2)
        total_samples = 0
        for batch in dataloader:
            total_samples += len(batch["x"])
        self.assertEqual(total_samples, len(self.dataset))

    def test_dataloader_drop_last(self):
        """Test that dataloader drops last incomplete batch when drop_last is True."""
        _logger.debug("Testing dataloader drop_last functionality.")
        dataloader = get_dataloader(self.dataset, batch_size=3, drop_last=True)
        total_samples = 0
        for batch in dataloader:
            total_samples += len(batch["x"])
        self.assertEqual(total_samples, 9)  # 10 samples with batch size 3 drops last sample


class TestScheduler(unittest.TestCase):
    """Test cases for different learning rate schedulers."""

    _logger.debug("------------ Setting up TestScheduler ------------")

    def setUp(self):
        """Set up test fixtures."""

        self.cfg = Config()
        self.cfg.unlock()
        self.cfg.learning_rate = 0.1
        self.cfg.lock()

        # Create a simple model and optimizer for testing
        self.model = torch.nn.Linear(10, 1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)

    def test_step_lr_scheduler(self):
        """Test StepLR scheduler decreases learning rate at specified step."""
        _logger.debug("Testing StepLR scheduler.")

        self.cfg.unlock()
        self.cfg.lr_step_size = 5
        self.cfg.lr_step_gamma = 0.5
        self.cfg.lock()

        scheduler = schedulers.StepLR(self.optimizer, self.cfg)
        initial_lr = self.optimizer.param_groups[0]["lr"]

        # Track learning rates
        lrs = [initial_lr]
        for epoch in range(15):
            scheduler.step()
            lrs.append(self.optimizer.param_groups[0]["lr"])

        # Verify LR decreases at step_size intervals
        self.assertAlmostEqual(lrs[0], 0.1, places=5)
        self.assertAlmostEqual(lrs[5], 0.05, places=5)  # After 5 epochs: 0.1 * 0.5
        self.assertAlmostEqual(lrs[10], 0.025, places=5)  # After 10 epochs: 0.1 * 0.5^2
        self.assertAlmostEqual(lrs[15], 0.0125, places=5)  # After 15 epochs: 0.1 * 0.5^3

        _logger.debug(f"StepLR learning rates: {lrs}")

    def test_multi_step_lr_scheduler(self):
        _logger.debug(f"Testing MultiStepLR scheduler is not yet implemented.")

    def test_exponential_lr_scheduler(self):
        _logger.debug(f"Testing ExponentialLR scheduler is not yet implemented.")

    def test_cosine_annealing_lr_scheduler(self):
        _logger.debug(f"Testing CosineAnnealingLR scheduler is not yet implemented.")

    def test_reduce_lr_on_plateau_scheduler(self):
        _logger.debug(f"Testing ReduceLROnPlateau scheduler is not yet implemented.")

    def test_cosine_annealing_warm_restarts_scheduler(self):
        _logger.debug(f"Testing CosineAnnealingWarmRestarts scheduler is not yet implemented.")

    def test_identity_scheduler(self):
        _logger.debug(f"Testing IdentityScheduler is not yet implemented.")
