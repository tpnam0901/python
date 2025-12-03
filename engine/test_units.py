import logging
import tempfile
import unittest

import mlflow

from engine.base import BaseEngine

_logger = logging.getLogger(f"{__name__}")
_logger.setLevel(logging.root.level)


class TestBaseEngine(unittest.TestCase):
    """Test cases for lock and unlock functionality in Config."""

    def test_base_engine_without_log_path(self):
        """Test BaseEngine initialization without log path."""
        _logger.debug("Testing BaseEngine initialization without log path.")

        class DummyClass(BaseEngine):
            def run(self):
                pass

        engine = DummyClass()
        self.assertIsNotNone(engine.logger)
        # Test logging prints to console
        _logger.setLevel(logging.DEBUG)
        with self.assertLogs(engine.logger, level="DEBUG") as log:
            engine.logger.debug("Test log entry.")
            self.assertIn("Test log entry.", log.output[0])
        _logger.setLevel(logging.root.level)

    def test_base_engine_with_log_path_and_handler(self):
        """Test BaseEngine initialization with log path."""
        _logger.debug("Testing BaseEngine initialization with log path.")

        class DummyClass(BaseEngine):
            def run(self):
                pass

        with tempfile.NamedTemporaryFile(delete=True) as temp_log_file:
            engine = DummyClass(log_path=temp_log_file.name)
            self.assertIsNotNone(engine.logger)
            handlers = engine.logger.handlers
            self.assertTrue(any(isinstance(h, logging.FileHandler) for h in handlers))
            # Test that the log file is created when logging
            _logger.setLevel(logging.DEBUG)
            engine.logger.debug("Test log entry.")
            with open(temp_log_file.name, "r") as f:
                log_contents = f.read()
                self.assertIn("Test log entry.", log_contents)
            _logger.setLevel(logging.root.level)

    def test_setup_mlflow(self):
        """Test MLflow setup in BaseEngine."""
        _logger.debug("Testing MLflow setup in BaseEngine.")

        class DummyClass(BaseEngine):
            def run(self):
                pass

        engine = DummyClass()
        engine.setup_mlflow(run_name="test_run", experiment_name="test_experiment")
        self.assertIsNotNone(engine.mlflow_id)
        self.assertEqual(engine.mlflow_run_name, "test_run")
        # Test logging with mlflow id and run name
        with mlflow.start_run(run_name=engine.mlflow_run_name, run_id=engine.mlflow_id) as run:
            self.assertEqual(run.info.run_id, engine.mlflow_id)
            self.assertEqual(run.info.run_name, engine.mlflow_run_name)


class TestTrainEngine(unittest.TestCase):
    """Test cases for TrainEngine."""

    def test_load_train_dataset(self):
        """Test loading of training dataset in TrainEngine."""
        from configs.base import Config
        from engine.train import TrainEngine

        _logger.debug("Testing loading of training dataset in TrainEngine.")
        cfg = Config()
        engine = TrainEngine(cfg)
        datasets = engine.load_train_dataset(
            batch_size=2,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )
        self.assertIn("train_dataset", datasets)
        self.assertIn("val_dataset", datasets)
        self.assertIn("train_loader", datasets)
        self.assertIn("val_loader", datasets)
        self.assertEqual(len(datasets["train_dataset"]), 100)
        self.assertEqual(len(datasets["val_dataset"]), 20)
        train_batch = next(iter(datasets["train_loader"]))
        self.assertEqual(train_batch["x"][0] * 2, train_batch["y"][0])

    def test_calculate_loss(self):
        """Test loss calculation in TrainEngine."""
        import torch

        from configs.example import Config as ExampleConfig
        from engine.train import TrainEngine

        _logger.debug("Testing loss calculation in TrainEngine.")
        cfg = ExampleConfig()
        engine = TrainEngine(cfg)

        # Create dummy data
        targets = {"y": torch.tensor([[2.0], [4.0], [6.0]])}
        logits = {"logits": torch.tensor([[1.5], [3.5], [5.5]])}

        # Calculate loss
        loss_dict = engine.calculate_loss(logits, targets)
        self.assertIn("total_loss", loss_dict)
        self.assertIn("mse_loss", loss_dict)
        # In this case, MSE loss should be ((0.5^2 + 0.5^2 + 0.5^2) / 3) = 0.25
        self.assertAlmostEqual(loss_dict["mse_loss"].item(), 0.25, places=4)


class TestEvaluateEngine(unittest.TestCase):
    """Test cases for EvaluateEngine."""

    def test_load_checkpoint_no_file(self):
        """Test loading checkpoint when no file exists."""
        import tempfile

        from configs.example import Config
        from engine.evaluate import EvaluateEngine

        _logger.debug("Testing loading checkpoint with no existing file in EvaluateEngine.")
        cfg = Config()
        cfg.ckpt_dir = tempfile.mkdtemp()
        cfg.current_time = "non_existent_time"

        engine = EvaluateEngine(cfg)

        model = engine.build_model()
        with self.assertRaises(FileNotFoundError):
            engine.load_checkpoint(model)

    def test_load_checkpoint_with_file(self):
        """Test loading checkpoint when file exists."""
        import os
        import tempfile

        import torch

        from configs.example import Config
        from engine.evaluate import EvaluateEngine

        _logger.debug("Testing loading checkpoint with existing file in EvaluateEngine.")
        cfg = Config()
        cfg.ckpt_dir = tempfile.mkdtemp()
        cfg.current_time = "test_time"

        # Create a dummy model and save a checkpoint
        engine = EvaluateEngine(cfg)
        model = engine.build_model()
        os.makedirs(os.path.join(cfg.ckpt_dir, cfg.current_time), exist_ok=True)
        ckpt_path = os.path.join(cfg.ckpt_dir, cfg.current_time, "latest.pth")
        torch.save(model.state_dict(), ckpt_path)

        # Now load the checkpoint
        engine.load_checkpoint(model)  # Should not raise an error
