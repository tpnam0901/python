import logging

logging.root.setLevel(logging.INFO)
logging.basicConfig(level=logging.root.level, format="%(name)s - %(levelname)s - %(message)s")

import argparse
import random

import numpy as np
import torch

import engine
from configs.base import Config, import_config


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cfg",
        "--config",
        type=str,
        default="configs/base.py",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "-cfg_ckpt",
        "--config_ckpt",
        type=str,
        default="",
        help="Path to the checkpoint .json file to load configuration from.",
    )
    parser.add_argument(
        "-e",
        "--engine",
        choices=["t", "i", "e"],
        required=True,
        help="Engine type to use: TrainEngine (t), InferenceEngine (i), EvaluateEngine (e).",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = arg_parser()
    cfg: Config = import_config(args.config)

    if args.engine in ["i", "e"]:
        assert (
            args.config_ckpt != ""
        ), "Checkpoint configuration file must be provided for Inference and Evaluation engines."
        cfg.load(args.config_ckpt)

    SEED = cfg.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    if args.engine == "t":
        trainer = engine.TrainEngine(cfg)
        trainer.run()
    elif args.engine == "i":
        inferencer = engine.InferenceEngine(cfg)
        inferencer.run()
    elif args.engine == "e":
        evaluator = engine.EvaluateEngine(cfg)
        evaluator.run()
