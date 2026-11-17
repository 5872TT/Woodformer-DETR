"""Train or validate the FEWD-DETR reference configuration."""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import src.misc.dist as dist
from src.core import YAMLConfig
from src.solver import TASKS


DEFAULT_CONFIG = os.path.join(ROOT, "configs", "rtdetr", "fewd_rtdetr_r18_ouludefect.yml")


def main():
    parser = argparse.ArgumentParser(description="Train/validate FEWD-DETR using the RT-DETR solver")
    parser.add_argument("-c", "--config", default=DEFAULT_CONFIG, help="FEWD-DETR YAML configuration")
    parser.add_argument("-r", "--resume", default=None, help="checkpoint to resume or validate")
    parser.add_argument("-t", "--tuning", default=None, help="checkpoint for compatible-weight tuning")
    parser.add_argument("--test-only", action="store_true", help="run validation instead of training")
    parser.add_argument("--amp", action="store_true", default=None,
                        help="enable automatic mixed precision (the YAML setting is used otherwise)")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args()

    if args.resume and args.tuning:
        parser.error("--resume and --tuning cannot be used together")
    if args.test_only and not args.resume:
        parser.error("--test-only requires --resume <checkpoint>")
    if not os.path.isfile(args.config):
        parser.error(f"configuration not found: {args.config}")

    dist.init_distributed()
    if args.seed is not None:
        dist.set_seed(args.seed)

    cfg_options = {"resume": args.resume, "tuning": args.tuning}
    if args.amp:
        cfg_options["use_amp"] = True
    cfg = YAMLConfig(args.config, **cfg_options)
    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    if args.test_only:
        solver.val()
    else:
        solver.fit()


if __name__ == "__main__":
    main()
