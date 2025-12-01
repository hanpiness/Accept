# Copyright (c) wilson.xu. All rights reserved.

import os
import yaml
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare accelerate script.")
    parser.add_argument(
        "--config_file",
        default="default_config.yaml",
        type=str)
    parser.add_argument(
        "--cache_dir",
        default="/home/jovyan",
        type=str)
    parser.add_argument(
        "--mixed_precision",
        default="fp16",
        type=str)
    parser.add_argument(
        "--zero_stage",
        default=2,
        type=int)
    parser.add_argument(
        "--gradient_accumulation_steps",
        default=8,
        type=int)
    parser.add_argument(
        "--gradient_clipping",
        default=1.0,
        type=float)

    args = parser.parse_args()

    cache_dir = os.path.join(args.cache_dir, ".cache/huggingface/accelerate")
    os.makedirs(cache_dir, exist_ok=True)
    args.cache_dir = cache_dir

    return args


if __name__ == "__main__":
    args = parse_args()

    # 0. Open config.yaml to modify
    root_dir = os.path.abspath(os.path.join(__file__, *(['..'] * 1)))
    with open(os.path.join(root_dir, args.config_file), encoding="utf-8") as f:
        config = yaml.safe_load(f.read())

    # 节点数量
    num_nodes = int(os.environ.get("NNODES", 1))
    # GPU数量
    num_gpus = int(os.environ.get("NPROC_PER_NODE", 8))
    # 主节点IP地址
    main_process_ip = os.environ.get("MASTER_ADDR", None)
    # 节点序号
    rank = int(os.environ.get("NODE_RANK", 0))
    # DATA_OUTPUT_DIR
    out_dir = os.environ.get("DATA_OUTPUT_DIR", 0)

    config["machine_rank"] = rank
    config["main_process_ip"] = main_process_ip
    config["num_machines"] = num_nodes
    config["num_processes"] = num_nodes * num_gpus
    config["mixed_precision"] = args.mixed_precision

    if "deepspeed" in args.config_file:
        config["deepspeed_config"]["zero_stage"] = args.zero_stage
        config["deepspeed_config"]["gradient_accumulation_steps"] = args.gradient_accumulation_steps
        config["deepspeed_config"]["gradient_clipping"] = args.gradient_clipping

    with open(os.path.join(args.cache_dir, "default_config.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml.dump(config, allow_unicode=True))
