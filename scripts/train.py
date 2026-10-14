#!/usr/bin/env python3
"""Training script for MATA"""

import argparse
import torch
import yaml
from mata import MATAModel


def train():
    parser = argparse.ArgumentParser(description='Train MATA model')
    parser.add_argument('--config', default='configs/default_config.yaml')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output-dir', default='./checkpoints')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    model = MATAModel(**config['model'])
    print("Training MATA model...")
    print(f"Config: {config}")
    print(f"Data directory: {args.data_dir}")


if __name__ == '__main__':
    train()
