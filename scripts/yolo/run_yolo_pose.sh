#!/bin/bash

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80
export HTTP_PROXY=http://proxy:80
export HTTPS_PROXY=http://proxy:80

source .venv/bin/activate
.venv/bin/python3 ./scripts/yolo/train_yolo_pose.py

# salloc --gres=gpu:a100:1 -C a100_80 --partition=a100 --time=02:00:00