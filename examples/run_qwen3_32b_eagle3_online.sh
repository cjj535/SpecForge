#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
HOME=/home/c00882514

export REQUESTS_CA_BUNDLE=/home/c00882514/CA.crt
export SSL_CERT_FILE=/home/c00882514/CA.crt
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun \
    --standalone \
    --nproc_per_node 8 \
    $ROOT_DIR/scripts/train_eagle3.py \
    --chat-template qwen \
    --cache-dir $ROOT_DIR/cache \
    --tp-size 8 \
    --attention-backend sdpa \
    --target-model-backend hf \
    --num-epochs 10 \
    --draft-accumulation-steps 1 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --target-model-path $HOME/Qwen3-32B \
    --train-data-path $HOME/dataset/train.jsonl \
    --eval-data-path $HOME/dataset/eval.jsonl \
    --build-dataset-num-proc 100 \
    --output-dir $ROOT_DIR/outputs/qwen3-32b-eagle3 \
    --draft-model-config $ROOT_DIR/configs/qwen3-32b-eagle3.json \
    --report-to wandb \
    --wandb-project specforge-qwen3-32b \
    --wandb-name qwen3-32b-eagle3-tp4-online