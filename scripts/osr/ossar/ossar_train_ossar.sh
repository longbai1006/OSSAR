#!/bin/bash
# sh scripts/osr/arpl/imagenet200_train_arpl.sh

python main.py \
    --config configs/datasets/imagenet200/imagenet200.yml \
    configs/networks/ossar_net.yml \
    configs/pipelines/train/ossar_hrpl.yml \
    configs/preprocessors/base_preprocessor.yml \
    --network.feat_extract_network.name resnet18_224x224 \
    --optimizer.num_epochs 90 \
    --dataset.train.batch_size 256 \
    --num_gpus 1 --num_workers 16 \
    --merge_option merge \
    --seed 0
