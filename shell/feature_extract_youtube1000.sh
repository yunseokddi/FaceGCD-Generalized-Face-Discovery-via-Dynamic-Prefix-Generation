#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1 nohup torchrun \
    --nproc_per_node=2 \
    --master_port=25411 \
    extract_features.py \
    --pretrained \
    --return-embed \
    --save-images \
    --pin-mem \
    --experiment gcd_youtubefaces_1000_part_fvit_pretrain_prefix10 \
    --amp \
    --prefix_tuning \
    --prefix_length 10 \
    --data-path youtube_faces_1000 \
    --dataset youtubefaces_1000 \
    --amp-impl apex \
    --save_dir results \
    --landmark_cnn \
    --pretrained_weights checkpoint.pth \
    --checkpoint_weights model_best.pth.tar \
    --patch_size 8 \
    --input-size 3 112 112 > ./nohup_logs/feature_gcd_youtubefaces_1000_part_fvit_pretrain_prefix10.out &