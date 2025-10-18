#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1 nohup torchrun \
        --nproc_per_node=2 \
        --master_port=32411 \
        SSK.py \
        --experiment GCD_base_prefix_gen \
        --dataset youtubefaces_1000 \
        --K 1000 \
        --max_kmeans_iter 500 \
        --k_means_init 10 \
        --experiment_idx gcd_youtubefaces_1000_part_fvit_pretrain_prefix10 \
        --save_dir results \
        --data-path youtube_faces_1000 > ./nohup_logs/ssk_gcd_youtubefaces_1000_part_fvit_pretrain_prefix10.out &