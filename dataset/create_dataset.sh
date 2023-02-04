#!/bin/bash

function func_basic {
    for (( scene_idx=0; scene_idx<num_data; scene_idx=scene_idx+1 )); do
        /usr/bin/python3 create_data.py \
            --seed_offset $seed_offset \
            --min_objects $min_objects \
            --max_objects $max_objects \
            --scene_idx $scene_idx
    done
}

export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export KUBRIC_USE_GPU=true

seed_offset=0
num_data=52000
min_objects=3
max_objects=6
func_basic

seed_offset=$num_data
num_data=1000
min_objects=7
max_objects=10
func_basic
