CONFIG_NAME=6B nsys profile --stats=true --sample=cpu --trace=cuda,cudnn,cublas,nvtx,osrt,oshmem --cudabacktrace=kernel:1000000,sync:1000000,memory:1000000 --duration=1200 --wait=all -o model6b-profile /home/ziyyang/v-wutianyuan/miniconda3/envs/rlhf/bin/accelerate launch --num_processes 7 \
    --config_file configs/accelerate/zero2-bf16.yaml \
    examples/hh/ppo_hh.py \
    --config-path examples/hh/configs/gptj-baseline-bugfix-ppo_loop-lr2e-6-ppoepoch1.json