export CONFIG_NAME=6B
export  OMP_NUM_THREADS=1

accelerate launch --num_processes 7 \
    --config_file configs/accelerate/zero2-bf16.yaml \
    pipeline_ppo_runner/run.py\
    --config-path examples/hh/configs/gptj-baseline-bugfix-ppo_loop-lr2e-6-ppoepoch1.json
