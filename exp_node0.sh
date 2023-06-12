# 5.15
accelerate launch --num_processes 7 \
    --config_file configs/accelerate/zero2-bf16.yaml \
    examples/hh/ppo_hh.py \
    --config-path examples/hh/configs/gptj-baseline-bugfix-ppo_loop-lr2e-6-ppoepoch1.json


# 5.14
# git checkout dev_bugfix/ppo_loop

# accelerate launch --num_processes 7 \
#     --config_file configs/accelerate/zero2-bf16.yaml \
#     examples/hh/ppo_hh.py \
#     --config-path examples/hh/configs/gptj-baseline-bugfix-ppo_loop-lr2e-6.json

# accelerate launch --num_processes 7 \
#     --config_file configs/accelerate/zero2-bf16.yaml \
#     examples/hh/ppo_hh.py \
#     --config-path examples/hh/configs/gptj-baseline-bugfix-ppo_loop-lr8e-7.json

# git checkout dev

# accelerate launch --num_processes 7 \
#     --config_file configs/accelerate/zero2-bf16.yaml \
#     examples/hh/ppo_hh.py \
#     --config-path examples/hh/configs/gptj-lr8e-7.json

# 5.13

# accelerate launch --num_processes 7 \
#     --config_file configs/accelerate/zero2-bf16.yaml \
#     examples/hh/ppo_hh.py \
#     --config-path examples/hh/configs/gptj-baseline.json

# accelerate launch --num_processes 7 \
#     --config_file configs/accelerate/zero2-bf16.yaml \
#     examples/hh/ppo_hh.py \
#     --config-path examples/hh/configs/gptj-lr2e-6.json

# git checkout dev_bugfix/ppo_loop

# accelerate launch --num_processes 7 \
#     --config_file configs/accelerate/zero2-bf16.yaml \
#     examples/hh/ppo_hh.py \
#     --config-path examples/hh/configs/gptj-baseline-bugfix-ppo_loop.json

