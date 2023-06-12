# 5.16
accelerate launch --num_processes 7 \
    --config_file configs/accelerate/zero2-bf16.yaml \
    examples/hh/ppo_hh.py \
    --config-path examples/hh/configs/llama-7b-lr2e-6-bugfix.json

# 5.15
# accelerate launch --num_processes 7 \
#     --config_file configs/accelerate/zero2-bf16.yaml \
#     examples/hh/ppo_hh.py \
#     --config-path examples/hh/configs/gptj-rmstatic-sft-bugfix-lr2e-6.json

# accelerate launch --num_processes 7 \
#     --config_file configs/accelerate/zero2-bf16.yaml \
#     examples/hh/ppo_hh.py \
#     --config-path examples/hh/configs/gptj-rmstatic-sft-bugfix-lr2e-6-ppoepoch1.json

# 5.14
# accelerate launch --num_processes 7 \
#     --config_file configs/accelerate/zero2-bf16.yaml \
#     examples/hh/ppo_hh.py \
#     --config-path examples/hh/configs/gptj-ppoepoch1-lr2e-6.json

# accelerate launch --num_processes 7 \
#     --config_file configs/accelerate/zero2-bf16.yaml \
#     examples/hh/ppo_hh.py \
#     --config-path examples/hh/configs/gptj-rmstatic-sft-lr2e-6-ppoepoch1.json

# 5.13

# accelerate launch --num_processes 7 \
#     --config_file configs/accelerate/zero2-bf16.yaml \
#     examples/hh/ppo_hh.py \
#     --config-path examples/hh/configs/vicuna-7b-lr8e-7-ppoepoch1.json

# accelerate launch --num_processes 7 \
#     --config_file configs/accelerate/zero2-bf16.yaml \
#     examples/hh/ppo_hh.py \
#     --config-path examples/hh/configs/gptj-rmstatic-sft-baseline.json

# training is not stable
# accelerate launch --num_processes 7 \
#     --config_file configs/accelerate/zero2-bf16.yaml \
#     examples/hh/ppo_hh.py \
#     --config-path examples/hh/configs/gptj-ppoepoch1-steps40000.json

# lr scheduler still buggy
# accelerate launch --num_processes 7 \
#     --config_file configs/accelerate/zero2-bf16.yaml \
#     examples/hh/ppo_hh.py \
#     --config-path examples/hh/configs/gptj-cosine.json
