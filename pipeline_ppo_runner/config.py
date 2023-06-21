import os

from trlx.data.default_configs import (
    ModelConfig,
    OptimizerConfig,
    PPOConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)


def get_default_config():
    default_config = TRLConfig(
        train=TrainConfig(
            tracker=None,
            seq_length=1024,
            epochs=10000,
            total_steps=10000,
            batch_size=4,
            checkpoint_interval=1000,
            eval_interval=500,
            pipeline="PromptPipeline",
            # trainer="PipelinedPPOTrainer",
            trainer="AcceleratePPOTrainer",
            checkpoint_dir="checkpoints/ppo_hh",
        ),
        model=ModelConfig(model_path="EleutherAI/gpt-j-6B", num_layers_unfrozen=2),
        tokenizer=TokenizerConfig(tokenizer_path="EleutherAI/gpt-j-6B", truncation_side="left"),
        optimizer=OptimizerConfig(name="adamw", kwargs=dict(lr=8e-6, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)),
        scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=8e-6)),
        method=PPOConfig(
            name="PPOConfig",
            num_rollouts=64,
            chunk_size=16,
            ppo_epochs=4,
            init_kl_coef=0.05,
            target=6,
            horizon=10000,
            gamma=1,
            lam=0.95,
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=1,
            scale_reward="running",
            ref_mean=None,
            ref_std=None,
            cliprange_reward=10,
            gen_kwargs=dict(
                max_new_tokens=128,
                top_k=0,
                top_p=1.0,
                do_sample=True,
            ),
        ),
    )
    config_name = os.environ.get("CONFIG_NAME")
    if config_name == "125M":
        default_config.train.batch_size = 32
        default_config.train.total_steps = 1500
        default_config.train.checkpoint_dir = "checkpoints/ppo_hh_125M"
        default_config.model.model_path = "Dahoas/pythia-125M-static-sft"
        default_config.tokenizer.tokenizer_path = "EleutherAI/gpt-neox-20b"
        default_config.method.num_rollouts = 128
    elif config_name == "1B":
        default_config.train.batch_size = 8
        default_config.train.total_steps = 2500
        default_config.optimizer.kwargs["lr"] = 6e-6
        default_config.scheduler.kwargs["eta_min"] = 6e-6
        default_config.train.checkpoint_dir = "checkpoints/ppo_hh_1B"
        default_config.model.model_path = "Dahoas/pythia-1B-static-sft"
        default_config.tokenizer.tokenizer_path = "EleutherAI/gpt-neox-20b"
        default_config.method.chunk_size = 16
    elif config_name == "6B":
        default_config.train.batch_size = 4
        default_config.train.seq_length = 512
        default_config.train.total_steps = 6000
        default_config.train.checkpoint_dir = "checkpoints/ppo_hh_6B"
        default_config.model.model_path = "Dahoas/pythia-6B-static-sft"
        default_config.tokenizer.tokenizer_path = "EleutherAI/gpt-neox-20b"
        default_config.method.chunk_size = 16
    elif config_name == "20B":
        default_config.train.seq_length = 512
        default_config.train.batch_size = 1
        default_config.train.total_steps = 8000
        default_config.optimizer.kwargs["lr"] = 1e-6
        default_config.scheduler.kwargs["eta_min"] = 1e-6
        default_config.train.checkpoint_dir = "checkpoints/ppo_hh_20B"
        default_config.model.model_path = "EleutherAI/gpt-neox-20b"
        default_config.tokenizer.tokenizer_path = "EleutherAI/gpt-neox-20b"
        default_config.method.num_rollouts = 16
        default_config.method.chunk_size = 4
        default_config.method.ppo_epochs = 2
    return default_config
