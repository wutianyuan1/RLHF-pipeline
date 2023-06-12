import os
import torch
import gc
import warnings
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from trlx.data.configs import TRLConfig
from trlx.data.default_configs import (
    default_ilql_config,
    default_ppo_config,
    default_sft_config,
)
from trlx.utils import set_seed
from trlx.utils.loading import get_pipeline, get_trainer


def print_all_tensors(output_file):
    for obj in gc.get_objects():
        try:
            flag = True
            shape = None
            device = None
            # if torch.is_tensor(obj):
            #     shape = obj.shape
            #     device = obj.device
            # elif ((hasattr(obj, 'data') and torch.is_tensor(obj.data))):
            #     shape = obj.data.shape
            #     device = obj.data.device
            if isinstance(obj, torch.nn.Module):
                shape = None
                device = {k : (v.shape, v.device) for (k, v) in obj.named_parameters()}
            else:
                flag = False
            if flag and (device is not None) and device != 'cpu':
                output_file.write("object type: {}, shape: {}, device: {}\n".format(type(obj), shape, device))
        except:
            pass

def cuda_memory_summary(total_gpus=8):
    pid = os.getpid()
    rank = os.environ.get("RANK", "0")
    with open(f"gpuusage/pid_{pid}_rank_{rank}.txt", 'w') as f:
        print_all_tensors(f)
        for i in range(total_gpus):
            f.write(f"GPU{i}:\n")
            f.write(torch.cuda.memory_summary(device=f'cuda:{i}'))
            f.write("-"*100 + '\n')

def train(  # noqa: C901
    model_path: Optional[str] = None,
    reward_fn: Optional[Callable[[List[str], List[str], List[str]], List[float]]] = None,
    dataset: Optional[Iterable[Tuple[str, float]]] = None,
    samples: Optional[List[str]] = None,
    rewards: Optional[List[float]] = None,
    prompts: Optional[List[str]] = None,
    eval_prompts: Optional[List[str]] = None,
    metric_fn: Optional[Callable[[List[str], List[str], List[str]], Dict[str, List[float]]]] = None,
    config: Optional[TRLConfig] = None,
    stop_sequences: Optional[List[str]] = [],
):
    """
    Dispatches online, offline reinforcement training or supervised finetuning
    depending on whether a reward function or a list of samples & rewards, or only list of samples is given

    Args:
        model_path (Optional[str]): Path to either huggingface checkpoint or a local directory
        config (Optional[TRLConfig]): TRLX configuration object
        reward_fn (Optional[Callable[[List[str], List[str], List[str]], List[float]]]):
            Function to rate batches of generated samples. Its arguments are
            (`samples`, `prompts`, `outputs`) and the return is a list of `rewards`
        dataset (List[Union[str, List[str]]], List[float]):
            Lists of samples and rewards for offline training. (Use `samples` and `rewards` instead)
        samples (List[Union[str, List[str]]]):
            List of strings or a list of prompts (questions or environment states) and outputs which are
            meant to be optimized. In the latter case the following form is expected:
            (prompt_0: str, output_0: str, prompt_1: str, output_1: str ...).
            Giving a single string `s` for the sample is a shorthand for (`tokenizer.bos_token`, `s`)
        rewards (List[float]):
            List of real numbers measuring the goodness of each sample
        prompts (`List[str]` or `List[Dict[str, Any]]`): Prompts to use for generations during online training.
            If a dict is passed as prompt, it must have a required key `"prompt"`, all the extra keys would be
            passed along the generation for that prompt as a keyword argument to reward function.
        eval_prompts (List[str] or `List[Dict[str, Any]]`): Prompts to use for periodical validation of training
        metric_fn (Optional[Callable[[List[str], List[str], List[str]], Dict[str, List[float]]]]):
            Function to compute statistics on batches of generated samples. Its arguments are the same
            as in `reward_fn` (`samples`, `prompts`, `outputs`) but the return is dictionary with keys
            as metric's name and values and lists of numeric values per each sample in batch
        stop_sequences (Optional[List[str]]):
            String sequences to trim generations (both for generating of experience and evaluation) up to its
            encounter in them. Generations will not contain them and also will also be right-stripped
    """
    if config is None:
        warnings.warn(
            "Passing the `config` argument implicitly is depreciated, use or"
            "adapt some from `trlx/data/default_configs.py` instead"
        )
        if reward_fn:
            config = default_ppo_config()
        elif rewards:
            config = default_ilql_config()
        else:
            config = default_sft_config()

    set_seed(config.train.seed)

    if dataset:
        warnings.warn("the `dataset` argument is being depreciated, split it into `samples` and `rewards` instead")
        samples, rewards = dataset

    if model_path:
        config.model.model_path = model_path

    trainer = get_trainer(config.train.trainer)(
        config=config,
        reward_fn=reward_fn,
        metric_fn=metric_fn,
        stop_sequences=stop_sequences,
        **config.train.trainer_kwargs,
    )
    # resume
    # TODO: ignore prompt loader's inner state for now

    batch_size = config.train.batch_size * int(os.environ.get("WORLD_SIZE", 1))
    max_prompt_length = config.train.seq_length - config.method.gen_kwargs["max_new_tokens"]

    # Online training against a reward function (e.g. PPO)
    if reward_fn:
        # print("!!->> reward fn")
        prompts = prompts or [trainer.tokenizer.bos_token] * batch_size

        if eval_prompts is None:
            eval_prompts = prompts[:batch_size]

        pipeline = get_pipeline(config.train.pipeline)(
            prompts, max_prompt_length, trainer.tokenizer, add_special_tokens=config.model.model_arch_type == "seq2seq"
        )
        trainer.add_prompt_pipeline(pipeline)

        if eval_prompts is None:
            eval_prompts = prompts[:batch_size]
        print("!!!!make exper:", config.method.num_rollouts)
        trainer.make_experience(config.method.num_rollouts)
        # print("!!!make exp done")

    # Offline training from the collected samples (e.g. SFT, ILQL)
    elif samples:
        # print("!!->> samples")
        if rewards is not None:
            if len(samples) != len(rewards):
                raise ValueError(f"Number of samples {len(samples)} should match the number of rewards {len(rewards)}")

        if eval_prompts is None:
            eval_prompts = [trainer.tokenizer.bos_token] * batch_size

        if rewards is not None:
            trainer.make_experience(samples, rewards, config.train.seq_length)
        else:
            trainer.make_experience(samples, config.train.seq_length)
    else:
        raise ValueError("Either `samples` or `reward_fn` should be given for training")

    eval_pipeline = get_pipeline(config.train.pipeline)(
        eval_prompts, max_prompt_length, trainer.tokenizer, add_special_tokens=config.model.model_arch_type == "seq2seq"
    )
    trainer.add_eval_pipeline(eval_pipeline)
    # print("!!->> learn:", trainer)
    # cuda_memory_summary()
    trainer.learn()
    return trainer
