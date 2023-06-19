import json
import os
import uuid
from time import time
from typing import Callable, List

import torch
import torch.nn.functional as F

import trlx.utils.logging as logging
from trlx.data.accelerate_base_datatypes import PromptBatch
from trlx.data.configs import TRLConfig
from trlx.data.ppo_types import PPORLElement
from trlx.trainer import register_trainer
from trlx.trainer.accelerate_ppo_trainer import AcceleratePPOTrainer
from trlx.utils import Clock
from trlx.utils.modeling import gather_dict, logprobs_of_labels

logger = logging.get_logger(__name__)


def generate_samples(self_obj, stats, prompt_iterator, accelerator, tokenizer, queue):
    # Get next batch in prompt dataset
    batch: PromptBatch = next(prompt_iterator)

    rollout_generate_time = time()
    # Generate samples from the language model (similar to using HuggingFace `generate` method)
    samples = self_obj.generate(batch["input_ids"], batch["attention_mask"])
    stats["time/rollout_generate"] = time() - rollout_generate_time

    prompt_tensors = batch.input_ids
    device = samples.device

    prompt_sizes = torch.tensor([prompt_tensors.shape[1]] * len(prompt_tensors), device=device)
    padded_samples = accelerator.pad_across_processes(
        samples, dim=1, pad_index=tokenizer.eos_token_id, pad_first=False
    )
    padded_prompts = accelerator.pad_across_processes(
        prompt_tensors, dim=1, pad_index=tokenizer.eos_token_id, pad_first=False
    )
    gathered_samples = accelerator.gather(padded_samples)
    gathered_prompts = accelerator.gather(padded_prompts)
    gathered_prompt_sizes = accelerator.gather(prompt_sizes)
    if accelerator.is_main_process:
        queue.push(gathered_samples, gathered_prompts, gathered_prompt_sizes)



def post_process(self_obj, config, stats, tokenizer, model, ref_model, tbar):
    if torch.distributed.is_initialized():
        scores = torch.empty(len(samples), device=device)
        torch.distributed.scatter(scores, all_scores)
    else:
        scores = all_scores[0].clone().detach()
    str_samples, str_prompts, str_outputs = self_obj.decode(prompt_tensors, samples, append_eos_token=True)

    # Pad the sample outputs
    outputs = tokenizer(str_outputs).input_ids
    if config.model.model_arch_type == "seq2seq":
        # add <pad> to the start of the output
        for i in range(len(outputs)):
            outputs[i] = [tokenizer.pad_token_id] + outputs[i]

    outputs = list(map(torch.LongTensor, outputs))
    maxsize = max(map(len, outputs))
    outputs = [
        F.pad(
            output,
            (0, maxsize - len(output)),
            value=tokenizer.pad_token_id,
        )
        for output in outputs
    ]
    sample_outputs = torch.vstack(outputs).to(device)

    if config.method.cliprange_reward:
        scores = torch.clip(scores, -config.method.cliprange_reward, config.method.cliprange_reward)

    # store statistics of the initial rollout as reference
    if self_obj.ref_mean is None:
        self_obj.ref_mean, self_obj.ref_std = scores.mean(), scores.std()
    all_scores_mean, all_scores_std = self_obj.running_moments.update(scores)
    stats["rollout_scores/mean"] = all_scores_mean.item()
    stats["rollout_scores/std"] = all_scores_std.item()
    stats["rollout_scores/running_mean"] = self_obj.running_moments.mean.item()
    stats["rollout_scores/running_std"] = self_obj.running_moments.std.item()

    if config.method.scale_reward == "running":
        scores /= self_obj.running_moments.std
    elif config.method.scale_reward == "ref":
        scores /= self_obj.ref_std

    # Precompute logprobs, values
    if config.model.model_arch_type == "seq2seq":
        attention_mask = batch.attention_mask.to(device)
        prompt_tensors = batch.input_ids.to(device)
        decoder_attention_mask = sample_outputs.not_equal(tokenizer.pad_token_id)
        decoder_attention_mask[:, 0] = 1
        with torch.no_grad():
            outputs = model(
                input_ids=prompt_tensors,
                attention_mask=attention_mask,
                decoder_input_ids=sample_outputs,
                decoder_attention_mask=decoder_attention_mask,
            )
            logits = outputs.logits
            values = outputs.value
            if hasattr(model, "frozen_head"):
                ref_logits = model.forward_hydra(
                    input_ids=prompt_tensors,
                    attention_mask=attention_mask,
                    decoder_input_ids=sample_outputs,
                    decoder_attention_mask=decoder_attention_mask,
                    return_dict=True,
                ).logits
            else:
                ref_logits = ref_model(
                    input_ids=prompt_tensors,
                    attention_mask=attention_mask,
                    decoder_input_ids=sample_outputs,
                    decoder_attention_mask=decoder_attention_mask,
                    return_dict=True,
                ).logits
    else:
        all_tokens = torch.cat((prompt_tensors.to(device), sample_outputs), dim=1)
        attention_mask = all_tokens.not_equal(tokenizer.pad_token_id).long().to(device)
        with torch.no_grad():
            logits, *_, values = model(
                all_tokens,
                attention_mask=attention_mask,
            )
            # TODO(dahoas): When hydra model works need to also support generation on hydra head
            if hasattr(model, "frozen_head"):
                ref_logits = model.forward_hydra(
                    all_tokens,
                    attention_mask=attention_mask,
                    return_dict=True,
                ).logits
            else:
                ref_logits = ref_model(
                    all_tokens,
                    attention_mask=attention_mask,
                    return_dict=True,
                ).logits
                ref_logits = ref_logits.to(device)

    if config.model.model_arch_type == "seq2seq":
        logprobs = logprobs_of_labels(logits[:, :-1, :], sample_outputs[:, 1:])
        ref_logprobs = logprobs_of_labels(ref_logits[:, :-1, :], sample_outputs[:, 1:])
    else:
        logprobs = logprobs_of_labels(logits[:, :-1, :], all_tokens[:, 1:])
        ref_logprobs = logprobs_of_labels(ref_logits[:, :-1, :], all_tokens[:, 1:])

    n_samples: int = samples.shape[0]

    # Estimate the KL divergence between the model and reference model
    if config.model.model_arch_type == "seq2seq":
        attention_mask = sample_outputs != tokenizer.pad_token_id
        start = 0
    else:
        start = prompt_tensors.shape[1] - 1

    log_ratio = (logprobs - ref_logprobs) * attention_mask[:, :-1]
    kl = log_ratio.exp() - 1 - log_ratio
    mean_kl_per_token = kl.mean()
    mean_kl = kl.sum(1).mean()

    logprobs = logprobs.cpu()
    ref_logprobs = ref_logprobs.cpu()
    prompt_tensors = prompt_tensors.cpu()
    sample_outputs = sample_outputs.cpu()
    values = values.cpu()[:, :-1]

    # Get the logprobs and values, for tokens that are not padding,
    # from the start of the prompt up to the <eos> token, while also including the latter
    # (these are taken from the student model and not the reference model)
    ends = start + attention_mask[:, start:].sum(1) + 1
    all_values = [values[ix, start : ends[ix]] for ix in range(n_samples)]
    all_logprobs = [logprobs[ix, start : ends[ix]] for ix in range(n_samples)]

    kl_penalty = self_obj.kl_ctl.value * -log_ratio.cpu()
    kl_penalty = [xs[start : ends[ix]] for ix, xs in enumerate(kl_penalty)]

    rollout_count = 0

    for sample_idx in range(n_samples):
        rewards = kl_penalty[sample_idx]
        rewards[-1] += scores[sample_idx].cpu()

        ppo_rl_elements.append(
            PPORLElement(
                query_tensor=prompt_tensors[sample_idx],
                response_tensor=sample_outputs[sample_idx],
                logprobs=all_logprobs[sample_idx],
                values=all_values[sample_idx],
                rewards=rewards,
            )
        )

        rollout_count += 1

    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(mean_kl, torch.distributed.ReduceOp.AVG)

    stats["time/rollout_time"] = clock.tick()
    stats["policy/sqrt_kl"] = torch.sqrt(mean_kl).item()
    stats["policy/kl_per_token"] = torch.sqrt(mean_kl_per_token).item()
    accumulated_stats.append(stats)

    tbar.set_description(f"[rollout {len(ppo_rl_elements)} / {num_rollouts}]")
    tbar.update(min(rollout_count, num_rollouts))



@register_trainer
class PipelinedPPOTrainer(AcceleratePPOTrainer):
    """Pipelined PPO Accelerate Trainer"""

    def __init__(self, config: TRLConfig, **kwargs):
        print("Here!!")
        super().__init__(config, **kwargs)
       
    def make_experience(self, num_rollouts: int = 1024, iter_count: int = 0):  # noqa:
        """Make experiences

        Takes `chunk_size` number of prompts from `prompt_iterator`, samples
        from the model and then computes the KL against a reference model. Finally it
        then appends PPOElements to trainer's `store`.

        Args:
            num_rollouts: Number of rollouts to generate
            iter_count: Total number of updates run (i.e. number of updates run for all batches & epochs)
        """
        t_start = time()
        logger.info("Collecting rollouts")
        tbar = logging.tqdm(
            total=num_rollouts,
            disable=os.environ.get("RANK", 0) != "0",
            desc=f"[rollout 0 / {num_rollouts}]",
            # Lower progress bar by 1 if we're in WARNING mode or above to avoid hiding high priority progress
            # bars (e.g. loss progress in trainers)
            position=logging.get_verbosity() >= logging.WARNING,
            # Leave progress bar if we're in INFO mode or lower to avoid spamming in suppressed verbosity levels
            leave=logging.get_verbosity() < logging.WARNING,
        )

        clock = Clock()
        ppo_rl_elements = []
        accumulated_stats = []

        while len(ppo_rl_elements) < num_rollouts:
            stats = {}

            metadata = gather_dict({k: v for k, v in batch.items() if k != "input_ids" and k != "attention_mask"})

            if self.accelerator.is_main_process:
                all_str_samples, all_str_prompts, all_str_outputs = self.decode(
                    gathered_prompts, gathered_samples, gathered_prompt_sizes, append_eos_token=True
                )

                rollout_score_time = time()
                all_scores = torch.tensor(
                    self.reward_fn(
                        samples=all_str_samples, prompts=all_str_prompts, outputs=all_str_outputs, **metadata
                    ),
                    dtype=torch.float,
                    device=device,
                )
                stats["time/rollout_score"] = time() - rollout_score_time

                all_scores = list(all_scores.reshape(self.accelerator.num_processes, -1).unbind())
            else:
                all_scores = None

            
        tbar.close()

        stats = {k: sum([xs[k] for xs in accumulated_stats]) / len(accumulated_stats) for k in stats}
        stats["kl_ctl_value"] = self.kl_ctl.value
        self.mean_kl = stats["policy/sqrt_kl"] ** 2
        self.accelerator.log(stats, step=iter_count)

        # Push samples and rewards to trainer's rollout storage
        self.push_to_store(ppo_rl_elements)
        t_end = time()
        print("***** Rollout cost:", t_end - t_start, "*****")
