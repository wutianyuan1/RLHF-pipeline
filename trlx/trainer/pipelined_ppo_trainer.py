import os
import torch
import torch.nn.functional as F
import numpy as np
import trlx.utils.logging as logging

from time import time
from trlx.data.accelerate_base_datatypes import PromptBatch
from trlx.data.configs import TRLConfig
from trlx.data.ppo_types import PPORLElement
from trlx.trainer import register_trainer
from trlx.trainer.accelerate_ppo_trainer import AcceleratePPOTrainer
from trlx.utils import Clock
from trlx.utils.modeling import logprobs_of_labels


logger = logging.get_logger(__name__)


@register_trainer
class PipelinedPPOTrainer(AcceleratePPOTrainer):
    """Pipelined PPO Accelerate Trainer"""

    def __init__(self, config: TRLConfig, **kwargs):
        super().__init__(config, **kwargs)
        if self.accelerator.is_main_process:
            self.reward_proc, self.in_queue, self.out_queue = self.reward_fn
    
    def make_experience_rollout(self, batch, stats):
        rollout_generate_time = time()
        # Generate samples from the language model (similar to using HuggingFace `generate` method)
        samples = self.generate(batch["input_ids"], batch["attention_mask"])
        stats["time/rollout_generate"] = time() - rollout_generate_time

        prompt_tensors = batch.input_ids
        device = samples.device

        prompt_sizes = torch.tensor([prompt_tensors.shape[1]] * len(prompt_tensors), device=device)
        padded_samples = self.accelerator.pad_across_processes(
            samples, dim=1, pad_index=self.tokenizer.eos_token_id, pad_first=False
        )
        padded_prompts = self.accelerator.pad_across_processes(
            prompt_tensors, dim=1, pad_index=self.tokenizer.eos_token_id, pad_first=False
        )
        gathered_samples = self.accelerator.gather(padded_samples)
        gathered_prompts = self.accelerator.gather(padded_prompts)
        gathered_prompt_sizes = self.accelerator.gather(prompt_sizes)

        if self.accelerator.is_main_process:
            all_str_samples, all_str_prompts, all_str_outputs = self.decode(
                gathered_prompts, gathered_samples, gathered_prompt_sizes, append_eos_token=True
            )
            send_data = {
                "samples": np.array(all_str_samples, dtype='U5000'),
                "prompts": np.array(all_str_prompts, dtype='U5000'),
                "original_output": np.array(all_str_outputs, dtype='U5000'),
            }
            self.in_queue.put(send_data)
        return samples, prompt_tensors

    def make_experience_postprocess(self, prompt_tensors, samples, stats, start_clock):
        assert self.config.model.model_arch_type != "seq2seq"

        device = samples.device

        if self.accelerator.is_main_process:
            all_scores = self.out_queue.get()
            all_scores = torch.tensor(all_scores['rewards'], dtype=torch.float, device=self.accelerator.device)
            all_scores = list(all_scores.reshape(self.accelerator.num_processes, -1).unbind())
        else:
            all_scores = None

        scores = torch.empty(len(samples), device=device)
        torch.distributed.scatter(scores, all_scores)
        
        cur_iter_ppo_rl_elements = []
        str_samples, str_prompts, str_outputs = self.decode(prompt_tensors, samples, append_eos_token=True)
        # Pad the sample outputs
        outputs = self.tokenizer(str_outputs).input_ids
        outputs = list(map(torch.LongTensor, outputs))
        maxsize = max(map(len, outputs))
        outputs = [
            F.pad(
                output,
                (0, maxsize - len(output)),
                value=self.tokenizer.pad_token_id,
            )
            for output in outputs
        ]
        sample_outputs = torch.vstack(outputs).to(device)
        if self.config.method.cliprange_reward:
            scores = torch.clip(scores, -self.config.method.cliprange_reward, self.config.method.cliprange_reward)

        # store statistics of the initial rollout as reference
        if self.ref_mean is None:
            self.ref_mean, self.ref_std = scores.mean(), scores.std()
        all_scores_mean, all_scores_std = self.running_moments.update(scores)
        stats["rollout_scores/mean"] = all_scores_mean.item()
        stats["rollout_scores/std"] = all_scores_std.item()
        stats["rollout_scores/running_mean"] = self.running_moments.mean.item()
        stats["rollout_scores/running_std"] = self.running_moments.std.item()

        if self.config.method.scale_reward == "running":
            scores /= self.running_moments.std
        elif self.config.method.scale_reward == "ref":
            scores /= self.ref_std

        # Precompute logprobs, values
        all_tokens = torch.cat((prompt_tensors.to(device), sample_outputs), dim=1)
        attention_mask = all_tokens.not_equal(self.tokenizer.pad_token_id).long().to(device)
        with torch.no_grad():
            logits, *_, values = self.model(
                all_tokens,
                attention_mask=attention_mask,
            )
            # TODO(dahoas): When hydra model works need to also support generation on hydra head
            if hasattr(self.model, "frozen_head"):
                ref_logits = self.model.forward_hydra(
                    all_tokens,
                    attention_mask=attention_mask,
                    return_dict=True,
                ).logits
            else:
                ref_logits = self.ref_model(
                    all_tokens,
                    attention_mask=attention_mask,
                    return_dict=True,
                ).logits
                ref_logits = ref_logits.to(device)


        logprobs = logprobs_of_labels(logits[:, :-1, :], all_tokens[:, 1:])
        ref_logprobs = logprobs_of_labels(ref_logits[:, :-1, :], all_tokens[:, 1:])
        n_samples: int = samples.shape[0]

        # Estimate the KL divergence between the model and reference model
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

        kl_penalty = self.kl_ctl.value * -log_ratio.cpu()
        kl_penalty = [xs[start : ends[ix]] for ix, xs in enumerate(kl_penalty)]

        rollout_count = 0

        for sample_idx in range(n_samples):
            rewards = kl_penalty[sample_idx]
            rewards[-1] += scores[sample_idx].cpu()

            cur_iter_ppo_rl_elements.append(
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

        stats["time/rollout_time"] = start_clock.tick()
        stats["policy/sqrt_kl"] = torch.sqrt(mean_kl).item()
        stats["policy/kl_per_token"] = torch.sqrt(mean_kl_per_token).item()
        return cur_iter_ppo_rl_elements, rollout_count

       
    def make_experience(self, num_rollouts: int = 1024, iter_count: int = 0):  # noqa:
        """Make experiences

        Takes `chunk_size` number of prompts from `prompt_iterator`, samples
        from the model and then computes the KL against a reference model. Finally it
        then appends PPOElements to trainer's `store`.

        Args:
            num_rollouts: Number of rollouts to generate
            iter_count: Total number of updates run (i.e. number of updates run for all batches & epochs)
        """
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

        all_iter_stats = []
        all_iter_samples = []
        all_iter_prompts = []
        n_samples = 0
        while n_samples < num_rollouts:
            stats = {}
            # Get next batch in prompt dataset
            batch: PromptBatch = next(self.prompt_iterator)
            ## TODO: fix device
            samples, prompt_tensors = self.make_experience_rollout(batch, stats)

            all_iter_prompts.append(prompt_tensors)
            all_iter_stats.append(stats)
            all_iter_samples.append(samples)
            n_samples += samples.shape[0]

        torch.distributed.barrier()

        cur_iter = 0
        while len(ppo_rl_elements) < num_rollouts:
            samples = all_iter_samples[cur_iter]
            stats = all_iter_stats[cur_iter]
            prompt_tensors = all_iter_prompts[cur_iter]
            cur_iter_ppo_rl_elements, rollout_count =\
                self.make_experience_postprocess(
                prompt_tensors, samples, stats, clock)

            cur_iter += 1
            ppo_rl_elements += cur_iter_ppo_rl_elements
            accumulated_stats.append(stats)
            tbar.set_description(f"[rollout {len(ppo_rl_elements)} / {num_rollouts}]")
            tbar.update(min(rollout_count, num_rollouts))
        tbar.close()

        stats = {k: sum([xs[k] for xs in accumulated_stats]) / len(accumulated_stats) for k in stats}
        stats["kl_ctl_value"] = self.kl_ctl.value
        self.mean_kl = stats["policy/sqrt_kl"] ** 2
        self.accelerator.log(stats, step=iter_count)

        # Push samples and rewards to trainer's rollout storage
        self.push_to_store(ppo_rl_elements)
