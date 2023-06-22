import multiprocessing
import torch
import math
import os
import numpy as np

from torch import nn
from time import time
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from pipeline.shmqueue import ShmQueue
from multiprocessing.shared_memory import SharedMemory


class RewardModel(nn.Module):
    def __init__(self, checkpoint_path, eos_token_id):
        super().__init__()
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
        self.transformer = model.transformer
        self.v_head = nn.Linear(model.config.n_embd, 1, bias=False)
        self.eos_token_id = eos_token_id

    def forward(self, input_ids):
        states = self.transformer(input_ids)[0]
        rewards = self.v_head(states).squeeze(-1)
        ends = torch.argmax((input_ids == self.eos_token_id).float(), dim=1).view(-1, 1)
        returns = torch.gather(rewards, 1, ends).squeeze(-1)
        return returns


class RewardProcess(multiprocessing.Process):
    def __init__(self,
                 reward_model: RewardModel,
                 reward_tokenizer: AutoTokenizer,
                 reward_device: torch.device,
                 reward_batch_size: int,
                 delta_reward: bool,
                 in_queue: ShmQueue,
                 out_queue: ShmQueue):
        super().__init__()
        self.reward_model = reward_model
        self.reward_tokenizer = reward_tokenizer
        self.reward_device = reward_device
        self.reward_batch_size = reward_batch_size
        self.delta_reward = delta_reward
        self.in_queue = in_queue
        self.out_queue = out_queue
    
    def get_reward(self, samples):
        input = self.reward_tokenizer(
            samples,
            padding=True,
            truncation=True,
            max_length=self.reward_tokenizer.max_len_single_sentence,
            return_tensors="pt",
        ).to(self.reward_device)

        mbs = self.reward_batch_size
        out = []
        for i in range(math.ceil(len(samples) / mbs)):
            batch_ixs = slice(i * mbs, (i + 1) * mbs)
            input_ids = input.input_ids[batch_ixs]
            rewards = self.reward_model(input_ids)
            out.extend(rewards)
        return torch.hstack(out)

    def reward_fn(self, samples, prompts, original_output, **kwargs):
        samples = [s + self.reward_tokenizer.eos_token for s in samples]
        rewards = self.get_reward(samples)

        if not self.delta_reward:
            return rewards

        original_samples = [p + o + self.reward_tokenizer.eos_token for p, o in zip(prompts, original_output)]
        original_rewards = self.get_reward(original_samples)
        return rewards - original_rewards
    
    def run(self):
        logfile = open('rewproc.log', 'w')
        while True:
            try:
                print("waiting for data...", file=logfile, flush=True)
                in_data = self.in_queue.get()
                s_t = time()
                print("recv time:", s_t, file=logfile, flush=True)
                samples = in_data['samples']
                prompts = in_data['prompts']
                original_output = in_data['original_output']
                rewards = self.reward_fn(samples, prompts, original_output)
                print("reward:", rewards, rewards.shape, file=logfile, flush=True)
                out_data = {
                    "rewards": rewards.detach().cpu().numpy(),
                }
                e_t = time()
                print("reward calculation time:", e_t - s_t, file=logfile, flush=True)
                self.out_queue.put(out_data)
            except KeyboardInterrupt:
                break
        self.in_queue.shm.close()
        self.out_queue.shm.close()


def create_reward_fn():  # noqa:  C901
    reward_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    reward_tokenizer.pad_token = reward_tokenizer.eos_token
    reward_tokenizer.truncation_side = "left"

    if os.environ.get("RANK", "0") == "0":
        reward_model = RewardModel("EleutherAI/gpt-j-6B", reward_tokenizer.eos_token_id)
        directory = snapshot_download("Dahoas/gptj-rm-static", revision="676bfd4d")
        for fpath in os.listdir(directory):
            if fpath.endswith(".pt") or fpath.endswith(".bin"):
                checkpoint = os.path.join(directory, fpath)
                break

        reward_model.load_state_dict(torch.load(checkpoint))
        reward_model.eval()
        reward_model.requires_grad_(False)
        reward_device = torch.cuda.device_count() - 1
        reward_model = reward_model.half().to(reward_device)
        reward_batch_size = 48
        delta_reward = True

        multiprocessing.set_start_method('spawn', force=True)
        total_items = 7 * 16  # nprocs * batch_size
        in_specs = {
            "samples": np.array(['a' for _ in range(total_items)], dtype='U5000'),
            "prompts": np.array(['a' for _ in range(total_items)], dtype='U5000'),
            "original_output": np.array(['a' for _ in range(total_items)], dtype='U5000'),
        }
        out_specs = {
            "rewards": np.zeros((16 * 7,), dtype=np.float16),
        }
        in_shm = SharedMemory("in_shm", create=True, size=100 * 2**20)
        in_queue = ShmQueue(10, in_shm, in_specs)
        out_shm = SharedMemory("out_shm", create=True, size=100 * 2**20)
        out_queue = ShmQueue(10, out_shm, out_specs)

        reward_process = RewardProcess(
            reward_model, reward_tokenizer, reward_device, reward_batch_size,
            delta_reward, in_queue, out_queue)
        reward_process.start()
        reward_specs = (reward_process, in_queue, out_queue)
    else:
        reward_specs = True

    return reward_specs
