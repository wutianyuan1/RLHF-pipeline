import torch
import logging
import os
import numpy as np

from accelerate import Accelerator
from pipeline.model import FakeLLM
from pipeline.shmqueue import ShmQueue
from time import time


class FakePPOTrainer():
    def __init__(self, reward_spec, num_features: int, start_t: float):
        if isinstance(reward_spec, tuple):
            self.reward_proc, self.in_queue, self.out_queue = reward_spec
            self.in_queue: ShmQueue = self.in_queue
            self.out_queue: ShmQueue = self.out_queue
        else:
            self.reward_fn = reward_spec
        self.num_features = num_features
        self.accelerator = Accelerator()

        if self.accelerator.state.deepspeed_plugin is not None:
            # by accelerate's default, arguments in `model.forward` would be casted to half
            if "fp16" in self.accelerator.state.deepspeed_plugin.deepspeed_config:
                self.accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["auto_cast"] = False

        self.accelerator.init_trackers(project_name="test_proj")
        self.model = FakeLLM(sent_len=self.num_features, device=self.accelerator.device).to(self.accelerator.device)
        self.start_t = start_t
    
    def fake_postforward(self, samples, reward):
        pi_1 = self.model(samples)
        pi_2 = self.model(samples)
        return reward + pi_1/pi_2

    def make_experience_rollout(self, cur_iter):
        samples = self.model.generate(batch_size=1)
        logging.info(f"Rollout get: {samples}, t={time() - self.start_t}")
        all_samples = self.accelerator.gather(samples)
        logging.info(f"Gathered all_samples: {all_samples}, t={time() - self.start_t}")
        all_samples = all_samples.reshape((self.accelerator.num_processes, self.num_features))
        if self.accelerator.is_main_process:
            # all_scores = self.reward_fn(all_samples)
            data = {
                "samples": all_samples.cpu().numpy(),
                "iter": np.array([cur_iter], dtype=np.int32),
            }
            logging.info(f"sending {data}, t={time() - self.start_t}")
            self.in_queue.put(data)

    def make_experience_calc_reward(self):
        if self.accelerator.is_main_process:
            # if self.out_queue.empty():
            #     all_scores = torch.zeros(self.accelerator.num_processes, 
            #                              dtype=torch.float32,
            #                              device=self.accelerator.device)
            #     logging.info("Empty queue")
            # else:
            all_scores = self.out_queue.get()
            all_scores = torch.tensor(all_scores['reward'], device=self.accelerator.device)
            logging.info(f"All rewards: {all_scores}, t={time() - self.start_t}")
            all_scores = list(all_scores.reshape(self.accelerator.num_processes, -1).unbind())
        else:
            all_scores = None

        assert torch.distributed.is_initialized()
        my_scores = torch.empty(1, device=self.accelerator.device)
        torch.distributed.scatter(my_scores, all_scores)

        logging.info(f"Rank {os.environ.get('RANK')}: my_reward={my_scores}, t={time() - self.start_t}")
        # final_reward = self.fake_postforward(samples, my_scores)
        # logging.info(f"Rank {os.environ.get('RANK')}: final ret={final_reward}")
    
    def fake_train(self, num_iters):
        logging.info(f'Start training on device: {self.accelerator.device}, t={time() - self.start_t}')
        for i in range(num_iters):
            self.make_experience_rollout(cur_iter=i)
        for _ in range(num_iters):
            self.make_experience_calc_reward()

        # release resources
        if self.accelerator.is_main_process:
            self.reward_proc.terminate()
            self.in_queue.shm.close()
            self.out_queue.shm.close()

