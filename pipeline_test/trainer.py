import torch
import logging
import os

from typing import Callable
from accelerate import Accelerator
from pipeline_test.model import FakeLLM


class FakePPOTrainer():
    def __init__(self, reward_fn: Callable, num_features: int):
        self.reward_fn = reward_fn
        self.num_features = num_features
        self.accelerator = Accelerator()

        if self.accelerator.state.deepspeed_plugin is not None:
            # by accelerate's default, arguments in `model.forward` would be casted to half
            if "fp16" in self.accelerator.state.deepspeed_plugin.deepspeed_config:
                self.accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["auto_cast"] = False

        self.accelerator.init_trackers(project_name="test_proj")
        self.model = FakeLLM(sent_len=self.num_features, device=self.accelerator.device).to(self.accelerator.device)
    
    def fake_postforward(self, samples, reward):
        pi_1 = self.model(samples)
        pi_2 = self.model(samples)
        return reward + pi_1/pi_2

    def make_experience(self):
        samples = self.model.generate(batch_size=1)
        logging.info(f"Rollout get: {samples}")
        all_samples = self.accelerator.gather(samples)
        logging.info(f"Gathered all_samples: {all_samples}")
        all_samples = all_samples.reshape((self.accelerator.num_processes, self.num_features))
        if self.accelerator.is_main_process:
            all_scores = self.reward_fn(all_samples)
            logging.info(f"All rewards: {all_scores}")
            all_scores = list(all_scores.reshape(self.accelerator.num_processes, -1).unbind())
        else:
            all_scores = None
        if torch.distributed.is_initialized():
            my_scores = torch.empty(len(samples), device=self.accelerator.device)
            torch.distributed.scatter(my_scores, all_scores)
        logging.info(f"Rank {os.environ.get('RANK')}: my_reward={my_scores}")
        final_reward = self.fake_postforward(samples, my_scores)
        logging.info(f"Rank {os.environ.get('RANK')}: final ret={final_reward}")
    
    def fake_train(self):
        logging.info(f'Start training on device: {self.accelerator.device}')
        for _ in range(1):
            self.make_experience()
