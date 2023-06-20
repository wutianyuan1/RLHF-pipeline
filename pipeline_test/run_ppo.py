import typer
import os
import torch
import logging
import numpy as np
from multiprocessing.shared_memory import SharedMemory
import torch.multiprocessing as multiprocessing

from time import sleep
from functools import partial
from pipeline_test.model import FakeRewardModel
from pipeline_test.trainer import FakePPOTrainer
from pipeline_test.shmqueue import ShmQueue


NUM_FEATURES = 3
NUM_PROCESSES = 7


class RewardProcess(multiprocessing.Process):
    def __init__(self,
                 in_queue: ShmQueue,
                 out_queue: ShmQueue):
        super().__init__()
        self.in_queue = in_queue
        self.out_queue = out_queue

    def run(self):
        reward_model = FakeRewardModel(NUM_FEATURES)
        reward_device = torch.cuda.device_count() - 1
        reward_model = reward_model.to(reward_device)
        while True:
            try:
                print("Waiting for data")
                data = self.in_queue.get()
                print(f"get data: {data}")
                samples = torch.tensor(data['samples'], device=reward_device)
                all_scores = reward_model(samples)
                out_data = {
                    "reward": all_scores.detach().cpu().numpy(),
                    "iter": np.array(data['iter'], dtype=np.int32)
                }
                print(f"output reward data: {out_data}")
                self.out_queue.put(out_data)
            except KeyboardInterrupt:
                break


def create_reward_fn():
    if os.environ.get("RANK") == '0':
        multiprocessing.set_start_method('spawn')
        in_mem = SharedMemory(create=True, size=100 * 2**20)
        in_data_specs = {
            "samples": np.zeros((NUM_PROCESSES, 3), dtype=np.float32),
            "iter": np.zeros(1, dtype=np.int32),
        }
        out_mem = SharedMemory(create=True, size=100 * 2**20)
        out_data_specs = {
            "reward": np.zeros((NUM_PROCESSES, 1), dtype=np.float32),
            "iter": np.zeros(1, dtype=np.int32),
        }
        in_queue = ShmQueue(10, in_mem, in_data_specs)
        out_queue = ShmQueue(10, out_mem, out_data_specs)
        reward_proc = RewardProcess(in_queue, out_queue)
        reward_proc.start()
        return (reward_proc, in_queue, out_queue)
    else:
        return True


def fake_train(reward_fn):
    trainer = FakePPOTrainer(reward_fn, NUM_FEATURES)
    trainer.fake_train()


app = typer.Typer()
@app.command()
def main():
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(filename=f"logs/output_rank{os.environ.get('RANK')}.log",
                        filemode='w')
    reward_fn = create_reward_fn()
    fake_train(reward_fn)


if __name__ == '__main__':
    app()
