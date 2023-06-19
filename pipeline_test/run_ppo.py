import typer
import os
import torch
import logging

from time import sleep
from pipeline_test.model import FakeRewardModel
from pipeline_test.trainer import FakePPOTrainer


NUM_FEATURES = 3


def create_reward_fn():
    if os.environ.get("RANK") == '0':
        reward_model = FakeRewardModel(NUM_FEATURES)
        reward_device = torch.cuda.device_count() - 1
        reward_model = reward_model.to(reward_device)
        def get_reward(samples):
            sleep(1)
            return reward_model(samples.to(reward_device))
        return get_reward
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
