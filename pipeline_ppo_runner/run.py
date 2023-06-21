import json
import typer
import trlx

from datasets import load_dataset
from itertools import islice
from pathlib import Path
from typing import Optional
from trlx.data.default_configs import TRLConfig
from pipeline_ppo_runner.config import get_default_config
from pipeline_ppo_runner.reward import create_reward_fn


app = typer.Typer()
@app.command()
def main(config_path: Optional[Path]=None):
    # load config from file
    hparams = {}
    if config_path is not None:
        hparams = json.loads(config_path.read_text())
        print(f"Loaded config from file {config_path}")
        print(f"changes: {hparams}")
    default_config = get_default_config()
    config = TRLConfig.update(default_config, hparams)

    # load reward model and datasets
    reward_fn = create_reward_fn()
    dataset = load_dataset("Dahoas/rm-static")
    prompts = [{"prompt": x["prompt"], "original_output": x["chosen"]} for x in dataset["train"]]
    eval_prompts = [{"prompt": x["prompt"], "original_output": x["chosen"]} for x in islice(dataset["test"], 280)]

    # start training
    trlx.train(
        prompts=prompts,
        eval_prompts=eval_prompts,
        reward_fn=reward_fn,
        config=config,
        stop_sequences=["Human:", "human:", "Assistant:", "assistant:"],
    )

if __name__ == "__main__":
    app()
