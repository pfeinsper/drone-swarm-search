import os
import pathlib
from DSSE import DroneSwarmSearch
from DSSE.environment.wrappers import TopNProbsWrapper
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
import torch
from torch import nn


class MLPModel(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space,
        act_space,
        num_outputs,
        model_config,
        name,
        **kw,
    ):
        print("OBSSPACE: ", obs_space)
        TorchModelV2.__init__(
            self, obs_space, act_space, num_outputs, model_config, name, **kw
        )
        nn.Module.__init__(self)

        self.model = nn.Sequential(
            nn.Linear(obs_space.shape[0], 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.policy_fn = nn.Linear(256, num_outputs)
        self.value_fn = nn.Linear(256, 1)

    def forward(self, input_dict, state, seq_lens):
        input_ = input_dict["obs"].float()
        value_input = self.model(input_)
        
        self._value_out = self.value_fn(value_input)
        return self.policy_fn(value_input), state

    def value_function(self):
        return self._value_out.flatten()


def env_creator(args):
    env = DroneSwarmSearch(
        drone_amount=16,
        grid_size=40,
        dispersion_inc=0.05,
        person_initial_position=(10, 10),
    )
    env = TopNProbsWrapper(env, 10)
    return env


if __name__ == "__main__":
    ray.init()

    env_name = "DSSE"

    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
    ModelCatalog.register_custom_model("MLPModel", MLPModel)

    config = (
        PPOConfig()
        .environment(env=env_name)
        .rollouts(num_rollout_workers=6, rollout_fragment_length="auto")
        .training(
            train_batch_size=8192,
            lr=1e-5,
            gamma=0.99999,
            lambda_=0.9,
            use_gae=True,
            clip_param=0.4,
            grad_clip=None,
            entropy_coeff=0.1,
            vf_loss_coeff=0.25,
            vf_clip_param=4200,
            sgd_minibatch_size=1024,
            num_sgd_iter=10,
            model={
                "custom_model": "MLPModel",
                "_disable_preprocessor_api": True,
            },
        )
        .experimental(_disable_preprocessor_api=True)
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "1")))
    )

    curr_path = pathlib.Path().resolve()
    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 10_000_000 if not os.environ.get("CI") else 50000},
        checkpoint_freq=10,
        storage_path=f"{curr_path}/ray_res/" + env_name,
        config=config.to_dict(),
    )
