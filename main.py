import sys

sys.path.append("../")

import ray
from ray import tune
from ray.air.config import RunConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.examples.models.shared_weights_model import TorchSharedWeightsModel
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env

from environments.DuopolyEnv import DuopolyEnv

env_config = {"max_price": 1200, "min_price": 500, "memory_size": 5, "num_seller": 2}


def env_creator(env_config):
    return DuopolyEnv(env_config)


rllib_config = (
    PPOConfig()
    .environment("duopoly_env", disable_env_checking=False)
    .framework("torch")
    .multi_agent(
        policies={
            "policy": PolicySpec(
                observation_space=env_creator(env_config).observation_space,
                action_space=env_creator(env_config).action_space,
                config=PPOConfig.overrides(framework_str="torch"),
            ),
        },
        policy_mapping_fn=lambda agent_id, *args, **kwargs: "policy",
        policies_to_train=["policy"],
    )
)


def main():
    ray.init()
    register_env("duopoly_env", env_creator)

    stop = {"training_iteration": 5}
    results = tune.Tuner(
        "PPO",
        param_space=rllib_config.to_dict(),
        run_config=RunConfig(stop=stop, verbose=1),
    ).fit()


if __name__ == "__main__":
    main()
