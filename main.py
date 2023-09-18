import sys

sys.path.append("../")

import ray
from ray import tune
from ray.air.config import RunConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.policy.policy import PolicySpec
from ray.train.rl import RLTrainer
from ray.tune.registry import register_env

from environments.DuopolyEnv import DuopolyEnv


def main():
    ray.init()
    register_env("duopoly_env", lambda _: DuopolyEnv({}))
    rllib_config = (
        DQNConfig()
        .environment("duopoly_env", disable_env_checking=False)
        .framework("torch")
        .multi_agent(
            policies={
                "policy": PolicySpec(config=DQNConfig.overrides(framework_str="torch")),
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "policy",
            policies_to_train=["policy"],
        )
        .rl_module(
            rl_module_spec=MultiAgentRLModuleSpec(
                module_specs={"policy": SingleAgentRLModuleSpec()}
            ),
        )
    )
    stop = {"training_iteration": 5}
    results = tune.Tuner(
        "DQN",
        param_space=rllib_config.to_dict(),
        run_config=RunConfig(stop=stop, verbose=1),
    ).fit()


if __name__ == "__main__":
    main()
