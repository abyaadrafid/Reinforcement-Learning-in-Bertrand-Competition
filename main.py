import sys

sys.path.append("../")

import ray
from ray import tune
from ray.air.config import RunConfig, ScalingConfig
from ray.rllib.policy.policy import PolicySpec
from ray.train.rl import RLTrainer

from environments.DuopolyEnv import DuopolyEnv


def main():
    ray.init()
    rllib_config = {
        "multiagent": {
            "policies": {
                # Use the PolicySpec namedtuple to specify an individual policy:
                "agent0": PolicySpec(
                    policy_class=None,  # infer automatically from Algorithm
                    observation_space=DuopolyEnv(
                        {}
                    ).observation_space,  # infer automatically from env
                    action_space=DuopolyEnv(
                        {}
                    ).action_space,  # infer automatically from env
                    config={"gamma": 0.85},
                ),
                "agent1": PolicySpec(
                    policy_class=None,  # infer automatically from Algorithm
                    observation_space=DuopolyEnv(
                        {}
                    ).observation_space,  # infer automatically from env
                    action_space=DuopolyEnv(
                        {}
                    ).action_space,  # infer automatically from env
                    config={"gamma": 0.85},
                ),
            },
            "policy_mapping_fn": lambda agent_id, **kwargs: agent_id,
        },
    }
    trainer = RLTrainer(
        run_config=RunConfig(stop={"training_iteration": 5}),
        scaling_config=ScalingConfig(num_workers=2, use_gpu=False),
        algorithm="DQN",
        config=rllib_config,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
