from typing import Any, Dict

import torch
from rllib.environment import SystemEnvironment
from rllib.environment.systems import InvertedPendulum

from exps.inverted_pendulum.util import PendulumReward
from curriculum_experiments.environment_parameter import ContinuousParameter
from curriculum_experiments.environment_wrapper import EnvironmentWrapper
import numpy as np


def large_state_termination(state, action, next_state=None):
    """Termination condition for environment."""
    if not isinstance(state, torch.Tensor):
        state = torch.tensor(state)
    if not isinstance(action, torch.Tensor):
        action = torch.tensor(action)

    done = torch.any(torch.abs(state) > 200, dim=-1) | torch.any(
        torch.abs(action) > 200, dim=-1
    )
    return (
        torch.zeros(*done.shape, 2)
            .scatter_(dim=-1, index=(~done).long().unsqueeze(-1), value=-float("inf"))
            .squeeze(-1)
    )


class PendulumContinuousWrapper(EnvironmentWrapper):
    def __init__(self):
        super().__init__()
        self.name = "Pendulum-v2"
        self.parameters = {
            "mass": ContinuousParameter(0.1, 3),
            "length": ContinuousParameter(0.1, 5),
            "friction": ContinuousParameter(0.001, 0.1),
            "gravity": ContinuousParameter(5, 20),
        }

    def create_env(self, parameter_values: Dict[str, Any]):
        initial_distribution = torch.distributions.Uniform(
            torch.tensor([np.pi, -0.0]), torch.tensor([np.pi, +0.0])
        )
        reward_model = PendulumReward(action_cost=0.1)
        environment = SystemEnvironment(
            InvertedPendulum(mass=parameter_values["mass"],
                             length=parameter_values["length"],
                             friction=parameter_values["friction"],
                             step_size=1 / 80,
                             gravity=parameter_values["gravity"]),
            reward=reward_model,
            initial_state=initial_distribution.sample,
            termination_model=large_state_termination,
        )
        environment.reset()
        return environment
