from typing import Any, Dict

import torch
from rllib.environment import SystemEnvironment
from rllib.environment.systems import InvertedPendulum

from curriculum_experiments.environment_parameter import ContinuousParameter
from curriculum_experiments.environment_wrapper import EnvironmentWrapper
import os

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


import gym.error

from rllib.environment.mujoco.locomotion import LocomotionEnv
from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv


class MBHalfCheetahEnv(LocomotionEnv, HalfCheetahEnv):
    """Half-Cheetah Environment."""

    def __init__(self, ctrl_cost_weight=0.1, expected_speed=0.0):
        self.base_mujoco_name = "HalfCheetah-v3"
        self.expected_speed = expected_speed
        LocomotionEnv.__init__(
            self,
            dim_pos=1,
            dim_action=(6,),
            ctrl_cost_weight=ctrl_cost_weight,
            forward_reward_weight=1.0,
            healthy_reward=0.0,
        )
        HalfCheetahEnv.__init__(
            self, ctrl_cost_weight=ctrl_cost_weight, forward_reward_weight=1.0
        )
        self.dim_state=18
        self.dim_action=6
        self.num_states=None
        self.num_actions=None
        self.goal=None

    def step(self, action):
        x_position_before = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]
        x_velocity = ((x_position_after - x_position_before)
                      / self.dt)

        ctrl_cost = self.control_cost(action)

        # Note: Changed to be difference from expected speed
        forward_reward = self._forward_reward_weight * np.abs(x_velocity - self.expected_speed)

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        info = {
            'x_position': x_position_after,
            'x_velocity': x_velocity,
            'expected_velocity': self.expected_speed,

            'reward_run': forward_reward,
            'reward_ctrl': -ctrl_cost
        }

        return observation, reward, done, info


class HalfCheetahWrapper(EnvironmentWrapper):
    def __init__(self):
        super().__init__()
        self.name = "MBHalfCheetah-v0"
        self.parameters = {
            "expected_speed": ContinuousParameter(0.0, 2.0),
        }

    def create_env(self, parameter_values: Dict[str, Any]):
        environment = MBHalfCheetahEnv(expected_speed=parameter_values["expected_speed"])
        return environment
