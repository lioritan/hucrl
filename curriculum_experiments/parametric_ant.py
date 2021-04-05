from typing import Any, Dict

import torch
from rllib.environment import SystemEnvironment
from rllib.environment.systems import InvertedPendulum

from exps.inverted_pendulum.util import PendulumReward
from curriculum_experiments.environment_parameter import ContinuousParameter
from curriculum_experiments.environment_wrapper import EnvironmentWrapper
import numpy as np

import gym.error
from gym.envs.mujoco.ant_v3 import AntEnv
from rllib.environment.mujoco.locomotion import LargeStateTermination, LocomotionEnv


class MBAntEnv(LocomotionEnv, AntEnv):
    """Ant Environment."""

    def __init__(self, ctrl_cost_weight=0.1, goal_x=0.0, goal_y=0.0):
        self.base_mujoco_name = "Ant-v3"
        self.goal = np.array([goal_x, goal_y])
        LocomotionEnv.__init__(
            self,
            dim_pos=2,
            dim_action=(8,),
            ctrl_cost_weight=ctrl_cost_weight,
            forward_reward_weight=1.0,
            healthy_reward=1.0,
        )
        AntEnv.__init__(
            self,
            ctrl_cost_weight=ctrl_cost_weight,
            contact_cost_weight=0.0,
            healthy_reward=1.0,
            terminate_when_unhealthy=True,
        )
        self._termination_model = LargeStateTermination(
            z_dim=2, healthy_z_range=self._healthy_z_range
        )

    def step(self, action):
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        forward_reward = x_velocity
        healthy_reward = self.healthy_reward

        # Note: add goal reward
        goal_reward = -np.sum(np.abs(xy_position_after - self.goal)) + 4.0  # make it happy, not suicidal

        rewards = forward_reward + healthy_reward + goal_reward
        costs = ctrl_cost + contact_cost

        reward = rewards - costs
        done = self.done
        observation = self._get_obs()
        info = {
            'reward_forward': forward_reward,
            'reward_ctrl': -ctrl_cost,
            'reward_contact': -contact_cost,
            'reward_survive': healthy_reward,

            'x_position': xy_position_after[0],
            'y_position': xy_position_after[1],
            'distance_from_origin': np.linalg.norm(xy_position_after, ord=2),

            'x_velocity': x_velocity,
            'y_velocity': y_velocity,
            'forward_reward': forward_reward,
        }

        return observation, reward, done, info


class AntWrapper(EnvironmentWrapper):
    def __init__(self):
        super().__init__()
        self.name = "MBAnt-v0"
        self.parameters = {
            "goal_x": ContinuousParameter(-3.0, 3.0),
            "goal_y": ContinuousParameter(-3.0, 3.0),

        }

    def create_env(self, parameter_values: Dict[str, Any]):
        environment = MBAntEnv(goal_x=parameter_values["goal_x"], goal_y=parameter_values["goal_y"])
        return environment
