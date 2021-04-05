import datetime
import os
import pickle

import torch
from dotmap import DotMap
from rllib.dataset.transforms import ActionScaler, MeanFunction, DeltaState
from rllib.model import AbstractModel
from rllib.reward.utilities import tolerance
from tqdm import tqdm
import numpy as np
from exps import get_mpc_agent, get_mb_mpo_agent
from exps.inverted_pendulum.util import PendulumReward, StateTransform, get_mbmpo_parser
from curriculum_experiments.const_task_teacher import ConstTaskTeacher
from curriculum_experiments.parametric_pendulum_continuous import PendulumContinuousWrapper, \
    large_state_termination
from curriculum_experiments.random_teacher import RandomTeacher
from curriculum_experiments.task_difficulty_estimate import estimate_task_difficulties


def run_on_parameteric_env(steps_per_task, tasks, wrapper, easy_task, state_func, folder, desc):
    teacher = ConstTaskTeacher({"params": easy_task}, wrapper)
    ref_env = wrapper.create_env(easy_task)

    parser = get_mbmpo_parser()
    parser.set_defaults(
        action_cost=0.1,
        train_episodes=20,
        environment_max_steps=steps_per_task,
        plan_horizon=8,
        sim_num_steps=steps_per_task,
        sim_initial_states_num_trajectories=8,
        sim_initial_dist_num_trajectories=8,
        model_kind="ProbabilisticEnsemble",
        model_learn_num_iter=21,
        model_opt_lr=1e-3,
        seed=1,
        plot_train_results=False,
        render_train=False,
        render_test=False,
    )
    args = parser.parse_args()
    params = DotMap(vars(args))

    student = get_mb_mpo_agent(
        ref_env.observation_space.shape,
        ref_env.action_space.shape,
        params,
        state_func,
        action_scale=ref_env.action_scale,
        transformations=[
            ActionScaler(scale=ref_env.action_scale),
            MeanFunction(DeltaState()),  # AngleWrapper(indexes=[1])
        ],
        input_transform=StateTransform(),
        termination_model=large_state_termination,
        initial_distribution=torch.distributions.Uniform(
            torch.tensor([-np.pi, -0.005]), torch.tensor([np.pi, +0.005])
        ),
    )

    for i in tqdm(range(tasks)):
        teacher.train_k_actions(student, steps_per_task)

    # difficulty_estimates, task_params = estimate_task_difficulties(student, wrapper, 5, 3, steps_per_task)

    # with open(f"./results/{date_string}/difficulty/{wrapper.name}/data_all.pkl", "wb") as fptr:
    #     pickle.dump((difficulty_estimates, task_params), fptr)
    with open(f"{folder}/hist_{desc}.pkl", "wb") as fptr:
        pickle.dump(teacher.history.history, fptr)
    student.save(f"{folder}/student_{desc}.agent", directory=".")


class TolerantPendulumReward(AbstractModel):
    """Reward for Inverted Pendulum."""

    def __init__(self, action_cost=0):
        super().__init__(dim_state=(2,), dim_action=(1,), model_kind="rewards")
        self.action_cost = action_cost
        self.reward_offset = 0

    def forward(self, state, action, next_state):
        """See `abstract_reward.forward'."""
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.get_default_dtype())
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, dtype=torch.get_default_dtype())

        cos_angle = torch.cos(state[..., 0])
        velocity = state[..., 1]

        angle_tolerance = tolerance(cos_angle, lower=0.8, upper=1.0, margin=0.2)
        velocity_tolerance = tolerance(velocity, lower=-1.0, upper=1.0, margin=1.0)
        state_cost = angle_tolerance * velocity_tolerance

        action_tolerance = tolerance(action[..., 0], lower=-0.2, upper=0.2, margin=0.2)
        action_cost = self.action_cost * (action_tolerance - 1)

        cost = state_cost + action_cost

        return cost.unsqueeze(-1), torch.zeros(1)


if __name__ == "__main__":
    easy_params = {
        "mass": 0.5,
        "length": 0.5,
        "friction": 0.001,
        "gravity": 10
    }
    date_string = datetime.datetime.today().strftime('%Y-%m-%d %H')
    folder_name = f"./results/{date_string}/model-sanity/Pendulum-v2"
    os.makedirs(folder_name, exist_ok=True)
    run_on_parameteric_env(400, 50, PendulumContinuousWrapper(), easy_params, PendulumReward(action_cost=0.1),
                           folder_name, "normal1")
    run_on_parameteric_env(400, 50, PendulumContinuousWrapper(), easy_params, PendulumReward(action_cost=0.1),
                           folder_name, "normal2")
    run_on_parameteric_env(400, 50, PendulumContinuousWrapper(), easy_params, TolerantPendulumReward(action_cost=0.1),
                           folder_name, "tolerant")
