import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rllib.util.rollout import rollout_agent
from rllib.util.training.utilities import Evaluate

from curriculum_experiments.environment_parameter import DiscreteParameter
from curriculum_experiments.environment_wrapper import EnvironmentWrapper
from curriculum_experiments.random_teacher import RandomTeacher


def evaluate(action_limit, eval_task_params, env_wrapper, student):
    eval_env = env_wrapper.create_env(eval_task_params)
    with Evaluate(student):
        rollout_agent(
            eval_env,
            student,
            max_steps=action_limit,
            num_episodes=1,
            render=False,
        )
        returns = np.mean(student.logger.get("eval_return")[-1:])
        return returns, None


def estimate_task_difficulties(student, task_wrapper: EnvironmentWrapper, num_segments=5, trials_per_task=3,
                               max_episode_len=200):
    task_estimates = np.zeros((num_segments ** len(task_wrapper.parameters), trials_per_task))
    task_params = []
    free_ind = 0

    param_names = list(task_wrapper.parameters.keys())
    param_ranges = [np.arange(task_wrapper.parameters[n].min_val,
                              task_wrapper.parameters[n].max_val + 1,
                              (task_wrapper.parameters[n].max_val - task_wrapper.parameters[n].min_val) / (
                                          num_segments - 1))
                    for n in param_names]

    indices = np.zeros(len(param_names), dtype=np.int)
    while free_ind < task_estimates.shape[0]:
        chosen_params = [param_ranges[i][indices[i]] for i in range(len(param_names))]
        for i, param_name in enumerate(param_names):
            if isinstance(task_wrapper.parameters[param_name], DiscreteParameter):
                chosen_params[i] = int(round(chosen_params[i]))
        task_params.append(chosen_params)
        for trial_num in range(trials_per_task):
            try:
                total_reward, _ = evaluate(max_episode_len,
                                           {param_names[i]: chosen_params[i] for i in range(len(param_names))},
                                           task_wrapper,
                                           student)
            except:
                print(chosen_params)
                total_reward = 0
            task_estimates[free_ind, trial_num] = total_reward

        free_ind += 1
        indices[-1] = (indices[-1] + 1) % num_segments
        if indices[-1] == 0:
            for i in range(-2, -len(indices) - 1, -1):
                indices[i] = (indices[i] + 1) % num_segments
                if indices[i] != 0:
                    break

    return task_estimates, task_params
