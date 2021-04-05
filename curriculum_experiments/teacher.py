from abc import ABC, abstractmethod
import numpy as np

from typing import Dict, Tuple, Any

from rllib.util.training.agent_training import train_agent

from curriculum_experiments.history import History

rews = []
dones = []


def my_callback(agent, environment, episode: int):
    global rews, dones
    rews = [o.reward.item() for o in agent.last_trajectory]
    dones = [o.done.item() for o in agent.last_trajectory]


class Teacher(ABC):
    def __init__(self, teacher_parameters, environment_wrapper, history_parameters=None, seed=None):
        self.history = History(history_parameters)
        self.env_wrapper = environment_wrapper
        self.seed = seed
        self.eval_data = []

    def train_k_actions(self, student, action_limit: int, eval_task_params=None, pretrain=False):
        training_task, params = self.generate_task()
        trajectory, rewards, dones = self.train_student_on_task(student, training_task, action_limit, eval_task_params,
                                                                pretrain)
        self.history.update(params, trajectory, rewards, dones)
        self.update_teacher_policy()

    def train_student_on_task(self, student, training_task, action_limit, eval_task_params=None, pretrain=False):
        train_agent(student, environment=training_task, callbacks=[my_callback], plot_flag=False, callback_frequency=1)

        # return the trajectory and rewards
        global rews, dones
        return (None,), rews, dones

    def evaluate(self, action_limit, eval_task_params, student):
        eval_env = self.env_wrapper.create_env(eval_task_params)
        student.set_env(eval_env)
        s = eval_env.reset()
        total_reward = 0
        episode_length = 0
        for i in range(action_limit):
            a, _ = student.predict(observation=s, deterministic=True)
            s, r, d, _ = eval_env.step(a)
            episode_length += 1
            total_reward += r
            if d == 1:
                break
        return total_reward, episode_length

    @abstractmethod
    def generate_task(self):
        pass

    @abstractmethod
    def update_teacher_policy(self):
        pass
