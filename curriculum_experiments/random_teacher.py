
from typing import Tuple, Dict, Any

from curriculum_experiments.teacher import Teacher


class RandomTeacher(Teacher):
    def __init__(self, teacher_parameters, environment_parameters):
        super().__init__(teacher_parameters, environment_parameters)

    def generate_task(self):
        chosen_params = {}
        for param_name in self.env_wrapper.parameters.keys():
            sampled_value = self.env_wrapper.parameters[param_name].sample()  # TODO: seed
            chosen_params[param_name] = sampled_value
        new_env = self.env_wrapper.create_env(chosen_params)
        return new_env, chosen_params

    def update_teacher_policy(self):
        pass
