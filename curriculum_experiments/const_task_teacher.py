
from typing import Tuple, Dict, Any

from curriculum_experiments.teacher import Teacher


class ConstTaskTeacher(Teacher):
    def __init__(self, teacher_parameters, environment_parameters):
        super().__init__(teacher_parameters, environment_parameters)
        self.task_params = teacher_parameters["params"]

    def generate_task(self):
        new_env = self.env_wrapper.create_env(self.task_params)
        return new_env, self.task_params

    def update_teacher_policy(self):
        pass
