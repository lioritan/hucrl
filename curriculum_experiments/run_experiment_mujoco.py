import datetime
import os
import pickle

from rllib.algorithms.mpc import CEMShooting
from tqdm import tqdm

from curriculum_experiments.parametric_ant import AntWrapper
from curriculum_experiments.parametric_half_cheetah import HalfCheetahWrapper
from curriculum_experiments.random_teacher import RandomTeacher
from curriculum_experiments.task_difficulty_estimate import estimate_task_difficulties
from hucrl.agent import MPCAgent
from hucrl.model.hallucinated_model import HallucinatedModel


def run_on_parameteric_env(steps_per_task, tasks, wrapper, easy_task):
    random_teacher = RandomTeacher(None, wrapper)

    ref_env = wrapper.create_env(easy_task)
    dynamical_model = HallucinatedModel.default(ref_env, beta=1.0)

    student = MPCAgent.default(
        mpc_policy=CEMShooting(dynamical_model=dynamical_model,
                               reward_model=ref_env.reward_model(),
                               horizon=10,
                               num_samples=200),
        environment=ref_env,
        thompson_sampling=False,
        exploration_episodes=1,
    )

    date_string = datetime.datetime.today().strftime('%Y-%m-%d %H')
    os.makedirs(f"./results/{date_string}/difficulty/{wrapper.name}", exist_ok=True)

    for i in tqdm(range(tasks)):
        random_teacher.train_k_actions(student, steps_per_task)
        if i % 50 == 0 and i > 0:
            difficulty_estimates, task_params = estimate_task_difficulties(student, wrapper, 5, 3, steps_per_task)
            with open(f"./results/{date_string}/difficulty/{wrapper.name}/data_{i}.pkl", "wb") as fptr:
                pickle.dump((difficulty_estimates, task_params), fptr)

    difficulty_estimates, task_params = estimate_task_difficulties(student, wrapper, 5, 3, steps_per_task)

    with open(f"./results/{date_string}/difficulty/{wrapper.name}/data_all.pkl", "wb") as fptr:
        pickle.dump((difficulty_estimates, task_params), fptr)
    with open(f"./results/{date_string}/difficulty/{wrapper.name}/hist.pkl", "wb") as fptr:
        pickle.dump(random_teacher.history.history, fptr)
    student.save(f"./results/{date_string}/difficulty/{wrapper.name}/student.agent", directory=".")


if __name__ == "__main__":
    easy_params = {
        "expected_speed": 0.1,
    }
    run_on_parameteric_env(1000, 250, HalfCheetahWrapper(), easy_params)

    easy_params2 = {
        "goal_x": 0.1,
        "goal_y": 0.1
    }
    run_on_parameteric_env(1000, 250, AntWrapper(), easy_params2)
