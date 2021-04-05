import datetime
import os
import pickle

from rllib.algorithms.mpc import CEMShooting
from rllib.environment import AbstractEnvironment, parse_space
from rllib.policy import MPCPolicy
from tqdm import tqdm

from curriculum_experiments.parametric_ant import AntWrapper
from curriculum_experiments.parametric_half_cheetah import HalfCheetahWrapper
from curriculum_experiments.random_teacher import RandomTeacher
from curriculum_experiments.task_difficulty_estimate import estimate_task_difficulties
from hucrl.agent import MPCAgent
from hucrl.environment.hallucination_wrapper import HallucinationWrapper
from hucrl.model.hallucinated_model import HallucinatedModel


class SillyEnvironmentWrapper(AbstractEnvironment):
    def __init__(self, env, seed=None, **kwargs):
        self.env = env
        self.env.seed(seed)

        dim_action, num_actions = parse_space(self.env.action_space)
        dim_state, num_states = parse_space(self.env.observation_space)
        if num_states > -1:
            num_states += 1  # Add a terminal state.

        super().__init__(
            dim_action=dim_action,
            dim_state=dim_state,
            action_space=self.env.action_space,
            observation_space=self.env.observation_space,
            num_actions=num_actions,
            num_states=num_states,
            num_observations=num_states,
        )
        self._time = 0
        self.metadata = self.env.metadata

    def add_wrapper(self, wrapper, **kwargs):
        """Add a wrapper for the environment."""
        self.env = wrapper(self.env, **kwargs)

        dim_action, num_actions = parse_space(self.env.action_space)
        dim_state, num_states = parse_space(self.env.observation_space)
        if num_states > -1:
            num_states += 1  # Add a terminal state.

        super().__init__(
            dim_action=dim_action,
            dim_state=dim_state,
            action_space=self.env.action_space,
            observation_space=self.env.observation_space,
            num_actions=num_actions,
            num_states=num_states,
            num_observations=num_states,
        )
        self._time = 0

    def pop_wrapper(self):
        """Pop last wrapper."""
        self.env = self.env.env

        dim_action, num_actions = parse_space(self.env.action_space)
        dim_state, num_states = parse_space(self.env.observation_space)
        if num_states > -1:
            num_states += 1  # Add a terminal state.

        super().__init__(
            dim_action=dim_action,
            dim_state=dim_state,
            action_space=self.env.action_space,
            observation_space=self.env.observation_space,
            num_actions=num_actions,
            num_states=num_states,
            num_observations=num_states,
        )
        self._time = 0

    def step(self, action):
        """See `AbstractEnvironment.step'."""
        next_state, reward, done, info = self.env.step(action)
        if self.num_states > 0 and done:  # Move to terminal state.
            next_state = self.num_states - 1
        self._time += 1
        return next_state, reward, done, info

    def render(self, mode="human"):
        """See `AbstractEnvironment.render'."""
        return self.env.render(mode)

    def close(self):
        """See `AbstractEnvironment.close'."""
        self.env.close()

    def reset(self):
        """See `AbstractEnvironment.reset'."""
        self._time = 0
        return self.env.reset()

    @property
    def goal(self):
        """Return current goal of environment."""
        if hasattr(self.env, "goal"):
            return self.env.goal
        return None

    @property
    def state(self):
        """See `AbstractEnvironment.state'."""
        if hasattr(self.env, "state"):
            return self.env.state
        elif hasattr(self.env, "s"):
            return self.env.s
        elif hasattr(self.env, "_get_obs"):
            return getattr(self.env, "_get_obs")()
        else:
            raise NotImplementedError("Strange state")

    @state.setter
    def state(self, value):
        if hasattr(self.env, "state"):
            self.env.state = value
        elif hasattr(self.env, "s"):
            self.env.s = value
        elif hasattr(self.env, "set_state"):
            self.env.set_state(
                value[: len(self.env.sim.data.qpos)],
                value[len(self.env.sim.data.qpos) :],
            )
        else:
            raise NotImplementedError("Strange state")

    @property
    def time(self):
        """See `AbstractEnvironment.time'."""
        return self._time

    @property
    def name(self):
        """Return class name."""
        return self.env_name


def run_on_parameteric_env(steps_per_task, tasks, wrapper, easy_task):
    random_teacher = RandomTeacher(None, wrapper)


    ref_env = wrapper.create_env(easy_task)
    wrapped_env = SillyEnvironmentWrapper(ref_env)
    dynamical_model = HallucinatedModel.default(wrapped_env, beta=1.0)
    wrapped_env.add_wrapper(HallucinationWrapper)

    student = MPCAgent(
        mpc_policy=MPCPolicy(mpc_solver=CEMShooting(dynamical_model=dynamical_model,
                                                    reward_model=ref_env.reward_model(),
                                                    horizon=10,
                                                    num_samples=200)),
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
