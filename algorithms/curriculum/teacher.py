from abc import ABCMeta
import logging
import ray
import numpy as np

from ray.rllib.algorithms.algorithm import Algorithm
from algorithms.curriculum.task_generators import (
    UniformTaskGenerator,
    ALPGMMTaskGenerator,
    VACLTaskGenerator,
    ContextualBanditTaskGenerator,
)
from ray.rllib.utils.typing import AlgorithmConfigDict, ResultDict

logger = logging.getLogger(__name__)


class Teacher(metaclass=ABCMeta):
    def __init__(
        self,
        trainer: Algorithm,
        trainer_config: AlgorithmConfigDict,
    ):
        """Initializes a Teacher instance."""
        self.trainer = trainer
        self.config = trainer_config

    def update_curriculum(self, result: ResultDict):
        """Method containing curriculum logic. Called after train step."""
        return result


class UniformTeacher(Teacher):
    def __init__(
        self,
        trainer: Algorithm,
        trainer_config: AlgorithmConfigDict,
    ):
        super().__init__(
            trainer,
            trainer_config,
        )
        self.teacher_config = trainer_config["teacher_config"]
        self.task_generator = UniformTaskGenerator.options(name="task_generator").remote(
            seed=self.config["seed"],
            num_agents=self.teacher_config.get("num_agents", [1]),
        )


class ALPGMMTeacher(Teacher):
    def __init__(
        self,
        trainer: Algorithm,
        trainer_config: AlgorithmConfigDict,
    ):
        super().__init__(
            trainer,
            trainer_config,
        )
        self.teacher_config = trainer_config["teacher_config"]
        self.task_generator = ALPGMMTaskGenerator.options(name="task_generator").remote(
            seed=self.config["seed"],
            num_agents=self.teacher_config.get("num_agents", [1]),
            gmm_fitness_func=self.teacher_config.get("gmm_fitness_func", "aic"),
            warm_start=self.teacher_config.get("warm_start", False),
            nb_em_init=self.teacher_config.get("nb_em_init", 1),
            fit_rate=self.teacher_config.get("fit_rate", 250),
            alp_window_size=self.teacher_config.get("alp_window_size", None),
            potential_ks=self.teacher_config.get("potential_ks", np.arange(2, 11, 1)),
            random_task_ratio=self.teacher_config.get("random_task_ratio", 0.2),
            nb_bootstrap=self.teacher_config.get("nb_bootstrap", None),
            initial_dist=self.teacher_config.get("initial_dist", None),
        )

    def update_curriculum(self, result: ResultDict) -> dict:
        infos = ray.get(self.task_generator.get_infos.remote())
        for k, v in infos.items():
            result[f"alp_gmm_{k}"] = v
        return result


class VACLTeacher(Teacher):
    def __init__(
        self,
        trainer: Algorithm,
        trainer_config: AlgorithmConfigDict,
    ):
        super().__init__(
            trainer,
            trainer_config,
        )
        self.teacher_config = trainer_config["teacher_config"]
        self.task_generator = VACLTaskGenerator.options(name="task_generator").remote(
            seed=self.config["seed"],
            solved_prop=self.teacher_config["solved_prop"],
            num_agents=self.teacher_config["num_agents"],
            buffer_length=self.teacher_config["buffer_length"],
            reproduction_num=self.teacher_config["reproduction_num"],
            epsilon=self.teacher_config["epsilon"],
            delta=self.teacher_config["delta"],
            h=self.teacher_config["h"],
            Rmin=self.teacher_config["Rmin"],
            Rmax=self.teacher_config["Rmax"],
            del_switch=self.teacher_config["del_switch"],
            topk=self.teacher_config["topk"],
            num_initial_tasks=self.teacher_config["num_initial_tasks"],
        )

    def update_curriculum(self, result: ResultDict) -> dict:
        self.task_generator.update_buffer.remote()
        self.task_generator.compute_gradient.remote()
        infos = ray.get(self.task_generator.get_infos.remote())
        for k, v in infos.items():
            result[f"vacl_{k}"] = v
        return result


class ContextualBanditTeacher(Teacher):
    def __init__(
        self,
        trainer: Algorithm,
        trainer_config: AlgorithmConfigDict,
    ):
        super().__init__(
            trainer,
            trainer_config,
        )
        self.teacher_config = trainer_config["teacher_config"]
        self.task_generator = ContextualBanditTaskGenerator.options(name="task_generator").remote(
            seed=self.config["seed"],
            num_contexts=self.teacher_config.get("num_contexts", 3),
            gamma=self.teacher_config.get("gamma", 0.3),
            num_agents=self.teacher_config.get("num_agents", [1]),
            min_rew=self.teacher_config.get("min_rew", 0),
            max_rew=self.teacher_config.get("max_rew", 1),
        )

    def update_curriculum(self, result: ResultDict) -> dict:
        context = None
        if self.trainer.get_policy("high_level_policy"):  # HRL
            context = getattr(self.trainer.get_policy("high_level_policy").model, "last_hx", [0])
        elif self.trainer.get_policy():  # ATT-COM
            context = getattr(self.trainer.get_policy().model, "last_hx", [0])
        elif self.trainer.get_policy("shared_policy"):  # PPO
            context = getattr(self.trainer.get_policy("shared_policy").model, "last_hx", [0])
        if context is None:
            context = [0]
        # self.task_generator.update_context.remote(context)
        if "evaluation" in result:
            self.task_generator.update_eval_reward.remote(result["evaluation"]["custom_metrics"]["score_mean"])
            self.task_generator.update_context.remote(context)

        probs = ray.get(self.task_generator.context_task_probs.remote())
        for i, p in enumerate(probs):
            for j, val in enumerate(p):
                result[f"bandit_context{i}_task{j}_prob"] = val

        return result


TEACHERS = {
    "uniform": UniformTeacher,
    "bandit": ContextualBanditTeacher,
    "alp-gmm": ALPGMMTeacher,
    "vacl": VACLTeacher,
}
