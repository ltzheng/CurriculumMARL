"""Registry of algorithm names."""

import importlib
import traceback

from ray.rllib.algorithms.registry import ALGORITHMS as RLLIB_ALGORITHMS
from ray.rllib.contrib.registry import CONTRIBUTED_ALGORITHMS


def _import_ppo_context():
    from algorithms.ppo.ppo_context import PPOContext
    from ray.rllib.algorithms.ppo.ppo import PPOConfig

    return PPOContext, PPOConfig().to_dict()


def _import_ppo_comm():
    from algorithms.ppo.ppo_comm import PPOComm
    from ray.rllib.algorithms.ppo.ppo import PPOConfig

    return PPOComm, PPOConfig().to_dict()


def _import_ppo_hrl():
    from algorithms.ppo.ppo_hrl import PPOHRL, PPOHRLConfig

    return PPOHRL, PPOHRLConfig().to_dict()


def _import_ppo_curriculum():
    from algorithms.curriculum.curriculum import PPOCurriculum, PPOCurriculumConfig

    return PPOCurriculum, PPOCurriculumConfig().to_dict()


def _import_ppo_comm_curriculum():
    from algorithms.curriculum.curriculum import PPOCommCurriculum, PPOCurriculumConfig

    return PPOCommCurriculum, PPOCurriculumConfig().to_dict()


def _import_ppo_hrl_curriculum():
    from algorithms.curriculum.curriculum import PPOHRLCurriculum, PPOHRLCurriculumConfig

    return PPOHRLCurriculum, PPOHRLCurriculumConfig().to_dict()


ALGORITHMS = dict(
    RLLIB_ALGORITHMS, **{
        "PPO-context": _import_ppo_context,
        "PPO-comm": _import_ppo_comm,
        "PPO-hrl": _import_ppo_hrl,
        "PPO-curriculum": _import_ppo_curriculum,
        "PPO-comm-curriculum": _import_ppo_comm_curriculum,
        "PPO-hrl-curriculum": _import_ppo_hrl_curriculum,
    }
)


def get_algorithm_class(alg: str, return_config=False) -> type:
    """Returns the class of a known Trainer given its name."""

    try:
        return _get_algorithm_class(alg, return_config=return_config)
    except ImportError:
        from ray.rllib.algorithms.mock import _algorithm_import_failed

        class_ = _algorithm_import_failed(traceback.format_exc())
        config = class_.get_default_config()
        if return_config:
            return class_, config
        return class_


def _get_algorithm_class(alg: str, return_config=False) -> type:
    if alg in ALGORITHMS:
        class_, config = ALGORITHMS[alg]()
    elif alg in CONTRIBUTED_ALGORITHMS:
        class_, config = CONTRIBUTED_ALGORITHMS[alg]()
    elif alg == "script":
        from ray.tune import script_runner

        class_, config = script_runner.ScriptRunner, {}
    elif alg == "__fake":
        from ray.rllib.algorithms.mock import _MockTrainer

        class_, config = _MockTrainer, _MockTrainer.get_default_config()
    elif alg == "__sigmoid_fake_data":
        from ray.rllib.algorithms.mock import _SigmoidFakeData

        class_, config = _SigmoidFakeData, _SigmoidFakeData.get_default_config()
    elif alg == "__parameter_tuning":
        from ray.rllib.algorithms.mock import _ParameterTuningTrainer

        class_, config = (
            _ParameterTuningTrainer,
            _ParameterTuningTrainer.get_default_config(),
        )
    else:
        raise Exception("Unknown algorithm {}.".format(alg))

    if return_config:
        return class_, config
    return class_
