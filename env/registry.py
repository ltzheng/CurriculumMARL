"""Registry of environment names."""
from env.callbacks import PvEMetricsCallback
from env.policy_mappings import parameter_sharing_policy_mapping_fn


def _import_gfootball_qmix(cfg):
    from gym.spaces import Tuple
    from env.football.multi_agent_env import FootballPvEEnv
    env = FootballPvEEnv(**cfg)
    agent_list = [f"agent_{i}" for i in list(range(env.num_agents))]
    grouping = {"group_1": agent_list}
    obs_space = Tuple([env.observation_space for _ in agent_list])
    act_space = Tuple([env.action_space for _ in agent_list])
    return env.with_agent_groups(
        grouping, obs_space=obs_space, act_space=act_space,
    )


def _import_gfootball_pve(cfg):
    from env.football.multi_agent_env import FootballPvEEnv
    return FootballPvEEnv(**cfg)


def _import_gfootball_pve_com(cfg):
    from env.football.com_env import FootballPvEComEnv
    return FootballPvEComEnv(**cfg)


def _import_gfootball_pve_hrl(cfg):
    from env.football.hierarchical_env import FootballPvEHierarchicalEnv
    return FootballPvEHierarchicalEnv(**cfg)


def _import_gfootball_pve_hrl_com(cfg):
    from env.football.hierarchical_env import FootballPvEHierarchicalComEnv
    return FootballPvEHierarchicalComEnv(**cfg)


def _import_gfootball_curriculum(cfg):
    from env.football.curriculum_env import FootballCurriculumPvEEnv
    return FootballCurriculumPvEEnv(**cfg)


def _import_gfootball_curriculum_com(cfg):
    from env.football.curriculum_env import FootballCurriculumPvEComEnv
    return FootballCurriculumPvEComEnv(**cfg)


def _import_gfootball_curriculum_hrl(cfg):
    from env.football.curriculum_env import FootballCurriculumPvEHierarchicalEnv
    return FootballCurriculumPvEHierarchicalEnv(**cfg)


def _import_gfootball_curriculum_hrl_com(cfg):
    from env.football.curriculum_env import FootballCurriculumPvEHierarchicalComEnv
    return FootballCurriculumPvEHierarchicalComEnv(**cfg)


def _import_starcraft_qmix(cfg):
    from gym.spaces import Tuple
    from env.starcraft.multi_agent_env import StarCraft2PvEEnv
    env = StarCraft2PvEEnv(**cfg)
    agent_list = [f"agent_{i}" for i in list(range(env.num_agents))]
    grouping = {"group_1": agent_list}
    obs_space = Tuple([env.observation_space for _ in agent_list])
    act_space = Tuple([env.action_space for _ in agent_list])
    return env.with_agent_groups(
        grouping, obs_space=obs_space, act_space=act_space,
    )


def _import_starcraft_pve(cfg):
    from env.starcraft.multi_agent_env import StarCraft2PvEEnv
    return StarCraft2PvEEnv(**cfg)


def _import_starcraft_pve_com(cfg):
    from env.starcraft.com_env import StarCraft2PvEComEnv
    return StarCraft2PvEComEnv(**cfg)


def _import_starcraft_pve_hrl(cfg):
    from env.starcraft.hierarchical_env import StarCraft2PvEHierarchicalEnv
    return StarCraft2PvEHierarchicalEnv(**cfg)


def _import_starcraft_pve_hrl_com(cfg):
    from env.starcraft.hierarchical_env import StarCraft2PvEHierarchicalComEnv
    return StarCraft2PvEHierarchicalComEnv(**cfg)


def _import_starcraft_curriculum(cfg):
    from env.starcraft.curriculum_env import StarCraft2CurriculumPvEEnv
    return StarCraft2CurriculumPvEEnv(**cfg)


def _import_starcraft_curriculum_com(cfg):
    from env.starcraft.curriculum_env import StarCraft2CurriculumPvEComEnv
    return StarCraft2CurriculumPvEComEnv(**cfg)


def _import_starcraft_curriculum_hrl(cfg):
    from env.starcraft.curriculum_env import StarCraft2CurriculumPvEHierarchicalEnv
    return StarCraft2CurriculumPvEHierarchicalEnv(**cfg)


def _import_starcraft_curriculum_hrl_com(cfg):
    from env.starcraft.curriculum_env import StarCraft2CurriculumPvEHierarchicalComEnv
    return StarCraft2CurriculumPvEHierarchicalComEnv(**cfg)


ENVIRONMENTS = {
    "gfootball_qmix": _import_gfootball_qmix,
    "gfootball_pve": _import_gfootball_pve,
    "gfootball_pve_com": _import_gfootball_pve_com,
    "gfootball_pve_hrl": _import_gfootball_pve_hrl,
    "gfootball_pve_hrl_com": _import_gfootball_pve_hrl_com,
    "gfootball_curriculum": _import_gfootball_curriculum,
    "gfootball_curriculum_com": _import_gfootball_curriculum_com,
    "gfootball_curriculum_hrl": _import_gfootball_curriculum_hrl,
    "gfootball_curriculum_hrl_com": _import_gfootball_curriculum_hrl_com,
    "starcraft_qmix": _import_starcraft_qmix,
    "starcraft_pve": _import_starcraft_pve,
    "starcraft_pve_com": _import_starcraft_pve_com,
    "starcraft_pve_hrl": _import_starcraft_pve_hrl,
    "starcraft_pve_hrl_com": _import_starcraft_pve_hrl_com,
    "starcraft_curriculum": _import_starcraft_curriculum,
    "starcraft_curriculum_com": _import_starcraft_curriculum_com,
    "starcraft_curriculum_hrl": _import_starcraft_curriculum_hrl,
    "starcraft_curriculum_hrl_com": _import_starcraft_curriculum_hrl_com,
}


def get_env_class(env: str) -> type:
    """Returns the class of a known environment given its name."""

    if env in ENVIRONMENTS:
        class_ = ENVIRONMENTS[env]
    else:
        raise Exception("Unknown environment {env}.")

    return class_


POLICY_MAPPINGS = {
    "parameter_sharing": parameter_sharing_policy_mapping_fn,
}

CALLBACKS = {
    "PvEMetrics": PvEMetricsCallback,
}
