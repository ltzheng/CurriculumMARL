import argparse
import yaml
import logging

import ray
from ray.tune.experiment.config_parser import _make_parser
from ray.tune.progress_reporter import CLIReporter
from ray.tune.result import DEFAULT_RESULTS_DIR
from ray.tune.resources import resources_to_json
from ray.tune.tune import run_experiments
from ray.tune.registry import register_trainable, register_env
from ray.tune.schedulers import create_scheduler
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_torch

from algorithms.registry import ALGORITHMS, get_algorithm_class
from env.registry import ENVIRONMENTS, get_env_class, POLICY_MAPPINGS, CALLBACKS
from models.registry import MODELS, get_model_class, ACTION_DISTS, get_action_dist_class

EXAMPLE_USAGE = """
python train.py -f configs/football/ppo/5v5.yaml
"""

# TODO
logger = logging.getLogger("ray.rllib")
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter(
        "%(asctime)s\t%(levelname)s %(filename)s:%(lineno)s -- %(message)s"
    )
)
logger.addHandler(handler)
logger.propagate = False

for key in list(ALGORITHMS.keys()):
    register_trainable(key, get_algorithm_class(key))
for key in list(ENVIRONMENTS.keys()):
    register_env(key, get_env_class(key))
for key in list(MODELS.keys()):
    ModelCatalog.register_custom_model(key, get_model_class(key))
for key in list(ACTION_DISTS.keys()):
    ModelCatalog.register_custom_action_dist(key, get_action_dist_class(key))

# Try to import both backends for flag checking/warnings.
torch, _ = try_import_torch()


def create_parser(parser_creator=None):
    parser = _make_parser(
        parser_creator=parser_creator,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Train a reinforcement learning agent.",
        epilog=EXAMPLE_USAGE,
    )

    # See also the base parser definition in ray/tune/experiment/__config_parser.py
    parser.add_argument(
        "--ray-address",
        default=None,
        type=str,
        help="Connect to an existing Ray cluster at this address instead "
        "of starting a new one.",
    )
    parser.add_argument(
        "--ray-ui", action="store_true", help="Whether to enable the Ray web UI."
    )
    parser.add_argument(
        "--local-mode",
        action="store_true",
        help="Run ray in local mode for easier debugging.",
    )
    parser.add_argument(
        "--ray-num-cpus",
        default=None,
        type=int,
        help="--num-cpus to use if starting a new cluster.",
    )
    parser.add_argument(
        "--ray-num-gpus",
        default=None,
        type=int,
        help="--num-gpus to use if starting a new cluster.",
    )
    parser.add_argument(
        "--ray-num-nodes",
        default=None,
        type=int,
        help="Emulate multiple cluster nodes for debugging.",
    )
    parser.add_argument(
        "--ray-object-store-memory",
        default=None,
        type=int,
        help="--object-store-memory to use if starting a new cluster.",
    )
    parser.add_argument(
        "--experiment-name",
        default="default",
        type=str,
        help="Name of the subdirectory under `local_dir` to put results in.",
    )
    parser.add_argument(
        "--local-dir",
        default=DEFAULT_RESULTS_DIR,
        type=str,
        help="Local dir to save training results to. Defaults to '{}'.".format(
            DEFAULT_RESULTS_DIR
        ),
    )
    parser.add_argument(
        "--upload-dir",
        default="",
        type=str,
        help="Optional URI to sync training results to (e.g. s3://bucket).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume previous Tune experiments.",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Whether to attempt to enable tracing for eager mode.",
    )
    parser.add_argument(
        "--env", default=None, type=str, help="The gym environment to use."
    )
    parser.add_argument(
        "-f",
        "--config-file",
        default=None,
        type=str,
        help="If specified, use config options from this file. Note that this "
        "overrides any trial-specific options set via flags above.",
    )
    parser.add_argument(
        "--seed",
        default=None,
        type=int,
    )

    return parser


def run(args, parser):
    if args.config_file:
        with open(args.config_file) as f:
            experiments = yaml.safe_load(f)
    else:
        # Note: keep this in sync with tune/experiment/__config_parser.py
        experiments = {
            args.experiment_name: {
                "run": args.run,
                "checkpoint_freq": args.checkpoint_freq,
                "checkpoint_at_end": args.checkpoint_at_end,
                "keep_checkpoints_num": args.keep_checkpoints_num,
                "checkpoint_score_attr": args.checkpoint_score_attr,
                "local_dir": args.local_dir,
                "resources_per_trial": (
                    args.resources_per_trial
                    and resources_to_json(args.resources_per_trial)
                ),
                "stop": args.stop,
                "config": dict(args.config, env=args.env),
                "restore": args.restore,
                "num_samples": args.num_samples,
                "sync_config": {
                    "upload_dir": args.upload_dir,
                },
            }
        }

    verbose = 1
    for exp in experiments.values():
        if args.seed is not None:
            exp["config"]["seed"] = args.seed
        metric_columns = exp.pop("metric_columns", None)
        if not exp.get("run"):
            parser.error("the following arguments are required: --run")
        if not exp.get("env") and not exp.get("config", {}).get("env"):
            parser.error("the following arguments are required: --env")
        if exp["config"].get("multiagent"):
            policy_mapping_name = exp["config"]["multiagent"].get("policy_mapping_fn")
            if isinstance(policy_mapping_name, str):
                exp["config"]["multiagent"]["policy_mapping_fn"] = POLICY_MAPPINGS[policy_mapping_name]
        if exp["config"].get("callbacks"):
            calback_name = exp["config"].get("callbacks")
            if isinstance(calback_name, str):
                exp["config"]["callbacks"] = CALLBACKS[calback_name]

    if args.ray_num_nodes:
        # Import this only here so that train.py also works with
        # older versions (and user doesn't use `--ray-num-nodes`).
        from ray.cluster_utils import Cluster

        cluster = Cluster()
        for _ in range(args.ray_num_nodes):
            cluster.add_node(
                num_cpus=args.ray_num_cpus or 1,
                num_gpus=args.ray_num_gpus or 0,
                object_store_memory=args.ray_object_store_memory,
            )
        ray.init(address=cluster.address)
    else:
        ray.init(
            include_dashboard=args.ray_ui,
            address=args.ray_address,
            object_store_memory=args.ray_object_store_memory,
            num_cpus=args.ray_num_cpus,
            num_gpus=args.ray_num_gpus,
            local_mode=args.local_mode,
        )

    progress_reporter = CLIReporter(
        print_intermediate_tables=verbose >= 1,
        metric_columns=metric_columns,
    )

    run_experiments(
        experiments,
        scheduler=create_scheduler(args.scheduler, **args.scheduler_config),
        resume=args.resume,
        verbose=verbose,
        progress_reporter=progress_reporter,
        concurrent=True,
    )

    ray.shutdown()


def main():
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)


if __name__ == "__main__":
    main()
