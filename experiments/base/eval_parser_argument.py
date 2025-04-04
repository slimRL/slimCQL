import argparse


def add_eval_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-en",
        "--experiment_name",
        help="Experiment name.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-an",
        "--algo_name",
        help="Algorithm name.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-s",
        "--seed",
        help="Seed.",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-e",
        "--epoch",
        help="Epoch.",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-horizon",
        "--horizon",
        help="Horizon for truncation.",
        type=int,
        default=27_000,
    )
    parser.add_argument(
        "-nespi",
        "--n_evaluation_steps_per_epoch",
        help="Evaluation steps per epoch.",
        type=int,
        default=125_000,
    )
    parser.add_argument(
        "-ee",
        "--epsilon_eval",
        help="Epsilon to use for evaluation.",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "-dm",
        "--delete_models",
        help="Delete models for all but the last epoch.",
        default=False,
        action="store_true",
    )
