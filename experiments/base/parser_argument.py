import argparse
from functools import wraps
from typing import Callable, List


def output_added_arguments(add_algo_arguments: Callable) -> Callable:
    @wraps(add_algo_arguments)
    def decorated(parser: argparse.ArgumentParser) -> List[str]:
        unfiltered_old_arguments = list(parser._option_string_actions.keys())

        add_algo_arguments(parser)

        unfiltered_arguments = list(parser._option_string_actions.keys())
        unfiltered_added_arguments = [
            argument for argument in unfiltered_arguments if argument not in unfiltered_old_arguments
        ]

        return [
            argument.strip("-")
            for argument in unfiltered_added_arguments
            if argument.startswith("--") and argument not in ["--help"]
        ]

    return decorated


@output_added_arguments
def add_base_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-en",
        "--experiment_name",
        help="Experiment name.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-s",
        "--seed",
        help="Seed of the experiment.",
        type=int,
        required=True,
    )
    parser.add_argument(
        "-dw",
        "--disable_wandb",
        help="Disable wandb.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "-f",
        "--features",
        nargs="*",
        help="List of features for the Q-networks.",
        type=int,
        default=[200, 200],
    )
    parser.add_argument(
        "-rbc",
        "--replay_buffer_capacity",
        help="Dataset (Fixed replay) size.",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        help="Batch size for training.",
        type=int,
        default=32,
    )
    parser.add_argument(
        "-n",
        "--update_horizon",
        help="Value of n in n-step TD update.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-gamma",
        "--gamma",
        help="Discounting factor.",
        type=float,
        default=0.99,
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        help="Learning rate.",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "-at",
        "--architecture_type",
        help="Type of architecture.",
        type=str,
        default="fc",
        choices=["cnn", "impala", "fc"],
    )


@output_added_arguments
def add_fqi_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-nbi",
        "--n_bellman_iterations",
        help="Number of Bellman iterations to perform.",
        type=int,
        default=30,
    )

    parser.add_argument(
        "-nfs",
        "--n_fitting_steps",
        help="Number of gradient update steps per Bellman iteration.",
        type=int,
        default=5,
    )
