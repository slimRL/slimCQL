import os
import sys
import json
import argparse


from experiments.base.eval_parser_argument import add_eval_arguments
from experiments.base.evaluate import evaluate
from slimdqn.environments.car_on_hill import CarOnHill


def run(argvs=sys.argv[1:]):
    parser = argparse.ArgumentParser("Evaluate offline agent.")
    add_eval_arguments(parser)
    args = parser.parse_args(argvs)

    env_name = os.path.abspath(__file__).split("/")[-2]
    p = json.load(open(f"experiments/{env_name}/exp_output/{args.experiment_name}/parameters.json", "rb"))
    p["save_path"] = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"../{env_name}/exp_output/{args.experiment_name}/{args.algo_name}",
    )
    env = CarOnHill()

    evaluate(p, args, env)


if __name__ == "__main__":
    run()
