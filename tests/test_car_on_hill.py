import os
import shutil
import subprocess
import unittest


class TestCarOnHill(unittest.TestCase):
    def test_fqi(self):
        save_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../experiments/car_on_hill/exp_output/_test_fqi",
        )
        print("SAVE PATH IS ", save_path)
        if os.path.exists(save_path):
            shutil.rmtree(save_path)

        returncode = subprocess.run(
            [
                "python3",
                "experiments/car_on_hill/fqi.py",
                "--experiment_name",
                "_test_fqi",
                "--seed",
                "1",
                "--disable_wandb",
                "--features",
                "25",
                "15",
                "--replay_buffer_capacity",
                "100",
                "--batch_size",
                "3",
                "--update_horizon",
                "1",
                "--gamma",
                "0.99",
                "--learning_rate",
                "1e-4",
                "--n_bellman_iterations",
                "1",
                "--n_fitting_steps",
                "10",
                "--architecture_type",
                "fc",
            ]
        ).returncode
        assert returncode == 0, "The command should not have raised an error."

        # shutil.rmtree(save_path)
