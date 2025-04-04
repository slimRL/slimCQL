import os
import shutil
import subprocess
import unittest


class TestCarOnHill(unittest.TestCase):
    def test_dqn(self):
        save_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../experiments/car_on_hill/exp_output/_test_dqn",
        )
        print("SAVE PATH IS ", save_path)
        if os.path.exists(save_path):
            shutil.rmtree(save_path)

        returncode = subprocess.run(
            [
                "python3",
                "experiments/car_on_hill/dqn.py",
                "--experiment_name",
                "_test_dqn",
                "--seed",
                "1",
                "--disable_wandb",
                "--features",
                "25",
                "15",
                "--replay_buffer_capacity",
                "10",
                "--batch_size",
                "3",
                "--update_horizon",
                "1",
                "--gamma",
                "0.99",
                "--learning_rate",
                "1e-4",
                "--architecture_type",
                "fc",
                "--target_update_frequency",
                "1",
                "--n_buffers_to_load",
                "1",
                "--n_epochs",
                "1",
                "--n_fitting_steps",
                "10",
            ]
        ).returncode
        assert returncode == 0, "The command should not have raised an error."

        shutil.rmtree(os.path.join(save_path, "../../replay_buffer/uniform_10"))
        shutil.rmtree(save_path)
