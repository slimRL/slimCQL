import os
import gzip
from tqdm import tqdm
import numpy as np
import tensorflow as tf

GAME = "VideoPinball"
RUN = 1
SRC_DIR = "experiments/atari/datasets/rlu_dataset"
DEST_DIR = "experiments/atari/datasets/numpy_dataset"


def atari_example_to_rlds(example_bytes: tf.Tensor):
    data = tf.io.parse_single_example(
        example_bytes,
        {
            "episode_idx": tf.io.FixedLenFeature([], tf.int64),
            "checkpoint_idx": tf.io.FixedLenFeature([], tf.int64),
            "episode_return": tf.io.FixedLenFeature([], tf.float32),
            "actions": tf.io.VarLenFeature(tf.int64),
            "observations": tf.io.VarLenFeature(tf.string),
            "unclipped_rewards": tf.io.VarLenFeature(tf.float32),
            "discounts": tf.io.VarLenFeature(tf.float32),
        },
    )

    actions = tf.sparse.to_dense(data["actions"])
    rewards = tf.sparse.to_dense(data["unclipped_rewards"])
    discounts = tf.sparse.to_dense(data["discounts"])
    obs_png_bytes = tf.sparse.to_dense(data["observations"], default_value=b"")

    def _decode_one(x):
        img = tf.io.decode_png(x, channels=1)  # [84,84,1], uint8
        return img

    observations = tf.map_fn(_decode_one, obs_png_bytes, dtype=tf.uint8, back_prop=False)

    T = tf.shape(actions)[0]

    is_first = tf.concat([[True], tf.zeros(T - 1, dtype=tf.bool)], axis=0)
    is_last = tf.concat([tf.zeros(T - 1, dtype=tf.bool), [True]], axis=0)
    is_term = tf.zeros_like(actions, dtype=tf.bool)

    terminal_cond = tf.equal(discounts[-1], 0.0)
    is_term = tf.cond(
        terminal_cond, lambda: tf.concat([tf.zeros(T - 1, dtype=tf.bool), [True]], axis=0), lambda: is_term
    )

    discounts = tf.cond(terminal_cond, lambda: tf.concat([discounts[1:], [0.0]], axis=0), lambda: discounts)

    return {
        "episode_id": data["episode_idx"],
        "checkpoint_id": data["checkpoint_idx"],
        "episode_return": data["episode_return"],
        "steps": {
            "observation": observations,
            "action": actions,
            "reward": rewards,
            "discount": discounts,
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_term,
        },
    }


def episode_generator(ds: tf.data.Dataset):
    for ep in ds:
        ep_id = ep["episode_id"].numpy()
        steps = ep["steps"]
        obs = steps["observation"].numpy()
        acts = steps["action"].numpy()
        rews = steps["reward"].numpy()
        last_flags = steps["is_last"].numpy()
        term_flags = steps["is_terminal"].numpy()

        yield ep_id, obs, acts, rews, last_flags, term_flags


os.makedirs(f"{DEST_DIR}/{GAME}/{RUN}", exist_ok=True)
for ckpt in tqdm(range(50)):  # 50 checkpoints for every run in RLU dataset
    raw_ds = tf.data.TFRecordDataset([f"{SRC_DIR}/{GAME}/run_{RUN}-{ckpt:05d}-of-00050"], compression_type="GZIP")
    ds = raw_ds.map(atari_example_to_rlds, num_parallel_calls=tf.data.AUTOTUNE)

    episodes = []
    for ep_id, obs, acts, rews, last_f, term_f in episode_generator(ds):
        episodes.append((ep_id, obs, acts, rews, last_f, term_f))

    arrays = {
        "observations": np.concatenate([episode[1] for episode in episodes]).squeeze(),
        "actions": np.concatenate([episode[2] for episode in episodes]),
        "rewards": np.concatenate([episode[3] for episode in episodes]),
        "is_terminals": np.concatenate([episode[4] for episode in episodes]),
        "episode_ends": np.concatenate([episode[5] for episode in episodes]),
    }
    for attr, array in arrays.items():
        np.save(
            gzip.GzipFile(fileobj=open(f"{DEST_DIR}/{GAME}/{RUN}/{attr}_{ckpt}.gz", "wb")), array, allow_pickle=False
        )
