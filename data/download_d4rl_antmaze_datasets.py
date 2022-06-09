"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import pickle
import gym
import numpy as np
import d4rl
from collections import defaultdict
from icecream import ic
from tqdm import tqdm


def split_into_trajectories(
    dones_float, observations, next_observations, actions, rewards
):
    trajs = [defaultdict(list)]
    for i in tqdm(range(len(observations))):
        trajs[-1]["observations"].append(observations[i])
        trajs[-1]["actions"].append(actions[i])
        trajs[-1]["rewards"].append(rewards[i])
        trajs[-1]["next_observations"].append(next_observations[i])
        trajs[-1]["terminals"].append(dones_float[i])
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append(defaultdict(list))

    for traj in trajs:
        for kk, vv in traj.items():
            traj[kk] = np.array(vv)
    return trajs


for env_name in [
    "antmaze-umaze-v2",
    "antmaze-umaze-diverse-v2",
    "antmaze-medium-diverse-v2",
    "antmaze-medium-play-v2",
    "antmaze-large-diverse-v2",
    "antmaze-large-play-v2",
]:
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)
    MAX_EPISODE_LEN = env._max_episode_steps
    env.close()

    ic(env_name, MAX_EPISODE_LEN)

    dones_float = np.zeros_like(dataset["rewards"])

    for i in range(len(dones_float) - 1):
        if (
            np.linalg.norm(
                dataset["observations"][i + 1] - dataset["next_observations"][i]
            )
            > 1e-6
            or dataset["terminals"][i] == 1.0
        ):
            dones_float[i] = 1
        else:
            dones_float[i] = 0

    dones_float[-1] = 1

    trajectories = split_into_trajectories(
        dones_float,
        dataset["observations"],
        dataset["next_observations"],
        dataset["actions"],
        dataset["rewards"],
    )

    returns = np.array([np.sum(traj["rewards"]) for traj in trajectories])
    lengths = np.array([len(traj["rewards"]) for traj in trajectories])
    num_samples = np.sum(lengths)
    print(f"number of samples collected: {num_samples}")
    print(
        f"Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}"
        f"\n"
        f"Trajectory lengths: mean = {np.mean(lengths)}, std = {np.std(lengths)}, max = {np.max(lengths)}, min = {np.min(lengths)}"
    )

    with open(f"{env_name}.pkl", "wb") as f:
        pickle.dump(trajectories, f)
