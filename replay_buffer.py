"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import numpy as np


class ReplayBuffer(object):
    def __init__(self, capacity, trajectories=[]):
        self.capacity = capacity
        if len(trajectories) <= self.capacity:
            self.trajectories = trajectories
        else:
            returns = [traj["rewards"].sum() for traj in trajectories]
            sorted_inds = np.argsort(returns)  # lowest to highest
            self.trajectories = [
                trajectories[ii] for ii in sorted_inds[-self.capacity :]
            ]

        self.start_idx = 0

    def __len__(self):
        return len(self.trajectories)

    def add_new_trajs(self, new_trajs):
        if len(self.trajectories) < self.capacity:
            self.trajectories.extend(new_trajs)
            self.trajectories = self.trajectories[-self.capacity :]
        else:
            self.trajectories[
                self.start_idx : self.start_idx + len(new_trajs)
            ] = new_trajs
            self.start_idx = (self.start_idx + len(new_trajs)) % self.capacity

        assert len(self.trajectories) <= self.capacity
