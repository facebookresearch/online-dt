"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import torch
import numpy as np
import random


MAX_EPISODE_LEN = 1000


class SubTrajectory(torch.utils.data.Dataset):
    def __init__(
        self,
        trajectories,
        sampling_ind,
        transform=None,
    ):

        super(SubTrajectory, self).__init__()
        self.sampling_ind = sampling_ind
        self.trajs = trajectories
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        traj = self.trajs[self.sampling_ind[index]]
        if self.transform:
            return self.transform(traj)
        else:
            return traj

    def __len__(self):
        return len(self.sampling_ind)


class TransformSamplingSubTraj:
    def __init__(
        self,
        max_len,
        state_dim,
        act_dim,
        state_mean,
        state_std,
        reward_scale,
        action_range,
    ):
        super().__init__()
        self.max_len = max_len
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.state_mean = state_mean
        self.state_std = state_std
        self.reward_scale = reward_scale

        # For some datasets there are actions with values 1.0/-1.0 which is problematic
        # for the SquahsedNormal distribution. The inversed tanh transformation will
        # produce NAN when computing the log-likelihood. We clamp them to be within
        # the user defined action range.
        self.action_range = action_range

    def __call__(self, traj):
        si = random.randint(0, traj["rewards"].shape[0] - 1)

        # get sequences from dataset
        ss = traj["observations"][si : si + self.max_len].reshape(-1, self.state_dim)
        aa = traj["actions"][si : si + self.max_len].reshape(-1, self.act_dim)
        rr = traj["rewards"][si : si + self.max_len].reshape(-1, 1)
        if "terminals" in traj:
            dd = traj["terminals"][si : si + self.max_len]  # .reshape(-1)
        else:
            dd = traj["dones"][si : si + self.max_len]  # .reshape(-1)

        # get the total length of a trajectory
        tlen = ss.shape[0]

        timesteps = np.arange(si, si + tlen)  # .reshape(-1)
        ordering = np.arange(tlen)
        ordering[timesteps >= MAX_EPISODE_LEN] = -1
        ordering[ordering == -1] = ordering.max()
        timesteps[timesteps >= MAX_EPISODE_LEN] = MAX_EPISODE_LEN - 1  # padding cutoff

        rtg = discount_cumsum(traj["rewards"][si:], gamma=1.0)[: tlen + 1].reshape(
            -1, 1
        )
        if rtg.shape[0] <= tlen:
            rtg = np.concatenate([rtg, np.zeros((1, 1))])

        # padding and state + reward normalization
        act_len = aa.shape[0]
        if tlen != act_len:
            raise ValueError

        ss = np.concatenate([np.zeros((self.max_len - tlen, self.state_dim)), ss])
        ss = (ss - self.state_mean) / self.state_std

        aa = np.concatenate([np.zeros((self.max_len - tlen, self.act_dim)), aa])
        rr = np.concatenate([np.zeros((self.max_len - tlen, 1)), rr])
        dd = np.concatenate([np.ones((self.max_len - tlen)) * 2, dd])
        rtg = (
            np.concatenate([np.zeros((self.max_len - tlen, 1)), rtg])
            * self.reward_scale
        )
        timesteps = np.concatenate([np.zeros((self.max_len - tlen)), timesteps])
        ordering = np.concatenate([np.zeros((self.max_len - tlen)), ordering])
        padding_mask = np.concatenate([np.zeros(self.max_len - tlen), np.ones(tlen)])

        ss = torch.from_numpy(ss).to(dtype=torch.float32)
        aa = torch.from_numpy(aa).to(dtype=torch.float32).clamp(*self.action_range)
        rr = torch.from_numpy(rr).to(dtype=torch.float32)
        dd = torch.from_numpy(dd).to(dtype=torch.long)
        rtg = torch.from_numpy(rtg).to(dtype=torch.float32)
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.long)
        ordering = torch.from_numpy(ordering).to(dtype=torch.long)
        padding_mask = torch.from_numpy(padding_mask)

        return ss, aa, rr, dd, rtg, timesteps, ordering, padding_mask


def create_dataloader(
    trajectories,
    num_iters,
    batch_size,
    max_len,
    state_dim,
    act_dim,
    state_mean,
    state_std,
    reward_scale,
    action_range,
    num_workers=24,
):
    # total number of subt-rajectories you need to sample
    sample_size = batch_size * num_iters
    sampling_ind = sample_trajs(trajectories, sample_size)

    transform = TransformSamplingSubTraj(
        max_len=max_len,
        state_dim=state_dim,
        act_dim=act_dim,
        state_mean=state_mean,
        state_std=state_std,
        reward_scale=reward_scale,
        action_range=action_range,
    )

    subset = SubTrajectory(trajectories, sampling_ind=sampling_ind, transform=transform)

    return torch.utils.data.DataLoader(
        subset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )


def discount_cumsum(x, gamma):
    ret = np.zeros_like(x)
    ret[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        ret[t] = x[t] + gamma * ret[t + 1]
    return ret


def sample_trajs(trajectories, sample_size):

    traj_lens = np.array([len(traj["observations"]) for traj in trajectories])
    p_sample = traj_lens / np.sum(traj_lens)

    inds = np.random.choice(
        np.arange(len(trajectories)),
        size=sample_size,
        replace=True,
        p=p_sample,
    )
    return inds
