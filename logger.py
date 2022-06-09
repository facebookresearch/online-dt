"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

from datetime import datetime
import os
import utils


class Logger:
    def __init__(self, variant):

        self.log_path = self.create_log_path(variant)
        utils.mkdir(self.log_path)
        print(f"Experiment log path: {self.log_path}")

    def log_metrics(self, outputs, iter_num, total_transitions_sampled, writer):
        print("=" * 80)
        print(f"Iteration {iter_num}")
        for k, v in outputs.items():
            print(f"{k}: {v}")
            if writer:
                writer.add_scalar(k, v, iter_num)
                if k == "evaluation/return_mean_gm":
                    writer.add_scalar(
                        "evaluation/return_vs_samples",
                        v,
                        total_transitions_sampled,
                    )

    def create_log_path(self, variant):
        now = datetime.now().strftime("%Y.%m.%d/%H%M%S")
        exp_name = variant["exp_name"]
        prefix = variant["save_dir"]
        return f"{prefix}/{now}-{exp_name}"
