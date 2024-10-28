from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np
import torch
from gymnasium import spaces

from mani_skill.agents.controllers.pd_joint_pos_custom import PDJointPosControllerCustom, PDJointPosControllerCustomConfig


class PDJointPosVelControllerCustom(PDJointPosControllerCustom):
    config: "PDJointPosVelControllerCustomConfig"
    _target_qvel = None

    def _initialize_action_space(self):
        joint_limits = self._get_joint_limits()
        pos_low, pos_high = joint_limits[:, 0], joint_limits[:, 1]
        vel_low = np.broadcast_to(self.config.vel_lower, pos_low.shape)
        vel_high = np.broadcast_to(self.config.vel_upper, pos_high.shape)
        low = np.float32(np.hstack([pos_low, vel_low]))
        high = np.float32(np.hstack([pos_high, vel_high]))
        self.single_action_space = spaces.Box(low, high, dtype=np.float32)

    def reset(self):
        super().reset()
        if self._target_qvel is None:
            self._target_qvel = self.qvel.clone()
        else:
            self._target_qvel[self.scene._reset_mask] = torch.zeros_like(
                self._target_qpos[self.scene._reset_mask], device=self.device
            )


    def set_action(self, action: np.ndarray):
        action = self._preprocess_action(action)
        nq = len(action[0]) // 2

        self._step = 0
        self._start_qpos = self.qpos

        if self.config.use_delta:
            if self.config.use_target:
                self._target_qpos = self._target_qpos + action[:, :nq]
            else:
                self._target_qpos = self._start_qpos + action[:, :nq]
        else:
            # Compatible with mimic
            self._target_qpos = torch.broadcast_to(
                action[:, :nq], self._start_qpos.shape
            )

        self._target_qvel = action[:, nq:]

        if self.config.interpolate:
            self._step_size = (self._target_qpos - self._start_qpos) / self._sim_steps
        else:
            self.set_drive_targets(self._target_qpos, self._target_qvel)


    def before_simulation_step(self):
        self._step += 1

        # Compute the next target via a linear interpolation
        if self.config.interpolate:
            targets = self._start_qpos + self._step_size * self._step
            self.set_drive_targets(targets, self._target_qvel)


@dataclass
class PDJointPosVelControllerCustomConfig(PDJointPosControllerCustomConfig):
    controller_cls = PDJointPosVelControllerCustom
    vel_lower: Union[float, Sequence[float]] = -1.0
    vel_upper: Union[float, Sequence[float]] = 1.0
