from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np
import torch
from gymnasium import spaces

from abc_pickup_extra.sapien3.kinematics_utils import compute_forward_kinematics, compute_jacobian
from concepts.math.rotationlib_wxyz import quat2axisangle
from mani_skill.agents.controllers.base_controller import BaseController, ControllerConfig
from mani_skill.utils import common
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array, DriveMode


class PDJointPosControllerCustom(BaseController):
    config: "PDJointPosControllerCustomConfig"
    _start_qpos = None
    _target_qpos = None

    def _get_joint_limits(self):
        qlimits = (
            self.articulation.get_qlimits()[0, self.active_joint_indices].cpu().numpy()
        )
        # Override if specified
        if self.config.lower is not None:
            qlimits[:, 0] = self.config.lower
        if self.config.upper is not None:
            qlimits[:, 1] = self.config.upper
        return qlimits

    def _initialize_action_space(self):
        joint_limits = self._get_joint_limits()
        low, high = joint_limits[:, 0], joint_limits[:, 1]
        self.single_action_space = spaces.Box(low, high, dtype=np.float32)

    def set_drive_property(self):
        n = len(self.joints)
        self.stiffness = np.broadcast_to(self.config.stiffness, n)
        self.damping = np.broadcast_to(self.config.damping, n)
        self.force_limit = np.broadcast_to(self.config.force_limit, n)
        self.friction = np.broadcast_to(self.config.friction, n)
        self.alpha = self.config.alpha

        for i, joint in enumerate(self.joints):
            drive_mode = self.config.drive_mode
            if not isinstance(drive_mode, str):
                drive_mode = drive_mode[i]
            # disable the built-in controller by setting stiffness and damping to 0
            joint.set_drive_properties(
                0, 0, force_limit=self.force_limit[i].item(), mode="force"
            )
            joint.set_friction(self.friction[i].item())

    def reset(self):
        super().reset()
        self._step = 0  # counter of simulation steps after action is set
        self._last_qvel = self.qvel.clone()
        self._last_f = None
        if self._start_qpos is None:
            self._start_qpos = self.qpos.clone()
        else:

            self._start_qpos[self.scene._reset_mask] = self.qpos[
                self.scene._reset_mask
            ].clone()
        if self._target_qpos is None:
            self._target_qpos = self.qpos.clone()
        else:
            self._target_qpos[self.scene._reset_mask] = self.qpos[
                self.scene._reset_mask
            ].clone()

    def set_drive_targets(self, targets, velocity_targets=None):
        if not self.articulation.get_links()[0].disable_gravity:
            raise NotImplementedError("This controller only works with gravity disabled")
        self._qvel = self.qvel.clone()
        error = targets - self._start_qpos
        qf = self.articulation.get_qf()
        f_1 = common.to_tensor(self.stiffness) * error
        self._last_qvel = self._qvel * self.alpha + self._last_qvel * (1-self.alpha)
        if velocity_targets is None:
            f_2 = -common.to_tensor(self.damping) * self._last_qvel
        else:
            f_2 = -common.to_tensor(self.damping) * (self._last_qvel - common.to_tensor(velocity_targets))
        force_limit = common.to_tensor(self.force_limit).reshape(1, -1)
        # print('f error: ', f_1)
        # print('f damping: ', f_2)
        f = f_1 + f_2
        f = torch.clamp(f, -force_limit, force_limit)
        qf[..., :7] = f
        self.articulation.set_qf(qf)

    def set_action(self, action: Array):
        action = self._preprocess_action(action)
        action = common.to_tensor(action)
        self._step = 0
        self._start_qpos = self.qpos
        if self.config.use_delta:
            if self.config.use_target:
                self._target_qpos = self._target_qpos + action
            else:
                self._target_qpos = self._start_qpos + action
        else:
            # Compatible with mimic controllers. Need to clone here otherwise cannot do in-place replacements in the reset function
            self._target_qpos = torch.broadcast_to(
                action, self._start_qpos.shape
            ).clone()
        if self.config.interpolate:
            self._step_size = (self._target_qpos - self._start_qpos) / self._sim_steps
        else:
            self.set_drive_targets(self._target_qpos)

    def before_simulation_step(self):
        self._step += 1

        # Compute the next target via a linear interpolation
        if self.config.interpolate:
            targets = self._start_qpos + self._step_size * self._step
            self.set_drive_targets(targets)

    def get_state(self) -> dict:
        if self.config.use_target or True:
            return {"target_qpos": self._target_qpos}
        return {}

    def set_state(self, state: dict):
        if self.config.use_target or True:
            self._target_qpos = state["target_qpos"]
            self.set_drive_targets(self._target_qpos)


@dataclass
class PDJointPosControllerCustomConfig(ControllerConfig):
    lower: Union[None, float, Sequence[float]]
    upper: Union[None, float, Sequence[float]]
    stiffness: Union[float, Sequence[float]]
    damping: Union[float, Sequence[float]]
    alpha: float
    force_limit: Union[float, Sequence[float]] = 1e10
    friction: Union[float, Sequence[float]] = 0.0
    use_delta: bool = False
    use_target: bool = False
    interpolate: bool = False
    normalize_action: bool = True
    drive_mode: Union[Sequence[DriveMode], DriveMode] = "force"
    controller_cls = PDJointPosControllerCustom


class PDJointPosMimicControllerCustom(PDJointPosControllerCustom):
    def _get_joint_limits(self):
        joint_limits = super()._get_joint_limits()
        diff = joint_limits[0:-1] - joint_limits[1:]
        assert np.allclose(diff, 0), "Mimic joints should have the same limit"
        return joint_limits[0:1]


class PDJointPosMimicControllerConfig(PDJointPosControllerCustomConfig):
    controller_cls = PDJointPosMimicControllerCustom
