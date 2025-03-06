from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np
import sapien
import torch
from gymnasium import spaces

from mani_skill.agents.controllers.utils.kinematics import compute_forward_kinematics, compute_jacobian
from concepts.math.rotationlib_wxyz import quat2axisangle
from mani_skill.agents.controllers.base_controller import BaseController, ControllerConfig
from mani_skill.utils import common
from mani_skill.utils.geometry.rotation_conversions import quaternion_multiply, quaternion_invert
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array, DriveMode


class PDJointPosControllerCompliant(BaseController):
    config: "PDJointPosControllerCompliantConfig"
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
        self.ee_stiffness = common.to_tensor(np.broadcast_to(self.config.ee_stiffness, 6))
        self.ee_damping = common.to_tensor(np.broadcast_to(self.config.ee_damping, 6))
        self.null_space_stiffness = common.to_tensor(np.broadcast_to(self.config.null_space_stiffness, n))
        self.null_space_damping = common.to_tensor(np.broadcast_to(self.config.null_space_damping, n))
        self.force_limit = common.to_tensor(np.broadcast_to(self.config.force_limit, n))
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
        self._global_step = 0  # counter of simulation steps after reset
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

        current_ee_pose = self.articulation.get_links()[7].pose
        ee_pose_target = compute_forward_kinematics(self.pinocchio_model, 7, targets)
        # TODO (Yuyao Liu @ 2024-11-26): the current error is not computed correctly. Refer to https://github.com/rachelholladay/franka_ros_interface/blob/kinetic/franka_ros_controllers/src/cartesian_impedance_controller.cpp
        # We use sapien.Pose here because it is much more efficient than maniskill.Pose for inv() operation
        # ee_pose_target = sapien.Pose(ee_pose_target.p, ee_pose_target.q)
        # ee_error = ee_pose_target * sapien.Pose(current_ee_pose.p[0].cpu().numpy(), current_ee_pose.q[0].cpu().numpy()).inv()
        ee_pose_target = Pose.create_from_pq(ee_pose_target.p, ee_pose_target.q)
        ee_error_translation = ee_pose_target.p - current_ee_pose.p
        ee_error_q = quaternion_multiply(ee_pose_target.q, quaternion_invert(current_ee_pose.q)).cpu().numpy()
        ee_error_axis, ee_error_angle = quat2axisangle(ee_error_q.reshape(-1))
        ee_error_axis_angle = ee_error_angle * ee_error_axis
        ee_error_axis_angle = common.to_tensor(ee_error_axis_angle).reshape(1, 3)
        ee_error_spatial_twist = torch.cat([ee_error_translation, ee_error_axis_angle], dim=-1)

        jacobian = torch.from_numpy(compute_jacobian(self.pinocchio_model, 7, targets, local=False)).float()
        jacobian_pinv = torch.pinverse(jacobian)

        # task space control
        self._qvel = self.qvel.clone()
        ee_vel = self._qvel[:7] @ jacobian.T
        last_ee_vel = self._last_qvel[:7] @ jacobian.T
        # alpha = 0.1
        # alpha = 1
        # ee_stiffness = common.to_tensor([75] * 3 + [7.5] * 3)
        # ee_stiffness = common.to_tensor([150] * 3 + [15] * 3)
        # ee_damping = 2 * torch.sqrt(ee_stiffness)
        ee_vel_momentum = self.alpha * ee_vel + (1 - self.alpha) * last_ee_vel
        ee_f_1 = self.ee_stiffness * ee_error_spatial_twist
        ee_f_2 = -self.ee_damping * ee_vel_momentum
        f_task = (ee_f_1 + ee_f_2) @ jacobian

        # null space control
        # null_space_stiffness = common.to_tensor([150] * 4 + [75] * 3) / 2
        # null_space_damping = 2 * torch.sqrt(null_space_stiffness)
        # null_space_damping[6] = null_space_damping[6] / 2
        null_space_error = targets - self._start_qpos
        null_space_f_1 = self.null_space_stiffness * null_space_error
        self._last_qvel = self._qvel * self.alpha + self._last_qvel * (1 - self.alpha)
        null_space_f_2 = -self.null_space_damping * self._last_qvel
        I = torch.eye(7)
        null_space_projection = I - jacobian_pinv @ jacobian
        f_null = (null_space_f_1 + null_space_f_2) @ null_space_projection.T
        # f_null = 0

        qf = self.articulation.get_qf()
        force_limit = self.force_limit.reshape(1, -1)
        # print(ee_f_1, ee_f_2, qf, ee_error, ee_error_spatial_twist)
        if self._global_step % 1000 == 0:
            # print('ee_error: ', ee_error)
            print('ee_error_spatial_twist: ', ee_error_spatial_twist)
            print('ee_f_1: ', ee_f_1)
            print('ee_f_2: ', ee_f_2)
            print('f_task: ', f_task)
            print('f_null: ', f_null)
            print('-'*50)
        f = f_task + f_null
        f = torch.clamp(f, -force_limit, force_limit)

        # saturate the force
        # delta_f_max = 10
        # if self._last_f is not None:
        #     difference = f - self._last_f
        #     f = self._last_f + torch.clamp(difference, -delta_f_max, delta_f_max)
        #     self._last_f = f
        # else:
        #     self._last_f = torch.zeros_like(f)
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
        self._global_step += 1

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
class PDJointPosControllerCompliantConfig(ControllerConfig):
    lower: Union[None, float, Sequence[float]]
    upper: Union[None, float, Sequence[float]]
    ee_stiffness: Union[float, Sequence[float]]
    ee_damping: Union[float, Sequence[float]]
    null_space_stiffness: Union[float, Sequence[float]]
    null_space_damping: Union[float, Sequence[float]]
    alpha: float
    force_limit: Union[float, Sequence[float]] = 1e10
    friction: Union[float, Sequence[float]] = 0.0
    use_delta: bool = False
    use_target: bool = False
    interpolate: bool = False
    normalize_action: bool = True
    drive_mode: Union[Sequence[DriveMode], DriveMode] = "force"
    controller_cls = PDJointPosControllerCompliant


class PDJointPosMimicControllerCompliant(PDJointPosControllerCompliant):
    def _get_joint_limits(self):
        joint_limits = super()._get_joint_limits()
        diff = joint_limits[0:-1] - joint_limits[1:]
        assert np.allclose(diff, 0), "Mimic joints should have the same limit"
        return joint_limits[0:1]


class PDJointPosMimicControllerConfig(PDJointPosControllerCompliantConfig):
    controller_cls = PDJointPosMimicControllerCompliant
