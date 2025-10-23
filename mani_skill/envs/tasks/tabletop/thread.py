from typing import Dict, Union, Any

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.utils.geometry.rotation_conversions import axis_angle_to_quaternion, quaternion_multiply
from mani_skill.utils.structs.pose import Pose
from mani_skill.agents.robots.fetch.fetch import Fetch
from mani_skill.agents.robots.panda.panda import Panda
from mani_skill.agents.robots.panda.panda_wristcam import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building.actors.mjcf import build_model
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


@register_env("Thread-v1", max_episode_steps=100)
class ThreadEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda", "panda_wristcam", "fetch"]
    agent: Union[Panda, PandaWristCam, Fetch]
    needle_height = 0.02
    tripod_height = 0.096
    ring_size = 0.012
    needle_pick_up_point = np.array([0.0, 0.06, 0.0])
    tripod_insert_point = np.array([0.0, 0.0, 0.088])
    tripod_insert_direction = np.array([1.0, 0.0, 0.0])
    
    needle_bound = dict(
        x=(-0.2, 0.05),
        y=(0.15, 0.25),
        z_rot=(-7. * np.pi / 6., np.pi / 6.),
    )
    tripod_bound = dict(
        x=(-0.1, 0.15),
        y=(-0.2, -0.1),
        z_rot=(np.pi / 6., 5. * np.pi / 6.),
    )

    def __init__(
        self,
        *args,
        robot_uids="panda",
        robot_init_qpos_noise=0.02,
        num_envs=1,
        reconfiguration_freq=None,
        control_freq=10,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.model_id = None
        if reconfiguration_freq is None:
            if num_envs == 1:
                reconfiguration_freq = 1
            else:
                reconfiguration_freq = 0
        self.target_id_list = None
        self.name_to_points_map = {
            'robot_id_list': 'num_robot_points',
            'target_id_list': 'num_target_points',
        }
        
        # walk around: self.control_freq is not settable
        self.sim_control_freq = control_freq

        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            num_envs=num_envs,
            **kwargs,
        )

        self.robot_id_list = []
        for obj_id, obj in sorted(self.segmentation_id_map.items()):
            if "panda" in obj.name:
                self.robot_id_list.append(obj_id)

    def reset(self, seed: Union[None, int, list[int]] = None, options: Union[None, dict] = None):
        '''
        no matter reconfigure is true or false, always set reconfigure to true
        '''
        if options is None:
            options = dict()
        options["reconfigure"] = True
        return super().reset(seed=seed, options=options)

    def goal_pose(self):
        init_target_pos = self.tripod_insert_point
        init_target_quat = axis_angle_to_quaternion(torch.tensor([0, 0, -torch.pi/2]).view(1, 3)).cpu().numpy().squeeze()
        init_target_pose = Pose.create_from_pq(init_target_pos, init_target_quat)
        return self.tripod.pose * init_target_pose
    
    @property
    def _default_sim_config(self):
        return SimConfig(
            control_freq=self.sim_control_freq,
        )

    # @property
    # def _default_sensor_configs(self):
    #     pose = sapien_utils.look_at(eye=[0.3, 0.3, 0.6], target=[-0.1, 0, 0.2])
    #     return [CameraConfig("base_camera", pose, 768, 768, np.pi / 2, 0.01, 100)]

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0.0, 0.6], target=[-0.1, 0, 0.1])
        pose = sapien_utils.look_at(eye=[1.3, 0.0, 1.35], target=[1.3-1, 0, 1.35-1])
        return [CameraConfig("base_camera", pose, 256, 256, np.pi / 8, 0.01, 100)]

    # @property
    # def _default_human_render_camera_configs(self):
    #     pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
    #     return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)
    
    @property
    def _default_human_render_camera_configs(self):
        # pose = sapien_utils.look_at(eye=[0.3, 0.0, 0.35], target=[-0.2, 0, -0.15])
        pose = sapien_utils.look_at(eye=[1.3, 0.0, 1.35], target=[1.3-1, 0, 1.35-1])
        # print(pose)
        # exit()
        return [CameraConfig("base_camera", pose, 1024, 1024, np.pi / 8, 0.01, 100)]

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        global WARNED_ONCE
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build(material=[1, 1, 1])

        assert self.num_envs == 1, "Only support single env for now"
        
        self.needle = build_model(
            self.scene,
            model="needle",
            name='needle',
        )
        
        self.tripod = build_model(
            self.scene,
            model="tripod",
            name='tripod',
        )

    def _after_reconfigure(self, options: dict):
        pass
    
    @staticmethod
    def generate_pos(num_envs, bound_dict):
        """
        example:
        bound_dict = dict(
            x=(-0.2, 0.05),
            y=(0.15, 0.25),
            z_rot=(-7. * np.pi / 6., np.pi / 6.),
        )
        """
        xyz = torch.zeros((num_envs, 3))
        xyz[:, 0] = torch.rand((num_envs, 1)) * (bound_dict['x'][1] - bound_dict['x'][0]) + bound_dict['x'][0]
        xyz[:, 1] = torch.rand((num_envs, 1)) * (bound_dict['y'][1] - bound_dict['y'][0]) + bound_dict['y'][0]
        xyz[:, 2] = 0.1
        angle = torch.rand((num_envs, 1)) * (bound_dict['z_rot'][1] - bound_dict['z_rot'][0]) + bound_dict['z_rot'][0]
        angle = angle.view(-1, 1) # (num_envs, 1)
        axis = torch.tensor([0, 0, 1], device=xyz.device).view(1, 3).expand(num_envs, -1) # (num_envs, 3)
        axis_angle = axis * angle # (num_envs, 3)
        q = axis_angle_to_quaternion(axis_angle) # (num_envs, 4)
        
        return xyz, q
    
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz_needle, q_needle = self.generate_pos(b, self.needle_bound)
            xyz_tripod, q_tripod = self.generate_pos(b, self.tripod_bound)
            xyz_needle[:, 2] = self.needle_height
            xyz_tripod[:, 2] = self.tripod_height
            qs = randomization.random_quaternions(b*2, lock_x=True, lock_y=True)
            qs = qs.view(b, 2, 4)

            self.needle.set_pose(Pose.create_from_pq(xyz_needle, q_needle))
            self.tripod.set_pose(Pose.create_from_pq(xyz_tripod, q_tripod))
            
            if self.robot_uids == "panda" or self.robot_uids == "panda_wristcam":
                # fmt: off
                qpos = np.array(
                    [0.0, 0, 0, -np.pi * 2 / 3, 0, np.pi * 2 / 3, np.pi / 4, 0.04, 0.04]
                )
                # fmt: on
                qpos[:-2] += self._episode_rng.normal(
                    0, self.robot_init_qpos_noise, len(qpos) - 2
                )
                self.agent.reset(qpos)
                self.agent.robot.set_root_pose(sapien.Pose([-0.615, 0, 0]))

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        return obs

    def evaluate(self):
        # target_shape = self.shapes[0]
        # is_touching = self.agent.is_touching(target_shape)
        # is_robot_static = self.agent.is_static(0.2)
        
        # target_position_xy = target_shape.pose.p[..., :2]
        # robot_position_xy = self.agent.tcp.pose.p[..., :2]
        # distance = torch.norm(target_position_xy - robot_position_xy, dim=-1)
        # is_centering = distance < self.centering_threshold
        
        # return {
        #     "success": is_touching & is_robot_static & is_centering,
        #     "is_touching": is_touching,
        #     "is_centering": is_centering,
        #     "is_robot_static": is_robot_static,
        # }
        return {
            "success": torch.zeros(self.num_envs, device=self.device, dtype=torch.bool),
        }

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # TODO: this is dummy reward
        return torch.zeros(self.num_envs, device=self.device)


