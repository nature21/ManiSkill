import numpy as np
import sapien

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig


@register_agent()
class GalaxeaR1UpperBody(BaseAgent):
    uid = "galaxea_r1_upperbody"
    # urdf_path = f"{PACKAGE_ASSET_DIR}/robots/galaxea_r1/r1_upperbody.urdf"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/galaxea_r1/r1_upperbody_glb.urdf"
    load_multiple_collisions = True

    # you may need to use this modify the friction values of some links in order to make it possible to e.g. grasp objects or avoid sliding on the floor
    urdf_config = dict()

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [0.0]*16
            ),
            pose=sapien.Pose(),
        )
    )

    # you will need to implement your controllers to control the robot here. If not implemented, ManiSkill will create two default controllers.
    # One does PD joint position control, and the other is PD joint delta position control
    # @property
    # def _controller_configs(self):
    #     raise NotImplementedError()

    # @property
    # def _sensor_configs(self):
    #     # Add custom cameras mounted to a link on the robot, or remove this if there aren't any you wish to simulate
    #     return [
    #         CameraConfig(
    #             uid="your_custom_camera_on_this_robot",
    #             p=[0.0464982, -0.0200011, 0.0360011],
    #             q=[0, 0.70710678, 0, 0.70710678],
    #             width=128,
    #             height=128,
    #             fov=1.57,
    #             near=0.01,
    #             far=100,
    #             entity_uid="your_mounted_camera",
    #         )
    #     ]
