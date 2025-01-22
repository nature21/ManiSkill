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
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/galaxea_r1/r1_upperbody_coacd.urdf"
    load_multiple_collisions = True

    # you may need to use this modify the friction values of some links in order to make it possible to e.g. grasp objects or avoid sliding on the floor
    urdf_config = dict()
    # ['torso_joint1', 'torso_joint2', 'torso_joint3', 'torso_joint4', 'left_arm_joint1', 'right_arm_joint1',
    #  'left_arm_joint2', 'right_arm_joint2', 'left_arm_joint3', 'right_arm_joint3', 'left_arm_joint4',
    #  'right_arm_joint4', 'left_arm_joint5', 'right_arm_joint5', 'left_arm_joint6', 'right_arm_joint6',
    #  'left_gripper_axis1', 'left_gripper_axis2', 'right_gripper_axis1', 'right_gripper_axis2']

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [
                    0.8, # torso_joint_1
                    -1.4, # torso_joint_2
                    -1.0, # torso_joint_3
                    0.0, # torso_joint_4
                    1.56, # left_arm_joint_1
                    -1.56, # right_arm_joint_1
                    2.94, # left_arm_joint_2
                    2.94, # right_arm_joint_2
                    -2.54, # left_arm_joint_3
                    -2.54, # right_arm_joint_3
                    0.0, # left_arm_joint_4
                    0.0, # right_arm_joint_4
                    0.0, # left_arm_joint_5
                    0.0, # right_arm_joint_5
                    0.0, # left_arm_joint_6
                    0.0, # right_arm_joint_6
                    0.0, # left_gripper_axis1
                    0.0, # left_gripper_axis2
                    0.0, # right_gripper_axis1
                    0.0, # right_gripper_axis2
                ]
            ),
            pose=sapien.Pose([-0.4, 0, -0.75]),
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

if __name__ == '__main__':
    import mani_skill.examples.demo_robot as demo_robot_script
    demo_robot_script.main()