from copy import deepcopy

import numpy as np
import sapien

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.galaxea_r1.r1_arm import GalaxeaR1Arm
from mani_skill.utils import sapien_utils


@register_agent()
class GalaxeaR1LeftArm(GalaxeaR1Arm):
    uid = "galaxea_r1_left_arm"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/galaxea_r1/r1_left_arm_simplified.urdf"
    load_multiple_collisions = True

    # you may need to use this modify the friction values of some links in order to make it possible to e.g. grasp objects or avoid sliding on the floor
    urdf_config = dict()

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [
                    1.16, # left_arm_joint_1
                    2.94, # left_arm_joint_2
                    -2.54, # left_arm_joint_3
                    0.0, # left_arm_joint_4
                    0.0, # left_arm_joint_5
                    0.0, # left_arm_joint_6
                    0.0, # left_gripper_axis1
                    0.0, # left_gripper_axis2
                ]
            ),
            # qpos=np.array(
            #     [
            #         0.0,  # left_arm_joint_1
            #         0.0,  # left_arm_joint_2
            #         0.0,  # left_arm_joint_3
            #         0.0,  # left_arm_joint_4
            #         0.0,  # left_arm_joint_5
            #         0.0,  # left_arm_joint_6
            #         0.0,  # left_gripper_axis1
            #         0.0,  # left_gripper_axis2
            #     ]
            # ),
            pose=sapien.Pose([-0.266794, 2.31713e-06, 0.192149], [0.980131, 0.000445157, 0.198338, 0.00219929])
        ),
        pre_grasp = Keyframe(
            qpos=np.array(
                [-0.38801375, 2.5683095, -1.0184467, 0.0033506425, 1.5687183, -1.5379351, 0.04999999, 0.049999963]
            ),
            pose=sapien.Pose([-0.266794, 2.31713e-06, 0.192149], [0.980131, 0.000445157, 0.198338, 0.00219929])
        )
    )

    def __init__(self, *args, **kwargs):
        self.arm_joint_names = [
            'left_arm_joint1',
            'left_arm_joint2',
            'left_arm_joint3',
            'left_arm_joint4',
            'left_arm_joint5',
            'left_arm_joint6'
        ]

        self.gripper_joint_names = [
            "left_gripper_axis1",
            "left_gripper_axis2",
        ]

        self.ee_link_name = "left_arm_link6"
        super().__init__(*args, **kwargs)

    def _after_init(self):
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )
        self.finger1_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "left_gripper_link1"
        )
        self.finger2_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "left_gripper_link2"
        )
        super()._after_init()
        # self.tcp.set_collision_group_bit(group=2, bit_idx=30, bit=1)
        # self.tcp.set_collision_group_bit(group=3, bit_idx=30, bit=1)
        # self.finger1_link.set_collision_group_bit(group=2, bit_idx=30, bit=1)
        # self.finger2_link.set_collision_group_bit(group=3, bit_idx=30, bit=1)


if __name__ == '__main__':
    import mani_skill.examples.demo_robot as demo_robot_script
    demo_robot_script.main()