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
class GalaxeaR1RightArm(GalaxeaR1Arm):
    uid = "galaxea_r1_right_arm"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/galaxea_r1/r1_right_arm_simplified.urdf"
    load_multiple_collisions = True

    # you may need to use this modify the friction values of some links in order to make it possible to e.g. grasp objects or avoid sliding on the floor
    urdf_config = dict()

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [
                    -1.16, # right_arm_joint_1
                    2.94, # right_arm_joint_2
                    -2.54, # right_arm_joint_3
                    0.0, # right_arm_joint_4
                    0.0, # right_arm_joint_5
                    0.0, # right_arm_joint_6
                    0.0, # right_gripper_axis1
                    0.0, # right_gripper_axis2
                ]
            ),
            # qpos=np.array(
            #     [
            #         0.0,  # right_arm_joint_1
            #         0.0,  # right_arm_joint_2
            #         0.0,  # right_arm_joint_3
            #         0.0,  # right_arm_joint_4
            #         0.0,  # right_arm_joint_5
            #         0.0,  # right_arm_joint_6
            #         0.0,  # right_gripper_axis1
            #         0.0,  # right_gripper_axis2
            #     ]
            # ),
            pose=sapien.Pose([-0.266794, 2.31713e-06, 0.192149], [0.980131, 0.000445157, 0.198338, 0.00219929])
        ),
        pre_grasp = Keyframe(
            qpos=np.array(
                [0.38926113, 2.5617487, -1.0116353, -0.017751142, -1.5716947, -1.5902961, 0.049999982, 0.049999975]
            ),
            pose=sapien.Pose([-0.266794, 2.31713e-06, 0.192149], [0.980131, 0.000445157, 0.198338, 0.00219929])
        )
    )

    def __init__(self, *args, **kwargs):
        self.arm_joint_names = [
            'right_arm_joint1',
            'right_arm_joint2',
            'right_arm_joint3',
            'right_arm_joint4',
            'right_arm_joint5',
            'right_arm_joint6'
        ]

        self.gripper_joint_names = [
            "right_gripper_axis1",
            "right_gripper_axis2",
        ]

        self.ee_link_name = "right_arm_link6"
        super().__init__(*args, **kwargs)

    def _after_init(self):
        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )
        self.finger1_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "right_gripper_link1"
        )
        self.finger2_link = sapien_utils.get_obj_by_name(
            self.robot.get_links(), "right_gripper_link2"
        )
        # self.tcp.set_collision_group_bit(group=2, bit_idx=31, bit=1)
        # self.tcp.set_collision_group_bit(group=3, bit_idx=31, bit=1)
        # self.finger1_link.set_collision_group_bit(group=2, bit_idx=31, bit=1)
        # self.finger2_link.set_collision_group_bit(group=3, bit_idx=31, bit=1)


if __name__ == '__main__':
    import mani_skill.examples.demo_robot as demo_robot_script
    demo_robot_script.main()