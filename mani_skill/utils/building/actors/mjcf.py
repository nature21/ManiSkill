from typing import Optional, Union
import os

import sapien
from mani_skill.utils.building.actor_builder import ActorBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array
from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.io_utils import load_json
from mani_skill.utils.building.actors.common import _build_by_type
from mani_skill import PACKAGE_ASSET_DIR

MJCF_DIR = os.path.join(PACKAGE_ASSET_DIR, 'mjcf_models')
MJCF_FILES = {
    'needle': 'needle_obj.xml', 
    'tripod': 'tripod_obj.xml'
}

def build_model(
    scene: ManiSkillScene,
    model: str,
    name: str,
):
    assert model in MJCF_FILES, f'{model} not in {MJCF_FILES.keys()}'
    mjcf_file = os.path.join(MJCF_DIR, MJCF_FILES[model])
    loader = scene.create_mjcf_loader()
    builders = loader.parse(str(mjcf_file))
    actor_builder: ActorBuilder = builders["actor_builders"][0]
    actor_builder.set_initial_pose(Pose.create_from_pq([0, 0, 0], [1, 0, 0, 0]))
    return actor_builder.build_dynamic(name=name)
    
    