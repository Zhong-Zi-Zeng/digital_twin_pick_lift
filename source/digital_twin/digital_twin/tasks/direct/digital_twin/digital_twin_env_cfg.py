# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.markers import FRAME_MARKER_CFG
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils import configclass

from digital_twin.tasks.robots import JETCOBOT_CFG

ASSET_DIR = Path.cwd() / "assets"

# ── FrameTransformer visualiser ─────────────────────────────────────────
_MARKER_CFG = FRAME_MARKER_CFG.copy()
_MARKER_CFG.markers["frame"].scale = (0.05, 0.05, 0.05)
_MARKER_CFG.prim_path = "/Visuals/FrameTransformer"

@configclass
class DigitalTwinEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0
    # spaces definition
    # action_space  : arm joints (6) + gripper (1) = 7
    # observation_space: joint_pos_rel (6) + joint_vel_rel (6) + obj_pos_b (3) + ee_pos_b (3) + last_action (7)= 25
    action_space = 7
    observation_space = 25
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = JETCOBOT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(  
                pos=(-0.21786, 0.48402, 0.73747),
                rot=(0.7071, 0, 0, -0.7071),
                joint_pos={  
                    "Joint_1": 0.0,
                    "Joint_2": -0.757,      
                    "Joint_3": -0.678,  
                    "Joint_4": 0.0,  
                    "Joint_5": 0.0,  
                    "Joint_6": 0.0,  
                    "gripper_controller": -0.422,  
                }
            )
        )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True)

    # end-effector frame transformer (true EE = Link_6 + offset)
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/jetcobot/base_link",
        debug_vis=True,
        visualizer_cfg=_MARKER_CFG,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/jetcobot/Link_6",
                name="end_effector",
                offset=OffsetCfg(
                    pos=[0.12374, 0.0, 0.01131],
                ),
            ),
        ],
    )

    # USD paths
    env_usd: str = str(ASSET_DIR / "lab_flatten.usd")

    # Cube config
    cube_properties = RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        )
    
    object_cfg: str = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Cube",
            init_state=RigidObjectCfg.InitialStateCfg
                (pos=[-0.2216, 0.30368, 0.8], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/blue_block.usd",
                scale=(0.3, 0.3, 0.3),
                rigid_props=cube_properties,
                semantic_tags=[("class", "cube_1")],
            ),
        )
   