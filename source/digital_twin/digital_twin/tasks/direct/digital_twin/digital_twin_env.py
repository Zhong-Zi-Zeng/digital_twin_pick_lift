# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import TiledCamera, FrameTransformer
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform, subtract_frame_transforms
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

from .digital_twin_env_cfg import DigitalTwinEnvCfg


class DigitalTwinEnv(DirectRLEnv):
    cfg: DigitalTwinEnvCfg

    def __init__(self, cfg: DigitalTwinEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ── Joint indices ────────────────────────────────────────────────────
        arm_ids, _ = self.robot.find_joints(
            ["Joint_1", "Joint_2", "Joint_3", "Joint_4", "Joint_5", "Joint_6"]
        )
        self._arm_joint_ids: list[int] = arm_ids   # 6 indices

        gripper_ids, _ = self.robot.find_joints(["gripper_controller"])
        self._gripper_joint_idx: int = int(gripper_ids[0])

        # ── Link_6 body index (for Jacobian lookup only) ────────────────────
        ee_ids, _ = self.robot.find_bodies("Link_6")
        self._ee_body_idx: int = int(ee_ids[0])

        # ── IK controller ───────────────────────────────────────────────────
        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
        )
        self._ik_controller = DifferentialIKController(
            cfg=ik_cfg,
            num_envs=self.num_envs,
            device=self.device,
        )

        # ── EE delta scale (metres per step) ────────────────────────────────
        self._ee_delta_scale: float = 0.005  # 5 mm per step

        # ── Task height constants (metres) ───────────────────────────────────
        self._init_obj_z: float = self.cfg.object_cfg.init_state.pos[2]  # 0.85 m
        self._target_obj_z: float = self._init_obj_z + 0.05              # 0.90 m (5 cm lift)
        self._drop_threshold: float = self._init_obj_z - 0.10            # 0.75 m

        # ── Gripper position limits ──────────────────────────────────────────
        self._gripper_closed: float = -0.58
        self._gripper_open: float = 0.14

        # ── Actions buffers (for obs) ────────────────────────────────────────
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._prev_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

        print("[DigitalTwinEnv] Robot bodies:", self.robot.body_names)
        print("[DigitalTwinEnv] EE body idx:", self._ee_body_idx)
        print("[DigitalTwinEnv] Arm joint ids:", self._arm_joint_ids)

    # ── Scene setup ─────────────────────────────────────────────────────────

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)

        env = sim_utils.UsdFileCfg(usd_path=self.cfg.env_usd, scale=(1.0, 1.0, 1.0))
        env.func(prim_path="/World/envs/env_.*/scene", cfg=env)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.object = RigidObject(self.cfg.object_cfg)

        # self.scene.sensors["camera"] = self.camera
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object
        self.ee_frame = FrameTransformer(self.cfg.ee_frame)
        self.scene.sensors["ee_frame"] = self.ee_frame

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        light_cfg = sim_utils.DomeLightCfg(
            intensity=1000.0, color_temperature=6500, color=(1.0, 1.0, 1.0)
        )
        light_cfg.func(
            "/World/Light", light_cfg,
            translation=(0.0, 0.0, 2.5),
            orientation=(0.5, 0.5, 0.5, 0.5),
        )

    # ── Action pipeline ──────────────────────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._prev_actions = self.actions.clone()
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # actions: (N, 7) = [dx, dy, dz, drx, dry, drz, gripper]
        a = self.actions  # (N, 7)

        # ── EE delta command: pos (3) + rot (3) scaled ──────────────────────
        ee_cmd = a[:, :6] * self._ee_delta_scale  # (N, 6)

        # ── Current EE pose (from FrameTransformer, includes offset) ──────
        ee_pos_w = self.ee_frame.data.target_pos_w[:, 0, :]    # (N, 3)
        ee_quat_w = self.ee_frame.data.target_quat_w[:, 0, :]  # (N, 4) wxyz
        ee_pos = ee_pos_w - self.scene.env_origins  # env-local

        # ── Jacobian ────────────────────────────────────────────────────────
        J_all = self.robot.root_physx_view.get_jacobians()  # (N, num_bodies, 6, num_dof)
        J = J_all[:, self._ee_body_idx, :, self._arm_joint_ids]  # (N, 6, 6)

        # ── IK solve ────────────────────────────────────────────────────────
        q_arm = self.robot.data.joint_pos[:, self._arm_joint_ids]  # (N, 6)

        self._ik_controller.set_command(ee_cmd, ee_pos=ee_pos, ee_quat=ee_quat_w)
        q_des = self._ik_controller.compute(
            jacobian=J,
            joint_pos=q_arm,
            ee_pos=ee_pos,
            ee_quat=ee_quat_w,
        )  # (N, 6)

        # ── Gripper: binary open / close ────────────────────────────────────
        gripper_open = a[:, 6] > 0.0
        gripper_target = torch.where(
            gripper_open,
            torch.full((self.num_envs,), self._gripper_open, device=self.device),
            torch.full((self.num_envs,), self._gripper_closed, device=self.device),
        )  # (N,)

        # ── Assemble full joint target ──────────────────────────────────────
        full_target = self.robot.data.joint_pos.clone()
        full_target[:, self._arm_joint_ids] = q_des
        full_target[:, self._gripper_joint_idx] = gripper_target

        self.robot.set_joint_position_target(full_target)

    # ── Observation ──────────────────────────────────────────────────────────

    def _get_observations(self) -> dict:
        # Robot root pose (for frame transforms)
        robot_pos_w  = self.robot.data.root_pos_w   # (N, 3)
        robot_quat_w = self.robot.data.root_quat_w  # (N, 4)

        # Object position in robot root frame (same as Isaac Lab lift mdp)
        obj_pos_w = self.object.data.root_pos_w     # (N, 3)
        obj_pos_b, _ = subtract_frame_transforms(robot_pos_w, robot_quat_w, obj_pos_w)  # (N, 3)

        # EE position in robot root frame
        ee_pos_w = self.ee_frame.data.target_pos_w[:, 0, :]              # (N, 3)
        ee_pos_b, _ = subtract_frame_transforms(robot_pos_w, robot_quat_w, ee_pos_w)    # (N, 3)

        # joint pos relative to default, joint vel relative to default
        joint_pos_rel = (
            self.robot.data.joint_pos[:, self._arm_joint_ids]
            - self.robot.data.default_joint_pos[:, self._arm_joint_ids]
        )  # (N, 6)
        joint_vel_rel = (
            self.robot.data.joint_vel[:, self._arm_joint_ids]
            - self.robot.data.default_joint_vel[:, self._arm_joint_ids]
        )  # (N, 6)

        # obs = [joint_pos_rel (6), joint_vel_rel (6), obj_pos_b (3), ee_pos_b (3), last_action (4)] = 22
        obs = torch.cat([joint_pos_rel, joint_vel_rel, obj_pos_b, ee_pos_b, self._prev_actions], dim=-1)
        return {"policy": obs}

    # ── Reward ───────────────────────────────────────────────────────────────

    def _get_rewards(self) -> torch.Tensor:
        ee_pos_w = self.ee_frame.data.target_pos_w[:, 0, :]  # (N, 3)
        obj_pos_w = self.object.data.root_pos_w               # (N, 3)

        # 1) Reaching reward: 1 - tanh(dist / std), weight=1.0  [same as object_ee_distance]
        dist_ee_obj = torch.norm(obj_pos_w - ee_pos_w, dim=-1)
        r_reach = (1.0 - torch.tanh(dist_ee_obj / 0.1)) * 1.0

        # 2) Grasp reward: encourage closing gripper when EE is within 1~2cm of object
        gripper_pos = self.robot.data.joint_pos[:, self._gripper_joint_idx]
        gripper_closed_ratio = (self._gripper_open - gripper_pos) / (self._gripper_open - self._gripper_closed)
        gripper_closed_ratio = torch.clamp(gripper_closed_ratio, 0.0, 1.0)
        proximity = torch.clamp((0.02 - dist_ee_obj) / 0.01, 0.0, 1.0)  # 0 at >=2cm, 1 at <=1cm
        r_grasp = proximity * gripper_closed_ratio * 5.0

        # 3) Lifting reward: binary 1.0 when object is barely off the table (~2cm), weight=15.0  [same as object_is_lifted]
        is_lifted = (obj_pos_w[:, 2] > self._init_obj_z + 0.02).float()
        r_lift = is_lifted * 15.0

        # 3) Goal tracking (coarse): only when lifted, tanh-kernel on dist to target height, weight=16.0
        #    [analogous to object_goal_tracking — provides gradient from table → target]
        height_diff = torch.abs(obj_pos_w[:, 2] - self._target_obj_z)
        r_goal_coarse = is_lifted * (1.0 - torch.tanh(height_diff / 0.3)) * 16.0

        # 4) Goal tracking (fine): same but tighter kernel, weight=5.0  [object_goal_tracking_fine_grained]
        r_goal_fine = is_lifted * (1.0 - torch.tanh(height_diff / 0.05)) * 5.0

        return r_reach + r_grasp + r_lift + r_goal_coarse + r_goal_fine

    # ── Done conditions ──────────────────────────────────────────────────────

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        obj_pos_local = self.object.data.root_pos_w - self.scene.env_origins  # (N, 3)
        obj_z = obj_pos_local[:, 2]

        # Terminate if object dropped below threshold
        dropped = obj_z < self._drop_threshold

        # Success if object lifted to target height
        lifted = obj_z >= self._target_obj_z

        terminated = dropped | lifted
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, truncated

    # ── Episode reset ────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        super()._reset_idx(env_ids)

        # Reset robot joints to default pose
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._prev_actions[env_ids] = 0.0

        # Reset object with small XY randomisation (±3 cm)
        obj_state = self.object.data.default_root_state[env_ids].clone()
        obj_state[:, :3] += self.scene.env_origins[env_ids]  # local → world frame
        xy_noise = sample_uniform(-0.03, 0.03, (len(env_ids), 2), device=self.device)
        obj_state[:, :2] += xy_noise
        obj_state[:, 7:] = 0.0   # zero velocities
        self.object.write_root_state_to_sim(obj_state, env_ids=env_ids)
