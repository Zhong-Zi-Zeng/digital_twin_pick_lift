"""Train pick-and-lift with CNN policy (image + vector observations)."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train CNN RL agent with skrl.")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default="Template-Digital-Twin-Direct-v0")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--checkpoint", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True  # required for TiledCamera

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Imports after sim launch ──────────────────────────────────────────────────

import os
import gymnasium as gym
from datetime import datetime

import torch
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
from skrl.resources.preprocessors.torch import RunningStandardScaler

from isaaclab.envs import DirectRLEnvCfg
from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
import digital_twin.tasks  # noqa: F401

from digital_twin.tasks.direct.digital_twin.digital_twin_env_cfg import DigitalTwinEnvCfg
from digital_twin.tasks.direct.digital_twin.agents.models import CNNPolicy, CNNValue


def main():
    # ── Create env ────────────────────────────────────────────────────────────
    env_cfg = DigitalTwinEnvCfg()
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    device = env.device

    # ── Models ────────────────────────────────────────────────────────────────
    policy = CNNPolicy(env.observation_space, env.action_space, device=device)
    value  = CNNValue(env.observation_space, env.action_space, device=device)
    models = {"policy": policy, "value": value}

    # ── Memory ────────────────────────────────────────────────────────────────
    ROLLOUTS = 32
    memory = RandomMemory(memory_size=ROLLOUTS, num_envs=env.num_envs, device=device)

    # ── PPO config ────────────────────────────────────────────────────────────
    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = ROLLOUTS
    cfg["learning_epochs"] = 8
    cfg["mini_batches"] = 8
    cfg["discount_factor"] = 0.99
    cfg["lambda"] = 0.95
    cfg["learning_rate"] = 5e-4
    cfg["learning_rate_scheduler"] = None
    cfg["grad_norm_clip"] = 1.0
    cfg["ratio_clip"] = 0.2
    cfg["value_clip"] = 0.2
    cfg["clip_predicted_values"] = True
    cfg["entropy_loss_scale"] = 0.0
    cfg["value_loss_scale"] = 2.0
    cfg["rewards_shaper_scale"] = 0.1
    cfg["state_preprocessor"] = None
    cfg["state_preprocessor_kwargs"] = {}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    cfg["seed"] = args_cli.seed

    log_dir = os.path.join(
        "logs", "skrl", "digital_twin_pick_lift",
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_ppo_cnn",
    )
    cfg["experiment"] = {
        "directory": os.path.abspath(log_dir),
        "experiment_name": "",
        "write_interval": "auto",
        "checkpoint_interval": "auto",
    }

    agent = PPO(
        models=models,
        memory=memory,
        cfg=cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    if args_cli.checkpoint:
        print(f"[INFO] Loading checkpoint: {args_cli.checkpoint}")
        agent.load(args_cli.checkpoint)

    # ── Trainer ───────────────────────────────────────────────────────────────
    timesteps = 500_000
    if args_cli.max_iterations:
        timesteps = args_cli.max_iterations * ROLLOUTS

    trainer = SequentialTrainer(
        cfg={"timesteps": timesteps, "environment_info": "log"},
        env=env,
        agents=agent,
    )
    trainer.train()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
