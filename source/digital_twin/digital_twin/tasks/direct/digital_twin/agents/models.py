from __future__ import annotations

import torch
import torch.nn as nn

from skrl.models.torch import GaussianMixin, DeterministicMixin, Model

VEC_OBS_DIM = 19          # joint_pos_rel(6) + joint_vel_rel(6) + prev_actions(7)
IMG_C, IMG_H, IMG_W = 3, 128, 128
PIXEL_FLAT_DIM = IMG_C * IMG_H * IMG_W  # 49152


def _split_obs(flat_obs: torch.Tensor):
    """Split flat observation into vector and image parts."""
    vec = flat_obs[:, :VEC_OBS_DIM]                          # (N, 19)
    pixels = flat_obs[:, VEC_OBS_DIM:]                       # (N, 49152)
    pixels = pixels.view(-1, IMG_C, IMG_H, IMG_W)            # (N, 3, 128, 128)
    return vec, pixels


class CNNPolicy(GaussianMixin, Model):
    """CNN (pixels) + MLP (vector) policy for pick-and-lift.

    Input: flat tensor (N, 49171) = [vec(19), pixels_flat(49152)]
    Output: Gaussian action distribution over (N, 7)
    """

    def __init__(self, observation_space, action_space, device,
                 clip_actions=False, clip_log_std=True,
                 min_log_std=-20.0, max_log_std=2.0):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)

        # ── CNN encoder for image ────────────────────────────────────────────
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            cnn_out_dim = self.cnn(torch.zeros(1, IMG_C, IMG_H, IMG_W)).shape[1]

        # ── MLP encoder for vector obs ───────────────────────────────────────
        self.vec_encoder = nn.Sequential(
            nn.LayerNorm(VEC_OBS_DIM),
            nn.Linear(VEC_OBS_DIM, 64), nn.ELU(),
            nn.Linear(64, 64), nn.ELU(),
        )

        # ── Combined head ────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(cnn_out_dim + 64, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
        )
        self.mean_layer = nn.Linear(128, self.num_actions)
        self.log_std_param = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        vec, pixels = _split_obs(inputs["states"])

        cnn_feat = self.cnn(pixels)
        vec_feat = self.vec_encoder(vec)
        x = self.head(torch.cat([cnn_feat, vec_feat], dim=-1))

        mean = self.mean_layer(x)
        log_std = self.log_std_param.expand_as(mean)
        return mean, log_std, {}


class CNNValue(DeterministicMixin, Model):
    """CNN + MLP value function (same backbone as policy)."""

    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            cnn_out_dim = self.cnn(torch.zeros(1, IMG_C, IMG_H, IMG_W)).shape[1]

        self.vec_encoder = nn.Sequential(
            nn.LayerNorm(VEC_OBS_DIM),
            nn.Linear(VEC_OBS_DIM, 64), nn.ELU(),
            nn.Linear(64, 64), nn.ELU(),
        )

        self.head = nn.Sequential(
            nn.Linear(cnn_out_dim + 64, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 1),
        )

    def compute(self, inputs, role):
        vec, pixels = _split_obs(inputs["states"])

        cnn_feat = self.cnn(pixels)
        vec_feat = self.vec_encoder(vec)
        value = self.head(torch.cat([cnn_feat, vec_feat], dim=-1))
        return value, {}
