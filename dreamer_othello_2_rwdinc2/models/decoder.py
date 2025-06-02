import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent

# TODO: 다른 사이즈 보드, 다른 cell state 추가에도 호환 가능하도록 채널 계산 필요 (파라미터로 받아야 함)
# 채널 위치 맞는지 확인

class Decoder2D(nn.Module):
    def __init__(self, args, obs_channel=7):
        super(Decoder2D, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(args.state_size + args.deterministic_size, 512),
            nn.ELU(),
            nn.Unflatten(1, (128, 2, 2)),
            nn.ConvTranspose2d(128, 64, 4),  # 6x6x128 -> 14x14x64
            nn.ELU(),
            nn.ConvTranspose2d(64, 32, 4),  # 14x14x64 -> 31x31x32
            nn.ELU(),
            nn.ConvTranspose2d(32, obs_channel, 3, padding=1),  # 31x31x32 -> 64x64x3
            # nn.Upsample(size=(8, 8), mode='bilinear', align_corners=False),
        )

    def forward(self, state, deterministic, target_size):
        x = torch.cat([state, deterministic], dim=-1)
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        pred = self.layers(x)
        # (batch*seq, obs_channel, 64, 64)
        # pred_shape = pred.shape
        # print("after decoder: ", pred_shape)
        pred = F.interpolate(pred, size=(target_size[-2], target_size[-1]), mode='bilinear', align_corners=False)
        pred = pred.reshape(*shape[:-1], *target_size[2:])
        # print(pred.max(), '     ', pred.min())
        # print(pred.shape)
        m = Normal(pred, 1)
        dist = Independent(m, 3)
        # print("pred:\n", pred)
        # print("normal:\n", m)
        # print("dist:\n", dist)
        # print('=================================')
        return dist


class Decoder1D(nn.Module):
    def __init__(self, args, obs_size):
        super(Decoder1D, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(args.state_size + args.deterministic_size, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, obs_size)
        )

    def forward(self, state, deterministic):
        x = torch.cat([state, deterministic], dim=-1)
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        pred = self.decoder(x)
        pred = pred.reshape(*shape[:-1], -1)
        dist = Normal(pred, 1)
        dist = Independent(dist, 1)
        return dist
