import torch.nn as nn

# TODO: 다른 사이즈 보드, 다른 cell state 추가에도 호환 가능하도록 채널 계산 필요 (현재 6x6에서는 안 통함)

class Encoder2D(nn.Module):
    def __init__(self, args, obs_channel=7):
        super(Encoder2D, self).__init__()
        self.observation_size = args.observation_size
        self.encoder = nn.Sequential(
            nn.Conv2d(obs_channel, 32, 3, padding=1),  # 7x8x8 -> 32x8x8
            nn.ELU(),
            nn.Conv2d(32, 64, 4),  # 32x8x8 -> 64x5x5
            nn.ELU(),
            nn.Conv2d(64, 128, 4),  # 64x5x5 -> 128x2x2
            nn.Flatten(),
            nn.Linear(512, self.observation_size),
        )
        self.encoder_adaptive = nn.Sequential(
            nn.Conv2d(obs_channel, 32, 3, padding=1),  # 7x8x8 -> 32x8x8
            nn.ELU(),
            nn.Conv2d(32, 64, 3),  # 32x8x8 -> 64x6x6
            nn.ELU(),
            nn.Conv2d(64, 128, 3),  # 64x6x6 -> 128x4x4
            nn.ELU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(512, self.observation_size),
        )
        # self.encoder1 = nn.Sequential(
        #     nn.Conv2d(obs_channel, 32, 3, padding=1),  # 7x8x8 -> 32x8x8
        #     nn.ELU(),)
        # self.encoder2 = nn.Sequential(
        #     nn.Conv2d(32, 64, 3),  # 32x8x8 -> 64x6x6
        #     nn.ELU(),)
        # self.encoder3 = nn.Sequential(
        #     nn.Conv2d(64, 128, 3),  # 64x6x6 -> 128x4x4
        #     nn.ELU(),)
        # self.encoder4 = nn.Sequential(
        #     nn.Conv2d(128, 256, 3),  # 128x4x4 -> 256x2x2
        #     nn.Flatten(),)
        # self.encoder5 = nn.Sequential(
        #     nn.Linear(1024, self.observation_size),
        # )

    def forward(self, obs):
        # print("before enc: ", obs.shape)
        # print("0: ", obs.shape)
        # x = self.encoder1(obs)
        # print("1: ", x.shape)
        # x = self.encoder2(x)
        # print("2: ", x.shape)
        # x = self.encoder3(x)
        # print("3: ", x.shape)
        # x = self.encoder4(x)
        # print("4: ", x.shape)
        # return self.encoder5(x)
        return self.encoder_adaptive(obs)


class Encoder1D(nn.Module):
    def __init__(self, args, obs_size):
        super(Encoder1D, self).__init__()
        self.observation_size = args.observation_size
        self.encoder = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, self.observation_size),
        )

    def forward(self, obs):
        return self.encoder(obs)
