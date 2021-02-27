import torch
import torch.nn as nn
import torch.nn.functional as F


class Network(nn.Module):
    def __init__(self, action_size=4):
        super(Network, self).__init__()
        self.action_space = action_size
        self.conv1 = nn.Conv2d(5, 64, kernel_size=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=2)
        self.fc = nn.Linear(1440, 512)
        self.head = Head(action_size)

    def forward(self, x):
        need_reshape = False
        if len(x.shape) == 5:
            need_reshape = True
            length = x.shape[0]
            bs = x.shape[1]
            x = x.reshape(length * bs, *x.shape[2:])

        x = F.leaky_relu(self.conv1(x), inplace=True)
        x = F.leaky_relu(self.conv2(x), inplace=True)
        x = x.view(x.shape[0], -1)
        x = F.leaky_relu(self.fc(x), inplace=True)
        logits, values = self.head(x)
        logits[torch.isnan(logits)] = 1e-12
        if need_reshape:
            logits, values = logits.view(length, bs, -1).permute(0, 2, 1), values.view(length, bs)
        # action = torch.softmax(logits, 1).multinomial(1)
        return logits, values


class Head(nn.Module):
    def __init__(self, action_space):
        super().__init__()
        self.actor_linear = nn.Linear(512, action_space)
        self.critic_linear = nn.Linear(512, 1)

    def forward(self, x):
        logits = self.actor_linear(x)
        values = self.critic_linear(x)
        return logits, values
