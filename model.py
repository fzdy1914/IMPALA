import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=3)
        self.bn = nn.BatchNorm2d(output_dim)

    def forward(self, x):
        h = torch.cat([x[:, :, :, -1:], x, x[:, :, :, :1]], dim=3)
        h = torch.cat([h[:, :, -1:], h, h[:, :, :1]], dim=2)
        h = self.conv(h)
        h = self.bn(h)
        return h


class DenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers, filters = 12, 32
        self.conv0 = ConvBlock(5, filters)
        self.blocks = nn.ModuleList([ConvBlock(filters, filters) for _ in range(layers)])
        self.head_p = nn.Linear(filters, 4, bias=False)
        self.head_v = nn.Linear(filters * 2, 1, bias=False)

    def forward(self, x):
        need_reshape = False
        if len(x.shape) == 5:
            need_reshape = True
            length = x.shape[0]
            bs = x.shape[1]
            x = x.reshape(length * bs, *x.shape[2:])

        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = F.relu_(h + block(h))

        h_head = (h * x[:,:1]).view(h.size(0), h.size(1), -1).sum(-1)
        h_avg = h.view(h.size(0), h.size(1), -1).mean(-1)
        logits = self.head_p(h_head)
        values = torch.tanh(self.head_v(torch.cat([h_head, h_avg], 1)))
        if need_reshape:
            logits, values = logits.view(length, bs, -1).permute(0, 2, 1), values.view(length, bs)

        return logits, values


# class Network(nn.Module):
#     def __init__(self, action_size=4):
#         super(Network, self).__init__()
#         self.action_space = action_size
#         self.conv1 = nn.Conv2d(5, 64, kernel_size=2)
#         self.conv2 = nn.Conv2d(64, 32, kernel_size=2)
#         self.fc = nn.Linear(1440, 512)
#         self.head = Head(action_size)
#
#     def forward(self, x):
#         need_reshape = False
#         if len(x.shape) == 5:
#             need_reshape = True
#             length = x.shape[0]
#             bs = x.shape[1]
#             x = x.reshape(length * bs, *x.shape[2:])
#
#         x = F.leaky_relu(self.conv1(x), inplace=True)
#         x = F.leaky_relu(self.conv2(x), inplace=True)
#         x = x.view(x.shape[0], -1)
#         x = F.leaky_relu(self.fc(x), inplace=True)
#         logits, values = self.head(x)
#         logits[torch.isnan(logits)] = 1e-12
#         if need_reshape:
#             logits, values = logits.view(length, bs, -1).permute(0, 2, 1), values.view(length, bs)
#         # action = torch.softmax(logits, 1).multinomial(1)
#         return logits, values
#
#
# class Head(nn.Module):
#     def __init__(self, action_space):
#         super().__init__()
#         self.actor_linear = nn.Linear(512, action_space)
#         self.critic_linear = nn.Linear(512, 1)
#
#     def forward(self, x):
#         logits = self.actor_linear(x)
#         values = self.critic_linear(x)
#         return logits, values
