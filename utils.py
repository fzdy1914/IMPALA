import torch


class ParameterServer(object):
    def __init__(self, lock):
        self.lock = lock
        self.weight = None

    def pull(self):
        with self.lock:
            return self.weight

    def push(self, weigth):
        with self.lock:
            self.weight = weigth


def make_time_major(batch):
    boards = []
    actions = []
    rewards = []
    dones = []
    logits = []
    for t in batch:
        boards.append(t.boards)
        actions.append(t.actions)
        rewards.append(t.rewards)
        dones.append(t.dones)
        logits.append(t.logits)
    boards = torch.stack(boards).transpose(0, 1)
    actions = torch.stack(actions).transpose(0, 1)
    rewards = torch.stack(rewards).transpose(0, 1)
    dones = torch.stack(dones).transpose(0, 1)
    logits = torch.stack(logits).permute(1, 2, 0)
    return boards, actions, rewards, dones, logits
