# Copyright (c) 2018-present, Anurag Tiwari.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Actor to generate trajactories"""
import numpy as np
import torch

from board_stack import encode_state_stack
from model import Network
from parameters import NUM2ACTION


class Trajectory(object):
    """class to store trajectory data."""
    def __init__(self):
        self.boards = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.logits = []
        self.actor_id = None
        self.player_id = None

    def append(self, board, action, reward, done, logit):
        self.boards.append(board)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.logits.append(logit)

    def finish(self):
        self.boards = torch.stack(self.boards)
        self.actions = torch.cat(self.actions, 0).squeeze()
        self.rewards = torch.cat(self.rewards, 0).squeeze()
        self.dones = torch.cat(self.dones, 0).squeeze()
        self.logits = torch.cat(self.logits, 0)

    def cuda(self):
        self.boards = self.boards.cuda()
        self.actions = self.actions.cuda()
        self.rewards = self.rewards.cuda()
        self.dones = self.dones.cuda()
        self.logits = self.logits.cuda()

    def get_last(self):
        """must call this function before finish()"""
        board = self.boards[-1]
        logits = self.logits[-1]
        action = self.actions[-1]
        reward = self.rewards[-1]
        done = self.dones[-1]
        return board, action, reward, done, logits

    @property
    def length(self):
        return len(self.rewards)

    def __repr__(self):
        if self.actions is list:
            return "actor_id: %s, player_id: %s, \n boards: %s, \n logits: %s, \n action: %s, \n reward: %s, \n done: %s, \n" \
                   % \
                   (self.actor_id, self.player_id, len(self.boards), len(self.logits), len(self.actions),
                    len(self.rewards), len(self.dones))
        else:
            return "actor_id: %s, player_id: %s, \n boards: %s, \n logits: %s, \n action: %s, \n reward: %s, \n done: %s, \n" \
                   % \
                   (self.actor_id, self.player_id, self.boards.shape, self.logits.shape, self.actions.shape, self.rewards.shape, self.dones.shape)


def actor(idx, ps, data, env, args):
    """Simple actor """
    played_games = 0
    current_total_length = 0
    length = args.length
    model = Network(action_size=4)
    # save_path = args.save_path
    # load_path = args.load_path
    env.start()
    """Run the env for n steps and return a trajectory rollout."""

    with torch.no_grad():
        init_logit = torch.zeros((1, 4), dtype=torch.float32)
        init_action = torch.zeros((1, 1), dtype=torch.int64)
        init_reward = torch.zeros((1, 1)).float()
        init_done = torch.tensor(False, dtype=torch.bool).view(1, 1)

        trajectory_list = [Trajectory(), Trajectory(), Trajectory(), Trajectory()]

        while True:
            model.load_state_dict(ps.pull())

            state = env.reset(4)
            boards, _, _, _ = encode_state_stack(state)
            boards = torch.from_numpy(np.stack(boards)).float()

            for i in range(4):
                trajectory_list[i].actor_id = idx
                trajectory_list[i].player_id = i

                if trajectory_list[i].length == length:
                    persistent_state = trajectory_list[i].get_last()
                    trajectory_list[i].finish()
                    data.put(trajectory_list[i])
                    trajectory_list[i] = Trajectory()
                    trajectory_list[i].append(*persistent_state)

                trajectory_list[i].append(board=boards[i], action=init_action, reward=init_reward, done=init_done,
                                          logit=init_logit)

            individual_logits = [list(), list(), list(), list()]
            while not env.done():
                boards, _, _, _ = encode_state_stack(state)
                logits, values = model(torch.from_numpy(np.stack(boards)).float())
                for i in range(4):
                    individual_logits[i].append(logits[i].view(1, 4).detach())
                actions = torch.softmax(logits, 1).multinomial(1)
                actions = [NUM2ACTION[i.item()] for i in actions]
                state = env.step(actions)

            rollout_list = env.get_rollout_list()
            for i in range(4):
                for j in range(len(rollout_list[i])):
                    if trajectory_list[i].length == length:
                        persistent_state = trajectory_list[i].get_last()
                        trajectory_list[i].finish()
                        data.put(trajectory_list[i])
                        trajectory_list[i] = Trajectory()
                        trajectory_list[i].append(*persistent_state)

                    boards = torch.from_numpy(rollout_list[i][j][0]).float()
                    action = torch.tensor(rollout_list[i][j][1]).to(torch.int64).view(1, 1)
                    rewards = torch.tensor(rollout_list[i][j][2]).float().view(1, 1)
                    done = torch.tensor(rollout_list[i][j][3]).bool().view(1, 1)
                    trajectory_list[i].append(board=boards, action=action, reward=rewards, done=done,
                                              logit=individual_logits[i][j])

            played_games += 1
            current_total_length += max([len(lst) for lst in rollout_list])
            if played_games % 1000 == 0:
                print("Actor: {}, Played Game: {}, Avg Len: {}".format(idx, played_games, current_total_length / 1000))
                current_total_length = 0
