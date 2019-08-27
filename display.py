from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import torch
import pygame
import random
import argparse
import numpy as np
from collections import deque
from common import v_wrap
from environment.environment import Environment
from agent.agent import Agent
from gym.envs.classic_control import rendering
# def render(self):
    #     if self.viewer is None:
    #         self.viewer = rendering.SimpleImageViewer()
    #     self.viewer.imshow(self.ale.getScreenRGB2())
    #     return self.viewer.isopen

        # cv2.imshow('screen', self.ale.getScreenRGB()[:, :, ::-1])
        # cv2.waitKey(1)
BLUE = (128, 128, 255)
RED = (255, 192, 192)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

class MovieWriter(object):
    def __init__(self, file_name, frame_size, fps):
        """
        frame_size is (w, h)
        """
        self._frame_size = frame_size
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.vout = cv2.VideoWriter()
        success = self.vout.open(file_name, fourcc, fps, frame_size, True)
        if not success:
            print("Create movie failed: {0}".format(file_name))

    def add_frame(self, frame):
        """
        frame shape is (h, w, 3), dtype is np.uint8
        """
        self.vout.write(frame)

    def close(self):
        self.vout.release()
        self.vout = None


class StateHistory(object):
    def __init__(self):
        self._states = deque(maxlen=3)

    def add_state(self, state):
        self._states.append(state)

    @property
    def is_full(self):
        return len(self._states) >= 3

    @property
    def states(self):
        return list(self._states)


class ValueHistory(object):
    def __init__(self):
        self._values = deque(maxlen=100)

    def add_value(self, value):
        self._values.append(value)

    @property
    def is_empty(self):
        return len(self._values) == 0

    @property
    def values(self):
        return self._values


class Display(object):
    def __init__(self, args, display_size, saver):
        pygame.init()
        self.args = args
        self.surface = pygame.display.set_mode(display_size, 0, 24)
        pygame.display.set_caption('UNREAL')
        args.action_size = Environment.get_action_size(args.env_name)
        self.global_network = Agent(1, args)
        saver.restore(self.global_network)
        self.global_network.eval()
        self.environment = Environment.create_environment(args.env_name)
        self.font = pygame.font.SysFont(None, 20)
        self.value_history = ValueHistory()
        self.state_history = StateHistory()
        self.distribution = torch.distributions.Categorical
        self.episode_reward = 0

    def update(self):
        self.surface.fill(BLACK)
        self.process()
        pygame.display.update()

    def choose_action(self, pi):
        m = self.distribution(pi)
        return m.sample().item()

    def scale_image(self, image, scale):
        return image.repeat(scale, axis=0).repeat(scale, axis=1)

    def draw_text(self, str, left, top, color=WHITE):
        text = self.font.render(str, True, color, BLACK)
        text_rect = text.get_rect()
        text_rect.left = left
        text_rect.top = top
        self.surface.blit(text, text_rect)

    def draw_center_text(self, string, center_x, top):
        text = self.font.render(string, True, WHITE, BLACK)
        text_rect = text.get_rect()
        text_rect.centerx = center_x
        text_rect.top = top
        self.surface.blit(text, text_rect)

    def show_pixel_change(self, pixel_change, left, top, rate, label):
        """
        Show pixel change
        """
        pixel_change_ = np.clip(pixel_change * 255.0 * rate, 0.0, 255.0)
        data = pixel_change_.astype(np.uint8)
        data = np.stack([data for _ in range(3)], axis=2)
        data = self.scale_image(data, 4)
        image = pygame.image.frombuffer(data, (20 * 4, 20 * 4), 'RGB')
        self.surface.blit(image, (left + 8 + 4, top + 8 + 4))
        self.draw_center_text(label, left + 100 / 2, top + 100)

    def show_policy(self, pi):
        """
        Show action probability.
        """
        start_x = 10

        y = 150

        for i in range(len(pi)):
            width = pi[i] * 100
            pygame.draw.rect(self.surface, WHITE, (start_x, y, width, 10))
            y += 20
        self.draw_center_text("PI", 50, y)

    def show_image(self, state):
        """
        Show input image
        """
        state_ = state * 255.0
        data = state_.astype(np.uint8)
        image = pygame.image.frombuffer(data, (84, 84), 'RGB')
        self.surface.blit(image, (8, 8))
        self.draw_center_text("input", 50, 100)

    def show_value(self):
        if self.value_history.is_empty:
            return

        min_v = float("inf")
        max_v = float("-inf")

        values = self.value_history.values

        for v in values:
            min_v = min(min_v, v)
            max_v = max(max_v, v)

        top = 150
        left = 150
        width = 100
        height = 100
        bottom = top + width
        right = left + height

        d = max_v - min_v
        last_r = 0.0
        for i, v in enumerate(values):
            r = (v - min_v) / d
            if i > 0:
                x0 = i - 1 + left
                x1 = i + left
                y0 = bottom - last_r * height
                y1 = bottom - r * height
                pygame.draw.line(self.surface, BLUE, (x0, y0), (x1, y1), 1)
            last_r = r

        pygame.draw.line(self.surface, WHITE, (left, top), (left, bottom), 1)
        pygame.draw.line(self.surface, WHITE, (right, top), (right, bottom), 1)
        pygame.draw.line(self.surface, WHITE, (left, top), (right, top), 1)
        pygame.draw.line(self.surface, WHITE, (left, bottom), (right, bottom), 1)

        self.draw_center_text("V", left + width / 2, bottom + 10)

    def show_reward_prediction(self, rp_c, reward):
        start_x = 310
        reward_index = 0
        if reward == 0:
            reward_index = 0
        elif reward > 0:
            reward_index = 1
        elif reward < 0:
            reward_index = 2

        y = 150

        labels = ["0", "+", "-"]

        for i in range(len(rp_c)):
            width = rp_c[i] * 100

            if i == reward_index:
                color = RED
            else:
                color = WHITE
            pygame.draw.rect(self.surface, color, (start_x + 15, y, width, 10))
            self.draw_text(labels[i], start_x, y - 1, color)
            y += 20

        self.draw_center_text("RP", start_x + 100 / 2, y)

    def show_reward(self):
        self.draw_text("REWARD: {}".format(int(self.episode_reward)), 310, 10)

    def process(self):
        last_action = self.environment.last_action
        last_reward = np.clip(self.environment.last_reward, -1, 1)
        last_action_reward = ExperienceFrame.concat_action_and_reward(last_action, self.args.action_size,
                                                                      last_reward)

        if self.args.use_pixel_change:
            pi_values, v_value, pc_q = self.global_network(
                v_wrap(self.environment.last_state[None, :], self.args.device),
                v_wrap(last_action_reward[None, :], self.args.device), 'pvp')
            pc_q = pc_q.squeeze().cpu().numpy()
        else:
            pi_values, v_value = self.global_network(v_wrap(self.environment.last_state[None, :], self.args.device),
                                                     v_wrap(last_action_reward[None, :], self.args.device), 'b')

        self.value_history.add_value(v_value.squeeze().cpu().numpy())
        action = self.choose_action(pi_values.squeeze())
        state, reward, terminal, pixel_change = self.environment.process(action)
        self.episode_reward += reward

        if terminal:
            self.environment.reset()
            self.episode_reward = 0

        self.show_image(state)
        self.show_policy(pi_values)
        self.show_value()
        self.show_reward()

        if self.args.use_pixel_change:
            self.show_pixel_change(pixel_change, 100, 0, 3.0, "PC")
            self.show_pixel_change(pc_q[action, :, :], 200, 0, 0.4, "PC Q")

        if self.args.use_reward_prediction:
            if self.state_history.is_full:
                rp_c = self.global_network(self.state_history.states, 'rp')
                self.show_reward_prediction(rp_c.squeeze().cpu().numpy(), reward)
        self.state_history.add_state(state)

    def get_frame(self):
        data = self.surface.get_buffer().raw
        return data


def main(args):
    display_size = (440, 400)
    save_path = os.path.join(args.checkpoint_path, args.env_name)
    saver = Saver(save_path)
    display = Display(args, display_size, saver)
    clock = pygame.time.Clock()
    running = True
    writer = MovieWriter("out.mov", display_size, fps=15)
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        display.update()
        clock.tick(15)
        frame_str = display.get_frame()
        d = np.fromstring(frame_str, dtype=np.uint8)
        d = d.reshape((display_size[1], display_size[0], 3))
        writer.add_frame(d)
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UNREAL')
    parser.add_argument("--env_name", type=str, default="Breakout-v0", help="Name of a map to use.")
    parser.add_argument('--seed', type=int, default=123, help='Random seed')
    parser.add_argument("--use_pixel_change", default=True, help="whether to use pixel change")
    parser.add_argument("--use_value_replay", default=True, help="whether to use value function replay")
    parser.add_argument("--use_reward_prediction", default=True, help="whether to use reward prediction")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoint_path",
                        help="Path for saving checkpoint")
    parser.add_argument('--gpu', type=int, default=0, help='Disable CUDA')
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(random.randint(1, 10000))
    if torch.cuda.is_available():
        args.device = torch.device('cuda:{}'.format(args.gpu))
        torch.cuda.random.manual_seed_all(random.randint(1, 10000))
        torch.backends.cudnn.enabled = False  # Disable nondeterministic ops
    else:
        args.device = torch.device('cpu')
    main(args)
