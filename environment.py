import cv2
import torch
import random
import atari_py
import torch.multiprocessing as mp
from collections import deque
from kaggle_environments import make

from board_stack import encode_env_stack


# class Atari:
#     def __init__(self, game_name, seed, max_episode_length=1e10, history_length=4, reward_clip=1, device='cpu'):
#         self.device = device
#         self.ale = atari_py.ALEInterface()
#         self.ale.setInt('random_seed', seed)
#         self.ale.setInt('max_num_frames_per_episode', max_episode_length)
#         self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
#         self.ale.setInt('frame_skip', 0)
#         self.ale.setBool('color_averaging', False)
#         self.ale.loadROM(atari_py.get_game_path(game_name))  # ROM loading must be done after setting options
#         actions = self.ale.getMinimalActionSet()
#         self.actions = dict(zip(range(len(actions)), actions))
#         self.reward_clip = reward_clip
#         self.lives = 0  # Life counter (used in DeepMind training)
#         self.life_termination = False  # Used to check if resetting only from loss of life
#         self.window = history_length  # Number of frames to concatenate
#         self.state_buffer = deque([], maxlen=history_length)
#         self.training = True  # Consistent with model training mode
#         self.viewer = None
#
#     def _get_state(self):
#         state = cv2.resize(self.ale.getScreenGrayscale(), (84, 84), interpolation=cv2.INTER_LINEAR)
#         return torch.tensor(state, dtype=torch.float32, device=self.device).div_(255)
#
#     def _reset_buffer(self):
#         for _ in range(self.window):
#             self.state_buffer.append(torch.zeros(84, 84, device=self.device))
#
#     def reset(self):
#         if self.life_termination:
#             self.life_termination = False  # Reset flag
#             self.ale.act(0)  # Use a no-op after loss of life
#         else:
#             # Reset internals
#             self._reset_buffer()
#             self.ale.reset_game()
#             # Perform up to 30 random no-ops before starting
#             for _ in range(random.randrange(30)):
#                 self.ale.act(0)  # Assumes raw action 0 is always no-op
#                 if self.ale.game_over():
#                     self.ale.reset_game()
#         # Process and return "initial" state
#         observation = self._get_state()
#         self.state_buffer.append(observation)
#         self.lives = self.ale.lives()
#         return torch.stack(list(self.state_buffer), 0)
#
#     def step(self, action):
#         # Repeat action 4 times, max pool over last 2 frames
#         frame_buffer = torch.zeros(2, 84, 84, device=self.device)
#         reward, done = 0, False
#         for t in range(4):
#             reward += self.ale.act(self.actions.get(action))
#             if t == 2:
#                 frame_buffer[0] = self._get_state()
#             elif t == 3:
#                 frame_buffer[1] = self._get_state()
#             done = self.ale.game_over()
#             if done:
#                 break
#         observation = frame_buffer.max(0)[0]
#         self.state_buffer.append(observation)
#         # Detect loss of life as terminal in training mode
#         if self.training:
#             lives = self.ale.lives()
#             if self.lives > lives > 0:  # Lives > 0 for Q*bert
#                 self.life_termination = not done  # Only set flag when not truly done
#                 done = True
#             self.lives = lives
#         # Return state, reward, done
#         reward = max(min(reward, self.reward_clip), -self.reward_clip)
#         return torch.stack(list(self.state_buffer), 0), reward, done
#
#     # Uses loss of life as terminal signal
#     def train(self):
#         self.training = True
#
#     # Uses standard terminal signal
#     def eval(self):
#         self.training = False
#
#     def action_size(self):
#         return len(self.actions)
#
#     def close(self):
#         if self.viewer is not None:
#             self.viewer.close()
#             self.viewer = None


class HungryGeese:
    def __init__(self, debug=False):
        self.env = make("hungry_geese", debug=debug)

    def get_rollout_list(self):
        return encode_env_stack(self.env)

    def step(self, actions):
        self.env.step(actions)
        return self.env.state

    def reset(self, num):
        self.env.reset(num)
        return self.env.state

    def done(self):
        return self.env.done

    def close(self):
        del self.env
        pass


class EnvironmentProxy(object):
    def __init__(self, constructor_kwargs):
        self._constructor_kwargs = constructor_kwargs

    def start(self):
        self.conn, conn_child = mp.Pipe()
        self._process = mp.Process(target=self.worker, args=(self._constructor_kwargs, conn_child))
        self._process.start()
        result = self.conn.recv()
        if isinstance(result, Exception):
            raise result

    def close(self):
        try:
            self.conn.send((2, None))
            self.conn.close()
        except IOError:
            raise IOError
        print("closed normal")
        self._process.join()

    def reset(self, num):
        self.conn.send([0, num])
        state = self.conn.recv()
        if state is None:
            raise ValueError
        return state

    def step(self, action):
        self.conn.send([1, action])
        state = self.conn.recv()
        if state is None:
            raise ValueError
        return state

    def get_rollout_list(self):
        self.conn.send([3, None])
        rollout_list = self.conn.recv()
        return rollout_list

    def done(self):
        self.conn.send([4, None])
        done = self.conn.recv()
        return done

    def worker(self, constructor_kwargs, conn):
        try:
            env = HungryGeese(**constructor_kwargs)
            conn.send(None)  # Ready.
            while True:
                # Receive request.
                command, arg = conn.recv()
                if command == 0:
                    conn.send(env.reset(arg))
                elif command == 1:
                    conn.send(env.step(arg))
                elif command == 2:
                    env.close()
                    conn.close()
                    break
                elif command == 3:
                    conn.send(env.get_rollout_list())
                elif command == 4:
                    conn.send(env.done())
                else:
                    print("bad command: {}".format(command))
        except Exception as e:
            if 'env' in locals() and hasattr(env, 'close'):
                try:
                    env.close()
                    print("closed error")
                except:
                    pass
            conn.send(e)