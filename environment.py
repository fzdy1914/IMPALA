import torch.multiprocessing as mp
from kaggle_environments import make

from board_stack_plus import encode_env_stack_plus


class HungryGeese:
    def __init__(self, debug=False):
        self.env = make("hungry_geese", debug=debug)

    def get_rollout_list(self):
        return encode_env_stack_plus(self.env)

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