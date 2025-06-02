import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, max_size, obs_space_dim, action_dim, device=None, enough_size=10_000):
        self.max_size = int(max_size)
        self.enough_size = int(enough_size)
        if device is None:
            device = torch.device('cpu')
        self._obs_buffer = torch.zeros((self.max_size,) + obs_space_dim).to(device)
        self._action_buffer = torch.zeros((self.max_size,) + action_dim).to(device)
        self._action_log_prob_buffer = torch.zeros(self.max_size, 1).to(device)
        self._trans_buffer = torch.zeros((self.max_size,) + obs_space_dim).to(device)
        self._reward_buffer = torch.zeros(self.max_size, 1).to(device)
        self._done_buffer = torch.zeros((self.max_size, 1),  device=device)
        self._idx = 0
        self._is_enough = False
        self._current_size = 0
    
    def push(self, obs, action, action_log_prob, trans, reward, done):
        self._obs_buffer[self._idx] = obs
        self._action_buffer[self._idx] = action
        self._action_log_prob_buffer[self._idx] = action_log_prob
        self._trans_buffer[self._idx] = trans
        self._reward_buffer[self._idx] = reward
        self._done_buffer[self._idx] = done

        self._idx += 1
        self._current_size += 1
        self._current_size = min(self._current_size, self.max_size)

        if not self._is_enough and self._idx == self.enough_size:
            self._is_enough = True
        if self._idx == self.max_size:
            self._idx = 0
    
    def is_enough(self):
        return self._is_enough

    def sample(self, size=1):
        if not self.is_enough():
            return None
        idx = torch.randint(0, self._current_size, (size,))
        return {
            'obs': self._obs_buffer[idx],
            'action': self._action_buffer[idx],
            'action_log_prob': self._action_log_prob_buffer[idx],
            'next_obs': self._trans_buffer[idx],
            'reward': self._reward_buffer[idx],
            'done': self._done_buffer[idx],
        }
'''
class NumpyReplayBuffer:
    def __init__(self, max_size, obs_space_dim, action_dim, device=None, enough_size=10_000):
        self.max_size = int(max_size)
        self.enough_size = int(enough_size)
        self._obs_buffer = np.zeros((self.max_size,) + obs_space_dim)
        self._action_buffer = np.zeros((self.max_size,) + action_dim)
        self._action_log_prob_buffer = np.zeros(self.max_size, 1)
        self._trans_buffer = np.zeros((self.max_size,) + obs_space_dim)
        self._reward_buffer = np.zeros(self.max_size, 1)
        self._done_buffer = np.zeros((self.max_size, 1))
        self._idx = 0
        self._is_enough = False
        self._current_size = 0
    
    def push(self, obs, action, action_log_prob, trans, reward, done):
        self._obs_buffer[self._idx] = obs
        self._action_buffer[self._idx] = action
        self._action_log_prob_buffer[self._idx] = action_log_prob
        self._trans_buffer[self._idx] = trans
        self._reward_buffer[self._idx] = reward
        self._done_buffer[self._idx] = done

        self._idx += 1
        self._current_size += 1
        self._current_size = min(self._current_size, self.max_size)

        if not self._is_enough and self._idx == self.enough_size:
            self._is_enough = True
        if self._idx == self.max_size:
            self._idx = 0
    
    def is_enough(self):
        return self._is_enough

    def sample(self, size=1):
        if not self.is_enough():
            return None
        idx = np.random.randint(0, self._current_size, (size,))
        return {
            'obs': torch.tensor(self._obs_buffer[idx]),
            'action': torch.tensor(self._action_buffer[idx]),
            'action_log_prob': torch.tensor(self._action_log_prob_buffer[idx]),
            'next_obs': torch.tensor(self._trans_buffer[idx]),
            'reward': torch.tensor(self._reward_buffer[idx]),
            'done': torch.tensor(self._done_buffer[idx]),
        }
'''

def get_flattened_shape(shape_object):
    total_dim = 1
    for dim in shape_object:
        total_dim *= dim
    return total_dim
