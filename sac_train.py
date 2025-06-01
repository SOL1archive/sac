import logging

from pprint import pprint
from icecream import ic

from tqdm.auto import trange
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import gymnasium as gym
from gymnasium.wrappers.transform_reward import TransformReward

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam

import wandb

from train_config import TrainConfig
from utils import ReplayBuffer, get_flattened_shape
from models import PolicyNet, SoftQNet, SoftVNet

class Trainer:
    def __init__(self, config: TrainConfig, device=None):
        self.config = config
        wandb.init(
            project="sac_rl",
            config=config.__dict__,
            reinit=True,
        )

        if device is not None:
            pass
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        self.device = device

        self.global_step_cnt = 0
        self.env = gym.make(config.env_name, render_mode='rgb_array')
        self.env = TransformReward(self.env, lambda r: self.config.reward_scale * r)

        self.replay_buffer = ReplayBuffer(
            max_size=config.replay_buffer_size,
            obs_space_dim=self.env.observation_space.shape,
            action_dim=self.env.action_space.shape,
            device=self.device,
        )
        
        obs_dim = get_flattened_shape(self.env.observation_space.shape)
        action_dim = get_flattened_shape(self.env.action_space.shape)

        self.policy = PolicyNet(
            obs_dim=obs_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            action_dim=action_dim,
        ).to(self.device)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=config.learning_rate)
    
        self.soft_q_lt = []
        for _ in range(2):
            soft_q = SoftQNet(
                obs_dim=obs_dim,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                action_dim=action_dim,
            ).to(self.device)
            soft_q_optimizer = Adam(soft_q.parameters(), lr=config.learning_rate)
            self.soft_q_lt.append(
                [soft_q, soft_q_optimizer]
            )
    
        self.soft_v = SoftVNet(
            obs_dim=obs_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
        ).to(self.device)
        self.soft_v_optimizer = Adam(self.soft_v.parameters(), lr=config.learning_rate)
    
        self.target_v = SoftVNet(
            obs_dim=obs_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
        ).to(self.device)
        self.target_v.load_state_dict(self.soft_v.state_dict())
    
    def _update_target_v(self):
        for soft_v_param, target_v_param in zip(self.soft_v.parameters(), self.target_v.parameters()):
            target_v_param.data = self.config.tau * soft_v_param.data + (1.0 - self.config.tau) * target_v_param.data
    
    def estimate_q(self, obs, action):
        q_estims = []
        for soft_q, _ in self.soft_q_lt:
            soft_q.eval()
            soft_q.requires_grad_(False)
            q_estims.append(
                soft_q(obs, action)
            )
            soft_q.train()
            soft_q.requires_grad_(True)
        final_q = torch.min(*q_estims)
        return final_q
    
    def estimate_target_v(self, obs):
        with torch.no_grad():
            self.target_v.eval()
            self.target_v.requires_grad_(False)
            target_v_out = self.target_v(obs)
            self.target_v.train()
            self.target_v.requires_grad_(True)
        return target_v_out

    def _update_v(self, obs):
        soft_v_pred = self.soft_v(obs)
        self.soft_v_optimizer.zero_grad()
        #ic(soft_v_pred.shape, q_pred.shape, batch_action_log_prob.shape)
        with torch.no_grad():
            self.policy.eval()
            action, action_log_prob, _ = self.policy.sample(obs)
            self.policy.train()
            q_pred = self.estimate_q(obs, action)
        soft_v_loss = F.mse_loss(soft_v_pred, q_pred - self.config.alpha * action_log_prob)
        soft_v_loss.backward()
        self.soft_v_optimizer.step()
        wandb.log({
            'global_steps': self.global_step_cnt, 
            'train/v_loss': soft_v_loss.item()
        })
        return soft_v_loss.item()
    
    def _update_q(self, obs, action, reward, next_obs, done):
        q_losses_lt = []
        for soft_q, soft_q_optimizer in self.soft_q_lt:
            soft_q_pred = soft_q(obs, action)
            target_v_out = self.estimate_target_v(next_obs)
            soft_q_optimizer.zero_grad()
            #ic(soft_q_pred.shape, batch_reward.shape, soft_v_out.shape)
            soft_q_loss = F.mse_loss(soft_q_pred, reward + (1 - done) * self.config.discount_rate * target_v_out)
            q_losses_lt.append(soft_q_loss.item())
            soft_q_loss.backward()
            soft_q_optimizer.step()
        wandb.log({
            'global_steps': self.global_step_cnt, 
            'train/q1_loss': q_losses_lt[0], 
            'train/q2_loss': q_losses_lt[1]
        })
        return q_losses_lt
    
    def _update_policy(self, obs):
        pred_action, pred_log_prob, (mean, std) = self.policy.sample(obs)
        soft_q_out = self.estimate_q(obs, pred_action)
        self.policy_optimizer.zero_grad()
        #ic(pred_log_prob.shape, soft_q_out.shape)
        policy_loss = (self.config.alpha * pred_log_prob - soft_q_out).mean()
        policy_loss.backward()
        self.policy_optimizer.step()
        wandb.log({
            'global_steps': self.global_step_cnt, 
            'train/policy_loss': policy_loss.item(),
            'train/action_mean': mean,
            'train/action_std': std,
            'train/action_in_update': pred_action.detach().cpu(),
            'train/entropy_in_update': pred_log_prob.detach().cpu(),
        })
        return policy_loss.item()
    
    def param_update(self, samples):
        if samples is None:
            return
        batch_obs = samples['obs'].to(self.device)
        batch_action = samples['action'].to(self.device)
        batch_reward = samples['reward'].to(self.device)
        batch_next_obs = samples['next_obs'].to(self.device)
        batch_done = samples['done'].to(self.device)

        self._update_v(batch_obs)
        self._update_q(batch_obs, batch_action, batch_reward, batch_next_obs, batch_done)
        self._update_policy(batch_obs)
        self._update_target_v()

    @torch.no_grad
    def eval(self):
        eval_env = gym.make(self.config.env_name, render_mode='rgb_array')
        eval_env = TransformReward(eval_env, lambda r: self.config.reward_scale * r)

        return_lt = []
        for _ in range(self.config.eval_num_episodes):
            episode_end = False
            obs, info = eval_env.reset()
            img_lt = [np.transpose(eval_env.render(), (2, 0, 1))]
            return_ = 0
            step_cnt = 0
            while not episode_end:
                action, log_prob, _ = self.policy.sample(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device))
                action = action.squeeze().cpu().numpy()
                log_prob = log_prob.squeeze().cpu().numpy()
                next_obs, reward, terminated, truncated, info = eval_env.step(action)
                obs = next_obs
                episode_end = terminated or truncated
                img_lt.append(np.transpose(eval_env.render(), (2, 0, 1)))

                return_ += reward * (self.config.discount_rate ** step_cnt)
                step_cnt += 1
            
            return_lt.append(return_)
        return_mean = sum(return_lt) / len(return_lt)
        wandb.log({
            'global_steps': self.global_step_cnt, 
            'eval/sample_return': return_, 
            'eval/sample_video': wandb.Video(
                np.stack(img_lt),
                fps=16,
                format='mp4',
            ),
            'eval/sample_episode_length': len(img_lt),
            'eval/return_mean': return_mean,
        })
        
        return return_mean

    def env_step(self, obs, action, log_prob):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        episode_end = terminated or truncated
        self.replay_buffer.push(
            torch.tensor(obs, dtype=torch.float32),
            torch.tensor(action, dtype=torch.float32),
            torch.tensor(log_prob, dtype=torch.float32),
            torch.tensor(next_obs, dtype=torch.float32),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(episode_end, dtype=torch.float32),
        )
        if episode_end:
            obs, info = self.env.reset()
        else:
            obs = next_obs
        return obs

    def train(self):
        self.global_step_cnt = 0
        eval_expected_return = []
        episode_end = False
        obs, info = self.env.reset()

        while not self.replay_buffer.is_enough():
            action = self.env.action_space.sample()
            log_prob = np.array(0)
            obs = self.env_step(obs, action, log_prob)

        obs, info = self.env.reset()
        for _ in trange(self.config.num_train_steps, mininterval=60.):
            with torch.no_grad():
                action, log_prob, _ = self.policy.sample(torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device))
            action = action.squeeze().cpu().numpy()
            log_prob = log_prob.squeeze().cpu().numpy()
            obs = self.env_step(obs, action, log_prob)

            samples = self.replay_buffer.sample(self.config.batch_size)
            self.param_update(samples)

            if self.global_step_cnt % self.config.eval_steps == 0:
                eval_expected_return.append(self.eval())
            self.global_step_cnt += 1
        return eval_expected_return
    
    def close(self):
        self.env.close()

def main():
    config = TrainConfig()
    pprint(config)
    trainer = Trainer(config)
    return_lt = trainer.train()
    trainer.close()
    return_series = pd.Series(return_lt)
    pprint(return_series)
    return_series = return_series.plot.line()
    plt.savefig('./sac-2.png')

if __name__ == '__main__':
    main()
