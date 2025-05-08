import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
# Do not modify the input of the 'act' function and the '__init__' function. 
class Pi_FC(torch.nn.Module):
    def __init__(self, obs_size, action_size):
        super(Pi_FC, self).__init__()
        self.fc1 = torch.nn.Linear(obs_size, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.mu = torch.nn.Linear(256, action_size)
        self.log_sigma = torch.nn.Linear(256, action_size)

    def forward(self, x, deterministic=False, with_logprob=False):
        y1 = F.relu(self.fc1(x))
        y2 = F.relu(self.fc2(y1))
        mu = self.mu(y2)

        if deterministic:
            # used for evaluating policy
            action = torch.tanh(mu)
            log_prob = None
        else:
            log_sigma = self.log_sigma(y2)
            log_sigma = torch.clamp(log_sigma,min=-20.0,max=2.0)
            sigma = torch.exp(log_sigma)
            dist = Normal(mu, sigma)
            x_t = dist.rsample()
            if with_logprob:
                log_prob = dist.log_prob(x_t).sum(1)
                log_prob -= (2*(np.log(2) - x_t - F.softplus(-2*x_t))).sum(1)
            else:
                log_prob = None
            action = torch.tanh(x_t)

        return action, log_prob

# Parse observation dictionary returned by Deepmind Control Suite
def process_observation(time_step):
    o_1 = np.array([])
    for k in time_step.observation:
        if time_step.observation[k].shape:
            o_1 = np.concatenate((o_1, time_step.observation[k].flatten()))
        else :
            o_1 = np.concatenate((o_1, np.array([time_step.observation[k]])))
    r = time_step.reward
    done = time_step.last()
    
    return o_1, r, done
class Agent(object):
    def __init__(self):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        self.obs_size = 67   # 你要根據 dm_control 環境修改
        self.action_size = 21  # 例如 humanoid 的動作維度

        self.actor = Pi_FC(self.obs_size, self.action_size).to(self.device)

        # 載入訓練好的模型參數（請替換 checkpoint 路徑）
        checkpoint = torch.load('./model.ckpt', map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor.eval()

    def act(self, observation):
        def flatten_observation(observation):
            return observation.flatten()
        observation = flatten_observation(observation)
        obs_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            action, _ = self.actor(obs_tensor, deterministic=True)
        return action.cpu().numpy()[0]
