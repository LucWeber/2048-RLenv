import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import torch
import torch.nn.functional as F

from RLenv_2048.utils import get_config
from RLenv_2048.policies import baselineREINFORCEpolicy
from RLenv_2048.env.RLenv2048 import RLenv2048


def preprocess_data():
  states, actions, rewards = [], [], []
  PATH = './RLenv_2048/pretrain_data/'
  files = os.listdir(PATH)
  for file in files:
    states_temp, actions_temp, rewards_temp = torch.load(open(os.path.join(PATH, file), 'rb'), map_location='cpu')
    states.append(states_temp)
    actions.append(actions_temp)
    rewards.append(rewards_temp)
  
  states = torch.cat(states)
  actions = torch.cat(actions)
  rewards = torch.cat(rewards)
  print(states.shape, actions.shape, rewards.shape)

  return states, actions, rewards

def train_agent(epochs, config):
  env = RLenv2048(mode='agent')

  model = baselineREINFORCEpolicy(env, **vars(config)).model
  states, actions, _ = preprocess_data()

  model.train()
      
  for epoch in range(epochs):
        # Shuffle the dataset
        permutation = torch.randperm(states.size(0))
        shuffled_states = states[permutation]
        shuffled_actions = actions[permutation]

        # Assuming the use of mini-batch training
        #batch_size = config.batch_size
        losses = []
        for state, action in zip(shuffled_states, shuffled_actions):
            #batch_states = shuffled_states[i:i+batch_size]
            #batch_actions = shuffled_actions[i:i+batch_size]
            #breakpoint()
            probs = model.forward(state.float().flatten().unsqueeze(0))
            action_one_hot = F.one_hot(action, num_classes=model.num_actions)
            loss = F.cross_entropy(probs.squeeze(), action_one_hot.float().squeeze())

            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
            losses.append(loss.item())
        print(f'Epoch: {epoch}, Loss: {sum(losses)/len(losses)}')

def plot_stats():   
    PATH = './RLenv_2048/pretrain_data/'
    files = os.listdir(PATH)
    stats = {'reward': [], 'model': [], 'total_sessions': [], 't_max': [], 'gamma': [], 'epsilon': [], 'entropy_term': [], 'lr': []}

    for file in files:
        positions = [5, 6, 8, 10, 12, 14, 16, 18]
        reward, model, total_sessions, t_max, gamma, epsilon, entropy_term, lr = (string for i, string in enumerate(file.split('_')) if i in positions)

        stats['reward'].append(int(reward))
        stats['model'].append(model)
        stats['total_sessions'].append(total_sessions)
        stats['t_max'].append(int(t_max))
        stats['gamma'].append(gamma)
        stats['epsilon'].append(epsilon)
        stats['entropy_term'].append(entropy_term)
        stats['lr'].append(lr)
    
    df = pd.DataFrame(stats)
    hue_order = ['Transformer4L', 'Transformer8L', 'Transformer12L', 'Transformer16L', 'Transformer20L']
    fig = plt.figure()
    sns.histplot(data=df, x='reward', hue='model', hue_order=hue_order, binwidth=1000, multiple='dodge', palette='viridis')
    plt.savefig(f'./RLenv_2048/plots/highest_rewards_histogram_models.png')

    fig = plt.figure()
    sns.histplot(data=df, x='reward', hue='total_sessions', binwidth=2000, multiple='dodge', palette='viridis')
    plt.savefig(f'./RLenv_2048/plots/highest_rewards_histogram_n_sessions.png')
       
    fig = plt.figure()
    sns.histplot(data=df, x='reward', hue='t_max', binwidth=1000, multiple='dodge', palette='viridis')
    plt.savefig(f'./RLenv_2048/plots/highest_rewards_histogram_tmax.png')

    fig = plt.figure()
    sns.histplot(data=df, x='reward', hue='gamma', binwidth=1000, multiple='dodge', palette='viridis')
    plt.savefig(f'./RLenv_2048/plots/highest_rewards_histogram_gamma.png')

    fig = plt.figure()
    sns.histplot(data=df, x='reward', hue='lr', binwidth=1000, multiple='dodge', palette='viridis')
    plt.savefig(f'./RLenv_2048/plots/highest_rewards_histogram_lr.png')


if __name__ == "__main__":
  # GET ARGS
  config = get_config()
  print(config)
  plot_stats()

  train_agent(epochs=10, config=config)

