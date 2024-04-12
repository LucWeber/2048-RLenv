import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch

from RLenv_2048.utils import get_config

def preprocess_data():
  states, actions, rewards = [], [], []
  PATH = './pretrain_data/'
  files = os.listdir(PATH)
  for file in files:
    states_temp, actions_temp, rewards_temp = torch.load(open(os.path.join(PATH, file), 'rb'))
    states.append(states_temp)
    actions.append(actions_temp)
    rewards.append(rewards_temp)
  states = torch.cat(states)
  actions = torch.cat(actions)
  rewards = torch.cat(rewards)
  return states, actions, rewards
    
if __name__ == "__main__":
  # GET ARGS
  config = get_config()
  print(config)

  states, actions, rewards = preprocess_data()
  print(states.shape, actions.shape, rewards.shape)
    
