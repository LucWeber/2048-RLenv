import argparse
from types import SimpleNamespace
import pandas as pd
import matplotlib.pyplot as plt

from RLenv_2048.configs import default_config
from RLenv_2048.models import MODEL_REGISTER


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed', help='seed', type=int, default=default_config['seed'])
  parser.add_argument('--model_name', help='model_type if training from scratch; name of model file if training from checkpoint', 
                      type=str, default=default_config['model_name'])
  parser.add_argument('--gamma', help='discount factor for reward', type=float, default=default_config['gamma'])
  parser.add_argument('--epsilon', help='for epsilon-greedy action selection', type=float, default=default_config['epsilon'])
  parser.add_argument('--entropy_term', help='for entropy regularization (to encourage exploration)', type=float, default=default_config['entropy_term'])
  parser.add_argument('--train', help='train the policy', action='store_true')
  parser.add_argument('--visualize', help='visualize the policy', action='store_true')
  parser.add_argument('--run_inferences', help='run inferences for the policy (generate pretraining data).',  action='store_true')
  parser.add_argument('--total_sessions', help='number of sessions to run.', type=int,  default=default_config['total_sessions'])
  parser.add_argument('--exp', help='Any string identifying an experiment', type=str,  default=default_config['exp'])

  args = parser.parse_args()
  args_dict = vars(parser.parse_args())
  
  args_dict['model_type'] = args.model_name if args.model_name in MODEL_REGISTER else args.model_name.split('_')[1]

  return args_dict

def merge_configs(defaults, overrides):
    config = defaults.copy()  # Start with the defaults
    config.update((k, v) for k, v in overrides.items() if v is not None)
    config['save_name'] = f'{overrides["model_type"]}_sess_{config["total_sessions"]}_tmax_{config["t_max"]}_gamma_{config["gamma"]}_epsilon_{config["epsilon"]}_entropy_{config["entropy_term"]}_lr_{config["lr"]}_{config["exp"]}'
    return config

def get_config():
    args = parse_args()
    config = merge_configs(default_config, args)
    return SimpleNamespace(**config) 
    
def train_policy(config, policy, policy_baseline=None, **kwargs):
  ''' train a given policy '''            
  # TRAIN POLICY
  rewards = policy.train(total_sessions=config.total_sessions, **kwargs)
  rewards_baseline = policy_baseline.run_sessions(config.total_sessions, config.t_max) if policy_baseline is not None else [0.0] * len(rewards)

  # PLOT RESULTS
  fig = plt.figure()
  plt.plot(rewards, label=config.model_name)
  if policy_baseline is not None:
      plt.plot(rewards_baseline, label='baseline', alpha=0.5)
  plt.legend()
  
  # SAVE REWARDS
  plt.savefig(f'./plots/rewards_{config.save_name}.png')
  rewards_df = pd.DataFrame.from_dict({'trained': rewards, 'baseline': rewards_baseline})
  os.makedirs("./results", exist_ok=True)
  rewards_df.to_csv(f'./results/rewards_{config.save_name}.csv')

  print(f'Baseline mean reward: {sum(rewards_baseline) / len(rewards_baseline)}')
  print(f'Trained mean reward: {sum(rewards) / len(rewards)}')

  return policy, policy_baseline
