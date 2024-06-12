import sys
import os 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from RLenv_2048.env.RLenv2048 import RLenv2048
from RLenv_2048.policies import RandomPolicy, baselineREINFORCEpolicy
from RLenv_2048.utils import get_config, train_policy
from RLenv_2048.visualization import GridVisualization

# TODO: make learning runable in batches (to stabilize)
# TODO: introduce some kind of config dict
# TODO: save best models during the training process
# TODO: cleanup code!
# TODO: implement pretraining
# TODO: organize repo into environment and experiments


if __name__ == "__main__":
  # GET ARGS
  config = get_config()
  print(config)

  # INIT ENVIRONMENT
  env = RLenv2048(mode='agent')

  # INIT POLICY
  policy_baseline = RandomPolicy(env, verbose=0)
  policy = 'state_action_pairs_high_reward_1820_openai_gpt4_sess_1_tmax_20000_gamma_0.95_epsilon_0.15_entropy_0.0_lr_0.0001_soft_test_OAI_interface.pt'
  #'state_action_pairs_high_reward_16920_Transformer12L_sess_3000_tmax_15000_gamma_0.95_epsilon_0.0_entropy_0.0_lr_0.0001_soft_grid_search.pt' #
  #policy = baselineREINFORCEpolicy(env, **vars(config)) 

  if config.train:
    results = train_policy(config, policy, policy_baseline)
    
  if config.visualize:
    # VISUALIZE POLICY
    sys.setrecursionlimit(2000)
    grid_visualization = GridVisualization(env=env, policy=policy, sleep_time=0.005, title='trained_policy')
    #grid_visualization = GridVisualization(env=env, policy=policy_baseline, sleep_time=0.1, title='random_baseline')
  if config.run_inferences:
    # RUN SESSIONS
    policy.run_sessions(n_sessions=config.total_sessions, t_max=config.t_max, save_high_reward_sessions=True)
    
