import sys
from env_2048 import Env_2048

from policies import RandomPolicy, baselineREINFORCEpolicy
from utils import get_config, train_policy

from visualization import GridVisualization

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
  env = Env_2048(mode='agent')

  # INIT POLICY
  policy_baseline = RandomPolicy(env, verbose=0)
  policy = baselineREINFORCEpolicy(env, **vars(config)) 

  if config.train:
    train_policy(config, policy, policy_baseline)
  if config.visualize:
    # VISUALIZE POLICY
    sys.setrecursionlimit(2000)
    grid_visualization = GridVisualization(env=env, policy=policy, sleep_time=0.1, title='trained_policy')
    grid_visualization = GridVisualization(env=env, policy=policy_baseline, sleep_time=0.1, title='random_baseline')
  if config.run_inferences:
    # RUN SESSIONS
    policy.run_sessions(n_sessions=config.total_sessions, t_max=config.t_max, save_high_reward_sessions=True)
    