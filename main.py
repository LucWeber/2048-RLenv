import sys
import argparse
sys.path.insert(0, '/home/lucas/Desktop/2048-RL-env')
import matplotlib.pyplot as plt
from env_2048 import Env_2048

from policies import RandomPolicy, baselineREINFORCEpolicy

from visualization import GridVisualization
import logging

# TODO: check if setup is able to learn basic env (if state 1 -> 2 if state 2 -> 3 if state 3 -> 4 ..)
#     -> model is not able to learn this. Find out why. Steps?
#     -> create rule-based model that solves it to see whether env works well
#      -> create a very simple MLP to learn the problem
#       -> if it cannot learn it -> check the grads
# TODO: adapt script such that it is runable on GPU (correct CUDA-memory-error)
# TODO: create transformer model and run it for longer on GPU
# TODO: make learning runable in batches (to stabilize)
# TODO: cleanup code!


if __name__ == "__main__":
  # GET ARGS & SET HYPERPARAMS
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed', help='seed', type=int, default=0)
  parser.add_argument('--model_type', help='model_type', type=str, default=None)
  args = parser.parse_args()
  logging.info(args)


  mode = 'agent'
  visualize = False
  total_sessions = 200
  t_max = 100
  model_type = args.model_type if args.model_type is not None else 'MLP'

  # INIT ENVIRONMENT
  env = Env_2048(mode=mode)

  # INIT POLICY
  policy_random = RandomPolicy(env, verbose=1)
  policy = baselineREINFORCEpolicy(env, model_type=model_type, t_max=t_max, verbose=1)

  # TRAIN POLICY
  rewards = policy.train(total_sessions=total_sessions)
  rewards_baseline = policy_random.run_sessions(total_sessions, t_max)

  # PLOT RESULTS
  plt.plot(rewards, model_type)
  plt.plot(rewards_baseline, label='baseline')
  plt.legend()

  if visualize:
    grid_visualization = GridVisualization(env=env, model=policy, sleep_time=0.1)
    grid_visualization = GridVisualization(env=env, model=policy_random, sleep_time=0.1)
