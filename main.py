import sys
import argparse
sys.path.insert(0, '/home/lucas/Desktop/2048-RL-env')
import matplotlib.pyplot as plt
from env_2048 import Env_2048
from env_sanity_check import EnvDummy
from policies import RandomPolicy, baselineREINFORCEpolicy

from visualization import GridVisualization
import logging

# TODO: don't allow illegal moves by design of algorithm/env
# TODO: check if setup is able to learn basic env (if state 1 -> 2 if state 2 -> 3 if state 3 -> 4 ..)
#     -> model is not able to learn this. Find out why. Steps?
#     -> create rule-based model that solves it to see whether env works well
#      -> create a very simple MLP to learn the problem
#       -> if it cannot learn it -> check the grads
# TODO: adapt script such that it is runable on HPC


if __name__ == "__main__":
  # GET ARGS & SET HYPERPARAMS
  parser = argparse.ArgumentParser()
  parser.add_argument('--seed', help='seed', type=int, default=0)
  args = parser.parse_args()
  logging.info(args)


  mode = 'agent'
  #mode = 'human'
  visualize = False
  total_sessions = 200
  t_max = 100

  # INIT ENVIRONMENT
  env = Env_2048(mode=mode)
  env = EnvDummy(mode=mode)

  # INIT POLICY
  policy_random = RandomPolicy(env, verbose=1)
  policy = baselineREINFORCEpolicy(env, model_type='ConvNet', t_max=t_max, verbose=1)
  policy = baselineREINFORCEpolicy(env, model_type='MLP', t_max=t_max, verbose=1)

  # TRAIN POLICY
  rewards = policy.train(total_sessions=total_sessions)
  rewards_baseline = policy_random.run_sessions(total_sessions, t_max)

  # PLOT RESULTS
  plt.plot(rewards, 'ConvNet')
  plt.plot(rewards_baseline, label='baseline')
  plt.legend()

  if visualize:
    grid_visualization = GridVisualization(env=env, model=policy, sleep_time=0.1)
    #grid_visualization = GridVisualization(env=env, model=policy_random, sleep_time=0.1)
