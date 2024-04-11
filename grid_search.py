from env_2048 import Env_2048

from policies import RandomPolicy, baselineREINFORCEpolicy
from utils import get_config, train_policy

# TODO: make learning runable in batches (to stabilize)
# TODO: save best models during the training process
# TODO: cleanup code!
# TODO: save runs with reward > 7000 for pretraining of new models
# TODO: implement data preprocessing and pretraining

if __name__ == "__main__":

  total_sessions = [4000]
  t_max = [15000]
  gammas = [0.99, 0.3, 0.5, 0.8, 0.95,] #  discount factor for reward
  epsilons = [0.0, 0.1, 0.3] #for epsilon-greedy action selection
  entropy_terms = [0.1, 0.2, 0.3, 0.5] # 0.01, 0.05,  for entropy regularization (to encourage exploration)
  lrs = [1e-4, 5e-5 ] #

  for lr in lrs:
    for gamma in gammas:
      for epsilon in epsilons:
        for entropy_term in entropy_terms:
          for total_sessions in total_sessions:

            config = get_config()
            config.total_sessions = total_sessions
            config.t_max = t_max
            config.gamma = gamma
            config.epsilon = epsilon
            config.entropy_term = entropy_term
            config.lr = lr
            config.save_name = f'{config.model_type}_sess_{total_sessions}_tmax_{t_max}_gamma_{gamma}_epsilon_{epsilon}_entropy_{entropy_term}_lr_{lr}_{config.exp}'
            print(config)

            # INIT ENVIRONMENT
            env = Env_2048(mode='agent')

            # INIT POLICY
            policy_baseline = RandomPolicy(env, verbose=0)
            policy = baselineREINFORCEpolicy(env, **config) 

            assert config.train, 'Training is disabled. Set config.train=True to train the model.'
            train_policy(config, policy, policy_baseline)
