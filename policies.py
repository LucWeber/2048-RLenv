import numpy as np
import torch
import models
import os
from tqdm import tqdm
from collections import Counter

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    device_name = 'cuda:0'
else:
    device_name = 'cpu'
    
device = torch.device(device_name)


class RandomPolicy:
    def __init__(self, env, verbose=0):
        self.env = env
        self.verbose = verbose

    def train(self, total_sessions=1):
        return [0] * total_sessions

    def run_sessions(self, n_sessions, t_max):
        total_rewards = []
        for session in range(n_sessions):
            obs = self.env.reset()
            rewards_sum = 0
            for i in range(t_max):
                action = self.predict(obs)
                obs, rewards, done, info = self.env.step(action)
                rewards_sum += rewards
                if done: break
            total_rewards.append(rewards_sum)
        return total_rewards

    def predict(self, state=None):
        return self.env.action_space.sample()


class baselineREINFORCEpolicy:
    """
    TODO: adapt for batch-processing
    adapt such that you cannot see where it is from
    """

    def __init__(self, env, model_type='MLP', t_max=1000, verbose=0):
        self.env = env
        self.lr = 1e-3
        # TODO: generalize this to other dimensionality:
        input_size = env.observation_space.shape[0] * env.observation_space.shape[1]
        self.model = getattr(models, model_type)(input_size=input_size,
                                                 num_actions=env.action_space.n,
                                                 learning_rate=self.lr)

        self.model = self.model.to(device)

        self.model_type = model_type
        self.verbose = verbose
        self.t_max = t_max
        self.gamma = 0.8
        self.training_steps = 0
        self.total_n_illegal_moves = []

    def train(self, total_sessions=1):
        """ train policy network for given amount of sessions """
        final_rewards = []
        self.training_steps = total_sessions
        for i, session in enumerate(tqdm(range(total_sessions))):
            final_rewards = self.train_session(final_rewards)
            self.total_n_illegal_moves.append(self.env.n_illegal_actions)
        self.save_model()
        return final_rewards

    def train_session(self, final_rewards):
        states, actions, rewards, log_probs = generate_session(env=self.env, model=self.model,
                                                               t_max=self.t_max)

        rewards = torch.tensor(rewards).unsqueeze(dim=0)
        log_probs = torch.cat(log_probs).unsqueeze(dim=0)

        self.update_policy(rewards, log_probs)
        final_rewards.append(self.env.total_score)

        return final_rewards

    def print_distribution_actions(self, actions):
        """ check distribution of actions for error analysis """
        if self.verbose:
            a_for_counting = [int(a) for a in actions]
            print(Counter(a_for_counting))

    def predict(self, state, sampling='greedy'):
        input = torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
        action = self.model.get_action(input)
        return action[0].item()

    def predict1(self, state, sampling='greedy'):
        ''' commented out for debugging'''
        if self.model_type == 'MLP':
            input = np.asarray(state).flatten()
        else:
            input = np.expand_dims(np.asarray(state), axis=0)
        action, _ = self.model.get_action(input)
        return action

    def update_policy0(self, rewards, log_probs):
        discounted_rewards = get_cumulative_rewards(rewards, self.gamma)

        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
                discounted_rewards.std() + 1e-9)  # normalize discounted rewards

        policy_gradient = []
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * Gt)

        self.model.optimizer.zero_grad()

        policy_gradient = torch.stack(policy_gradient).sum()

        policy_gradient.backward()
        self.model.optimizer.step()

    def update_policy(self, batch_rewards, batch_log_probs):
        policy_gradients = []

        for rewards in batch_rewards:
            discounted_rewards = get_cumulative_rewards(rewards, self.gamma)
            discounted_rewards = torch.tensor(discounted_rewards)
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
                    discounted_rewards.std() + 1e-9)  # normalize discounted rewards

            for log_prob, Gt in zip(batch_log_probs, discounted_rewards):
                policy_gradients.append(-log_prob * Gt)

        self.model.optimizer.zero_grad()
        policy_gradient = torch.sum(torch.stack(policy_gradients, dim=0), dim=1)  # .sum()
        policy_gradient.backward()
        self.model.optimizer.step()

    def save_model(self):
        save_folder = f'./saved_models'
        os.makedirs(save_folder, exist_ok=True)

        save_file = f'REINFORCE_{self.model_type}_gamma_{self.gamma}_lr_{self.lr}_steps_' \
                    f'{self.training_steps}_tmax_{self.t_max}.ckpt'

        save_path = os.path.join(save_folder, save_file)

        torch.save(self.model.state_dict(), save_path)
        print(f'Successfully saved model to {save_path}!')


def generate_session(env, model, t_max):
    """
    Returns sequences of states, actions, and rewards.
    """
    # arrays to record session
    states, actions, rewards, log_probs = [], [], [], []
    state = env.reset()

    for _ in range(t_max):
        '''
        if self.model_type == 'MLP':
            input = np.asarray(state).flatten()
        else:
        '''
        # this is for faking a batch:
        input = torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
        input.requires_grad = True
        action, log_prob = model.get_action(input)
        log_probs.append(log_prob)

        # Sample action with given probabilities.
        new_state, reward, done, info = env.step(int(action))

        # record session history to train later
        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = new_state
        if done:
            break

    #print(f'number illegal actions: {env.n_illegal_actions}')

    return states, actions, rewards, log_probs


def get_cumulative_rewards(rewards, gamma=0.99):
    """
  rewards: rewards at each step
  gamma: discount for reward
  """
    discounted_rewards = []

    for rew in range(len(rewards)):
        Gt = 0
        pw = 0
        for r in rewards[rew:]:
            Gt = Gt + gamma ** pw * r
            pw = pw + 1
        discounted_rewards.append(Gt)

    return discounted_rewards


def get_cumulative_rewards2(rewards, gamma=0.99):
    """
    recursive implementation:
    rewards: rewards at each step
    gamma: discount for reward
    """

    def cum_sum(rewards_all, cum_sums=[]):
        if not cum_sums:
            cum_sums = cum_sum(rewards_all, [rewards_all[0]])
            return cum_sums
        elif len(cum_sums) == len(rewards_all):
            return cum_sums
        else:
            wts = gamma ** ((len(cum_sums)) - torch.arange(0, len(cum_sums)))
            itr = len(cum_sums)
            cum_sums = cum_sums + [rewards_all[itr] + sum([c * float(wts[i]) for i, c in enumerate(rewards_all[:itr])])]
            cum_sums = cum_sum(rewards_all, cum_sums)
            return cum_sums

    rewards = list(reversed(rewards))
    result = cum_sum(rewards)
    return list(reversed(result))
