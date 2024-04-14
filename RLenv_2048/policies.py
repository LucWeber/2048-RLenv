import numpy as np
import torch
import os
from tqdm import tqdm
from collections import Counter
import pickle as pkl

from RLenv_2048.models import get_model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

    def __init__(self, env, model_name='MLP', t_max=1000, verbose=0, gamma=0.99, epsilon=0.0, entropy_term=0.1, lr=1e-4, save_name='model', sampling='soft', **kwargs):
        
        input_size = env.observation_space.shape[0] * env.observation_space.shape[1]

        self.env = env
        self.lr = lr
        self.save_name = save_name
        
        self.model, self.model_type = get_model(model_name=model_name, input_size=input_size, num_actions=env.action_space.n,lr=lr, sampling=sampling, device=device)
        self.model = self.model.to(device)

        self.verbose = verbose
        self.t_max = t_max
        self.epsilon = epsilon # for epsilon-greedy action selection
        self.gamma = gamma
        self.entropy_term = entropy_term
        self.training_steps = 0
        self.total_n_illegal_moves = []

    def train(self, total_sessions=1):
        """ train policy network for given amount of sessions """
        final_scores = []
        self.training_steps = total_sessions
        current_best_score = 3000 # we want to save models that perform better than this across recent sessions
        
        pbar = tqdm(range(total_sessions), desc="Train policy")
        for i, session in enumerate(pbar):
            final_scores, current_best_score = self.train_session(final_scores, current_best_score)
            self.total_n_illegal_moves.append(self.env.n_illegal_actions)
            pbar.set_postfix(Score=f'{final_scores[-1]}', refresh=False)
        self.save_model()
        return final_scores

    def train_session(self, final_scores, current_best_score, save_high_reward_sessions=True):
        states, actions, rewards, log_probs = generate_session(env=self.env, model=self.model,
                                                               t_max=self.t_max, epsilon=self.epsilon)

        if save_high_reward_sessions:
            if self.env.total_score > 8000:
                save_folder = f'./RLenv_2048/pretrain_data'
                os.makedirs(save_folder, exist_ok=True)
                save_file = f'state_action_pairs_high_reward_{self.env.total_score}_{self.save_name}.pt'
                torch.save((torch.tensor(states).cpu(), torch.stack(actions).cpu(), torch.tensor(rewards).cpu()), open(os.path.join(save_folder, save_file), 'wb'))
                print(f'Saved state-action pairs!')

        rewards = torch.tensor(rewards).unsqueeze(dim=0)
        log_probs = torch.cat(log_probs).unsqueeze(dim=0)

        self.update_policy(rewards, log_probs)
        final_scores.append(self.env.total_score)

        # check most recent 4 scores, compare it with the best score and save model if it is better
        current_score = torch.mean(torch.tensor(final_scores if len(final_scores) < 4 else final_scores[-4:] , dtype=float))
        if current_score > current_best_score:
            print(f'New best score: {current_score:.2f}. Saving model.')
            self.save_model(best=True, verbose=False)
            current_best_score = current_score

        return final_scores, current_best_score

    def run_sessions(self, n_sessions, t_max, save_high_reward_sessions=True):
        with torch.no_grad():
            pbar = tqdm(range(n_sessions), desc="Extracting state-action pairs")
            for i, session in enumerate(pbar):
                states, actions, rewards, _ = generate_session(env=self.env, model=self.model,
                                                                    t_max=t_max, epsilon=self.epsilon)
                self.print_distribution_actions(actions)
                if save_high_reward_sessions:
                    if self.env.total_score > 8000:
                        save_folder = f'./RLenv_2048/pretrain_data'
                        os.makedirs(save_folder, exist_ok=True)
                        save_file = f'state_action_pairs_high_reward_{self.env.total_score}_{self.save_name}.pt'
                        torch.save((torch.tensor(states).cpu(), torch.stack(actions).cpu(), torch.tensor(rewards).cpu()), open(os.path.join(save_folder, save_file), 'wb'))
                        print(f'Saved state-action pairs!')
                pbar.set_postfix(reward=f'{self.env.total_score}', refresh=False)

    def print_distribution_actions(self, actions):
        """ check distribution of actions for error analysis """
        if self.verbose:
            a_for_counting = [int(a) for a in actions]
            print(Counter(a_for_counting))

    def predict(self, state, sampling='greedy'):
        inputs = torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0).to(device)
        action = self.model.get_action(inputs)
        return action[0].item()


    def update_policy(self, batch_rewards, batch_log_probs):
        policy_gradients = []

        entropy = 0
        for rewards, log_probs in zip(batch_rewards, batch_log_probs):
            discounted_rewards = get_cumulative_rewards(rewards, self.gamma)
            discounted_rewards = torch.tensor(discounted_rewards)
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
                    discounted_rewards.std() + 1e-9)  # normalize discounted rewards

            for log_prob, Gt in zip(log_probs, discounted_rewards):
                policy_gradients.append(-log_prob * Gt)
            
            # Calculate entropy of the action probabilities
            probs = torch.exp(log_probs)
            entropy += -(probs * log_probs).sum(-1).mean()
            #regularization_term = -(self.entropy_term * entropy)

        entropy /= len(batch_rewards)

        self.model.optimizer.zero_grad()

        policy_gradient = (1 - self.entropy_term) * torch.stack(policy_gradients).sum() - (self.entropy_term * entropy)
        policy_gradient.backward()
        self.model.optimizer.step()


    def save_model(self, best=False, verbose=True):
        save_folder = f'./RLenv_2048/saved_models'
        os.makedirs(save_folder, exist_ok=True)

        save_file = f'REINFORCE_{self.save_name}{"_best" if best else ""}.ckpt'
        save_path = os.path.join(save_folder, save_file)

        torch.save(self.model.state_dict(), save_path)
        if verbose:
            print(f'Successfully saved model to {save_path}!')
        
        save_file = f'REINFORCE_{self.save_name}{"_best" if best else ""}_optimizer.ckpt'
        save_path = os.path.join(save_folder, save_file)
        torch.save(self.model.optimizer.state_dict(), save_path)
        if verbose:
            print(f'Successfully saved optimizer to {save_path}!')


def generate_session(env, model, t_max, epsilon=0.0):
    """
    Returns sequences of states, actions, and rewards.
    """
    # arrays to record session
    states, actions, rewards, log_probs = [], [], [], []
    state = env.reset()

    for _ in range(t_max):
        # this is for faking a batch:
        inputs = torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0).to(device)
        inputs.requires_grad = True
        legal_actions = env.get_legal_actions()
        action, log_prob = model.get_action(inputs, legal_actions, epsilon=epsilon)
        log_probs.append(log_prob)

        # take action
        new_state, reward, done, info = env.step(int(action))

        # record session history to train later
        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state = new_state
        if done:
            break
    
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
