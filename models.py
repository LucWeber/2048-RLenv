import torch as t
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_size, num_actions, hidden_size=32, learning_rate=3e-4):
        super(MLP, self).__init__()

        self.num_actions = num_actions
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        #x = F.relu(self.linear2(x))
        x = F.softmax(self.linear3(x), dim=1)
        return x

    def get_probs(self, state):
        return self.forward(state)

    def get_action(self, state, epsilon=0.0):
        if type(state) == np.ndarray:
            state = t.from_numpy(state).float()
        try:
            state = state.flatten().unsqueeze(0)  # TODO:remove this for batch-processing
        except:
            pass
        probs = self.forward(state)
        if np.random.random() < epsilon: # epsilon-greedy
            sampled_actions = np.random.choice(len(probs))
        else:
            sampled_actions = Categorical(probs).sample()
        #sampled_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_probs = t.log(probs[range(len(probs)), sampled_actions])
        #sampled_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        #log_prob = t.log(probs.squeeze(0)[sampled_action])
        return sampled_actions, log_probs

class ConvNet(nn.Module):
    def __init__(self, input_size, num_actions, learning_rate=3e-4):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2, stride=1, padding=1),
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=2, stride=1, padding=1),
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=2, stride=1, padding=1),
            #nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )
        self.fc_lambda = nn.Linear(400, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.num_actions = num_actions

    def forward(self, state):
        out = self.layer1(state)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc_lambda(out)
        out = F.softmax(out, dim=1)
        return out

    def get_probs(self, state):
        return self.forward(state)

    def get_action(self, state, epsilon=0.0):
        state = state.unsqueeze(dim=1)  # add channel; & needs to be float for Conv2D -> is this bad?
        probs = self.forward(state)

        if np.random.random() < epsilon: # epsilon-greedy
            sampled_actions = np.random.choice(len(probs))
        else:
            sampled_actions = Categorical(probs).sample()
        #sampled_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_probs = t.log(probs[range(len(probs)), sampled_actions])
        return sampled_actions, log_probs

# TODO: create transformer model
class Transformer(nn.Module):
    def __init__(self, input_size, num_actions, hidden_size, learning_rate=3e-4):
        super(Transformer, self).__init__()

        self.num_actions = num_actions
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.softmax(self.linear2(x), dim=1)
        return x

    def get_probs(self, state):
        return self.forward(state)

    def get_action(self, state):
        state = state.unsqueeze(0)  # TODO:remove this for batch-processing
        probs = self.forward(Variable(state))
        if np.random.random() < epsilon: # epsilon-greedy
            sampled_actions = np.random.choice(len(probs))
        else:
            sampled_action = Categorical(probs.detach())
        log_prob = t.log(probs.squeeze(0)[sampled_action])
        return sampled_action, log_prob
