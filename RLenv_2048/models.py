import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.autograd import Variable

import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class MLP(nn.Module):
    def __init__(self, input_size, num_actions, hidden_size=64, learning_rate=3e-4, sampling='soft'):
        super(MLP, self).__init__()

        self.num_actions = num_actions
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, #maximize=True
                                    )
        self.sampling = sampling

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.softmax(self.linear3(x), dim=1)
        return x

    def get_probs(self, state):
        return self.forward(state)

    def get_action(self, state, legal_actions, epsilon=0.0):

        state = state.flatten().unsqueeze(0)  # TODO:remove this for batch-processing
        probs = self.forward(state)[0,0,:]

        if np.random.random() < epsilon or True not in legal_actions: # epsilon-greedy + random action if no legal actions (last turn of game)
            sampled_actions = torch.tensor(np.random.choice(len(probs)), device=device)
        else:
            # Setting probability of illegal actions to 0
            probs = probs.where(torch.tensor(legal_actions, device=device), .0)

            if self.sampling == 'soft':
                sampled_actions = Categorical(probs).sample()
            elif self.sampling == 'greedy':
                sampled_actions = torch.argmax(probs)
        
        log_probs = torch.log(probs[sampled_actions]).unsqueeze(0).unsqueeze(0)

        return sampled_actions, log_probs

class ConvNet(nn.Module):
    def __init__(self, input_size, num_actions, learning_rate=3e-4, sampling='soft'):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2)
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )
        self.fc_lambda = nn.Linear(800, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.num_actions = num_actions
        self.sampling = sampling

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

    def get_action(self, state, legal_actions, epsilon=0.0):

        state = state.flatten().unsqueeze(0)  # TODO:remove this for batch-processing
        probs = self.forward(state)[0,0,:]

        if np.random.random() < epsilon or True not in legal_actions: # epsilon-greedy + random action if no legal actions (last turn of game)
            sampled_actions = torch.tensor(np.random.choice(len(probs)), device=device)
        else:
            # Setting probability of illegal actions to 0
            probs = probs.where(torch.tensor(legal_actions, device=device), .0)

            if self.sampling == 'soft':
                sampled_actions = Categorical(probs).sample()
            elif self.sampling == 'greedy':
                sampled_actions = torch.argmax(probs)
        
        log_probs = torch.log(probs[sampled_actions]).unsqueeze(0).unsqueeze(0)

        return sampled_actions, log_probs



class Transformer(nn.Module):
    def __init__(self, input_size, num_actions, hidden_size=64, nhead=4, nlayers=2, learning_rate=3e-4, sampling='soft'):
        super(Transformer, self).__init__()
        self.model_type = 'Transformer'
        self.num_actions = num_actions
        self.pos_encoder = PositionalEncoding(hidden_size)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=nlayers)
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.init_weights()
        self.sampling = sampling

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.encoder(src) * np.sqrt(self.num_actions)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return torch.softmax(output, dim=-1)
        
    def get_action(self, state, legal_actions, epsilon=0.0):

        state = state.flatten().unsqueeze(0)  # TODO:remove this for batch-processing
        probs = self.forward(state)[0,0,:]

        if np.random.random() < epsilon or True not in legal_actions: # epsilon-greedy + random action if no legal actions (last turn of game)
            sampled_actions = torch.tensor(np.random.choice(len(probs)), device=device)
        else:
            # Setting probability of illegal actions to 0
            probs = probs.where(torch.tensor(legal_actions, device=device), .0)

            if self.sampling == 'soft':
                sampled_actions = Categorical(probs).sample()
            elif self.sampling == 'greedy':
                sampled_actions = torch.argmax(probs)
        
        log_probs = torch.log(probs[sampled_actions]).unsqueeze(0).unsqueeze(0)

        return sampled_actions, log_probs


class Transformer4L(nn.Module):
    def __init__(self, input_size, num_actions, hidden_size=128, nhead=8, nlayers=4, learning_rate=3e-4, sampling='soft'):
        super(Transformer4L, self).__init__()
        self.model_type = 'Transformer'
        self.num_actions = num_actions
        self.pos_encoder = PositionalEncoding(hidden_size)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=nlayers)
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.init_weights()
        self.sampling = sampling

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.encoder(src) * np.sqrt(self.num_actions)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return torch.softmax(output, dim=-1)
        
    def get_action(self, state, legal_actions, epsilon=0.0):

        state = state.flatten().unsqueeze(0)  # TODO:remove this for batch-processing
        probs = self.forward(state)[0,0,:]

        if np.random.random() < epsilon or True not in legal_actions: # epsilon-greedy + random action if no legal actions (last turn of game)
            sampled_actions = torch.tensor(np.random.choice(len(probs)), device=device)
        else:
            # Setting probability of illegal actions to 0
            probs = probs.where(torch.tensor(legal_actions, device=device), .0)


            if self.sampling == 'soft':
                sampled_actions = Categorical(probs).sample()
            elif self.sampling == 'greedy':
                sampled_actions = torch.argmax(probs)
        
        log_probs = torch.log(probs[sampled_actions]).unsqueeze(0).unsqueeze(0)

        return sampled_actions, log_probs

class Transformer8L(nn.Module):
    def __init__(self, input_size, num_actions, hidden_size=128, nhead=8, nlayers=8, learning_rate=3e-4, sampling='soft'):
        super(Transformer8L, self).__init__()
        self.model_type = 'Transformer'
        self.num_actions = num_actions
        self.pos_encoder = PositionalEncoding(hidden_size)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=nlayers)
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.init_weights()
        self.sampling = sampling # otpions: soft vs. greedy

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.encoder(src) * np.sqrt(self.num_actions)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return torch.softmax(output, dim=-1)
        
    def get_action(self, state, legal_actions, epsilon=0.0):

        state = state.flatten().unsqueeze(0)  # TODO:remove this for batch-processing
        probs = self.forward(state)[0,0,:]

        if np.random.random() < epsilon or True not in legal_actions: # epsilon-greedy + random action if no legal actions (last turn of game)
            sampled_actions = torch.tensor(np.random.choice(len(probs)), device=device)
        else:
            # Setting probability of illegal actions to 0
            probs = probs.where(torch.tensor(legal_actions, device=device), .0)

            if self.sampling == 'soft':
                try:
                    sampled_actions = Categorical(probs).sample()
                except:
                    print('Some error with the probs. See probs:')
                    print(probs)
                    print('See legal actions:')
                    print(legal_actions)
                    print('Continue with random action')
                    sampled_actions = torch.tensor(np.random.choice(len(probs)), device=device)

            elif self.sampling == 'greedy':
                sampled_actions = torch.argmax(probs)
        
        log_probs = torch.log(probs[sampled_actions]).unsqueeze(0).unsqueeze(0)

        return sampled_actions, log_probs


class Transformer12L(nn.Module):
    def __init__(self, input_size, num_actions, hidden_size=128, nhead=8, nlayers=12, learning_rate=3e-4, sampling='soft'):
        super(Transformer12L, self).__init__()
        self.model_type = 'Transformer'
        self.num_actions = num_actions
        self.pos_encoder = PositionalEncoding(hidden_size)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=nlayers)
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.init_weights()
        self.sampling = sampling # otpions: soft vs. greedy

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.encoder(src) * np.sqrt(self.num_actions)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return torch.softmax(output, dim=-1)
        
    def get_action(self, state, legal_actions, epsilon=0.0):

        state = state.flatten().unsqueeze(0)  # TODO:remove this for batch-processing
        probs = self.forward(state)[0,0,:]

        if np.random.random() < epsilon or True not in legal_actions: # epsilon-greedy + random action if no legal actions (last turn of game)
            sampled_actions = torch.tensor(np.random.choice(len(probs)), device=device)
        else:
            # Setting probability of illegal actions to 0
            probs = probs.where(torch.tensor(legal_actions, device=device), .0)

            if self.sampling == 'soft':
                try:
                    sampled_actions = Categorical(probs).sample()
                except:
                    print('Some error with the probs. See probs:')
                    print(probs)
                    print('See legal actions:')
                    print(legal_actions)
                    print('Continue with random action')
                    sampled_actions = torch.tensor(np.random.choice(len(probs)), device=device)

            elif self.sampling == 'greedy':
                sampled_actions = torch.argmax(probs)
        
        log_probs = torch.log(probs[sampled_actions]).unsqueeze(0).unsqueeze(0)

        return sampled_actions, log_probs

class Transformer16L(nn.Module):
    def __init__(self, input_size, num_actions, hidden_size=128, nhead=8, nlayers=16, learning_rate=3e-4, sampling='soft'):
        super(Transformer16L, self).__init__()
        self.model_type = 'Transformer'
        self.num_actions = num_actions
        self.pos_encoder = PositionalEncoding(hidden_size)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=nlayers)
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.init_weights()
        self.sampling = sampling # otpions: soft vs. greedy

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.encoder(src) * np.sqrt(self.num_actions)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return torch.softmax(output, dim=-1)
        
    def get_action(self, state, legal_actions, epsilon=0.0):

        state = state.flatten().unsqueeze(0)  # TODO:remove this for batch-processing
        probs = self.forward(state)[0,0,:]

        if np.random.random() < epsilon or True not in legal_actions: # epsilon-greedy + random action if no legal actions (last turn of game)
            sampled_actions = torch.tensor(np.random.choice(len(probs)), device=device)
        else:
            # Setting probability of illegal actions to 0
            probs = probs.where(torch.tensor(legal_actions, device=device), .0)

            if self.sampling == 'soft':
                sampled_actions = Categorical(probs).sample()
            elif self.sampling == 'greedy':
                sampled_actions = torch.argmax(probs)
        
        log_probs = torch.log(probs[sampled_actions]).unsqueeze(0).unsqueeze(0)

        return sampled_actions, log_probs
 

class Transformer20L(nn.Module):
    def __init__(self, input_size, num_actions, hidden_size=128, nhead=8, nlayers=20, learning_rate=3e-4, sampling='soft'):
        super(Transformer20L, self).__init__()
        self.model_type = 'Transformer'
        self.num_actions = num_actions
        self.pos_encoder = PositionalEncoding(hidden_size)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=nlayers)
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.init_weights()
        self.sampling = sampling # otpions: soft vs. greedy

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.encoder(src) * np.sqrt(self.num_actions)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return torch.softmax(output, dim=-1)
        
    def get_action(self, state, legal_actions, epsilon=0.0):

        state = state.flatten().unsqueeze(0)  # TODO:remove this for batch-processing
        probs = self.forward(state)[0,0,:]

        if np.random.random() < epsilon or True not in legal_actions: # epsilon-greedy + random action if no legal actions (last turn of game)
            sampled_actions = torch.tensor(np.random.choice(len(probs)), device=device)
        else:
            # Setting probability of illegal actions to 0
            probs = probs.where(torch.tensor(legal_actions, device=device), .0)

            if self.sampling == 'soft':
                try:
                    sampled_actions = Categorical(probs).sample()
                except:
                    print('Some error with the probs. See probs:')
                    print(probs)
                    print('See legal actions:')
                    print(legal_actions)
                    print('Continue with random action')
                    sampled_actions = torch.tensor(np.random.choice(len(probs)), device=device)

            elif self.sampling == 'greedy':
                sampled_actions = torch.argmax(probs)
        
        log_probs = torch.log(probs[sampled_actions]).unsqueeze(0).unsqueeze(0)

        return sampled_actions, log_probs



class Transformer32L(nn.Module):
    def __init__(self, input_size, num_actions, hidden_size=128, nhead=8, nlayers=32, learning_rate=3e-4, sampling='soft'):
        super(Transformer32L, self).__init__()
        self.model_type = 'Transformer'
        self.num_actions = num_actions
        self.pos_encoder = PositionalEncoding(hidden_size)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=nlayers)
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.init_weights()
        self.sampling = sampling # otpions: soft vs. greedy

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.encoder(src) * np.sqrt(self.num_actions)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return torch.softmax(output, dim=-1)
        
    def get_action(self, state, legal_actions, epsilon=0.0):

        state = state.flatten().unsqueeze(0)  # TODO:remove this for batch-processing
        probs = self.forward(state)[0,0,:]

        if np.random.random() < epsilon or True not in legal_actions: # epsilon-greedy + random action if no legal actions (last turn of game)
            sampled_actions = torch.tensor(np.random.choice(len(probs)), device=device)
        else:
            # Setting probability of illegal actions to 0
            probs = probs.where(torch.tensor(legal_actions, device=device), .0)

            if self.sampling == 'soft':
                try:
                    sampled_actions = Categorical(probs).sample()
                except:
                    print('Some error with the probs. See probs:')
                    print(probs)
                    print('See legal actions:')
                    print(legal_actions)
                    print('Continue with random action')
                    sampled_actions = torch.tensor(np.random.choice(len(probs)), device=device)

            elif self.sampling == 'greedy':
                sampled_actions = torch.argmax(probs)
        
        log_probs = torch.log(probs[sampled_actions]).unsqueeze(0).unsqueeze(0)

        return sampled_actions, log_probs
  

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

MODEL_REGISTER = {'MLP': MLP, 'ConvNet': ConvNet, 'Transformer': Transformer, 'Transformer4L': Transformer4L, 'Transformer8L': Transformer8L, 'Transformer12L': Transformer12L,
                  'Transformer16L': Transformer16L, 'Transformer20L': Transformer20L, 'Transformer32L': Transformer32L}

def get_model(model_name, 
              input_size, 
              num_actions, 
              lr, 
              device,
              sampling='soft',
              **kwargs
              ):
    if model_name in MODEL_REGISTER:
            model_type = model_name
            model = MODEL_REGISTER[model_type](input_size=input_size,
                                                num_actions=num_actions,
                                                learning_rate=lr,
                                                sampling=sampling)
    else:
        model_type = model_name.split('_')[1]
        model = MODEL_REGISTER[model_type](input_size=input_size,
                                            num_actions=num_actions,
                                            learning_rate=lr,
                                            sampling=sampling)
        
        sd = torch.load(f'./RLenv_2048/saved_models/{model_name}.ckpt', map_location=device)
        model.load_state_dict(sd)
        try:
            sd_optimizer = torch.load(f'./RLenv_2048/saved_models/{model_name}_optimizer.ckpt', map_location=device)
            model.optimizer.load_state_dict(sd_optimizer)
            model.optimizer.param_groups[0]['lr'] = lr
            for state in model.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        except:
            print('No optimizer found for model. Continue with newly initialised optimizer.')
    return model, model_type
