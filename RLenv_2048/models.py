import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.autograd import Variable
from transformers import AutoModelForCausalLM, AutoTokenizer
#from RLenv_2048.utils import retry_with_exponential_backoff

try:
    import random
    import time
    import openai
    from openai import OpenAI
    
except:
    pass

import numpy as np
from RLenv_2048.prompts import DEFAULT_PROMPT_HF, DEFAULT_PROMPT_OAI

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# define a retry decorator
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple = (openai.RateLimitError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


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

class HFlocal_Wrapper():
    def __init__(self, model_type="mistralai/Mistral-7B-Instruct-v0.2", use_chat_template=True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_type, device_map='auto')#.to(device)
        self.model = AutoModelForCausalLM.from_pretrained(model_type, load_in_8bit=True,pad_token_id=self.tokenizer.eos_token_id, device_map='auto')#.to(device)

        self.model_type = model_type
        self.optimizer = None
        self.use_chat_template = use_chat_template
        self.prompt = DEFAULT_PROMPT_HF if not use_chat_template else DEFAULT_PROMPT_OAI
        self.previous_state = ''
        self.previous_action = 0

    def forward(self, state, legal_actions):
        
        if self.use_chat_template:
            state = state[0].int().tolist()
            state = '\n'.join([str(s) for s in state])
            state += f'\n\nCurrent legal actions are {[i for i, legal in enumerate(legal_actions) if legal]} \n\nPlease only provide an integer as next action index as an answer, without any additional output. Your action:'
            print(state)
            if self.previous_state=='':
                self.previous_state = state
                return torch.tensor(0)
            chat = [{"role": "user", "content": self.prompt.format(state=self.previous_state)},
                    {"role": "assistant", "content": f'{self.previous_action}'},
                    {"role": "user", "content": self.prompt.format(state=state)},
            ]

            input_ids = self.tokenizer.apply_chat_template(chat, tokenize=True, return_tensors='pt')
            self.previous_state = state
        else:
            input_ids = self.tokenizer.encode(self.prompt.format(state=state.tolist()), return_tensors='pt') #.to(self.device)
        output = self.model.generate(input_ids, max_new_tokens=2, num_return_sequences=1, do_sample=False, #return_full_text=False
                                     )
        out = self.tokenizer.decode(output[0], skip_special_tokens=True)
        try:
            out = int(out)
        except:
            out = np.random.choice(len(legal_actions))
        self.previous_action = out
        return torch.tensor(out)
    
    def get_action(self, state, legal_actions, epsilon=0.0):
        if True not in legal_actions:
            return torch.tensor(np.random.choice(len(legal_actions))), None
        else:
            return self.forward(state, legal_actions), None
    
class OAIWrapper():
    def __init__(self, model_type="gpt-3.5-turbo-0125"):
        self.client = OpenAI()
        self.model_type = model_type
        self.optimizer = None
        self.device = device
        self.prompt = DEFAULT_PROMPT_OAI
        self.previous_state = ''
        self.test = False

    @retry_with_exponential_backoff
    def completions_with_backoff(self, **kwargs):
        return self.client.chat.completions.create(**kwargs)

    def forward(self, state, legal_actions):
        
        state = state[0].int().tolist()
        state = '\n'.join([str(s) for s in state])
        state += f'\n\n current legal actions are {[i for i, legal in enumerate(legal_actions) if legal]} '
        print(state)
        stuck = self.previous_state == state
        
        if not self.test and not stuck:
            #self.client.chat.completions.create
            response = self.completions_with_backoff(
            model=self.model_type,
            messages=[
                {"role": "system", "content": self.prompt.format(state=state)},
                {"role": "user", "content": "Please only provide the next action index as an answer, without any additional output."}
            ],
            temperature=.0,
            )
            response = int(float(response.choices[0].message.content))
            self.previous_state = state
        else:
            response = np.random.choice(len(legal_actions))
            self.previous_state = state
        return torch.tensor(response)

    
    def get_action(self, state, legal_actions, epsilon=0.0):
        
        if True not in legal_actions:
            return torch.tensor(np.random.choice(len(legal_actions))), None
        else:
            return self.forward(state, legal_actions), None


MODEL_REGISTER = {'MLP': MLP, 
                  'ConvNet': ConvNet, 
                  'Transformer': Transformer, 
                  'Transformer4L': Transformer4L, 
                  'Transformer8L': Transformer8L, 
                  'Transformer12L': Transformer12L,
                  'Transformer16L': Transformer16L, 
                  'Transformer20L': Transformer20L, 
                  'Transformer32L': Transformer32L,
                  #'wrapper_hf': HF_Wrapper,
                  'wrapper_hflocal': HFlocal_Wrapper,
                  'wrapper_openai': OAIWrapper}

def get_model(model_name, 
              input_size, 
              num_actions, 
              lr, 
              device,
              sampling='soft',
              **kwargs
              ):
    
    if 'wrapper_' in model_name:
        model_type = model_name.split('_')[2]
        wrapper_type = '_'.join(model_name.split('_')[:2])
        model = MODEL_REGISTER[wrapper_type](model_type)

    elif model_name in MODEL_REGISTER:
        model_type = model_name
        model = MODEL_REGISTER[model_type](input_size=input_size,
                                            num_actions=num_actions,
                                            learning_rate=lr,
                                            sampling=sampling).to(device)
    else:
        model_type = model_name.split('_')[1]
        model = MODEL_REGISTER[model_type](input_size=input_size,
                                            num_actions=num_actions,
                                            learning_rate=lr,
                                            sampling=sampling).to(device)
        
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
