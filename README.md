2048-RLenv
===========

This is a [OpenAI-gym](https://github.com/openai/gym)-styled RL-environment for the game [2048](https://github.com/gabrielecirulli/2048) by Gabriele Cirulli. The game-logic and visualization is adapted from [Yangshun's python implementation](https://github.com/yangshun/2048-python) of the game.
<p align="center">
  <img src="https://github.com/LucWeber/2048-RLenv/blob/master/game_animation_small.gif" />
</p>
The repository contains the game environment, the possibility to train policies for the environment (with REINFORCE and a few policy networks preimplmented) and the possibility to visualize a given policy.


### Setup
___________
Set things up by running:
```bash
conda create -n 2048-RL python=3.10
conda activate 2048-RL
pip install -r requirements.txt
```
Train a REINFORCE policy with default hyperparameters using a 4-layer Transformer as policy network by running:
```bash
python RLenv_2048/scripts/main.py --train \
                                  --exp my_first_run \
                                  --model_name Transformer4L
```

Here, the arg `--model_name` specifies the policy network to use. If you train from scratch, use one of the pre-implemented model-classes from `RLenv_2048/models` or implement your own in the same file (don't forget to add it to the `MODEL_REGISTER` on the bottom!). Otherwise, you can also use a pre-trained network. To do so, use the file-name of the saved model as `--model_name`, like so:

```bash
python RLenv_2048/scripts/main.py --visualize \
                                  --model_name REINFORCE_Transformer4L_sess_2000_tmax_10000_gamma_0.99_epsilon_0.0_entropy_0.1_lr_0.0001_greedy
```
Note that I replaced the `--train` flag with `--visualize`. We can train but also visualize or simply `--run_inference` on a given policy that we are interested in.

### Best scores
___________

The currently highest performing model score **16,920 points**. It is a `Transformer12L` with $\gamma$ = 0.95, $\epsilon$ = 0.0, entropy-term = 0.0, lr = 1e-4 and soft sampling.

### This repository is still WORK IN PROGRESS 🔧 
___________

Upcoming features are:
- script to pretrain new agents on state-action pairs collected from high-scoring agents


