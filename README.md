2048-RLenv
===========

This is a [OpenAI-gym](https://github.com/openai/gym)-styled RL-environment for the game [2048](https://github.com/gabrielecirulli/2048) by Gabriele Cirulli. The game-logic and visualization is adapted from [Yangshun's python implementation](https://github.com/yangshun/2048-python) of the game.
<p align="center">
  <img src="https://github.com/LucWeber/2048-RLenv/blob/master/game_animation_small.gif" />
</p>
The repository contains the game environment, the possibility to train policies for the environment (with REINFORCE and a few policy networks preimplmented) and the possibility to visualize a given policy.

To set things up, run:
```bash
conda create -n 2048-RL python=3.10
conda activate 2048-RL
pip install -r requirements.txt
```
Train a REINFORCE policy with default hyperparameters using a 4-layer Transformer as policy network, run:
```bash
python RLenv_2048/scripts/main.py --train \
                --exp my_first_run \
                --model_name Transformer4L
```
Here, `--train` indicates that we want to train a policy (alternatively, you can `--visualize` or simply `--run_inference` on a given policy).

### Best scores
___________

The currently highest performing model score **12,616 points**. It is a `Transformer8L` with $\gamma$ = 0.99, $\epsilon$ = 0.1, entropy-term = 0.1 and lr = 1e-4.

### This repository is still WORK IN PROGRESS 🔧 
___________

Upcoming features are:
- script to pretrain new agents on state-action pairs collected from high-scoring agents


