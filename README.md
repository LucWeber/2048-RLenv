2048-RLenv
===========

This is a [OpenAI-gym](https://github.com/openai/gym)-styled RL-environment for the game [2048](https://github.com/gabrielecirulli/2048) by Gabriele Cirulli. The game-logic and visualization is adapted from [yangshun's python implementation](https://github.com/yangshun/2048-python) of the game.

To set things up, run:

    $ conda create -n 2048-RL python=3.8
    $ pip install -r requirements.txt
    $ source activate 2048-RL

To train a policy, run:
    
    $ python main.py 
