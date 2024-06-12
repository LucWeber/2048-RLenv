DEFAULT_PROMPT_HF = """You are playing 2048. You have four actions to choose from:

0: up; 
1: down; 
2: left;
3: right.

The current state of the game is the following:

{state}

Please only provide the next action index as an answer, without any additional output.

Action:"""

DEFAULT_PROMPT_OAI = """You are playing 2048. You have four actions to choose from:

0: up; 
1: down; 
2: left;
3: right.

The current state of the game is the following:

{state}"""
