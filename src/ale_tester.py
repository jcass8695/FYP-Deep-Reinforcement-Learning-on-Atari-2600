'''
For experimenting with ALE and testing things out
'''
from random import randrange
import numpy as np
from ale_python_interface import ALEInterface

ale = ALEInterface()
ale.setInt(b'random_seed', 123)
ale.setBool(str.encode('display_screen'), True)
ale.loadROM(str.encode('./roms/space_invaders.bin'))

specific_actions = ale.getMinimalActionSet()

try:
    for episode in range(10):
        total_reward = 0
        while not ale.game_over():
            a = specific_actions[randrange(len(specific_actions))]
            reward = ale.act(a)
            total_reward += reward

        print('Episode ended with score:', total_reward)
        ale.reset_game()

except KeyboardInterrupt:
    print('Shutting Down')
