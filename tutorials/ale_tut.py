import sys
from random import randrange
from ale_python_interface import ALEInterface

if len(sys.argv) < 2:
    print('Usage:', sys.argv[0], 'rom_file')
    sys.exit()

ale = ALEInterface()
ale.setInt(b'random_seed', 123)
ale.setBool(b'display_screen', True)
rom_file = str.encode(sys.argv[1])
ale.loadROM(rom_file)

# Get the list of legal actions
legal_actions = ale.getLegalActionSet()


for episode in range(10):
    total_reward = 0
    while not ale.game_over():
        a = legal_actions[randrange(len(legal_actions))]
        reward = ale.act(a)
        total_reward += reward

    print('Episode ended with score:', total_reward)
    ale.reset_game()
