import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from collections import deque
from ale_python_interface import ALEInterface
import numpy as np
from dqn import DeepQN
from preprocessor import Preprocessor
from replaymemory import ReplayMemory


class SpaceInvaders():
    def __init__(self, display=False, load_model=False):
        self.ale = ALEInterface()

        self.ale.setInt(str.encode('random_seed'), np.random.randint(100))
        self.ale.setBool(str.encode('display_screen'), display)
        self.ale.loadROM(str.encode('./roms/space_invaders.bin'))

        self.legal_actions = list(self.ale.getMinimalActionSet())

        # Squeeze because getScreenGrayScale returns a shape of (210, 160, 1)
        self.frame_shape = np.squeeze(self.ale.getScreenGrayscale()).shape
        self.network_input_shape = self.frame_shape + (3,)  # (210, 160, 3)
        self.frame_buffer = deque(maxlen=3)
        self.replay_memory = ReplayMemory(2000, 32)
        self.preprocessor = Preprocessor(self.frame_shape)
        self.dqn = DeepQN(
            self.network_input_shape,
            self.legal_actions,
            self.replay_memory,
            load=load_model
        )

        print('Space Invaders Loaded!')
        print('Displaying: ', display)
        print('Action Set: ', self.legal_actions)
        print('Frame Shape: ', self.frame_shape)
        print('Network Input Shape: ', self.network_input_shape)

    def train(self, max_frames=1000):
        total_reward = 0
        frame_counter = 0
        alive_counter = 0

        # Initialize frame buffer
        self.frame_buffer.append(np.squeeze(self.ale.getScreenGrayscale()))
        self.frame_buffer.append(np.squeeze(self.ale.getScreenGrayscale()))
        self.frame_buffer.append(np.squeeze(self.ale.getScreenGrayscale()))

        while frame_counter < max_frames:
            gameover = False
            initial_state = self.preprocessor.stack_frames(self.frame_buffer)
            action = self.dqn.predict_move(initial_state)
            print('Predicted Action: ', action)
            self.frame_buffer.clear()

            # Play for 3 frames and stack'em up
            for _ in range(3):
                # Save model every 10% of the way through training
                if frame_counter % max_frames * 0.1 == 0:
                    print('Frame: ', frame_counter)
                    self.dqn.save_network('model.h5')

                reward = self.ale.act(action)
                self.frame_buffer.append(
                    np.squeeze(self.ale.getScreenGrayscale()))

                total_reward += reward
                frame_counter += 1
                alive_counter += 1

            if self.ale.game_over():
                print('Gameover!')
                print('F: {}, S: {}'.format(alive_counter, total_reward))
                gameover = True
                total_reward = 0
                alive_counter = 0
                self.ale.reset_game()

            new_state = self.preprocessor.stack_frames(self.frame_buffer)
            self.replay_memory.add(
                initial_state,
                action,
                reward,
                gameover,
                new_state
            )

            self.dqn.replay_training()

    def simulate_intelligent(self):
        done = False
        total_reward = 0

        # Initialize frame buffer
        self.frame_buffer.append(np.squeeze(self.ale.getScreenGrayscale()))
        self.frame_buffer.append(np.squeeze(self.ale.getScreenGrayscale()))
        self.frame_buffer.append(np.squeeze(self.ale.getScreenGrayscale()))

        print('Simulating Game....')
        while not done:
            state = self.preprocessor.stack_frames(self.frame_buffer)
            action = self.dqn.predict_move(state)
            total_reward += self.ale.act(action)

            # Pushes oldest frame out
            self.frame_buffer.append(np.squeeze(self.ale.getScreenGrayscale()))
            if self.ale.game_over():
                done = True

        print('Game Over')
        print('Frames Survived: ', self.ale.getEpisodeFrameNumber())
        print('Score: ', total_reward)
        self.ale.reset_game()

    def simulate_random(self):
        done = False
        total_reward = 0
        print('Simulating game randomly')

        while not done:
            action = np.random.choice(self.ale.getMinimalActionSet())
            total_reward += self.ale.act(action)

            if self.ale.game_over():
                done = True

        print('Game Over')
        print('Frames Survived: ', self.ale.getEpisodeFrameNumber())
        print('Score: ', total_reward)
        self.ale.reset_game()


if __name__ == '__main__':
    game = SpaceInvaders(display=False, load_model=True)
    # game.train()
    game.simulate_random()
    game.simulate_intelligent()
