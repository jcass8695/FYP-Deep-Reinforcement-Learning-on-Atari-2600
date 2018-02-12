#! ../venv/bin/python

import os
import sys
from collections import deque
from datetime import datetime
from ale_python_interface import ALEInterface
import numpy as np
import matplotlib.pyplot as plt
from dqn import DeepQN
from preprocessor import Preprocessor
from replaymemory import ReplayMemory

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class SpaceInvaders():
    def __init__(self, display=False, load_model=False):
        self.name = 'space_invaders'
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
            self.name,
            load=load_model
        )

        print('Space Invaders Loaded!')
        print('Displaying: ', display)
        print('Action Set: ', self.legal_actions)
        print('Frame Shape: ', self.frame_shape)
        print('Network Input Shape: ', self.network_input_shape)

    def train(self, max_frames=100000):
        total_reward = 0
        frame_counter = 0
        alive_counter = 0

        # Initialize frame buffer
        self.frame_buffer.append(np.squeeze(self.ale.getScreenGrayscale()))
        self.frame_buffer.append(np.squeeze(self.ale.getScreenGrayscale()))
        self.frame_buffer.append(np.squeeze(self.ale.getScreenGrayscale()))

        try:
            while frame_counter < max_frames:
                print('Frame: ', frame_counter)
                gameover = False
                initial_state = self.preprocessor.stack_frames(
                    self.frame_buffer)
                action = self.dqn.predict_move(initial_state)
                self.frame_buffer.clear()

                # Play for 3 frames and stack'em up
                for _ in range(3):
                    # Save model every 500 frames
                    if frame_counter % 500 == 0:
                        self.dqn.save_network(self.name)

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

        except:
            self.dqn.save_network(self.name)
            self.ale.reset_game()
            print('Stopped on frame: ', frame_counter)

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
        return total_reward

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

    def save_results(self, pathx, pathy, xdata, ydata):
        ''' Save a set of metrics to numpy arrays specified by pathx, pathy '''

        try:
            xdata_loaded = np.load(pathx)
            ydata_loaded = np.load(pathy)
            xdata_loaded = np.append(xdata_loaded, xdata)
            ydata_loaded = np.append(ydata_loaded, ydata)
        except FileNotFoundError:
            if '.npy' not in pathx or '.npy' not in pathy:
                print('Paths must include the .npy extension')
                return

        print('Saved {}, {} at {}'.format(pathx, pathx, datetime.now()))
        np.save(pathx, xdata)
        np.save(pathy, ydata)

    def plot_results(self, pathx, pathy):
        try:
            xdata = np.load(pathx)
            ydata = np.load(pathy)
        except FileNotFoundError:
            if '.npy' not in pathx or '.npy' not in pathy:
                print('Paths must include the .npy extension')
            else:
                print('No data to plot found at {} or {}'.format(pathx, pathy))

            return

        plt.plot(xdata, ydata, 'ro')
        plt.show()


if __name__ == '__main__':
    game = SpaceInvaders(display=False, load_model=False)
    interval = 500
    max_frames = 500000
    games_to_play = 1

    for i in range(max_frames // interval):
        game.train(interval)
        running_score = 0
        for _ in range(games_to_play):
            running_score += game.simulate_intelligent()

        running_score /= games_to_play

        # Save the Average Score over 10 games for this interval
        game.save_results(
            './data/{}_avgscorex_dqn.npy'.format(game.name),
            './data/{}_avgscorey_dqn.npy'.format(game.name),
            (i + 1) * interval,
            running_score
        )

        sys.exit()
