#!/home/support/apps/cports/rhel-7.x86_64/gnu/Python/3.6.4/bin/python3
# The shebang is for usage on Trinity's Boole weird module thing (sigh`git)
import os
import sys
import traceback
import argparse
from collections import deque
from datetime import datetime
from ale_python_interface import ALEInterface
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from dqn import DeepQN
from double_dqn import DoubleDQN
from preprocessor import Preprocessor
from replaymemory import ReplayMemory

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
parser = argparse.ArgumentParser()
parser.add_argument(
    'type',
    help='The type of Deep Learning to use',
    type=str,
    choices=['dqn', 'ddqn']
)
parser.add_argument(
    'max_frames',
    help='The max number of frames to train the model for',
    type=int
)
parser.add_argument(
    'interval_frames',
    help='The number of frames to run in between game simulation testing',
    type=int
)
parser.add_argument(
    '-l',
    '--load_model',
    help='Load a previously trained model?',
    action='store_true'
)
parser.add_argument(
    '-d',
    '--display',
    help='Display viedo output of game?',
    action='store_true'
)
args = parser.parse_args()


class SpaceInvaders():
    def __init__(self, dl_type, display=False, load_model=False):
        self.name = 'space_invaders'
        self.ale = ALEInterface()
        self.ale.setInt(str.encode('random_seed'), np.random.randint(100))
        self.ale.setBool(str.encode('display_screen'), display)
        self.ale.loadROM(str.encode('./roms/space_invaders.bin'))

        self.legal_actions = list(self.ale.getMinimalActionSet())

        # Squeeze because getScreenGrayScale returns a shape of (210, 160, 1)
        self.frame_shape = np.squeeze(self.ale.getScreenGrayscale()).shape
        self.network_input_shape = self.frame_shape + (3,)  # (210, 160, 3)
        self.frame_buffer = deque(maxlen=3)  # Holds the 3 most recent frames
        self.replay_memory = ReplayMemory(2000, 32)
        self.preprocessor = Preprocessor(self.frame_shape)
        if dl_type == 'dqn':
            self.agent = DeepQN(
                self.network_input_shape,
                self.legal_actions,
                self.replay_memory,
                self.name,
                load=load_model
            )
        else:
            self.agent = DoubleDQN(
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
        print('Agent: ', dl_type)

    def train(self, max_frames=100000):
        total_reward = 0
        frame_counter = 0
        alive_counter = 0

        # Initialize frame buffer
        self.frame_buffer.append(np.squeeze(self.ale.getScreenGrayscale()))
        self.frame_buffer.append(np.squeeze(self.ale.getScreenGrayscale()))
        self.frame_buffer.append(np.squeeze(self.ale.getScreenGrayscale()))

        try:
            while frame_counter + 3 < max_frames:
                gameover = False
                initial_state = self.preprocessor.stack_frames(
                    self.frame_buffer
                )

                action = self.agent.predict(initial_state)
                self.frame_buffer.clear()

                # Play for 3 frames and stack'em up
                for _ in range(3):
                    # Save model every 500 frames
                    if frame_counter % 500 == 0:
                        self.agent.save_model()

                    # If using a target model, update it's weights
                    # from q model's every 500 frames
                    if hasattr(self.agent, 'tau') and frame_counter % self.agent.tau == 0:
                        self.agent.update_targetmodel()

                    reward = self.ale.act(action)
                    self.frame_buffer.append(
                        np.squeeze(self.ale.getScreenGrayscale()))

                    print('Frame: ', frame_counter)
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

                self.agent.replay_train()

        except Exception as e:
            exc_type, exc_val, exc_trace = sys.exc_info()
            traceback.print_exception(exc_type, exc_val, exc_trace)
            self.agent.save_model()
            self.ale.reset_game()
            print('Exception on frame: ', frame_counter)

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
            action = self.agent.predict(state)
            total_reward += self.ale.act(action)

            # Pushes oldest frame out
            self.frame_buffer.append(np.squeeze(self.ale.getScreenGrayscale()))
            if self.ale.game_over():
                done = True

        frames_survived = self.ale.getEpisodeFrameNumber()
        print('Game Over')
        print('Frames Survived: ', frames_survived)
        print('Score: ', total_reward)
        self.ale.reset_game()
        return total_reward, frames_survived

    def simulate_random(self):
        done = False
        total_reward = 0
        print('Simulating game randomly')

        while not done:
            action = np.random.choice(self.ale.getMinimalActionSet())
            total_reward += self.ale.act(action)

            if self.ale.game_over():
                done = True

        frames_survived = self.ale.getEpisodeFrameNumber()
        print('Game Over')
        print('Frames Survived: ', frames_survived)
        print('Score: ', total_reward)
        self.ale.reset_game()
        return total_reward, frames_survived

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
            else:
                xdata_loaded = xdata
                ydata_loaded = ydata

        np.save(pathx, xdata_loaded)
        np.save(pathy, ydata_loaded)
        print('Saved {}, {} at {}'.format(pathx, pathy, datetime.now()))

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
    game = SpaceInvaders(
        args.type,
        display=args.display,
        load_model=args.load_model
    )

    interval = args.interval_frames
    max_frames = args.max_frames
    games_to_play = 10

    for i in range(max_frames // interval):
        game.train(interval)
        running_score = 0
        frames_survived = 0
        for _ in range(games_to_play):
            game_scores = game.simulate_intelligent()
            running_score += game_scores[0]
            frames_survived += game_scores[1]

        running_score /= games_to_play
        frames_survived /= games_to_play

        # Save the Average Score and Frames survived over 10 games for this interval
        game.save_results(
            './data/{}_avgscorex_{}.npy'.format(game.name, args.type),
            './data/{}_avgscorey_{}.npy'.format(game.name, args.type),
            (i + 1) * interval,
            running_score
        )

        game.save_results(
            './data/{}_avgframes_survx_{}.npy'.format(game.name, args.type),
            './data/{}_avgframes_survy_{}.npy'.format(game.name, args.type),
            (i + 1) * interval,
            frames_survived
        )

    # Silence weird TF exception https://github.com/tensorflow/tensorflow/issues/3388
    K.clear_session()
