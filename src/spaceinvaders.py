import os
from collections import deque
from ale_python_interface import ALEInterface
import numpy as np
import matplotlib.pyplot as plt
from dqn import DeepQN
from preprocessor import Preprocessor
from replaymemory import ReplayMemory

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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
                if frame_counter % (max_frames * 0.1) == 0:
                    print('Checkpoint on frame: ', frame_counter)
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

    def save_results(self, pathx, pathy, frames_trained_for, scores):
        '''
        Saves the scores of individual games and the number of frames that the model
        was trained for as numpy arrays. Intended to be used to visualize performance
        in final report

        pathx: str path to saved numpy array of saved frames_trained_for values
        pathy: same but for saved scores
        frames_trained_for: number of frames the model had been trained for to produce the given scores
        scores: list of scores obtained from a simulation
        '''
        x = [frames_trained_for for _ in range(len(scores))]
        try:
            xdata = np.load(pathx)
            ydata = np.load(pathy)
            xdata = np.concatenate([xdata, x])
            ydata = np.concatenate([ydata, scores])
        except FileNotFoundError:
            if '.npy' not in pathx or '.npy' not in pathy:
                print('Paths must include the .npy extension')
                return

            xdata = x
            ydata = scores

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
                print('No data to plot')

            return

        plt.plot(xdata, ydata, 'ro')
        plt.show()


if __name__ == '__main__':
    game = SpaceInvaders(display=False, load_model=True)
    # game.train()
    # game.simulate_random()
    collected_scores = []
    for _ in range(8):
        collected_scores.append(game.simulate_intelligent())

    game.save_results('dqn_xdata.npy', 'dqn_ydata.npy', 2000, collected_scores)
    game.plot_results('dqn_xdata.npy', 'dqn_ydata.npy')
