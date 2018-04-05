from traceback import print_exc
from collections import deque
import bz2
import pickle
from ale_python_interface import ALEInterface
import numpy as np
from replaymemory import ReplayMemory
from dqn import DeepQN
from double_dqn import DoubleDQN
from dueling_dqn import DuelingDQN


class Agent():
    def __init__(self, game, agent_type, display, load_model, record, test):
        self.name = game
        self.agent_type = agent_type
        self.ale = ALEInterface()
        self.ale.setInt(str.encode('random_seed'), np.random.randint(100))
        self.ale.setBool(str.encode('display_screen'), display or record)
        if record:
            self.ale.setString(str.encode('record_screen_dir'), str.encode('./data/recordings/{}/{}/tmp/'.format(game, agent_type)))

        self.ale.loadROM(str.encode('./roms/{}.bin'.format(self.name)))
        self.action_list = list(self.ale.getMinimalActionSet())
        self.frame_shape = np.squeeze(self.ale.getScreenGrayscale()).shape
        if test:
            self.name += '_test'

        if 'space_invaders' in self.name:
            # Account for blinking bullets
            self.frameskip = 2
        else:
            self.frameskip = 3

        self.frame_buffer = deque(maxlen=self.frameskip + 1)
        if load_model:
            self.load_replaymemory()
        else:
            self.replay_memory = ReplayMemory(100000, 32)

        model_input_shape = self.frame_shape + (self.frameskip+1,)
        model_output_shape = len(self.action_list)

        if agent_type == 'dqn':
            self.model = DeepQN(
                model_input_shape,
                model_output_shape,
                self.action_list,
                self.replay_memory,
                self.name,
                load_model
            )
        elif agent_type == 'double':
            self.model = DoubleDQN(
                model_input_shape,
                model_output_shape,
                self.action_list,
                self.replay_memory,
                self.name,
                load_model
            )

        else:
            self.model = DuelingDQN(
                model_input_shape,
                model_output_shape,
                self.action_list,
                self.replay_memory,
                self.name,
                load_model
            )

        print('{} Loaded!'.format(' '.join(self.name.split('_')).title()))
        print('Displaying: ', display)
        print('Frame Shape: ', self.frame_shape)
        print('Frame Skip: ', self.frameskip)
        print('Action Set: ', self.action_list)
        print('Model Input Shape: ', model_input_shape)
        print('Model Output Shape: ', model_output_shape)
        print('Agent: ', agent_type)

    def training(self, steps):
        '''
        Trains the agent for :training_interval_frames.

        Returns the average model loss
        '''

        loss = []

        # Initialize frame buffer
        self.frame_buffer.append(np.squeeze(self.ale.getScreenGrayscale()))
        self.frame_buffer.append(np.squeeze(self.ale.getScreenGrayscale()))
        self.frame_buffer.append(np.squeeze(self.ale.getScreenGrayscale()))
        self.frame_buffer.append(np.squeeze(self.ale.getScreenGrayscale()))

        try:
            for step in range(steps):
                gameover = False
                initial_state = np.stack(self.frame_buffer, axis=-1)
                action = self.model.predict_action(initial_state)

                # Backup data
                if step % 5000 == 0:
                    self.model.save_model()
                    self.model.save_hyperparams()

                # If using a target model check for weight updates
                if hasattr(self.model, 'tau'):
                    if self.model.tau == 0:
                        self.model.update_target_model()
                        self.model.tau = 5000
                    else:
                        self.model.tau -= 1

                # Frame skipping technique https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/
                lives_before = self.ale.lives()
                for _ in range(self.frameskip):
                    self.ale.act(action)

                reward = self.ale.act(action)
                self.frame_buffer.append(np.squeeze(self.ale.getScreenGrayscale()))

                lives_after = self.ale.lives()
                if lives_after < lives_before:
                    gameover = True  # Taking advice from dude on reddit
                    reward = -1

                if self.ale.game_over():
                    gameover = True
                    reward = -1
                    self.ale.reset_game()

                new_state = np.stack(self.frame_buffer, axis=-1)

                # Experiment with clipping rewards for stability purposes
                reward = np.clip(reward, -1, 1)
                self.replay_memory.add(
                    initial_state,
                    action,
                    reward,
                    gameover,
                    new_state
                )

                loss += self.model.replay_train()
        except:
            print_exc()
            self.model.save_model()
            self.model.save_hyperparams()
            self.save_replaymemory()
            raise KeyboardInterrupt

        return np.mean(loss, axis=0)

    def simulate_random(self):
        print('Simulating game randomly')
        done = False
        total_reward = 0
        while not done:
            action = np.random.choice(self.ale.getMinimalActionSet())
            reward = self.ale.act(action)
            total_reward += reward
            if self.ale.game_over():
                reward = -1
                done = True

            reward = np.clip(reward, -1, 1)
            if reward != 0:
                print(reward)

        frames_survived = self.ale.getEpisodeFrameNumber()
        print('Game Over')
        print('Frames Survived: ', frames_survived)
        print('Score: ', total_reward)
        self.ale.reset_game()
        return total_reward, frames_survived

    def simulate_intelligent(self, evaluating=False):
        print('Simulating game intelligently')
        done = False
        total_score = 0

        self.frame_buffer.append(np.squeeze(self.ale.getScreenGrayscale()))
        self.frame_buffer.append(np.squeeze(self.ale.getScreenGrayscale()))
        self.frame_buffer.append(np.squeeze(self.ale.getScreenGrayscale()))
        self.frame_buffer.append(np.squeeze(self.ale.getScreenGrayscale()))
        while not done:
            state = np.stack(self.frame_buffer, axis=-1)
            action = self.model.predict_action(state, evaluating)

            for _ in range(self.frameskip):
                self.ale.act(action)

            # Remember, ale.act returns the increase in game score with this action
            total_score += self.ale.act(action)

            # Pushes oldest frame out
            self.frame_buffer.append(np.squeeze(self.ale.getScreenGrayscale()))
            if self.ale.game_over():
                done = True

        frames_survived = self.ale.getEpisodeFrameNumber()
        print('Game Over')
        print('Frames Survived: ', frames_survived)
        print('Score: ', total_score)
        self.ale.reset_game()
        return total_score, frames_survived

    def save_replaymemory(self):
        with bz2.BZ2File('./data/{}/{}_replaymem.obj'.format(self.agent_type, self.name), 'wb') as f:
            pickle.dump(self.replay_memory, f, protocol=pickle.HIGHEST_PROTOCOL)
            print('Saved replay memory')

    def load_replaymemory(self):
        try:
            with bz2.BZ2File('./data/{}/{}_replaymem.obj'.format(self.agent_type, self.name), 'rb') as f:
                self.replay_memory = pickle.load(f)
                print('Loaded replay memory')
        except FileNotFoundError:
            print('No replay memory file found')
            raise KeyboardInterrupt
