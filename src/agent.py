from traceback import print_exc
from collections import deque
from ale_python_interface import ALEInterface
import numpy as np
from replaymemory import ReplayMemory
from dqn import DeepQN
from double_dqn import DoubleDQN
from dueling_dqn import DuelingDQN


class Agent():
    def __init__(self, game, agent_type, display=False, load_model=False, record=False):
        self.name = game
        self.ale = ALEInterface()
        self.ale.setInt(str.encode('random_seed'), np.random.randint(100))
        self.ale.setBool(str.encode('display_screen'), display or record)
        if record:
            self.ale.setString(str.encode('record_screen_dir'), str.encode('./data/recordings/{}/{}/tmp/'.format(game, agent_type)))

        self.ale.loadROM(str.encode('./roms/{}.bin'.format(self.name)))
        self.action_list = list(self.ale.getMinimalActionSet())
        self.frame_shape = np.squeeze(self.ale.getScreenGrayscale()).shape
        self.frame_buffer = deque(maxlen=3)
        self.replay_memory = ReplayMemory(500000, 32)

        model_input_shape = self.frame_shape + (3,)
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
        total_reward = 0
        alive_counter = 0

        # Initialize frame buffer
        self.frame_buffer.append(np.squeeze(self.ale.getScreenGrayscale()))
        self.frame_buffer.append(np.squeeze(self.ale.getScreenGrayscale()))
        self.frame_buffer.append(np.squeeze(self.ale.getScreenGrayscale()))

        try:
            for step in range(steps):
                gameover = False
                intial_state = np.stack(self.frame_buffer, axis=-1)
                action = self.model.predict_action(intial_state)
                if step % 5000 == 0:
                    self.model.save_model()

                if hasattr(self.model, 'tau') and step % self.model.tau == 0:
                    self.model.update_target_model()

                # Play for 3 frames and stack 'em up
                for _ in range(3):
                    reward = self.ale.act(action)
                    self.frame_buffer.append(np.squeeze(self.ale.getScreenGrayscale()))
                    total_reward += reward
                    alive_counter += 1

                if self.ale.game_over():
                    gameover = True
                    total_reward = 0
                    alive_counter = 0
                    self.ale.reset_game()

                new_state = np.stack(self.frame_buffer, axis=-1)
                self.replay_memory.add(
                    intial_state,
                    action,
                    reward,
                    gameover,
                    new_state
                )

                loss += self.model.replay_train()

        except OSError:
            raise KeyboardInterrupt
        except:
            print_exc()
            self.model.save_model()
            raise KeyboardInterrupt

        return np.mean(loss, axis=0)

    def simulate_random(self):
        print('Simulating game randomly')
        done = False
        total_reward = 0
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

    def simulate_intelligent(self, evaluating=False):
        print('Simulating game intelligently')
        done = False
        total_reward = 0

        self.frame_buffer.append(np.squeeze(self.ale.getScreenGrayscale()))
        self.frame_buffer.append(np.squeeze(self.ale.getScreenGrayscale()))
        self.frame_buffer.append(np.squeeze(self.ale.getScreenGrayscale()))
        while not done:
            state = np.stack(self.frame_buffer, axis=-1)
            action = self.model.predict_action(state, evaluating)
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
