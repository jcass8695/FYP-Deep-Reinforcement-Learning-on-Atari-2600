import sys
from datetime import datetime
from keras.models import load_model, Sequential
from keras.layers import Conv2D, Dense, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import numpy as np
from replaymemory import ReplayMemory


class DeepQN():
    def __init__(self, input_shape, valid_action_set, replay_memory: ReplayMemory, load=False):
        # Parameters
        self.epsilon = 0.1  # Exploration rate
        self.epsilon_floor = 0.05
        self.epsilon_decay_rate = 0.99
        self.gamma = 0.95  # Discount rate
        self.learning_rate = 0.00001

        self.minibatch_size = 64
        self.memory = replay_memory
        self.input_shape = input_shape
        self.valid_action_set = valid_action_set
        self.tb_callback = TensorBoard(log_dir='../tblogs', write_images=True)

        if load:
            self.q_model = self.load_network('./model.h5')
        else:
            self.q_model = self.build_q_network()

    def build_q_network(self):
        '''
        Builds the CNN (from https://github.com/yilundu/DQN-DDQN-on-Space-Invaders/blob/master/deep_Q.py),
        that takes preprocessed frames as input and outputs
        Q-Values for each available action in the game.

        Frames have already been converted to greyscale for
        computational efficiency as color does not effect gameplay
        '''

        model = Sequential()
        model.add(Conv2D(
            32,
            (8, 8),
            input_shape=self.input_shape,
            strides=(4, 4),
            activation='relu'
        ))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(len(self.valid_action_set), activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def predict_move(self, state):
        q_values = self.q_model.predict(
            np.expand_dims(state, 0),  # Add extra axis
            batch_size=1
        )

        # Translate the index of the highest Q value action to an action
        # in the games set. This is required as the games action set may look
        # like this: [0, 1, 2, 3, 6, 17, 18]
        optimal_policy = np.argmax(q_values)
        optimal_action = self.valid_action_set[optimal_policy]
        if np.random.random() < self.epsilon:
            optimal_action = np.random.choice(self.valid_action_set)

        return optimal_action

    def replay_training(self):
        minibatch = self.memory.sample()
        for state, action, reward, done, next_state in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(
                    self.q_model.predict(np.expand_dims(next_state, 0), batch_size=1)))

            target_f = self.q_model.predict(np.expand_dims(state, 0))
            target_f[0][self.valid_action_set.index(action)] = target

            self.q_model.fit(
                np.expand_dims(state, 0),
                target_f,
                verbose=0,
                callbacks=[self.tb_callback]
            )

        if self.epsilon > self.epsilon_floor:
            self.epsilon *= self.epsilon_decay_rate

    def save_network(self, path):
        self.q_model.save(path)
        print("Successfully saved network at ", datetime.now())

    def load_network(self, path):
        try:
            print("Succesfully loaded network.")
            return load_model(path)
        except ValueError:
            print('Model does not exist at that path')
            print('Exiting program...')
            sys.exit()
