import sys
from keras.models import load_model, Sequential
from keras.layers import Conv2D, Dense, Flatten
from keras.optimizers import Adam
import numpy as np
from dqn import DeepQN
from replaymemory import ReplayMemory


class DoubleDeepQN(DeepQN):
    def __init__(self, input_shape, valid_action_set, replay_memory: ReplayMemory, load=False):
        super().__init__(self, input_shape, valid_action_set,
                         replay_memory, load)

        if load:
            self.target_model = self.load_network('./target_model.h5')
        else:
            self.target_model = self.build_target_network()

    def build_target_network(self):
        # Creates a target network as described in DeepMind paper
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
        model.compile(loss='mse', optimizer=Adam(self.learning_rate))
        return model

    def replay_training(self):
        minibatch = self.memory.sample()
        for state, action, reward, done, next_state in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(
                    self.q_model.predict(np.expand_dims(next_state, 0), batch_size=1)))

            target_f = self.q_model.predict(np.expand_dims(state, 0))
            target_f[0][self.valid_action_set.index(action)] = target

            self.q_model.fit(np.expand_dims(state, 0),
                             target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_floor:
            self.epsilon *= self.epsilon_decay_rate
