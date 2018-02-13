import sys
from datetime import datetime
from nn_base import NN
from keras.models import load_model
import numpy as np
from replaymemory import ReplayMemory


class DeepQN(NN):
    def __init__(self, input_shape, output_shape, replay_memory: ReplayMemory, game_name, load=False):
        super().__init__(input_shape, output_shape, replay_memory, game_name, load)
        if load:
            self.qmodel = self.load_model()
        else:
            self.qmodel = self.build_qmodel()

    def predict(self, state):
        qvalues = self.qmodel.predict(
            np.expand_dims(state, 0),  # Add extra axis
            batch_size=1
        )

        # Translate the index of the highest Q value action to an action
        # in the games set. This is required as the games action set may look
        # like this: [0, 1, 2, 3, 6, 17, 18]
        optimal_policy = np.argmax(qvalues)
        optimal_action = self.output_shape[optimal_policy]
        if np.random.random() < self.epsilon:
            optimal_action = np.random.choice(self.output_shape)

        return optimal_action

    def replay_train(self):
        ''' Playing Atari with Deep Reinforcement Learning (DeepMind, 2013), Algorithm 1 '''

        minibatch = self.replay_memory.sample()
        for state, action, reward, done, next_state in minibatch:
            target = reward
            if not done:
                # Future discounted reward = R + gamma * Q(ns, a)
                target = (reward + self.gamma * np.amax(
                    self.qmodel.predict(np.expand_dims(next_state, 0), batch_size=1)))

            ypred = self.qmodel.predict(np.expand_dims(state, 0))
            ypred[0][self.output_shape.index(action)] = target

            self.qmodel.fit(
                np.expand_dims(state, 0),
                ypred,
                verbose=0
            )

        if self.epsilon > self.epsilon_floor:
            self.epsilon *= self.epsilon_decay_rate

    def save_model(self):
        self.qmodel.save('./data/{}_qmodel.h5'.format(self.game_name))
        print('Saved model at ', datetime.now())

    def load_model(self):
        try:
            return load_model('./data/{}_qmodel.h5'.format(self.game_name))
        except ValueError:
            print('Failed to load model for DQN')
            sys.exit()
