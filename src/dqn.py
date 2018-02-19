from datetime import datetime
import numpy as np
from keras.models import load_model
from nn_base import NN
from replaymemory import ReplayMemory


class DeepQN(NN):
    def __init__(self, input_shape, output_shape, action_list, replay_memory: ReplayMemory, game_name, load=False):
        super().__init__(input_shape, output_shape, replay_memory, game_name)
        if load:
            self.qmodel = self.load_model()
        else:
            self.qmodel = self.build_qmodel()

        self.action_list = action_list

    def predict_action(self, state, evaluating=False):
        qvalues = self.qmodel.predict(np.expand_dims(state, 0), batch_size=1)

        # Translate the index of the highest Q value action to an action
        # in the games set. This is required as the games action set may look
        # like this: [0, 1, 2, 3, 6, 17, 18]
        optimal_policy = np.argmax(qvalues)
        optimal_action = self.action_list[optimal_policy]
        if not evaluating and np.random.random() < self.epsilon:
            optimal_action = np.random.choice(self.action_list)
        elif evaluating and np.random.random() < 0.05:
            optimal_action = np.random.choice(self.action_list)

        return optimal_action

    def replay_train(self):
        ''' Playing Atari with Deep Reinforcement Learning (DeepMind, 2013), Algorithm 1 '''

        loss = []
        minibatch = self.replay_memory.sample()
        for state, action, reward, done, next_state in minibatch:
            target = reward
            if not done:
                # Future discounted reward = R + gamma * Q(ns, a)
                target = (reward + self.gamma * np.amax(self.qmodel.predict(np.expand_dims(next_state, 0), batch_size=1)))

            ypred = self.qmodel.predict(np.expand_dims(state, 0))
            ypred[0][self.action_list.index(action)] = target
            loss.append(self.qmodel.fit(np.expand_dims(state, 0), ypred, verbose=0).history['loss'][0])
        if self.epsilon > self.epsilon_floor:
            self.epsilon *= self.epsilon_decay_rate

        return loss

    def save_model(self):
        self.qmodel.save('./data/dqn/{}_qmodel_dqn.h5'.format(self.game_name))
        print('Saved model at ', datetime.now())

    def load_model(self):
        try:
            return load_model('./data/dqn/{}_qmodel_dqn.h5'.format(self.game_name))
        except OSError:
            print('Failed to load model for DQN')
            raise KeyboardInterrupt
