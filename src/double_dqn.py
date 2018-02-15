import sys
from datetime import datetime
from keras.models import load_model
import numpy as np
from nn_base import NN
from replaymemory import ReplayMemory


class DoubleDQN(NN):
    def __init__(self, input_shape, output_shape, replay_memory: ReplayMemory, game_name, load=False):
        super().__init__(input_shape, output_shape, replay_memory, game_name)
        self.tau = 500  # Number of frames before updating the target network weights

        if load:
            self.qmodel, self.targetmodel = self.load_model()
        else:
            self.qmodel = self.build_qmodel()
            self.targetmodel = self.build_qmodel()

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
        minibatch = self.replay_memory.sample()
        for state, action, reward, done, next_state in minibatch:
            target = reward
            if not done:
                # Use the qmodel to select an action and the target model to
                # evaluate the Q values
                future_best_action_index = np.argmax(self.qmodel.predict(
                    np.expand_dims(next_state, 0), batch_size=1
                ))
                future_action_qvals = self.targetmodel.predict(
                    np.expand_dims(next_state, 0), batch_size=1
                )[0]

                target = reward + self.gamma * \
                    future_action_qvals[future_best_action_index]

            ypred = self.qmodel.predict(np.expand_dims(state, 0))
            # In the future if we do <action> the Q value will now be influenced
            # by the max Q value of the next states best action (discounted future reward)
            ypred[0][self.output_shape.index(action)] = target
            loss = self.qmodel.fit(
                np.expand_dims(state, 0),
                ypred,
                verbose=0
            ).history['loss']

        if self.epsilon > self.epsilon_floor:
            self.epsilon *= self.epsilon_decay_rate

        return loss

    def update_targetmodel(self):
        self.targetmodel.set_weights(self.qmodel.get_weights())

    def save_model(self):
        self.qmodel.save('./data/{}_qmodel.h5'.format(self.game_name))
        self.targetmodel.save(
            './data/{}_targetmodel.h5'.format(self.game_name)
        )

        print('Saved models at ', datetime.now())

    def load_model(self):
        try:
            qmodel = load_model(
                './data/{}_qmodel.h5'.format(self.game_name)
            )
            tmodel = load_model(
                './data/{}_targetmodel.h5'.format(self.game_name)
            )

            return qmodel, tmodel
        except OSError:
            print('Failed to load models for DDQN')
            sys.exit()
