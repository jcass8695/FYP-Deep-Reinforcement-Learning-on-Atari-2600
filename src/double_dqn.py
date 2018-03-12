from datetime import datetime
from keras.models import load_model
import numpy as np
from nn_base import NN
from replaymemory import ReplayMemory


class DoubleDQN(NN):
    def __init__(self, input_shape, output_shape, action_list, replay_memory: ReplayMemory, game_name, load=False):
        super().__init__(input_shape, output_shape, replay_memory, game_name)
        self.tau = 5000  # Number of training steps before updating the target network weights

        if load:
            self.q_model, self.target_model = self.load_model()
        else:
            self.q_model = self.build_qmodel()
            self.target_model = self.build_qmodel()

        self.action_list = action_list

    def predict_action(self, state, evaluating=False):
        qvalues = self.q_model.predict(np.expand_dims(state, 0), batch_size=1)

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
        loss = []
        minibatch = self.replay_memory.sample()
        for state, action, reward, done, next_state in minibatch:
            target = reward
            if not done:
                # Use the qmodel to select an action and the target model to
                # evaluate the Q values
                future_best_action_index = np.argmax(self.q_model.predict(
                    np.expand_dims(next_state, 0), batch_size=1
                ))
                future_action_qvals = self.target_model.predict(
                    np.expand_dims(next_state, 0), batch_size=1
                )[0]

                target = reward + self.gamma * future_action_qvals[future_best_action_index]

            ypred = self.q_model.predict(np.expand_dims(state, 0))

            # In the future if we do <action> the Q value will now be influenced
            # by the max Q value of the next states best action (discounted future reward)
            ypred[0][self.action_list.index(action)] = target
            loss.append(self.q_model.fit(np.expand_dims(state, 0), ypred, verbose=0).history['loss'][0])

            if self.epsilon > self.epsilon_floor:
                self.epsilon -= self.epsilon_decay_rate

        return loss

    def update_target_model(self):
        self.target_model.set_weights(self.q_model.get_weights())

    def save_model(self):
        self.q_model.save('./data/double/{}_qmodel_double.h5'.format(self.game_name))
        self.target_model.save('./data/double/{}_targetmodel_double.h5'.format(self.game_name))
        super().save_model()

    def load_model(self):
        try:
            qmodel = load_model('./data/double/{}_qmodel_double.h5'.format(self.game_name))
            tmodel = load_model('./data/double/{}_targetmodel_double.h5'.format(self.game_name))
            super().load_model()
            return qmodel, tmodel
        except OSError:
            print('Failed to load models for DoubleDQN')
            raise KeyboardInterrupt
