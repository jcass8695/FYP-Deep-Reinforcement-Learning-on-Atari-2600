from datetime import datetime
from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, Flatten, merge, Input
from keras.optimizers import Adam
from keras.models import load_model
import keras.backend as K
import numpy as np
from nn_base import NN
from replaymemory import ReplayMemory


class DuelingDQN(NN):
    def __init__(self, input_shape, output_shape, action_list, replay_memory: ReplayMemory, game_name, load=False):
        super().__init__(input_shape, output_shape, replay_memory, game_name)
        self.tau = 5000  # Number of training steps before updating the target network weights

        if load:
            self.q_model, self.target_model = self.load_model()
        else:
            self.q_model = self.build_qmodel()
            self.target_model = self.build_qmodel()

        self.action_list = action_list

    def build_qmodel(self):
        input_layer = Input(shape=self.input_shape)
        layers = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(input_layer)
        layers = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(layers)
        layers = Conv2D(64, (3, 3), activation='relu')(layers)
        layers = Flatten()(layers)
        q_branch = Dense(512)(layers)
        q_branch = Dense(1)(layers)
        advantage_branch = Dense(512)(layers)
        advantage_branch = Dense(self.output_shape)(layers)
        branch_exit = merge([advantage_branch, q_branch], mode=lambda x: x[0]+x[1] - K.mean(x[0]), output_shape=[self.output_shape])
        model = Model(inputs=[input_layer], outputs=[branch_exit])
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

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
            loss.append(self.q_model.fit(
                np.expand_dims(state, 0),
                ypred,
                verbose=0
            ).history['loss'][0])

        if self.epsilon > self.epsilon_floor:
            self.epsilon *= self.epsilon_decay_rate

        return loss

    def update_target_model(self):
        self.target_model.set_weights(self.q_model.get_weights())

    def save_model(self):
        self.q_model.save('./data/duel/{}_qmodel_duel.h5'.format(self.game_name))
        self.target_model.save('./data/duel/{}_targetmodel_duel.h5'.format(self.game_name))
        print('Saved models at ', datetime.now())

    def load_model(self):
        try:
            qmodel = load_model('./data/duel/{}_qmodel_duel.h5'.format(self.game_name))
            tmodel = load_model('./data/duel/{}_targetmodel_duel.h5'.format(self.game_name))
            return qmodel, tmodel
        except OSError:
            print('Failed to load models for DuelingDQN')
            raise KeyboardInterrupt
