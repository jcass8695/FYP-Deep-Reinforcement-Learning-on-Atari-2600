from abc import ABC, abstractmethod
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten
from keras.optimizers import Adam
from replaymemory import ReplayMemory


class NN(ABC):
    def __init__(self, input_shape, output_shape, replay_memory: ReplayMemory, game_name):
        # Model Parameters
        self.epsilon = 0.1  # Exploration rate
        self.epsilon_floor = 0.05  # Min Exploration rate
        self.epsilon_decay_rate = 0.99
        self.gamma = 0.95  # Discount rate
        self.learning_rate = 0.00001
        self.minibatch_size = 64

        self.replay_memory = replay_memory
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.game_name = game_name  # The name of the game to which this model is attached

    def build_qmodel(self):
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
        model.add(Dense(len(self.output_shape), activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    @abstractmethod
    def predict(self, state):
        pass

    @abstractmethod
    def replay_train(self):
        pass

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def load_model(self):
        pass
