from abc import ABC, abstractmethod
from datetime import datetime
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten
from keras.optimizers import Adam
from replaymemory import ReplayMemory


class NN(ABC):
    def __init__(self, input_shape, output_shape, replay_memory: ReplayMemory, game_name):
        # Model Parameters
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_floor = 0.1  # Min Exploration rate
        self.epsilon_decay_rate = (self.epsilon - self.epsilon_floor) / 1000000
        self.gamma = 0.99  # Discount rate
        self.learning_rate = 0.00025
        self.minibatch_size = 32

        self.replay_memory = replay_memory
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.game_name = game_name  # The name of the game to which this model is attached

    def build_qmodel(self):
        '''
        Builds the CNN (from Human-Level control paper - Model Architecture),
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
        model.add(Dense(self.output_shape, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    @abstractmethod
    def predict_action(self, state, evaluating=False):
        pass

    @abstractmethod
    def replay_train(self):
        pass

    @abstractmethod
    def save_model(self):
        print('Saved models at ', datetime.now())

    @abstractmethod
    def load_model(self):
        print('Model loaded at', datetime.now())

    @abstractmethod
    def save_hyperparams(self):
        print('Saved Hyperparameters at ', datetime.now())

    @abstractmethod
    def load_hyperparams(self):
        print('Hyperparameters loaded at', datetime.now())
