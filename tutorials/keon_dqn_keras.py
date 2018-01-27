from collections import deque
from statistics import mean
from time import time
import random
import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import TensorBoard


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.replay_size = 128
        self.scores = []
        self.model = self.build_model()
        # self.tb = TensorBoard(log_dir='./logs/{}'.format(time()))
        print('Agent')
        print('State Size:', self.state_size)
        print('Action Size:', self.action_size)

    def build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        print('Action:', act_values)
        # Returns index of action with highest reward.
        # Have to index into act_values as predict returns a 2-dimension array
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        try:
            minibatch = random.sample(self.memory, batch_size)
        except ValueError:
            minibatch = random.sample(self.memory, len(self.memory))

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                # Again, remember predict returns a 2-dimension array
                # np.amax gives us the maximum future rewarded action
                target = reward + self.gamma * \
                    np.amax(self.model.predict(next_state)[0])

            # Predict a set of rewards for the given state
            target_f = self.model.predict(state)
            target_f[0][action] = target
            # self.model.fit(state, target_f, verbose=0, callbacks=[self.tb])
            self.model.fit(state, target_f, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def main():
    episodes = 500
    frames = 5000
    env = gym.make('Pong-v0')
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    # Iterate the game
    for episode in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state.shape[0]])
        for frame in range(frames):
            # env.render()

            # Decide action
            action = agent.act(state)
            # Advance the game to the next frame based on the action.
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, next_state.shape[0]])

            # Remember the previous state, action, reward, and done
            agent.remember(state, action, reward, next_state, done)

            # make next_state the new current state for the next frame.
            state = next_state
            if done:
                if episode % 1 == 0:
                    print("E: {}/{}, S: {}".format(episode, episodes, frame))

                # train the agent with the experience of the episode
                agent.replay(agent.replay_size)

                agent.scores.append(frame)
                break

    print('Episodes: {}\nAvg Score: {}\nMemory Replay Size: {}'.format(
        episodes, mean(agent.scores), agent.replay_size))


if __name__ == "__main__":
    main()
