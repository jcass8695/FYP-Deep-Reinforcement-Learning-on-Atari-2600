from os import getcwd
from statistics import median, mean
from collections import Counter
import random
import gym
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

LR = 0.001
env = gym.make('CartPole-v0')
env.reset()
goal_steps = 500
score_requirement = 50
initial_games = 10000
num_episodes = 10
max_steps = 200


def get_training_data():
    # [(observation, action), ...]
    training_data = []

    # All scores from every frame
    scores = []

    # All scores that were above the threshold
    accepted_scores = []

    for _ in range(initial_games):
        score = 0
        # Moves taken in this episode
        game_memory = []
        prev_observation = []
        env.reset()
        for _ in range(max_steps):
            action = env.action_space.sample()
            observation, reward, done, _ = env.step(action)

            # Observation is returned as a result of taking the current action
            # So we pair the previous observation with the current action
            if len(prev_observation) > 0:
                game_memory.append((prev_observation, action))

            prev_observation = observation
            score += reward
            if done:
                break

        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                # Convert action to one-hot encoding
                if data[1] == 1:
                    one_hot_action = [0, 1]
                else:
                    one_hot_action = [1, 0]

                training_data.append((data[0], one_hot_action))

        scores.append(score)
    np.save('training_data.npy', np.array(training_data))

    # Stats
    print('Avg Accepted Score:', mean(accepted_scores))
    print('Median of Accepted Scores:', median(accepted_scores))
    print(Counter(accepted_scores))
    return training_data


def neural_network_model(input_size):
    network = input_data(shape=[None, input_size, 1], name="input")

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 2, activation='softmax')
    network = regression(
        network,
        optimizer='adam',
        learning_rate=LR,
        loss='categorical_crossentropy',
        name='targets'
    )

    model = tflearn.DNN(network)
    return model


def train_model(training_data, model=False):
    X = np.array([i[0] for i in training_data]).reshape(
        -1,
        len(training_data[0][0]),
        1
    )

    Y = [i[1] for i in training_data]
    if model is False:
        model = neural_network_model(input_size=len(X[0]))

    model.fit(
        {'input': X},
        {'targets': Y},
        n_epoch=5,
        snapshot_step=500,
        show_metric=True
    )

    return model


def predict_moves(model):
    scores = []
    choices = []
    for _ in range(num_episodes):
        score = 0
        prev_observation = []
        game_memory = []
        prev_observation = []
        env.reset()
        for step in range(max_steps):
            env.render()
            if len(prev_observation) == 0:
                action = env.action_space.sample()
            else:
                action = np.argmax(model.predict(
                    np.array(prev_observation).reshape(
                        -1,
                        len(prev_observation),
                        1
                    )
                )[0])

            choices.append(action)
            new_observation, reward, done, info = env.step(action)
            prev_observation = new_observation
            game_memory.append((new_observation, action))
            score += reward
            # print('E:{} F:{} S:{}'.format(episode, step, reward))
            if done:
                break

        scores.append(score)

    print('Avg Score:', mean(scores))
    print('Choice 1\'s:{} Choice 0\'s:{}'.format(
        choices.count(1), choices.count(0))
    )


td = get_training_data()
predict_moves(train_model(td))
