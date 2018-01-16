import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter

LR = 0.001
env = gym.make('CartPole-v0')
env.reset()
goal_steps = 500
score_requirement = 50
initial_games = 10000
num_episodes = 50
max_steps = 200


def random_inputs():
    # 1 Episode = 1 Game
    for episode in range(num_episodes):
        print('E:', episode)
        avg_reward = 0
        env.reset()

        # Each episode will run for 200 frames and then we stop
        for frame in range(max_steps):
            # Draws the game
            env.render()

            # Assigns a random 0/1 (Left/Right) action
            action = env.action_space.sample()

            # Takes the random action and returns these 4 parameters
            observation, reward, done, info = env.step(action)
            avg_reward += reward

            # If we die before the 200 frames (likely) then end episode
            if done:
                print('Died at F:', frame)
                print('Avg Reward:', avg_reward / frame)
                break


def learning_with_random_inputs():
    # [(observation, action), ...]
    training_data = []

    # All scores from every frame
    scores = []

    # All scores that were above the threshold
    accepted_scores = []

    for _ in range(num_episodes):
        score = 0
        # Moves taken in this episode
        game_memory = []
        prev_observation = []

        for step in range(max_steps):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

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
        env.reset()

    np.save('training_data.npy', np.array(training_data))

    # Stats
    print('Avg Accepted Score:', mean(accepted_scores))
    print('Median of Accepted Scores:', median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data


learning_with_random_inputs()
