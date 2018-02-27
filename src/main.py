#!/home/support/apps/cports/rhel-6.x86_64/gnu/Python/3.5.2/bin/python3

import os
from argparse import ArgumentParser
from traceback import print_exc
import tensorflow as tf
import keras.backend as K
from keras.backend.tensorflow_backend import set_session
from agent import Agent
import util

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = ArgumentParser()
parser.add_argument('game', help='Select which game to play', type=str, choices=['space_invaders', 'breakout'])
parser.add_argument('deep_learning_mode', help='The type of Deep Learning to use', type=str, choices=['dqn', 'double', 'duel'])
parser.add_argument('training_steps', default=25000, nargs='?', help='The number of steps (3 frames), to run during a training epoch?', type=int)
parser.add_argument('training_epochs', default=20, nargs='?', help='The number of training epochs to run', type=int)
parser.add_argument('evalutation_games', default=10, nargs='?', help='The number of games to evaluate on', type=int)
parser.add_argument('-l', '--load_model', default=True, help='Use this flag to start with a new model', action='store_false')
parser.add_argument('-d', '--display', help='Display video output of game?', action='store_true')
parser.add_argument('-r', '--record', help='Record a simulation of the game?', action='store_true')
args = parser.parse_args()


def main():
    agent = Agent(
        args.game,
        args.deep_learning_mode,
        display=args.display,
        load_model=args.load_model,
        record=args.record
    )

    if args.record:
        util.record(agent, './data/recordings/{}/{}/'.format(args.game, args.deep_learning_mode))
        return

    games_to_play = args.evalutation_games
    try:
        for epoch in range(args.training_epochs):
            running_score = 0
            frames_survived = 0
            print('Training Epoch: ', epoch + 1)
            avg_loss = agent.training(args.training_steps)
            for _ in range(games_to_play):
                agent_scores = agent.simulate_intelligent(evaluating=True)
                running_score += agent_scores[0]
                frames_survived += agent_scores[1]

            running_score /= games_to_play
            frames_survived /= games_to_play

            # Save the Average Score and Frames survived over 10 agents for this interval
            util.save_results(
                './data/{1}/{0}_avgscore_{1}.npy'.format(agent.name, args.deep_learning_mode),
                running_score
            )

            util.save_results(
                './data/{1}/{0}_avgframes_surv_{1}.npy'.format(agent.name, args.deep_learning_mode),
                frames_survived
            )

            # Save the average model loss over each training epoch
            # There are interval / 3 training epochs per interval
            util.save_results(
                './data/{1}/{0}_loss_{1}.npy'.format(agent.name, args.deep_learning_mode),
                avg_loss
            )

    except KeyboardInterrupt:
        print('Quitting...')


def play_and_display_intelligent():
    agent = Agent(
        args.game,
        args.deep_learning_mode,
        display=True,
        load_model=True
    )
    agent.simulate_intelligent(evaluating=True)


if __name__ == '__main__':
    main()
    # play_and_display_intelligent()
    K.clear_session()
