import os
from argparse import ArgumentParser
import keras.backend as K
from agent import Agent
import util


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = ArgumentParser()
parser.add_argument(
    'game',
    help='Select which game to play',
    type=str,
    choices=['space_invaders', 'breakout']
)
parser.add_argument(
    'type',
    help='The type of Deep Learning to use',
    type=str,
    choices=['dqn', 'ddqn']
)
parser.add_argument(
    'max_frames',
    help='The max number of frames to train the model for',
    type=int
)
parser.add_argument(
    'interval_frames',
    help='The number of frames to run in between game simulation testing',
    type=int
)
parser.add_argument(
    '-l',
    '--load_model',
    help='Load a previously trained model?',
    action='store_true'
)
parser.add_argument(
    '-d',
    '--display',
    help='Display video output of game?',
    action='store_true'
)
args = parser.parse_args()

if __name__ == '__main__':
    agent = Agent(
        args.game,
        args.type,
        display=args.display,
        load_model=args.load_model
    )

    interval = args.interval_frames
    max_frames = args.max_frames
    games_to_play = 10
    try:
        for i in range(max_frames // interval):
            running_score = 0
            frames_survived = 0
            print('Training Epoch: ', i + 1)
            avg_loss = agent.training(interval)
            for _ in range(games_to_play):
                agent_scores = agent.simulate_intelligent()
                running_score += agent_scores[0]
                frames_survived += agent_scores[1]

            running_score /= games_to_play
            frames_survived /= games_to_play

            # Save the Average Score and Frames survived over 10 agents for this interval
            util.save_results(
                './data/{}_avgscorex_{}.npy'.format(agent.name, args.type),
                './data/{}_avgscorey_{}.npy'.format(agent.name, args.type),
                (i + 1) * interval,
                running_score
            )

            util.save_results(
                './data/{}_avgframes_survx_{}.npy'.format(agent.name, args.type),
                './data/{}_avgframes_survy_{}.npy'.format(agent.name, args.type),
                (i + 1) * interval,
                frames_survived
            )

            # Save the average model loss over each training epoch
            # There are interval / 3 training epochs per interval
            util.save_results(
                './data/{}_lossx_{}.npy'.format(agent.name, args.type),
                './data/{}_lossy_{}.npy'.format(agent.name, args.type),
                (i + 1),
                avg_loss
            )
    except KeyboardInterrupt:
        print('Quitting...')

    K.clear_session()
