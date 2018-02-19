from pprint import pprint
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt


def save_results(pathx, pathy, xdata, ydata):
    ''' Save a set of metrics to numpy arrays specified by pathx, pathy '''

    try:
        xdata_loaded = np.load(pathx)
        ydata_loaded = np.load(pathy)
        xdata_loaded = np.append(xdata_loaded, xdata)
        ydata_loaded = np.append(ydata_loaded, ydata)

    except FileNotFoundError:
        if '.npy' not in pathx or '.npy' not in pathy:
            print('Paths must include the .npy extension')
            raise FileNotFoundError
        else:
            xdata_loaded = xdata
            ydata_loaded = ydata

    np.save(pathx, xdata_loaded)
    np.save(pathy, ydata_loaded)
    print('Saved {}, {} at {}'.format(pathx, pathy, datetime.now()))


def print_results(game, mode):
    if game not in ['space_invaders', 'breakout']:
        print('Invalid game name: ', game)
        return

    if mode not in ['dqn', 'ddqn']:
        print('Invalid DL Mode: ', mode)
        return

    print('Epochs')
    pprint(np.load('./data/{1}/{0}_avgscorex_{1}.npy'.format(game, mode)))
    print('Average Score')
    pprint(np.load('./data/{1}/{0}_avgscorey_{1}.npy'.format(game, mode)))
    print('-----------------------')
    print('Epochs')
    pprint(np.load('./data/{1}/{0}_avgframes_survx_{1}.npy'.format(game, mode)))
    print('Average Frames Survived')
    pprint(np.load('./data/{1}/{0}_avgframes_survy_{1}.npy'.format(game, mode)))
    print('-----------------------')
    print('Epochs')
    print(np.load('./data/{1}/{0}_lossx_{1}.npy'.format(game, mode)))
    print('Average Frames Survived')
    print(np.load('./data/{1}/{0}_lossy_{1}.npy'.format(game, mode)))


def plot_results(pathx, pathy):
    try:
        xdata = np.load(pathx)
        ydata = np.load(pathy)
        plt.plot(xdata, ydata, 'r')
        plt.show()
    except FileNotFoundError:
        if '.npy' not in pathx or '.npy' not in pathy:
            print('Paths must include the .npy extension')
        else:
            print('No data to plot found at {} or {}'.format(pathx, pathy))

        raise FileNotFoundError
