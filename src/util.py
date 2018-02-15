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
