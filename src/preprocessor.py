import numpy as np


class Preprocessor():
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def stack_frames(self, frames):
        ''' Stacks the last 3 frames as input into the CNN '''
        return np.stack(frames, axis=-1)
