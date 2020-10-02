#!/usr/bin/env python

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import sys

if __name__ == '__main__':
    argv = sys.argv
    if len(argv) <= 1:
        print('Specify pickle file as parameter.')
    else:
        plt.figure().add_subplot(111, projection='3d')
        file = open(argv[1], 'rb')
        image = pickle.load(file)
        image['plotfun'](image)
        plt.show()
