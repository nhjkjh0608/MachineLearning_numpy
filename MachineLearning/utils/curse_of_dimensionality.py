import matplotlib.pyplot as plt
import numpy as np


def get_edge_length(fraction, dimension):
    return fraction ** (1 / dimension)


def show_curse_of_dimensionality():
    x = np.linspace(0, 1)
    for i in range(1,11,2):
        plt.plot(x, get_edge_length(x, i), label=str(i)+'dim')
    plt.legend()
    plt.xlabel('Fraction of data')
    plt.ylabel('Length')
    plt.show()
