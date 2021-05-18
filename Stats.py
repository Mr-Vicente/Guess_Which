import matplotlib.pyplot as plt
import numpy as np

def draw_hist(indices, distances):
    plt.figure()
    indices = np.array(indices)
    distances = np.array(distances)
    result = plt.bar(indices, distances, color='c', edgecolor='k', alpha=0.65)
    plt.axhline(distances.mean(), color='k', linestyle='dashed', linewidth=1)
    plt.axhline(np.median(distances), color='k', linestyle='dashed', linewidth=1)
    plt.show()