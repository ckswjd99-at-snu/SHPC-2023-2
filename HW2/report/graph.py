import numpy as np
import matplotlib.pyplot as plt

# Raw data. 
# Iterated 10 times. 
# Format: [thread, GFLOPS]
data1 = np.array([
    [   1,   12.109473],
    [   2,   24.986466],
    [   4,   44.867514],
    [   8,   80.882804],
    [  16,  159.467801],
    [  32,  230.937032],
    [  64,  319.222889],
    [ 128,  186.833631],
    [ 256,   93.317154],
]).transpose(1, 0)

data2 = np.array([
    [  16,  159.467801],
    [  32,  230.937032],
    [  48,  163.719001],
    [  64,  319.222889],
    [  80,  168.679404],
    [  96,  103.772299],
    [  112, 133.486816],
    [  128, 186.833631],
]).transpose(1, 0)

plt.plot(*data1)
plt.show()

plt.plot(*data2)
plt.show()