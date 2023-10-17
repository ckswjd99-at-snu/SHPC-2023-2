import numpy as np
import matplotlib.pyplot as plt

# Raw data. 
# Iterated 10 times. 
# Format: [thread, GFLOPS]
data1 = np.array([
    [   1,   29.587183],
    [   2,   67.573975],
    [   4,  126.452963],
    [   8,  250.105695],
    [  16,  402.960087],
    [  32,  541.627833],
    [  64,  585.510650],
    [ 128,  329.937241],
    [ 256,  236.764712],
]).transpose(1, 0)

data2 = np.array([
    [  16,  402.960087],
    [  32,  541.627833],
    [  48,  220.415782],
    [  64,  585.510650],
    [  80,  238.057714],
    [  96,  108.586687],
    [  112, 167.649999],
    [  128, 329.937241],
]).transpose(1, 0)

plt.grid(True, axis='y', alpha=0.3)
plt.bar(np.arange(len(data1[1, :])), data1[1, :], width=0.5)
plt.xticks(np.arange(len(data1[1, :])), labels=data1[0, :].astype(np.int32))
plt.xlabel('# of threads')
plt.ylabel('GFLOPS')
plt.show()

plt.grid(True, axis='y', alpha=0.3)
plt.bar(np.arange(len(data2[1, :])), data2[1, :], width=0.5)
plt.xticks(np.arange(len(data2[1, :])), labels=data2[0, :].astype(np.int32))
plt.xlabel('# of threads')
plt.ylabel('GFLOPS')
plt.show()