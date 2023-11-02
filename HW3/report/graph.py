import numpy as np
import matplotlib.pyplot as plt

# Raw data. 
# Iterated 10 times. 
# Format: [thread, GFLOPS]
data1 = np.array([
    [   1,   35.199595],
    [   2,   71.601795],
    [   4,  137.892078],
    [   8,  240.236484],
    [  16,  445.992250],
    [  32,  483.251362],
    [  64,  961.032611],
    [ 128,  369.857605],
    [ 256,  234.607769],
]).transpose(1, 0)

data2 = np.array([
    [  16,  445.992250],
    [  32,  483.251362],
    [  48,  229.006018],
    [  64,  961.032611],
    [  80,  271.003438],
    [  96,  118.718092],
    [  112, 176.362913],
    [  128, 369.857605],
]).transpose(1, 0)

data3 = np.array([
    [0, 234.237554],
    [1, 234.219445],
    [2, 236.203331],
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

plt.grid(True, axis='y', alpha=0.3)
plt.bar(np.arange(len(data3[1, :])), data3[1, :], width=0.5)
plt.xticks(np.arange(len(data3[0, :])), labels=["static", "dynamic", "guided"])
plt.yticks(range(0, 300, 100))
plt.ylabel('GFLOPS')
plt.show()