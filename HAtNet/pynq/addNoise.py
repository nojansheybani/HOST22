import numpy as np
import os

dir = os.listdir("./valid")

for file in dir:
    weights = np.load("./valid/" + file)
    noise = np.random.normal(0,1,weights.shape)
    noisyWeights = weights + noise
    print(np.sum(weights) - np.sum(noisyWeights))
    np.save("./invalid/"+file, noisyWeights)
    