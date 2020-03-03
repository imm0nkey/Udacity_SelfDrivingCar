import numpy as np

n_features = 6
weights = np.random.normal(scale=1/n_features**.5, size=n_features)
print(weights)
