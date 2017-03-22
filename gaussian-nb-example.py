import numpy as np


features = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
labels = np.array([1, 1, 1, 2, 2, 2])

#train
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features, labels)

result = clf.predict([[-0.8, -1]])
print(result)