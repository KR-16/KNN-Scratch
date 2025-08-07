import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
    
    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    # helper function
    def _predict_point(self, x):
        # compute distances
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]

        # get the indices of k nearest - closest k
        k_indices = np.argsort(distances)[:self.k]

        # get corresponding labels
        k_labels = [self.y_train[i] for i in k_indices]

        # Majority Vote
        most_common = Counter(k_labels).most_common()
        return most_common[0][0]
    
    def predict(self, X_test):
        predictions = [self._predict_point(x) for x in X_test]
        return np.array(predictions)