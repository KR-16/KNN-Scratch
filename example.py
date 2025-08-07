import numpy as np
from knn import KNN
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

model = KNN(k=6)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(preds)
ac = accuracy_score(y_test, preds)
print(f"Accuracy: {ac:.2f}")