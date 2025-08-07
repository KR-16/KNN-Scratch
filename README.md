# KNN-Scratch
KNN From Scratch

## KNN Algorithm
1. Step1: Select the optimal value of K
  a. K is the number of nearest neighbors that needs to be considered while making predictions
2. Step2: Calculate the distance
  a. To measure the similarity between target and training data points.
  b. Distance is calculated between data points in the dataset and target point.
3. Step3: Finding Nearest Neighbors
   a. The k data points with smallest distances to the target point are nearest neighbors.
4. Step4: Voting for Classification or Mean for Regression
   a. For classification, the class with the most votes among the nearest neighbors is chosen.
   b. For regression, the mean of the target values of the nearest neighbors is calculated.


## Prediction Function
* distances.append - saves the distance between the target is from the testing point, along with the label
* distances.sort - sorts the distances in ascending order
* k_nearest_labels - pciks the labels of the k closest points
* Counter - counts the frequency of each label in the k nearest labels