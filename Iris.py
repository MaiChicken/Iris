
import math
import random
import numpy as np

#try building own KNN abd

class OwnKNN():
    def dist(self, first, second):
        distance = 0
        for i in range(len(first)):
            distance += pow(first[i]-second[i],2)
        return math.sqrt(distance)
    def fit(self, x_train, y_train, k):
        self.x_train = x_train
        self.y_train = y_train
        self.k = k

    def predict(self, test):
        prediction = []
        for i in range(len(test)):
            distance = []
            for j in range(len(self.x_train)):
                distance.append(self.dist(test[i],self.x_train[j]))
            sorted_distance = sorted(distance)
            temp_predict = []
            for k in range(self.k):
                a = np.where(np.isclose(distance, sorted_distance[k]))
                temp_predict += a[0].tolist()
            original_label = []
            for k in range(self.k):
                original_label.append(self.y_train[temp_predict[k]])
            prediction.append(max(set(original_label), key = original_label.count))
        return prediction

from sklearn import datasets
import sklearn.model_selection
import sklearn.metrics
iris = datasets.load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.2)
clf = OwnKNN()
clf.fit(x_train, y_train, k=3)
prediction = clf.predict(x_test)
print(prediction)

print(sklearn.metrics.accuracy_score(y_test, prediction))


