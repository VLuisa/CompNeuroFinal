import os
import struct
from mnist import read, show
import numpy as np
from collections import Counter
import time

start_time = time.time()

training_data = list(read(dataset = "training", path = "./data"))
testing_data = list(read(dataset = "testing", path = "./data"))

# Find distance between trainDigit and testDigit
def calcDistL1(testval, trainval):
    dist = np.sum(np.abs(testval-trainval))
    return dist

def nearestNeighbor(testval):
    testlabel, testpixels = testval
    min_dist = 10000000
    distances = []
    for i in xrange(len(training_data)):
        tlabel, tpixels = training_data[i]
        dist = calcDistL1(testpixels, tpixels)
        data = (dist, tlabel)
        distances.append(data)
    distances.sort()
    return distances

def knn(k, testDigitIndex):
    # show(testing_data[testDigitIndex][1])
    label = testing_data[testDigitIndex][0]
    # print("Test Digit Label: " + str(label))
    sorted_dists = nearestNeighbor(testing_data[testDigitIndex])
    knnObjs = sorted_dists[:k]

    # strip list to be list of nearest labels
    nearest = []
    for x in xrange(len(knnObjs)):
        t, l = knnObjs[x]
        nearest.append(l)

    this_data = Counter(nearest)
    vote = this_data.most_common(1)
    prediction = vote[0][0]
    return (label, prediction)

# def calcAccuracy():

totalcorrect = 0
# totalincorrect = 0
for i in xrange(len(testing_data)):
    result = knn(5, i)
    l, p = result
    # print("I think this is " + str(p))
    if l == p:
        # print("Correct")
        totalcorrect = totalcorrect + 1
        # print(totalcorrect)
    # else:
        # print("Incorrect")

accuracy = totalcorrect / len(testing_data)
print("Accuracy: " + str(accuracy))

print("--- %s seconds ---" % (time.time() - start_time))
# print("I think this is a " + str(knn(1, 1)))
# print("I think this is a " + str(knn(1, 5)))
# print("I think this is a " + str(knn(1, 3)))
# for i in xrange(10):
#     print(nearestNeighbor(testing_data[i]))
