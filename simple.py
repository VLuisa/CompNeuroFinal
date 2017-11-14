import os
import struct
from mnist import read, show
import numpy as np
import math

training_data = list(read(dataset = "training", path = "./data"))
# print(len(training_data))
tlabel, tpixels = training_data[0]

# Find distance between trainDigit and testDigit
def calcDistL1(testval, trainval):
    dist = np.sum(np.abs(testval-trainval))
    return dist

def nearestNeighbor(testval):
    testlabel, testpixels = testval
    print(testlabel)
    print(testpixels)
    min_dist = 10000000
    for i in xrange(len(training_data)):
        tlabel, tpixels = training_data[i]
        dist = calcDistL1(testpixels, tpixels)
        if dist < min_dist:
            min_dist = dist
        data = (tlabel, min_dist)
    return data

# def predict(testNum, k):
    # Calc distance between to numbers
    # for i in range(len(training_data)):


testing_data = list(read(dataset = "testing", path = "./data"))
label, pixels = testing_data[0]
testval = (label, pixels)

# print(label)
# show(pixels)

print("Distance:")
# print(calcDistL1(pixels, tpixels))
# print(calcDistL1(pixels, pixels))
for i in xrange(10):
    print(nearestNeighbor(testing_data[i]))
