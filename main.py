'''
Created by Kevin De Angeli
Date: 2019-11-14
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sym
from PIL import Image
import random
import copy

def readData(showWxample=False):
    #pixels = list(img.getdata())
    #img = Image.open('example2.JPG')
    img = Image.open('pictureData.ppm')
    print(type(img))
    X = np.array(img)  # 120x120x3 Data
    picShape =[]
    picShape.append(X.shape[0])
    picShape.append(X.shape[1])
    X = X.reshape(X.shape[0] * X.shape[1], 3)
    if showWxample == True:
        img.show()
        print("Data Example")
        print(X[0:5])
        print(" ")
        print("Number of entries: ", X.shape[0])
        print("Min of the dataset ", np.amin(X,axis=0))
        print("Max of the dataset ", np.amax(X,axis=0))
        print(" ")
    return X, picShape


def displayPicture(imageArray, picShape):
    newImage = imageArray.reshape((picShape[0],picShape[1],3))
    img = Image.fromarray(newImage, 'RGB')
    img.show()

class K_means(object):

    def __init__(self, data):
        self.X= data
        self.iterations = 0
        self.C = [] #Clusters

    def train(self, k, iterationsLimit = -1):
        '''
        Note 1: the loop will continue until the clusters stop
        changing or until the optional iterationsLimit parameter
        is reached. -1 means no limit.

        Note 2: A common issue is to have empty clusters.
        For this the function clustersUpdate calls reInitializeEmptyClusters


        Steps:
        1. Initialize k clusters at a random position
        2. Label points based on closest neighbor (function: closestCluster)
        3. Update Clusters (function clustersUpdate)

        '''
        #could put low= 0 , high= 256. But I wanted ti try this
        #self.C = [np.random.randint(low=np.amin(self.X), high= np.amax(self.X), size=self.X.shape[1]) for i in range(k)]
        self.C = [np.random.randint(low=0, high= 256, size=self.X.shape[1]) for i in range(k)]

        self.C = np.array(self.C)
        #self.C = self.C.reshape((k,self.X.shape[1]))

        C_old = np.array([])


        while not self.finishLoopCheck(oldClusters=C_old, iterationsLim=iterationsLimit):
            print("Iteration: ", self.iterations)
            C_old=copy.deepcopy(self.C)  # To copy C by value not by reference
            #print(self.C)

            dataAssignment = self.closestCluster()
            self.clustersUpdate(dataAssignment)

            self.iterations+=1


    def finishLoopCheck(self, oldClusters, iterationsLim):
        if iterationsLim == self.iterations:
            return True
        else:
            return np.array_equal(oldClusters, self.C) #Clusters didn't change ?


    def closestCluster(self):
        '''
        Create a list where each data point is associated with a
        clusters. Then it returns the list of clusters.


        '''
        clusterAssignment = []
        for i in self.X:    #For each dataPoint
            dist = []
            for k in self.C: #For each cluster.
                dist.append(np.linalg.norm(i-k))
            min = np.amin(dist)
            index = dist.index(min)
            clusterAssignment.append(index)

        #return a list of size X where each element specifies the cluster.
        return  np.array(clusterAssignment)

    def reInitializeEmptyClusters(self, CIndex):
        '''
        Re-initialize clusters at randon.
        This is used when clusters are empty.
        '''

        newCoordinates = np.random.randint(low=0, high=256, size=self.X.shape[1])
        self.C[CIndex] = np.array(newCoordinates)


    def clustersUpdate(self, clusterAssignments):
        '''
        In order to handle "empty clusters" I re-initialized those clusters randonly.
        '''
        #clusterAssignments = np.array(clusterAssignments)
        newClusterCoordinate=[]
        #update self.C based on clusterAssignments

        for i in range(self.C.shape[0]):
            if i not in clusterAssignments:
                print("Not there: ", i)
                self.reInitializeEmptyClusters(CIndex= i)
                continue
            findDataPoints = clusterAssignments == i

            dataPointsCoordinates = self.X[findDataPoints]
            newClusterCoordinate = np.average(dataPointsCoordinates,axis=0)
            self.C[i] = newClusterCoordinate


    def mergeDataPoints(self):
        '''
        This function change the value of the
        data points based on the value of the closest neighboor.
        '''
        dataAssignment = self.closestCluster()

        for i in range(self.C.shape[0]):
            selectPoints = dataAssignment == i
            self.X[selectPoints] = self.C[i]

        return self.X





def main():
    data, picShape = readData(showWxample=True)

    kMeans = K_means(data)
    kMeans.train(k=20, iterationsLimit= 3)
    newImage = kMeans.mergeDataPoints()
    displayPicture(newImage,picShape )




if __name__ == "__main__":
    main()