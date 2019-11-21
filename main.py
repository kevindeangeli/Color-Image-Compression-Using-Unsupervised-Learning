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
from sklearn.cluster import MeanShift

def readData(showWxample=False):
    #pixels = list(img.getdata())
    #img = Image.open('example2.JPG')
    img = Image.open('pictureData.ppm')
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
        '''
        Stop the program if the clusters' position stop changing or
        the limit number of iterations has been reached.
        '''
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
                print("Empty Cluster: ", i)
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

class WinnerTakeAll(object):
    def __init__(self, data):
        self.X = data
        self.C = 0
        self.iterations = 0
        self.epselon = 0

    def finishLoopCheck(self, oldClusters, iterationsLim):
        '''
        Stop the program if the clusters' position stop changing or
        the limit number of iterations has been reached.
        '''
        if iterationsLim == self.iterations:
            return True
        else:
            return np.array_equal(oldClusters, self.C) #Clusters didn't change ?

    def closestCluster(self, testPoint):
        '''
        Compute euclidian distance of a testPoit to each of the clusters.
        Return the index of the closest cluster.
        '''
        dist = []
        for i in range(self.C.shape[0]):
            dist.append(np.linalg.norm(self.C[i] - testPoint))
            min = np.amin(dist)
            WinnerIndex = dist.index(min)

        return  WinnerIndex



    def clustersUpdate(self):
        '''
        For each data point. Find the closet cluster (function closestCluster)
        and update the position of the cluster based on W_new = W_old + epselon( X - W_old)
        '''
        newClusterCoordinate=[]
        #update self.C based on clusterAssignments

        for k in self.X:
            winnerCluster = self.closestCluster(k)
            newClusterCoordinate = self.C[winnerCluster] + self.epselon*(k - self.C[winnerCluster])
            self.C[winnerCluster] = newClusterCoordinate


    def train(self, k, iterrationsLimit = -1, epselon=0.1):
        '''
        Randomly initialize clusters and then update them.
        '''
        self.epselon = epselon
        self.C = k
        self.C = [np.random.randint(low=0, high=256, size=self.X.shape[1]) for i in range(k)]
        self.C = np.array(self.C)
        C_old = [] #old clusters. Used to check if they stopped changing.

        while not self.finishLoopCheck(oldClusters=C_old, iterationsLim=iterrationsLimit):
            print("Iteration: ", self.iterations)
            C_old = copy.deepcopy(self.C)  # To copy C by value not by reference
            #dataAssignment = self.closestCluster()
            self.clustersUpdate()
            self.iterations += 1


    def closestClusterForMerging(self):
        '''
        Create a list where each data point is associated with a
        cluster. Then it returns the list of clusters.
        This is used to create the final picture.
        Once the cluster position are set, assign the coordinate of
        the cluster to each of the the data points that are close.
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

    def mergeDataPoints(self):
        '''
        This function change the value of the
        data points based on the value of the closest neighboor.
        '''
        dataAssignment = self.closestClusterForMerging()

        for i in range(self.C.shape[0]):
            selectPoints = dataAssignment == i
            self.X[selectPoints] = self.C[i]

        return self.X

class KohonenMaps(object):

    def __init__(self, data):
        self.X = data
        self.xmax= None
        self.ymax= None
        self.learningRate = None
        self.currentEpoch = 1
        self.learnRateFunc = self.timeInverse
        self.totalEpochs = None
        self.C = None #It contains a dic where keys are 2D-grid coordinates, and values are points in the data space.


    def train(self,xmax=10, ymax=10, epochs=4,  learningRate = "time inverse"):
        self.xmax= xmax
        self.ymax= ymax
        self.totalEpochs = epochs
        self.learningRate = learningRate
        if learningRate == "time proportional":
            self.learnRateFunc = self.timeProportional

        self.initializeGrid()

        for k in range(self.totalEpochs): #for each epoch
            for i in self.X: #For each data point
                clusters = np.array(list(self.C.values()))
                gridCoordinates = np.array(list(self.C.keys()))

                winnerIndex = self.findWinnerNeuron(i, clusters)
                self.updateClusters(gridCoordinates[winnerIndex],clusters[winnerIndex],i)
                self.currentEpoch+=1


    def updateClusters(self, winnerCoordinates, clusterCoordiante, testPoint):
        '''
        Based on the winner, update all the clusters.
        :param winnerCoordinates:
        :param clusterCoordiante:
        :param testPoint:
        '''

        winnerCoordinates = np.array(winnerCoordinates)
        coordinateDifference = testPoint - clusterCoordiante[0]

        for i in range(self.xmax):
            for j in range(self.ymax):
                clusterGridCoordinate = np.array([i,j])
                gridDistance = np.sum(np.abs(winnerCoordinates-clusterGridCoordinate))
                newCoordinates = self.updateCordinates((i,j),gridDistance,coordinateDifference)
                self.C[(i,j)] = newCoordinates

    def updateCordinates(self, coordinate, gridDistance,coordinateDifference):
        '''
        Update the cluster based on the eqution:
        W_k+1 = W_k + LearRateFunc()*Neighborhood Function
        Here, the neighbordhood function used is 1/exp(gridDifference/2)
        Note that all clusters are being updated, but the farthest away in the
        2D grid are not being affected much. Some paper use the "radio" idea
        to identify which ones should be updated.

        :param coordinate: 2D grid coordinate
        :param gridDistance: 2D grid distance
        :param coordinateDifference: Difference between the winner cluster and the test point
        :return:
        '''
        #add diferenece as a parameter.
        value = self.C[coordinate]
        newVal = value + (self.learnRateFunc()* (1/np.exp(gridDistance/2))*coordinateDifference)
        return newVal




    def findWinnerNeuron(self, testPoint,clusters):
        '''
        Compute euclidian distance of a testPoit to each of the clusters.
        Return the index of the closest cluster.
        '''
        dist = []
        for i in range(clusters.shape[0]):
            dist.append(np.linalg.norm(clusters[i] - testPoint))
        min = np.amin(dist)
        #print(dist)
        WinnerIndex = dist.index(min)
        return WinnerIndex #coordinates of the winner


    def initializeGrid(self):
        '''
        Create a grid dictionary where the key is the value in a
        2D matric (i,j), and the key is the coordinates of that point.
        '''
        totalNeurons = self.xmax*self.ymax
        #initialClusters = np.array([np.random.randint(low=0, high= 256, size=self.X.shape[1]) for i in range(totalNeurons)])

        #Create mapping dictionary from grid to coordinates:
        self.C = {}
        for i in range(self.xmax):
            for j in range(self.ymax):
                key = (i,j)
                value = np.array([np.random.randint(low=0, high= 256, size=self.X.shape[1])])
                self.C[key] = value




    def mergeDataPoints(self):
        '''
        This function change the value of the
        data points based on the value of the closest neighboor.
        '''
        clusters = np.array(list(self.C.values()))

        dataAssignment = self.closestCluster(clusters)

        for i in range(clusters.shape[0]):
            selectPoints = dataAssignment == i
            self.X[selectPoints] = clusters[i]
        return self.X

    def closestCluster(self, clusters):
        '''
        This function is called by mergeDataPoints only. (for this class)
        Create a list where each data point is associated with a
        clusters (closest). Then it returns the list of clusters.
        '''
        clusterAssignment = []
        for i in self.X:  # For each dataPoint
            dist = []
            for k in clusters:  # For each cluster.
                dist.append(np.linalg.norm(i - k))
            min = np.amin(dist)
            index = dist.index(min)
            clusterAssignment.append(index)

        # return a list of size X where each element specifies the cluster.
        return np.array(clusterAssignment)




    def timeInverse(self):
        '''
        One of the Learning rate functions
        :return: (.9)**k
        '''
        return (.9) ** (self.currentEpoch)

    def timeProportional(self):
        '''
        One of the Learning rate functions
        :return: (1-k/K)
        '''
        return (1 - (self.currentEpoch / self.totalEpochs))



def MSE(A,B):
    '''
    :param A: array of pixels of image 1
    :param B: array of pixels of image 2
    :return: Mean Square Value.
    '''
    mse = np.subtract(A.astype(np.int16), B.astype(np.int16))
    mse = mse**2
    mse = np.sum(mse) / A.shape[0]
    return mse

def PSNR(A,B):
    '''
    :param A: array of pixels of image 1
    :param B: array of pixels of image 2
    :return: PSNR score.
    '''
    mse = MSE(A,B)
    return 20 * np.log10(255/np.sqrt(mse))



def main():
    data, picShape = readData(showWxample=False)
    originalPic, picShape = readData(showWxample=False)
    print(picShape)




    '''
    kMeans = K_means(data)
    kMeans.train(k=16, iterationsLimit= 10)
    newImage0 = kMeans.mergeDataPoints()
    displayPicture(newImage0,picShape )
    print(MSE(originalPic,newImage0))
    print(PSNR(originalPic,newImage0))
    '''



    '''
    #data, picShape = readData(showWxample=False)
    winner_take_all = WinnerTakeAll(data)
    winner_take_all.train(k=16, iterrationsLimit= 10, epselon = 0.18) #.1 works
    newImage2 = winner_take_all.mergeDataPoints()
    displayPicture(newImage2, picShape)
    print(MSE(originalPic,newImage2))
    print(PSNR(originalPic,newImage2))
    '''


    '''
    SOM = KohonenMaps(data)
    SOM.train(xmax=4, ymax=4, epochs=5, learningRate="time inverse")
    newImage3 = SOM.mergeDataPoints()
    displayPicture(newImage3,picShape)
    print(MSE(originalPic,newImage3))
    print(PSNR(originalPic,newImage3))
    '''

    bandwidth=10
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(data)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("With a bandwidth of size: ", bandwidth, "Number of clusters: ", n_clusters_)

if __name__ == "__main__":
    main()