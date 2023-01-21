from python_speech_features import mfcc # creates a variable called mfcc, which is an object that represents the MFCC features of a speech signal
import scipy.io.wavfile as wav#imports scipy, which is used to read in audio files as waveforms
import numpy as np# uses numpy to create a list of all possible values for the MFCC feature vector for each frame in the file

from tempfile import TemporaryFile#reads in an audio file using TemporaryFile() and stores it into wav with pickle()
import os #to access os for memory access and modifications
import pickle# imports pickle so we can save our data into this file later on.
import random #imports random so we can generate some noise to test our system out on.
import operator# import operator so we can use mathematical functions like exponentials or square roots when calculating MFCC values from speech features.

import math
import numpy as np

#The code calculates the distance between two points.
#The first point is instance1 and the second point is instance2
#The code calculates the distance by using a dot product of two vectors, cm1 and cm2.
# It then uses logarithms to calculate how far away each vector is from its respective origin (mm1 and mm2).
#Then it subtracts k from this value to get the final result.
#The code calculates that if you take a unit length along one axis, multiply it by another unit length along another axis, add them together, then divide by 2k where k is some constant number, you will get your desired result which in this case would be 0 because there are no units on either axis so they cancel out
def distance(instance1 , instance2 , k ):
    distance =0 
    mm1 = instance1[0] 
    cm1 = instance1[1]
    mm2 = instance2[0]
    cm2 = instance2[1]
    distance = np.trace(np.dot(np.linalg.inv(cm2), cm1)) 
    distance+=(np.dot(np.dot((mm2-mm1).transpose() , np.linalg.inv(cm2)) , mm2-mm1 )) 
    distance+= np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
    distance-= k
    return distance

#The code starts by creating a list of distances between the training set and each instance.
#Then, it creates a list of all the neighbors for each instance.
#The code then sorts these lists in order from smallest to largest distance.
#The function getNeighbors() is called with three parameters: trainingSet, instance, k It starts by creating an empty list named distances which will be used to store the distances between instances and their corresponding training sets.
#It then iterates through every element in the range (len(trainingSet)): dist = distance(trainingSet[x], instance, k )+ distance(instance, trainingSet[x], k) .
#This line calculates how far away an instance is from its corresponding training set using Euclidean Distance formula and stores that value into dist .
# It then appends this value onto another list named neighbors which will contain all of the distances calculated so far.
#Finally it sorts these two lists in order from smallest to largest distance using operator.itemgetter() .
#The code is a function that returns the neighbors of an instance in a given training set.
#The first line creates an empty list called distances which will be used to store the distance between each instance and its neighbors.
# The next for loop iterates through all of the instances in the training set, and calculates the distance between them and their respective neighbors.
#The result of this calculation is then appended to distances list, with (trainingSet[x][2], dist) as its key value.
#This key value would correspond to where x is found in the list of instances, with 2 being the index number of that particular instance within its corresponding training set.
#After sorting distances by key value, we have created an array called neighbors

def getNeighbors(trainingSet, instance, k):
    distances = []
    for x in range (len(trainingSet)):
        dist = distance(trainingSet[x], instance, k )+ distance(instance, trainingSet[x], k)
        distances.append((trainingSet[x][2], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

#The code is a function that takes in an array of neighbors and returns the class with the highest vote.
#The code starts by creating a dictionary called "classVote" which is initialized to be empty.
# Then, for each neighbor, it checks if that neighbor is already in the list of classes.
#If so, then their vote count is incremented by one; otherwise they are added to the list of classes with a vote count of one.
#The sorted list returned from this function has only one element: 0[0].
#This means that there was only one class with more than 50% votes (the first number) and all other classes had less than 50% votes (the second number).
#The code will return the nearest class to the given neighbors.

def nearestClass(neighbors):
    classVote = {}

    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVote:
            classVote[response]+=1 
        else:
            classVote[response]=1

    sorter = sorted(classVote.items(), key = operator.itemgetter(1), reverse=True)
    return sorter[0][0]

#The code is trying to find the accuracy of a prediction.
#It starts with an empty list, and then iterates through all the predictions in the test set.
#For each one, it checks if that prediction is correct or not by checking if that prediction is equal to the last element in the list (the end of the test set).
#If so, it adds 1 to its count for that particular item.
#Then it returns this sum divided by how many items are in its list.
#The code first creates a variable called "correct" which will be used as a counter for how many times each item was correct out of all items in their respective lists.
#The next line sets up an empty list with no elements inside it; this will be used as our test set later on when we want to calculate accuracy rates for different predictions within our dataset.
#The next few lines create variables called "predictions" and "testSet".

def getAccuracy(testSet, predictions):
    correct = 0 
    for x in range (len(testSet)):
        if testSet[x][-1]==predictions[x]:
            correct+=1
    return 1.0*correct/len(testSet)

directory = "Data/genres_original/"
f= open("my.dat" ,'wb') #opens my.dat in writing in binary mode
i=0

# The code starts by reading the file from the directory.
#It then iterates through each of the files in that directory, and for each one it reads a 20ms segment of audio data into memory.
#The code then calculates an mfcc feature vector using this data, which is stored in a pickle object called "feature".
#The next part of the code iterates through all 11 directories in order to find any other folders with files named "wav" (the name of the sound file).
#For each folder found, it reads another 20ms segment of audio data into memory and calculates an mfcc feature vector using this data.
#This process continues until there are no more folders left to read or there are no more wav files to calculate features for.
#The code attempts to calculate the mean matrix for a given feature.
#The code calculates the mean matrix by taking the mean of each row in the covariance matrix and then adding them together.

for folder in os.listdir(directory):
    i+=1
    if i==11 :
        break   
    for file in os.listdir(directory+folder):  
        (rate,sig) = wav.read(directory+folder+"/"+file)
        mfcc_feat = mfcc(sig,rate ,winlen=0.020, appendEnergy = False)
        covariance = np.cov(np.matrix.transpose(mfcc_feat))
        mean_matrix = mfcc_feat.mean(0)
        feature = (mean_matrix , covariance , i)
        pickle.dump(feature , f)

f.close()

dataset = []

# The code opens the file my.dat with the open() function and then loops through all of the lines in it until it finds one where random.random() is less than or equal to split, which will be when it has found a line that should go into trSet (the training set) and teSet (the test set).
#Then for each line in dataset, if random.random() is greater than or equal to split, then we append that line into trSet; otherwise we append that line into teSet.
#The code is used to load a dataset into memory.
#The code will loop through the entire dataset and randomly select a subset of data from it, which is then appended to the trSet and teSet lists.

def loadDataset(filename , split , trSet , teSet):
    with open("my.dat" , 'rb') as f:
        while True:
            try:
                dataset.append(pickle.load(f))
            except EOFError:
                f.close()
                break  

    for x in range(len(dataset)):
        if random.random() <split :      
            trSet.append(dataset[x])
        else:
            teSet.append(dataset[x])  

trainingSet = []
testSet = []
loadDataset("my.dat" , 0.66, trainingSet, testSet)

leng = len(testSet)
predictions = []
for x in range (leng):
    predictions.append(nearestClass(getNeighbors(trainingSet ,testSet[x] , 5))) 

accuracy1 = getAccuracy(testSet , predictions)
print('Accuracy: ')
print(accuracy1)
