import numpy as np
import scipy
from scipy import signal
import scipy.io as sio
import os
import cv2
import math
import pickle
import pandas as pd

# define body part slices
nose = np.s_[0: 2]
leftEar = np.s_[3: 5]
rightEar = np.s_[6: 8]
neck = np.s_[9: 11]
spine1 = np.s_[12: 14]
leftShoulder = np.s_[15: 17]
rightShoulder = np.s_[18: 20]
spine2 = np.s_[21: 23]
spine3 = np.s_[24: 26]
tail1 = np.s_[27: 29]
tail2 = np.s_[30: 32]
tail3 = np.s_[33: 35]
tail4 = np.s_[36: 38]
tail5 = np.s_[39: 41]

#list containing slices
bodyPartList = [nose, leftEar, rightEar, neck, spine1, leftShoulder, rightShoulder, spine2, spine3, tail1, tail2, tail3, tail4, tail5]

# dictionary containing indexes for liklihood values
liklihoodIndex = ({
    'nose' : 2, 'leftEar' : 5, 'rightEar' : 8, 'neck' : 11, 'spine1' : 14, 'leftShoulder' : 17, 'rightShoulder' : 20, 
    'spine2' : 23, 'spine3' : 26, 'tail1' : 29, 'tail2' : 32, 'tail3' : 35, 'tail4' : 38, 'tail5' : 41
})

path = 'X:\ibn-vision\DATA\SUBJECTS_AB'  

problematicEscapes=[]
LiklihoodThreshold=0.90
houseBoundary=30
jumpThreshold=8
crossFrameThresh=70


def FindJumps(coor, jumpThreshold):
    """
    input: coordinate, either x or y's
    find large jumps in tracking
    """

    rolledCoor = np.roll(coor, 1)
    frameDifferences = abs(rolledCoor[1:] - coor[1:])
    frameJumps = list(np.where(frameDifferences > jumpThreshold)[0])
    return frameJumps


def medianFilteringInterpolation(coors):
    """
    first applies median filtering
    then applies average interpolation over the larger gaps
    good explanation of this function: https://stackoverflow.com/questions/6518811
    good explanation of lamda function: https://realpython.com/python-lambda
    """

    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    scipy.signal.medfilt(coors, 3)
    nans, x = nan_helper(coors)
    if False in nans:
        coors[nans] = np.interp(x(nans), x(~nans), coors[~nans])

directory_list = os.listdir(path)
for directory in directory_list:
    file_list = os.listdir(path + '/' + directory)
    for file in file_list:
        if 'escape' in file:
            response = np.load(path + '/' + directory + '/' + file)
            # iterate over each escape
            escapeColumnsForStacking = []
            for bodyPart, likely in zip(bodyPartList, liklihoodIndex):
                # iterate over each body part
                bodyPartCoors = response[bodyPart]
                X = np.copy(bodyPartCoors[0])
                Y = np.copy(bodyPartCoors[1])
                likelyVals = np.copy(response[liklihoodIndex[likely]])

                # find where tracking goes awry
                lowLiklihood = list(np.where(likelyVals < LiklihoodThreshold)[0])

                framesToRemove = list(set(lowLiklihood + FindJumps(X, jumpThreshold) + FindJumps(Y, jumpThreshold)))

                # removing those frames from cordinates
                for i in framesToRemove:
                    X[i] = np.nan
                    Y[i] = np.nan

                medianFilteringInterpolation(Y)
                medianFilteringInterpolation(X)

                escapeColumnsForStacking.append(X)
                escapeColumnsForStacking.append(Y)
                escapeColumnsForStacking.append(likelyVals)

                ProcessedEscape = np.column_stack(escapeColumnsForStacking).T
            np.save(path + '/' + directory + '/' + file[:-4] + 'processed.npy', ProcessedEscape)






















