import os
import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

def readPic(picPath):
    imgValues = []
    img = None
    if os.path.isfile(picPath):
        img = cv2.imread(picPath, cv2.IMREAD_COLOR)
    if img is not None:
        for x in range(32):
            imgValues.append([])
            for y in range(32):
                imgValues[x].append(0)
                if img[x][y][0] > 0:
                    imgValues[x][y] = 1
    return imgValues

def readAllPicsInFolder(folderPath):
    dirs = os.listdir(folderPath)
    pics = []
    indx = 0
    for i in range(len(dirs)):
        if "moving" not in dirs[i]:
            pics.append([])
            pics[indx] = readPic(folderPath + '/' + dirs[i])
            indx +=1
    return pics

def extractStationaryAndNonStationaryParts(picData, stationaryLimit):
    #stationary limit should be from 10%-40% prefferably 20%
    newPicData = []
    for x in range(32):
        newPicData.append([])
        for y in range(32):
            decisionArr = []
            for i in range(len(picData)):
                if int(picData[i][x][y]) == 1:
                    decisionArr.append(1)
                else:
                    decisionArr.append(0)
            if len(picData) > 0:
                if float(sum(decisionArr)/len(picData)) > (1-stationaryLimit):
                    newPicData[x].append(1)
                else:
                    newPicData[x].append(0)
    return newPicData

def extractMovingParts(picData, movingLimit):
    # Moving limit should be from 20 % to 80% prefferably 60%
    newPicData = []
    for x in range(32):
        newPicData.append([])
        for y in range(32):
            decisionArr = []
            for i in range(len(picData)):
                if int(picData[i][x][y]) == 1:
                    decisionArr.append(1)
                else:
                    decisionArr.append(0)
            if float(sum(decisionArr)/len(picData)) < float(1- ((1-movingLimit)/2)) and float(sum(decisionArr)/len(picData)) > float(((1-movingLimit)/2)) and len(picData) > 0:
                newPicData[x].append(1)
            else:
                newPicData[x].append(0)
    return newPicData

def writePic(extractedData, writeWhere):
    blank_image = np.zeros((32,32,3), np.uint8)
    for i in range(32):
        for y in range(32):
            if extractedData[i][y] == 1:
                blank_image[i][y] = [255, 255, 255]
    cv2.imwrite(writeWhere, blank_image)

import os
labels = ['stairs', 'bending', 'kneeling', 'lying', 'sitting', 'standing', 'walk']
folderName = 'test'
allowedToPass = 0.4
label = 'walk' # label
picDir = None
picData = readAllPicsInFolder(picDir)
extractedMovingParts = extractMovingParts(picData, allowedToPass)
extractedStationaryAndNonStationaryParts = extractStationaryAndNonStationaryParts(picData, 1 - (allowedToPass * 2))
if not os.path.exists(os.path.join(".", label, folderName, picDir, 'moving' + str(allowedToPass))):
    os.makedirs(os.path.join(".", label, folderName, picDir, 'moving' + str(allowedToPass)))
writePic(extractedMovingParts, os.path.join(".", label, folderName, picDir, 'moving' + str(allowedToPass), "result_move.png"))
writePic(extractedStationaryAndNonStationaryParts, os.path.join(".", label, folderName, picDir, 'moving' + str(allowedToPass), "result-stationary.png"))