import numpy as np
import constants
import random



#this function takes in a csv path and returns the number of rows minus the top row
def numberOfRows(csv):
    currentFile = open(csv)
    singleLine = currentFile.readline()

    inc = 0
    while True:
        singleLine = currentFile.readline()
        if singleLine == '':
            break
        inc += 1
    return inc


#this function takes in a data csv and returns a dxn matrix and a label matrix. It also adds a 1 on the first index of the data array
#to account for the w0 
#@param it takes in a CSV to create a data arr out of 
#@return it returns a array with index 0 being the data Array and index1 being the label Array
def createDataArray(csv):
    currentFile = open(csv)
    singleLine = currentFile.readline()
    numberOfRow = numberOfRows(csv) 
    labelArr = np.zeros(shape=(numberOfRow, 1))

    inc = 0
    while True:
        singleLine = currentFile.readline()
        if singleLine == '':
            break

        singleLine = singleLine.split(",")
        singleLine[-1] = singleLine[-1].replace("\n","")
        for i in range(0, singleLine.__len__()):
            singleLine[i] = int(singleLine[i])

        #this removes the label into the label Array and adds a 1 on the front for w0
        labelArr[inc] = singleLine.pop(-1)
        singleLine.insert(0, 1)

        if(inc == 0):
            DataArr = np.zeros(shape=(numberOfRow, singleLine.__len__()))

        for i in range(0,singleLine.__len__()):
            DataArr[inc][i] = singleLine[i]
        inc += 1
    return [DataArr, labelArr]


#this function returns an array of the optimized weights
def LRTraining(dataArr, learnRate, maxItterations, regConstant, nPlusOneLen):
    weightArr = np.zeros(shape=(nPlusOneLen, 1))
    random.seed(392)
    for inc in range(0, nPlusOneLen):
        weightArr[inc] = random.random()
    
    for t in range(1,maxItterations):
        yPredArr = np.zeros(shape=(len(dataArr), 1))
        zedArr = np.zeros(shape=(len(dataArr), 1))

        for i in range(0, len(dataArr)):
            zedArr[i] = 0
            for j in range(1, nPlusOneLen):
                zedArr[i] = zedArr[i] + (weightArr[j] * dataArr[i][j])

            yPredArr[i] = (1/(1+np.exp(-1 * zedArr[i])))

        gradientVec = np.zeros(shape=(nPlusOneLen, 1))

        for j in range(0,nPlusOneLen - 1):
            gradientVec[j] = 0
            for i in range(1, len(dataArr)):
                gradientVec[j] = gradientVec[j] + (dataArr[i][j] * ())



# print(numberOfRows(constants.enronBOWTrainPaths[0]))
dataStuff = createDataArray(constants.enronBOWTrainPaths[0])
LRTraining(dataStuff[0], 1, 5, 1, dataStuff[0].shape[1])