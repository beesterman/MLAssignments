import numpy as np
import constants
import random
from NBTraining import finalReport


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

def numberOfSpamInData(labelArr):
    count = 0
    for label in labelArr:
        if label == 1:
            count += 1
    return count


def weightToCSV(weightArr):
    currentFile = open(constants.projectDataPath + "storedWeights.csv", "w")
    currentFile.write(str(len(weightArr)))
    currentFile.write(',')

    for i in range(0, len(weightArr)):
        currentFile.write(str(weightArr[i][0]))
        currentFile.write(',')
    


def CSVtoWeight(csv):
    currentFile = open(csv)
    singleLine = currentFile.readline()
    singleLine = singleLine.split(',')
    
    weightArr = np.zeros(shape=(int(singleLine[0]), 1))

    for i in range(0, len(singleLine) - 2):
        weightArr[i][0] = float(singleLine[i + 1])



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
def LRTraining(dataArr, labelArr, learnRate, maxItterations, regConstant, nPlusOneLen):
    weightArr = np.zeros(shape=(nPlusOneLen, 1))
    random.seed(392)
    for inc in range(0, nPlusOneLen):
        weightArr[inc][0] = random.random()
    
    for t in range(1,maxItterations):
        yPredArr = np.zeros(shape=(len(dataArr), 1))
        zedArr = np.zeros(shape=(len(dataArr), 1))

        for i in range(0, len(dataArr)):
            zedArr[i] = 0
            for j in range(0, nPlusOneLen):
                zedArr[i][0] = zedArr[i][0] + (weightArr[j][0] * dataArr[i][j])
            
            yPredArr[i][0] = (1/(1+np.exp(-1 * zedArr[i][0])))

        gradientVec = np.zeros(shape=(nPlusOneLen, 1))

        for j in range(0,nPlusOneLen - 1):
            gradientVec[j] = 0
            for i in range(1, len(dataArr)):
                gradientVec[j] = gradientVec[j] + (dataArr[i][j] * (labelArr[i] - yPredArr[i][0]))
            
            if j != 0:
                gradientVec[j] = gradientVec[j] - (regConstant * weightArr[j][0])
            
            weightArr[j][0] = weightArr[j][0] + (learnRate * gradientVec[j][0])
    
    return weightArr

#this function takes in a dataset and performes the math to test if it 
def LRTest(dataArr, labelArr, weightArr, dataArrRows, set):
    
    actuallAmmountOfSpam = numberOfSpamInData(labelArr)
    totalEmails = 0
    correctEmails = 0
    predictedAsSpam = 0
    correctSpamEmails = 0


    

    for i in range(0,dataArrRows):
        

        result = np.divide(1,np.add(1,np.exp(np.dot(np.transpose(np.multiply(-1, weightArr)), dataArr[i]))))
        

        totalEmails += 1
        if(result >= 0.5):
            prediction = 1
        else:
            prediction = 0
        
        print("what it predicted: " + str(prediction) + " what it was: " + str(labelArr[i][0]))

        if(prediction == labelArr[i][0]):
            correctEmails += 1
        if(prediction == 1):
            predictedAsSpam += 1
        if(prediction == 1 and labelArr[i][0] == 1):
            correctSpamEmails += 1
        
    finalReport(set, totalEmails, correctEmails, predictedAsSpam, correctSpamEmails, actuallAmmountOfSpam)

#this function processes data form the csv form, trains and then tests it defaults to a 70, 30 split for validation data
#@perams Traincsv is the path to the traiining csv, Testcsv is the path to the test csv, trainonwholeDataset is 0 for traiing only on one dataset with a 70/30 split
# 1 is for traingin on the training set and then testing on the test set, set is for the number enron you are testing
def ProcessDataFromCSV(Traincsv, Testcsv, trainOnWholeDataset, set ):
    dataStuff = createDataArray(Traincsv)
    dataArr = dataStuff[0]
    labelArr = dataStuff[1]


    if(trainOnWholeDataset == 0):
        rows = numberOfRows(Traincsv)
        rowsToTest = int(np.ceil(np.multiply(.7, rows)))
        trainingSet = dataArr[75:rowsToTest + 75, :]
        trainingLabels = labelArr[75:rowsToTest+75]
        validationSet = np.vstack((dataArr[:75, :], dataArr[rowsToTest+75:, : ]))
        validationLabels = np.vstack((labelArr[:75, :], labelArr[rowsToTest+75:, : ]))
    else:
        trainingSet = dataArr
        trainingLabels = labelArr
        dataStuff = createDataArray(Testcsv)
        validationSet = dataStuff[0]
        validationLabels = dataStuff[1]

    weightArr = LRTraining(trainingSet, trainingLabels, .01, 20, .01, trainingSet.shape[1])
    # weightArr = CSVtoWeight(constants.projectDataPath + "storedWeights.csv")

    weightToCSV(weightArr)

    LRTest(validationSet, validationLabels, weightArr, len(validationSet), set)





# ProcessDataFromCSV(constants.enronBerTrainPaths[0], constants.enronBerTestPaths[0], 1, 1)
