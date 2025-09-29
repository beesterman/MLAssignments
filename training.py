import numpy as np
import pandas as pd
import constants
from constants import completeVocabPath
from fileParser import createVocabDict


def getLenOfVocab(set):
    match set:
        case 0: 
            vocabFile = open(completeVocabPath)
            onlyLine = vocabFile.readline()
            splitOnlyLine = onlyLine.split(",")
        case 1:
            vocabFile = open(constants.enronVocabPaths[set-1])
            onlyLine = vocabFile.readline()
            splitOnlyLine = onlyLine.split(",")
        case 2:
            vocabFile = open(constants.enronVocabPaths[set-1])
            onlyLine = vocabFile.readline()
            splitOnlyLine = onlyLine.split(",")
        case 3:
            vocabFile = open(constants.enronVocabPaths[set-1])
            onlyLine = vocabFile.readline()
            splitOnlyLine = onlyLine.split(",")
        
    # print("enron" + str(set) + " vocab count: ", splitOnlyLine.__len__() -1)

    return splitOnlyLine.__len__() -1

#this takes in the number of the enron you want the priors for then calculates it
#@peram takes in a integer 1-3 coresponding to enron1-4 
#@return returns an array [X,Y] with X corespongind the spam prior and Y coresponding to not spam prior
def calculatePriors(set):
    match set:
        case 1:
            dataFrame = pd.read_csv(constants.enronBOWTrainPaths[0])
        case 2:
            dataFrame = pd.read_csv(constants.enronBOWTrainPaths[1])
        case 3:
            dataFrame = pd.read_csv(constants.enronBOWTrainPaths[2])
    
    selectedColumn = dataFrame["label"]
    totalDocs = 0
    spamDocs = 0

    #finding the number of total and spam documents
    for data in selectedColumn:
        if data == 1:
            spamDocs += 1
            totalDocs += 1
        else:
            totalDocs += 1
    
    spamPrior = spamDocs / totalDocs
    notSpamPrior = 1 - spamPrior
    
    return [spamPrior, notSpamPrior]

#this takes in the token and calculates the total number of times it occours in the training data
#@peram takes the token and the enron number that you would like (1-3), you also input the class num (1 for spam 0 for nonspam)
#@return returns the total number of times that the token occours in the testing corpus
def getCountOfTokenOccouranceInTrainDataPerClass(tokenind, set, classNum):
    match set:
        case 1:
            BOWFile = open(constants.enronBOWTrainPaths[0])
        case 2:
            BOWFile = open(constants.enronBOWTrainPaths[1])
        case 3:
            BOWFile = open(constants.enronBOWTrainPaths[2])

    singleline = BOWFile.readline()
    totalOccourances = 0

    while singleline != '':

        singleline = BOWFile.readline()
        if singleline == '':
            break
        singleline = singleline.split(",")
        singleline[-1] = singleline[-1].replace("\n","")
        if singleline[-1] == str(classNum):
            totalOccourances += int(singleline[tokenind])
    # print("this is the total occourance: ", totalOccourances)

    return totalOccourances

#this function gives back the total value of all text occourances in teh class corpus. 
#@peram takes in the set num and classnum
#@return returns the total number of words that occour in the corpus of the given class.
def getTotalTextValue(set, classNum):
    totalTextValue = 0
    
    currenFile = open(constants.enronBOWTrainPaths[set-1])
    singleLine = currenFile.readline()
    
    while singleLine != '':
        singleLine = currenFile.readline()
        tokenArr = singleLine.split(",")
        tokenArr[-1] = tokenArr[-1].replace("\n", "")
        
        if tokenArr[-1] == str(classNum):
            for i in range(0,len(tokenArr) -1):
                totalTextValue += int(tokenArr[i])
            # print(totalTextValue)

        
    
    # print(totalTextValue)
    return totalTextValue

#this function gives back the log probability of a certain token based on the class
#@peram takes in the token set number and class number
#@return returns the log probability of that individual token
def calculateProbOfToken(tokenind, set, classNum, lenOfVocab, textOccourPerClass):
    numerator = getCountOfTokenOccouranceInTrainDataPerClass(tokenind, set, classNum) + 1
    denominator = textOccourPerClass + lenOfVocab
    return np.log2(numerator / denominator)

#this function gives back the classification of a single email
#@peram takes in the non split token array of a single email
#@return returns the classification of the given email
def classifySingleEmail(email, set):
    tokenArray = email.split(",")
    actualClass = tokenArray[-1].replace("\n","")
    tokenArray.pop()
    priorArr = calculatePriors(set)

    #calculating the sum of all the log probs of the email for spam
    sumOfIndProbs = 0
    vocabLen = getLenOfVocab(set)
    textOccurPerClass = getTotalTextValue(set, 1)


    print(len(tokenArray))
    for ind in range(0,len(tokenArray)):
        if int(tokenArray[ind]) > 0:
            sumOfIndProbs += np.pow(calculateProbOfToken(ind, set, 1, vocabLen, textOccurPerClass), ind)
    probOfSpam = np.log2(priorArr[0]) + sumOfIndProbs
    print("this is the prob of spam: ", probOfSpam)

    #calculating the sum of all the log probs of a email for not spam
    sumOfIndProbs = 0
    textOccurPerClass = getTotalTextValue(set, 1)

    for ind in range(0,len(tokenArray)):
        if int(tokenArray[ind]) > 0:
            sumOfIndProbs += np.pow(calculateProbOfToken(ind, set, 0, vocabLen, textOccurPerClass), ind)
    probOfNonSpam = np.log2(priorArr[1]) + sumOfIndProbs
    print("this is the prob of ham: ", probOfSpam)

    if probOfSpam > probOfNonSpam:
        return [1,actualClass]
    else:
        return [0,actualClass]
    



def classifyGivenData(bowFile, set):
    currentFile = open(bowFile)
    singleLine = currentFile.readline()

    while True:
        singleLine = currentFile.readline()
        if(singleLine == ''):
            break
        resutlArr = classifySingleEmail(singleLine, 1)
        print("what it predicted: " + str(resutlArr[0]) + " what it was: " + str(resutlArr[1]))
    

# calculatePriors(1)
# getCountOfTokenOccouranceInTrainDataPerClass(0, 1, 0)
# getLenOfVocab(0)
# getLenOfVocab(1)
# getLenOfVocab(2)
# getLenOfVocab(3)
# getTotalTextValue(1, 1)
# calculateProbOfToken(0, 1, 1)
# file = open(constants.enronBOWTrainPaths[0])
# singleline = file.readline()
# singleline = file.readline()
# singleline = file.readline()
# singleline = file.readline()
# print("email class: " ,classifySingleEmail(singleline, 1))
classifyGivenData(constants.enronBOWTrainPaths[0], 1)