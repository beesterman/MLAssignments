import numpy as np
import pandas as pd
import constants
from constants import completeVocabPath
from fileParser import createVocabDict, createVocabArray


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

#this returns a dictionary of the words and thier occourances per set and classnum
#@peram takes the enron number that you would like (1-3), you also input the class num (1 for spam 0 for nonspam)
#@return returns a dictionary with all of the words and thier counts for that class
def getCountOfTokenOccouranceInTrainDataPerClass(set, classNum):
    match set:
        case 1:
            BOWFilePath = constants.enronBOWTrainPaths[0]
        case 2:
            BOWFilePath = constants.enronBOWTrainPaths[1]
        case 3:
            BOWFilePath = constants.enronBOWTrainPaths[2]

    vocabArray = createVocabArray(set)
    vocabDict = createVocabDict(set)
    totalOccourances = 0


    totalOccourances = 0
    BOWFile = open(BOWFilePath)
    singleline = BOWFile.readline()

    while singleline != '':
        singleline = BOWFile.readline()
        if singleline == '':
            break
        singleline = singleline.split(",")
        singleline[-1] = singleline[-1].replace("\n","")
        if singleline[-1] == str(classNum):
            for inc in range(0, len(singleline)- 1):
                vocabDict[vocabArray[inc]] += int(singleline[inc])

    # for i in range(0, 100):
    #     print("this word: " + vocabArray[i] + " had this many occourances: " + str(vocabDict[vocabArray[i]]) + " for this classNum: " + str(classNum))

    return vocabDict

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
def classifySingleEmail(email, set, dictOfSpamOccour, dictOfNotSpamOccour):
    tokenArray = email.split(",")
    actualClass = tokenArray[-1].replace("\n","")
    tokenArray.pop()
    priorArr = calculatePriors(set)
    vocabArr = createVocabArray(set)

    #calculating the sum of all the log probs of the email for spam
    sumOfIndProbs = 0
    vocabLen = getLenOfVocab(set)
    textOccurPerClass = getTotalTextValue(set, 1)

    for ind in range(0,len(tokenArray)):
        if int(tokenArray[ind]) > 0:
            # this is the Tct/sum Tct for all vocab + vocab
            #this gives a 0 for unknown words
            try:
                sumHold = dictOfSpamOccour[vocabArr[ind]]
            except:
                sumHold = 0
            sumOfIndProbs += np.log2(((sumHold + 1)/ (textOccurPerClass + vocabLen)) * int(tokenArray[ind]))
    probOfSpam = np.log2(priorArr[0]) + sumOfIndProbs
    # print("this is the prob of spam: ", probOfSpam)

    #calculating the sum of all the log probs of a email for not spam
    sumOfIndProbs = 0
    
    textOccurPerClass = getTotalTextValue(set, 1)

    for ind in range(0,len(tokenArray)):
        if int(tokenArray[ind]) > 0:
            # this is the Tct/sum Tct for all vocab + vocab
            #this gives a 0 for unknown words
            try:
                sumHold = dictOfNotSpamOccour[vocabArr[ind]]
            except:
                sumHold = 0
            sumOfIndProbs += np.log2(((sumHold + 1)/ (textOccurPerClass + vocabLen)) * int(tokenArray[ind]))
    probOfNonSpam = np.log2(priorArr[1]) + sumOfIndProbs
    # print("this is the prob of ham: ", probOfSpam)

    if probOfSpam > probOfNonSpam:
        return [1,int(actualClass)]
    else:
        return [0,int(actualClass)]
    


#this function takes in a single BOW file and returns the correct and totoal
#@peram takes in the non split token array of a single email
#@return returns the classification of the given email
def classifyGivenData(bowFile, set):
    currentFile = open(bowFile)
    singleLine = currentFile.readline()

    spamOccour = getCountOfTokenOccouranceInTrainDataPerClass(set, 1)
    notSpamOccour = getCountOfTokenOccouranceInTrainDataPerClass(set, 0)

    actualAmmountOfSpam = totalNumberOfSpam(bowFile)

    totalEmails = 0
    correctEmails = 0
    correctSpamEmails = 0
    predictedAsSpam = 0

    while True:
        singleLine = currentFile.readline()
        if(singleLine == ''):
            break
        resultArr = classifySingleEmail(singleLine, 1, spamOccour, notSpamOccour)
        print("what it predicted: " + str(resultArr[0]) + " what it was: " + str(resultArr[1]))
        
        totalEmails += 1

        if(resultArr[0] == resultArr[1]):
            correctEmails += 1
        if(resultArr[0] == 1):
            predictedAsSpam += 1
        if(resultArr[0] == 1 and resultArr[1] == 1):
            correctSpamEmails += 1
    
    print(totalEmails)
    print(correctEmails)
    print(predictedAsSpam)
    print(correctSpamEmails)
    print(actualAmmountOfSpam)
    

    finalReport(set, totalEmails, correctEmails, predictedAsSpam, correctSpamEmails, actualAmmountOfSpam)





    #this ourputs a final report for a given set of inputs
def finalReport(set, totalEmails, correctEmails, predictedAsSpam, correctSpamEmails, actualAmmountOfSpam):
    acuracy = correctEmails / totalEmails
    precision = correctSpamEmails / predictedAsSpam
    recall = correctSpamEmails / actualAmmountOfSpam
    f1Score = 2 * ((precision * recall) / (precision + recall))

    currentFile = open(constants.projectDataPath + "enron" + str(set) + "FinalReport.txt", "w")

    currentFile.write("-------------------------------------\n")
    currentFile.write("accuracy = " + str(acuracy) + "\n")
    currentFile.write("precision = " + str(precision) + "\n")
    currentFile.write("recall = " + str(recall) + "\n")
    currentFile.write("F1 score = " + str(f1Score) + "\n")
    currentFile.write("-------------------------------------\n")



#this takes file in and counts the number of spam
def totalNumberOfSpam(csv):
    currentFile = open(csv)
    singleLine = currentFile.readline()
    count = 0

    while True:
        singleLine = currentFile.readline()
        if singleLine == '':
            break
        if singleLine[-2] == '1':
            count += 1

    return count

# calculatePriors(1)
# getCountOfTokenOccouranceInTrainDataPerClass(1, 0)
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
classifyGivenData(constants.enronBOWTestPaths[2], 3)
# totalNumberOfSpam(constants.enronBOWTestPaths[0])
# finalReport(1,450,437,118,118,131)

