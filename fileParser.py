#%%
from constants import originalDataPath, projectDataPath, originalDataTrainingPaths, completeVocabPath, originalDataTestPaths, enronVocabPaths
import os
import pandas as pd

# %%
#stopword list from https://gist.github.com/sebleier/554280#file-nltk-s-list-of-english-stopwords also includes words from email data
stopwords = ["to","cc","am","pm","subject","i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
punctuation = ['+','.','$','%','^','*','#','±','??','??????','????','°','???','','@','.', '!', '?', ',', ':', ';', '(', ')', '[', ']', '{', '}', "'", '"', '-', '—', '/', '...', '&']

#%%
#function pulls in all of the training data and creates one massive completeVocab.csv
def constructVocabulary():
    #initalize empty dictionary for vocabulary and opening complete vocab file
    vocabularyFile = open((projectDataPath + "completeVocab.csv"), "w")
    vocabulary = dict()
    #itterate over the directories in originalDataTrainingPaths and 
    for currentDirecotry in originalDataTrainingPaths:
        # gets list of current files in the directory
        listOfFilesInCurrDirecotry = os.listdir(currentDirecotry)

        for file in listOfFilesInCurrDirecotry:
            currentOpenFile = open(currentDirecotry + "/" + file)
            singleLine = " "
            while singleLine != '':
                
                #some emails have non unicode chars, this skips it
                try:
                    singleLine = currentOpenFile.readline()
                except:
                    continue

                singleLineTokens = cleanSingleLine(singleLine)
                for token in singleLineTokens:
                    #this gets rid of all the non utf-8 charecters
                    try:
                        token.encode("ascii", "strict")
                    except:
                        continue
                    
                    if(token in vocabulary):
                        vocabulary[token] += 1
                    else:
                        vocabulary[token] = 1

    print("complete vocab size: ",vocabulary.__len__())
    for token in vocabulary:
        vocabularyFile.write(token + ",")
    
    #parsing the vocab for individual enrons
    enron = 0
    for currentDirecotry in originalDataTrainingPaths:
        if ("/enron1/train/spam" in currentDirecotry):
            currentFile = open(projectDataPath + "enron1Vocab.csv", "w")
            enron = 1
            vocabulary = dict()
        elif ("/enron2/train/spam" in currentDirecotry):
            currentFile = open(projectDataPath + "enron2Vocab.csv", "w")  
            enron = 2
            vocabulary = dict()
        elif ("/enron4/train/spam" in currentDirecotry):
            currentFile = open(projectDataPath + "enron4Vocab.csv", "w")
            enron = 4
            vocabulary = dict()
        elif ("/enron1/train/ham" in currentDirecotry):
            currentFile = open(projectDataPath + "enron1Vocab.csv", "a")
            enron = 1
        elif ("/enron2/train/ham" in currentDirecotry):
            currentFile = open(projectDataPath + "enron2Vocab.csv", "a")
            enron = 2
        elif ("/enron4/train/ham" in currentDirecotry):
            currentFile = open(projectDataPath + "enron4Vocab.csv", "a")
            enron = 4
            
        listOfFilesInCurrDirecotry = os.listdir(currentDirecotry)

        for file in listOfFilesInCurrDirecotry:
            currentOpenFile = open(currentDirecotry + "/" + file)
            singleLine = " "
            while singleLine != '':
                
                #some emails have non unicode chars, this skips it
                try:
                    singleLine = currentOpenFile.readline()
                except:
                    continue

                if singleLine == '':
                    continue

                singleLineTokens = cleanSingleLine(singleLine)
                for token in singleLineTokens:
                    if(token in vocabulary):
                        vocabulary[token] += 1
                    else:
                        vocabulary[token] = 1

        if("ham" in currentDirecotry):
            print("enron" + str(enron) + " vocab size:",vocabulary.__len__())
            for token in vocabulary:
                currentFile.write(token + ",")
        
        



#%%
#function to take in a single sentnece and return an array of tokens.
#@perams: a single line of text as a string
#@returns: an array of tokens with cleaned inputs
def cleanSingleLine(singleLine):
    #send all chars to lower case and performs other replacements on the sentence
    singleLine = singleLine.lower()
    singleLine = singleLine.replace("\n", "")
    singleLine = singleLine.replace("subject:", "")

    # splits the line into an array then checks to see if it is punctuation, a stopword, or a numeric value
    singleLineTokens = singleLine.split(" ")

    # making a bastardized for loop from c++
    lengthOfSingleToken = len(singleLineTokens)
    i = 0
    while(i != lengthOfSingleToken):
        word = singleLineTokens[i]

        if(word in punctuation):
            singleLineTokens.pop(singleLineTokens.index(word))
            i -= 1
            lengthOfSingleToken -= 1
        elif(word in stopwords):
            singleLineTokens.pop(singleLineTokens.index(word))
            i -= 1
            lengthOfSingleToken -= 1
        elif (word.isnumeric()):
            singleLineTokens.pop(singleLineTokens.index(word))
            i -= 1
            lengthOfSingleToken -= 1
        elif ('?' in word):
            singleLineTokens.pop(singleLineTokens.index(word))
            i -= 1
            lengthOfSingleToken -= 1
        i += 1
    
    return singleLineTokens

#this function takes the complete vocab and produces a csv with every email as a row and the count of each word in each email.
def BOWDataCreator():
    
    # creating an array with all of the testing and training data paths
    allPaths = [originalDataTrainingPaths, originalDataTestPaths]

    for testAndTrainPaths in allPaths:
        for currentDirecotry in testAndTrainPaths:
            if ("/enron1/train" in currentDirecotry):
                currentFile = open(projectDataPath + "enron1_bow_train.csv", "a")
            elif ("/enron2/train" in currentDirecotry):
                currentFile = open(projectDataPath + "enron2_bow_train.csv", "a")   
            elif ("/enron4/train" in currentDirecotry):
                currentFile = open(projectDataPath + "enron4_bow_train.csv", "a")
            elif ("/enron1/test" in currentDirecotry):
                currentFile = open(projectDataPath + "enron1_bow_test.csv", "a")
            elif ("/enron2/test" in currentDirecotry):
                currentFile = open(projectDataPath + "enron2_bow_test.csv", "a")
            elif ("/enron4/test" in currentDirecotry):
                currentFile = open(projectDataPath + "enron4_bow_test.csv", "a")
            
            if ('1' in currentDirecotry):
                vocabDict = createVocabDict(1)
                individualVocabDict = createVocabDict(1)
            elif ('2' in currentDirecotry):
                vocabDict = createVocabDict(2)
                individualVocabDict = createVocabDict(2)
            elif ('4' in currentDirecotry):
                vocabDict = createVocabDict(3)
                individualVocabDict = createVocabDict(3)

            # gets list of current files in the directory
            listOfFilesInCurrDirecotry = os.listdir(currentDirecotry)
            print(currentDirecotry, listOfFilesInCurrDirecotry.__len__())
            #openes a individual file
            for file in listOfFilesInCurrDirecotry:
                currentOpenFile = open(currentDirecotry + "/" + file)
                singleLine = " "
                while singleLine != '':
                    #some emails have non unicode chars, this skips it
                    try:
                        singleLine = currentOpenFile.readline()
                    except:
                        continue

                    cleanedLine = cleanSingleLine(singleLine)
                    
                    #getting all of the tokens and counting the number of times they appear in the file
                    for token in cleanedLine:

                        #this gets rid of all the non utf-8 charecters
                        try:
                            token.encode("ascii", "strict")
                        except:
                            continue

                        if(token in vocabDict):
                            individualVocabDict[token] += 1

                #writing the number of each word into the file
                for word in individualVocabDict:
                    currentFile.write(str(individualVocabDict[word]) + ",")
                if("ham" in currentDirecotry):
                    currentFile.write(str(0))
                if("spam" in currentDirecotry):
                    currentFile.write(str(1))
                currentFile.write("\n")

                #reseting the number of words back to zero for use in the next file
                for word in individualVocabDict:
                    individualVocabDict[word] = 0

            currentFile.close()
                    



# this function returns a dictionary of all the unique vocab words all initialized to 0
#@peram: takes in a number 0-3 coresponging to completeVocab, enron1vocab, enron2vocab, and enron3vocab
def createVocabDict(set):
    vocabDict = dict()
    
    match set:
        case 0: 
            vocabFile = open(completeVocabPath)
            onlyLine = vocabFile.readline()
            splitOnlyLine = onlyLine.split(",")
        case 1:
            vocabFile = open(enronVocabPaths[set-1])
            onlyLine = vocabFile.readline()
            splitOnlyLine = onlyLine.split(",")
        case 2:
            vocabFile = open(enronVocabPaths[set-1])
            onlyLine = vocabFile.readline()
            splitOnlyLine = onlyLine.split(",")
        case 3:
            vocabFile = open(enronVocabPaths[set-1])
            onlyLine = vocabFile.readline()
            splitOnlyLine = onlyLine.split(",")


    for token in splitOnlyLine:
        #this gets rid of all the non utf-8 charecters
        try:
            token.encode("ascii", "strict")
        except:
            continue

        vocabDict[token] = 0
            
    return vocabDict

#this function creates empty BOW files with the complete header
def createBOWFiles():
    names = ["enron1_bow_train.csv","enron2_bow_train.csv", "enron4_bow_train.csv","enron1_bow_test.csv","enron2_bow_test.csv","enron4_bow_test.csv"]
    for name in names:
        currentFile = open(projectDataPath + name, "w")
        
        if ('1' in name):
            vocabDict = createVocabDict(1)
        elif ('2' in name):
            vocabDict = createVocabDict(2)
        elif ('4' in name):
            vocabDict = createVocabDict(3)

        for token in vocabDict:
            if(token == ""):
                continue
            #this gets rid of all the non utf-8 charecters
            try:
                token.encode("ascii", "strict")
            except:
                continue

            currentFile.write(token + ",")
        currentFile.write("label")
        currentFile.write("\n")
        currentFile.close()
#this function creates empty Bernouli files with the complete header
def createBernouliFiles():
    names = ["enron1_bernoulli_train.csv", "enron1_bernoulli_test.csv","enron2_bernoulli_train.csv","enron2_bernoulli_test.csv","enron4_bernoulli_train.csv","enron4_bernoulli_test.csv"]
    vocabDict = createVocabDict(0)
    for name in names:
        currentFile = open(projectDataPath + name, "w")

        if ('1' in name):
            vocabDict = createVocabDict(1)
        elif ('2' in name):
            vocabDict = createVocabDict(2)
        elif ('4' in name):
            vocabDict = createVocabDict(3)


        for token in vocabDict:
            if(token == ""):
                continue
            #this gets rid of all the non utf-8 charecters
            try:
                token.encode("ascii", "strict")
            except:
                continue
            currentFile.write(token + ",")
        currentFile.write("label")
        currentFile.write("\n")
        currentFile.close()
#this creates the populated bernouli data files
def bernouliDataCreator():
    
    # creating an array with all of the testing and training data paths
    allPaths = [originalDataTrainingPaths, originalDataTestPaths]

    for testAndTrainPaths in allPaths:
        for currentDirecotry in testAndTrainPaths:
            if ("/enron1/train" in currentDirecotry):
                currentFile = open(projectDataPath + "enron1_bernoulli_train.csv", "a")
            elif ("/enron2/train" in currentDirecotry):
                currentFile = open(projectDataPath + "enron2_bernoulli_train.csv", "a")   
            elif ("/enron4/train" in currentDirecotry):
                currentFile = open(projectDataPath + "enron4_bernoulli_train.csv", "a")
            elif ("/enron1/test" in currentDirecotry):
                currentFile = open(projectDataPath + "enron1_bernoulli_test.csv", "a")
            elif ("/enron2/test" in currentDirecotry):
                currentFile = open(projectDataPath + "enron2_bernoulli_test.csv", "a")
            elif ("/enron4/test" in currentDirecotry):
                currentFile = open(projectDataPath + "enron4_bernoulli_test.csv", "a")


            if ('1' in currentDirecotry):
                vocabDict = createVocabDict(1)
                individualVocabDict = createVocabDict(1)
            elif ('2' in currentDirecotry):
                vocabDict = createVocabDict(2)
                individualVocabDict = createVocabDict(2)
            elif ('4' in currentDirecotry):
                vocabDict = createVocabDict(3)
                individualVocabDict = createVocabDict(3)


            # gets list of current files in the directory
            listOfFilesInCurrDirecotry = os.listdir(currentDirecotry)
            print(currentDirecotry, listOfFilesInCurrDirecotry.__len__())
            #openes a individual file
            for file in listOfFilesInCurrDirecotry:
                currentOpenFile = open(currentDirecotry + "/" + file)
                singleLine = " "
                while singleLine != '':
                    #some emails have non unicode chars, this skips it
                    try:
                        singleLine = currentOpenFile.readline()
                    except:
                        continue

                    cleanedLine = cleanSingleLine(singleLine)
                    
                    #getting all of the tokens and counting the number of times they appear in the file
                    for token in cleanedLine:
                        if(token in vocabDict):
                            individualVocabDict[token] = 1

                #writing the number of each word into the file
                for word in individualVocabDict:
                    currentFile.write(str(individualVocabDict[word]) + ",")
                if("ham" in currentDirecotry):
                    currentFile.write(str(0))
                if("spam" in currentDirecotry):
                    currentFile.write(str(1))
                currentFile.write("\n")

                #reseting the number of words back to zero for use in the next file
                for word in individualVocabDict:
                    individualVocabDict[word] = 0

            currentFile.close()


# %%
# constructVocabulary()
# BOWDataCreator()
# bernouliDataCreator()
# %%
# createBOWFiles()
# createBernouliFiles()


# %%
