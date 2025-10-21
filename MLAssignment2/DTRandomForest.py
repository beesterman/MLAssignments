import constants
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


def hyperparamTesting():
    numOfEstimators = [50,100,200]
    criterion = ["gini", "entropy", "log_loss"]
    maxDepth = [10,20,None]

    inputFile = pd.read_csv(constants.basePath + "finalCsvs/DTTuning.csv", header=None)
    testCSV = open(constants.basePath + "finalCsvs/" + "DTRandomForestTuning.csv", "w")
    testCSV.write("file,numOfEstimators,criteria,maxDepth,accuracy,f1"+ "\n")

    
    i = 0
    for row in inputFile.itertuples(index=False):
        if(i == 0):
            i += 1
            continue

        bestF1 = 0
        writeHold = []

        df = pd.read_csv(row[0], header=None)
        print(row[0])
        trainData = df.iloc[:,:-1]
        trainLabels = df.iloc[:,-1]

        df2 = pd.read_csv(row[0].replace("test", "valid"),header=None)
        validData = df2.iloc[:,:-1]
        validLabels = df2.iloc[:,-1]

        # print(trainData)
        # print(trainLabels)

        for estimatorNum in numOfEstimators:
            for criteria in criterion:
                for mDepth in maxDepth:
                    
                    classifier = RandomForestClassifier(n_estimators=estimatorNum, criterion=criteria,max_depth=mDepth, )
                    classifier.fit(trainData, trainLabels)


                    validPredictLabels = classifier.predict(validData)

                    accuracy = accuracy_score(validLabels, validPredictLabels)
                    f1 = f1_score(validLabels, validPredictLabels)

                    if (f1 > bestF1):
                        bestF1 = f1
                        writeHold = [row[0], estimatorNum, criteria, mDepth, accuracy, f1]
                    
        testCSV.write(str(writeHold[0]) + "," + str(writeHold[1]) + "," + str(writeHold[2]) + "," + str(writeHold[3]) + "," + str(writeHold[4]) + "," + str(writeHold[5]) + "\n")


def testingOnData():

    inputFile = pd.read_csv(constants.basePath + "finalCsvs/DTRandomForestTuning.csv", header=None)
    outfile = open(constants.basePath + "finalCsvs/DTRandomForestTesting.csv", "w")
    outfile.write("file,numOfEstimators,criteria,maxDepth,accuracy,f1"+ "\n")
    

    i = 0
    for row in inputFile.itertuples(index=False):
        if(i == 0):
            i += 1
            continue



        print(row[0])

        df = pd.read_csv(row[0], header=None)
        df2 = pd.read_csv(row[0].replace("train", "valid"), header=None)

        df3 = pd.concat([df, df2], axis=0)
        trainData = df3.iloc[:,:-1]
        trainLabels = df3.iloc[:,-1]


        try:
            depth = int(row[3])
        except: 
            depth = None
        

        classifier = RandomForestClassifier(n_estimators=int(row[1]), criterion=row[2],max_depth=depth)
        classifier.fit(trainData, trainLabels)    


        df4 = pd.read_csv(row[0].replace("train", "test"), header=None)
        testData = df4.iloc[:,:-1]
        testLabels = df4.iloc[:,-1]
        predictedLabels = classifier.predict(testData)

        accuracy = accuracy_score(testLabels, predictedLabels)
        f1 = f1_score(testLabels, predictedLabels)

        outfile.write(row[0] + "," + row[1] + "," + row[2] + "," + row[3] + "," + str(accuracy) + "," + str(f1) + "\n")




# hyperparamTesting()
# testingOnData()