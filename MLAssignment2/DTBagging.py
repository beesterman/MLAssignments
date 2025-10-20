import constants
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd


def hyperparamTesting():
    criterion = ["gini", "entropy", "log_loss"]
    splitter = ["best", "random"]
    depths = [None,1,5,10,15,20]

    testCSV = open(constants.basePath + "finalCsvs/" + "DTTuning.csv", "w")
    testCSV.write("file,criteria,split,depth,accuracy,f1"+ "\n")

    

    for trainVal in constants.testTrainValues:
        bestF1 = 0
        writeHold = []

        filename = constants.originalDataPath + "train_c" + str(trainVal[0]) + "_d" + str(trainVal[1]) + ".csv"
        df = pd.read_csv(filename, header=None)
        print(filename)
        trainData = df.iloc[:,:-1]
        trainLabels = df.iloc[:,-1]

        for criteria in criterion:
            for split in splitter:
                for depth in depths:
                    dt =  DecisionTreeClassifier(criterion=criteria, splitter=split, max_depth=depth)
                    dt.fit(trainData, trainLabels)
                    
                    df2 = pd.read_csv(constants.originalDataPath + "valid_c" + str(trainVal[0]) + "_d" + str(trainVal[1]) + ".csv",header=None)
                    validData = df2.iloc[:,:-1]
                    validLabels = df2.iloc[:,-1]

                    validPredictLabels = dt.predict(validData)

                    accuracy = accuracy_score(validLabels, validPredictLabels)
                    f1 = f1_score(validLabels, validPredictLabels)

                    if (f1 > bestF1):
                        bestF1 = f1
                        writeHold = [filename, criteria, split, depth, accuracy, f1]
                    
        testCSV.write(str(writeHold[0]) + "," + str(writeHold[1]) + "," + str(writeHold[2]) + "," + str(writeHold[3]) + "," + str(writeHold[4]) + "," + str(writeHold[5]) + "\n")


def testingOnData():

    inputFile = pd.read_csv(constants.basePath + "finalCsvs/DTTuning.csv", header=None)
    outfile = open(constants.basePath + "finalCsvs/DTTesting.csv", "w")
    outfile.write("file,criteria,split,depth,accuracy,f1"+ "\n")
    
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

        dt =  DecisionTreeClassifier(criterion=row[1], splitter=row[2], max_depth=depth)
        dt.fit(trainData, trainLabels)

        df4 = pd.read_csv(row[0].replace("train", "test"))
        testData = df3.iloc[:,:-1]
        testLabels = df3.iloc[:,-1]
        predictedLabels = dt.predict(testData)

        accuracy = accuracy_score(testLabels, predictedLabels)
        f1 = f1_score(testLabels, predictedLabels)

        outfile.write(row[0] + "," + row[1] + "," + row[2] + "," + str(depth) + "," + str(accuracy) + "," + str(f1) + "\n")




hyperparamTesting()
testingOnData()