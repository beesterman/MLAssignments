projectDataPath = "./projectData/"
originalDataPath = "./originalData/"
originalDataTrainingPaths = [
        (originalDataPath + "enron1/train/spam"),
        (originalDataPath + "enron1/train/ham"),
        (originalDataPath + "enron2/train/spam"),
        (originalDataPath + "enron2/train/ham"),
        (originalDataPath + "enron4/train/spam"),
        (originalDataPath + "enron4/train/ham")
        ]
originalDataTestPaths = [
        (originalDataPath + "enron1/test/spam"),
        (originalDataPath + "enron1/test/ham"),
        (originalDataPath + "enron2/test/spam"),
        (originalDataPath + "enron2/test/ham"),
        (originalDataPath + "enron4/test/spam"),
        (originalDataPath + "enron4/test/ham")
        ]
completeVocabPath = (projectDataPath + "completeVocab.csv")