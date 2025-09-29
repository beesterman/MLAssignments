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
enronVocabPaths = [
        (projectDataPath + "enron1Vocab.csv"),
        (projectDataPath + "enron2Vocab.csv"),
        (projectDataPath + "enron4Vocab.csv"),
]
enronBOWTrainPaths = [
        (projectDataPath + "enron1_bow_train.csv"),
        (projectDataPath + "enron2_bow_train.csv"),
        (projectDataPath + "enron4_bow_train.csv"),
]
# 0 coresponds to global and then 1-3 to ther respective enrons
globalVocabSizes = [
        24174,
        8815,
        9070,
        16506
]