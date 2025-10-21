from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score
import constants


# Load MNIST dataset
X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
X = X / 255.0 # Normalize pixel values to [0,1]

# Split into training (60K) and test (10K) sets
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

def trainDT(inFile, X_train, X_test, y_train, y_test):
    print("begining DecisionTree")
    dt = DecisionTreeClassifier(criterion="entropy",max_depth=10)
    dt.fit(X_train, y_train)

    predictedLabels = dt.predict(X_test)

    accuracy = accuracy_score(y_test, predictedLabels)

    inFile.write("DecisionTree," + str(accuracy) + "\n")

def trainBagging(inFile, X_train, X_test, y_train, y_test):
    print("begining Bagging")
    dt = DecisionTreeClassifier(criterion="entropy",max_depth=10)
    
    classifier = BaggingClassifier(estimator=dt,n_estimators=20,bootstrap=True,max_features=3)
    classifier.fit(X_train, y_train)

    predictedLabels = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, predictedLabels)

    inFile.write("Bagging," + str(accuracy) + "\n")


def trainRandomForest(inFile, X_train, X_test, y_train, y_test):
    print("begining RandomForest")
    classifier = RandomForestClassifier(n_estimators=50, criterion="gini",max_depth=20)
    classifier.fit(X_train, y_train)

    predictedLabels = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, predictedLabels)

    inFile.write("RandomForest," + str(accuracy) + "\n")

def trainBoosting(inFile, X_train, X_test, y_train, y_test):
    print("begining Boosting")
    classifier = GradientBoostingClassifier(loss="log_loss", learning_rate=0.1,n_estimators=100)
    classifier.fit(X_train, y_train)

    predictedLabels = classifier.predict(X_test)

    accuracy = accuracy_score(y_test, predictedLabels)

    inFile.write("Boosting," + str(accuracy) + "\n")



inFile = open(constants.basePath + "finalCsvs/mnist.csv", 'w')
inFile.write("classifier,accuracy\n")

trainDT(inFile, X_train, X_test, y_train, y_test)
trainBagging(inFile, X_train, X_test, y_train, y_test)
trainRandomForest(inFile, X_train, X_test, y_train, y_test)
trainBoosting(inFile, X_train, X_test, y_train, y_test)