import Database
from sklearn.neighbors import KNeighborsClassifier


class Classifier:
    def __init__(self, apiKey, pwrd):
        self.db = Database.DBManager(apiKey, pwrd)

    def classifyByKnn(self, k, trainX, trainY, testX, testY=None):
        neigh = KNeighborsClassifier(n_neighbors=k, algorithm="auto", n_jobs=-1)
        print("Fitting training data...")
        neigh.fit(trainX, trainY)
        print("Predicting test data...")
        predictedY = neigh.predict(testX)
        if testY is None:
            return predictedY
        else:
            correct = 0
            for i in range(0, len(predictedY)):
                if predictedY[i] == testY[i]:
                    correct += 1
            accuracy = correct / len(predictedY)
            return accuracy, predictedY
