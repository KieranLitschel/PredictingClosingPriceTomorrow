import time
from sklearn.neighbors import KNeighborsClassifier


class Classifier:
    def __init__(self, trainX, trainY, testX, testY=None):
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY

    def classifyByKnnInRange(self, ks, returnPredictions=False, returnAccuracy=True, printProgress=True, printTime=True):
        results = {}
        for i in ks:
            print('Trying k=%s' % i)
            start = time.time()
            result = self.classifyByKnn(i, returnPredictions, returnAccuracy, printProgress=False, coresToUse=4)
            if printTime:
                print('Took %.1f seconds.' % (time.time()-start))
            results[i] = result
            if printProgress:
                if returnPredictions and returnAccuracy:
                    print("k=%s gave accuracy of %.4f%%" % (i, result[i] * 100))
                else:
                    print("k=%s gave accuracy of %.4f%%" % (i, result * 100))
        return results

    def classifyByKnn(self, k, returnPredictions=True, returnAccuracy=False, printProgress=True, coresToUse=-1):
        neigh = KNeighborsClassifier(n_neighbors=k, algorithm="auto", n_jobs=coresToUse) # Using ball_tree as kd_tree runs out of memory
        if printProgress:
            print("Fitting training data...")
        neigh.fit(self.trainX, self.trainY)
        if printProgress:
            print("Predicting test data...")
        if returnPredictions:
            predictedY = neigh.predict(self.testX)
            if not returnAccuracy:
                return predictedY
            else:
                correct = 0
                for i in range(0, len(predictedY)):
                    if predictedY[i] == self.testY[i]:
                        correct += 1
                accuracy = correct / len(predictedY)
                if returnPredictions:
                    return accuracy, predictedY
                else:
                    return accuracy
        else:
            return neigh.score(self.testX, self.testY)
