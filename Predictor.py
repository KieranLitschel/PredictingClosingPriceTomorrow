import time
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score


def graphTwoForComparison(ks, fWith, fWithout, addedFeature):
    plt.title("KNN classification with and without " + addedFeature)
    plt.xlabel("Number of neighbours")
    plt.ylabel("Accuracy classifying test set (%)")
    yWith = []
    yWithout = []
    for k in ks:
        yWith.append(fWith[k])
        yWithout.append(fWithout[k])
    plt.plot(ks, yWith, color='blue', label='With ' + addedFeature)
    plt.plot(ks, yWithout, color='red', label='Without ' + addedFeature)
    plt.legend()
    plt.show()


class Classifier:
    def __init__(self, trainX, trainY, testX, testY=None):
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY
        self.noOfFeatures = self.trainX.shape[1]

    def classifyByKnnInRange(self, ks, returnPredictions=False, returnAccuracy=True, printProgress=True, printTime=True,
                             graphIt=True, graphTitle=""):
        results = {}
        for i in ks:
            print('Trying k=%s' % i)
            start = time.time()
            result = self.classifyByKnn(i, returnPredictions, returnAccuracy, printProgress=False, coresToUse=4)
            if printTime:
                print('Took %.1f seconds.' % (time.time() - start))
            results[i] = result
            if printProgress:
                if returnPredictions and returnAccuracy:
                    print("k=%s gave accuracy of %.4f%%" % (i, result[0]))
                else:
                    print("k=%s gave accuracy of %.4f%%" % (i, result))
        if graphIt:
            if graphTitle != "":
                plt.title(graphTitle)
            else:
                plt.title("KNN classification with %s features" % len(self.trainX[0]))
            plt.xlabel("Number of neighbours")
            plt.ylabel("Accuracy classifying test set (%)")
            y = []
            for i in ks:
                y.append(results[i])
            plt.plot(ks, y)
            plt.show()
        return results

    def classifyByKnn(self, k, returnPredictions=True, returnAccuracy=False, printProgress=True, coresToUse=-1):
        neigh = KNeighborsClassifier(n_neighbors=k, algorithm="auto", n_jobs=coresToUse)
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
                accuracy = (correct / len(predictedY)) * 100
                if returnPredictions:
                    return accuracy, predictedY
                else:
                    return accuracy
        else:
            return neigh.score(self.testX, self.testY) * 100

    def classifyByLogRegRiseOrFall(self, name):
        # Generate tensorflow graph
        with tf.name_scope("placeholders"):
            tfTrainX = tf.placeholder(tf.float32, self.trainX.shape)
            tfTrainY = tf.placeholder(tf.float32, self.trainY.shape)
            tfTestX = tf.placeholder(tf.float32, self.testX.shape)
        with tf.name_scope("weights"):
            W = tf.Variable(tf.random_normal((self.noOfFeatures, 1)))
            b = tf.Variable(tf.random_normal((1,)))
        with tf.name_scope("prediction"):
            y_logit = tf.squeeze(tf.matmul(self.testX, W) + b)
            # the sigmoid gives the class probability of 1
            y_one_prob = tf.sigmoid(y_logit)
            # Rounding P(y=1) will give the correct prediction.
            y_pred = tf.round(y_one_prob)

        with tf.name_scope("loss"):
            # Compute the cross-entropy term for each datapoint
            entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=self.testY)
            # Sum all contributions
            l = tf.reduce_sum(entropy)
        with tf.name_scope("optim"):
            train_op = tf.train.AdamOptimizer(.01).minimize(l)

        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", l)
            merged = tf.summary.merge_all()

        train_writer = tf.summary.FileWriter('/predcloseprice/' + name, tf.get_default_graph())

        n_steps = 1000
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # Train model
            for i in range(n_steps):
                feed_dict = {tfTrainX: self.trainX, tfTrainY: self.trainY}
                _, summary, loss = sess.run([train_op, merged, l], feed_dict=feed_dict)
                print("loss: %f" % loss)
                train_writer.add_summary(summary, i)

            # Make Predictions
            y_pred_np = sess.run(y_pred, feed_dict={tfTestX: self.testX})

        score = accuracy_score(self.testY, y_pred_np)
        print("Training Set Accuracy: %f" % score)
