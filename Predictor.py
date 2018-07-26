import time
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources
from sklearn.ensemble import RandomForestClassifier


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
    def __init__(self, trainX, trainY, testX, testY=None, validX=None, validY=None, noOfClasses=None, coresToUse=6):
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY
        self.noOfFeatures = self.trainX.shape[1]
        self.noOfClasses = noOfClasses
        self.validX = validX
        self.validY = validY
        self.coresToUse = coresToUse

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

    def classifyByKnn(self, k, returnPredictions=True, returnAccuracy=False, printProgress=True):
        neigh = KNeighborsClassifier(n_neighbors=k, algorithm="auto", n_jobs=self.coresToUse)
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

    # This method is based off the template from the sample code of the book Tensorflow for Deep Learning
    def classifyByLogRegRiseOrFall(self, name, n_steps, learningRate):
        # Generate tensorflow graph
        with tf.name_scope("placeholders"):
            tfTrainX = tf.placeholder(tf.float32, self.trainX.shape)
            tfTrainY = tf.placeholder(tf.float32, self.trainY.shape)
            tfTestX = tf.placeholder(tf.float32, self.testX.shape)
        with tf.name_scope("weights"):
            W = tf.Variable(tf.random_normal((self.noOfFeatures, 1)))
            b = tf.Variable(tf.random_normal((1,)))
        with tf.name_scope("prediction"):
            y_logit = tf.squeeze(tf.matmul(self.trainX, W) + b)
            # the sigmoid gives the class probability of 1
            y_one_prob = tf.sigmoid(y_logit)
            # Rounding P(y=1) will give the correct prediction.
            y_pred = tf.round(y_one_prob)

        with tf.name_scope("loss"):
            # Compute the cross-entropy term for each datapoint
            entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=self.trainY)
            # Sum all contributions
            l = tf.reduce_sum(entropy)
        with tf.name_scope("optim"):
            train_op = tf.train.AdamOptimizer(learningRate).minimize(l)

        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", l)
            merged = tf.summary.merge_all()

        train_writer = tf.summary.FileWriter('/predcloseprice/' + name, tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # Train model
            for i in range(n_steps):
                feed_dict = {tfTrainX: self.trainX, tfTrainY: self.trainY}
                _, summary, loss = sess.run([train_op, merged, l], feed_dict=feed_dict)
                print("loss: %f" % loss)
                train_writer.add_summary(summary, i)

            # Make Predictions
            y_pred_train = sess.run(y_pred, feed_dict={tfTrainX: self.trainX})
            y_pred_test = sess.run(tf.round(tf.sigmoid(tf.squeeze(tf.matmul(self.testX, W) + b))),
                                   feed_dict={tfTestX: self.testX})

        score = accuracy_score(self.trainY, y_pred_train)
        print("Training Set Accuracy: %f" % score)
        score = accuracy_score(self.testY, y_pred_test)
        print("Test Set Accuracy: %f" % score)

        train_writer.close()

    # Note that tensor_forest is not supported on windows in the build of tensorflow I used in this project, so this method not well tested
    def classifyByTFRandomForest(self, noOfEpochs, noOfTrees, maxNoOfNodes):
        if self.noOfClasses is None:
            print("Warning: No of classes must be defined in constructor")
        else:
            with tf.name_scope("placeholders"):
                X = tf.placeholder(tf.float32, shape=[None, self.noOfFeatures])
                Y = tf.placeholder(tf.int32, shape=[None])

            with tf.name_scope("forest"):
                hParams = tensor_forest.ForestHParams(num_classes=self.noOfClasses, num_features=self.noOfFeatures,
                                                      num_trees=noOfTrees, max_nodes=maxNoOfNodes).fill()
                forestGraph = tensor_forest.RandomForestGraphs(hParams)

            with tf.name_scope("optimisers"):
                trainOp = forestGraph.training_graph(X, Y)
                lossOp = forestGraph.training_loss(X, Y)

            with tf.name_scope("accuracy"):
                inferOp, _, _ = forestGraph.inference_graph(X)
                correctPrediction = tf.equal(tf.arg_max(inferOp, 1), tf.cast(Y, tf.int64))
                accuracyOp = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))

            initVars = tf.group(tf.global_variables_initializer, resources.shared_resources())

            losses = []

            with tf.Session() as sess:
                sess.run(initVars)
                for i in range(1, noOfEpochs + 1):
                    _, l = sess.run([trainOp, lossOp],
                                    feed_dict={X: self.trainX, Y: tf.cast(self.trainY, tf.int32)})
                    if i % 50 == 0 or i == 1:
                        acc = sess.run(accuracyOp,
                                       feed_dict={X: self.trainX, Y: tf.cast(self.trainY, tf.int32)})
                        print('Step %i, Loss: %f, Acc: %f' % (i, l, acc))

                    losses.append(l)

                trainAccs = sess.run(accuracyOp, feed_dict={X: self.trainX, Y: tf.cast(self.trainY, tf.int32)})
                print("Training accuracy:", trainAccs)
                if self.validX is not None:
                    validAccs = sess.run(accuracyOp, feed_dict={X: self.validX, Y: tf.cast(self.validY, tf.int32)})
                    print("Validation accuracy:", validAccs)
                testAccs = sess.run(accuracyOp, feed_dict={X: self.trainX, Y: tf.cast(self.trainY, tf.int32)})
                print("Test accuracy:", testAccs)

            return losses

    def classifyBySKLRandomForestInRange(self, ks, change, noOfTrees=10, maxFeaturesPerTree="auto", minDepth=1, seed=0,
                                         returnPredictions=False, returnAccuracy=True, printProgress=True,
                                         printTime=True, graphIt=True, graphTitle=""):
        results = []
        for i in ks:
            if change == "noOfTrees":
                print('Trying noOfTrees=%s' % i)
                noOfTrees = i
            elif change == "maxFeaturesPerTree":
                print('Trying maxFeaturesPerTree=%s' % i)
                maxFeaturesPerTree = i
            elif change == "minDepth":
                print('Trying minDepth=%s' % i)
                minDepth = i
            elif change == "seed":
                print('Trying seed=%s' % i)
                seed = i
            else:
                print("Variable to change is unrecognised, terminating experiment")
                break
            start = time.time()
            result = self.classifyBySKLRandomForest(noOfTrees=noOfTrees, maxFeaturesPerTree=maxFeaturesPerTree,
                                                    minDepth=minDepth, seed=seed, printProgress=False,
                                                    returnPredictions=returnPredictions,
                                                    returnAccuracy=returnAccuracy, predictTest=False)
            if printTime:
                print('Took %.1f seconds.' % (time.time() - start))
            if printProgress:
                if returnPredictions and returnAccuracy:
                    print("Gave valid accs: %.4f%%" %
                          result[1])
                else:
                    print("Gave valid accs: %.4f%%" %
                          result)
            results.append(result)
        if graphIt:
            if graphTitle != "":
                plt.title(graphTitle)
            else:
                plt.title("Random Forest classification changing %s" % change)
            plt.xlabel(change)
            plt.ylabel("Accuracy on validation set (%)")
            y = []
            for i in range(0, len(results)):
                if returnPredictions:
                    y.append(results[i][1])
                else:
                    y.append(results[i])
            plt.plot(ks, y)
            plt.show()
        return results

    def classifyBySKLRandomForest(self, noOfTrees=10, maxFeaturesPerTree="auto", minDepth=1, seed=0, printProgress=True,
                                  returnAccuracy=True, returnPredictions=False, predictTest=True):
        clf = RandomForestClassifier(n_estimators=noOfTrees, max_features=maxFeaturesPerTree,
                                     n_jobs=self.coresToUse, random_state=seed, min_samples_leaf=minDepth)
        if printProgress:
            print("Generating forest...")
        clf.fit(self.trainX, self.trainY)
        if returnPredictions:
            if printProgress:
                print("Making predictions...")
            if predictTest:
                prediction = clf.predict(self.testX)
            else:
                prediction = clf.predict(self.validX)
            if not returnAccuracy:
                return prediction
            else:
                if printProgress:
                    print("Calculating accuracy")
                if predictTest:
                    acc = accuracy_score(self.testY, prediction) * 100
                else:
                    acc = accuracy_score(self.validY, prediction) * 100
                if returnPredictions:
                    return [prediction, acc]
                else:
                    return acc
        else:
            if printProgress:
                print("Making predictions and calculating accuracy...")
            if predictTest:
                accs = clf.score(self.testX, self.testY) * 100
            else:
                accs = clf.score(self.validX, self.validY) * 100
            return accs
