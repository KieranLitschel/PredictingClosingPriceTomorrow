import time
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import math
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import KerasSearchCV
import os


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


def graphChangeInAccs(values, results, xlbl, cv=True):
    if cv:
        accuracies = [results[key]['mean'] for key in results.keys()]
        stds = [results[key]['std'] for key in results.keys()]
        fig, ax1 = plt.subplots()
        ax1.plot(values, accuracies, 'r')
        ax1.set_xlabel(xlbl)
        ax1.set_ylabel('Average accuracy using 4-fold CV (%)', color='r')
        ax1.tick_params('y', colors='r')

        ax2 = ax1.twinx()
        ax2.plot(values, stds, 'b')
        ax2.set_ylabel('Standard deviation of accuracy using 4-fold CV (%)', color='b')
        ax2.tick_params('y', colors='b')

        fig.tight_layout()
        fig.set_size_inches(8, 6, forward=True)
        plt.title("Effect of changing " + xlbl)
        plt.show()
    else:
        accuracies = [results[key]['acc'] for key in results.keys()]
        plt.title("Effect of changing " + xlbl)
        plt.xlabel(xlbl)
        plt.ylabel("Accuracy classifying validation set (%)")
        plt.plot(values, accuracies)
        plt.show()


def continue_search(path=os.path.dirname(os.path.abspath(__file__)), pickle_path="KSCV.pickle"):
    KSCV = KerasSearchCV.Host(path, pickle_path, True)
    if KSCV.file_found:
        KSCV.start()
        return KSCV.getResults()
    else:
        return None


def RandomSearchCVToCSV(RSCV):
    line = ""
    for parameter in RSCV.grid_scores_[0].parameters.keys():
        line += str(parameter) + ","
    lines = line + "\n"
    for iteration in RSCV.grid_scores_:
        line = ""
        for parameter in iteration.parameters.keys():
            line += str(iteration.parameters[parameter]) + ","  # params
        for accs in iteration.cv_validation_scores:
            line += str(accs) + ","  # accs
        line += iteration.__str__().split(" std: ")[1].split(",")[0] + ","  # std acc
        line += str(iteration.mean_validation_score)  # mean acc
        lines += line + "\n"
    with open('RSCVResults.csv', 'w') as f:
        f.write(lines)
        f.close()


class Classifier:
    def __init__(self, trainX, trainY, testX=None, testY=None, validX=None, validY=None, noOfClasses=None, n_jobs=6,
                 usePOfData=100):
        if usePOfData == 100:
            self.trainX = trainX
            self.trainY = trainY
            self.testX = testX
            self.testY = testY
            self.validX = validX
            self.validY = validY
        else:
            np.random.seed(0)
            cTrain = np.random.choice(len(trainX), size=int(math.floor(len(trainX) * (usePOfData / 100))),
                                      replace=False)
            self.trainX = trainX[cTrain]
            self.trainY = trainY[cTrain]
            if testX is not None:
                cTest = np.random.choice(len(testX), size=int(math.floor(len(testX) * (usePOfData / 100))),
                                         replace=False)
                self.testX = testX[cTest]
                if testY is not None:
                    self.testY = testY[cTest]
            if validX is not None:
                cValid = np.random.choice(len(validX), size=int(math.floor(len(validX) * (usePOfData / 100))),
                                          replace=False)
                self.validX = validX[cValid]
                self.validY = validY[cValid]
        self.noOfFeatures = self.trainX.shape[1]
        self.noOfClasses = noOfClasses
        self.n_jobs = n_jobs


class KNNClassifierMethods(Classifier):

    def __init__(self, trainX, trainY, testX, testY, n_jobs=6):
        Classifier.__init__(self, trainX, trainY, testX, testY, n_jobs=n_jobs)

    def classifyByKnnInRange(self, ks, returnPredictions=False, returnAccuracy=True, printProgress=True, printTime=True,
                             graphIt=True, graphTitle=""):
        results = {}
        for i in ks:
            print('Trying k=%s' % i)
            start = time.time()
            result = self.classifyByKnn(i, returnPredictions, returnAccuracy, printProgress=False, n_jobs=4)
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
        neigh = KNeighborsClassifier(n_neighbors=k, algorithm="auto", n_jobs=self.n_jobs)
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


class LogisticRegressionClassiferMethods(Classifier):
    def __init__(self, trainX, trainY, testX, testY, noOfClasses, n_jobs=6):
        Classifier.__init__(self, trainX, trainY, testX, testY, noOfClasses=noOfClasses, n_jobs=n_jobs)

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


class RandomForestClassifierMethods(Classifier):

    def __init__(self, trainX, trainY, testX, testY, validX=None, validY=None, noOfClasses=None, n_jobs=6):
        Classifier.__init__(self, trainX, trainY, testX, testY, validX, validY, noOfClasses, n_jobs)

    # Note that tensor_forest is not supported on windows in the build of tensorflow I used in this project, so this method not well tested
    def classifyByTFRandomForest(self, noOfEpochs, n_estimators, maxNoOfNodes):
        if self.noOfClasses is None:
            print("Warning: No of classes must be defined in constructor")
        else:
            with tf.name_scope("placeholders"):
                X = tf.placeholder(tf.float32, shape=[None, self.noOfFeatures])
                Y = tf.placeholder(tf.int32, shape=[None])

            with tf.name_scope("forest"):
                hParams = tensor_forest.ForestHParams(num_classes=self.noOfClasses, num_features=self.noOfFeatures,
                                                      num_trees=n_estimators, max_nodes=maxNoOfNodes).fill()
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

    def classifyBySKLRandomForestInRange(self, ks, change, n_estimators=10, max_features="auto", min_samples_leaf=1,
                                         seed=0,
                                         minSamplesSplit=2, bootstrap=True, maxDepth=None, returnPredictions=False,
                                         returnAccuracy=True, printProgress=True, printTime=True, graphIt=True,
                                         graphTitle=""):
        results = []
        for i in ks:
            if change == "n_estimators":
                print('Trying n_estimators=%s' % i)
                n_estimators = i
            elif change == "max_features":
                print('Trying max_features=%s' % i)
                max_features = i
            elif change == "min_samples_leaf":
                print('Trying min_samples_leaf=%s' % i)
                min_samples_leaf = i
            elif change == "seed":
                print('Trying seed=%s' % i)
                seed = i
            elif change == "minSamplesSplit":
                print('Trying minSampleSplit=%s' % i)
                seed = i
            elif change == "maxDepth":
                print('Trying maxDepth=%s' % i)
                seed = i
            else:
                print("Variable to change is unrecognised, terminating experiment")
                break
            start = time.time()
            result = self.classifyBySKLRandomForest(n_estimators=n_estimators, max_features=max_features,
                                                    min_samples_leaf=min_samples_leaf, seed=seed, printProgress=False,
                                                    returnPredictions=returnPredictions,
                                                    returnAccuracy=returnAccuracy, predictTest=False,
                                                    minSamplesSplit=minSamplesSplit, bootstrap=bootstrap,
                                                    maxDepth=maxDepth)
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
            if graphIt and len(results) > 1:
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
                plt.plot(ks[0:len(results)], y)
                plt.show()
        return results

    def classifyBySKLRandomForest(self, n_estimators=10, max_features="auto", min_samples_leaf=1, seed=0,
                                  printProgress=True,
                                  returnAccuracy=True, returnPredictions=False, predictTest=True, minSamplesSplit=2,
                                  bootstrap=True, maxDepth=None):
        clf = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features,
                                     n_jobs=self.n_jobs, random_state=seed, min_samples_leaf=min_samples_leaf,
                                     min_samples_split=minSamplesSplit, bootstrap=bootstrap, max_depth=maxDepth)
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

    def evaluateBySKLRandomForest(self, n_estimators=100, max_features=3, min_samples_leaf=98, seed=0,
                                  getImportances=True):
        print("Running cross-validation...")
        rfc = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features,
                                     min_samples_leaf=min_samples_leaf, random_state=seed)
        scores = cross_val_score(rfc, self.trainX, self.trainY, n_jobs=self.n_jobs, cv=4)
        acc = np.mean(scores)
        std = np.std(scores)
        if getImportances:
            print("Finding feature importances...")
            rfc.n_jobs = self.n_jobs
            rfc.fit(self.trainX, self.trainY)
            featureImportances = rfc.feature_importances_
            return {'scores': scores, 'acc': acc, 'std': std, 'featureImportances': featureImportances}
        else:
            return {'scores': scores, 'acc': acc, 'std': std}

    def random_search(self, n_estimators, max_features, min_samples_leaf, iterations=10, seed=0, cv=4,
                      path=os.path.dirname(os.path.abspath(__file__)), pickle_path="KSCV.pickle"):
        param_grid = {'n_estimators': n_estimators, 'max_features': max_features, 'min_samples_leaf': min_samples_leaf}
        if cv == 1:
            validX = self.validX
            validY = self.validY
        else:
            validX = None
            validY = None
        kscv = KerasSearchCV.Host(path, pickle_path)
        kscv.create_new_sklearn(self.trainX, self.trainY, RandomForestClassifier, "random", param_grid, iterations, cv,
                                self.n_jobs, seed, validX, validY)
        kscv.start()
        return kscv.getResults()


class NeuralNetworkClassifierMethods(Classifier):

    def __init__(self, trainX, trainY, testX=None, testY=None, validX=None, validY=None, threads=2, memory_frac=0.8,
                 usePOfData=100):
        Classifier.__init__(self, trainX, trainY, testX, testY, validX, validY, n_jobs=1, usePOfData=usePOfData)
        self.trainY = keras.utils.to_categorical(self.trainY)
        if validX is not None:
            self.validY = keras.utils.to_categorical(self.validY)
        if testX is not None:
            self.testY = keras.utils.to_categorical(self.testY)
        self.threads = threads
        self.total_memory = memory_frac

    def create_model(self, layers, dropout_rate=0.5, learning_rate=0.001):
        model = keras.Sequential()
        first = True
        i = 0
        for layer in layers:
            if layer.L2:
                reg = keras.regularizers.l2(layer.lmbda)
            else:
                reg = keras.regularizers.l1(layer.lmbda)
            if first:
                model.add(keras.layers.Dense(layer.neurons, activation=layer.activation, kernel_regularizer=reg,
                                             input_shape=(self.trainX.shape[1],), name="Layer-" + str(i)))
                first = False
            else:
                model.add(keras.layers.Dense(layer.neurons, activation=layer.activation, kernel_regularizer=reg,
                                             name="Layer-" + str(i)))
            if dropout_rate != 0:
                model.add(keras.layers.Dropout(dropout_rate, name="dropout-" + str(i)))
            i += 1
        model.add(keras.layers.Dense(4, activation=tf.nn.softmax, name="output"))
        optimizer = keras.optimizers.Adam(lr=learning_rate)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def create_single_layer_model(self, L2, lmbda, neurons, activation, dropout_rate=0.5, learning_rate=0.001):
        layers = [self.CustomLayer(L2, lmbda, neurons, activation)]
        return self.create_model(layers, dropout_rate, learning_rate)

    def random_search_single_layer(self, seed=0, verbose=2, n_iter=5):
        np.random.seed(seed)
        tf.set_random_seed(seed)
        model = keras.wrappers.scikit_learn.KerasClassifier(self.create_single_layer_model, verbose=0)
        batch_sizes = [128, 256, 512, 768, 1024]
        epochs = [10, 50, 100, 200, 300, 400, 500]
        L2s = [True, False]
        lmbdas = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
        neurons = range(2, 2 * self.trainX.shape[1])
        activations = [tf.nn.softmax, tf.nn.sigmoid, tf.nn.tanh, tf.nn.relu]
        param_dist = dict(L2=L2s, lmbda=lmbdas, neurons=neurons, activation=activations, batch_size=batch_sizes,
                          epochs=epochs)
        rscv = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=n_iter, cv=4,
                                  random_state=seed,
                                  verbose=verbose, refit=False)
        rscv.fit(self.trainX, self.trainY)
        return rscv

    def grid_search_single_layer(self, batch_sizes, epochs, L2s, lmbdas, neurons, activations, dropout_rates,
                                 learning_rates, seed=0, cv=4, path=os.path.dirname(os.path.abspath(__file__)),
                                 pickle_path="KSCV.pickle", tensorboard_on=False):
        param_grid = dict(L2=L2s, lmbda=lmbdas, neurons=neurons, activation=activations, batch_size=batch_sizes,
                          epochs=epochs, dropout_rate=dropout_rates, learning_rate=learning_rates)
        KSCV = KerasSearchCV.Host(path, pickle_path, False)
        KSCV.create_new(trainX=self.trainX, trainY=self.trainY, model_constructor=self.create_single_layer_model,
                        search_type="grid", param_grid=param_grid, cv=cv, threads=self.threads,
                        total_memory=self.total_memory, seed=seed, validX=self.validX, validY=self.validY,
                        tensorboard_on=tensorboard_on)
        KSCV.start()
        return KSCV.getResults()

    def sklearn_grid_search_single_layer(self, batch_sizes, epochs, L2s, lmbdas, neurons, activations, dropout_rates,
                                         verbose=2, seed=0):
        try:
            memory_frac = self.total_memory
            keras.backend.clear_session()
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=memory_frac)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            keras.backend.set_session(sess)

            np.random.seed(seed)
            tf.set_random_seed(seed)
            model = keras.wrappers.scikit_learn.KerasClassifier(self.create_single_layer_model, verbose=0)
            param_grid = dict(L2=L2s, lmbda=lmbdas, neurons=neurons, activation=activations, batch_size=batch_sizes,
                              epochs=epochs, dropout_rate=dropout_rates)
            gscv = GridSearchCV(estimator=model, param_grid=param_grid, cv=4, verbose=verbose, refit=False)
            gscv.fit(self.trainX, self.trainY)
            return gscv
        finally:
            keras.backend.clear_session()

    def create_two_layer_model(self, fstNeurons, fstActivation, sndNeurons, sndActivation, L2, lmbda, dropout_rate=0.5,
                               learning_rate=0.001):
        layers = [self.CustomLayer(L2, lmbda, fstNeurons, fstActivation),
                  self.CustomLayer(L2, lmbda, sndNeurons, sndActivation)]
        return self.create_model(layers, dropout_rate, learning_rate)

    def grid_search_two_layer(self, fstNeurons, fstActivations, sndNeurons, sndActivations, L2s, lmbdas, batch_sizes,
                              epochs, dropout_rates, learning_rates, seed=0, cv=4,
                              path=os.path.dirname(os.path.abspath(__file__)), pickle_path="KSCV.pickle",
                              tensorboard_on=False, custom_object_scope=None, histogram_freq=0):
        param_grid = dict(fstNeurons=fstNeurons, fstActivation=fstActivations, sndNeurons=sndNeurons,
                          sndActivation=sndActivations, L2=L2s, lmbda=lmbdas, batch_size=batch_sizes, epochs=epochs,
                          dropout_rate=dropout_rates, learning_rate=learning_rates)
        KSCV = KerasSearchCV.Host(path, pickle_path, False)
        KSCV.create_new(trainX=self.trainX, trainY=self.trainY, model_constructor=self.create_two_layer_model,
                        search_type="grid", param_grid=param_grid, cv=cv, threads=self.threads,
                        total_memory=self.total_memory, seed=seed, validX=self.validX, validY=self.validY,
                        tensorboard_on=tensorboard_on, custom_object_scope=custom_object_scope,
                        histogram_freq=histogram_freq)
        KSCV.start()
        return KSCV.getResults()

    def custom_search_two_layer(self, params, seed=0, cv=4,
                                path=os.path.dirname(os.path.abspath(__file__)), pickle_path="KSCV.pickle",
                                tensorboard_on=False, custom_object_scope=None, histogram_freq=0):
        KSCV = KerasSearchCV.Host(path, pickle_path, False)
        KSCV.create_new(trainX=self.trainX, trainY=self.trainY, model_constructor=self.create_two_layer_model,
                        search_type="custom", param_grid=params, cv=cv, threads=self.threads,
                        total_memory=self.total_memory, seed=seed, validX=self.validX, validY=self.validY,
                        tensorboard_on=tensorboard_on, custom_object_scope=custom_object_scope,
                        histogram_freq=histogram_freq)
        KSCV.start()
        return KSCV.getResults()

    def skelarn_random_search_two_layer(self, seed=0, verbose=2, n_iter=4):
        np.random.seed(seed)
        tf.set_random_seed(seed)
        model = keras.wrappers.scikit_learn.KerasClassifier(self.create_two_layer_model, verbose=0)
        batch_sizes = [128, 256, 512, 768, 1024]
        epochs = [10, 50, 100, 200, 300, 400, 500]
        L2s = [True, False]
        lmbdas = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
        neurons = range(2, 2 * self.trainX.shape[1])
        activations = [tf.nn.leaky_relu, tf.nn.tanh, tf.nn.relu]
        param_dist = dict(fstL2=L2s, fstLmbda=lmbdas, fstNeurons=neurons, fstActivation=activations, sndL2=L2s,
                          sndLmbda=lmbdas, sndNeurons=neurons, sndActivation=activations, batch_size=batch_sizes,
                          epochs=epochs)
        rscv = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=n_iter, cv=4,
                                  random_state=seed, verbose=verbose, refit=False)
        rscv.fit(self.trainX, self.trainY)
        return rscv

    def random_search_two_layer(self, fstNeurons, fstActivations, sndNeurons, sndActivations, L2s, lmbdas, batch_sizes,
                                epochs, dropout_rates, learning_rates, iterations=20, seed=0, cv=4,
                                path=os.path.dirname(os.path.abspath(__file__)), pickle_path="KSCV.pickle",
                                tensorboard_on=False, custom_object_scope=None, histogram_freq=0):
        param_grid = dict(fstNeurons=fstNeurons, fstActivation=fstActivations, sndNeurons=sndNeurons,
                          sndActivation=sndActivations, L2=L2s, lmbda=lmbdas, batch_size=batch_sizes, epochs=epochs,
                          dropout_rate=dropout_rates, learning_rate=learning_rates)
        KSCV = KerasSearchCV.Host(path, pickle_path, False)
        KSCV.create_new(trainX=self.trainX, trainY=self.trainY, model_constructor=self.create_two_layer_model,
                        search_type="random", iterations=iterations, param_grid=param_grid, cv=cv, threads=self.threads,
                        total_memory=self.total_memory, seed=seed, validX=self.validX, validY=self.validY,
                        tensorboard_on=tensorboard_on, custom_object_scope=custom_object_scope,
                        histogram_freq=histogram_freq)
        KSCV.start()
        return KSCV.getResults()

    class CustomLayer:
        def __init__(self, L2, lmbda, neurons, activation):
            self.L2 = L2
            self.lmbda = lmbda
            self.neurons = neurons
            self.activation = activation
