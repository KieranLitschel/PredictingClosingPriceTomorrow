# Table of Contents
* [**Introduction**](https://github.com/KieranLitschel/PredictingClosingPriceTomorrow#introduction)
* [**Progress Log**](https://github.com/KieranLitschel/PredictingClosingPriceTomorrow#progress-log)
  * [**Predicting using KNN**](https://github.com/KieranLitschel/PredictingClosingPriceTomorrow#predicting-using-knn)
  * [**Predicting using logistic regression**](https://github.com/KieranLitschel/PredictingClosingPriceTomorrow#predicting-using-logistic-regression)
  * [**Predicting using random forests**](https://github.com/KieranLitschel/PredictingClosingPriceTomorrow#predicting-using-random-forests)

# Introduction
This is the beginning of a project where I will be using the 20 year history of the current members of the S&P 500 to train a neural network to be able to classify the closing price tomorrow of any given stock into a fixed number of bounds.
# Progress Log
## Predicting using KNN
I have decided to try KNN as it is very simple to train, this will also give me a target to beat when training my neural network. I will classify the the percentage change in closing price tomorrow as one of four of the following classes:

* adjClosePChange <-1%
* -1% <= adjClosePChange < 0%
* 0% <= adjClosePChange < 1%
* 1% <= adjClosePChange

The data with split as 80% for training, and 20% for testing. I will be standardizing all features based on the mean and standard deviation of the training data.
### Results of using features: adjClosePChange, pDiffClose5SMA, pDiffClose8SMA, pDiffClose13SMA
Note that:
* adjClosePChange is the percentage change between the adjusted closing price the day before we are predicting and the day before that one.
* pDiffCloseNSMA, where N is a number, is the percentage difference between the closing price the day before we are predicting, and the N day simple moving average, tracing back N days from the day before the day we are predicting. I chose to use 5, 8, and 13 as they are fibonaci numbers which are commonly compared against each other when assessing a stock.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/KNN/KNN%20with%204%20features.png" alt="KNN with 4 features, plateus at approx 31% accuracy" style="width: 10px;"/>

Conclusions:
* Begins to plateu at about 100 neighbours, trailing off at around 31% accuracy.
* There are 4 classes so we would expect an accuracy of around 25% if it was classifying randomly and there was no pattern in the data, so trailing off at 31% accuracy suggests there is some pattern.
* Considering we are using around 1.9 million samples for training and testing, we should see an increase in accuracy if we increase the number of features.
### Results of adding feature RSI
Note that:
* RSI is the relative strength indicator ([see here](https://www.investopedia.com/terms/r/rsi.asp) for more information.
* I used the features adjClosePChange, pDiffClose5SMA, pDiffClose8SMA, pDiffClose13SMA, and RSI, although on the red line plotted below I show the results from the last experiment for comparison which I did not train using RSI.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/KNN/KNN%20with%205%20features%20(rsi).png" alt="KNN with and without RSI" style="width: 10px;"/>

Conclusions:
* Adding RSI increased accuracy on the test set by 0.26%.
* The low increase in accuracy would suggest the variation of RSI is already captured by other features.
* Will leave it as a feature for now, but will consider applying PCA to dataset when I have more features.
### Results of adding Bollinger Band features
Note that:
* Due to a bug I lost the allocation of the samples the test and training set that I used in the last experiment, so there may be slight variations where comparing this graph to the previous, but nothing significant.
* I added 3 new features, all related to [bollinger bands](https://www.investopedia.com/terms/b/bollingerbands.asp), they are:
  <dl>
    <dt>pDiffCloseUpperBB and pDiffCloseLowerBB</dt>
      <dd>The percentage difference between the closing price on a day and the lower bollinger band and upper bollinger band respectively.</dd>
    <dt>pDiff20SMAAbsBB</dt>
      <dd>This is the difference between the SMA used and the upper bollinger band, to help identify when bollinger bands are squeezing.       </dd>
  </dl>

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/KNN/KNN%20with%208%20features%20(bollinger%20bands).png" alt="KNN with and without RSI" style="width: 10px;"/>

Conclusions:
* Adding the 3 features led to a 0.77% increase in accuracy.
* I was expecting much larger increases in accuracy than I am achieving, as at this rate I don't think I will be able to surpass an accuracy of 40% using KNN, suggesting it is not well suited to the problem. Despite this I will continue to test with KNN while implementing new features, as it is very quick to test and allows me to judge the progress I am making. Once I've generate all the features I have planned I will train a neural network to solve the problem, which I expect will perform much better.
### Results of adding percentage difference between SMAs
I hypothesised that there might be more that could be learnt from looking at the difference between the 5, 8, and 13 day SMAs with each other, so I added 3 features to capture this. Below are the results of adding these 3 features.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/KNN/KNN%20with%2011%20features%20(difference%20between%20SMAs).png" alt="KNN with and without difference between SMAs" style="width: 10px;"/>

It is clear that adding these features does not help improve accuracy, probably because all they have to contribute is captured by the first 3 features I added related to SMAs, hence I will not use them as features in future, but may consider adding them again when applying PCA.
### Results of adding MACD
The [MACD technical indicator](https://www.investopedia.com/terms/m/macd.asp) is used to identify when is best to enter and exit a trade, so I thought it might be able to give some information that will help determine what will happen to the stock tomorrow. The first feature I added was the MACD histogram, which is the difference between the MACD value (the difference between the fast and slow EMA) and the signal (the EMA of the MACD value). This value should help identify crossovers and dramatic rises.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/KNN/KNN%20with%209%20features%20(MACD%20Histogram).png" alt="KNN with and without MACD Histogram" style="width: 10px;"/>

The results of this experiment are very similair to the last one, but note how the two lines do not cross beyond 30 neighbours, and we also see a 0.13% increase in acuracy when k=100. For these reasons I think that the MACD is improving the accuracy of the classifier. However the increase in accuracy is rather disappointing, and I hypothesised that an issue may be that it is impossible for the model to tell which way the MACD histogram is moving and how fast, so I decided to add a feature to represent the difference between the MACD histogram at the current period and at the last period.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/KNN/KNN%20with%2010%20features%20(MACD%20Histogram%20delta).png" alt="KNN with and without MACD Histogram delta" style="width: 10px;"/>

But from looking at the results from the experiment above, there is very little difference between the lines with and without the difference, suggesting there is nothing more to be learned from this feature. Hence I will leave the difference out of future models for the time being.
### Results of adding Stochastic Oscillator
I chose to add the [stochastic oscilator](https://en.wikipedia.org/wiki/Stochastic_oscillator) as up to this point all features have been generated using the adjusted closing price, but the stochastic oscilator calculates momentum using the high, low, and close. The downside to this is that as the Alpha Vantage API does not supply an adjusted high and low price, so I have to use the raw high, low, and close prices. This means that stock splits and dividends will lead the oscilator to give false signals, but these are not frequent events so I hope that this indicator will still be able to give me a boost in accuracy.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/KNN/KNN%20with%2011%20features%20(stochastic%20oscillator).png" alt="KNN with and without stochastic oscillator" style="width: 10px;"/>

We see an increase of accuracy of 0.19% as a result of adding this feature, and it is clear from the graph that it does increase accuracy.
Increasing the number of features has made testing new features significantly slower, as a result in future experiments I will try neighbours in the range 60 to 110, as there appears to be a trend that from 60 onwards accuracy begins to plateu.
### Results of adding ADX
I decided to include [ADX](https://www.investopedia.com/articles/trading/07/adx-trend-indicator.asp) as it is a very popular indicator that is used to determine trends in price movemement and whether they are trailing off or strengthening, which should help indicate which way the closing price will move tomorrow.

Unfortunately the testing time increased again, and it was taking 10 minutes to test for each value of k, hence I have decided I will no longer experiment with KNN, and just focus on implementing features and then training a neural network.

## Predicting using logistic regression
I decided that in the interest of time I will just train the model with the features I have used so far, and once I have trained a model that performs relatively accurately I will investigate whether more features would be beneficial. Next I experimented with logistic regression, and to start with I attempted to train a model to predict whether the stock will be higher or lower at end of the next day, as this was a much simpler problem to solve with logistic regression and would allow me to quickly test how suitable logistic regression is for the problem. I wrote the code in TensorFlow, I chose to use the adam optimiser so that I could train the model quickly with fewer hyperparameters.

But I found that the best accuracy I could achieve was 51%, and bizarrely increasing the value of the learning rate increased the accuracy on the training and test set despite the value of the loss function plateuing at a higher value. It is also worth noting that our accuracy for the 2 class problem is only 1% higher than if we assigned classes at random, and despite our 4 class problem being more complicated we achieved an accuracy 7% higher than if we assigned classes at random. These observations lead me to the hypothesis that our data is not linearly seperable. This would expain why KNN outperformed logistic regression with a harder problem, as KNN is capable of forming non-linear decision boundaries, whereas logistic regression is only capable of forming linear ones.

To test this hypothesis I reran KNN for all features I had selected up to and including the stochastic oscillator using the 2 class problem. The results of which you can see below.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/KNN/predicting%20rise%20or%20fall%20KNN%20with%2011%20features%20(up%20to%20stochastic%20oscillator).png" alt="KNN classifying rise or fall of stock" style="width: 10px;"/>

In previous experiments I increased the number of neighbours each time by 5, but for this experiment I increased the number by 10 each time so the graph isn't as smooth. But it is clear that with more neighbours we could achieve a higher accuracy, with no clear indication that the increase in accuracy is slowing down as the number of neighbours increases. However, it is worth noting that the increase in accuracy each time is not significant, so it is not clear by what margin KNN outperforms logistic regression. So the conclusion from this experiment is that KNN does not give a definitive answer whether the data is linearly seperable or not. Though considering the performance of logistic regression I think it is safe to assume the data is not linearly seperable. Hence I will now experiment with random forests as they are able to cope with non-linearly seperable data.

## Predicting using random forests
### Experiment 1a - changing number of trees
I will be using the implementation of random forests given in scikit learn as the version of TensorFlow I am using does not support random forests on windows. I first experimented with the number of trees, leaving all other parameters as default.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/Random%20Forest/Random%20Forests%20-%20Experiment%201a.png" alt="Random forests tuning the number of trees" style="width: 10px;"/>

The accuracy does not seem to plateu as quickly as we saw for KNN, with increase in accuracy being more gradual and not appearing to tail off. Despite this the time to generate the forests increases rapidly, with it taking 210 seconds for 70 trees, and 300 seconds for 100 trees. Consequently although 100 trees gives a better accuracy I have chosen 70 as the best value for the number of trees in this experiment, with no significant improvement in accuracy after this point.

### Experiment 1b - 70 trees, changing maximum number of features

Next I adjusted max_features, which is the maximum number of features that are considered at each split, the default value was 4, but I tried all possible values (1 to 13). 

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/Random%20Forest/Random%20Forests%20-%20Experiment%201b.png" alt="Random forests tuning the maximum number of features" style="width: 10px;"/>

The results indicate the maximum number of features has little effect on accuracy, but it has a big effect on time, with it taking around 200 seconds to generate the tree and prediction for a maximum of 4 features which gives an accuracy of 31.9679%, and around 100 seconds to do the same with only 1 feature per tree which gives an accuracy of 31.8909%. Hence I will continue my experiments with 1 feature per tree for now.

### Experiment 1c - 1 feature per tree, changing number of trees

Now that generating the tree is much faster, I thought it was worth experimenting to see if I could get a higher accuracy with more trees.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/Random%20Forest/Random%20Forests%20-%20Experiment%201c.png" alt="Random forests 1 feature per tree, tuning the number of trees" style="width: 10px;"/>

I decided 100 trees was the best compromise between accuracy and time, with it giving an accuracy of 32.0716%, an increase of almost 0.2%, whilst increase time to generate and predict by only 40 seconds.

### Experiment 1d - 100 trees, 1 feature per tree, changing minSamplesLeaf

minSamplesLeaf is the minimum number of samples to be at a leaf node. Increasing this should help reduce overfitting of the test set. 

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/Random%20Forest/Random%20Forests%20-%20Experiment%201d.png" alt="Random forests 100 trees, 1 feature per tree, changing minSamplesLeaf" style="width: 10px;"/>

There is no clear trend with the graph, so it is harder to choose the best value. However a value of 50 gives the highest accuracy at 32.9638%, also reducing the time to generate and predict to 105 seconds, considering this improvement I will choose a value of 50 for future experiments. 

### Experiment 2a - 70 trees, changing minSamplesLeaf

It seems odd to me that our best value for maximum number of features is 1, as this suggests that it doesn't matter what features you use to split the data following each split, implying that features are independant of each other. Considering this I have decided it is worth investigating optimising minSamplesLeaf before maximum number of features, and seeing how this impacts when we investigate maximum number of features. 

For this experiment I will be starting with 70 trees as this was the best found for trees without tuning any other hyperparameters.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/Random%20Forest/Random%20Forests%20-%20Experiment%202a.png" alt="Changing maximum number of feature" style="width: 10px;"/>

Clearly a value of 60 performs the best here, increasing accuracy to 32.9231% compared to 31.9679% without, and a decrease in time from 210 seconds to 170 seconds.

### Experiment 2b, 2c, 2d

I have not included the graphs of these experiments here as I found no values that increased performance, but the results of these experiments can be found in the results folder in this repository. The experiments are detailed below:

Experiment 2b - Tried maximum of 70 trees and a value of 60 for minSamplesLeaf, changing the maximum number of features, but a maximum of 3 features had been auto selected in the previous experiments and this turned out to be the best value.

Experiment 2c - Tried a value of 60 for minSamplesLeaf and a maximum of 3 features, changing the number of trees, but 70 trees still appeared to be the optimal number.

Experiment 2d - Tried 70 trees, a maximum of 3 features, and a value of 60 for minSamplesLeaf, changing minSamplesSplit, which is the minimum samples required to split an internal node. But found the default value of 2 performed the best.

### Experiment 1f

I did not include the graph of this experiment in the progress log as I saw no improvement. This experiment was similair to 2d, fixing the number of trees to 100, 1 feature per tree, and a value of 50 for minSamplesLeaf, changing the value of minSamplesSplit. But I found no increase in performance from the default value of 2.

### Experiment 3a

It seems like 32.9638% accuracy is the best I can achieve manually tuning the parameters. So I've decided to apply scikit learn's method for automatic hyperparameter selection called RandomizedSearchCV. I've chosen to start with tuning the number of trees between 1 and 100, the maximum number of features to consider at each split between 1 and 8, and the minimum number of samples to be at a leaf node between 1 and 100, as these seem most likely to have the most impact on accuracy and from my experiments this range of values seems a good compromise between time to generate the tree and accuracy. I've also not investigated the effect of changing the criterion from gini to entropy, so I've decided to tune this as well. To compromise between time and avoiding overfitting, I've decided to use 4-fold cross-validation for evaluating each set of hyperparameters.

In the end I ended up evaluating 55 random settings from the range of hyperparameters. I've decided to evaluate the settings that gave the top 20 values for accuracy to narrow down my range for the hyperparameters.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/Random%20Forest/Random%20Forest%20Experiment%203a%20-%20Top%2020.PNG" alt="Top 20 results from experiment 3a" style="width: 10px;"/>

I seperated min_samples_leaf and n_estimators into 10 bounds, and counted the number of times each hyperparameter occured in each bound in the top 20 accuracies, and weighted each bound by the ratio of the number of times it was sampled in the 55 random settings. The results are displayed in the following graphs.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/Random%20Forest/Random%20Forest%20Experiment%203a%20-%20Top%2020%20in%20Graphs.PNG" alt="Top 20 results from experiment 3a in graphs" style="width: 10px;"/>

A higher score for a bound in relation to other bounds for each hyperparameter suggests it is more significant to accuracy. Considering this it appears that we can reduce the bounds to 50 to 100 estimators (trees), 3 to 6 features at each split, and the minimum number of samples to be at a leaf node to 50 to 100. We will use this range in experiment 3b.

### Experiment 3b

I conducted the next experiment with RanomizedSearchCV with the range of hyperparameters described above. I tried 25 random settings and also included the results from 3a that were in the range being investigated in the results table. This time I looked at the results with the top 5 accuracies.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/Random%20Forest/Random%20Forest%20Experiment%203b%20-%20Top%205.PNG" alt="Top 5 results from experiment 3b" style="width: 10px;"/>

The best accuracy found exceeded that discovered in 3a, suggesting that we are on the right track. I analysed the parameters for the top 5 settings using the same method as before.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/Random%20Forest/Random%20Forest%20Experiment%203b%20-%20Top%205%20in%20Graphs.PNG" alt="Top 5 results from experiment 3b in graphs" style="width: 10px;"/>

The graphs suggest that the best set of hyperparameters have 90 to 100 estimators, require 90 to 100 samples to be at a leaf, and somewhere between 3 and 4 max features. Interestingly the pattern we saw with entropy being the best criterion has reversed, with it being outperformed by gini. Considering this, for my next experiment I will investigate the ranges described and limit the criterion to gini.

### Experiment 3c

I conducted the next experiment with RanomizedSearchCV with the range of hyperparameters described above with 10 random settings using RandomizedSearchCV.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/Random%20Forest/Random%20Forest%20Experiment%203c%20-%20Results.PNG" alt="Results from experiment 3c" style="width: 10px;"/>

I've included all the results and no graphs this time as now we are dealing with fewer parameters and settings it is easier to analyse by looking at. One thing that imediatly jumped out at me was the top 3 results, that were all in the same range, and all exceeded the best accuracies seen so far, despite their accuracies being cross validated unlike those from all other experiments. It does not necessarily mean that there are not better results attainable in the range, but in the interest of time it seems a good idea to put a final focus on investigating a range around these results. So I've decided than in my final experiment for random forests I'll conduct GridSearchCV over the range of 95 to 100 estimators and 95 to 100 minimum samples to be at a leaf. I'll fix the maximum number of feature to 3 and use the gini criterion.

### Experiment 3d
