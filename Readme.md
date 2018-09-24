# Table of Contents
* [**Introduction**](https://github.com/KieranLitschel/PredictingClosingPriceTomorrow#introduction)
* [**Future plans**](https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Readme.md#future-plans)
* [**Progress Log**](https://github.com/KieranLitschel/PredictingClosingPriceTomorrow#progress-log)
  * [**Summary of key results**](https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Readme.md#summary-of-key-results)
    * [**The final feature set**](https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Readme.md#the-final-feature-set)
    * [**Results for the 4-class problem**](https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Readme.md#results-for-the-4-class-problem)
  * [**Predicting using KNN**](https://github.com/KieranLitschel/PredictingClosingPriceTomorrow#predicting-using-knn)
    * [**Results of adding SMAs**](https://github.com/KieranLitschel/PredictingClosingPriceTomorrow#results-of-using-features-adjclosepchange-pdiffclose5sma-pdiffclose8sma-pdiffclose13sma)
    * [**Results of adding RSI**](https://github.com/KieranLitschel/PredictingClosingPriceTomorrow#results-of-adding-feature-rsi)
    * [**Results of adding Bollinger Bands**](https://github.com/KieranLitschel/PredictingClosingPriceTomorrow#results-of-adding-bollinger-band-features)
    * [**Results of adding difference between SMAs**](https://github.com/KieranLitschel/PredictingClosingPriceTomorrow#results-of-adding-percentage-difference-between-smas)
    * [**Results of adding MACD**](https://github.com/KieranLitschel/PredictingClosingPriceTomorrow#results-of-adding-macd)
    * [**Results of adding Stochastic Oscilator**](https://github.com/KieranLitschel/PredictingClosingPriceTomorrow#results-of-adding-stochastic-oscillator)
    * [**Results of adding ADX**](https://github.com/KieranLitschel/PredictingClosingPriceTomorrow#results-of-adding-adx)
  * [**Predicting using logistic regression**](https://github.com/KieranLitschel/PredictingClosingPriceTomorrow#predicting-using-logistic-regression)
  * [**Predicting using random forests**](https://github.com/KieranLitschel/PredictingClosingPriceTomorrow#predicting-using-random-forests)
    * [**Experiments 1 and 2: Adjusting hyperparameters manually**](https://github.com/KieranLitschel/PredictingClosingPriceTomorrow#experiment-1a---changing-number-of-trees)
    * [**Experiment 3: Adjusting hyperparameters using RandomizedSearchCV and GridSearchCV**](https://github.com/KieranLitschel/PredictingClosingPriceTomorrow#experiment-3a)
    * [**Results of experiments 1, 2, and 3**](https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Readme.md#results-of-experiments-1-2-and-3)
    * [**Analysing the importance of each feature**](https://github.com/KieranLitschel/PredictingClosingPriceTomorrow#analysing-the-importance-of-each-feature)
    * [**Conclusions drawn from experimenting with random forests**](https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Readme.md#conclusion)
  * [**Adding more technical indicators**](https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Readme.md#adding-more-technical-indicators)
    * [**Results of adding OBV**](https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Readme.md#results-of-adding-obv)
    * [**Results of adding OBV and adjClose gradients**](https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Readme.md#results-of-adding-obv-and-adjusted-close-gradients)
  * [**Adding financial ratios**](https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Readme.md#adding-fundamental-ratios)
    * [**Introduction**](https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Readme.md#introduction-1)
    * [**Cleaning up the data**](https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Readme.md#cleaning-up-the-data)
    * [**Retuning hyperparameters**](https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Readme.md#retuning-hyperparameters)
  * [**Predicting using neural networks**](https://github.com/KieranLitschel/PredictingClosingPriceTomorrow#predicting-using-neural-networks)
    * [**Results of random search for neural network with one hidden layer**](https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Readme.md#results-of-random-search-for-network-with-one-hidden-layer)
    * [**Tuning network with one hidden layer**](https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Readme.md#tuning-network-with-one-hidden-layer)
    
# Introduction

This is the beginning of a project where I will be using the 20 year history of the current members of the S&P 500 to train a neural network to be able to classify the closing price tomorrow of any given stock into a fixed number of bounds.

# Future Plans

In early September 2018 I started back at university, and I imagine won't have any more time to work on the project until Christmas. This semester I am taking a course in machine learning, which will help me to reflect on what I've done so far, and what I should focus on doing next to attain a better accuracy. Below I've listed a few things I've already learnt about and think might help me improve my project further.

* Reasses which features are important using what I learnt from my machine learning course
  * One thing I have learnt so far is that you should pick features which have similair values for samples belonging to the same class and that are at the same time as dissimilair as possible from different classes. I could do this by visualizing each feature individually, discarding of those which are ambiguous about which class the data belongs to, and seeing if accuracy increases without those features.
  * Another is that decision trees aren't good at handling non-monotonic data. I should consider whether numerical features are monotonic or not when visualizing my data, and features which are not monotonic I should quantize, converting each non-monotonic feature to multiple categorical features. This may allow the random forest to attain an even higher accuracy.
  * One other is that leaving outliers in the data can make it harder for models to be learn, with large values offsetting the mean and standard deviation when numerical features are normalized. I should think about how to identify outliers in my data and then remove them. I tackled this problem briefly before, and found that some values that at first appeared to be outliers turned out to be valid data. I should consider converting features with such values to logarithmic scales to make it easiers for models to be able to learn from them.
* Another thing I should reconsider is how to fill in the missing values from the financial ratios. My current method using the maximum and minimum values of a feature where it makes sense to might not be the best way to go about it, as it looks likes this confuses the regessor into predicting abnormally large values for other missing features. I should consider using regression to predict all missing values, and also try using the mean or median instead of linear regression.
* I also noticed when training neural networks is that the weights the network applied to the financial ratios were a lot closer to 0 than the technical indicators. This suggests the network is finding it difficult to learn from the raw data. I should consider how to make it easier for them to be understood by the network.

# Progress Log
## Summary of key results

The rest of the progress log is written in chronologigal order, but this section will be updated as I create new models.

### The final feature set

The finalized feature set consists of the following technical indicators and financial ratios, see later sections for details on non-standard ones.

The following table lists all features related to technical indicators. These are all calculated from historical stock quotes for each trading day which are fetched via the [Alpha Vantage API](https://www.alphavantage.co/).

| Variable name(s)                                | Explanation                                                                    |
|:-----------------------------------------------:|:------------------------------------------------------------------------------:|
| adjClosePChange                                 | Percentage change in closing price since the previous trading day              |
| pDiffClose5SMA, pDiffClose8SMA, pDiffClose13SMA | Percentage difference between the closing price and the 5, 8, and 13 day SMA's |
| rsi                                             | Relative strength index                                                        |
| pDiffCloseUpperBB                               | Percentage difference between the upper bollinger band and adjusted closing price |
| pDiffCloseLowerBB                               | Percentage difference between the lower bollinger band and adjusted closing price |
| pDiff20SMAAbsBB                                 | Percentage difference between the upper bollinger band and the 20-day SMA used to calculated the bollinger bands |
| macdHist                                        | The value for the macd histogram |
| deltaMacdHist                                   | The change in the macd histogram since the last trading day |
| stochPK                                         | The value of %k for the full stochastic oscilator |
| stochPD                                         | The value of %d for the full stochastic oscilator |
| adx                                             | Average directional index                                                      |
| pDiffPdiNdi                                     | The percentage difference of the postivie directional indicator from the negative directional indicator used in ADX |
| obvGrad5, obvGrad8, obvGrad13                   | The gradient of 5, 8, and 13 day SMA's for on-balance volume                   |
| adjCloseGrad5, adjCloseGrad8, adjCloseGrad13, adjCloseGrad20, adjCloseGrad35, adjCloseGrad50 | The gradient of the 5, 8, 13, 20, 35, and 50 day SMA's for adjusted closing price |

The following table lists all features related to financial ratios. These are fetched from the Wharton Research Data Services database on financial ratios. The techincal indicators for each trading day are joined with the most recent quote for ratios, but quotes for ratios are only given once per month, so ratios for each trading day can be up to a month out of date, but nonetheless they provide more insight than technical indicators alone.

| Variable name(s)| Explanation             |
|:---------------:|:-----------------------:|
| de_ratio        | Debt to equity ratio    |
| curr_ratio      | Current ratio           |
| quick_ratio     | Quick ratio             |
| roe             | Return on equity        |
| npm             | Net profit margin       |              
| pe_exi          | P/E (Diluted, Excl EI)  |
| pe_inc          | P/E (Diluted, Incl EI)  |
| ps              | Price to sales          |
| ptb             | Price to book           |
| roa             | Return on assets        |
| intcov_ratio    | Interest coverage ratio |
| at_turn         | Asset turnover ratio    |

### Results for the 4-class problem

The 4 class problem is described at the beginning of KNN, but in brief we are trying to predict the closing price of the following trading day into one of the following 4 bands:

* adjClosePChange <-1%
* -1% <= adjClosePChange < 0%
* 0% <= adjClosePChange < 1%
* 1% <= adjClosePChange

| Method                               | Details | Accuracy on test set |
|:------------------------------------:|:-------:|:--------------------:|
| Random forest                        | 650 estimators, 100 min samples per leaf, 9 max features per split | 33.311% |
| Neural network with one hidden layer | layers: \[input: 38 neurons -> hidden layer: 60 neurons, relu -> dropout layer: 0.3 dropout rate -> output layer: 4 neurons, softmax\], batch size of 756, learning rate of 10^-4, 4390 epochs, L1 regularization with a value of 10^-9 for lambda | 32.835% |
| Neural network with two hidden layers | layers: \[input: 38 neurons -> hidden layer: 70 neurons, leaky relu -> dropout layer: 0.08 dropout rate -> hidden layer: 70 neurons, leaky relu -> dropout layer: 0.08 dropout rate -> output layer: 4 neurons, softmax\], batch size of 756, learning rate of 10^-4, 1855 epochs | 33.526% |

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

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/KNN/KNN%20with%204%20features.png" alt="KNN with 4 features, plateus at approx 31% accuracy" style="width: 10px;"/>

Conclusions:
* Begins to plateu at about 100 neighbours, trailing off at around 31% accuracy.
* There are 4 classes so we would expect an accuracy of around 25% if it was classifying randomly and there was no pattern in the data, so trailing off at 31% accuracy suggests there is some pattern.
* Considering we are using around 1.9 million samples for training and testing, we should see an increase in accuracy if we increase the number of features.
### Results of adding feature RSI
Note that:
* RSI is the relative strength indicator ([see here](https://www.investopedia.com/terms/r/rsi.asp) for more information.
* I used the features adjClosePChange, pDiffClose5SMA, pDiffClose8SMA, pDiffClose13SMA, and RSI, although on the red line plotted below I show the results from the last experiment for comparison which I did not train using RSI.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/KNN/KNN%20with%205%20features%20(rsi).png" alt="KNN with and without RSI" style="width: 10px;"/>

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

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/KNN/KNN%20with%208%20features%20(bollinger%20bands).png" alt="KNN with and without RSI" style="width: 10px;"/>

Conclusions:
* Adding the 3 features led to a 0.77% increase in accuracy.
* I was expecting much larger increases in accuracy than I am achieving, as at this rate I don't think I will be able to surpass an accuracy of 40% using KNN, suggesting it is not well suited to the problem. Despite this I will continue to test with KNN while implementing new features, as it is very quick to test and allows me to judge the progress I am making. Once I've generate all the features I have planned I will train a neural network to solve the problem, which I expect will perform much better.
### Results of adding percentage difference between SMAs
I hypothesised that there might be more that could be learnt from looking at the difference between the 5, 8, and 13 day SMAs with each other, so I added 3 features to capture this. Below are the results of adding these 3 features.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/KNN/KNN%20with%2011%20features%20(difference%20between%20SMAs).png" alt="KNN with and without difference between SMAs" style="width: 10px;"/>

It is clear that adding these features does not help improve accuracy, probably because all they have to contribute is captured by the first 3 features I added related to SMAs, hence I will not use them as features in future, but may consider adding them again when applying PCA.
### Results of adding MACD
The [MACD technical indicator](https://www.investopedia.com/terms/m/macd.asp) is used to identify when is best to enter and exit a trade, so I thought it might be able to give some information that will help determine what will happen to the stock tomorrow. The first feature I added was the MACD histogram, which is the difference between the MACD value (the difference between the fast and slow EMA) and the signal (the EMA of the MACD value). This value should help identify crossovers and dramatic rises.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/KNN/KNN%20with%209%20features%20(MACD%20Histogram).png" alt="KNN with and without MACD Histogram" style="width: 10px;"/>

The results of this experiment are very similair to the last one, but note how the two lines do not cross beyond 30 neighbours, and we also see a 0.13% increase in acuracy when k=100. For these reasons I think that the MACD is improving the accuracy of the classifier. However the increase in accuracy is rather disappointing, and I hypothesised that an issue may be that it is impossible for the model to tell which way the MACD histogram is moving and how fast, so I decided to add a feature to represent the difference between the MACD histogram at the current period and at the last period.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/KNN/KNN%20with%2010%20features%20(MACD%20Histogram%20delta).png" alt="KNN with and without MACD Histogram delta" style="width: 10px;"/>

But from looking at the results from the experiment above, there is very little difference between the lines with and without the difference, suggesting there is nothing more to be learned from this feature. Hence I will leave the difference out of future models for the time being.
### Results of adding Stochastic Oscillator
I chose to add the [stochastic oscilator](https://en.wikipedia.org/wiki/Stochastic_oscillator) as up to this point all features have been generated using the adjusted closing price, but the stochastic oscilator calculates momentum using the high, low, and close. The downside to this is that as the Alpha Vantage API does not supply an adjusted high and low price, so I have to use the raw high, low, and close prices. This means that stock splits and dividends will lead the oscilator to give false signals, but these are not frequent events so I hope that this indicator will still be able to give me a boost in accuracy.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/KNN/KNN%20with%2011%20features%20(stochastic%20oscillator).png" alt="KNN with and without stochastic oscillator" style="width: 10px;"/>

We see an increase of accuracy of 0.19% as a result of adding this feature, and it is clear from the graph that it does increase accuracy.
Increasing the number of features has made testing new features significantly slower, as a result in future experiments I will try neighbours in the range 60 to 110, as there appears to be a trend that from 60 onwards accuracy begins to plateu.
### Results of adding ADX
I decided to include [ADX](https://www.investopedia.com/articles/trading/07/adx-trend-indicator.asp) as it is a very popular indicator that is used to determine trends in price movemement and whether they are trailing off or strengthening, which should help indicate which way the closing price will move tomorrow.

Unfortunately the testing time increased again, and it was taking 10 minutes to test for each value of k, hence I have decided I will no longer experiment with KNN, and just focus on implementing features and then training a neural network.

## Predicting using logistic regression
I decided that in the interest of time I will just train the model with the features I have used so far, and once I have trained a model that performs relatively accurately I will investigate whether more features would be beneficial. Next I experimented with logistic regression, and to start with I attempted to train a model to predict whether the stock will be higher or lower at end of the next day, as this was a much simpler problem to solve with logistic regression and would allow me to quickly test how suitable logistic regression is for the problem. I wrote the code in TensorFlow, I chose to use the adam optimiser so that I could train the model quickly with fewer hyperparameters.

But I found that the best accuracy I could achieve was 51%, and bizarrely increasing the value of the learning rate increased the accuracy on the training and test set despite the value of the loss function plateuing at a higher value. It is also worth noting that our accuracy for the 2 class problem is only 1% higher than if we assigned classes at random, and despite our 4 class problem being more complicated we achieved an accuracy 7% higher than if we assigned classes at random. These observations lead me to the hypothesis that our data is not linearly seperable. This would expain why KNN outperformed logistic regression with a harder problem, as KNN is capable of forming non-linear decision boundaries, whereas logistic regression is only capable of forming linear ones.

To test this hypothesis I reran KNN for all features I had selected up to and including the stochastic oscillator using the 2 class problem. The results of which you can see below.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/KNN/predicting%20rise%20or%20fall%20KNN%20with%2011%20features%20(up%20to%20stochastic%20oscillator).png" alt="KNN classifying rise or fall of stock" style="width: 10px;"/>

In previous experiments I increased the number of neighbours each time by 5, but for this experiment I increased the number by 10 each time so the graph isn't as smooth. But it is clear that with more neighbours we could achieve a higher accuracy, with no clear indication that the increase in accuracy is slowing down as the number of neighbours increases. However, it is worth noting that the increase in accuracy each time is not significant, so it is not clear by what margin KNN outperforms logistic regression. So the conclusion from this experiment is that KNN does not give a definitive answer whether the data is linearly seperable or not. Though considering the performance of logistic regression I think it is safe to assume the data is not linearly seperable. Hence I will now experiment with random forests as they are able to cope with non-linearly seperable data.

## Predicting using random forests
### Experiment 1a - changing number of trees
I will be using the implementation of random forests given in scikit learn as the version of TensorFlow I am using does not support random forests on windows. I first experimented with the number of trees, leaving all other parameters as default.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/Random%20Forest/Random%20Forests%20-%20Experiment%201a.png" alt="Random forests tuning the number of trees" style="width: 10px;"/>

The accuracy does not seem to plateu as quickly as we saw for KNN, with increase in accuracy being more gradual and not appearing to tail off. Despite this the time to generate the forests increases rapidly, with it taking 210 seconds for 70 trees, and 300 seconds for 100 trees. Consequently although 100 trees gives a better accuracy I have chosen 70 as the best value for the number of trees in this experiment, with no significant improvement in accuracy after this point.

### Experiment 1b - 70 trees, changing maximum number of features

Next I adjusted max_features, which is the maximum number of features that are considered at each split, the default value was 4, but I tried all possible values (1 to 13). 

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/Random%20Forest/Random%20Forests%20-%20Experiment%201b.png" alt="Random forests tuning the maximum number of features" style="width: 10px;"/>

The results indicate the maximum number of features has little effect on accuracy, but it has a big effect on time, with it taking around 200 seconds to generate the tree and prediction for a maximum of 4 features which gives an accuracy of 31.9679%, and around 100 seconds to do the same with only 1 feature per tree which gives an accuracy of 31.8909%. Hence I will continue my experiments with 1 feature per tree for now.

### Experiment 1c - 1 feature per tree, changing number of trees

Now that generating the tree is much faster, I thought it was worth experimenting to see if I could get a higher accuracy with more trees.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/Random%20Forest/Random%20Forests%20-%20Experiment%201c.png" alt="Random forests 1 feature per tree, tuning the number of trees" style="width: 10px;"/>

I decided 100 trees was the best compromise between accuracy and time, with it giving an accuracy of 32.0716%, an increase of almost 0.2%, whilst increase time to generate and predict by only 40 seconds.

### Experiment 1d - 100 trees, 1 feature per tree, changing minSamplesLeaf

minSamplesLeaf is the minimum number of samples to be at a leaf node. Increasing this should help reduce overfitting of the test set. 

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/Random%20Forest/Random%20Forests%20-%20Experiment%201d.png" alt="Random forests 100 trees, 1 feature per tree, changing minSamplesLeaf" style="width: 10px;"/>

There is no clear trend with the graph, so it is harder to choose the best value. However a value of 50 gives the highest accuracy at 32.9638%, also reducing the time to generate and predict to 105 seconds, considering this improvement I will choose a value of 50 for future experiments. 

### Experiment 2a - 70 trees, changing minSamplesLeaf

It seems odd to me that our best value for maximum number of features is 1, as this suggests that it doesn't matter what features you use to split the data following each split, implying that features are independant of each other. Considering this I have decided it is worth investigating optimising minSamplesLeaf before maximum number of features, and seeing how this impacts when we investigate maximum number of features. 

For this experiment I will be starting with 70 trees as this was the best found for trees without tuning any other hyperparameters.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/Random%20Forest/Random%20Forests%20-%20Experiment%202a.png" alt="Changing maximum number of feature" style="width: 10px;"/>

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

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/Random%20Forest/Random%20Forest%20Experiment%203a%20-%20Top%2020.PNG" alt="Top 20 results from experiment 3a" style="width: 10px;"/>

I seperated min_samples_leaf and n_estimators into 10 bounds, and counted the number of times each hyperparameter occured in each bound in the top 20 accuracies, and weighted each bound by the ratio of the number of times it was sampled in the 55 random settings. The results are displayed in the following graphs.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/Random%20Forest/Random%20Forest%20Experiment%203a%20-%20Top%2020%20in%20Graphs.PNG" alt="Top 20 results from experiment 3a in graphs" style="width: 10px;"/>

A higher score for a bound in relation to other bounds for each hyperparameter suggests it is more significant to accuracy. Considering this it appears that we can reduce the bounds to 50 to 100 estimators (trees), 3 to 6 features at each split, and the minimum number of samples to be at a leaf node to 50 to 100. We will use this range in experiment 3b.

### Experiment 3b

I conducted the next experiment with RanomizedSearchCV with the range of hyperparameters described above. I tried 25 random settings and also included the results from 3a that were in the range being investigated in the results table. This time I looked at the results with the top 5 accuracies.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/Random%20Forest/Random%20Forest%20Experiment%203b%20-%20Top%205.PNG" alt="Top 5 results from experiment 3b" style="width: 10px;"/>

The best accuracy found exceeded that discovered in 3a, suggesting that we are on the right track. I analysed the parameters for the top 5 settings using the same method as before.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/Random%20Forest/Random%20Forest%20Experiment%203b%20-%20Top%205%20in%20Graphs.PNG" alt="Top 5 results from experiment 3b in graphs" style="width: 10px;"/>

The graphs suggest that the best set of hyperparameters have 90 to 100 estimators, require 90 to 100 samples to be at a leaf, and somewhere between 3 and 4 max features. Interestingly the pattern we saw with entropy being the best criterion has reversed, with it being outperformed by gini. Considering this, for my next experiment I will investigate the ranges described and limit the criterion to gini.

### Experiment 3c

I conducted the next experiment with RanomizedSearchCV with the range of hyperparameters described above with 10 random settings using RandomizedSearchCV.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/Random%20Forest/Random%20Forest%20Experiment%203c%20-%20Results.PNG" alt="Results from experiment 3c" style="width: 10px;"/>

I've included all the results and no graphs this time as now we are dealing with fewer parameters and settings it is easier to analyse by looking at. One thing that imediatly jumped out at me was the top 3 results, that were all in the same range, and all exceeded the best accuracies seen so far, despite their accuracies being cross validated unlike those from all other experiments. It does not necessarily mean that there are not better results attainable in the range, but in the interest of time it seems a good idea to put a final focus on investigating a range around these results. So I've decided than in my final experiment for random forests I'll conduct GridSearchCV over the range of 95 to 100 estimators and 95 to 100 minimum samples to be at a leaf. I'll fix the maximum number of feature to 3 and use the gini criterion.

### Experiment 3d

Below are the top 5 results of the grid search.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/Random%20Forest/Random%20Forest%20Experiment%203d%20-%20Top%205%20in%20Graphs.PNG" alt="Top 5 results from experiment 3d in graphs" style="width: 10px;"/>

Unfortunately we did not see much improvement from experiment 3c, with our best found result only have a standard deviation of 0.01% lower, and an accuracy 0.01% higher. But none the less this is an improvement over 3c.

### Results of experiments 1, 2, and 3

Below are the results of the best chosen hyper parameters from each of the three experiments.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/Random%20Forest/Random%20Forest%20Experiment%201%2C%202%2C%203%20results.PNG" alt="Results of experiments 1, 2, and 3" style="width: 10px;"/>

Notie that accuracies differ slightly as I am using 4-fold cross validation, something I didn't do in experiments 1 and 2, and I suspect there's a slight difference in experiment 3 as I am using cross_val_score as opposed to GridCV, so I imagine folds are being allocated samples slightly differently. The hyper parameters chosen in experiment 3 perform the best, so we will use these as the final ones for our random forest.

### Analysing the importance of each feature

Having decided on the best hyper parameters for the random forest, I decided to investigate the importance the forest had assigned to each of the features.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/Random%20Forest/Signifance%20of%20Features%20Based%20on%20RF%20Exp%203d.PNG" alt="Importance of features based on random forest from experiment 3c" style="width: 10px;"/>

The results are somewhat surprising, with features relating to bollinger bands being far more significant than other features. Their significance does make sense though, with pDiff20SMAAbsBB capturing the squeeze of the bollinger bands which is indicative of volatility, and pDiffCloseUpperBB being the difference between upper bollinger band and the closing price, which can be indicative of a stock being overbought when closer to the upper band, which would point to a fall in price following shortly. The difference of the lower band from the closing price being less significant is somewhat confusing, but this may be because a lot of information can be gathered from the the upper band when the closing price is much smaller than it. 

One thing that I struggled to decide when first implementing the technical indicators was the period for each of them to capture. All but RSI rely on an EMA's or SMA's, and the smaller periods these capture the more sensitive they are to change, and the larger the less sensitive they are. I decided to leave them at default values when I implemented them, but I decided now it was worth seeing if there was any pattern between the period they captured and the significance of them to the forest.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/Random%20Forest/Signifance%20of%20Features%20compared%20to%20period.PNG" alt="Comparison of feature significance to period length" style="width: 10px;"/>

There seems like there might be some pattern, with the 5 most significant features having a period in the range of 10 to 20. Considering this I decided it was worth experimenting with the worst performing indicators, the stochastic oscilator and the MACD. I made a faster MACD, halving the period length of the slow and fast line to 6 and 13 periods respectively, and a slower stochastic oscilator, quadrupling the slow and fast period to 12 and 20 days respectively.

Unfortuantely this had little effect on accuracy, with replacing the old features with the new ones actually decreasing accuracy by 0.1%, and the significance of the new features only marginally differing only marginally from the originals. I did not try retuning the hyperparameters, but considering the change in significance being small it doesn't seem like there would be much of a change in accuracy even if I reoptimised the hyperparameters. Consequently I have concluded it is probably more to do with what the stochastic oscilator and MACD fundamentally behave that makes them less significant than other features, rather than them being less significant because they have different period lengths. Consequently I will revert to using the features with their original period lengths.

### Conclusions drawn from experimenting with random forests

The random forest classifier was an improvement over KNN as it drastically reduced prediction time, with the model being pregenerated, but disappointingly we only saw around a 0.6% increase in accuracy. It seems that the model is dominated by bollinger bands, with the other features contributing little in comparison. We would have expected to see an increase in accuracy by considering more features at each split, but we didn't, with considering less features than default at each split performing best. Considering all of this, it seems that the best course of action is to add more features, which I plan to do next.

## Adding more technical indicators

### Results of adding OBV

The complexity with OBV (on-balance volume, [see here](https://traderhq.com/trading-indicators/understanding-on-balance-volume-and-how-to-use-it/)) is that the important thing is not it's value, but the trend in movement considering the trend in movement of price. I've decided the best way to capture this is by estimating the line of best fit through the OBV and also the prices using linear regression, and taking the estimated gradients to estimate the trend of each.

It is not useful to compare these gradients directly, particularly as they will vary greatly depending on the time period examined. For example the volume traded of Microsoft around the tech bubble in the 2000's is much higher than the volume traded today, so the rate of change in volume traded is much higher around the 2000's too. Also we can't even compare the rate of change of different stocks around the same time period, as for example members of the FTSE 100 will have a much higher volume traded than members of the FTSE 250, so we run into the same problem. It may be worth investigating in future standardizing each stocks rate of change of volume determining the standard deviation and mean using a fixed number of periods in the past, but I am uncertain of how effective this would be, so I will leave it for now.

So instead of comparing gradients directly I will compare the rate of change of OBV and rate of change of adjusted closing price for each sample, using a fixed number of period prior to each sample to estimate the rates of change. It is important to be able to determine when the rate of change is flat, I will do this by dividing absolute value of the OBV gradient by the the average of the absolute values of the OBV in the period it was selected from, if this value is less than or equal to 0.05 (chosen by hand) I will consider the OBV gradient flat.

I have also formalised the information in the last paragraph into mathematics to make it clearer.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/Maths/OBV%20supporting%20equations.PNG" alt="Equations that support OBV formalised into maths" style="width: 10px;"/>

Using all of the above I will assign each sample into one of four categories:

* Price continues to fall - If the price trends down, the OBV trends down, and the OBV gradient is not flat, then this suggests the price will continue to fall, so we assign it a value of 0
* Bearish divergence - If the price is trending up, but OBV is down or the OBV gradient is flat, then this suggests bearish divergence, so we expect prices will start to fall, and we assign it a value of 1
* Bullish divergence - If the price is trending down, but OBV is trending up or the OBV gradient is flat, then this suggests bullish divergence, so we expect prices will start to rise, and we assign it a value of 2
* Prices continue to rise - If the price trends up, the OBV trends up, and the OBV gradient is not flat, then this suggests prices will continue to rise, so we assign it a value of 3

I am not certain what value of n will perform best, so I've decided to add features for n = 5, 8, and 13, as these values performed well with the SMAs.

Disappointingly using training a random forest with our best features and testing using 4 fold cross validation with the new features and all the previous ones, accuracy only increased by around 0.01% and standard deviation increased by 0.02%. The significance assigned to each feature suggests to us the issue is not the hyperparameters, with the new features each being assigned a significance of around 1/15th of the previous least significant features.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/Random%20Forest/Random%20Forests%20-%20Significance%20of%20Adding%20OBV%20Predictors.PNG" alt="Bar chart of significance of adding OBV related features" style="width: 10px;"/>

I tried tweaking the threshold to classify the OBV gradient as flat, trying 0.01 and 0.1, but neither gave much improvement on using a value of 0.05. I thought about what might be wrong, and came to the conclusion it may be to do with how the gradient is calculated as flat. Dividing the gradient by the average of the absolute OBV works if the OBV is oscillating around 0. But over time successful stocks trend towards very large values of OBV's, meaning that the OBVGradRatio becomes very small. Consequently I decided to try without considering whether the gradient is flat, just going based on if its positive or negative, to see if we would see any improvement in performance.

After making this change the significance of each feature doubled, but were still 1/10th of the next smallest, and accuracy remained below that without these features. The final thing I thought I'd try is just using the raw gradients as features, which I will continue in the next section, as although I thought it shouldn't work, I could have been wrong, so it was worth trying, as otherwise there was no point keeping features related to OBV.

### Results of adding OBV and adjusted close gradients

Note that this section continues on from the previous.

Suprisingly using the raw gradients as features proved successful, as we see below, with each feature having a significance of around 2%.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/Random%20Forest/Random%20Forests%20-%20Significance%20of%20Adding%20Gradients.PNG" alt="Bar chart of significance of adding gradients" style="width: 10px;"/>

It is worth noting that the accuracy only increases by 0.1% using 4-fold cross validation, but I have not retuned the hyperparameters so I am not too concerned about this. 

I investigated to see if I could workout why I was wrong and features containing raw gradients were significant. I think the reason is although I am correct about gradients varying a lot depending on the company and its point in history, the combination of looking at 500 companies over 20 years gives a large range of gradients, which seems to be the reason why these issues are overcome.

The bar chart is ordered from left to right from most significant to least significant, and interestingly it seems to suggest that the longer the period examined, the more significant the gradient. As a result of this observation, I decided it was worth investigating larger period lengths. So I tried increasing to 20, 35, and 50 day periods, to see if significance increased further.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/Random%20Forest/Random%20Forests%20-%20Significance%20of%20Adding%20Gradients%2020%2C%2035%2C%2050.PNG" alt="Bar chart of significance of adding gradients of longer periods" style="width: 10px;"/>

The significance of the OBV gradients does not change much, but the significance of the gradients of the adjusted close increases significantly. Consequently I've decided I will include the adjusted close gradients of 5, 8, 13, 20, 35, and 50, and will keep with the OBV gradients of 5, 8, and 13. Below are the significance of including those features.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/Random%20Forest/Random%20Forests%20-%20Significance%20of%20Final%20Gradients.PNG" alt="Bar chart of significance of adding final gradients" style="width: 10px;"/>

For the record, these features increase accuracy using 4-fold cross validation to 33.0955%, and decrease standard deviation to 0.2366%. This is only a minor improvement, but there should be a more substantial improvement when we retune the hyperparameters.

## Adding financial ratios

### Introduction

Up until now I have only been using technical indicators to make predictions, but I have now obtained access to data on financial ratios through Wharton Research Data Services. The downside to this is that they only have monthly data on the period 1990 to 2015, so I will only be able to make predictions on past data. But I think the benefit outweighs this downside, as it is impossible to value a company based purely on technical indicators, so financial ratios should allow me to make far better predictions.

As I will be using a slightly different period I have rerun the same tests I performed in the last experiment with the new set of data. There is little variation, with the accuracy using 4-fold cross validation being 33.0197% with a standard deviation of 0.2874%. The feature significances do not change much either, but are displayed below for reference anyway.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/Random%20Forest/Random%20Forests%20-%20Significance%20Before%20Adding%20Fundamentals.PNG" alt="Bar chart of significance of features using funamentals data set" style="width: 10px;"/>

### Cleaning up the data

The first issue I've encountered is that quarterly reports are not associated with a ticker symbol, but a permno number, most stocks have multiple permno numbers, being issued a new one when there is a major restructuring. This adds a layer of complexity as permno numbers can describe overlapping periods, and there is no way to automate the process of deciding which permno is more relevant. Consequently I've decided the best way to go about this is by favouring smaller permno numbers to larger ones when choosing which one to use. This is because a permno number is permenant, so they are not reissued, meaning larger numbers were issued later than smaller ones. However if a smaller permno does not cover the entire 1990 to 2015 period, I will make use of the next greatest permno number for the period it doesn't cover. This is a somewhat arbitrary decision, but it should keep ratios consistent which is important for training.

The other important thing to consider is handling missing data, as each row in the fundamentals table corresponds to 30 rows in timeseriesdaily table, it is not feesible to just throw them away. To tackle I got rid of ratios that were missing in over 50% of the data, and carefully selected important ratios with those missing in between 10% and 50% of the data, and discarded the rest. I used rules of thumb to replace missing values in the columns that I'd keep that were missing more than 10% of the data, for example if the equation involved debt, it was reasonably likely that the data was missing as the company had no debt, so I interpreted the equation that corresponded to the ratio and assigned it a value of 0 or the maximum value observed in the column to reflect what the ratio would approach as the debt approached it's true value of 0. Next I selected all the remaining samples that were missing at least one column, this equated to about 25% of the data. I grouped these by the columns they were missing, and in each group selected the non-null columns, and used those to predict the missing columns using linear regression with the data that had no missing columns as training data.

### Retuning hyperparameters

To start with I tried using all 53 ratios I had collected from the WRDS database. The results are displayed below:

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/Random%20Forest/Random%20Forests%20with%20all%20ratios.PNG" alt="Table of results of retuning hyperparameters with all ratios" style="width: 10px;"/>

Clearly the default parameters we used in previous experiments, but even after reoptimising, tripling the number of features and estimators the new features made the model worse. I suspect this was because a lot of them weren't relevant and were just adding noise, so I did some research and selected 12 of the 56 ratios to keep, you can see the full list of those I kept in the spreadsheet in Results\Random Forests\Adding financial ratios.xlsx in the Believed Significant Features tab. I first tried the hyperparameters I had chosen when first tuning and the results were promising, with only a 0.1% drop off in accuracy despite increasing the number of features by 50%. Next I experimented with a larger range of estimators and features than I had before, to see if I could beat previous scores, and you can see the results below.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/Random%20Forest/Random%20Forests%20with%20selected%20ratios.PNG" alt="Table of results of retuning hyperparameters with selected ratios" style="width: 10px;"/>

I managed to, but I only achieved around a 0.01% increase in accuracy. As a result of this I suspect I have exhausted any improvement in accuracy I can gain using random forests. As a result I will now focus on building a neural network, but as a final experiment I will try using a much larger range of estimators and features when building the random forests, experimenting with between 500 and 1000 estimators, and 6 to 17 maximum features at each split. Training these and evaluating them using 4-fold cross validation will take a lot of time, so I will make use of Google Compute Engine and run this experiment on the cloud in virtual machines.

The top 10 results of 26 are displayed below, and somewhat unsurprisingly we only gained an increase 0.1% in accuracy, suggesting that we have exhausted any increase we can get from random forests. 

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/Random%20Forest/Random%20Forests%20with%20selected%20ratios%20extended%20search.PNG" alt="Table of results of extended search of retuning hyperparameters with selected ratios" style="width: 10px;"/>

## Predicting using neural networks

I've decided to train the neural network in TensorFlow as it is a modern library that will integrate well with training using Google Compute. I've experimented with low-level TensorFlow and Keras, and decided that Keras is most appropriate for this project as it will allow me to adjust the model quickly. 

To start with I used the Adam optimizer, as this seemed like the simplest optimizer to tune. I also started with a neural network with one hidden layer, as this is simplest to optimize.

### Results of random search for network with one hidden layer

I started by running 60 iterations of RandomizedSearchCV experimenting with modifying the number of neurons and epochs, the activation function, the batch size, whether to use L1 or L2 regularization, and what value of lambda to use for regularization. Below are the top 10 results.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/Neural%20Networks/Top%2010%20results%20from%20random%20search%20of%20single%20hidden%20layer.PNG" alt="Top 10 results from random search for a neural network with a single hidden layer" style="width: 10px;"/>

Noticably the top 10 is dominated by networks using rectified linear units as the hidden layer. As a result I decided to retune the best result to see if I could improve the accuracy further.

### Tuning network with one hidden layer

It is infeasible to tune hyperparameters using the whole network, so I used a sample. First I tried using a sample of 1% of the data. The experiments I conducted are sumarised in the results folder for neural networks in the repo in Hyperparameters.docx, and the results are displayed in graphs of experiments 1 - 3. I have decided not to include any detail here though because the sample size proved too small, with there being a lot of noise in the accuracy meaning it was difficult to optimise, and when I tried my retuned hyperparameters on the whole training set I found accuracy decreased from those found by the random search, suggesting the sample did not generalize the dataset well and that I had overfitted the sample as a result.

Consequently I decided to optimise using 10% of the data. But first I decided to tune the number of epochs using the whole dataset. The results are displayed below.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/Neural%20Networks/Experiment%204%20-%20Changing%20Number%20of%20Epochs.png" alt="Effect of changing the number of epochs" style="width: 10px;"/>

This proved to be successful, with 200 epochs giving the best accuracy of 32.0992%, with a standard deviation of 0.1491%. This also significantly reduced training times, with it reducing time to train the network from 200 minutes to 80 minutes, a significant improvement. This also made it much more feesible to tune hyperparameters using 10% of the data.

To start with I trained the network with the new optimal found hyperparameters to give a baseline value. I found the accuracy to be 31.9259% with a standard deviation of 0.372%. This was very close to the accuracy found using the whole dataset suggesting the sample size was suitably large. Next I retuned the number of neurons.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/Neural%20Networks/Experiment%205%20-%20Changing%20Number%20of%20Neurons.png" alt="Effect of changing the number of neurons" style="width: 10px;"/>

There is quite a lot of noise in the graph, but the increase in accuracy appears to plateu after around 40 neurons. But I decided to use 75 neurons as this gave the lowest standard deviation of 0.167%, and also gave a mean accuracy of 32.178%

Next I attempted to retune regularization, trying both L1 and L2 regularization with a range of values of lambda, you can see the results in the results folder in images for experiments 6 and 7 respectively. I have only included the results for L2 (experiment 7) below, as they both show the same trend, with accuracy continuing to increase as lambda decreased.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/Neural%20Networks/Experiment%207%20-%20Changing%20L2.png" alt="Effect of changing the value of lambda with L2 regularization" style="width: 10px;"/>

I hypothesised this might have been because in combination regularization and dropout led to underfitting, so I tried setting lambda to 0, and sure enough the accuracy increased to 32.232% with a standard deviation of 0.291%.

Consequently next I decided to retune the dropout rate, using a value of 0 for lambda so that regularization did not intefere with the experiment.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/Neural%20Networks/Experiment%208%20-%20Changing%20Dropout%20Rate.png" alt="Effect of changing the dropout rate" style="width: 10px;"/>

I found that setting the dropout rate to 0.3 gave the best accuracy of 32.297% with a standard deviation of 0.326%.

I decided it was worth also experimenting with L1 and L2 regularization without dropout to see which of the three performed the best. Consequently I set the dropout rate to 0, and tried a range of values for lambda. Note that each value of the y-axis was attained using a value of lambda of 10^(-x).

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/Neural%20Networks/Experiment%209%20-%20Changing%20L1.png" alt="Effect of changing the value of lambda with L1 regularization" style="width: 10px;"/>

I found that 0.001 that using 0.001 as the value for lambda gave the best accuracy of 32.232% with a standard deviation of 0.291%.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/Neural%20Networks/Experiment%2010%20-%20Changing%20L2.png" alt="Effect of changing the value of lambda with L2 regularization" style="width: 10px;"/>

I found that using 0.001 as the value for lambda gave the best stable accuracy of 32.128% with a standard deviation of 0.271%.

Of the three dropout appears to perform the best, so I used that with a dropout rate of 0.3. But I decided it was worth seeing if any greater accuracy could be achieved using the new dropout rate and L1 regularization, so I investigated this. Note that in the graph below, the value of lambda for each value of x corresponds to 10^(-x).

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/Neural%20Networks/Experiment%2011%20-%20Changing%20L1.png" alt="Effect of changing the value of lambda with L1 regularization and dropout" style="width: 10px;"/>

From the above I found that a value of 10^-9 gave the best accuracy, giving a mean accuracy of 32.361% with a standard deviation of 0.229%.

Next I retuned the number of neruons again, in case the change in dropout and regularization choice meant the optimal value changes.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/Neural%20Networks/Experiment%2011%20-%20Changing%20L1.png" alt="Changing the number of neurons" style="width: 10px;"/>

From the above I discovered the new optimal was 60 neurons, giving a mean accuracy of 32.327% and a standard deviation of 0.211%. It is worth noting this accuracy is lower than that found in the last experiment, but this is probably down to the random numbers used being slightly different.

Next I tried to retune the batch size and number of epochs using the whole training set.

The first experiment used a batch size of 756.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/Neural%20Networks/Experiment%2013%20-%20Number%20of%20epochs%20(756%20batch%20size).png" alt="Changing the number of epochs with a batch size of 756" style="width: 10px;"/>

The second experiment used a batch size of 512.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/Neural%20Networks/Experiment%2014%20-%20Number%20of%20epochs%20(512%20batch%20size).png" alt="Changing the number of epochs with a batch size of 512" style="width: 10px;"/>

And the third a batch size of 256.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/Neural%20Networks/Experiment%2015%20-%20Number%20of%20epochs%20(256%20batch%20size).png" alt="Changing the number of epochs with a batch size of 256" style="width: 10px;"/>

I found that changing the batch size had very little impact on accuracy, but increased training time significantly. Consequently I went with a batch size of 756, as it was the fastest too train, and also yielded the highest accuracy of 32.715% with a standard deviation of 0.225% in 200 epochs.

In experiments 16 to 26 I retuned the learning rate and number of epochs. I have omitted most of the results here as several experiments proved unsuccessfuly, for example trying to tune using cross-validation and only on a sample of the training set proved unsuccessfuly, and in the end I used the whole training set for tuning, using the validation set to validate the training set wasn't being overfitted. In the end a learning rate of 10^-4 proved optimal training over 4390 epochs, which gave an accuracy on the validation set of around 32.87%. The full results for the whole range of epochs are displayed below.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/Neural%20Networks/Experiment%2024%20-%20Changing%20the%20number%20of%20epochs.PNG" alt="Changing the number of epochs with a learning rate of 10^-4" style="width: 10px;"/>

Note that the graph above is slightly different from the usual, as I made use of TensorBoard for training which reports the validation accuracy after every epoch. The bold line is the smoothed accuracy for each epoch (with a smoothing factor of 0.9), and the faint line is the true values of accuracy.

Next I evaluated the model on the test set, a set of data which I had not used at all in training and tuning the network, this gave an accuracy of 32.84%, suggesting the hyperparameters were well chosen.

### Tuning network with 2 hidden layers

To start with I ran 60 random searches using 4-fold cross validation over a similair hyperparameter space to the one in single layer problem, with the only difference being I added another hidden layer. I based the starting point of my investigation off of the best result which gave a mean accuracy of 31.5% and had 59 neurons in the first hidden layer and 61 in the second. The first layer used the rectified linear unit activation function and the second layer had the leaky rectified linear unit activation function, but I decided to use leaky units in both layers to prevent dying neurons. The batch size of the best result was 256, but as I found in my last experiment batch size has very little impact on accuracy, so I decided to go with 756 as this would be quicker to train.

The best performing network had L2 regularization in both layers, with a value of 0.01 for lambda in the first layer and 0.005 in the second layer. Considering the results from the last experiment it seemed likely that a higher accuracy could be achieved with reduced values of lambda, as the best found values of lamda were much higher than the optimal I found in the last experiment. So I decided to optimise this first. To simplify the problem I decided to apply the same value of lambda and means of regularization to both hidden layers, as this should not have had a significant impact on accuracy and simplifies optimization.

First I investigated using dropout alone over 200 epochs without L1 or L2 regularization, with a dropout layer after each hidden layer. I decided to experiment using 4-fold cross validation and only 50 of the training data to save time. I experimented with the dropout rate, starting with a value of 0.1 and increasing it by 0.1 each time until accuracy began to fall. I found 0.1 to be the optimal dropout rate, so then experimented with a dropout rate of 0.02, increasing it by 0.02 each time until I found the best accuracy. I found a dropout rate of 0.08 gave the best mean accuracy of 33.108% with a standard deviation of 0.0265%, the loss and accuracy of the test set over each of the 4 folds with this dropout rate over the 200 epochs is displayed below.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/Neural%20Networks/Experiment%2028%20-%20Best%20dropout%20rate%20(0.08).PNG" alt="Loss and accuracy over test set of all 4-folds with dropout rate of 0.08" style="width: 10px;"/>

In experiments 29 and 30 I experimented with using only L2 and L1 regularization respectively, optimising the value of lambda for each using 4-fold cross validation over 50% of the training data.

The best value of lambda for L2 regularization was found to be 0.0001 and the accuracy and loss on the test set of all 4 folds is displayed below.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/Neural%20Networks/Experiment%2029%20-%20Best%20lambda%20(0.0001)%20for%20L2.PNG" alt="Loss and accuracy over test set of all 4-folds with L2 regularization and lambda as 0.0001" style="width: 10px;"/>

The best value of lambda for L1 regularization was found to be 0.00003 and the accuracy and loss on the test set of all 4 folds is displayed below.

<img src="https://github.com/KieranLitschel/PredictingClosingPriceTomorrow/blob/master/Results/4-class%20problem/Neural%20Networks/Experiment%2030%20-%20Best%20lambda%20(0.00003)%20for%20L1.PNG" alt="Loss and accuracy over test set of all 4-folds with L1 regularization and lambda as 0.00003" style="width: 10px;"/>
