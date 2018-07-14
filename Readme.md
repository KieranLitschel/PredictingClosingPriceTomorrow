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

<img src="https://github.com/KieranLitschel/Images/blob/master/KNN%20with%204%20features.png" alt="KNN with 4 features, plateus at approx 31% accuracy" style="width: 10px;"/>

Conclusions:
* Begins to plateu at about 100 neighbours, trailing off at around 31% accuracy.
* There are 4 classes so we would expect an accuracy of around 25% if it was classifying randomly and there was no pattern in the data, so trailing off at 31% accuracy suggests there is some pattern.
* Considering we are using around 1.9 million samples for training and testing, we should see an increase in accuracy if we increase the number of features.
### Results of adding feature RSI
Note that:
* RSI is the relative strength indicator ([see here](https://www.investopedia.com/terms/r/rsi.asp) for more information.
* I used the features adjClosePChange, pDiffClose5SMA, pDiffClose8SMA, pDiffClose13SMA, and RSI, although on the red line plotted below I show the results from the last experiment for comparison which I did not train using RSI.

<img src="https://github.com/KieranLitschel/Images/blob/master/KNN%20with%205%20features%20(rsi).png" alt="KNN with and without RSI" style="width: 10px;"/>

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

<img src="https://github.com/KieranLitschel/Images/blob/master/KNN%20with%208%20features%20(bollinger%20bands).png" alt="KNN with and without RSI" style="width: 10px;"/>

Conclusions:
* Adding the 3 features led to a 0.77% increase in accuracy.
* I was expecting much larger increases in accuracy than I am achieving, as at this rate I don't think I will be able to surpass an accuracy of 40% using KNN, suggesting it is not well suited to the problem. Despite this I will continue to test with KNN while implementing new features, as it is very quick to test and allows me to judge the progress I am making. Once I've generate all the features I have planned I will train a neural network to solve the problem, which I expect will perform much better.
### Results of adding percentage difference between SMAs
I hypothesised that there might be more that could be learnt from looking at the difference between the 5, 8, and 13 day SMAs with each other, so I added 3 features to capture this. Below are the results of adding these 3 features.

<img src="https://github.com/KieranLitschel/Images/blob/master/KNN%20with%2011%20features%20(difference%20between%20SMAs).png" alt="KNN with and without difference between SMAs" style="width: 10px;"/>

It is clear that adding these features does not help improve accuracy, probably because all they have to contribute is captured by the first 3 features I added related to SMAs, hence I will not use them as features in future, but may consider adding them again when applying PCA.
### Results of adding MACD
The [MACD technical indicator](https://www.investopedia.com/terms/m/macd.asp) is used to identify when is best to enter and exit a trade, so I thought it might be able to give some information that will help determine what will happen to the stock tomorrow. The first feature I added was the MACD histogram, which is the difference between the MACD value (the difference between the fast and slow EMA) and the signal (the EMA of the MACD value). This value should help identify crossovers and dramatic rises.

<img src="https://github.com/KieranLitschel/Images/blob/master/KNN%20with%209%20features%20(MACD%20Histogram).png" alt="KNN with and without MACD Histogram" style="width: 10px;"/>

The results of this experiment are very similair to the last one, but note how the two lines do not cross beyond 30 neighbours, and we also see a 0.13% increase in acuracy when k=100. For these reasons I think that the MACD is improving the accuracy of the classifier. However the increase in accuracy is rather disappointing, and I hypothesised that an issue may be that it is impossible for the model to tell which way the MACD histogram is moving and how fast, so I decided to add a feature to represent the difference between the MACD histogram at the current period and at the last period.

<img src="https://github.com/KieranLitschel/Images/blob/master/KNN%20with%2010%20features%20(MACD%20Histogram%20delta).png" alt="KNN with and without MACD Histogram delta" style="width: 10px;"/>

But from looking at the results from the experiment above, there is very little difference between the lines with and without the difference, suggesting there is nothing more to be learned from this feature. Hence I will leave the difference out of future models for the time being.
### Results of adding Stochastic Oscilator
I chose to add the [stochastic oscilator](https://en.wikipedia.org/wiki/Stochastic_oscillator) as up to this point all features have been generated using the adjusted closing price, but the stochastic oscilator calculates momentum using the high, low, and close. The downside to this is that as the Alpha Vantage API does not supply an adjusted high and low price, so I have to use the raw high, low, and close prices. This means that stock splits and dividends will lead the oscilator to give false signals, but these are not frequent events so I hope that this indicator will still be able to give me a boost in accuracy.
