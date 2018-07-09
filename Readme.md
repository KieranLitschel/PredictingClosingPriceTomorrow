# Introduction
This is the beginning of a project where I will be using the 20 year history of the current members of the S&P 500 to train a neural network to be able to classify the closing price tomorrow of any given stock into a fixed number of bounds.
# Progress Log
## Predicting using KNN
I have decided to try KNN as it is very simple to train, this will also give me a target to beat when training my neural network. I will classify the the percentage change in closing price tomorrow as one of four of the following classes: adjClosePChange < -1%, -1%<= adjClosePChange < 0%, 0% <= adjClosePChange<1, and adjClosePChange>=1%. The data with split as 80% for training, and 20% for testing.
### Results of using features: adjClosePChange, pDiffClose5SMA, pDiffClose8SMA, pDiffClose13SMA
<img src="https://github.com/KieranLitschel/Images/blob/master/KNN%20with%204%20features.png" alt="KNN with 4 features, plateus at approx 31% accuracy" style="width: 10px;"/>
