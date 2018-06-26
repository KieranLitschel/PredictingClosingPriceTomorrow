# Introduction
This is the beginning of a project where I will be using the 20 year history of the current members of the S&P 500 to train a neural network to be able to classify the closing price tomorrow of any given stock into a fixed number of bounds.
# Issues I encountered and how I handled them
## Cleaning up the data
I have made use of the Alpha Vantage API to fetch historical data but have noticed that there are some errors in the historic data. With nearly 2.3 million rows in the database it would be infeasible to handle these individually, so instead I am going to develop some heuristics to attempt to detect these errors, so that I can mark them as anomalies and handle them accordingly.
### Using the percentage change in closing price
The percentage change in closing price since yesterday will be used to classify the expected output of the network for each training sample, but I have noticed that sudden changes in price as a result of things like stock splitting can give huge percentage changes. There will be no way for my model to predict events like this, so I will be marking events like these as anomalies.
