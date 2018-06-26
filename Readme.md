# Introduction
This is the beginning of a project where I will be using the 20 year history of the current members of the S&P 500 to train a neural network to be able to classify the closing price tomorrow of any given stock into a fixed number of bounds.
# Issues I encountered and how I handled them
## Cleaning up the data
I made use of the Alpha Vantage API to fetch historical data but noticed that there are some errors in the historic data. With nearly 2.3 million rows in the database it would have been infeasible to handle these individually, so instead I developed some heuristics to attempt to detect these errors, so that they could be marked as anomalies and handled accordingly.
### Using the percentage change in closing price
The percentage change in closing price since yesterday will be used to classify the expected output of the network for each training sample, but I noticed that sudden changes in price as a result stock splitting can give huge percentage changes. I decided the best course of action is divide all stocks values by the number of stocks it was split into for each date prior to each split.
