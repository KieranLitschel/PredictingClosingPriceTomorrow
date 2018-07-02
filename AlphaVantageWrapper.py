import enum
import requests
import time


class AlphaVantage:
    def __init__(self, apiKey):
        self.apiKey = apiKey

    def requestDailyHistory(self, outputSize, ticker):
        if outputSize == OutputSize.FULL:
            response = requests.get(
                "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={0}&outputsize=full&apikey={1}".format(
                    ticker, self.apiKey))
        else:
            response = requests.get(
                "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={0}&apikey={1}".format(
                    ticker,
                    self.apiKey))
        history = response.json().get('Time Series (Daily)')
        return history

    def getDailyHistory(self, outputSize, ticker):
        history = self.requestDailyHistory(outputSize, ticker)
        # The API sometimes does not return the response we require if its overloaded, in which case we wait a bit
        # and try again
        while history is None:
            print('Last request failed. Retrying...')
            time.sleep(1.5)
            history = self.getDailyHistory(outputSize, ticker)
            if not (history is None):
                print('Request succeeded.')
        return history


class OutputSize(enum.Enum):
    FULL = enum.auto()
    COMPACT = enum.auto()
