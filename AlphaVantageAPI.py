import enum
import json
import requests


class AlphaVantage:
    def __init__(self, apiKey):
        self.apiKey = apiKey

    def getDailyHistory(self, outputsize, symbol):
        if outputsize == OutputSize.FULL:
            response = requests.get(
                "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={0}&outputsize=full&apikey={1}".format(
                    symbol, self.apiKey))
        else:
            response = requests.get(
                "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={0}&apikey={1}".format(symbol,
                                                                                                            self.apiKey))
        history = response.json()
        return history

class OutputSize(enum.Enum):
    FULL = enum.auto()
    COMPACT = enum.auto()
