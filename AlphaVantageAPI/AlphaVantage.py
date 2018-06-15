import enum

class AlphaVantage:
    def __init__(self,apiKey):
        self.apiKey=apiKey


    def getDailyHistory(self,outputsize):
        if (outputsize==OutputSize.FULL):
            

class OutputSize(enum.Enum):
    FULL = enum.auto()
    COMPACT = enum.auto()
