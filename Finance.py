class FinanceCalculator:

    def __init__(self, seriesSoFar=None, rsiPeriod=14):
        self.upward = []
        self.averageUpward = []
        self.downward = []
        self.averageDownward = []
        self.rsiPeriod = rsiPeriod
        if seriesSoFar is not None:
            for i in range(1, len(seriesSoFar)+1):
                self.RSI(seriesSoFar[0:i])

    def reset(self):
        self.__init__()

    def smaPDiff(self, series, period):
        if len(series) >= period:
            sma = sum(series[len(series) - period: len(series)]) / period
            close = series[-1]
            return ((sma - close) / close) * 100
        else:
            return None

    # The equation for RSI can vary, but I am using the one from this video https://www.youtube.com/watch?v=WZbOeFsSirM
    def RSI(self, series):
        period = self.rsiPeriod  # These first two variables just help readability of the code
        lenSeries = len(series)
        if lenSeries >= 2:
            if series[lenSeries - 1] > series[lenSeries - 2]:
                self.upward.append(series[lenSeries - 1]-series[lenSeries - 2])
                self.downward.append(0)
            elif series[lenSeries - 1] < series[lenSeries - 2]:
                self.downward.append(series[lenSeries - 2]-series[lenSeries - 1])
                self.upward.append(0)
            else:
                self.upward.append(0)
                self.downward.append(0)
        if lenSeries > period:
            if len(self.averageUpward) == 0:
                self.averageUpward.append(sum(self.upward[len(self.upward) - period:len(self.upward)]) / period)
                self.averageDownward.append(
                    sum(self.downward[len(self.downward) - period:len(self.downward)]) / period)
            else:
                self.averageUpward.append((self.averageUpward[-1] * (period - 1) + self.upward[-1]) / period)
                self.averageDownward.append((self.averageDownward[-1] * (period - 1) + self.downward[-1]) / period)
            relativeStrength = self.averageUpward[-1] / self.averageDownward[-1]
            return 100 - (100 / (relativeStrength + 1))
        else:
            return None
