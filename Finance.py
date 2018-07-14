import statistics


class FinanceCalculator:

    def __init__(self, seriesSoFar=None, rsiPeriod=14, emas=[12, 26]):
        self.upward = []
        self.averageUpward = []
        self.downward = []
        self.averageDownward = []
        self.rsiPeriod = rsiPeriod
        if seriesSoFar is not None:
            for i in range(1, len(seriesSoFar) + 1):
                self.RSI(seriesSoFar[0:i])
        self.lastEMA = {}
        self.prevDifferences = []
        self.prevSignal = None
        self.highs = []
        self.lows = []
        self.pKs = []
        self.pdms = []
        self.ndms = []
        self.trs = []
        self.closes = []
        self.pdisMinusNdis = []

    def reset(self):
        self.__init__()

    def smaPDiff(self, series, period):
        if len(series) >= period:
            sma = sum(series[len(series) - period: len(series)]) / period
            close = series[-1]
            return ((sma - close) / close) * 100
        else:
            return None

    def pDiffBetweenSMAs(self, series, periods):
        SMAs = []
        for period in periods:
            if len(series) >= period:
                SMAs.append(sum(series[len(series) - period: len(series)]) / period)
            else:
                SMAs.append(None)
        pDiffSMAs = []
        for i in range(0, len(SMAs)):
            for j in range(i + 1, len(SMAs)):
                if SMAs[i] is not None and SMAs[j] is not None and SMAs[j] != 0:
                    pDiffSMAs.append((SMAs[i] - SMAs[j]) / SMAs[j])
                else:
                    pDiffSMAs.append(None)
        return pDiffSMAs

    # The equation for RSI can vary, but I am using the one from this video https://www.youtube.com/watch?v=WZbOeFsSirM
    def RSI(self, series):
        period = self.rsiPeriod  # These first two variables just help readability of the code
        lenSeries = len(series)
        if lenSeries >= 2:
            if series[lenSeries - 1] > series[lenSeries - 2]:
                self.upward.append(series[lenSeries - 1] - series[lenSeries - 2])
                self.downward.append(0)
            elif series[lenSeries - 1] < series[lenSeries - 2]:
                self.downward.append(series[lenSeries - 2] - series[lenSeries - 1])
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
            if self.averageDownward[-1] == 0:
                return 100
            else:
                relativeStrength = self.averageUpward[-1] / self.averageDownward[-1]
                return 100 - (100 / (relativeStrength + 1))
        else:
            return None

    def bollingerBandsPDiff(self, series, n=20, k=2):
        if len(series) >= n:
            close = series[-1]
            period = series[len(series) - n: len(series)]
            sman = sum(period) / n
            periodStd = statistics.pstdev(period)
            upperBand = sman + periodStd * k
            lowerBand = sman - periodStd * k
            pDiffCloseUpperBand = ((upperBand - close) / close) * 100
            pDiffCloseLowerBand = ((lowerBand - close) / close) * 100
            pDiffSmaAbsBand = ((upperBand - sman) / sman) * 100
            return pDiffCloseUpperBand, pDiffCloseLowerBand, pDiffSmaAbsBand
        else:
            return None, None, None

    def EMA(self, series, period, label=""):
        if len(series) >= period:
            prevEMA = self.lastEMA.get(str(period) + label)
            if prevEMA is None:
                ema = sum(series[len(series) - period: len(series)]) / period
            else:
                factor = 2 / (period + 1)
                ema = ((series[-1] - prevEMA) * factor) + prevEMA
            self.lastEMA[str(period) + label] = ema
        else:
            ema = None
        return ema

    def MACD(self, series):
        if len(series) >= 26:
            fastEMA = self.EMA(series, 12, "MACDfastEMA")
            slowEMA = self.EMA(series, 26, "MACDslowEMA")
            difference = fastEMA - slowEMA
            self.prevDifferences.append(difference)
            if len(self.prevDifferences) >= 9:
                signal = self.EMA(self.prevDifferences, 9, "MACDsignal")
                histogram = difference - signal
            else:
                signal = None
                histogram = None
        else:
            difference = None
            signal = None
            histogram = None
        return difference, signal, histogram

    def updateHighLowClose(self, high, low, close):
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)

    def stochasticOscilator(self):
        close = self.closes[-1]
        fastKPeriod = 5
        slowKPeriod = 3
        if len(self.lows) >= fastKPeriod:
            l5 = min(self.lows[len(self.lows) - fastKPeriod: len(self.lows)])
            h5 = max(self.highs[len(self.highs) - fastKPeriod: len(self.highs)])
            if h5-l5!=0:
                pK = ((close - l5) / (h5 - l5)) * 100
                self.pKs.append(pK)
                if len(self.pKs) >= slowKPeriod:
                    pD = sum(self.pKs[len(self.pKs) - slowKPeriod: len(self.pKs)]) / slowKPeriod
                else:
                    pD = None
            else:
                self.pKs=[]
                pK=None
                pD=None
        else:
            pK = None
            pD = None
        return pK, pD

    def ADX(self):
        pdi=None
        ndi=None
        adx=None
        if len(self.closes)>=2:
            self.trs.append(max(self.highs[-1] - self.lows[-1], abs(self.highs[-1] - self.closes[-2]),
                                abs(self.lows[-1] - self.closes[-2])))
            moveUp = self.highs[-1] - self.highs[-2]
            moveDown = self.lows[-2] - self.lows[-1]
            if moveUp > moveDown and moveUp > 0:
                pdm = moveUp
            else:
                pdm = 0
            self.pdms.append(pdm)
            if moveDown > moveUp and moveDown > 0:
                ndm = moveDown
            else:
                ndm = 0
            self.ndms.append(ndm)
            if len(self.pdms) >= 14:
                pdmEMA = self.EMA(self.pdms, 14, "ADXpdm")
                ndmEMA = self.EMA(self.ndms, 14, "ADXndm")
                atr = self.EMA(self.trs, 14, "ADXatr")
                pdi = 100 * (pdmEMA / atr)
                ndi = 100 * (ndmEMA / atr)
                self.pdisMinusNdis.append(abs(pdi - ndi))
                if len(self.pdisMinusNdis) >= 14:
                    adx = 100 * self.EMA(self.pdisMinusNdis,14) / (pdi + ndi)
        return pdi,ndi,adx
