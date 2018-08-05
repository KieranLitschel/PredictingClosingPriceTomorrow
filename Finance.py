import statistics
import numpy as np
from sklearn import linear_model
import multiprocessing


class FinanceCalculator:

    def __init__(self, seriesSoFar=None, rsiPeriod=14, n_jobs=None):
        self.upward = []
        self.averageUpward = []
        self.downward = []
        self.averageDownward = []
        self.rsiPeriod = rsiPeriod
        if seriesSoFar is not None:
            for i in range(1, len(seriesSoFar) + 1):
                self.RSI(seriesSoFar[0:i])
        self.lastEMA = {}
        self.prevDifferences = {}
        self.prevSignal = None
        self.highs = []
        self.lows = []
        self.pKs = {}
        self.pdms = []
        self.ndms = []
        self.trs = []
        self.closes = []
        self.pdisMinusNdis = []
        self.obvs = {}
        self.adjCloses = []
        if n_jobs is None:
            if multiprocessing.cpu_count() - 2 > 0:
                self.jobs = multiprocessing.cpu_count() - 2
            else:
                self.jobs = 1
        else:
            self.jobs = n_jobs

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

    def MACD(self, series, slow, fast):
        name = str(slow) + "," + str(fast)
        if self.prevDifferences.get(name) is None:
            self.prevDifferences[name] = []
        if len(series) >= fast:
            fastEMA = self.EMA(series, slow, "MACDfastEMA" + name)
            slowEMA = self.EMA(series, fast, "MACDslowEMA" + name)
            difference = fastEMA - slowEMA
            self.prevDifferences[name].append(difference)
            if len(self.prevDifferences[name]) >= 9:
                signal = self.EMA(self.prevDifferences[name], 9, "MACDsignal" + name)
                histogram = difference - signal
            else:
                signal = None
                histogram = None
        else:
            difference = None
            signal = None
            histogram = None
        return difference, signal, histogram

    def updateHighLowClose(self, high, low, close, adjClose):
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)
        self.adjCloses.append(adjClose)

    def stochasticOscilator(self, fastKPeriod, slowKPeriod):
        if len(self.lows) >= fastKPeriod:
            close = self.closes[-1]
            l5 = min(self.lows[len(self.lows) - fastKPeriod: len(self.lows)])
            h5 = max(self.highs[len(self.highs) - fastKPeriod: len(self.highs)])
            if h5 - l5 != 0:
                if self.pKs.get(str(fastKPeriod) + "," + str(slowKPeriod)) is None:
                    self.pKs[str(fastKPeriod) + "," + str(slowKPeriod)] = []
                pK = ((close - l5) / (h5 - l5)) * 100
                self.pKs[str(fastKPeriod) + "," + str(slowKPeriod)].append(pK)
                if len(self.pKs[str(fastKPeriod) + "," + str(slowKPeriod)]) >= slowKPeriod:
                    pD = sum(self.pKs[str(fastKPeriod) + "," + str(slowKPeriod)][
                             len(self.pKs[str(fastKPeriod) + "," + str(slowKPeriod)]) - slowKPeriod: len(
                                 self.pKs[str(fastKPeriod) + "," + str(slowKPeriod)])]) / slowKPeriod
                else:
                    pD = None
            else:
                self.pKs[str(fastKPeriod) + "," + str(slowKPeriod)] = []
                pK = None
                pD = None
        else:
            pK = None
            pD = None
        return pK, pD

    def ADX(self):
        pdi = None
        ndi = None
        adx = None
        if len(self.closes) >= 2:
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
                    adx = 100 * self.EMA(self.pdisMinusNdis, 14) / (pdi + ndi)
        return pdi, ndi, adx

    def OBVGrad(self, volume, period):
        if self.obvs.get(period) is None:
            self.obvs[period] = [0]
        if len(self.adjCloses) >= 2:
            adjClosePChange = ((self.adjCloses[-1] - self.adjCloses[-2]) / self.adjCloses[-2]) * 100
        else:
            return None
        if adjClosePChange > 0:
            obv = self.obvs[period][-1] + volume
        elif adjClosePChange < 0:
            obv = self.obvs[period][-1] - volume
        elif adjClosePChange == 0:
            obv = self.obvs[period][-1]
        self.obvs[period].append(obv)
        OBVGrad = None
        if len(self.obvs[period]) > period:
            OBVSample = self.obvs[period][len(self.obvs[period]) - period: len(self.obvs[period])]
            OBVSample = np.array(OBVSample)
            x = np.arange(1, period + 1).reshape(period, 1)
            regr = linear_model.LinearRegression(n_jobs=self.jobs)
            regr.fit(x, OBVSample)
            OBVGrad = regr.coef_[0]
        return OBVGrad

    def adjCloseGrad(self, period):
        adjCloseGrad = None
        if len(self.adjCloses) > period:
            adjCloseSample = np.array(self.adjCloses[len(self.adjCloses) - period: len(self.adjCloses)])
            x = np.arange(1, period + 1).reshape(period, 1)
            regr = linear_model.LinearRegression(n_jobs=self.jobs)
            regr.fit(x, adjCloseSample)
            adjCloseGrad = regr.coef_[0]
        return adjCloseGrad