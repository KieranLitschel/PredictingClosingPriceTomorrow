import bs4 as bs
import requests
import mysql.connector
from mysql.connector import MySQLConnection, Error
import AlphaVantageWrapper as AVW
import datetime
import time
import random
import math
import numpy as np
from numpy import nonzero


# Modified version of method from https://pythonprogramming.net/sp500-company-list-python-programming-for-finance/
def getSP500Tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickersNSectors = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        sector = row.findAll('td')[3].text
        tickersNSectors.append((ticker, sector))
    return tickersNSectors


def pointToDate(point):
    dateComps = str(point).split('-')
    date = datetime.date(int(dateComps[0]), int(dateComps[1]), int(dateComps[2]))
    return date


def smaPDiff(series, period):
    if len(series) >= period:
        sma = sum(series[len(series) - period: len(series)]) / period
        close = series[-1]
        return ((sma - close) / close) * 100
    else:
        return None


class DBManager:
    def __init__(self, apiKey, pwrd):
        self.av = AVW.AlphaVantage(apiKey)
        self.pwrd = pwrd
        self.insertAllTSDQuery = "INSERT INTO timeseriesdaily(ticker,date,open,high,low,close,adjClose,volume,adjClosePChange,pDiffClose5SMA,pDiffClose8SMA,pDiffClose13SMA) " \
                                 "VALUES(%s,DATE(%s),%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"

    def insert(self, query, args, many=False):
        try:
            conn = mysql.connector.connect(host='localhost', database='stocks', user='root', password=self.pwrd)
            cursor = conn.cursor()
            if many:
                # If this fails you may need to increase the size of max_allowed_packet in the my.ini file for the
                # server
                if len(args) > 100000:
                    # Implemented this as I found that if the insertion had more than 100k args it failed
                    print('Beginning batch insertion into the database...')
                    batchNo = 1
                    for i in range(100000, len(args), 100000):
                        print('Inserting %s to %s' % (i - 100000, i))
                        cursor.executemany(query, args[i - 100000: i])
                        batchNo += 1
                        conn.commit()
                    if i < len(args) - 1:
                        print('Inserting %s to %s' % (i, len(args)))
                        cursor.executemany(query, args[i: len(args)])
                else:
                    cursor.executemany(query, args)
            else:
                cursor.execute(query, args)
            conn.commit()
        except Error as e:
            print(e)
        finally:
            cursor.close()
            conn.close()

    def select(self, query, args):
        try:
            conn = mysql.connector.connect(host='localhost', database='stocks', user='root', password=self.pwrd)
            cursor = conn.cursor()
            cursor.execute(query, args)
            return cursor.fetchall()
        except Error as e:
            print(e)
        finally:
            cursor.close()
            conn.close()

    def formClassBands(self, noOfClasses):
        adjClosePChanges = self.select("SELECT adjClosePChange FROM timeseriesdaily ORDER BY adjClosePChange ASC", ())
        bandSize = math.floor(len(adjClosePChanges) / noOfClasses)
        bands = []
        for i in range(1, noOfClasses):
            bands.append(adjClosePChanges[i * bandSize])
        return bands

    # There was no way around formatting the string to add the field name to the sql query, so I made this method to
    # ensure the injected field name is not malicious
    def getSafeName(self, noOfClasses, trainingPc, testPc, validationPc):
        if validationPc != 0:
            return "`%s_%s_%s_%s`" % (int(noOfClasses), int(trainingPc), int(testPc), int(validationPc))
        else:
            return "`%s_%s_%s`" % (int(noOfClasses), int(trainingPc), int(testPc))

    def updateSetMembers(self, classBands, trainingPc, testPc, validationPc=0):
        noOfClasses = len(classBands) + 1
        name = self.getSafeName(noOfClasses, trainingPc, testPc, validationPc)
        column_names = self.select(
            "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA='stocks' AND TABLE_NAME='mlsets'",
            ())
        for i in range(0, len(column_names)):
            column_names[i] = column_names[i][0]
        if name[1:-1] not in column_names:
            self.insert(
                "ALTER TABLE mlsets ADD %s INT NULL;" % self.getSafeName(noOfClasses, trainingPc, testPc, validationPc),
                ())
        bandNo = 0
        bandData = []
        shortest = float('inf')
        print("Fetching data from the database, %.2f%% complete." % (bandNo * 100 / noOfClasses))
        bandData.append(self.select(
            "SELECT t.ticker,t.date FROM timeseriesdaily AS t "
            "INNER JOIN mlsets AS m "
            "WHERE t.adjClosePChange < %s "
            "AND t.ticker = m.ticker "
            "AND t.date = m.date "
            "AND m.{0} IS NULL".format(self.getSafeName(noOfClasses, trainingPc, testPc, validationPc)),
            (classBands[0],)))
        if len(bandData[bandNo]) < shortest:
            shortest = len(bandData[bandNo])
        bandNo += 1
        print("Fetching data from the database, %.2f%% complete." % (bandNo * 100 / noOfClasses))
        while bandNo < len(classBands):
            bandData.append(
                self.select(
                    "SELECT t.ticker,t.date FROM timeseriesdaily AS t "
                    "INNER JOIN mlsets AS m "
                    "WHERE t.adjClosePChange >= %s "
                    "AND t.adjClosePChange < %s "
                    "AND t.ticker = m.ticker "
                    "AND t.date = m.date "
                    "AND m.{0} IS NULL".format(name),
                    (classBands[bandNo - 1], classBands[bandNo])))
            if len(bandData[bandNo]) < shortest:
                shortest = len(bandData[bandNo])
            bandNo += 1
            print("Fetching data from the database, %.2f%% complete." % (bandNo * 100 / noOfClasses))
        bandData.append(self.select(
            "SELECT t.ticker,t.date FROM timeseriesdaily AS t "
            "INNER JOIN mlsets AS m "
            "WHERE t.adjClosePChange >= %s "
            "AND t.ticker = m.ticker "
            "AND t.date = m.date "
            "AND m.{0} IS NULL".format(self.getSafeName(noOfClasses, trainingPc, testPc, validationPc)),
            (classBands[-1],)))
        if len(bandData[bandNo]) < shortest:
            shortest = len(bandData[bandNo])
        bandNo += 1
        print("Fetching data from the database, %.2f%% complete." % (bandNo * 100 / noOfClasses))
        args = []
        classNo = 0
        print('Determining classes...')
        argsDict = {}
        for band in bandData:
            random.shuffle(band)
            hdt = 0
            argsDict[classNo] = []
            argsDict[classNo + noOfClasses] = []
            while shortest - hdt >= 99:
                for pc in range(0, 100):
                    if pc < trainingPc:
                        args.append((classNo, band[hdt + pc][0], band[hdt + pc][1]))
                        argsDict[classNo].append((classNo, band[hdt + pc][0], band[hdt + pc][1]))
                    elif pc < trainingPc + testPc:
                        args.append((classNo + noOfClasses, band[hdt + pc][0], band[hdt + pc][1]))
                        argsDict[classNo + noOfClasses].append((classNo, band[hdt + pc][0], band[hdt + pc][1]))
                    else:
                        args.append((classNo + (2 * noOfClasses), band[hdt + pc][0], band[hdt + pc][1]))
                hdt += 100
            classNo += 1
        print('Determined classes.')
        query = "UPDATE mlsets SET {0}=%s WHERE ticker=%s AND date=%s".format(
            self.getSafeName(noOfClasses, trainingPc, testPc, validationPc))
        if len(args) == 1:
            self.insert(query, args[0])
        elif len(args) > 1:
            self.insert(query, args, True)
        print('Classes updated for field %s in table mlsets' % name)

    def getSetsFromField(self, setFieldName, reqFields, reqNotNulls = []):
        setInfo = setFieldName.split('_')
        noOfClasses = int(setInfo[0])
        query = "SELECT m.`%s`" % setFieldName
        for reqField in reqFields:
            query += ", " + reqField
        query += " FROM timeseriesdaily AS t INNER JOIN mlsets AS m " \
                 "WHERE t.ticker = m.ticker " \
                 "AND t.date = m.date " \
                 "AND m.`{0}` >= %s " \
                 "AND m.`{0}` <= %s ".format(setFieldName)
        for reqNotNull in reqNotNulls:
            query += "AND "+reqNotNull+" IS NOT NULL"
        trainX = np.array(self.select(query, (0, noOfClasses - 1)))
        trainY = trainX[:, 0].reshape(-1, 1)
        trainX = np.delete(trainX, 0, 1)
        testX = np.array(self.select(query, (noOfClasses, 2 * noOfClasses - 1)))
        testY = testX[:, 0].reshape(-1, 1)
        testX = np.delete(testX, 0, 1)
        if len(setInfo) == 3:
            return trainX, trainY, testX, testY
        if len(setInfo) == 4:
            validX = np.array(self.select(query, (2 * noOfClasses, 3 * noOfClasses - 1)))
            validY = validX[:, 0].reshape(-1, 1)
            validX = np.delete(validX, 0, 1)
            return trainX, trainY, testX, testY, validX, validY

    def timeseriesToArgs(self, ticker, points, history, args, lastUpdated=datetime.date.min):

        maxSMA = 13
        if lastUpdated == datetime.date.min:
            addingNewStock = True
        else:
            addingNewStock = False
        closeHist = []
        if not addingNewStock:  # Finish functionality in for loop
            query = "SELECT adjClose FROM timeseriesdaily " \
                    "WHERE ticker=%s " \
                    "AND date<=DATE(%s) " \
                    "ORDER BY date DESC LIMIT %s"
            closes = self.select(query, (ticker, lastUpdated, maxSMA))
            for close in reversed(closes):
                closeHist.append(close[0])
        first = True
        for point in reversed(points):
            pointInHistory = history.get(point)
            date = pointToDate(point)
            # We do not add records to the database that are recorded as today, as the values vary over the day
            if (date - lastUpdated).days >= 0 and (date - datetime.date.today()).days != 0:
                open = float(pointInHistory.get('1. open'))
                high = float(pointInHistory.get('2. high'))
                low = float(pointInHistory.get('3. low'))
                close = float(pointInHistory.get('4. close'))
                adjClose = float(pointInHistory.get('5. adjusted close'))
                volume = int(pointInHistory.get('6. volume'))
                if first:
                    if addingNewStock:
                        adjClosePChange = None
                    else:
                        result = self.select(
                            "SELECT adjClose FROM timeseriesdaily WHERE ticker = %s  ORDER BY date DESC LIMIT 1",
                            (ticker,))
                        adjCloseBefore = result[0][0]
                        adjClosePChange = ((adjClose - adjCloseBefore) / adjCloseBefore) * 100
                    first = False
                else:
                    adjClosePChange = ((adjClose - adjCloseBefore) / adjCloseBefore) * 100
                closeHist.append(adjClose)
                pDiffClose5SMA = smaPDiff(closeHist, 5)
                pDiffClose8SMA = smaPDiff(closeHist, 8)
                pDiffClose13SMA = smaPDiff(closeHist, 13)
                args.append((ticker, date, open, high, low, close, adjClose, volume, adjClosePChange,
                             pDiffClose5SMA, pDiffClose8SMA, pDiffClose13SMA))
                adjCloseBefore = adjClose

    def addNewStock(self, ticker, sector):
        lastUpdated = datetime.date.today()
        history = self.av.getDailyHistory(AVW.OutputSize.FULL, ticker)
        points = list(history.keys())
        firstDay = pointToDate(points[-1])
        query = "INSERT INTO tickers(ticker,sector,firstDay,lastUpdated) " \
                "VALUES(%s,%s,DATE(%s),DATE(%s))"
        args = (ticker, sector, firstDay, lastUpdated)
        self.insert(query, args)
        args = []
        self.timeseriesToArgs(ticker, points, history, args)
        self.insert(self.insertAllTSDQuery, args, True)
        print('Stock added successfully')

    def addManyNewStocks(self, tickersNSectors):
        completed = 0
        tickersArgs = []
        timeseriesArgs = []
        for (ticker, sector) in tickersNSectors:
            print("Fetching stock data, %.2f%% complete." % (completed * 100 / len(tickersNSectors)))
            lastUpdated = datetime.date.today()
            history = self.av.getDailyHistory(AVW.OutputSize.FULL, ticker)
            points = list(history.keys())
            firstDay = pointToDate(points[-1])
            tickersArgs.append((ticker, sector, firstDay, lastUpdated))
            self.timeseriesToArgs(ticker, points, history, timeseriesArgs)
            time.sleep(1)  # Can only make ~1 request to the API per second
            completed += 1
        query = "INSERT INTO tickers(ticker,sector,firstDay,lastUpdated) " \
                "VALUES(%s,%s,DATE(%s),DATE(%s))"
        self.insert(query, tickersArgs, True)
        self.insert(self.insertAllTSDQuery, timeseriesArgs, True)
        print('All stocks added')

    def readdAllStocks(self):
        tickersNSectors = self.select("SELECT ticker,sector FROM tickers", '')
        self.insert("DELETE FROM tickers", ())
        self.addManyNewStocks(tickersNSectors)

    def updateAllStocks(self):
        tickersNLastUpdated = self.select("SELECT ticker, lastUpdated FROM tickers", '')
        self.updateStocks(tickersNLastUpdated)
        print('All stocks updated')

    def updateStocks(self, tickersNLastUpdated):
        insertArgs = []
        updateArgs = []
        completed = 0
        for (ticker, lastUpdated) in tickersNLastUpdated:
            print("Fetching stock data, %.2f%% complete." % (completed * 100 / len(tickersNLastUpdated)))
            if (lastUpdated - datetime.date.today()).days != 0:
                # It's neccessary to keep track of today as bugs can occur if an update occurs over 2 days e.g. if it is
                # started at 11:59pm one night and continues into the next day
                today = datetime.date.today()
                updateArgs.append((today, ticker))
                if (today - lastUpdated).days > 100:
                    history = self.av.getDailyHistory(AVW.OutputSize.COMPACT, ticker)
                else:
                    history = self.av.getDailyHistory(AVW.OutputSize.FULL, ticker)
                points = list(history.keys())
                self.timeseriesToArgs(ticker, points, history, insertArgs, lastUpdated)
                time.sleep(1)  # Can only make ~1 request to the API per second
            completed += 1
        if len(insertArgs) > 1:
            self.insert(self.insertAllTSDQuery, insertArgs, True)
        else:
            insertArgs = insertArgs[0]
            self.insert(self.insertAllTSDQuery, insertArgs)
        query = "UPDATE tickers SET lastUpdated = DATE(%s) WHERE ticker = %s;"
        if len(updateArgs) > 1:
            self.insert(query, updateArgs, True)
        else:
            updateArgs = updateArgs[0]
            self.insert(query, updateArgs)
        print("100% complete.")
