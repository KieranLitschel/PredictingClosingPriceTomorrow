import bs4 as bs
import requests
import mysql.connector
from mysql.connector import MySQLConnection, Error
import AlphaVantageWrapper as AVW
import datetime
import time


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


class DBManager:
    def __init__(self, apiKey, pwrd):
        self.av = AVW.AlphaVantage(apiKey)
        self.pwrd = pwrd

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

    def timeseriesToArgs(self, ticker, points, history, args, lastUpdated=datetime.date.min):
        positionOfPrev = 1
        for point in points:
            pointInHistory = history.get(point)
            date = pointToDate(point)
            # We do not add records to the database that are recorded as today, as the values vary over the day
            if (date - lastUpdated).days > 0 and (date - datetime.date.today()).days != 0:
                open = float(pointInHistory.get('1. open'))
                high = float(pointInHistory.get('2. high'))
                low = float(pointInHistory.get('3. low'))
                close = float(pointInHistory.get('4. close'))
                volume = int(pointInHistory.get('5. volume'))
                if positionOfPrev == len(points):
                    if lastUpdated == datetime.date.min:
                        prevDate = datetime.date.min
                    else:
                        result = self.select("SELECT MAX(date) FROM timeseriesdaily WHERE ticker = %s", (ticker,))
                        prevDate = result[0][0]
                else:
                    prevDate = points[positionOfPrev]
                args.append((ticker, date, prevDate, open, high, low, close, volume))
            positionOfPrev += 1

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
        query = "INSERT INTO timeseriesdaily(ticker,date,prevDate,open,high,low,close,volume) " \
                "VALUES(%s,DATE(%s),DATE(%s),%s,%s,%s,%s,%s)"
        self.insert(query, args, True)

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
        query = "INSERT INTO timeseriesdaily(ticker,date,prevDate,open,high,low,close,volume) " \
                "VALUES(%s,DATE(%s),DATE(%s),%s,%s,%s,%s,%s)"
        self.insert(query, timeseriesArgs, True)
        print('All stocks added')

    def updateAllStocks(self):
        tickersNLastUpdated = self.select("SELECT ticker, lastUpdated FROM tickers", '')
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
        query = "INSERT INTO timeseriesdaily(ticker,date,prevDate,open,high,low,close,volume) " \
                "VALUES(%s,DATE(%s),DATE(%s),%s,%s,%s,%s,%s)"
        self.insert(query, insertArgs, True)
        query = "UPDATE tickers SET lastUpdated = DATE(%s) WHERE ticker = %s;"
        self.insert(query, updateArgs, True)
        print('All stocks updated')
