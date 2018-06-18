import bs4 as bs
import requests
import mysql.connector
from mysql.connector import MySQLConnection, Error
import AlphaVantageAPI as AVAPI
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
        self.av = AVAPI.AlphaVantage(apiKey)
        self.pwrd = pwrd

    def insert(self, query, args, multi):
        try:
            conn = mysql.connector.connect(host='localhost', database='stocks', user='root', password=self.pwrd)
            cursor = conn.cursor()
            if multi:
                cursor.executemany(query, args)
            else:
                # Tells the server to expect packets up 500MB in this session (1MB default)
                cursor.execute('SET SESSION max_allowed_packet=500M')
                cursor.execute(query, args)
            conn.commit()
        except Error as e:
            print(e)
        finally:
            cursor.close()
            conn.close()

    def addNewStock(self, ticker, sector):
        history = self.av.getDailyHistory(AVAPI.OutputSize.FULL, ticker)
        points = list(history.keys())
        firstDay = pointToDate(points[-1])
        lastDay = pointToDate(points[0])
        query = "INSERT INTO tickers(ticker,sector,firstDay,lastDay) " \
                "VALUES(%s,%s,DATE(%s),DATE(%s))"
        args = (ticker, sector, firstDay, lastDay)
        self.insert(query, args, False)
        args = []
        for point in points:
            pointInHistory = history.get(point)
            date = pointToDate(point)
            open = float(pointInHistory.get('1. open'))
            high = float(pointInHistory.get('2. high'))
            low = float(pointInHistory.get('3. low'))
            close = float(pointInHistory.get('4. close'))
            volume = int(pointInHistory.get('5. volume'))
            args.append((ticker, date, open, high, low, close, volume))
        query = "INSERT INTO timeseriesdaily(ticker,date,open,high,low,close,volume) " \
                "VALUES(%s,DATE(%s),%s,%s,%s,%s,%s)"
        self.insert(query, args, True)

    def addManyNewStocks(self, tickersNSector):
        history = {}
        completed = 0
        args = []
        for (ticker, sector) in tickersNSector:
            print("%.2f%% complete." % (completed * 100 / len(tickersNSector)))
            history[ticker] = self.av.getDailyHistory(AVAPI.OutputSize.FULL, ticker)
            # The API sometimes does not return the response we require if its overloaded, in which case we wait a bit
            # and try again
            while history[ticker] is None:
                print('Failed on ticker %s of %s. Retrying.' % (completed, len(tickersNSector)))
                time.sleep(1.5)
                history[ticker] = self.av.getDailyHistory(AVAPI.OutputSize.FULL, ticker)
                if not (history[ticker] is None):
                    print('Recovered successfully.')
            points = list(history[ticker].keys())
            firstDay = pointToDate(points[-1])
            lastDay = pointToDate(points[0])
            args.append((ticker, sector, firstDay, lastDay))
            time.sleep(1.5)  # Can only make ~1 request to the API per second
            completed += 1
        query = "INSERT INTO tickers(ticker,sector,firstDay,lastDay) " \
                "VALUES(%s,%s,DATE(%s),DATE(%s))"
        self.insert(query, args, True)
        args = []
        for (ticker, sector) in tickersNSector:
            points = history[ticker].keys()
            for point in points:
                pointInHistory = history.get(ticker).get(point)
                date = pointToDate(point)
                open = float(pointInHistory.get('1. open'))
                high = float(pointInHistory.get('2. high'))
                low = float(pointInHistory.get('3. low'))
                close = float(pointInHistory.get('4. close'))
                volume = int(pointInHistory.get('5. volume'))
                args.append((ticker, date, open, high, low, close, volume))
        query = "INSERT INTO timeseriesdaily(ticker,date,open,high,low,close,volume) " \
                "VALUES(%s,DATE(%s),%s,%s,%s,%s,%s)"
        self.insert(query, args, True)
