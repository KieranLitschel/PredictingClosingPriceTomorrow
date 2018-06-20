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

    def insert(self, query, args, many):
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

    def addNewStock(self, ticker, sector):
        history = self.av.getDailyHistory(AVW.OutputSize.FULL, ticker)
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

    def addManyNewStocks(self, tickersNSectors):
        history = {}
        completed = 0
        args = []
        for (ticker, sector) in tickersNSectors:
            print("Fetching stock data, %.2f%% complete." % (completed * 100 / len(tickersNSectors)))
            history[ticker] = self.av.getDailyHistory(AVW.OutputSize.FULL, ticker)
            # The API sometimes does not return the response we require if its overloaded, in which case we wait a bit
            # and try again
            while history[ticker] is None:
                print('Failed on ticker %s of %s. Retrying...' % (completed, len(tickersNSectors)))
                time.sleep(1.5)
                history[ticker] = self.av.getDailyHistory(AVW.OutputSize.FULL, ticker)
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
        for (ticker, sector) in tickersNSectors:
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
