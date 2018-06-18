import bs4 as bs
import requests
import mysql.connector
from mysql.connector import MySQLConnection, Error
import AlphaVantageAPI as AVAPI
import datetime


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
