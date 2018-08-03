import wrds
import numpy as np


class WRDS:
    def __init__(self, wrds_username):
        self.db = wrds.Connection(wrds_username=wrds_username)

    def getPermno(self, tickers):
        query = "SELECT DISTINCT ticker, permno FROM crsp.dse WHERE"
        if type(tickers) is list:
            first = True
            for ticker in tickers:
                if first:
                    query += " ticker=\'" + ticker + "\'"
                    first = False
                else:
                    query += " OR ticker=\'" + ticker + "\'"
        else:
            query += " ticker=\'" + tickers + "\'"
        response = np.array(self.db.raw_sql(query))
        permnos = {}
        for row in response:
            if permnos.get(row[0]) is None:
                permnos[row[0]] = [row[1]]
            else:
                permnos[row[0]].append(row[1])
        return permnos
