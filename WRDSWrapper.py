import wrds
import numpy as np


class WRDS:
    def __init__(self, wrds_username):
        self.db = wrds.Connection(wrds_username=wrds_username)

    def getPermnos(self, tickers):
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
            permnos[int(row[1])] = row[0]
        return permnos

    def getFinancials(self, permnos, columns):
        query = "SELECT permno,public_date"
        if type(columns) is list:
            for column in columns:
                query += "," + column
        else:
            query += ","+columns
        query += " FROM wrdsapps_finratios.firm_ratio WHERE "
        if type(permnos) is list:
            first = True
            for permno in permnos:
                if first:
                    query += "permno="+str(permno)
                    first = False
                else:
                    query += " OR permno="+str(permno)
        else:
            query += "permno="+permnos
        financials = np.array(self.db.raw_sql(query))
        return financials
