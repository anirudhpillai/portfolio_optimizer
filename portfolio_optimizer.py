import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import scipy.optimize as spo


def get_data(symbols, start_date, end_date):
    """Read stock data (adjusted close) for given symbols from Yahoo finance"""
    df = pd.DataFrame(index=
                    pd.date_range(start_date, end_date))

    for symbol in symbols:
        p = web.DataReader(symbol, "yahoo", start_date, end_date)
        df_temp = p.loc[:, ['Adj Close']]
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)

    df = df.dropna(axis=0, how='any') # dropping Nan

    return df


def portfolio_optimizer(symbols, start_date, end_date):
    df = get_data(symbols, start_date, end_date)

    def compute_sharpe_ratio(alloc):
        """Calculate sharpe ratio of the portfolio when given the allocation"""
        # create portfolio from df
        normed = df / df.ix[0]
        alloced = normed * alloc
        port_val = alloced.sum(axis=1)

        daily_returns = ((port_val / port_val.shift(1)) - 1)[1:]
        avg_daily_returns = daily_returns.mean()
        std_daily_returns = daily_returns.std()
        # daily risk free rate taken as 0
        sharpe_ratio = (avg_daily_returns / std_daily_returns) * (252 ** 0.5)
        return -sharpe_ratio # negative as we want to maximise sharpe ratio using a minimiser

    # initial alloc
    alloc = np.ones(len(symbols)) / len(symbols)
    bounds = [(0, 1) for i in symbols]
    # Says one minus the sum of all variables must be zero
    cons = ({'type': 'eq', 'fun': lambda x:  1 - sum(x)})
    return spo.minimize(compute_sharpe_ratio, alloc,bounds=bounds,
                        constraints=cons, method='SLSQP')


def main():
    symbols = ['AAPL', 'GOOG', 'MSFT']
    print(portfolio_optimizer(symbols, '2011-01-01', '2015-12-31'))


if __name__ == "__main__":
    main()
