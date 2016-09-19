import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web


def get_data(symbols, dates):
    """Read stock data (adjusted close) for given symbols from Yahoo finance"""
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols:
        symbols.insert(0, 'SPY')

    for symbol in symbols:
        p = web.DataReader(symbol, "yahoo", dates)
        df_temp = p.loc[:, ['Adj Close']]
        df_temp = df_temp.rename(columns={'Adj Close': symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])

    return df

def plot_data(df, title="Stock prices"):
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    plt.show()


def get_rolling_mean(values, window):
    """Return rolling mean of given values, using specified window size."""
    return pd.rolling_mean(values, window=window)


def get_rolling_std(values, window):
    """Return rolling standard deviation of given values, using specified window size."""
    return pd.rolling_std(values, window)


def get_bollinger_bands(rm, rstd):
    """Return upper and lower Bollinger Bands."""
    upper_band = rm + 2*rstd
    lower_band = rm - 2*rstd
    return upper_band, lower_band


def compute_daily_returns(df):
    """Compute and return the daily return values"""
    daily_returns = (df / df.shift(1)) - 1
    daily_returns.ix[0, :] = 0
    return daily_returns


def compute_cumulative_returns(df):
    """Compute and return the cumulative return values"""
    cumulative_returns = (df / df.iloc[0]) - 1
    return cumulative_returns


def fill_missing_values(df_data):
    """Fill missing values in data frame, in place."""
    df_data.fillna(method="ffill", inplace=True)
    df_data.fillna(method="bfill", inplace=True)


def compute_daily_portfolio_values(df, alloc, start_val):
    normed = df / df.ix[0]
    alloced = normed * alloc
    pos_vals = alloced * start_val
    port_val = pos_vals.sum(axis=1)
    return port_val

def compute_portfolio_stats(port_val):
    daily_returns = ((port_val / port_val.shift(1)) - 1)[1:]
    cumulative_returns = (port_val[-1] / port_val[0]) - 1
    avg_daily_returns = daily_returns.mean()
    std_daily_returns = daily_returns.std()
    sharpe_ratio = (avg_daily_returns / std_daily_returns) * (252 ** 0.5)  # daily risk free rate taken as 0
    return cumulative_returns, avg_daily_returns, std_daily_returns, sharpe_ratio


def test_run():
    # Read data
    dates = pd.date_range('2012-01-01', '2012-12-31')
    symbols = ['SPY', 'AAPL', 'MSFT', 'GOOG']
    df = get_data(symbols, dates)

    port_val = compute_daily_portfolio_values(df, [0, 0.2, 0, 0.8], 1000000)
    print(compute_portfolio_stats(port_val))
    # df = compute_cumulative_returns(df)
    # df[:1000]['GOOG'].plot()
    # plt.show()

    # dr = compute_daily_returns(df)
    # dr[:100][['GOOG', 'AAPL']].plot()
    # plt.show()

    # plotting histogram of daily returns
    # dr.hist() gives histogram of all stocks in df
    # dr['GOOG'].hist()

    """ Scatter Plot
    daily_returns = compute_daily_returns(df)

    daily_returns.plot(kind='scatter', x='SPY', y='GOOG')
    # Getting beta and alpha
    beta_GOOG, alpha_GOOG = np.polyfit(daily_returns['SPY'], daily_returns['GOOG'], 1)
    # plt.plot(x, m*x+c, line, color)
    plt.plot(daily_returns['SPY'], beta_GOOG*daily_returns['SPY'] + alpha_GOOG, '-', color='r')

    daily_returns.plot(kind='scatter', x='SPY', y='AAPL')
    # Getting beta and alpha
    beta_AAPL, alpha_AAPL = np.polyfit(daily_returns['SPY'], daily_returns['AAPL'], 1)
    # plt.plot(x, m*x+c, line, color)
    plt.plot(daily_returns['SPY'], beta_AAPL*daily_returns['SPY'] + alpha_AAPL, '-', color='r')

    plt.show()

    # Correlation
    print(daily_returns.corr(method='pearson'))
    """

    """ Compute Bollinger Bands
    rm_SPY = get_rolling_mean(df['SPY'], window=20)
    rstd_SPY = get_rolling_std(df['SPY'], window=20)
    upper_band, lower_band = get_bollinger_bands(rm_SPY, rstd_SPY)

    ax = df['SPY'].plot(title="Bollinger Bands", label='SPY')
    rm_SPY.plot(label='Rolling mean', ax=ax)
    upper_band.plot(label='upper band', ax=ax)
    lower_band.plot(label='lower band', ax=ax)

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc='upper left')
    plt.show()
    """

if __name__ == "__main__":
    test_run()
