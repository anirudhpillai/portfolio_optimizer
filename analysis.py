import os
import pandas as pd
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

def test_run():
    # Read data
    dates = pd.date_range('2012-01-01', '2012-12-31')
    symbols = ['SPY', 'AAPL', 'MSFT', 'GOOG']
    df = get_data(symbols, dates)

    df = compute_cumulative_returns(df)
    df[:1000]['GOOG'].plot()
    plt.show()

    # dr = compute_daily_returns(df)
    # dr[:100][['GOOG', 'AAPL']].plot()
    # plt.show()

    # # Compute Bollinger Bands
    # rm_SPY = get_rolling_mean(df['SPY'], window=20)
    #
    # # Compute rolling standard deviation
    # rstd_SPY = get_rolling_std(df['SPY'], window=20)
    #
    # # Compute upper and lower bands
    # upper_band, lower_band = get_bollinger_bands(rm_SPY, rstd_SPY)
    #
    # # Plot raw SPY values, rolling mean and Bollinger Bands
    # ax = df['SPY'].plot(title="Bollinger Bands", label='SPY')
    # rm_SPY.plot(label='Rolling mean', ax=ax)
    # upper_band.plot(label='upper band', ax=ax)
    # lower_band.plot(label='lower band', ax=ax)
    #
    # # Add axis labels and legend
    # ax.set_xlabel("Date")
    # ax.set_ylabel("Price")
    # ax.legend(loc='upper left')
    # plt.show()


if __name__ == "__main__":
    test_run()
