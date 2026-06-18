import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator


def stock_metrics(symbol):
    ticker = yf.Ticker(symbol)

    # Historical data
    df = ticker.history(period="1y", interval="1d")

    if df.empty:
        raise ValueError(f"No data found for {symbol}")

    close = df["Close"]
    volume = df["Volume"]

    # ==========================
    # Technical Indicators
    # ==========================

    current_price = close.iloc[-1]

    # MA200
    ma200 = close.rolling(window=200).mean().iloc[-1]

    # RSI(14)
    rsi = RSIIndicator(close=close, window=14).rsi().iloc[-1]

    # Relative Volume
    avg_volume = volume.rolling(window=20).mean().iloc[-1]
    relative_volume = volume.iloc[-1] / avg_volume

    # Support / Resistance (last 60 trading days)
    support = df["Low"].tail(60).min()
    resistance = df["High"].tail(60).max()

    # ==========================
    # Fundamental Indicators
    # ==========================

    info = ticker.info

    revenue_growth = info.get("revenueGrowth")
    roe = info.get("returnOnEquity")
    net_margin = info.get("profitMargins")
    pe_ratio = info.get("trailingPE")

    return {
        "Ticker": symbol.upper(),

        # Fundamental
        "Revenue Growth": revenue_growth,
        "ROE": roe,
        "Net Margin": net_margin,
        "P/E Ratio": pe_ratio,

        # Technical
        "Current Price": round(current_price, 2),
        "MA200": round(ma200, 2),
        "Above MA200": current_price > ma200,
        "RSI": round(rsi, 2),
        "Relative Volume": round(relative_volume, 2),
        "Support": round(support, 2),
        "Resistance": round(resistance, 2)
    }


# ============================================
# List of companies to analyze
# ============================================

tickers = [
    "TSLA",
    # "AAPL",
    # "NVDA",
    # "MSFT"
]

# ============================================
# Print metrics
# ============================================

for ticker in tickers:

    try:
        metrics = stock_metrics(ticker)

        print("=" * 70)
        print(f"TICKER: {metrics['Ticker']}")
        print("=" * 70)

        print("\nFUNDAMENTAL METRICS")
        print("-------------------")
        print(f"Revenue Growth : {metrics['Revenue Growth']}")
        print(f"ROE            : {metrics['ROE']}")
        print(f"Net Margin     : {metrics['Net Margin']}")
        print(f"P/E Ratio      : {metrics['P/E Ratio']}")

        print("\nTECHNICAL METRICS")
        print("-----------------")
        print(f"Current Price  : {metrics['Current Price']}")
        print(f"MA200          : {metrics['MA200']}")
        print(f"Above MA200    : {metrics['Above MA200']}")
        print(f"RSI            : {metrics['RSI']}")
        print(f"Relative Volume: {metrics['Relative Volume']}")
        print(f"Support        : {metrics['Support']}")
        print(f"Resistance     : {metrics['Resistance']}")

        print()

    except Exception as e:
        print(f"{ticker}: {e}")