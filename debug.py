import yfinance as yf
import pandas as pd


def debug_price_and_ma200(symbol):
    print("=" * 70)
    print(f"DEBUGGING {symbol}")
    print("=" * 70)

    ticker = yf.Ticker(symbol)

    # Download data
    df = ticker.history(period="1y", interval="1d")

    print("\n1. DataFrame Information")
    print("-" * 40)
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    if df.empty:
        print("\nERROR: DataFrame is empty!")
        return

    print("\n2. Last 10 rows")
    print("-" * 40)
    print(df.tail(10))

    close = df["Close"]

    print("\n3. Close prices")
    print("-" * 40)
    print(close.tail(10))

    print("\n4. Missing values")
    print("-" * 40)
    print("NaN in Close:", close.isna().sum())
    print("Valid Close values:", close.notna().sum())

    print("\n5. Current Price")
    print("-" * 40)
    print("close.iloc[-1] =", close.iloc[-1])

    if pd.isna(close.iloc[-1]):
        print("WARNING: Last closing price is NaN")
        print("Last valid closing price:",
              close.dropna().iloc[-1])

    print("\n6. MA200 Calculation")
    print("-" * 40)

    ma200_series = close.rolling(window=200).mean()

    print("Last 10 MA200 values:")
    print(ma200_series.tail(10))

    print("\nFinal MA200:", ma200_series.iloc[-1])

    if pd.isna(ma200_series.iloc[-1]):
        print("\nWARNING: MA200 is NaN")

        if len(close.dropna()) < 200:
            print("Reason: Less than 200 valid price points.")
        else:
            print("Reason: One or more NaN values exist in the rolling window.")

    print("\n7. Comparison")
    print("-" * 40)

    current_price = close.iloc[-1]
    ma200 = ma200_series.iloc[-1]

    print("Current Price:", current_price)
    print("MA200:", ma200)

    if pd.notna(current_price) and pd.notna(ma200):
        print("Above MA200:", current_price > ma200)
    else:
        print("Above MA200: Cannot compute because one value is NaN")

    print("\n8. Last valid values")
    print("-" * 40)
    print("Last valid Close:",
          close.dropna().iloc[-1])

    print("Last valid MA200:",
          ma200_series.dropna().iloc[-1])

    print("\nDebug complete.")


# Run
debug_price_and_ma200("TSLA")