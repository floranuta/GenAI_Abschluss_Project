import yfinance as yf

ticker = yf.Ticker("AAPL")  # или "MSFT", "GOOG"
hist = ticker.history(period="5d")

print(hist)
