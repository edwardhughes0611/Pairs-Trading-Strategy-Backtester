import yfinance as yf
import matplotlib.pyplot as plt

dat = yf.Ticker("NVDA")

df = dat.history(period="1mo")

fig, ax = plt.subplots()             # Create a figure containing a single Axes.
ax.plot(df.index, df['Close'])  # Plot some data on the Axes.
plt.show()                           # Show the figure.
print(df)