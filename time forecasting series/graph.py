import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("yahoo_stock.csv", parse_dates=['Date'])

# Set Date as index
df.set_index('Date', inplace=True)

# Plot the selected variables
plt.figure(figsize=(12, 6))
plt.plot(df['Open'], label='Open', color='blue')
plt.plot(df['High'], label='High', color='orange')
plt.plot(df['Low'], label='Low', color='green')
plt.plot(df['Close'], label='Close', color='red')
plt.plot(df['Adj Close'], label='Adj Close', color='purple')

# Labeling and aesthetics
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Time Series Plot of Stock Data')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
