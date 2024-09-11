import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the stocks and their respective weights in the portfolio
weights = {
    "AAPL": 0.0688, "AMGN": 0.0626, "AMZN": 0.0048, "AXP": 0.0005, "BA": 0.0434,
    "CAT": 0.0086, "CRM": 0.0538, "CSCO": 0.0019, "CVX": 0.0327, "DIS": 0.0071,
    "DOW": 0.0032, "GS": 0.0056, "HD": 0.0027, "HON": 0.0137, "IBM": 0.0775,
    "INTC": 0.0086, "JNJ": 0.0323, "JPM": 0.0275, "KO": 0.0519, "MCD": 0.0334,
    "MMM": 0.0073, "MRK": 0.0813, "MSFT": 0.0269, "NKE": 0.041, "PG": 0.0375,
    "TRV": 0.0227, "UNH": 0.0751, "V": 0.0523, "VZ": 0.0712, "WMT": 0.0438
}

# Initial investment
initial_investment = 100000

# Download historical data for the stocks and the Dow Jones Industrial Average as a benchmark
tickers = list(weights.keys()) + ['^DJI']
data = yf.download(tickers, start="2021-8-1", end="2024-8-1")['Adj Close']

# Separate the benchmark data
benchmark_data = data.pop('^DJI')

# Calculate daily returns for the stocks and the benchmark
returns = data.pct_change()
benchmark_returns = benchmark_data.pct_change()

# Calculate portfolio daily returns
portfolio_daily_returns = (returns * pd.Series(weights)).sum(axis=1)

# Calculate cumulative portfolio returns and benchmark returns
cumulative_portfolio_returns = (1 + portfolio_daily_returns).cumprod()
cumulative_benchmark_returns = (1 + benchmark_returns).cumprod()

# Calculate final portfolio value and benchmark value
final_portfolio_value = initial_investment * cumulative_portfolio_returns
final_benchmark_value = initial_investment * cumulative_benchmark_returns

# Calculate annual returns and Sharpe ratio for both portfolio and benchmark
annual_returns = portfolio_daily_returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
annual_benchmark_returns = benchmark_returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
annual_volatility = portfolio_daily_returns.resample('Y').std() * np.sqrt(252)
annual_benchmark_volatility = benchmark_returns.resample('Y').std() * np.sqrt(252)
risk_free_rate = 0.03  # Assume a risk-free rate of 3%
annual_sharpe_ratios = (annual_returns - risk_free_rate) / annual_volatility
annual_benchmark_sharpe = (annual_benchmark_returns - risk_free_rate) / annual_benchmark_volatility

# Calculate Beta and Alpha for the portfolio
covariance = np.cov(portfolio_daily_returns[1:], benchmark_returns[1:])
beta = covariance[0, 1] / covariance[1, 1]
alpha = annual_returns.mean() - (risk_free_rate + beta * (annual_benchmark_returns.mean() - risk_free_rate))

# Calculate and print excess returns for holding periods of 1, 2, 3, 4, and 5 years
holding_periods = [1, 2, 3, 4, 5]

for period in holding_periods:
    if len(final_portfolio_value) > period * 252:
        portfolio_period_return = final_portfolio_value[-1] / final_portfolio_value[-(period * 252)] - 1
        benchmark_period_return = final_benchmark_value[-1] / final_benchmark_value[-(period * 252)] - 1
        excess_return = portfolio_period_return - benchmark_period_return
        print(f"{period}-year Holding Period: Portfolio Return: {portfolio_period_return * 100:.2f}%, Benchmark Return: {benchmark_period_return * 100:.2f}%, Excess Return: {excess_return * 100:.2f}%")

# Print annual metrics for portfolio and benchmark
print("\nAnnual Returns and Sharpe Ratios:")
for year in sorted(annual_returns.index.year):
    print(f"{year} - Portfolio Return: {annual_returns[str(year)].mean() * 100:.2f}%, Sharpe Ratio: {annual_sharpe_ratios[str(year)].mean():.2f}")
    print(f"{year} - Benchmark Return: {annual_benchmark_returns[str(year)].mean() * 100:.2f}%, Sharpe Ratio: {annual_benchmark_sharpe[str(year)].mean():.2f}")

# Extract date range for plotting
start_date = final_portfolio_value.index.min().strftime('%Y-%m-%d')
end_date = final_portfolio_value.index.max().strftime('%Y-%m-%d')

# Plot both the portfolio and benchmark growth over time
plt.figure(figsize=(14, 7))
plt.plot(final_portfolio_value, label='Portfolio Value')
plt.plot(final_benchmark_value, color='red', label='DJI Index')
plt.title(f'Portfolio and Benchmark Value Over Time\n({start_date} to {end_date})')
plt.xlabel('Date')
plt.ylabel('Value in $')
plt.legend()
plt.grid(True)
plt.show()

# Calculate and print the final Sharpe ratio of the portfolio
final_annual_return = (final_portfolio_value[-1] / initial_investment) ** (1 / (len(final_portfolio_value) / 252)) - 1
final_annual_volatility = portfolio_daily_returns.std() * np.sqrt(252)
final_sharpe_ratio = (final_annual_return - risk_free_rate) / final_annual_volatility

# Print portfolio and benchmark metrics
print("\nPortfolio vs. Benchmark Metrics:")
print(f"Portfolio Beta: {beta:.2f}")
print(f"Portfolio Alpha: {alpha * 100:.2f}%")
print(f"Final portfolio value (end of period): ${final_portfolio_value.iloc[-1]:,.2f}")
print(f"Final benchmark value (end of period): ${final_benchmark_value.iloc[-1]:,.2f}")
print(f"Final Portfolio Sharpe Ratio: {final_sharpe_ratio:.2f}")
