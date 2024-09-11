# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
data = pd.read_csv('all_stocks_forecasts.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Calculate daily returns
returns = data.pct_change().dropna()

# Set parameters for Monte Carlo simulation
num_simulations = 100000
num_days = 252  # Approximately one trading year

# Set random seed for reproducibility
np.random.seed(42)

# Perform Monte Carlo simulation for each stock
simulation_results = {}
for stock in returns.columns:
    mean_return = returns[stock].mean()
    std_return = returns[stock].std()
    
    # Simulate stock prices
    simulated_paths = np.zeros((num_days, num_simulations))
    simulated_paths[0] = data[stock].iloc[-1]
    for t in range(1, num_days):
        random_returns = np.random.normal(mean_return, std_return, num_simulations)
        simulated_paths[t] = simulated_paths[t-1] * (1 + random_returns)
    
    simulation_results[stock] = simulated_paths

# Calculate portfolio metrics
num_portfolios = 100000
results = np.zeros((4, num_portfolios))
weights_record = []

for i in range(num_portfolios):
    # Generate random weights
    weights = np.random.random(len(returns.columns))
    weights /= np.sum(weights)
    weights_record.append(weights)
    
    # Calculate portfolio return and risk
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe_ratio = portfolio_return / portfolio_stddev
    
    # Store results
    results[0, i] = portfolio_return
    results[1, i] = portfolio_stddev
    results[2, i] = sharpe_ratio
    results[3, i] = np.sum(weights > 0)

# Find the optimal portfolio (highest Sharpe ratio)
max_sharpe_idx = np.argmax(results[2])
optimal_weights = weights_record[max_sharpe_idx]
optimal_portfolio_return = results[0, max_sharpe_idx]
optimal_portfolio_stddev = results[1, max_sharpe_idx]
optimal_sharpe_ratio = results[2, max_sharpe_idx]
optimal_cardinality = results[3, max_sharpe_idx]

# Calculate portfolio metrics
one_day_return = np.sum(returns.mean() * optimal_weights)
annualized_return = optimal_portfolio_return
sharpe_ratio = optimal_sharpe_ratio
daily_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(returns.cov(), optimal_weights)))
annual_volatility = daily_volatility * np.sqrt(252)

# Get the stock names
stock_names = returns.columns

# Map stock names to their weights in the optimal portfolio
optimal_weights_dict = dict(zip(stock_names, optimal_weights))

# Print the results
print("Optimal Portfolio Cardinality:", optimal_cardinality)
print("One-day Return of the Portfolio:", one_day_return)
print("Annualized Return of the Portfolio:", annualized_return)
print("Sharpe Ratio of the Portfolio:", sharpe_ratio)
print("Daily Volatility of the Portfolio:", daily_volatility)
print("Annual Volatility of the Portfolio:", annual_volatility)
print("Weights of the Stocks in the Optimal Portfolio:")
for stock, weight in optimal_weights_dict.items():
    print(f"{stock}: {weight:.4f}")

# Monte Carlo simulation for the portfolio based on optimal weights
portfolio_simulation = np.zeros((num_days, num_simulations))
initial_portfolio_value = data.iloc[-1].dot(optimal_weights)
portfolio_simulation[0] = initial_portfolio_value

for t in range(1, num_days):
    random_returns = np.random.multivariate_normal(returns.mean(), returns.cov(), num_simulations)
    portfolio_simulation[t] = portfolio_simulation[t-1] * (1 + np.dot(random_returns, optimal_weights))

# Plot the simulation results for the portfolio
plt.figure(figsize=(10, 6))
plt.plot(portfolio_simulation, lw=0.5, alpha=0.1, color='green')
plt.title('Monte Carlo Simulation for the Portfolio')
plt.xlabel('Days')
plt.ylabel('Portfolio Value')
plt.show()

# Calculate Value at Risk (VaR) at the 95% confidence level
confidence_level = 0.95
VaR = np.percentile(portfolio_simulation[-1], (1 - confidence_level) * 100)
print(f"Value at Risk (VaR) at {confidence_level * 100}% confidence level: {VaR:.2f}")

# Calculate Conditional Value at Risk (CVaR) at the 95% confidence level
tail_losses = portfolio_simulation[-1, portfolio_simulation[-1] <= VaR]
CVaR = tail_losses.mean()
print(f"Conditional Value at Risk (CVaR) at {confidence_level * 100}% confidence level: {CVaR:.2f}")

# Plot histogram of final portfolio values
plt.figure(figsize=(10, 6))
plt.hist(portfolio_simulation[-1], bins=50, alpha=0.75, color='blue', edgecolor='black')
plt.title('Distribution of Final Portfolio Values')
plt.xlabel('Final Portfolio Value')
plt.ylabel('Frequency')
plt.axvline(VaR, color='r', linestyle='dashed', linewidth=2, label=f'VaR: {VaR:.2f}')
plt.axvline(CVaR, color='g', linestyle='dashed', linewidth=2, label=f'CVaR: {CVaR:.2f}')
plt.legend()
plt.show()