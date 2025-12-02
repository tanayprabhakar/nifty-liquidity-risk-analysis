import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def fetch_data():
    """Fetches historical data for Nifty 50 and selected sectors."""
    tickers = {
        'Nifty 50': '^NSEI',
        'Nifty Bank': '^NSEBANK',
        'Nifty IT': '^CNXIT'
    }
    
    print("Fetching data from yfinance...")
    data = yf.download(list(tickers.values()), period="2y", progress=False)
    
    # Handle MultiIndex columns from yfinance (Price, Ticker)
    if isinstance(data.columns, pd.MultiIndex):
        try:
            data = data['Adj Close']
        except KeyError:
            data = data['Close']
            
    # Rename columns to friendly names
    inv_tickers = {v: k for k, v in tickers.items()}
    data.rename(columns=inv_tickers, inplace=True)
    
    # Drop any rows with missing values to ensure alignment
    data.dropna(inplace=True)
    return data

def calculate_rolling_beta(df, window=30):
    """Calculates the rolling beta for each column against Nifty 50."""
    # Calculate daily percent change
    returns = df.pct_change().dropna()
    
    market_col = 'Nifty 50'
    market_returns = returns[market_col]
    
    betas = pd.DataFrame(index=returns.index)
    
    # Calculate rolling variance of the market
    market_rolling_var = market_returns.rolling(window=window).var()
    
    for col in returns.columns:
        if col == market_col:
            continue
            
        # Calculate rolling covariance between sector and market
        rolling_cov = returns[col].rolling(window=window).cov(market_returns)
        
        # Calculate Beta
        beta = rolling_cov / market_rolling_var
        
        # --- Outlier Cleaning (Crucial Step) ---
        # Filter or clip beta values. Replace any beta < -2 or > 3 with NaN
        beta = beta.mask((beta < -2) | (beta > 3))
        
        betas[col] = beta
        
    return betas

def plot_betas(betas):
    """Plots the rolling betas with specified formatting."""
    # Use seaborn-whitegrid theme
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(12, 6))
    
    for col in betas.columns:
        plt.plot(betas.index, betas[col], label=f'{col} Beta')
        
    # Force Y-axis limits
    plt.ylim(0.0, 2.0)
    
    # Add dashed horizontal line at y=1.0
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Market Benchmark (1.0)')
    
    plt.title('30-Day Rolling Beta of Nifty Sectors (Outliers Removed)')
    plt.xlabel('Date')
    plt.ylabel('Beta')
    plt.legend()
    plt.tight_layout()
    
    output_file = 'rolling_beta_output.png'
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    plt.close()

def main():
    try:
        # 1. Data Preparation
        df = fetch_data()
        
        if df.empty:
            print("No data fetched. Please check your internet connection or ticker symbols.")
            return

        # 2. Calculation Logic & 3. Outlier Cleaning
        betas = calculate_rolling_beta(df)
        
        # 4. Visualization
        plot_betas(betas)
        print("Analysis complete.")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
